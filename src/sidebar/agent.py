"""
Voice Agent API client: the mouth and brain.

Manages the WebSocket connection to AssemblyAI's Voice Agent API.
Sends audio, receives replies, handles tool calls, and supports
mid-session context updates via session.update.

Integrates with speculative execution: the engine can cache a
pre-computed tool result; when the agent fires a matching tool.call,
the cached result is used instantly instead of re-executing.
"""

import asyncio
import json
import time

import websockets

from . import config
from .tools import TOOL_DEFINITIONS, execute_tool


class AgentClient:
    def __init__(self, audio_io, sight=None):
        self._audio_io = audio_io
        self._sight = sight
        self._ws = None
        self._audio_queue: asyncio.Queue | None = None
        self._connected = False
        self._ready_event = asyncio.Event()
        self._connection_closed = asyncio.Event()
        self._session_id: str | None = None
        self._speculative_cache: dict | None = None
        self._tool_tasks: dict[str, asyncio.Task] = {}
        self._tool_results: dict[str, str] = {}
        self._tool_batch_ids: list[str] = []
        self._tool_reply_done = False
        self._last_event_type = ""
        self._flushing_tool_results = False

    def _system_prompt(self, room_context: str = "") -> str:
        manifest = getattr(self._sight, "manifest", None)
        visual_context = manifest.context() if manifest else "(no manifest configured)"
        prompt = (
            f"{config.DEFAULT_SYSTEM_PROMPT}\n\n"
            f"## Active visual field\nAdapter: {getattr(self._sight, 'name', 'unknown')}\n"
            f"Manifest: {visual_context}"
        )
        if room_context:
            prompt += f"\n\n## Current conversation context\n{room_context}"
        return prompt

    @property
    def connected(self) -> bool:
        return self._connected

    def cache_speculative_result(self, tool_name: str, arguments: str, result: str):
        self._speculative_cache = {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "cached_at": time.monotonic(),
        }

    def clear_speculative_cache(self):
        self._speculative_cache = None

    async def connect(self, stop_event: asyncio.Event):
        self._ready_event.clear()
        self._connection_closed.clear()
        headers = {"Authorization": f"Bearer {config.API_KEY}"}

        async with websockets.connect(
            config.AGENT_WS_URL, additional_headers=headers
        ) as ws:
            self._ws = ws

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": self._system_prompt(),
                    "greeting": "",
                    "output": {"voice": config.DEFAULT_VOICE},
                    "tools": TOOL_DEFINITIONS,
                },
            }))

            self._audio_queue = self._audio_io.subscribe_agent()

            receive_task = asyncio.create_task(self._receive_loop(stop_event))
            send_task = asyncio.create_task(self._send_audio_loop(stop_event))
            try:
                await asyncio.gather(receive_task, send_task)
            finally:
                self._connection_closed.set()
                for task in (receive_task, send_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(receive_task, send_task, return_exceptions=True)
                self._discard_pending_tool_results()
                if self._audio_queue is not None:
                    self._audio_io.unsubscribe_agent(self._audio_queue)
                    self._audio_queue = None

    async def _send_audio_loop(self, stop_event: asyncio.Event):
        while not stop_event.is_set() and not self._connection_closed.is_set():
            if not self._ready_event.is_set():
                try:
                    await asyncio.wait_for(self._ready_event.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                if self._connection_closed.is_set():
                    break
            try:
                b64_audio = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5
                )
                if self._ws and self._connected:
                    await self._ws.send(json.dumps({
                        "type": "input.audio",
                        "audio": b64_audio,
                    }))
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                break

    async def _receive_loop(self, stop_event: asyncio.Event):
        try:
            async for raw_msg in self._ws:
                if stop_event.is_set():
                    break

                msg = json.loads(raw_msg)
                msg_type = msg.get("type", "")
                self._last_event_type = msg_type

                if msg_type == "session.ready":
                    self._session_id = msg.get("session_id", "?")
                    self._connected = True
                    while self._audio_queue is not None:
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self._ready_event.set()
                    print(f"[agent] session ready — id: {self._session_id}")

                elif msg_type == "session.error":
                    print(f"  [agent session error] {msg.get('code', '')}: {msg.get('message', msg)}")
                    break

                elif msg_type == "transcript.user.delta":
                    pass

                elif msg_type == "transcript.user":
                    text = msg.get("text", "")
                    if text.strip():
                        print(f"  \033[1m[you → agent]\033[0m {text}")

                elif msg_type == "transcript.agent":
                    text = msg.get("text", "")
                    print(f"  \033[1;34m[agent]\033[0m {text}")

                elif msg_type == "reply.audio":
                    self._audio_io.play_audio(msg.get("data", ""))

                elif msg_type == "reply.done":
                    self._audio_io.mark_agent_done_speaking()
                    status = msg.get("status", "")
                    if status == "interrupted":
                        print("  [agent interrupted]")
                        self.clear_speculative_cache()
                        self._discard_pending_tool_results()
                    else:
                        self._tool_reply_done = bool(self._tool_batch_ids)
                        await self._send_pending_tool_results()

                elif msg_type == "input.speech.started" and self._tool_reply_done:
                    self._discard_pending_tool_results()

                elif msg_type == "tool.call":
                    call_id = msg.get("call_id", "")
                    name = msg.get("name", "")
                    arguments = msg.get("arguments", "{}")
                    self._handle_tool_call(call_id, name, arguments)

                elif msg_type == "error":
                    print(f"  [agent error] {msg.get('message', msg)}")

        except websockets.ConnectionClosed:
            pass
        finally:
            self._connected = False
            self._connection_closed.set()

    def _handle_tool_call(self, call_id: str, name: str, arguments):
        if not self._tool_batch_ids:
            self._tool_reply_done = False
        self._tool_batch_ids.append(call_id)
        cached = self._speculative_cache
        if (cached and cached["tool_name"] == name
                and self._normalize_arguments(cached["arguments"])
                == self._normalize_arguments(arguments)):
            age_ms = (time.monotonic() - cached["cached_at"]) * 1000
            self._tool_results[call_id] = cached["result"]
            self._speculative_cache = None
            print(f"  \033[1;32m[tool — speculative hit!]\033[0m {name} (cached {age_ms:.0f}ms ago)")
        else:
            if cached:
                print(f"  \033[1;31m[tool — spec miss]\033[0m expected {cached['tool_name']} with matching arguments, got {name}")
                self._speculative_cache = None
            print(f"  [tool] {name}({arguments})")
            task = asyncio.create_task(self._execute_tool(name, arguments))
            self._tool_tasks[call_id] = task
            task.add_done_callback(lambda finished, cid=call_id: self._tool_finished(cid, finished))

    async def _execute_tool(self, name: str, arguments) -> str:
        try:
            result = await asyncio.to_thread(execute_tool, name, arguments, self._sight)
            print(f"  [tool → agent] {result}")
            return result
        except Exception as exc:
            return f"Tool execution failed: {exc}"

    def _tool_finished(self, call_id: str, task: asyncio.Task):
        self._tool_tasks.pop(call_id, None)
        if task.cancelled():
            return
        try:
            self._tool_results[call_id] = task.result()
        except Exception as exc:
            self._tool_results[call_id] = f"Tool execution failed: {exc}"
        if self._tool_reply_done and self._last_event_type == "reply.done":
            asyncio.create_task(self._send_pending_tool_results())

    async def _send_pending_tool_results(self):
        """Send completed tool results only after the matching reply.done."""
        if (not self._ws or not self._tool_reply_done
                or self._last_event_type != "reply.done" or not self._tool_batch_ids
                or self._flushing_tool_results):
            return
        if any(call_id not in self._tool_results for call_id in self._tool_batch_ids):
            return
        pending = [(call_id, self._tool_results[call_id]) for call_id in self._tool_batch_ids]
        self._flushing_tool_results = True
        try:
            for call_id, result in pending:
                await self._ws.send(json.dumps({
                    "type": "tool.result",
                    "call_id": call_id,
                    "result": json.dumps({"result": result}, ensure_ascii=False),
                }))
            self._clear_tool_batch()
        except websockets.ConnectionClosed:
            self._discard_pending_tool_results()
        finally:
            self._flushing_tool_results = False

    @staticmethod
    def _normalize_arguments(arguments) -> str:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)

    def _clear_tool_batch(self):
        for task in self._tool_tasks.values():
            task.cancel()
        self._tool_tasks.clear()
        self._tool_results.clear()
        self._tool_batch_ids.clear()
        self._tool_reply_done = False

    def _discard_pending_tool_results(self):
        self._clear_tool_batch()

    async def update_context(self, room_summary: str):
        if not self._ws or not self._connected:
            return
        try:
            await self._ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": self._system_prompt(room_summary),
                },
            }))
        except websockets.ConnectionClosed:
            pass

    async def disconnect(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "session.end"}))
            except websockets.ConnectionClosed:
                pass
            self._connected = False
