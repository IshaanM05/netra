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
    def __init__(self, audio_io):
        self._audio_io = audio_io
        self._ws = None
        self._audio_queue: asyncio.Queue | None = None
        self._connected = False
        self._session_id: str | None = None
        self._speculative_cache: dict | None = None

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
        headers = {"Authorization": f"Bearer {config.API_KEY}"}

        async with websockets.connect(
            config.AGENT_WS_URL, additional_headers=headers
        ) as ws:
            self._ws = ws

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": config.DEFAULT_SYSTEM_PROMPT,
                    "greeting": "",
                    "output": {"voice": config.DEFAULT_VOICE},
                    "tools": TOOL_DEFINITIONS,
                },
            }))

            self._audio_queue = self._audio_io.subscribe_agent()

            receive_task = asyncio.create_task(self._receive_loop(stop_event))
            send_task = asyncio.create_task(self._send_audio_loop(stop_event))

            await asyncio.gather(receive_task, send_task, return_exceptions=True)

    async def _send_audio_loop(self, stop_event: asyncio.Event):
        while not stop_event.is_set():
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

                if msg_type == "session.ready":
                    self._session_id = msg.get("session_id", "?")
                    self._connected = True
                    print(f"[agent] session ready — id: {self._session_id}")

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

                elif msg_type == "tool.call":
                    call_id = msg.get("call_id", "")
                    name = msg.get("name", "")
                    arguments = msg.get("arguments", "{}")
                    await self._handle_tool_call(call_id, name, arguments)

                elif msg_type == "error":
                    print(f"  [agent error] {msg.get('message', msg)}")

        except websockets.ConnectionClosed:
            pass
        finally:
            self._connected = False

    async def _handle_tool_call(self, call_id: str, name: str, arguments: str):
        cached = self._speculative_cache
        if cached and cached["tool_name"] == name:
            age_ms = (time.monotonic() - cached["cached_at"]) * 1000
            result = cached["result"]
            self._speculative_cache = None
            print(f"  \033[1;32m[tool — speculative hit!]\033[0m {name} → {result} (cached {age_ms:.0f}ms ago)")
        else:
            if cached:
                print(f"  \033[1;31m[tool — spec miss]\033[0m expected {cached['tool_name']}, got {name}")
                self._speculative_cache = None
            print(f"  [tool] {name}({arguments})")
            result = execute_tool(name, arguments)
            print(f"  [tool → agent] {result}")

        if self._ws:
            await self._ws.send(json.dumps({
                "type": "tool.result",
                "call_id": call_id,
                "result": result,
            }))

    async def update_context(self, room_summary: str):
        if not self._ws or not self._connected:
            return
        try:
            await self._ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "system_prompt": (
                        f"{config.DEFAULT_SYSTEM_PROMPT}\n\n"
                        f"## Current conversation context\n"
                        f"Here's what the people in the room have been saying:\n\n"
                        f"{room_summary}"
                    ),
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
