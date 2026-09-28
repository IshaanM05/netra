"""
Voice Agent API client: the mouth and brain.

Manages the WebSocket connection to AssemblyAI's Voice Agent API: streams mic
audio in, plays replies out, runs client-side tools, and pushes transcripts and
tool activity to the viewer.

Tool results follow the documented protocol: a result is sent only while
reply.done is the latest turn event (not during reply.started /
input.speech.started), and results are dropped when a reply is interrupted.

Speculative execution: on tool.call the speculator is asked to claim() a
result it already started from the user's partial transcript.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import websockets

from . import config
from .tools import execute_tool, tool_definitions, web_tools_enabled

# Everyday words dilute the keyterm boost (AssemblyAI guidance); keep only distinctive ones.
_COMMON_WORDS = {"motor", "gears", "gear", "drive", "fins", "tip", "heater", "cartridge", "fan", "blower",
                 "throat", "cold", "end", "print", "cooling", "clicking", "jam", "lever", "tube", "sensor",
                 "block", "heat", "sink", "break", "part", "layer", "door", "screw", "tension", "brass",
                 "element", "feed", "temperature", "thermal", "temp", "stepper", "arm", "assembly",
                 "planetary", "under", "extrusion", "stringing", "pull", "heating", "runaway"}


def _keyterms(manifest) -> list[str]:
    explicit = [str(t) for t in getattr(manifest, "keyterms", []) or []]
    words = []
    for part in manifest.parts:
        for phrase in [part.get("name"), *part.get("aliases", [])]:
            words += re.findall(r"[A-Za-z0-9]+", str(phrase))
    for entry in manifest.vocab:
        words += re.findall(r"[A-Za-z0-9]+", str(entry.get("term", "")))
    words += re.findall(r"[A-Za-z0-9]+", manifest.name)
    distinctive = [w for w in words if len(w) >= 4 and w.lower() not in _COMMON_WORDS] + \
                  [w for w in words if w.isupper() and len(w) >= 2]
    seen, out = set(), []
    for term in ["Netra", *explicit, *distinctive]:
        if term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out[:100]


def build_session(sight=None, room_context: str = "") -> dict:
    """The session.update payload. Shared by the live agent and the test harness."""
    session = {
        "system_prompt": config.build_system_prompt(sight, web_tools_enabled(), room_context),
        "greeting": "",  # must be a string: None silently invalidates the whole session.update
        "output": {"voice": config.DEFAULT_VOICE},
        "tools": tool_definitions(sight),
    }
    manifest = getattr(sight, "manifest", None)
    if manifest is not None:
        parts = ", ".join(str(p.get("name")) for p in manifest.parts)
        session["input"] = {
            "keyterms": _keyterms(manifest),
            "transcription_prompt": (
                f"Someone working hands-on with a {manifest.name}, talking to a voice assistant "
                f"named Netra. They name parts such as {parts}, and describe problems like "
                "clogs, clicking, under-extrusion, heat creep, thermal runaway and cold pulls."
            )[:1750],
        }
    return session


class AgentClient:
    def __init__(self, audio_io, sight=None, hub=None, speculator=None, initial_user_text: str | None = None):
        self._audio_io = audio_io
        self._sight = sight
        self._hub = hub
        self._speculator = speculator
        self._initial_user_text = initial_user_text
        self.session_error: str | None = None
        self._ws = None
        self._audio_queue: asyncio.Queue | None = None
        self._connected = False
        self._ready_event = asyncio.Event()
        self._connection_closed = asyncio.Event()
        self._session_id: str | None = None
        self._turn_state = ""  # latest of reply.started / input.speech.started / reply.done
        self._tool_tasks: dict[str, asyncio.Task] = {}
        self._ready_results: dict[str, str] = {}
        self._send_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def _emit(self, event: dict):
        if self._hub:
            self._hub.broadcast(event)

    async def _send(self, payload: dict):
        if not self._ws:
            return
        async with self._send_lock:
            await self._ws.send(json.dumps(payload, ensure_ascii=False))

    # ---------- connection ----------

    async def connect(self, stop_event: asyncio.Event):
        self._ready_event.clear()
        self._connection_closed.clear()
        headers = {"Authorization": f"Bearer {config.API_KEY}"}

        async with websockets.connect(config.AGENT_WS_URL, additional_headers=headers) as ws:
            self._ws = ws
            await self._send({"type": "session.update", "session": build_session(self._sight)})
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
                self._drop_tool_calls()
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
            try:
                b64_audio = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
                if self._connected:
                    await self._send({"type": "input.audio", "audio": b64_audio})
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
                if await self._handle_event(msg) is False:
                    break
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connected = False
            self._connection_closed.set()

    async def _handle_event(self, msg: dict) -> bool | None:
        t = msg.get("type", "")

        if t == "session.ready":
            self._session_id = msg.get("session_id", "?")
            registered = len(msg.get("config", {}).get("tools", []) or [])
            self._connected = True
            while self._audio_queue is not None:
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._ready_event.set()
            print(f"[agent] session ready — id: {self._session_id} ({registered} tools)")
            if self._initial_user_text:
                # The wake turn ("Hey Netra, where's the nozzle?") was heard before this socket
                # existed; hand it over so the first question gets answered.
                await self.say_as_user(self._initial_user_text)
            else:
                await self._send({"type": "reply.create", "instructions":
                                  "Let the user know you're listening, in five words or fewer."})

        elif t == "session.error":
            self.session_error = msg.get("code") or "error"
            print(f"  [agent session error] {msg.get('code', '')}: {msg.get('message', msg)}")
            if not self._connected:
                return False

        elif t == "session.ended":
            return False

        elif t in ("reply.started", "input.speech.started"):
            self._turn_state = t

        elif t == "transcript.user":
            text = msg.get("text", "")
            if text.strip():
                print(f"  \033[1m[you → agent]\033[0m {text}")
                self._emit({"type": "transcript", "who": "user", "text": text})

        elif t == "transcript.agent":
            text = msg.get("text", "")
            print(f"  \033[1;34m[agent]\033[0m {text}")
            if text.strip():
                self._emit({"type": "transcript", "who": "agent", "text": text})

        elif t == "reply.audio":
            self._audio_io.play_audio(msg.get("data", ""))

        elif t == "reply.done":
            self._turn_state = t
            self._audio_io.mark_agent_done_speaking()
            if msg.get("status") == "interrupted":
                print("  [agent interrupted]")
                self._drop_tool_calls()
            else:
                await self._flush_if_idle()

        elif t == "tool.call":
            self._handle_tool_call(msg.get("call_id", ""), msg.get("name", ""), msg.get("arguments", "{}"))

        elif t == "error":
            print(f"  [agent error] {msg.get('message', msg)}")
        return None

    # ---------- tools ----------

    def _handle_tool_call(self, call_id: str, name: str, arguments):
        self._emit({"type": "tool", "name": name, "args": _parse_arguments(arguments), "phase": "call"})
        hit = self._speculator.claim(name, arguments) if self._speculator else None
        if hit is not None:
            print(f"  \033[1;32m[tool — speculative hit!]\033[0m {name} started {hit.lead_ms:.0f}ms "
                  f"before the agent asked" + ("" if hit.ready else " (still finishing)"))
            if self._sight is not None and name in {"locate", "describe", "point", "walk_through"}:
                # Speculation only showed a ghost; the cheap local call commits the visual state.
                asyncio.create_task(asyncio.to_thread(execute_tool, name, arguments, self._sight))
            coro = self._await_speculation(name, hit)
        else:
            print(f"  [tool] {name}({arguments})")
            coro = self._execute_tool(name, arguments)
        task = asyncio.create_task(coro)
        self._tool_tasks[call_id] = task
        task.add_done_callback(lambda finished, cid=call_id: self._tool_finished(cid, finished))

    async def _execute_tool(self, name: str, arguments) -> str:
        started = time.monotonic()
        try:
            result = await asyncio.to_thread(execute_tool, name, arguments, self._sight)
        except Exception as exc:
            result = f"Tool execution failed: {exc}"
        elapsed = (time.monotonic() - started) * 1000
        print(f"  [tool → agent] {result[:300]} ({elapsed:.0f}ms)")
        self._emit({"type": "tool", "name": name, "phase": "result", "ms": round(elapsed)})
        return result

    async def _await_speculation(self, name: str, hit) -> str:
        started = time.monotonic()
        if not hit.ready:
            await asyncio.to_thread(hit.done.wait, 30)
        waited = (time.monotonic() - started) * 1000
        self._emit({"type": "tool", "name": name, "phase": "result", "ms": round(waited),
                    "speculative": True, "lead_ms": round(hit.lead_ms)})
        return hit.result or "Speculative tool call produced no result."

    def _tool_finished(self, call_id: str, task: asyncio.Task):
        self._tool_tasks.pop(call_id, None)
        if task.cancelled():
            return
        try:
            self._ready_results[call_id] = task.result()
        except Exception as exc:
            self._ready_results[call_id] = f"Tool execution failed: {exc}"
        asyncio.create_task(self._flush_if_idle())

    async def _flush_if_idle(self):
        """Send ready tool results, but only while reply.done is the latest turn event."""
        if self._turn_state != "reply.done" or not self._ready_results:
            return
        pending, self._ready_results = self._ready_results, {}
        try:
            for call_id, result in pending.items():
                await self._send({"type": "tool.result", "call_id": call_id,
                                  "result": json.dumps({"result": result}, ensure_ascii=False)})
        except websockets.ConnectionClosed:
            pass

    def _drop_tool_calls(self):
        for task in self._tool_tasks.values():
            task.cancel()
        self._tool_tasks.clear()
        self._ready_results.clear()

    # ---------- context ----------

    async def say_as_user(self, text: str):
        """Inject a user utterance and ask the agent to answer it now."""
        # conversation.message(role=user) alone does not reach the model; reply.create
        # instructions do, and the exchange stays in the conversation history.
        try:
            await self._send({"type": "reply.create", "instructions":
                              f'The user just said: "{text}". Treat this exactly as if they had spoken it: '
                              "reply to it now, following your system prompt and calling tools as usual."})
            self._emit({"type": "transcript", "who": "user", "text": text})
        except websockets.ConnectionClosed:
            pass

    async def refresh_session(self, room_summary: str = ""):
        """After a machine switch: new prompt, keyterms and transcription prompt (tools unchanged)."""
        if not self._ws or not self._connected:
            return
        session = build_session(self._sight, room_summary)
        for immutable in ("greeting", "output"):
            session.pop(immutable, None)
        try:
            await self._send({"type": "session.update", "session": session})
        except websockets.ConnectionClosed:
            pass

    async def announce(self, instructions: str):
        """Make the agent speak now (e.g. a background job finished)."""
        if not self._ws or not self._connected:
            return
        try:
            await self._send({"type": "reply.create", "instructions": instructions})
        except websockets.ConnectionClosed:
            pass

    async def update_context(self, room_summary: str):
        if not self._ws or not self._connected:
            return
        try:
            await self._send({"type": "session.update", "session": {
                "system_prompt": config.build_system_prompt(self._sight, web_tools_enabled(), room_summary)}})
        except websockets.ConnectionClosed:
            pass

    async def disconnect(self):
        if self._ws:
            try:
                await self._send({"type": "session.end"})
            except websockets.ConnectionClosed:
                pass
            self._connected = False


def _parse_arguments(arguments):
    if isinstance(arguments, str):
        try:
            return json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return {"raw": arguments}
    return arguments or {}
