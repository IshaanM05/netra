"""
Netra engine: orchestrates the dual-socket architecture.

- Universal-Streaming runs continuously (the ears): wake phrase, room context,
  and the partial transcripts that drive speculative execution
- Voice Agent API connects on demand (the mouth and brain) and stays up until
  the user says "Netra, stop" or it disconnects
- ViewerHub pushes visual actions, transcripts and speculation stats to the
  browser 3D viewer (Field mode)
"""

import asyncio
import os
import re
import signal
import time

from .audio import AudioIO
from .streaming import StreamingListener, Turn
from .agent import AgentClient
from .speculation import SpeculativeExecutor
from .tools import tool_definitions, web_tools_enabled
from .viewer import ViewerHub
from . import config

# Common mis-hearings of "Netra" by streaming STT.
_WAKE_VARIANTS = {"netra", "netraa", "nitra", "neetra", "netro", "netrah", "nethra", "natra", "neta"}
_STOP_PATTERN = re.compile(r"\b(?:stop listening|go to sleep|goodbye|that's all|thats all|stop)\b")


class NetraEngine:
    CONNECT_ATTEMPTS = 4

    def __init__(self):
        self._audio = AudioIO()
        self._agent: AgentClient | None = None
        self._stop_event = asyncio.Event()
        self._agent_active = False
        self._agent_task: asyncio.Task | None = None
        self._last_context_update = 0.0
        self._context_update_interval = 10.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._hub: ViewerHub | None = None
        self._speculator: SpeculativeExecutor | None = None
        self._sight = None
        self._streaming_listener: StreamingListener | None = None
        self._connecting = False

    @staticmethod
    def _speaker_tag(speaker: str | None) -> str:
        if speaker is None or speaker == "PENDING":
            return "..."
        return f"Speaker {speaker}"

    def _schedule(self, coro):
        if self._loop:
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _emit(self, event: dict):
        if self._hub:
            self._hub.broadcast(event)

    def _is_echo(self) -> bool:
        return self._agent_active and self._audio._is_echo_suppressed()

    # ---------- streaming callbacks (SDK thread) ----------

    def _on_partial_turn(self, turn: Turn):
        if self._is_echo():
            return
        tag = self._speaker_tag(turn.speaker)
        print(f"\r  [{tag}] {turn.text}    ", end="", flush=True)
        self._emit({"type": "partial", "text": turn.text})
        if self._agent_active and self._speculator:
            self._speculator.on_partial_turn(turn.text)

    def _on_final_turn(self, turn: Turn):
        if self._is_echo():
            return
        tag = self._speaker_tag(turn.speaker)
        print(f"\n  \033[1m[{tag}]\033[0m {turn.text}")
        self._emit({"type": "partial", "text": ""})

        if self._agent_active and self._speculator:
            self._speculator.on_final_turn(turn.text)

        woke = self._detect_wake(turn.text)
        if woke and self._agent_active and self._detect_stop(turn.text):
            print("  \033[1;33m[stop phrase]\033[0m")
            self._schedule(self._deactivate_agent())
            return
        if woke and not self._agent_active:
            print("  \033[1;32m[wake detected!]\033[0m")
            self._schedule(self._activate_agent(self._after_wake(turn.text)))
        if self._agent_active:
            self._schedule(self._async_context_update())

    @staticmethod
    def _detect_wake(text: str) -> bool:
        words = re.findall(r"[a-z]+", text.lower())
        return any(w in _WAKE_VARIANTS for w in words)

    @staticmethod
    def _after_wake(text: str) -> str | None:
        """The request that followed the wake word in the same turn, if any."""
        words = text.split()
        for i, word in enumerate(words):
            if re.sub(r"[^a-z]", "", word.lower()) in _WAKE_VARIANTS:
                rest = " ".join(words[i + 1:]).strip(" ,.!?")
                return rest if len(rest.split()) >= 2 else None
        return None

    @staticmethod
    def _detect_stop(text: str) -> bool:
        return bool(_STOP_PATTERN.search(text.lower()))

    # ---------- agent lifecycle ----------

    async def _activate_agent(self, initial_user_text: str | None = None):
        if self._agent_active:
            return
        self._agent_active = True
        self._connecting = True
        try:
            await self._connect_with_retry(initial_user_text)
        finally:
            self._connecting = False

    async def _connect_with_retry(self, initial_user_text: str | None):
        self._emit({"type": "status", "state": "connecting"})
        for attempt in range(1, self.CONNECT_ATTEMPTS + 1):
            print(f"[engine] activating agent (attempt {attempt})...")
            agent = AgentClient(self._audio, sight=self._sight, hub=self._hub,
                                speculator=self._speculator, initial_user_text=initial_user_text)
            self._agent = agent
            task = asyncio.create_task(agent.connect(self._stop_event))
            self._agent_task = task
            task.add_done_callback(lambda t, a=agent: self._on_agent_stopped(a, t))

            deadline = time.monotonic() + 10
            while not agent.connected and not task.done() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if agent.connected:
                self._emit({"type": "status", "state": "active"})
                if self._streaming_listener:
                    await agent.update_context(self._context())
                    self._last_context_update = time.monotonic()
                return
            if not task.done():
                task.cancel()
            reason = agent.session_error or "timeout"
            if self._stop_event.is_set():
                return
            backoff = min(2 ** attempt, 8)
            print(f"[engine] agent not ready ({reason}); retrying in {backoff}s")
            await asyncio.sleep(backoff)
        print("[engine] could not start the voice agent; say the wake phrase to try again")
        self._agent_active = False
        self._agent = None
        self._emit({"type": "status", "state": "listening", "error": "Voice agent unavailable"})

    async def _deactivate_agent(self):
        agent, task = self._agent, self._agent_task
        if agent:
            await agent.disconnect()
        if task and not task.done():
            task.cancel()

    def _on_agent_stopped(self, agent: AgentClient, task: asyncio.Task):
        if self._agent is not agent:
            return
        if not task.cancelled():
            error = task.exception()
            if error:
                print(f"[engine] agent connection ended: {error}")
        if self._connecting or self._stop_event.is_set():
            return  # connect failures are handled by the retry loop
        self._agent_active = False
        self._agent = None
        self._agent_task = None
        if self._speculator:
            self._speculator.cancel_all()
        if not self._stop_event.is_set():
            print(f"[engine] agent session ended. Say \"{config.WAKE_PHRASE}\" to start again.")
            self._emit({"type": "status", "state": "listening"})

    def _context(self) -> str:
        return self._streaming_listener.transcript.summary()

    async def _async_context_update(self):
        now = time.monotonic()
        if now - self._last_context_update < self._context_update_interval:
            return
        if self._agent and self._streaming_listener:
            await self._agent.update_context(self._context())
            self._last_context_update = now

    # ---------- main ----------

    def _startup_report(self):
        manifest = getattr(self._sight, "manifest", None)
        tools = [t["name"] for t in tool_definitions(self._sight)]
        print(f"[netra] sight: {self._sight.name}"
              + (f" | manifest: {manifest.name} ({len(manifest.parts)} parts, "
                 f"{len(manifest.procedures)} procedures)" if manifest else ""))
        print(f"[netra] tools: {', '.join(tools)}")
        if not web_tools_enabled():
            print("[netra] \033[33mweb tools off\033[0m — set ANAKIN_API_KEY to enable live search")
        if self._hub:
            print(f"[netra] \033[1;36m3D viewer: {self._hub.url}\033[0m")

    async def run(self):
        self._loop = asyncio.get_running_loop()
        manifest_sight = config.build_sight()
        if getattr(manifest_sight, "name", "") == "model":
            self._hub = ViewerHub()
            await self._hub.start(manifest_sight.manifest)
            manifest_sight.renderer = self._hub
        self._sight = manifest_sight
        self._speculator = SpeculativeExecutor(self._sight, self._hub)

        self._audio.start(self._loop)
        keyterms = []
        manifest = getattr(self._sight, "manifest", None)
        if manifest:
            for part in manifest.parts:
                keyterms += [str(part.get("name")), *map(str, part.get("aliases", []))]
            keyterms += [str(v.get("term")) for v in manifest.vocab]
        self._streaming_listener = StreamingListener(
            audio_queue=self._audio.subscribe_streaming(),
            on_partial_turn=self._on_partial_turn,
            on_final_turn=self._on_final_turn,
            keyterms=list(dict.fromkeys(k for k in keyterms if k)),
        )

        self._loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self._shutdown()))

        self._startup_report()
        print(f"[engine] listening. Say \"{config.WAKE_PHRASE}\" to talk to Netra; \"Netra, stop\" to end.")
        print("[engine] Ctrl+C to quit.\n")
        self._emit({"type": "status", "state": "listening"})
        if os.environ.get("NETRA_AUTOSTART", "").strip() in {"1", "true", "yes"}:
            self._schedule(self._activate_agent())

        await self._streaming_listener.run(self._stop_event)

    async def _shutdown(self):
        print("\n[engine] shutting down...")
        self._stop_event.set()

        if self._speculator and self._speculator.stats.total_speculations:
            print(f"[engine] {self._speculator.stats.summary()}")

        if self._agent:
            await self._agent.disconnect()
        if self._agent_task:
            self._agent_task.cancel()
            try:
                await self._agent_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._hub:
            await self._hub.stop()

        self._audio.stop()
        print("[engine] done.")
