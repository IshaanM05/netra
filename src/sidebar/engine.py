"""
Sidebar engine: orchestrates the dual-socket architecture.

- Universal-Streaming runs continuously (the ears)
- Voice Agent API connects on-demand when wake phrase detected (the mouth)
- Rolling context from the room transcript feeds into the agent
- Audio gate: mic audio only forwards to agent when active
- Speculative execution: predicts tool calls from partial turns
"""

import asyncio
import signal
import time

from .audio import AudioIO
from .streaming import StreamingListener, Turn
from .agent import AgentClient
from .speculation import SpeculativeExecutor
from . import config


class SidebarEngine:
    def __init__(self):
        self._audio = AudioIO()
        self._agent: AgentClient | None = None
        self._stop_event = asyncio.Event()
        self._agent_active = False
        self._agent_task: asyncio.Task | None = None
        self._last_context_update = 0.0
        self._context_update_interval = 10.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._speculator = SpeculativeExecutor()

    @staticmethod
    def _speaker_tag(speaker: str | None) -> str:
        if speaker is None or speaker == "PENDING":
            return "..."
        return f"Speaker {speaker}"

    def _schedule(self, coro):
        if self._loop:
            asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _is_echo(self) -> bool:
        return self._agent_active and self._audio._is_echo_suppressed()

    def _on_partial_turn(self, turn: Turn):
        if self._is_echo():
            return

        tag = self._speaker_tag(turn.speaker)
        print(f"\r  [{tag}] {turn.text}    ", end="", flush=True)

        if self._agent_active:
            call = self._speculator.on_partial_turn(turn.text)
            if call and call.result and self._agent:
                self._agent.cache_speculative_result(
                    call.tool_name, call.arguments, call.result
                )

    def _on_final_turn(self, turn: Turn):
        if self._is_echo():
            return

        tag = self._speaker_tag(turn.speaker)
        print(f"\n  \033[1m[{tag}]\033[0m {turn.text}")

        if self._agent_active:
            _result, was_hit = self._speculator.on_final_turn(turn.text)
            if not was_hit and self._agent:
                self._agent.clear_speculative_cache()

        if self._detect_wake(turn.text):
            print("  \033[1;32m[wake detected!]\033[0m")
            if not self._agent_active:
                self._schedule(self._activate_agent())

        if self._agent_active:
            self._schedule(self._async_context_update())

    def _detect_wake(self, text: str) -> bool:
        normalized = text.lower().strip()
        normalized = normalized.replace(",", "").replace(".", "").replace("!", "")
        wake_words = config.WAKE_PHRASE.lower().split()
        return all(w in normalized.split() for w in wake_words)

    async def _activate_agent(self):
        if self._agent_active:
            return

        print("[engine] activating agent...")
        self._agent_active = True
        self._agent = AgentClient(self._audio)

        self._agent_task = asyncio.create_task(
            self._agent.connect(self._stop_event)
        )

        await asyncio.sleep(1.5)

        if self._agent and self._streaming_listener:
            summary = self._streaming_listener.transcript.summary()
            await self._agent.update_context(summary)
            self._last_context_update = time.monotonic()
            print("[engine] context sent to agent")

    async def _async_context_update(self):
        now = time.monotonic()
        if now - self._last_context_update < self._context_update_interval:
            return
        if self._agent and self._streaming_listener:
            summary = self._streaming_listener.transcript.summary()
            await self._agent.update_context(summary)
            self._last_context_update = now

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._audio.start(self._loop)

        streaming_audio_queue = self._audio.subscribe_streaming()
        self._streaming_listener = StreamingListener(
            audio_queue=streaming_audio_queue,
            on_partial_turn=self._on_partial_turn,
            on_final_turn=self._on_final_turn,
        )

        self._loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(self._shutdown()),
        )

        print(f"[engine] Sidebar is listening. Say \"{config.WAKE_PHRASE}\" to activate the agent.")
        print("[engine] Ctrl+C to stop.\n")

        await self._streaming_listener.run(self._stop_event)

    async def _shutdown(self):
        print("\n[engine] shutting down...")
        self._stop_event.set()

        stats = self._speculator.stats
        if stats.total_speculations > 0:
            print(f"[engine] {stats.summary()}")

        if self._agent:
            await self._agent.disconnect()
        if self._agent_task:
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

        self._audio.stop()
        print("[engine] done.")
