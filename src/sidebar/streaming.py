"""
Universal-Streaming listener: diarized ears.

Connects to AssemblyAI's Universal-3.5 Pro streaming, receives partial/final
Turn events with speaker labels, and maintains a rolling transcript of the
room conversation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

from . import config


@dataclass
class Turn:
    speaker: str
    text: str
    is_final: bool
    turn_order: int | None = None


@dataclass
class RoomTranscript:
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 50

    def add_final_turn(self, turn: Turn):
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def summary(self) -> str:
        if not self.turns:
            return "(no conversation yet)"
        lines = []
        for t in self.turns:
            lines.append(f"Speaker {t.speaker}: {t.text}")
        return "\n".join(lines)


class StreamingListener:
    def __init__(
        self,
        audio_queue: asyncio.Queue,
        on_partial_turn: Callable[[Turn], None] | None = None,
        on_final_turn: Callable[[Turn], None] | None = None,
    ):
        self._audio_queue = audio_queue
        self._on_partial_turn = on_partial_turn
        self._on_final_turn = on_final_turn
        self._transcript = RoomTranscript()
        self._client: StreamingClient | None = None
        self._stop = False

    @property
    def transcript(self) -> RoomTranscript:
        return self._transcript

    def _handle_begin(self, _client, event: BeginEvent):
        print(f"[streaming] session started — id: {event.id}")

    def _handle_turn(self, _client, event: TurnEvent):
        turn = Turn(
            speaker=getattr(event, "speaker_label", "?"),
            text=event.transcript,
            is_final=event.end_of_turn,
            turn_order=getattr(event, "turn_order", None),
        )

        if event.end_of_turn:
            self._transcript.add_final_turn(turn)
            if self._on_final_turn:
                self._on_final_turn(turn)
        else:
            if self._on_partial_turn:
                self._on_partial_turn(turn)

    def _handle_termination(self, _client, event: TerminationEvent):
        duration = getattr(event, "audio_duration_seconds", "?")
        print(f"[streaming] session ended — audio duration: {duration}s")

    async def run(self, stop_event: asyncio.Event):
        self._client = StreamingClient(
            StreamingClientOptions(api_key=config.API_KEY)
        )
        self._client.on(StreamingEvents.Begin, self._handle_begin)
        self._client.on(StreamingEvents.Turn, self._handle_turn)
        self._client.on(StreamingEvents.Termination, self._handle_termination)

        self._client.connect(
            StreamingParameters(
                sample_rate=config.STREAMING_SAMPLE_RATE,
                speech_model="universal-3-5-pro",
                speaker_labels=True,
                keyterms_prompt=["Sidebar", "Hey Sidebar"],
            )
        )

        try:
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.5
                    )
                    self._client.stream(chunk)
                except asyncio.TimeoutError:
                    continue
        finally:
            self._client.disconnect(terminate=True)
