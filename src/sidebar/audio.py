"""
Audio I/O: shared mic capture and speaker playback.

Captures mic at 16kHz for streaming and resamples to 24kHz for the agent.
Playback at 24kHz with basic echo suppression (mutes mic forwarding to agent
while agent audio is playing).
"""

import asyncio
import base64
import threading
import time

import numpy as np
import sounddevice as sd

from . import config


class AudioIO:
    def __init__(self):
        self._streaming_subscribers: list[asyncio.Queue] = []
        self._agent_subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

        self._playback_stream: sd.OutputStream | None = None
        self._input_stream: sd.InputStream | None = None

        self._agent_speaking = False
        self._agent_speak_end_time = 0.0
        self._echo_guard_ms = 1500

    def subscribe_streaming(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._streaming_subscribers.append(q)
        return q

    def subscribe_agent(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._agent_subscribers.append(q)
        return q

    def _is_echo_suppressed(self) -> bool:
        if self._agent_speaking:
            return True
        return time.monotonic() < self._agent_speak_end_time

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if self._loop is None:
            return

        pcm16 = indata.copy()

        for q in self._streaming_subscribers:
            try:
                q.put_nowait(pcm16.tobytes())
            except asyncio.QueueFull:
                pass

        if not self._is_echo_suppressed():
            resampled = self._resample_16k_to_24k(pcm16)
            b64 = base64.b64encode(resampled).decode("ascii")
            for q in self._agent_subscribers:
                try:
                    q.put_nowait(b64)
                except asyncio.QueueFull:
                    pass

    @staticmethod
    def _resample_16k_to_24k(pcm16_bytes: np.ndarray) -> bytes:
        samples = pcm16_bytes.flatten().astype(np.float32)
        ratio = config.AGENT_SAMPLE_RATE / config.STREAMING_SAMPLE_RATE
        new_len = int(len(samples) * ratio)
        indices = np.arange(new_len) / ratio
        indices = np.clip(indices, 0, len(samples) - 1)
        resampled = np.interp(indices, np.arange(len(samples)), samples)
        return resampled.astype(np.int16).tobytes()

    def play_audio(self, pcm_b64: str):
        if not pcm_b64 or self._playback_stream is None:
            return
        self._agent_speaking = True
        pcm = np.frombuffer(base64.b64decode(pcm_b64), dtype=np.int16)
        self._playback_stream.write(pcm.reshape(-1, 1))

    def mark_agent_done_speaking(self):
        self._agent_speaking = False
        self._agent_speak_end_time = time.monotonic() + (self._echo_guard_ms / 1000)

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

        chunk_samples = int(
            config.STREAMING_SAMPLE_RATE * config.STREAMING_CHUNK_MS / 1000
        )

        self._input_stream = sd.InputStream(
            samplerate=config.STREAMING_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=chunk_samples,
            callback=self._mic_callback,
        )

        self._playback_stream = sd.OutputStream(
            samplerate=config.AGENT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=config.PLAYBACK_BUFFER_FRAMES,
        )

        self._input_stream.start()
        self._playback_stream.start()

    def stop(self):
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
        if self._playback_stream:
            self._playback_stream.stop()
            self._playback_stream.close()
