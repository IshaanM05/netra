"""
Hello-world: Universal-Streaming v3 with diarized speaker labels.

Captures mic audio and prints partial/final Turn events to the terminal,
tagged by speaker. Press Ctrl+C to stop.

Requires: ASSEMBLYAI_API_KEY in .env or environment.
"""

import os
import signal
import sys

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

load_dotenv()

SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 200
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

api_key = os.environ.get("ASSEMBLYAI_API_KEY")
if not api_key:
    print("Error: set ASSEMBLYAI_API_KEY in .env or environment")
    sys.exit(1)


def on_begin(_client, event: BeginEvent):
    print(f"\n[session started — id: {event.id}]")
    print("Listening... speak into your mic. Ctrl+C to stop.\n")


def on_turn(_client, event: TurnEvent):
    tag = "FINAL" if event.end_of_turn else "partial"
    speaker = getattr(event, "speaker_label", "?")
    prefix = f"[{tag}] Speaker {speaker}"

    if event.end_of_turn:
        print(f"\033[1m{prefix}:\033[0m {event.transcript}")
    else:
        print(f"\r{prefix}: {event.transcript}    ", end="", flush=True)


def on_termination(_client, event: TerminationEvent):
    duration = getattr(event, "audio_duration_seconds", "?")
    print(f"\n[session ended — audio duration: {duration}s]")


def main():
    client = StreamingClient(
        StreamingClientOptions(api_key=api_key)
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_termination)

    client.connect(
        StreamingParameters(
            sample_rate=SAMPLE_RATE,
            speech_model="universal-3-5-pro",
            speaker_labels=True,
        )
    )

    stop = False

    def on_sigint(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_sigint)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while not stop:
                data, _overflowed = stream.read(CHUNK_SAMPLES)
                client.stream(data.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nDisconnecting...")
        client.disconnect(terminate=True)
        print("Done.")


if __name__ == "__main__":
    main()
