"""
Hello-world: Voice Agent API round-trip.

Opens a WebSocket to the Voice Agent API, configures a session with a greeting
and one tool, streams mic audio in, and plays back the agent's spoken replies.
Press Ctrl+C to stop.

Requires: ASSEMBLYAI_API_KEY in .env or environment.
"""

import asyncio
import base64
import json
import os
import signal
import sys

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv

load_dotenv()

AGENT_WS_URL = "wss://agents.assemblyai.com/v1/ws"
SAMPLE_RATE_IN = 24_000
SAMPLE_RATE_OUT = 24_000
CHUNK_DURATION_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE_IN * CHUNK_DURATION_MS / 1000)
PLAYBACK_BUFFER_FRAMES = 4800  # 200ms buffer to prevent ALSA underruns

api_key = os.environ.get("ASSEMBLYAI_API_KEY")
if not api_key:
    print("Error: set ASSEMBLYAI_API_KEY in .env or environment")
    sys.exit(1)

SESSION_CONFIG = {
    "type": "session.update",
    "session": {
        "system_prompt": (
            "You are a helpful voice assistant. Keep answers brief and conversational. "
            "You have a web_search tool you can use to look things up."
        ),
        "greeting": "Hey! I'm Sidebar. Ask me anything, or just say hi.",
        "output": {
            "voice": "anna",
        },
        "tools": [
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web for a factual query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        ],
    },
}


async def handle_tool_call(ws, call_id: str, name: str, arguments: str):
    print(f"  [tool call] {name}({arguments})")
    result = f"Sorry, {name} is not implemented yet in this hello-world demo."
    await ws.send(json.dumps({
        "type": "tool.result",
        "call_id": call_id,
        "result": result,
    }))
    print(f"  [tool result sent]")


async def send_audio(ws, stop_event: asyncio.Event):
    loop = asyncio.get_event_loop()

    def mic_callback(indata, _frames, _time, _status):
        if stop_event.is_set():
            return
        pcm_bytes = indata.tobytes()
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        asyncio.run_coroutine_threadsafe(
            ws.send(json.dumps({"type": "input.audio", "audio": b64})),
            loop,
        )

    with sd.InputStream(
        samplerate=SAMPLE_RATE_IN,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
        callback=mic_callback,
    ):
        await stop_event.wait()


async def receive_messages(ws, stop_event: asyncio.Event):
    playback_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE_OUT,
        channels=1,
        dtype="int16",
        blocksize=PLAYBACK_BUFFER_FRAMES,
    )
    playback_stream.start()

    try:
        async for raw_msg in ws:
            if stop_event.is_set():
                break

            msg = json.loads(raw_msg)
            msg_type = msg.get("type", "")

            if msg_type == "session.ready":
                session_id = msg.get("session_id", "?")
                print(f"[session ready — id: {session_id}]")
                print("Listening... speak into your mic. Ctrl+C to stop.\n")

            elif msg_type == "transcript.user.delta":
                text = msg.get("delta", "")
                print(f"\r  [you] {text}    ", end="", flush=True)

            elif msg_type == "transcript.user":
                text = msg.get("text", "")
                print(f"\n  \033[1m[you]\033[0m {text}")

            elif msg_type == "transcript.agent":
                text = msg.get("text", "")
                print(f"  \033[1;34m[agent]\033[0m {text}")

            elif msg_type == "reply.audio":
                audio_b64 = msg.get("data", "")
                if audio_b64:
                    pcm = np.frombuffer(
                        base64.b64decode(audio_b64), dtype=np.int16
                    )
                    playback_stream.write(pcm.reshape(-1, 1))

            elif msg_type == "reply.done":
                status = msg.get("status", "")
                if status == "interrupted":
                    print("  [interrupted by user]")

            elif msg_type == "tool.call":
                call_id = msg.get("call_id", "")
                name = msg.get("name", "")
                arguments = msg.get("arguments", "{}")
                await handle_tool_call(ws, call_id, name, arguments)

            elif msg_type == "error":
                print(f"  [error] {msg.get('message', msg)}")

    except websockets.ConnectionClosed:
        pass
    finally:
        playback_stream.stop()
        playback_stream.close()


async def shutdown(ws, stop_event, receive_task, send_task):
    print("\nDisconnecting...")
    stop_event.set()
    try:
        await ws.send(json.dumps({"type": "session.end"}))
    except websockets.ConnectionClosed:
        pass
    receive_task.cancel()
    send_task.cancel()
    for task in (receive_task, send_task):
        try:
            await task
        except asyncio.CancelledError:
            pass
    print("Done.")


async def main():
    headers = {"Authorization": f"Bearer {api_key}"}
    stop_event = asyncio.Event()

    print("Connecting to Voice Agent API...")
    async with websockets.connect(AGENT_WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(SESSION_CONFIG))

        receive_task = asyncio.create_task(receive_messages(ws, stop_event))
        send_task = asyncio.create_task(send_audio(ws, stop_event))

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(
                shutdown(ws, stop_event, receive_task, send_task)
            ),
        )

        await asyncio.gather(receive_task, send_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
