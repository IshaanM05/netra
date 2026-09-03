#!/usr/bin/env python3
"""
Automated test harness for Voice Agent API prompt/tool iteration.
No voice playback — text responses and tool call verification only.
"""

import asyncio
import base64
import json
import sys

import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()

from src.sidebar.config import API_KEY, AGENT_WS_URL, DEFAULT_SYSTEM_PROMPT, DEFAULT_VOICE
from src.sidebar.tools import TOOL_DEFINITIONS, execute_tool

SAMPLE_RATE = 24_000


def text_to_pcm(text: str) -> bytes:
    import ctypes
    lib = ctypes.CDLL("libespeak-ng.so.1")
    lib.espeak_Initialize(1, 0, None, 0)
    lib.espeak_SetVoiceByName(b"en")
    lib.espeak_SetParameter(1, 150, 0)  # slower rate for clarity

    audio_chunks = []
    AUDIO_CB_TYPE = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p,
    )

    @AUDIO_CB_TYPE
    def callback(wav, numsamples, events):
        if wav and numsamples > 0:
            buf = (ctypes.c_short * numsamples)()
            ctypes.memmove(buf, wav, numsamples * 2)
            audio_chunks.append(bytes(buf))
        return 0

    lib.espeak_SetSynthCallback(callback)
    text_bytes = text.encode("utf-8")
    lib.espeak_Synth(text_bytes, len(text_bytes) + 1, 0, 0, 0, 0, None, None)
    lib.espeak_Synchronize()

    pcm_22k = b"".join(audio_chunks)
    samples = np.frombuffer(pcm_22k, dtype=np.int16).astype(np.float32)
    ratio = SAMPLE_RATE / 22050
    new_len = int(len(samples) * ratio)
    indices = np.arange(new_len) / ratio
    indices = np.clip(indices, 0, len(samples) - 1)
    resampled = np.interp(indices, np.arange(len(samples)), samples)
    return resampled.astype(np.int16).tobytes()


def pcm_to_b64_chunks(pcm: bytes, chunk_ms: int = 100) -> list[str]:
    chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * 2
    chunks = []
    for i in range(0, len(pcm), chunk_bytes):
        chunk = pcm[i:i + chunk_bytes]
        if len(chunk) < chunk_bytes:
            chunk += b"\x00" * (chunk_bytes - len(chunk))
        chunks.append(base64.b64encode(chunk).decode("ascii"))
    return chunks


TEST_CASES = [
    # --- get_time: direct ---
    {"query": "What time is it?", "expect_tool": "get_time"},
    {"query": "What is today's date?", "expect_tool": "get_time"},
    {"query": "What day of the week is it?", "expect_tool": "get_time"},
    {"query": "What time is it in India?", "expect_tool": "get_time"},
    # --- get_time: indirect ---
    {"query": "Do you know what time it is right now?", "expect_tool": "get_time"},
    {"query": "Tell me the current date.", "expect_tool": "get_time"},
    # --- calculate: direct ---
    {"query": "What is 24 times 5?", "expect_tool": "calculate"},
    {"query": "Calculate 100 divided by 7.", "expect_tool": "calculate"},
    {"query": "What is the square root of 144?", "expect_tool": "calculate"},
    {"query": "How much is 15 percent of 200?", "expect_tool": "calculate"},
    # --- calculate: indirect ---
    {"query": "If I have 3 dozen eggs, how many is that?", "expect_tool": "calculate"},
    {"query": "What is 99 plus 1?", "expect_tool": "calculate"},
    {"query": "How much do I tip on a 50 dollar bill at 20 percent?", "expect_tool": "calculate"},
    # --- define_word: direct ---
    {"query": "What does serendipity mean?", "expect_tool": "define_word"},
    {"query": "What does ubiquitous mean?", "expect_tool": "define_word"},
    # --- define_word: indirect ---
    {"query": "I heard the word pragmatic, what does that mean?", "expect_tool": "define_word"},
    {"query": "Can you tell me what ephemeral means?", "expect_tool": "define_word"},
    # --- edge: ambiguous phrasing ---
    {"query": "What is 7 to the power of 3?", "expect_tool": "calculate"},
    {"query": "What is the time in Tokyo right now?", "expect_tool": "get_time"},
    {"query": "What does the word cogent mean?", "expect_tool": "define_word"},
]


async def drain_events(ws, timeout=0.5):
    """Drain any buffered events, discarding them."""
    while True:
        try:
            await asyncio.wait_for(ws.recv(), timeout=timeout)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            break


async def send_and_wait(ws, query: str) -> dict:
    """Send audio for one query and wait for the complete response cycle."""
    pcm = text_to_pcm(query)
    chunks = pcm_to_b64_chunks(pcm)

    # Send speech audio
    for chunk in chunks:
        await ws.send(json.dumps({"type": "input.audio", "audio": chunk}))
        await asyncio.sleep(0.08)

    # Send 2s silence to trigger end-of-turn
    silence = np.zeros(int(SAMPLE_RATE * 2), dtype=np.int16).tobytes()
    for chunk in pcm_to_b64_chunks(silence):
        await ws.send(json.dumps({"type": "input.audio", "audio": chunk}))
        await asyncio.sleep(0.08)

    # Now collect events until reply.done
    tool_calls = []
    heard = ""
    agent_text = ""
    deadline = asyncio.get_event_loop().time() + 20

    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
        except asyncio.TimeoutError:
            break
        except websockets.ConnectionClosed:
            break

        msg = json.loads(raw)
        t = msg.get("type", "")

        if t == "transcript.user":
            heard = msg.get("text", "")
        elif t == "tool.call":
            call_id = msg.get("call_id", "")
            name = msg.get("name", "")
            args = msg.get("arguments", "{}")
            tool_calls.append(name)
            result = execute_tool(name, args)
            await ws.send(json.dumps({
                "type": "tool.result", "call_id": call_id, "result": result,
            }))
        elif t == "transcript.agent":
            agent_text = msg.get("text", "")
        elif t == "reply.done":
            break

    # Drain any trailing audio events
    await drain_events(ws, timeout=1.0)

    return {"heard": heard, "tool_calls": tool_calls, "agent_response": agent_text}


async def run_tests():
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with websockets.connect(AGENT_WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "greeting": "",
                "output": {"voice": DEFAULT_VOICE},
                "tools": TOOL_DEFINITIONS,
            },
        }))

        # Wait for session.ready
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "session.ready":
                tools = msg.get("config", {}).get("tools", [])
                print(f"[session ready — {len(tools)} tools registered]\n")
                break
            elif msg.get("type") == "session.error":
                print(f"SESSION ERROR: {msg}")
                return

        results = []
        for i, tc in enumerate(TEST_CASES):
            query = tc["query"]
            expect = tc["expect_tool"]

            print(f"  [{i+1}/{len(TEST_CASES)}] \"{query}\"", end=" ... ", flush=True)

            r = await send_and_wait(ws, query)

            hit = expect in r["tool_calls"]
            status = "\033[32mPASS\033[0m" if hit else "\033[31mFAIL\033[0m"
            print(f"{status}  heard=\"{r['heard']}\"  tools={r['tool_calls']}  response=\"{r['agent_response'][:60]}\"")

            results.append({**tc, **r, "hit": hit})

        await ws.send(json.dumps({"type": "session.end"}))

    hits = sum(1 for r in results if r["hit"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  Tool-call accuracy: {hits}/{total} ({hits/total*100:.0f}%)")
    print(f"{'='*60}")

    if hits < total:
        print("\n  Failures:")
        for r in results:
            if not r["hit"]:
                print(f"    - \"{r['query']}\" → heard=\"{r['heard']}\" tools={r['tool_calls']}")


if __name__ == "__main__":
    if not API_KEY:
        print("Error: set ASSEMBLYAI_API_KEY in .env")
        sys.exit(1)
    asyncio.run(run_tests())
