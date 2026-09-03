# Sidebar

A voice agent you can pull into a conversation — as a silent co-pilot listening to two people, or as a 1:1 coach that adapts as you go.

Built on AssemblyAI's Universal-3.5 Pro streaming and Voice Agent API, with a speculative-execution layer that fires tool calls against partial transcripts before a turn finalizes.

**AssemblyAI Voice Agent Hackathon** · Sep 2026

## Architecture

Two independent AssemblyAI connections:

- **Universal-Streaming** — the ears. Always listening, diarizing both speakers, watching partial/final turns for wake-phrase and speculative-intent detection.
- **Voice Agent API** — the mouth and brain. Activated when the agent is addressed; owns STT, reasoning, tool calls, and speech from that point on.

## Modes

| | Co-pilot (primary) | Coach (stretch) |
|---|---|---|
| Setup | Two people talk normally | 1:1 with the user |
| Agent behavior | Silent by default; speaks only when addressed | Always in conversation; adapts live |
| AssemblyAI use | Universal-Streaming + Voice Agent API | Voice Agent API alone |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### System dependencies

```bash
# Ubuntu/Debian — PortAudio for mic access
sudo apt-get install -y libportaudio2 portaudio19-dev
```

### Configuration

```bash
cp .env.example .env
# Add your AssemblyAI API key to .env
```

## Quick start

```bash
# Test diarized streaming (mic → speaker-labeled transcripts)
python hello_streaming.py

# Test voice agent round-trip (mic → agent reply through speakers)
python hello_agent.py
```

Both scripts exit cleanly on Ctrl+C.

## Stack

- Python 3.10+
- `assemblyai` SDK — Universal-Streaming v3
- `websockets` — Voice Agent API (raw WebSocket)
- `sounddevice` + `numpy` — local audio I/O
