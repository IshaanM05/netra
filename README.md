# Netra

Netra is an AssemblyAI Voice Agent API demo for a voice agent that can inspect a visual field and act through pluggable Sight adapters. Netra Desktop reads screen text locally with OCR and can point the cursor at a match. Netra Field loads a machine manifest and exposes part lookup, descriptions, and procedure guidance. The same voice session provides speech recognition, turn taking, reasoning, tool calls, and spoken output over AssemblyAI's Voice Agent API.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set ASSEMBLYAI_API_KEY in .env
python run.py
```

Say “Hey Netra” to activate the conversational agent. Install PortAudio on Linux for microphone access:

```bash
sudo apt-get install -y libportaudio2 portaudio19-dev
```

## Netra Desktop

Set `NETRA_SIGHT=screen` (the default) and leave `NETRA_MANIFEST` blank for generic desktop use. The screen adapter uses local Tesseract OCR to read visible text and `pyautogui` to move the pointer; it never clicks. Install the optional Python packages with `pip install pyautogui pytesseract` and install the Tesseract binary for OCR support. On macOS, grant Screen Recording and Accessibility permissions to the terminal. The adapter also attempts to read accessible element names through AppleScript.

To use app-specific vocabulary and procedures, set `NETRA_MANIFEST=manifests/sample-app.yaml`. Desktop does not require a manifest.

## Netra Field

Set `NETRA_SIGHT=model` and `NETRA_MANIFEST=manifests/sample-machine.yaml`. A physical machine manifest defines its model source, parts, vocabulary, and procedures. The current repository has no 3D renderer, so Field actions resolve against the manifest and report the part identifiers an attached renderer would use; they do not claim to move or highlight geometry.

Both manifest examples use the same YAML shape:

```yaml
manifest:
  domain: physical_machine # or screen_app
  name: Example machine
  source: assets/example.glb # physical_machine only
  parts: []                 # physical_machine only
  vocab: []
  procedures: []
```

## Tools and configuration

Netra Core registers `locate`, `point`, `expand`, `describe`, `walk_through`, `search_live`, and `scrape_live` with AssemblyAI's JSON-schema tool calling, alongside the original time, calculation, and dictionary tools. Set `ANAKIN_API_KEY` to enable web search and URL extraction through Anakin.io. Search returns structured results with source URLs; scrape extracts markdown from a specific HTTP or HTTPS URL. Without a key, the agent reports that live web tools are not configured.

The wake phrase listener uses AssemblyAI Universal-3.5 Pro streaming for always-on room transcription. The Voice Agent API handles the active voice conversation. This keeps the existing co-pilot behavior while the active agent uses AssemblyAI's end-to-end voice stack.

## Other examples

```bash
python hello_streaming.py # diarized Universal-Streaming transcript
python hello_agent.py     # standalone AssemblyAI Voice Agent API sample
```

The Voice Agent API expects 24 kHz mono PCM16 audio, and the live desktop adapter depends on local OS permissions and optional OCR/pointer packages. Visual support is intentionally capability-aware: current screen actions work on OCR text and the pointer, while model actions work against manifest data until a 3D renderer is connected.
