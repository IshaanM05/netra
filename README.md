# Netra — a voice agent that can see what you're working on

Hands full, eyes on the machine, manual out of reach. Netra is a voice assistant for hands-on repair: ask
**"Where's the heatbreak?"** and the part lights up in a live 3D model of your machine. Say **"My extruder
keeps clicking"** and it walks you through the fix one step at a time, highlighting each part as you go.
It pulls the assembly apart on request, checks the official docs on the web, and says safety notes first.

**One engine, any machine.** Say "switch to my bike" and the view, vocabulary and procedures change. Name a
machine Netra has never seen ("I'm working on a Breville Barista Express") and it **learns it live**: it finds the
official manual on the web, reads it, drafts a parts-and-procedures pack in about 15 seconds, and tells you when
it's ready. **Netra Desktop** does the same for software: it reads your screen, rings the button you asked
about and moves the pointer there.

Built for the **AssemblyAI Voice Agent Hackathon** on AssemblyAI's **Voice Agent API** (speech in, reasoning,
tool calls, speech out) and **Universal-3.5 Pro streaming** (always-on ears).

## What makes it different: speculative execution

Most voice agents wait for you to finish talking, then think, then act. Netra acts **while you're still
talking**. Universal-Streaming partial transcripts are scanned as you speak; when the intent is clear
("where's the heat…"), Netra fires the likely tool early:

- **Visual tools** (`locate`, `describe`, `walk_through`) are previewed against the machine manifest and shown
  as an amber **ghost highlight** before you finish the sentence.
- **Web search** is started in the background once the query stops changing, so the slowest call is already
  running when the agent decides to use it.

When the Voice Agent API emits its real `tool.call`, Netra checks whether the speculation matches
semantically (same part, same procedure step, overlapping search query). A hit returns the ready result
instantly and the ghost turns solid. A miss is discarded and the ghost cleared. Speculation is limited to
read-only tools, so a wrong guess can't change anything. The viewer shows fires, hits, misses and the
average head start live.

## Architecture

```
            ┌───────────────── Universal-3.5 Pro streaming (always on) ─────────────────┐
 mic ──┬──▶ │ wake phrase · room context · partial transcripts ─▶ SpeculativeExecutor    │──ghost──┐
       │    └────────────────────────────────────────────────────────────────────────────┘         │
       └──▶ Voice Agent API ── tool.call ─▶ tools ─▶ Sight adapter (ModelSight / ScreenSight) ─▶ ViewerHub ─▶ 3D viewer
                 ▲    speech out ◀──────────┘         ▲ manifest: parts · aliases · vocab · procedures
                 └── session: prompt, ≤10 tools, keyterms + transcription prompt from the manifest
```

| Module | Role |
|---|---|
| `src/sidebar/engine.py` | Orchestrates both sockets; fuzzy wake word ("Hey Netra"), wake-turn handoff, stop phrase, capacity retry |
| `src/sidebar/agent.py` | Voice Agent API client; documented `tool.result` timing; manifest-driven `keyterms` / `transcription_prompt` |
| `src/sidebar/speculation.py` | Intent prediction on partials, ghost highlights, background pre-fetch, semantic `claim()` |
| `src/sidebar/sight.py` | Manifest loader; `ModelSight` (phrase + fuzzy part resolution, procedure state, side-effect-free `preview`) and `ScreenSight` (OCR + pointer) |
| `src/sidebar/viewer.py` · `viewer/index.html` | WebSocket hub + three.js viewer (highlight, ghost, explode, focus, step card, transcript, sources, speculation meter) |
| `src/sidebar/tools.py` | Tool schemas and execution, including Anakin.io `search_live` / `scrape_live` |
| `src/sidebar/learn.py` | Learns an unknown machine from its manual: Anakin search/scrape → LLM Gateway draft → validated manifest |
| `src/sidebar/screen.py` · `overlay.py` | Netra Desktop: background OCR of the focused window, fuzzy label matching, on-screen ring, pointer |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y libportaudio2 portaudio19-dev   # Linux mic access
cp .env.example .env                                     # add ASSEMBLYAI_API_KEY (and ANAKIN_API_KEY for web)
python run.py --open                                     # say "Hey Netra, where's the heatbreak?"
```

`--autostart` skips the wake phrase. Say "Netra, stop" to end a session. Use headphones on stage (the
agent's voice is also gated from the mic while it speaks).

**Machine library.** Every `physical_machine` YAML in `manifests/` is a machine: the 3D-printer extruder and a
bicycle drivetrain ship as live 3D models. "Switch to my bike, I got a flat" loads the bike *and* starts the
flat-tire procedure in one turn (`load_machine` with `also_asked`). You can also pick from the viewer's dropdown.

**Learning new machines.** `load_machine` for an unknown machine starts `MachineLearner` in the background
(`src/sidebar/learn.py`): Anakin search → read the top two pages (headless-browser fallback for JS sites) →
AssemblyAI **LLM Gateway** drafts a manifest → validation and repair (ids, step→part references, junk safety
text) → saved to `manifests/learned/`. The viewer shows progress, the agent announces "it's ready", and the
machine appears as a labelled schematic you can locate parts on and walk procedures with.

**Your own machine.** Write a manifest (see `manifests/sample-machine.yaml`) with parts, aliases, vocab and
step-by-step procedures, where each step names the parts it touches. Point `source:` at a `.glb` whose node
names match the part ids, or keep the built-in procedural extruder.

**Netra Desktop.** `python run.py --desktop` (optionally `--manifest manifests/sample-app.yaml` for Figma vocabulary
and procedures). A background reader re-OCRs the focused window only when its pixels change (RapidOCR, pip-only,
no system Tesseract). Speculation triggers a fresh read the moment you start asking, so `locate` usually answers
from a reading that's already done (3 ms in our test vs about 2.5 s for a cold OCR pass). The target gets a
cyan ring on screen (amber while predicting), and the pointer moves to it. It never clicks. Linux/X11.

## Testing

```bash
python tests/test_sight.py         # offline: resolution, procedures, speculation, library, learned-pack repair, desktop
python tests/eval_agent.py -j 8    # live: 51 multi-turn scenarios (web + learning ones need ANAKIN_API_KEY) against the real Voice Agent API session
python tests/smoke_anakin.py       # live: one search + one scrape through Anakin.io
python tests/rehearse_viewer.py    # mic-free scripted demo in the viewer (http://127.0.0.1:8765)
```

`eval_agent.py` opens real sessions with the exact config the app ships and injects user turns as text. It
runs the real tools and asserts the tool chosen, the part or procedure step targeted, the viewer events and
key phrases in the reply. It covers aliases, speech-to-text typos ("heat brake", "thermister"), multi-turn
procedures with next / back / repeat / switch, safety-first replies, and refusing to invent parts that
aren't in the manifest, live web lookups, machine switching with follow-up requests, learning a new machine from
the web and then using it, and Desktop mode on a Figma screen. Current result: **51/51 scenarios, 71/71 turns;
request → `tool.call` p50 ≈ 0.7 s**.

## Voice Agent API notes (learned the hard way)

- `greeting: null` silently invalidates the whole `session.update`: no tools, no prompt. Use `""`.
- Send `tool.result` only while `reply.done` is the latest turn event; hold results through
  `reply.started` / `input.speech.started`, and drop them when a reply is interrupted.
- `conversation.message` with `role: user` did not reach the model in our tests. `reply.create` with
  `instructions` does, and stays in the conversation history. Netra uses this to hand over the request
  spoken in the same breath as the wake word.
- Keep 10 tools or fewer per session; Netra drops `get_time` / `define_word` in Field mode.
