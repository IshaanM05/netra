# Netra — a voice agent that can see what you're working on

Hands full, eyes on the machine, manual out of reach. Netra is a voice assistant for hands-on repair: ask
**"Where's the heatbreak?"** and the part lights up in a live 3D model of your machine. Say **"My extruder
keeps clicking"** and it walks you through the fix one step at a time, highlighting each part as you go.
It pulls the assembly apart on request, checks the official docs on the web, and says safety notes first.

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

**Your own machine.** Write a manifest (see `manifests/sample-machine.yaml`) with parts, aliases, vocab and
step-by-step procedures, where each step names the parts it touches. Point `source:` at a `.glb` whose node
names match the part ids, or keep the built-in procedural extruder.

**Netra Desktop.** `python run.py --sight screen --manifest ""` reads the screen with local OCR and moves the
pointer to what you ask about (it never clicks). Needs `pyautogui`, `pytesseract` and Tesseract.

## Testing

```bash
python tests/test_sight.py         # offline: resolution (incl. STT typos), procedures, speculation, wake parsing
python tests/eval_agent.py -j 6    # live: 36 multi-turn scenarios against the real Voice Agent API session
python tests/smoke_anakin.py       # live: one search + one scrape through Anakin.io
python tests/rehearse_viewer.py    # mic-free scripted demo in the viewer (http://127.0.0.1:8765)
```

`eval_agent.py` opens real sessions with the exact config the app ships and injects user turns as text. It
runs the real tools and asserts the tool chosen, the part or procedure step targeted, the viewer events and
key phrases in the reply. It covers aliases, speech-to-text typos ("heat brake", "thermister"), multi-turn
procedures with next / back / repeat / switch, safety-first replies, and refusing to invent parts that
aren't in the manifest, plus live web lookups. Current result: **40/40 scenarios, 53/53 turns; request → `tool.call` p50 ≈ 0.6 s**.

## Voice Agent API notes (learned the hard way)

- `greeting: null` silently invalidates the whole `session.update`: no tools, no prompt. Use `""`.
- Send `tool.result` only while `reply.done` is the latest turn event; hold results through
  `reply.started` / `input.speech.started`, and drop them when a reply is interrupted.
- `conversation.message` with `role: user` did not reach the model in our tests. `reply.create` with
  `instructions` does, and stays in the conversation history. Netra uses this to hand over the request
  spoken in the same breath as the wake word.
- Keep 10 tools or fewer per session; Netra drops `get_time` / `define_word` in Field mode.
