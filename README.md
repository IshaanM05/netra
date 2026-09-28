# Netra — a voice agent that can see what you're working on

Hands full, eyes on the machine, manual out of reach. **Netra** is a voice assistant for hands-on repair and
software help. Ask **"Where's the heatbreak?"** and the part lights up in a live 3D model of your machine. Say
**"My extruder keeps clicking"** and it walks you through the fix one step at a time, highlighting every part
it mentions. It says safety notes first and checks official manuals on the web when its own knowledge runs
out.

- **One engine, any machine.** "Switch to my bike, I got a flat" swaps the model and starts the flat-tire fix
  in one sentence.
- **Learns machines it has never seen.** "I'm working on a Breville Barista Express" makes it find the
  official manual online, read it and build a parts-and-procedures pack in about 15 seconds, then tell you
  it's ready.
- **Netra Desktop.** The same idea for software: it reads your screen, draws a ring around the button you
  asked about and moves the pointer there. It never clicks.
- **Acts before you finish talking.** See *speculative execution* below.

Built for the **AssemblyAI Voice Agent Hackathon** (September 2026) on AssemblyAI's **Voice Agent API**
(speech in, reasoning, tool calls, speech out), **Universal-3.5 Pro streaming** (always-on ears) and
**LLM Gateway** (drafting packs for new machines), with **Anakin.io** for live web search and page reading.

---

## Speculative execution: acting while you're still talking

Most voice agents wait for you to finish, then think, then act. Netra scans Universal-Streaming **partial
transcripts** as you speak. When the intent is clear ("where's the heat…"), it fires the likely tool early:

| What you're saying | What Netra does before you finish |
|---|---|
| a part name ("where's the nozzle…") | resolves it against the machine and shows an **amber ghost highlight** in 3D |
| a problem ("my extruder keeps clicking…") | previews step 1 of the matching procedure and ghosts its parts |
| "look up the official guide for…" | starts the web search in the background once the query stops changing |
| "where's the Share button…" (Desktop) | triggers a fresh screen read and rings the label in amber |

When the Voice Agent API emits its real `tool.call`, Netra checks whether the speculation **matches
semantically**: same part, same procedure step, or an overlapping search query. A hit returns the ready
result instantly and the ghost turns solid cyan. A miss is dropped and the ghost cleared. Speculation only
touches **read-only** tools, so a wrong guess can't change anything. A live meter shows fires, hits, misses
and the average head start.

## How it works

```
             ┌──────────────── Universal-3.5 Pro streaming (always on) ────────────────┐
 mic ──┬───▶ │ wake word · room context · partial transcripts ─▶ SpeculativeExecutor    │──ghost──┐
       │     └──────────────────────────────────────────────────────────────────────────┘         │
       └───▶ Voice Agent API ── tool.call ─▶ tools ─▶ Sight adapter ─────────────────────────▶ what you see
                  ▲   speech out ◀───────────┘          │  ModelSight: machine packs ─▶ 3D viewer (browser)
                  │                                     │  ScreenSight: live screen  ─▶ on-screen ring + pointer
                  └── session: prompt, ≤10 tools,       └─ MachineLearner: web manual ─▶ LLM Gateway ─▶ new pack
                      keyterms + transcription prompt
                      built from the active machine
```

| Module | Role |
|---|---|
| `src/sidebar/engine.py` | Orchestrates both sockets: fuzzy wake word ("Hey Netra"), hand-over of the request spoken with the wake word, stop phrase, retry when the API is at capacity, machine switching |
| `src/sidebar/agent.py` | Voice Agent API client: documented `tool.result` timing, session built from the active machine (prompt, tools, keyterms, transcription prompt), spoken announcements |
| `src/sidebar/speculation.py` | Intent prediction on partial transcripts, ghost highlights, background pre-fetch, semantic `claim()` |
| `src/sidebar/sight.py` | Machine packs (`Manifest`), `MachineLibrary`, `ModelSight`: part resolution tolerant of speech-to-text errors, procedure state (next / back / repeat), side-effect-free `preview()` |
| `src/sidebar/learn.py` | Learns an unknown machine: Anakin search + scrape → LLM Gateway draft → validated, repaired pack |
| `src/sidebar/screen.py` · `overlay.py` | Netra Desktop: background OCR of the focused window, label matching, on-screen ring, pointer |
| `src/sidebar/viewer.py` · `viewer/index.html` | WebSocket hub and three.js viewer: highlight, ghost, exploded view, camera focus, procedure card, captions, sources, machine picker, speculation meter |
| `src/sidebar/tools.py` | Tool schemas and execution, including Anakin.io `search_live` / `scrape_live` |

### Machine packs

A machine is one YAML file in `manifests/`. It lists parts (with the names people actually say), symptom
vocabulary, safety notes and step-by-step procedures where each step names the parts it touches. The pack
drives answers, 3D highlights, speech-recognition key terms and early guessing. Two packs ship with live 3D
models:

- `manifests/sample-machine.yaml`: Prusa MK4 extruder (13 parts, 4 procedures)
- `manifests/bicycle.yaml`: bicycle drivetrain and brakes (13 parts, 4 procedures)

Learned packs land in `manifests/learned/` and show as a labelled schematic. Point `source:` at a `.glb` whose
node names match the part ids to use a real 3D model. `manifests/sample-app.yaml` gives Netra Desktop Figma
vocabulary and procedures.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y libportaudio2 portaudio19-dev   # Linux microphone access
cp .env.example .env                                     # add ASSEMBLYAI_API_KEY and ANAKIN_API_KEY
```

| Command | What you get |
|---|---|
| `python run.py --open` | Machines + 3D viewer. Say "Hey Netra, where's the nozzle?" |
| `python run.py --open --autostart` | Same, without needing the wake word |
| `python run.py --desktop` | Netra Desktop on whatever window is focused (Linux/X11) |
| `python run.py --desktop --manifest manifests/sample-app.yaml` | Desktop with Figma vocabulary and procedures |
| `python tests/rehearse_viewer.py` | Microphone-free scripted demo at http://127.0.0.1:8765 |

Use headphones: the agent's voice is gated from the mic while it speaks, but speakers in a quiet room still
leak. Say "Netra, stop" to end a session.

## Testing

```bash
python tests/test_sight.py        # offline, instant: part resolution, procedures, speculation, library, packs, desktop
python tests/eval_agent.py -j 8   # live: 51 multi-turn conversations against the real Voice Agent API
python tests/smoke_anakin.py      # live: one web search + one page read
```

`eval_agent.py` opens real Voice Agent API sessions with the exact configuration the app ships. It types each
user turn in (no text-to-speech needed), runs the real tools, and checks the tool chosen, the part or procedure
step targeted, what the viewer was told to show, and key phrases in the spoken reply. Coverage:

- part names and nicknames, including speech-to-text errors ("heat brake", "thermister")
- multi-turn procedures with next / back / repeat / switch, and safety-first replies
- refusing to invent parts that aren't in the pack
- live web lookups, preferring the machine's own procedures over the web
- switching machines together with a follow-up request
- learning a new machine from the web and then using it
- Desktop mode on a Figma screen

**Current result: 51/51 conversations, 71/71 turns; median request → `tool.call` ≈ 0.7 s.**

## Project status

**Working and tested**
- Field mode: 3D viewer, highlights, exploded view, guided procedures, safety-first replies
- Speculative execution with ghost highlights, web pre-fetch and a live hit/miss meter
- Machine library (extruder, bicycle), switching by voice or picker, switching plus follow-up in one turn
- Learning new machines from their manuals, with a spoken "it's ready"
- Live web grounding with cited sources (Anakin.io)
- Netra Desktop: background screen reading, on-screen ring, pointer, Figma procedures
- Offline and live test suites

**In progress for the submission**
- End-to-end rehearsals with a real microphone in a noisy room (noise suppression, speculation timing)
- Auto-stop after a period of silence
- Bundling the 3D library locally so the viewer works without internet
- Demo video and write-up

**Known limits**
- Netra Desktop currently supports Linux/X11
- Learned machines appear as a schematic, not a 3D model
- Drafting packs uses the LLM Gateway models available to the account; larger models give richer packs

## Voice Agent API notes (learned the hard way)

- `greeting: null` silently invalidates the whole `session.update`: no tools, no prompt. Use `""`.
- Send `tool.result` only while `reply.done` is the latest turn event. Hold results through `reply.started`
  and `input.speech.started`, and drop them when a reply is interrupted.
- `conversation.message` with `role: user` did not reach the model in our tests. `reply.create` with
  `instructions` does, and stays in the conversation history. Netra uses it to hand over the request spoken
  in the same breath as the wake word.
- Keep 10 tools or fewer per session. Netra scopes tools to the active adapter.
- `input.keyterms` works best with rare single words ("heatbreak", "thermistor"), not phrases.
