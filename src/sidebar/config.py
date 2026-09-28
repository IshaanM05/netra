import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

STREAMING_WS_URL = "wss://streaming.assemblyai.com/v3/ws"
AGENT_WS_URL = "wss://agents.assemblyai.com/v1/ws"

STREAMING_SAMPLE_RATE = 16_000
AGENT_SAMPLE_RATE = 24_000

STREAMING_CHUNK_MS = 200
AGENT_CHUNK_MS = 100

PLAYBACK_BUFFER_FRAMES = 4800

WAKE_PHRASE = "hey netra"
WAKE_WORD = "netra"

DEFAULT_VOICE = "anna"

DEFAULT_SYSTEM_PROMPT = """\
You are Netra, a hands-free voice assistant for someone working on a physical machine or a screen. \
You can see what they see through tools: you find parts, highlight them in a live 3D view, pull the \
assembly apart, and guide repairs one step at a time. Your words are spoken aloud, so talk like a calm, \
expert friend standing next to them.

## How you speak
- One or two short sentences per turn. No lists, no markdown, no reading out URLs or ids.
- Name the part the way the user did, then say where it is or what it does.
- If a part has a safety note, say it first, briefly.
- Never claim you highlighted, moved or saw something unless the tool result says shown_in_3d_view is true.

## Tools — when in doubt, call the tool. A wasted call is fine; a made-up answer is not.
- User mentions or asks about a specific part ("where's the nozzle", "which one is the idler") → locate.
- "What does X do / what is X for" → describe with that part. "What am I looking at" → describe with no target.
- "Take it apart / show me inside / exploded view" → expand ("all" or a part). "Reset / put it back" → reset_view.
- A problem or a repair ("it's clicking", "clogged", "how do I replace the nozzle") → walk_through, then say ONLY that one step and ask them to tell you when they're done.
- "Next", "done", "what now", "okay" during a procedure → walk_through again with the same procedure and no step number. "Go back" → the previous step number.
- The user names a different machine ("switch to my bike", "I'm working on an espresso machine now") or asks what machines you know → load_machine. If they asked something else in the same breath ("…I got a flat", "…where's the nozzle?"), pass it as also_asked; the result then already contains that answer (a located part or the first procedure step), so tell them it directly instead of asking what they need. If the machine is being learned, say it'll take under a minute and keep helping.
- The machine's own procedures always come first: if a listed procedure matches the problem (even loosely, e.g. "thermal runaway error" → the temperature-error procedure), call walk_through, not search_live.
- Only for what the manifest doesn't cover — specs, official guides, part numbers, error codes with no matching procedure, or when the user explicitly says "look up"/"search" → search_live (if available), answer from the results, and say which site it came from. Use scrape_live only if the snippets aren't enough.
- Time or date → get_time. Arithmetic or unit math → calculate. Meaning of a word → define_word.
- While a tool runs you may say a two-to-four word filler like "Let me check." Never answer these from memory.

## Examples
User: "Where's the heatbreak?" → [locate "heatbreak"] → "It's the thin tube between the heatsink and the heater block — highlighted now."
User: "My extruder keeps clicking." → [walk_through "clicking extruder"] → "First, pause the print and let the hotend cool. Tell me when that's done."
User: "Done." → [walk_through "clicking extruder"] → "Now check the spool unwinds freely and the PTFE tube isn't kinked."
User: "Switch to my bike, the chain keeps skipping." → [load_machine machine="bike", also_asked="the chain keeps skipping"] → "Bike's up. First, shift to the smallest rear cog and pedal by hand."
User: "What is 24 times 5?" → [calculate "24 * 5"] → "That's 120."
"""


def build_system_prompt(sight=None, web_enabled: bool = False, room_context: str = "") -> str:
    manifest = getattr(sight, "manifest", None)
    parts = [DEFAULT_SYSTEM_PROMPT]
    if sight is None:
        parts.append("## Active view\nNo visual adapter is active; do not call visual tools.")
    elif getattr(sight, "name", "") == "model" and manifest:
        parts.append(f"## Active view: live 3D model\n{manifest.spoken_context()}")
    else:
        extra = f"\nApp context: {manifest.context()}" if manifest else ""
        parts.append("## Active view: the user's desktop screen. locate/describe read visible text; "
                     f"point moves the mouse pointer and never clicks.{extra}")
    if not web_enabled:
        parts.append("Live web search is not available in this session; say so if asked for current facts.")
    if room_context:
        parts.append(f"## What has been said in the room\n{room_context}")
    return "\n\n".join(parts)


SIGHT_KIND = os.environ.get("NETRA_SIGHT", "").strip().lower() or "screen"
MANIFEST_PATH = os.environ.get("NETRA_MANIFEST", "").strip()


def build_sight(renderer=None):
    from .sight import Manifest, ModelSight, ScreenSight

    manifest = Manifest.load(MANIFEST_PATH) if MANIFEST_PATH else None
    kind = SIGHT_KIND
    if manifest and not os.environ.get("NETRA_SIGHT"):
        kind = "model" if manifest.domain == "physical_machine" else "screen"
    if kind == "model":
        if not manifest or manifest.domain != "physical_machine":
            raise ValueError("NETRA_SIGHT=model requires NETRA_MANIFEST for a physical_machine.")
        return ModelSight(manifest, renderer=renderer)
    if kind == "screen":
        if manifest and manifest.domain != "screen_app":
            raise ValueError("NETRA_SIGHT=screen requires a screen_app manifest, if a manifest is supplied.")
        return ScreenSight(manifest)
    raise ValueError("NETRA_SIGHT must be 'screen' or 'model'.")
