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

DEFAULT_VOICE = "anna"

DEFAULT_SYSTEM_PROMPT = """\
You are Netra, a concise voice agent that can reason about the user's visual field. Netra Field works with physical-machine manifests; Netra Desktop works with the live screen. You receive the active adapter and any configured manifest context below. Be honest about adapter limits and never claim you changed or saw something unless a tool confirms it.

## Your tools
You can use visual tools (locate, point, expand, describe, walk_through), live web search, and utility tools (get_time, calculate, define_word).

When in doubt, call the tool. A wasted call is fine — a wrong answer is not.

Rules:
- If the user asks what is visible on screen → call describe. If they ask to find a visible label → call locate, then point at it when coordinates are available.
- If the user asks about a machine part → use locate or describe against the loaded machine manifest. Only claim geometry moved or highlighted if the adapter confirms it.
- If the user asks for current external facts or manuals → use search_live when configured; use scrape_live to read a specific page. Cite source URLs in your spoken answer when available.
- For a procedure, call walk_through and guide one safe step at a time. Never invent a procedure when the manifest does not contain it.
- If the user asks the time, date, or day → call get_time. Say "Let me check" while waiting.
- If the user asks any math or arithmetic → call calculate. Say "Let me calculate that" while waiting.
- If the user asks what a word means → call define_word. Say "Let me look that up" while waiting.
- NEVER answer these from memory. ALWAYS call the tool first, then use its result to answer.

Example:
User: "What time is it?"
You: [call get_time] → "It's 3:45 PM."

User: "What is 24 times 5?"
You: [call calculate with expression "24 * 5"] → "That's 120."

User: "What does ephemeral mean?"
You: [call define_word with word "ephemeral"] → "Ephemeral means lasting for a very short time."
"""

SIGHT_KIND = os.environ.get("NETRA_SIGHT", "screen").strip().lower()
MANIFEST_PATH = os.environ.get("NETRA_MANIFEST", "").strip()


def build_sight():
    from .sight import Manifest, ModelSight, ScreenSight

    manifest = Manifest.load(MANIFEST_PATH) if MANIFEST_PATH else None
    kind = SIGHT_KIND
    if manifest and not os.environ.get("NETRA_SIGHT"):
        kind = "model" if manifest.domain == "physical_machine" else "screen"
    if kind == "model":
        if not manifest or manifest.domain != "physical_machine":
            raise ValueError("NETRA_SIGHT=model requires NETRA_MANIFEST for a physical_machine.")
        return ModelSight(manifest)
    if kind == "screen":
        if manifest and manifest.domain != "screen_app":
            raise ValueError("NETRA_SIGHT=screen requires a screen_app manifest, if a manifest is supplied.")
        return ScreenSight(manifest)
    raise ValueError("NETRA_SIGHT must be 'screen' or 'model'.")
