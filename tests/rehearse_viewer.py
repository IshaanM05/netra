#!/usr/bin/env python3
"""
Drive the 3D viewer without a microphone: starts the real ViewerHub and plays a
scripted demo through the real ModelSight and SpeculativeExecutor.

    python tests/rehearse_viewer.py            # then open http://127.0.0.1:8765
    python tests/rehearse_viewer.py --loop     # repeat forever (booth / screenshots)
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sidebar.sight import Manifest, ModelSight  # noqa: E402
from src.sidebar.speculation import SpeculativeExecutor  # noqa: E402
from src.sidebar.viewer import ViewerHub  # noqa: E402


async def say(hub, who, text, pause=1.2):
    hub.broadcast({"type": "transcript", "who": who, "text": text})
    await asyncio.sleep(pause)


async def partials(hub, spec, text, word_ms=260):
    words = text.split()
    for i in range(1, len(words) + 1):
        partial = " ".join(words[:i])
        hub.broadcast({"type": "partial", "text": partial})
        spec.on_partial_turn(partial)
        await asyncio.sleep(word_ms / 1000)
    spec.on_final_turn(text)
    await asyncio.sleep(0.4)


async def tool(hub, spec, sight, name, args):
    hub.broadcast({"type": "tool", "name": name, "args": args, "phase": "call"})
    hit = spec.claim(name, args)
    sight.act(name, args)
    hub.broadcast({"type": "tool", "name": name, "phase": "result", "ms": 3,
                   **({"speculative": True, "lead_ms": round(hit.lead_ms)} if hit else {})})


async def script(hub, sight, spec):
    hub.broadcast({"type": "reset"})
    hub.broadcast({"type": "status", "state": "active"})
    await asyncio.sleep(2)
    await partials(hub, spec, "Where's the heatbreak on this thing?")
    await asyncio.sleep(0.6)  # agent thinking
    await tool(hub, spec, sight, "locate", {"query": "heatbreak"})
    await say(hub, "agent", "It's the thin tube between the heatsink and the heater block — highlighted now.", 4)

    await partials(hub, spec, "Show me how it all comes apart.")
    await tool(hub, spec, sight, "expand", {"target": "all"})
    await say(hub, "agent", "Here's the exploded view.", 4)

    await partials(hub, spec, "My extruder keeps clicking, can you walk me through it?")
    await asyncio.sleep(0.6)
    await tool(hub, spec, sight, "walk_through", {"procedure": "clicking extruder"})
    await say(hub, "agent", "First, pause the print and let the hotend cool. Tell me when that's done.", 4)
    await partials(hub, spec, "Done, what's next?")
    await tool(hub, spec, sight, "walk_through", {"procedure": "clicking extruder"})
    await say(hub, "agent", "Check the spool unwinds freely and the PTFE tube isn't kinked.", 4)
    await partials(hub, spec, "Okay. Next.")
    await tool(hub, spec, sight, "walk_through", {"procedure": "clicking extruder"})
    await say(hub, "agent", "Open the idler lever and brush any filament dust off the drive gears.", 5)

    hub.broadcast({"type": "sources", "query": "Prusa MK4 clogged nozzle", "results": [
        {"title": "Clogged nozzle / hotend (MK4) | Prusa Knowledge Base",
         "url": "https://help.prusa3d.com/article/clogged-nozzle-hotend-mk4", "snippet": ""}]})
    await partials(hub, spec, "Thanks. Reset the view.")
    await tool(hub, spec, sight, "reset_view", {})
    await asyncio.sleep(3)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--wait", type=float, default=6, help="seconds to wait for the browser first")
    args = parser.parse_args()
    manifest = Manifest.load(ROOT / "manifests" / "sample-machine.yaml")
    hub = ViewerHub()
    await hub.start(manifest)
    sight = ModelSight(manifest, renderer=hub)
    spec = SpeculativeExecutor(sight, hub)
    print(f"viewer: {hub.url}")
    await asyncio.sleep(args.wait)
    while True:
        await script(hub, sight, spec)
        print(spec.stats.summary())
        if not args.loop:
            break
    await asyncio.sleep(3600 if args.loop else 5)
    await hub.stop()


if __name__ == "__main__":
    asyncio.run(main())
