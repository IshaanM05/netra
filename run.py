#!/usr/bin/env python3
"""Run the Netra voice agent.

    python run.py                    # Field demo: sample extruder + 3D viewer, say "Hey Netra"
    python run.py --autostart --open # start talking immediately and open the viewer
    python run.py --sight screen --manifest ""   # Netra Desktop (OCR + pointer)
"""

import argparse
import asyncio
import os
import sys
import webbrowser

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MANIFEST = "manifests/sample-machine.yaml"


def main():
    parser = argparse.ArgumentParser(description="Netra: a voice agent that can see what you're working on.")
    parser.add_argument("--manifest", default=os.environ.get("NETRA_MANIFEST", DEFAULT_MANIFEST),
                        help="machine/app manifest YAML ('' for none)")
    parser.add_argument("--sight", default=os.environ.get("NETRA_SIGHT", ""), choices=["", "model", "screen"],
                        help="visual adapter; blank infers it from the manifest")
    parser.add_argument("--autostart", action="store_true", help="start the agent without the wake phrase")
    parser.add_argument("--open", action="store_true", help="open the 3D viewer in the browser")
    args = parser.parse_args()

    # config reads these at import time
    os.environ["NETRA_MANIFEST"] = args.manifest
    os.environ["NETRA_SIGHT"] = args.sight
    if args.autostart:
        os.environ["NETRA_AUTOSTART"] = "1"

    from src.sidebar.config import API_KEY
    from src.sidebar.engine import NetraEngine

    if not API_KEY:
        print("Error: set ASSEMBLYAI_API_KEY in .env or environment")
        sys.exit(1)
    if args.open:
        webbrowser.open("http://127.0.0.1:8765/")
    asyncio.run(NetraEngine().run())


if __name__ == "__main__":
    main()
