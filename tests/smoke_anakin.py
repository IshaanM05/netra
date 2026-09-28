#!/usr/bin/env python3
"""Live check of the Anakin.io web tools: one search, one scrape of the top result.

    python tests/smoke_anakin.py ["optional query"]
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.sidebar.tools import execute_tool, web_tools_enabled  # noqa: E402
from src.sidebar.viewer import RecordingHub  # noqa: E402


class _Sight:  # execute_tool reads the hub from sight.renderer
    renderer = RecordingHub()


def main():
    if not web_tools_enabled():
        sys.exit("ANAKIN_API_KEY is not set in .env")
    query = sys.argv[1] if len(sys.argv) > 1 else "Prusa MK4 clogged nozzle cold pull official guide"

    started = time.monotonic()
    raw = execute_tool("search_live", {"query": query}, _Sight())
    print(f"search_live {(time.monotonic() - started) * 1000:.0f}ms")
    try:
        results = json.loads(raw)["results"]
    except (ValueError, KeyError):
        sys.exit(f"search failed: {raw}")
    for r in results:
        print(f"  - {r['title']}\n    {r['url']}\n    {r['snippet'][:160]}")
    if not results:
        sys.exit("no results")

    started = time.monotonic()
    page = execute_tool("scrape_live", {"url": results[0]["url"]}, _Sight())
    print(f"\nscrape_live {(time.monotonic() - started) * 1000:.0f}ms")
    try:
        content = json.loads(page)["content"]
    except (ValueError, KeyError):
        sys.exit(f"scrape failed: {page}")
    print(f"  {len(content)} chars: {content[:300]!r}")
    print(f"\nviewer events: {[e['type'] for e in _Sight.renderer.events]}")


if __name__ == "__main__":
    main()
