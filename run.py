#!/usr/bin/env python3
"""Run the Sidebar engine."""

import asyncio
import sys

from src.sidebar.config import API_KEY
from src.sidebar.engine import SidebarEngine


def main():
    if not API_KEY:
        print("Error: set ASSEMBLYAI_API_KEY in .env or environment")
        sys.exit(1)

    engine = SidebarEngine()
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
