"""
Tool definitions and execution for the Voice Agent API.

Tools use the flat schema format: {type, name, description, parameters}.
"""

import json
import math
import re
import os
import time

import httpx
from .sight import SIGHT_TOOLS, TOOL_DEFINITIONS as SIGHT_TOOL_DEFINITIONS

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "calculate",
        "description": (
            "Call this when the user asks any math question, arithmetic, or calculation. "
            "Triggers: 'what is X times Y', 'calculate', 'how much is', 'X plus Y', "
            "'X divided by Y', 'square root of', 'percent of'. "
            "Do NOT answer math from memory — always call this tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '24 * 5' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "type": "function",
        "name": "get_time",
        "description": (
            "Call this when the user asks about the current time, date, or day. "
            "Triggers: 'what time is it', 'what is today', 'what day is it', "
            "'current time', 'what date is it', 'time in [place]'. "
            "Do NOT guess the time — always call this tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone name if a specific location is mentioned, e.g. Asia/Kolkata",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "define_word",
        "description": (
            "Call this when the user asks for the definition or meaning of a word. "
            "Triggers: 'what does X mean', 'define X', 'meaning of X', 'what is X'. "
            "Do NOT define words from memory — always call this tool first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The word to look up",
                }
            },
            "required": ["word"],
        },
    },
]

UTILITY_TOOL_DEFINITIONS = TOOL_DEFINITIONS

WEB_TOOL_DEFINITIONS = [
    {
        "type": "function", "name": "search_live",
        "description": (
            "Call this for anything current or not in the machine manifest: specs, part numbers, "
            "official guides, firmware, prices, error codes. Returns top web results with source URLs. "
            "Triggers: 'look up', 'search', 'find the official', 'what does the manual say', "
            "'what size', 'which version', 'is there a guide'. Say 'Let me look that up' while waiting."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A focused web search query, including the machine name"}},
            "required": ["query"]},
    },
    {
        "type": "function", "name": "scrape_live",
        "description": (
            "Call this to read one specific web page, usually a URL returned by search_live, "
            "when the snippet isn't enough to answer. Triggers: 'open that', 'read the guide', "
            "'what does that page say'."
        ),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "A full http(s) URL"}},
            "required": ["url"]},
    },
]


def web_tools_enabled() -> bool:
    return bool(os.environ.get("ANAKIN_API_KEY", "").strip())


def tool_definitions(sight=None) -> list[dict]:
    """Tools to register for this session: sight tools only with an adapter, web tools only with a key."""
    tools = list(UTILITY_TOOL_DEFINITIONS)
    if getattr(sight, "name", "") == "model":
        # AssemblyAI recommends <=10 tools per phase; Field mode keeps only calculate from the utilities.
        tools = [t for t in tools if t["name"] == "calculate"]
    if sight is not None:
        tools += SIGHT_TOOL_DEFINITIONS
    if web_tools_enabled():
        tools += WEB_TOOL_DEFINITIONS
    return tools


TOOL_DEFINITIONS = UTILITY_TOOL_DEFINITIONS + SIGHT_TOOL_DEFINITIONS + WEB_TOOL_DEFINITIONS


_SAFE_MATH = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
}


def execute_tool(name: str, arguments: str, sight=None) -> str:
    try:
        if isinstance(arguments, dict):
            args = arguments
        elif arguments:
            args = json.loads(arguments)
        else:
            args = {}
    except json.JSONDecodeError:
        return f"Error: invalid arguments JSON: {arguments}"

    if name in SIGHT_TOOLS:
        if sight is None:
            return "No visual adapter is active. Configure NETRA_SIGHT=model or screen."
        return sight.act(name, args)
    hub = getattr(sight, "renderer", None)
    if name == "search_live":
        return _search_live(args.get("query", ""), hub)
    if name == "scrape_live":
        return _scrape_live(args.get("url", ""), hub)

    if name == "calculate":
        return _calculate(args.get("expression", ""))
    elif name == "get_time":
        return _get_time(args.get("timezone", ""))
    elif name == "define_word":
        return _define_word(args.get("word", ""))
    else:
        return f"Unknown tool: {name}"


_WORD_TO_OP = {
    "plus": "+", "minus": "-", "times": "*", "multiplied by": "*",
    "divided by": "/", "over": "/", "to the power of": "**",
    "squared": "**2", "cubed": "**3", "percent of": "*0.01*",
}


def _calculate(expression: str) -> str:
    if not expression:
        return "Error: no expression provided"
    cleaned = expression.lower()
    for word, op in sorted(_WORD_TO_OP.items(), key=lambda x: -len(x[0])):
        cleaned = cleaned.replace(word, op)
    cleaned = re.sub(r'[^0-9+\-*/().,%\s a-zA-Z]', '', cleaned)
    try:
        result = eval(cleaned, {"__builtins__": {}}, _SAFE_MATH)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def _get_time(timezone: str = "") -> str:
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    try:
        if timezone:
            tz = ZoneInfo(timezone)
            now = datetime.now(tz)
            return now.strftime(f"%A, %B %d, %Y at %I:%M %p ({timezone})")
        else:
            now = datetime.now()
            return now.strftime("%A, %B %d, %Y at %I:%M %p (local time)")
    except Exception:
        now = datetime.now()
        return now.strftime("%A, %B %d, %Y at %I:%M %p (local time)")


def _define_word(word: str) -> str:
    if not word:
        return "Error: no word provided"
    try:
        resp = httpx.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
            timeout=5,
        )
        if resp.status_code != 200:
            return f"Could not find a definition for '{word}'."
        data = resp.json()
        if data and isinstance(data, list):
            meanings = data[0].get("meanings", [])
            if meanings:
                defs = meanings[0].get("definitions", [])
                if defs:
                    return f"{word}: {defs[0].get('definition', 'No definition found.')}"
        return f"Could not find a definition for '{word}'."
    except Exception as e:
        return f"Error looking up '{word}': {e}"


def _search_live(query: str, hub=None) -> str:
    """Search Anakin.io's synchronous web search endpoint."""
    query = str(query).strip()
    api_key = os.environ.get("ANAKIN_API_KEY", "").strip()
    if not query:
        return "Error: no search query provided."
    if not api_key:
        return "Live web search is unavailable: set ANAKIN_API_KEY to enable Anakin.io search."
    try:
        response = httpx.post(
            "https://api.anakin.io/v1/search",
            headers={"X-API-Key": api_key},
            json={"prompt": query, "limit": 3},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if not results:
            return "No live search results were found."
        trimmed = [{"title": r.get("title", ""), "url": r.get("url", ""),
                    "snippet": str(r.get("snippet", ""))[:400]} for r in results[:3]]
        if hub:
            hub.broadcast({"type": "sources", "query": query, "results": trimmed})
        return json.dumps({"query": query, "results": trimmed}, ensure_ascii=False)
    except (httpx.HTTPError, ValueError) as exc:
        return f"Live search failed: {exc}"


def _scrape_live(url: str, hub=None) -> str:
    """Fetch a single page through Anakin.io's inline URL scraper."""
    from urllib.parse import urlparse

    url = str(url).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Error: provide a complete http:// or https:// URL."
    api_key = os.environ.get("ANAKIN_API_KEY", "").strip()
    if not api_key:
        return "Live URL scraping is unavailable: set ANAKIN_API_KEY to enable Anakin.io scraping."
    try:
        response = httpx.post(
            "https://api.anakin.io/v1/url-scraper/scrape",
            headers={"X-API-Key": api_key},
            json={"url": url},
            timeout=httpx.Timeout(90, connect=10),
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        job_id = payload.get("id") or payload.get("jobId")
        if status == "failed":
            return f"Live URL scrape failed: {payload.get('error') or 'Anakin.io reported a failed scrape.'}"
        if status in {"pending", "processing"} and job_id:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                time.sleep(1)
                poll = httpx.get(
                    f"https://api.anakin.io/v1/url-scraper/{job_id}",
                    headers={"X-API-Key": api_key},
                    timeout=httpx.Timeout(10, connect=5),
                )
                poll.raise_for_status()
                payload = poll.json()
                status = payload.get("status")
                if status == "failed":
                    return f"Live URL scrape failed: {payload.get('error') or 'Anakin.io reported a failed scrape.'}"
                if status == "completed":
                    break
            else:
                return f"The page is still being scraped by Anakin.io (job {job_id}); try again shortly."
        elif status not in (None, "completed"):
            return f"Anakin.io returned an unrecognized scrape status: {status}."
        content = payload.get("markdown") or payload.get("content") or ""
        if not content:
            return "Anakin.io completed the scrape but returned no markdown content."
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", str(content))
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if hub:
            hub.broadcast({"type": "sources", "query": "page", "results": [
                {"title": payload.get("title") or url, "url": url, "snippet": text[:200]}]})
        return json.dumps({"url": url, "content": text[:4000]}, ensure_ascii=False)
    except (httpx.HTTPError, ValueError) as exc:
        return f"Live URL scrape failed: {exc}"
