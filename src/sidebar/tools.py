"""
Tool definitions and execution for the Voice Agent API.

Tools use the flat schema format: {type, name, description, parameters}.
"""

import json
import math
import re

import httpx

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


def execute_tool(name: str, arguments: str) -> str:
    try:
        if isinstance(arguments, dict):
            args = arguments
        elif arguments:
            args = json.loads(arguments)
        else:
            args = {}
    except json.JSONDecodeError:
        return f"Error: invalid arguments JSON: {arguments}"

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
