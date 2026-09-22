"""Visual-field adapters and the shared Netra manifest loader."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

_STOPWORDS = {"a", "an", "and", "are", "at", "for", "how", "i", "in", "is", "it",
              "me", "of", "on", "please", "the", "to", "what", "where", "with"}


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\w-]+", text.lower()) if word not in _STOPWORDS}


@dataclass
class Manifest:
    domain: str
    name: str
    source: str | None = None
    parts: list[dict[str, Any]] = field(default_factory=list)
    vocab: list[dict[str, str]] = field(default_factory=list)
    procedures: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, filename: str | Path) -> "Manifest":
        path = Path(filename).expanduser().resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Could not load manifest {path}: {exc}") from exc
        data = raw.get("manifest", raw) if isinstance(raw, dict) else None
        if not isinstance(data, dict):
            raise ValueError("Manifest must be a YAML mapping, optionally under 'manifest:'.")
        domain = data.get("domain")
        name = data.get("name")
        if domain not in {"physical_machine", "screen_app"}:
            raise ValueError("manifest.domain must be 'physical_machine' or 'screen_app'.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("manifest.name must be a non-empty string.")
        source = data.get("source")
        if domain == "physical_machine" and not isinstance(source, str):
            raise ValueError("physical_machine manifests require a source model path.")
        parts = data.get("parts", [])
        vocab = data.get("vocab", [])
        procedures = data.get("procedures", [])
        if not isinstance(parts, list) or not all(isinstance(x, dict) for x in parts):
            raise ValueError("manifest.parts must be a list of mappings.")
        if not isinstance(vocab, list) or not all(isinstance(x, dict) for x in vocab):
            raise ValueError("manifest.vocab must be a list of mappings.")
        if not isinstance(procedures, list) or not all(isinstance(x, dict) for x in procedures):
            raise ValueError("manifest.procedures must be a list of mappings.")
        if domain == "physical_machine" and not parts:
            raise ValueError("physical_machine manifests require at least one part.")
        return cls(domain, name.strip(), source, parts, vocab, procedures)

    def context(self) -> str:
        return json.dumps({"domain": self.domain, "name": self.name,
                           "source": self.source,
                           "parts": self.parts, "vocab": self.vocab,
                           "procedures": self.procedures}, ensure_ascii=False)


class Sight(Protocol):
    name: str
    def act(self, verb: str, arguments: dict[str, Any]) -> str: ...


class ModelSight:
    name = "model"

    def __init__(self, manifest: Manifest):
        self.manifest = manifest

    def _part(self, query: str) -> dict[str, Any] | None:
        terms = _tokens(query)
        for entry in self.manifest.vocab:
            vocab_terms = _tokens(str(entry.get("term", "")))
            if not vocab_terms or not vocab_terms.issubset(terms):
                continue
            target = str(entry.get("maps_to") or entry.get("part") or entry.get("element") or "").lower()
            if target:
                mapped = next((part for part in self.manifest.parts
                               if target in {str(part.get("id", "")).lower(),
                                             str(part.get("name", "")).lower()}), None)
                if mapped:
                    return mapped
        ranked = []
        for part in self.manifest.parts:
            haystack = " ".join(str(v) for v in part.values()).lower()
            score = sum(term in haystack for term in terms)
            if score:
                ranked.append((score, part))
        return max(ranked, key=lambda item: item[0])[1] if ranked else None

    def act(self, verb: str, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("target") or arguments.get("query") or "").strip()
        if verb == "walk_through":
            requested = str(arguments.get("procedure") or query).lower()
            terms = _tokens(requested)
            scored = [(len(terms & _tokens(str(item.get("name", "")))), item)
                      for item in self.manifest.procedures]
            scored = [item for item in scored if item[0] > 0]
            for _, procedure in sorted(scored, key=lambda item: item[0], reverse=True):
                steps = procedure.get("steps", [])
                if not isinstance(steps, list) or not steps:
                    return "The matching procedure has no steps configured."
                try:
                    step_number = int(arguments.get("step", 1))
                except (TypeError, ValueError):
                    step_number = 1
                if not 1 <= step_number <= len(steps):
                    return f"Step must be between 1 and {len(steps)}."
                return json.dumps({
                    "procedure": procedure.get("name"),
                    "step": step_number,
                    "total_steps": len(steps),
                    "instruction": steps[step_number - 1],
                    "next_step": step_number + 1 if step_number < len(steps) else None,
                }, ensure_ascii=False)
            return "No matching procedure is defined in this machine manifest."
        part = self._part(query)
        if not part:
            return f"No part matching '{query}' was found in {self.manifest.name}."
        if verb == "describe":
            return json.dumps(part, ensure_ascii=False)
        if verb == "locate":
            return f"Located {part.get('name', part)} in {self.manifest.name}. Part data: {json.dumps(part, ensure_ascii=False)}"
        if verb in {"point", "expand"}:
            return f"{verb.title()} requested for {part.get('name', part)}. The active model renderer must apply this action using part id {part.get('id', part.get('name', 'unknown'))}."
        return f"Unsupported visual action: {verb}."


class ScreenSight:
    """Desktop adapter. Uses OS accessibility names where available and a cursor pointer."""
    name = "screen"

    def __init__(self, manifest: Manifest | None = None):
        self.manifest = manifest
        self._last_match: tuple[str, int, int] | None = None

    def _screen_text(self) -> list[tuple[str, int, int]]:
        """Read visible text and its center coordinates using local OCR."""
        try:
            import pyautogui
            import pytesseract
            image = pyautogui.screenshot()
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            lines: dict[tuple[int, int, int], list[tuple[str, int, int, int, int]]] = {}
            for index, raw in enumerate(data["text"]):
                text = str(raw).strip()
                if not text or float(data["conf"][index]) < 30:
                    continue
                key = (int(data["block_num"][index]), int(data["par_num"][index]),
                       int(data["line_num"][index]))
                lines.setdefault(key, []).append((
                    text, int(data["left"][index]), int(data["top"][index]),
                    int(data["width"][index]), int(data["height"][index])))
            found = []
            for words in lines.values():
                words.sort(key=lambda word: word[1])
                left = min(word[1] for word in words)
                top = min(word[2] for word in words)
                right = max(word[1] + word[3] for word in words)
                bottom = max(word[2] + word[4] for word in words)
                found.append((" ".join(word[0] for word in words),
                              (left + right) // 2, (top + bottom) // 2))
            return found
        except ImportError:
            return []
        except Exception:
            return []

    def _accessibility_names(self) -> list[str]:
        if platform.system() != "Darwin":
            return []
        script = ('tell application "System Events" to tell (first process whose frontmost is true) '
                  'to get name of every UI element of window 1')
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True,
                                    text=True, timeout=3, check=False)
            return [item.strip() for item in result.stdout.split(",") if item.strip()]
        except (OSError, subprocess.TimeoutExpired):
            return []

    def act(self, verb: str, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("target") or arguments.get("query") or "").strip()
        if verb == "walk_through" and self.manifest:
            return ModelSight(self.manifest).act(verb, arguments)
        if verb == "search_live":
            return "Live search is handled by Netra Core."
        names = self._accessibility_names()
        screen_text = self._screen_text()
        if verb == "locate":
            vocab_hints = []
            if self.manifest:
                terms = _tokens(query)
                for item in self.manifest.vocab:
                    vocab_terms = _tokens(str(item.get("term", "")))
                    if vocab_terms and vocab_terms.issubset(terms):
                        vocab_hints.append(item)
            accessibility_matches = [name for name in names if query.lower() in name.lower()]
            matches = [entry for entry in screen_text if query.lower() in entry[0].lower()]
            if not matches and vocab_hints:
                concepts = set().union(*(_tokens(str(item.get("meaning", "")))
                                         for item in vocab_hints))
                matches = [entry for entry in screen_text if concepts & _tokens(entry[0])]
            if matches:
                self._last_match = matches[0]
                text, x, y = matches[0]
                return f"Found visible text '{text}' near ({x}, {y}). Call point with target '{text}' to move the pointer there."
            if accessibility_matches:
                return (f"Matching accessibility elements: {accessibility_matches[:10]}. "
                        "No OCR coordinates were available; provide screen coordinates to point.")
            if vocab_hints:
                return ("App guidance: " + json.dumps(vocab_hints, ensure_ascii=False)
                        + " No matching visible label was found.")
            return ("No matching accessibility element was found. "
                    "No matching visible text was found either. Describe the target or provide screen coordinates.")
        if verb == "describe":
            summary = " ".join(item[0] for item in screen_text[:100])
            if names:
                summary += f"\nAccessibility names: {names[:40]}"
            return summary[:4000] if summary else "Could not read screen text. Check screen capture and OCR permissions/dependencies."
        if verb == "point":
            x, y = arguments.get("x"), arguments.get("y")
            if x is None and y is None and query:
                match = next((entry for entry in screen_text if query.lower() in entry[0].lower()), None)
                if match:
                    self._last_match = match
            if x is None and y is None and self._last_match:
                _, x, y = self._last_match
            try:
                import pyautogui
                if x is not None and y is not None:
                    width, height = pyautogui.size()
                    px, py = int(x), int(y)
                    if not (0 <= px < width and 0 <= py < height):
                        return f"Coordinates must be within the screen ({width}x{height})."
                    pyautogui.moveTo(px, py, duration=0.25)
                    return f"Pointer moved to screen position ({px}, {py}) to indicate {query or 'the target'}."
                return "To point on screen, provide x and y coordinates. No click was performed."
            except ImportError:
                return "Screen pointing requires pyautogui. Install the optional desktop dependencies."
            except Exception as exc:
                return f"Could not move the screen pointer: {exc}"
        if verb == "expand":
            return "Screen zoom is not available through the current desktop adapter."
        return f"Unsupported screen action: {verb}."


TOOL_DEFINITIONS = [
    {"type": "function", "name": "locate", "description": "Find a named part or visible UI element.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"type": "function", "name": "point", "description": "Point at a part or screen location. Screen coordinates are optional x and y; this never clicks.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}}}},
    {"type": "function", "name": "expand", "description": "Expand or zoom into a model part or screen target when the active adapter supports it.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}},
    {"type": "function", "name": "describe", "description": "Describe a part or the visible desktop UI.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}}},
    {"type": "function", "name": "walk_through", "description": "Get one numbered step from a named troubleshooting procedure. Ask before continuing and request the next step number only after the user is ready.", "parameters": {"type": "object", "properties": {"procedure": {"type": "string"}, "step": {"type": "integer", "minimum": 1}}, "required": ["procedure"]}},
]
