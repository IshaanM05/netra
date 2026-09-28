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

_STOPWORDS = {"a", "an", "and", "are", "at", "can", "do", "does", "for", "how", "i", "in",
              "is", "it", "me", "my", "of", "on", "please", "show", "tell", "that", "the",
              "this", "to", "what", "where", "which", "with", "you", "your", "netra", "hey"}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> set[str]:
    return {_stem(word) for word in re.findall(r"[\w-]+", text.lower()) if word not in _STOPWORDS}


def _normalize(text: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))) + " "


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    phrase = _normalize(phrase)
    return len(phrase.strip()) > 0 and phrase in normalized_text


@dataclass
class Manifest:
    domain: str
    name: str
    source: str | None = None
    parts: list[dict[str, Any]] = field(default_factory=list)
    vocab: list[dict[str, str]] = field(default_factory=list)
    procedures: list[dict[str, Any]] = field(default_factory=list)
    base_dir: Path | None = None

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
        return cls(domain, name.strip(), source, parts, vocab, procedures, path.parent)

    @property
    def model_path(self) -> Path | None:
        """Resolved path of a .glb/.gltf source, if the manifest points at one that exists."""
        if not self.source or not self.source.lower().endswith((".glb", ".gltf")):
            return None
        path = Path(self.source)
        if not path.is_absolute() and self.base_dir:
            path = self.base_dir / path
        return path if path.exists() else None

    def context(self) -> str:
        return json.dumps({"domain": self.domain, "name": self.name,
                           "source": self.source,
                           "parts": self.parts, "vocab": self.vocab,
                           "procedures": self.procedures}, ensure_ascii=False)

    def spoken_context(self) -> str:
        """Compact manifest summary for the system prompt."""
        lines = [f"Machine: {self.name}"]
        for part in self.parts:
            aliases = ", ".join(part.get("aliases", []))
            lines.append(f"- {part.get('name')} (id {part.get('id')}; also called {aliases})")
        if self.procedures:
            lines.append("Procedures: " + "; ".join(str(p.get("name")) for p in self.procedures))
        return "\n".join(lines)


def _step_text(step: Any) -> str:
    return str(step.get("text", "")) if isinstance(step, dict) else str(step)


def _step_parts(step: Any) -> list[str]:
    return [str(p) for p in step.get("parts", [])] if isinstance(step, dict) else []


class Sight(Protocol):
    name: str
    def act(self, verb: str, arguments: dict[str, Any]) -> str: ...


class ModelSight:
    """Physical-machine adapter. Resolves parts from the manifest and drives the 3D viewer."""
    name = "model"

    _WHOLE_WORDS = {"all", "everything", "whole", "entire", "assembly", "extruder", "machine",
                    "it", "thing", "printer", "head", "the", "whole", "overview", "view", "none"}

    def _is_whole(self, query: str) -> bool:
        words = set(re.findall(r"[a-z]+", query.lower()))
        return not words or words <= self._WHOLE_WORDS | {self.manifest.name.lower()} | set(
            re.findall(r"[a-z]+", self.manifest.name.lower()))

    def __init__(self, manifest: Manifest, renderer=None):
        self.manifest = manifest
        self.renderer = renderer
        self._current_procedure: dict[str, Any] | None = None
        self._current_step = 0

    # ---------- resolution ----------

    def _phrases(self, part: dict[str, Any]) -> list[str]:
        return [str(part.get("id", "")).replace("_", " "), str(part.get("name", "")),
                *[str(a) for a in part.get("aliases", [])]]

    def resolve(self, query: str) -> dict[str, Any] | None:
        """Best-matching part for free text, or None. Phrase matches beat token overlap."""
        if not query:
            return None
        text = _normalize(query)
        best: tuple[float, dict[str, Any]] | None = None
        for part in self.manifest.parts:
            for phrase in self._phrases(part):
                if _contains_phrase(text, phrase):
                    score = 100 + len(phrase)
                    if not best or score > best[0]:
                        best = (score, part)
        by_id = {str(p.get("id")): p for p in self.manifest.parts}
        for entry in self.manifest.vocab:
            term = str(entry.get("term", ""))
            target = by_id.get(str(entry.get("maps_to") or entry.get("part") or ""))
            if target and _contains_phrase(text, term):
                score = 90 + len(term)
                if not best or score > best[0]:
                    best = (score, target)
        if best:
            return best[1]
        terms = _tokens(query)
        ranked = []
        for part in self.manifest.parts:
            name_terms = set().union(*(_tokens(p) for p in self._phrases(part)))
            score = 3 * len(terms & name_terms) + len(terms & _tokens(str(part.get("description", ""))))
            if score >= 3:
                ranked.append((score, part))
        return max(ranked, key=lambda item: item[0])[1] if ranked else None

    def mentioned_part(self, text: str) -> dict[str, Any] | None:
        """Part named explicitly (by phrase, not fuzzy tokens) in text — used for speculation."""
        normalized = _normalize(text)
        best = None
        for part in self.manifest.parts:
            for phrase in self._phrases(part):
                if _contains_phrase(normalized, phrase) and (not best or len(phrase) > best[0]):
                    best = (len(phrase), part)
        return best[1] if best else None

    def find_procedure(self, query: str) -> dict[str, Any] | None:
        if not query:
            return None
        text = _normalize(query)
        best = None
        for procedure in self.manifest.procedures:
            for phrase in [str(procedure.get("name", "")), *map(str, procedure.get("aliases", []))]:
                if _contains_phrase(text, phrase) and (not best or len(phrase) + 100 > best[0]):
                    best = (len(phrase) + 100, procedure)
        if best:
            return best[1]
        terms = _tokens(query) - {"walk", "through", "guide", "step", "steps", "procedure", "fix", "help"}
        scored = []
        for procedure in self.manifest.procedures:
            names = [str(procedure.get("name", "")), *map(str, procedure.get("aliases", []))]
            overlap = len(terms & set().union(*(_tokens(n) for n in names)))
            if overlap:
                scored.append((overlap, procedure))
        return max(scored, key=lambda item: item[0])[1] if scored else None

    # ---------- viewer ----------

    @property
    def _viewer_live(self) -> bool:
        return bool(self.renderer and self.renderer.has_clients)

    def _show(self, event: dict[str, Any]):
        if self.renderer:
            self.renderer.broadcast(event)

    def _part_payload(self, part: dict[str, Any], shown: bool) -> dict[str, Any]:
        payload = {"part_id": part.get("id"), "name": part.get("name"),
                   "description": part.get("description")}
        if part.get("safety"):
            payload["safety"] = part["safety"]
        payload["shown_in_3d_view"] = shown
        return payload

    # ---------- actions ----------

    def preview(self, verb: str, arguments: dict[str, Any]) -> str:
        """Same answer as act(), but without touching the viewer or procedure state (for speculation)."""
        return self.act(verb, arguments, commit=False)

    def act(self, verb: str, arguments: dict[str, Any], commit: bool = True) -> str:
        query = str(arguments.get("target") or arguments.get("query") or "").strip()
        show = self._show if commit else (lambda event: None)

        if verb == "walk_through":
            return self._walk_through(arguments, query, commit)

        if verb == "reset_view":
            if not commit:
                return json.dumps({"reset": True})
            self._show({"type": "reset"})
            self._current_procedure = None
            return json.dumps({"reset": True, "shown_in_3d_view": self._viewer_live})

        if verb == "expand":
            if self._is_whole(query):
                show({"type": "explode", "part_id": None})
                return json.dumps({"exploded": "whole assembly", "shown_in_3d_view": self._viewer_live})
            part = self.resolve(query)
            if not part:
                return f"No part matching '{query}' in {self.manifest.name}."
            show({"type": "explode", "part_id": part.get("id")})
            return json.dumps({"exploded_around": part.get("name"), **self._part_payload(part, self._viewer_live)},
                              ensure_ascii=False)

        if verb == "describe" and self._is_whole(query):
            return json.dumps({"machine": self.manifest.name,
                               "parts": [p.get("name") for p in self.manifest.parts],
                               "procedures": [p.get("name") for p in self.manifest.procedures]},
                              ensure_ascii=False)

        part = self.resolve(query)
        if not part:
            names = ", ".join(str(p.get("name")) for p in self.manifest.parts)
            return f"No part matching '{query}' in {self.manifest.name}. Known parts: {names}."

        if verb in {"locate", "point", "describe"}:
            show({"type": "highlight", "part_id": part.get("id"), "mode": "solid"})
            show({"type": "focus", "part_id": part.get("id")})
            return json.dumps(self._part_payload(part, self._viewer_live), ensure_ascii=False)
        return f"Unsupported visual action: {verb}."

    def _walk_through(self, arguments: dict[str, Any], query: str, commit: bool = True) -> str:
        requested = str(arguments.get("procedure") or query)
        procedure = self.find_procedure(requested)
        if procedure is None and self._current_procedure and (
                not requested or _tokens(requested) <= {"next", "previous", "back", "again", "repeat", "current"}):
            procedure = self._current_procedure
        if procedure is None:
            names = "; ".join(str(p.get("name")) for p in self.manifest.procedures)
            return f"No matching procedure in this manifest. Available: {names}."
        steps = procedure.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return "The matching procedure has no steps configured."
        raw_step = arguments.get("step")
        if raw_step is None:
            same = self._current_procedure is procedure
            step_number = self._current_step + 1 if same else 1
        else:
            try:
                step_number = int(raw_step)
            except (TypeError, ValueError):
                step_number = 1
        if step_number > len(steps):
            return json.dumps({"procedure": procedure.get("name"), "done": True,
                               "message": f"That was the last of {len(steps)} steps."})
        step_number = max(step_number, 1)
        step = steps[step_number - 1]
        part_ids = _step_parts(step)
        if commit:
            self._current_procedure = procedure
            self._current_step = step_number
            self._show({"type": "step", "procedure": procedure.get("name"), "step": step_number,
                    "total": len(steps), "instruction": _step_text(step), "part_ids": part_ids})
        return json.dumps({
            "procedure": procedure.get("name"),
            "step": step_number,
            "total_steps": len(steps),
            "instruction": _step_text(step),
            "parts": part_ids,
            "next_step": step_number + 1 if step_number < len(steps) else None,
            "shown_in_3d_view": self._viewer_live,
        }, ensure_ascii=False)


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
        if verb == "reset_view":
            return "The desktop adapter has no view to reset."
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


SIGHT_TOOLS = {"locate", "point", "expand", "describe", "walk_through", "reset_view"}

TOOL_DEFINITIONS = [
    {
        "type": "function", "name": "locate",
        "description": (
            "Call this when the user asks where a part or UI element is, or refers to one by name. "
            "It finds it and highlights it in the user's view. "
            "Triggers: 'where is the X', 'which one is the X', 'find the X', 'show me the X', "
            "'what's this X'. Always call this before talking about a specific part."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The part or element as the user named it, e.g. 'nozzle'"}},
            "required": ["query"]},
    },
    {
        "type": "function", "name": "point",
        "description": (
            "Call this to point at a part or screen location so the user can see it. Never clicks. "
            "Triggers: 'point to', 'point at', 'highlight', 'which one exactly'. "
            "For screens, x and y are optional pixel coordinates."
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Part or element name"},
            "x": {"type": "integer"}, "y": {"type": "integer"}}},
    },
    {
        "type": "function", "name": "expand",
        "description": (
            "Call this to show an exploded view — pull the parts apart so the user can see inside. "
            "Triggers: 'take it apart', 'exploded view', 'show me inside', 'how does it come apart', "
            "'break it down', 'expand'. Use target 'all' for the whole assembly or a part name to explode around it."
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "'all' or a part name"}},
            "required": ["target"]},
    },
    {
        "type": "function", "name": "describe",
        "description": (
            "Call this when the user asks what a part does, what it is for, or what is in view. "
            "Triggers: 'what does the X do', 'what is the X for', 'tell me about the X', "
            "'what am I looking at', 'what parts are there'. Leave target empty for an overview."
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Part name, or empty for an overview"}}},
    },
    {
        "type": "function", "name": "walk_through",
        "description": (
            "Call this when the user has a problem to fix or wants a procedure, and for every "
            "'next step'. Returns ONE step at a time and highlights the parts involved. "
            "Triggers: 'walk me through', 'how do I fix', 'it's clicking', 'clogged', 'how do I replace', "
            "'next', 'next step', 'what now', 'done, what's next', 'go back', 'repeat that'. "
            "Omit step to get the next step of the current procedure."
        ),
        "parameters": {"type": "object", "properties": {
            "procedure": {"type": "string", "description": "The problem or procedure in the user's words; for 'next' reuse the current procedure name"},
            "step": {"type": "integer", "minimum": 1, "description": "Specific step number; omit for the next step"}},
            "required": ["procedure"]},
    },
    {
        "type": "function", "name": "reset_view",
        "description": (
            "Call this to put the view back to normal: un-highlight, reassemble, recenter. "
            "Triggers: 'reset', 'put it back together', 'start over', 'clear that'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]
