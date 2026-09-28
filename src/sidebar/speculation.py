"""
Speculative execution on partial transcripts.

While the user is still talking, Universal-Streaming partials are scanned for
intent. Predicted read-only tool calls are fired early:

  - visual (locate/describe/walk_through): previewed against the manifest and
    shown as a translucent "ghost" highlight in the 3D view
  - search_live / define_word: the slow network call starts in the background

When the Voice Agent API later emits the real tool.call, the agent asks
claim(): if the speculation matches semantically (same part, same procedure,
overlapping search query), the ready result is returned instantly and the
ghost turns solid. Otherwise the speculation is dropped and the ghost cleared.

Stats (hits, misses, lead time) feed the viewer's speculation meter.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .tools import execute_tool, web_tools_enabled

PART_TOOLS = {"locate", "describe", "point"}
SPECULATABLE_TOOLS = PART_TOOLS | {"walk_through", "search_live", "define_word"}

_SEARCH_STOP = {"a", "an", "the", "for", "of", "on", "to", "me", "my", "up", "look", "search",
                "find", "please", "can", "you", "what", "is", "does", "do", "how", "about",
                "and", "in", "netra", "hey", "google", "say", "says", "tell"}

_SEARCH_INTENT = re.compile(
    r"\b(?:look up|look for|search(?: the web)?(?: for)?|google|find(?: me)?(?: the)? official"
    r"|find(?: me)? (?:a|the) (?:guide|manual|spec)|what does the manual say(?: about)?|check online(?: for)?)\b(.*)")
_PROCEDURE_INTENT = re.compile(
    r"\b(?:walk me through|how do i|how to|help me|fix|keeps|won't|isn't|is not|problem|issue|error|"
    r"clogged|clicking|jammed|broken|replace|swap)\b")
_DESCRIBE_INTENT = re.compile(r"\bwhat (?:does|do|is|'s)\b.*\b(?:do|for|does)\b")
_DEFINE_INTENT = [
    re.compile(r"(?:what(?:'s| is) (?:the )?(?:definition|meaning) of|define) ['\"]?(\w+)"),
    re.compile(r"what does (?:the word )?['\"]?(\w+)['\"]? mean"),
]


def _search_terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:6] for w in words if w not in _SEARCH_STOP and len(w) > 1}


def _json(text: str | None) -> dict:
    try:
        value = json.loads(text or "")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class SpeculativeCall:
    tool_name: str
    arguments: dict[str, Any]
    key: str
    partial_text: str
    fired_at: float = field(default_factory=time.monotonic)
    ready_at: float | None = None
    result: str | None = None
    part_ids: list[str] = field(default_factory=list)
    lead_ms: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)

    @property
    def ready(self) -> bool:
        return self.done.is_set()


@dataclass
class SpeculationStats:
    total_speculations: int = 0
    hits: int = 0
    misses: int = 0
    lead_ms: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        decided = self.hits + self.misses
        return self.hits / decided if decided else 0.0

    @property
    def avg_lead_ms(self) -> float:
        return sum(self.lead_ms) / len(self.lead_ms) if self.lead_ms else 0.0

    def as_event(self) -> dict:
        return {"type": "spec_stats", "fired": self.total_speculations, "hits": self.hits,
                "misses": self.misses, "avg_lead_ms": round(self.avg_lead_ms)}

    def summary(self) -> str:
        return (f"Speculations: {self.total_speculations} | Hits: {self.hits} | "
                f"Misses: {self.misses} | Hit rate: {self.hit_rate:.0%} | "
                f"Avg lead: {self.avg_lead_ms:.0f}ms")


class SpeculativeExecutor:
    SEARCH_STABLE_MS = 350
    MAX_AGE_S = 20.0

    def __init__(self, sight=None, hub=None):
        self._sight = sight
        self._hub = hub
        self._lock = threading.Lock()
        self._pending: dict[str, SpeculativeCall] = {}  # slot -> call ("part", "procedure", "search", "define")
        self._stats = SpeculationStats()
        self._search_candidate: tuple[str, float] | None = None
        self._searches_this_turn = 0

    @property
    def stats(self) -> SpeculationStats:
        return self._stats

    # ---------- viewer ----------

    def _emit(self, event: dict):
        if self._hub:
            self._hub.broadcast(event)

    def _emit_stats(self):
        self._emit(self._stats.as_event())

    # ---------- partial / final turns ----------

    def on_partial_turn(self, text: str):
        self._speculate(text, final=False)

    def on_final_turn(self, text: str):
        self._speculate(text, final=True)
        self._search_candidate = None
        self._searches_this_turn = 0

    def _speculate(self, text: str, final: bool):
        lower = text.lower().strip()
        if len(lower) < 6:
            return
        self._expire()
        if self._sight is not None and getattr(self._sight, "name", "") == "model":
            self._speculate_visual(lower, text)
        if web_tools_enabled():
            self._speculate_search(lower, text, final)
        self._speculate_define(lower, text)

    def _speculate_visual(self, lower: str, text: str):
        sight = self._sight
        procedure = None
        if _PROCEDURE_INTENT.search(lower):
            procedure = sight.find_procedure(lower)
            if procedure is not None and procedure is getattr(sight, "_current_procedure", None):
                procedure = None  # mid-procedure "next" steps depend on state; don't guess
        if procedure is not None:
            name = str(procedure.get("name"))
            args = {"procedure": name}
            if self._same("procedure", name):
                return
            result = sight.preview("walk_through", args)
            parts = _json(result).get("parts", [])
            self._fire("procedure", SpeculativeCall("walk_through", args, name, text, part_ids=parts), result)
            return
        part = sight.mentioned_part(lower)
        if part is None:
            return
        part_id = str(part.get("id"))
        if self._same("part", part_id):
            return
        tool = "describe" if _DESCRIBE_INTENT.search(lower) else "locate"
        args = {"query": str(part.get("name"))} if tool == "locate" else {"target": str(part.get("name"))}
        result = sight.preview(tool, args)
        self._fire("part", SpeculativeCall(tool, args, part_id, text, part_ids=[part_id]), result)

    def _speculate_search(self, lower: str, text: str, final: bool):
        match = _SEARCH_INTENT.search(lower)
        if not match:
            return
        query = match.group(1).strip(" ?.,")
        manifest = getattr(self._sight, "manifest", None)
        if manifest and not (_search_terms(manifest.name) & _search_terms(query)):
            query = f"{manifest.name} {query}"
        terms = _search_terms(query)
        if len(terms) < 3:
            return
        now = time.monotonic()
        candidate = self._search_candidate
        if candidate is None or _jaccard(_search_terms(candidate[0]), terms) < 0.8:
            self._search_candidate = (query, now)
            if not final:
                return
        elif not final and (now - candidate[1]) * 1000 < self.SEARCH_STABLE_MS:
            return
        with self._lock:
            existing = self._pending.get("search")
        if existing and _jaccard(_search_terms(existing.key), terms) >= 0.5:
            return
        if self._searches_this_turn >= 2:
            return
        self._searches_this_turn += 1
        call = SpeculativeCall("search_live", {"query": query}, query, text)
        self._fire_async("search", call)

    def _speculate_define(self, lower: str, text: str):
        for pattern in _DEFINE_INTENT:
            m = pattern.search(lower)
            if m and m.group(1) not in {"the", "a", "an", "it", "that", "this", "is"}:
                word = m.group(1)
                if not self._same("define", word):
                    self._fire_async("define", SpeculativeCall("define_word", {"word": word}, word, text))
                return

    # ---------- firing ----------

    def _same(self, slot: str, key: str) -> bool:
        with self._lock:
            call = self._pending.get(slot)
        return bool(call and call.key == key)

    def _fire(self, slot: str, call: SpeculativeCall, result: str):
        call.result = result
        call.ready_at = time.monotonic()
        call.done.set()
        self._replace(slot, call)
        for part_id in call.part_ids:
            self._emit({"type": "highlight", "part_id": part_id, "mode": "ghost"})
        print(f"  \033[1;33m[speculate]\033[0m {call.tool_name}({json.dumps(call.arguments)}) from partial")

    def _fire_async(self, slot: str, call: SpeculativeCall):
        self._replace(slot, call)
        print(f"  \033[1;33m[speculate]\033[0m {call.tool_name}({json.dumps(call.arguments)}) started in background")

        def run():
            result = execute_tool(call.tool_name, call.arguments, None)
            call.result = result
            call.ready_at = time.monotonic()
            call.done.set()
            elapsed = (call.ready_at - call.fired_at) * 1000
            print(f"  \033[1;33m[speculate]\033[0m {call.tool_name} ready ({elapsed:.0f}ms)")
            self._emit({"type": "spec", "kind": "ready", "tool": call.tool_name, "ms": round(elapsed)})

        threading.Thread(target=run, daemon=True).start()

    def _replace(self, slot: str, call: SpeculativeCall):
        with self._lock:
            old = self._pending.get(slot)
            self._pending[slot] = call
            self._stats.total_speculations += 1
        if old is not None:
            self._miss(old)
        self._emit({"type": "spec", "kind": "fire", "tool": call.tool_name,
                    "part_ids": call.part_ids, "label": call.key})
        self._emit_stats()

    def _miss(self, call: SpeculativeCall):
        self._stats.misses += 1
        if call.part_ids:
            self._emit({"type": "clear_ghost", "part_ids": call.part_ids})
        self._emit({"type": "spec", "kind": "miss", "tool": call.tool_name, "label": call.key})
        self._emit_stats()

    def _expire(self):
        now = time.monotonic()
        with self._lock:
            stale = [slot for slot, c in self._pending.items() if now - c.fired_at > self.MAX_AGE_S]
            expired = [self._pending.pop(slot) for slot in stale]
        for call in expired:
            self._miss(call)

    def cancel_all(self):
        with self._lock:
            calls = list(self._pending.values())
            self._pending.clear()
        for call in calls:
            self._miss(call)

    # ---------- agent side ----------

    def claim(self, tool_name: str, arguments) -> SpeculativeCall | None:
        """Called on the agent's real tool.call. Returns a matching speculation (possibly still
        in flight — wait on call.done), or None."""
        if tool_name not in SPECULATABLE_TOOLS:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                arguments = {}
        slot = self._slot_for(tool_name)
        with self._lock:
            call = self._pending.get(slot)
        if call is None:
            return None
        if not self._matches(call, tool_name, arguments):
            with self._lock:
                if self._pending.get(slot) is call:
                    self._pending.pop(slot)
            self._miss(call)
            return None
        with self._lock:
            self._pending.pop(slot, None)
        now = time.monotonic()
        call.lead_ms = (now - call.fired_at) * 1000
        self._stats.hits += 1
        self._stats.lead_ms.append(call.lead_ms)
        self._emit({"type": "spec", "kind": "hit", "tool": tool_name, "label": call.key,
                    "part_ids": call.part_ids, "lead_ms": round(call.lead_ms)})
        self._emit_stats()
        if tool_name != call.tool_name and self._sight is not None:
            # e.g. speculated locate, agent asked describe for the same part: same payload.
            call.result = self._sight.preview(tool_name, arguments)
        return call

    @staticmethod
    def _slot_for(tool_name: str) -> str:
        if tool_name in PART_TOOLS:
            return "part"
        return {"walk_through": "procedure", "search_live": "search", "define_word": "define"}[tool_name]

    def _matches(self, call: SpeculativeCall, tool_name: str, arguments: dict) -> bool:
        if tool_name in PART_TOOLS:
            query = str(arguments.get("query") or arguments.get("target") or "")
            part = self._sight.resolve(query) if self._sight is not None else None
            return bool(part and str(part.get("id")) == call.key)
        if tool_name == "walk_through":
            if arguments.get("step") not in (None, 1, "1"):
                return False
            procedure = self._sight.find_procedure(str(arguments.get("procedure", ""))) if self._sight else None
            if not procedure or str(procedure.get("name")) != call.key:
                return False
            # Same procedure and same step as the preview (state may have moved on since).
            now, then = _json(self._sight.preview("walk_through", arguments)), _json(call.result)
            return now.get("step") == then.get("step")
        if tool_name == "search_live":
            return _jaccard(_search_terms(call.key), _search_terms(str(arguments.get("query", "")))) >= 0.5
        if tool_name == "define_word":
            return str(arguments.get("word", "")).lower().strip() == call.key
        return False
