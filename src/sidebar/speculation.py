"""
Speculative execution on partial transcripts.

Fires tool calls against partial Turn events before the turn finalizes.
When the turn finalizes:
  - Match: commit the speculative result (near-zero latency)
  - Mismatch: cancel and re-run against the corrected final transcript

Tracks hit-rate and latency delta for the demo.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

from .tools import TOOL_DEFINITIONS, execute_tool


@dataclass
class SpeculativeCall:
    tool_name: str
    arguments: str
    partial_text: str
    result: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    committed: bool = False
    cancelled: bool = False


@dataclass
class SpeculationStats:
    total_speculations: int = 0
    hits: int = 0
    misses: int = 0
    latency_saved_ms: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if self.total_speculations == 0:
            return 0.0
        return self.hits / self.total_speculations

    @property
    def avg_latency_saved_ms(self) -> float:
        if not self.latency_saved_ms:
            return 0.0
        return sum(self.latency_saved_ms) / len(self.latency_saved_ms)

    def summary(self) -> str:
        return (
            f"Speculations: {self.total_speculations} | "
            f"Hits: {self.hits} | Misses: {self.misses} | "
            f"Hit rate: {self.hit_rate:.0%} | "
            f"Avg latency saved: {self.avg_latency_saved_ms:.0f}ms"
        )


# Tools that are safe to speculatively execute (cheap, no side effects, cancellable)
SPECULATABLE_TOOLS = {"calculate", "get_time", "define_word"}


class SpeculativeExecutor:
    def __init__(self):
        self._pending: SpeculativeCall | None = None
        self._stats = SpeculationStats()
        self._min_confidence_chars = 8

    @property
    def stats(self) -> SpeculationStats:
        return self._stats

    def on_partial_turn(self, text: str) -> SpeculativeCall | None:
        if len(text) < self._min_confidence_chars:
            return None

        prediction = self._predict_tool_call(text)
        if prediction is None:
            return None

        tool_name, arguments = prediction

        if tool_name not in SPECULATABLE_TOOLS:
            return None

        if (
            self._pending
            and self._pending.tool_name == tool_name
            and self._pending.arguments == arguments
        ):
            return self._pending

        call = SpeculativeCall(
            tool_name=tool_name,
            arguments=arguments,
            partial_text=text,
            start_time=time.monotonic(),
        )
        call.result = execute_tool(tool_name, arguments)
        call.end_time = time.monotonic()

        self._pending = call
        self._stats.total_speculations += 1

        elapsed = (call.end_time - call.start_time) * 1000
        print(f"  \033[1;33m[speculate]\033[0m {tool_name}({arguments}) → {call.result} ({elapsed:.0f}ms)")

        return call

    def on_final_turn(self, text: str) -> tuple[str | None, bool]:
        """
        Returns (result, was_hit).
        If there's a pending speculation that matches the final text,
        commit it (hit). Otherwise cancel it (miss) and return None.
        """
        if self._pending is None:
            return None, False

        call = self._pending
        self._pending = None

        final_prediction = self._predict_tool_call(text)

        if (
            final_prediction
            and final_prediction[0] == call.tool_name
            and final_prediction[1] == call.arguments
        ):
            call.committed = True
            self._stats.hits += 1
            latency_saved = (call.end_time - call.start_time) * 1000
            self._stats.latency_saved_ms.append(latency_saved)
            print(f"  \033[1;32m[spec hit]\033[0m committed {call.tool_name} result (saved {latency_saved:.0f}ms)")
            return call.result, True
        else:
            call.cancelled = True
            self._stats.misses += 1
            print(f"  \033[1;31m[spec miss]\033[0m cancelled {call.tool_name}, final text diverged")
            return None, False

    def cancel_pending(self):
        if self._pending:
            self._pending.cancelled = True
            self._pending = None

    def _predict_tool_call(self, text: str) -> tuple[str, str] | None:
        lower = text.lower().strip()

        time_patterns = [
            r"what(?:'s| is) the (?:current )?(?:time|date)",
            r"what time",
            r"what(?:'s| is) today",
            r"tell me the (?:time|date)",
            r"current (?:time|date)",
        ]
        for pat in time_patterns:
            if re.search(pat, lower):
                return ("get_time", "{}")

        calc_patterns = [
            r"(?:what(?:'s| is)|calculate|compute|how much is)\s+(.+?)(?:\?|$)",
            r"(\d+[\s]*[+\-*/^][\s]*\d+)",
            r"(?:square root|sqrt) of (\d+)",
        ]
        for pat in calc_patterns:
            m = re.search(pat, lower)
            if m:
                expr = m.group(1).strip().rstrip("?. ")
                if any(c in expr for c in "0123456789"):
                    return ("calculate", json.dumps({"expression": expr}))

        define_patterns = [
            r"(?:what(?:'s| is) (?:the )?(?:definition|meaning) of|define|what does) ['\"]?(\w+)['\"]?",
            r"what (?:does|is) ['\"]?(\w+)['\"]? mean",
        ]
        for pat in define_patterns:
            m = re.search(pat, lower)
            if m:
                word = m.group(1).strip()
                if word and word not in ("the", "a", "an", "is", "it", "that", "this"):
                    return ("define_word", json.dumps({"word": word}))

        return None
