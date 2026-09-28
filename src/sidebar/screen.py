"""
Netra Desktop: the visual field is the user's live screen.

ScreenReader keeps a fresh OCR reading of the focused window in the background
(RapidOCR, pip-only, no system Tesseract needed). It re-reads only when the
pixels change, and speculation can ask for an immediate re-read while the user
is still talking, so "where's the export button?" is usually answered from a
reading that is already done.

ScreenSight resolves spoken targets against that reading, moves the pointer
(it never clicks) and draws a ring around the target with the overlay process.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FILLER = re.compile(r"\b(?:the|a|an|button|menu|tab|icon|option|link|field|box|panel|item|on|screen|"
                     r"here|please|thing|called|labelled|labeled|that says|says)\b")


@dataclass
class TextItem:
    text: str
    box: tuple[int, int, int, int]   # screen coordinates x1, y1, x2, y2
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


@dataclass
class Reading:
    window: str
    items: list[TextItem] = field(default_factory=list)
    taken_at: float = 0.0
    ocr_ms: float = 0.0


def _active_window() -> tuple[str, tuple[int, int, int, int]] | None:
    """Title and screen rectangle of the focused window (X11)."""
    try:
        from Xlib import X, display
        d = display.Display()
        root = d.screen().root
        wid = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), X.AnyPropertyType).value[0]
        win = d.create_resource_object("window", wid)
        name = win.get_full_property(d.intern_atom("_NET_WM_NAME"), 0)
        geo = win.get_geometry()
        pos = win.translate_coords(root, 0, 0)
        title = name.value.decode("utf-8", "replace") if name and isinstance(name.value, bytes) else str(name.value if name else "")
        d.close()
        return title, (-pos.x, -pos.y, geo.width, geo.height)
    except Exception:
        return None


class ScreenReader:
    CHANGE_THRESHOLD = 2.0     # mean abs pixel difference (downsampled) that counts as "screen changed"
    IDLE_INTERVAL = 0.8

    def __init__(self):
        self._lock = threading.Lock()
        self._reading: Reading | None = None
        self._wake = threading.Event()
        self._fresh = threading.Condition()
        self._stop = False
        self._ocr = None
        self._last_thumb = None
        self.available = True
        self.error = ""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------- public ----------

    @property
    def reading(self) -> Reading | None:
        with self._lock:
            return self._reading

    def request_refresh(self):
        self._last_thumb = None  # force a re-read even if pixels look unchanged
        self._wake.set()

    def fresh_reading(self, max_age: float = 1.5, timeout: float = 4.0) -> Reading | None:
        """A reading no older than max_age seconds, waiting for one if needed."""
        deadline = time.monotonic() + timeout
        reading = self.reading
        if reading and time.monotonic() - reading.taken_at <= max_age:
            return reading
        self.request_refresh()
        with self._fresh:
            while time.monotonic() < deadline:
                reading = self.reading
                if reading and time.monotonic() - reading.taken_at <= max_age:
                    return reading
                self._fresh.wait(timeout=deadline - time.monotonic())
        return self.reading

    def stop(self):
        """Stop and wait for an in-flight OCR pass (exiting mid-inference aborts onnxruntime)."""
        self._stop = True
        self._wake.set()
        self._thread.join(timeout=5)

    # ---------- background loop ----------

    def _loop(self):
        try:
            import mss
            import numpy as np
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        except Exception as exc:  # missing optional deps
            self.available, self.error = False, f"screen reading unavailable: {exc}"
            return
        with mss.MSS() as grabber:
            while not self._stop:
                try:
                    self._tick(grabber, np)
                except Exception as exc:
                    self.error = str(exc)
                self._wake.wait(timeout=self.IDLE_INTERVAL)
                self._wake.clear()

    def _tick(self, grabber, np):
        window = _active_window()
        monitor = grabber.monitors[0]
        full = np.array(grabber.grab(monitor))[:, :, :3]
        if window:
            title, (x, y, w, h) = window
            x0, y0 = max(0, x - monitor["left"]), max(0, y - monitor["top"])
            crop = full[y0:y0 + h, x0:x0 + w]
            offset = (x0 + monitor["left"], y0 + monitor["top"])
        else:
            title, crop, offset = "screen", full, (monitor["left"], monitor["top"])
        if "netra-overlay" in title.lower() or crop.size == 0:
            return
        thumb = crop[::16, ::16].astype("int16")
        if self._last_thumb is not None and self._last_thumb.shape == thumb.shape and \
                float(np.abs(thumb - self._last_thumb).mean()) < self.CHANGE_THRESHOLD:
            return
        self._last_thumb = thumb
        started = time.monotonic()
        result, _ = self._ocr(np.ascontiguousarray(crop), use_cls=False)
        items = []
        for points, text, score in result or []:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            box = (int(min(xs)) + offset[0], int(min(ys)) + offset[1], int(max(xs)) + offset[0], int(max(ys)) + offset[1])
            if str(text).strip():
                items.append(TextItem(str(text).strip(), box, float(score)))
        reading = Reading(title, items, time.monotonic(), (time.monotonic() - started) * 1000)
        with self._lock:
            self._reading = reading
        with self._fresh:
            self._fresh.notify_all()


class OverlayClient:
    """Talks to the overlay subprocess; silently does nothing if it can't start (e.g. no display)."""

    def __init__(self):
        self._proc = None
        try:
            self._proc = subprocess.Popen([sys.executable, "-m", "src.sidebar.overlay"], stdin=subprocess.PIPE,
                                          text=True, cwd=str(Path(__file__).resolve().parents[2]))
        except OSError:
            self._proc = None

    def send(self, command: dict):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps(command) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def close(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.terminate()


def _clean_target(query: str) -> str:
    text = query.lower().strip(" ?.!,'\"")
    text = re.sub(r"^(?:where(?:'s| is)|find|show me|point (?:to|at)|locate|click|open|go to)\s+", "", text)
    return re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()


def match_items(items: list[TextItem], query: str) -> list[tuple[float, TextItem]]:
    """Rank visible text against a spoken target: exact > contained > fuzzy (OCR and STT typos)."""
    target = _clean_target(query)
    if not target:
        return []
    ranked = []
    for item in items:
        label = item.text.lower().strip(" .:…")
        if label == target:
            score = 3.0
        elif target in label and len(target) >= 3:
            score = 2.0 + len(target) / max(len(label), 1)
        elif label in target and len(label) >= 4:
            score = 1.5 + len(label) / len(target)
        else:
            score = difflib.SequenceMatcher(None, label, target).ratio()
            if score < 0.78:
                continue
        ranked.append((score, item))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].box[1], pair[1].box[0]))
    return ranked


class ScreenSight:
    """Desktop adapter: OCR of the focused window, pointer movement and an on-screen ring. Never clicks."""
    name = "screen"
    tools = {"locate", "point", "describe", "walk_through"}

    def __init__(self, manifest=None, reader: ScreenReader | None = None, overlay: OverlayClient | None = None,
                 move_pointer: bool = True):
        self.manifest = manifest
        self.renderer = None
        self.reader = reader or ScreenReader()
        self.overlay = overlay
        self.move_pointer = move_pointer
        self._procedures = None
        if manifest is not None and manifest.procedures:
            from .sight import ModelSight
            self._procedures = ModelSight(manifest)

    # ---------- resolution (shared with speculation) ----------

    def resolve(self, query: str, reading: Reading | None = None) -> dict[str, Any] | None:
        reading = reading or self.reader.reading
        if not reading:
            return None
        target = query
        if self.manifest is not None:  # app vocabulary: "artboard" -> the label the app actually shows
            for entry in self.manifest.vocab:
                if str(entry.get("term", "")).lower() in query.lower() and entry.get("label"):
                    target = str(entry["label"])
        ranked = match_items(reading.items, target)
        if not ranked:
            return None
        score, item = ranked[0]
        return {"id": item.text.lower(), "text": item.text, "box": item.box, "center": item.center,
                "match": round(score, 2), "window": reading.window}

    def mentioned_part(self, text: str):  # speculation hook: screen targets come from intent, not a part list
        return None

    def preview(self, verb: str, arguments: dict[str, Any]) -> str:
        return self.act(verb, arguments, commit=False)

    # ---------- actions ----------

    def act(self, verb: str, arguments: dict[str, Any], commit: bool = True) -> str:
        query = str(arguments.get("target") or arguments.get("query") or "").strip()
        if not self.reader.available:
            return f"Screen reading is not available ({self.reader.error}). Install mss and rapidocr_onnxruntime."
        if verb == "walk_through":
            if self._procedures is None:
                return "No procedures are configured for this app; answer from what's on screen or search the web."
            return self._procedures.act(verb, arguments, commit)
        if verb == "describe" and not query:
            reading = self.reader.fresh_reading(max_age=2.0)
            if not reading:
                return "Could not read the screen yet."
            lines = sorted(reading.items, key=lambda i: (i.box[1] // 20, i.box[0]))
            return json.dumps({"window": reading.window,
                               "visible_text": [i.text for i in lines][:80],
                               "read_ms_ago": round((time.monotonic() - reading.taken_at) * 1000)}, ensure_ascii=False)
        if verb in {"locate", "point", "describe"}:
            found = self.resolve(query)
            if found is None and commit:
                found = self.resolve(query, self.reader.fresh_reading(max_age=0.5))
            if found is None:
                reading = self.reader.reading
                sample = ", ".join(i.text for i in (reading.items[:25] if reading else []))
                return json.dumps({"found": False, "target": query,
                                   "hint": "Not visible in the focused window. Ask the user to open the right "
                                           "window or menu, or describe it differently.",
                                   "some_visible_text": sample}, ensure_ascii=False)
            if commit:
                if self.overlay:
                    self.overlay.send({"ring": list(found["box"]), "label": found["text"], "mode": "solid"})
                if self.move_pointer and verb in {"locate", "point"}:
                    try:
                        import pyautogui
                        pyautogui.moveTo(*found["center"], duration=0.35)
                    except Exception:
                        pass
            return json.dumps({"found": True, "text": found["text"], "window": found["window"],
                               "screen_position": found["center"], "ringed_on_screen": bool(commit and self.overlay),
                               "pointer_moved": bool(commit and self.move_pointer and verb != "describe")},
                              ensure_ascii=False)
        return f"The desktop adapter can't {verb}. It can locate, point, describe and walk through app procedures."

    def ghost(self, found: dict[str, Any]):
        if self.overlay:
            self.overlay.send({"ring": list(found["box"]), "label": found["text"], "mode": "ghost"})

    def clear_ghost(self):
        if self.overlay:
            self.overlay.send({"clear": True})
