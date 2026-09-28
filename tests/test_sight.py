#!/usr/bin/env python3
"""Offline tests (no network): part resolution, procedures, speculation, wake parsing.

    python tests/test_sight.py      # or: pytest tests/test_sight.py
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ANAKIN_API_KEY"] = ""  # keep speculation offline

from src.sidebar.engine import NetraEngine  # noqa: E402
from src.sidebar.learn import normalize_draft  # noqa: E402
from src.sidebar.screen import Reading, ScreenSight, TextItem, match_items  # noqa: E402
from src.sidebar.sight import MachineLibrary, Manifest, ModelSight  # noqa: E402
from src.sidebar.speculation import SpeculativeExecutor  # noqa: E402
from src.sidebar.tools import tool_definitions  # noqa: E402
from src.sidebar.viewer import RecordingHub  # noqa: E402

MANIFEST = Manifest.load(ROOT / "manifests" / "sample-machine.yaml")  # the extruder


def fresh():
    hub = RecordingHub()
    return ModelSight(MANIFEST, renderer=hub), hub


def test_resolution():
    sight, _ = fresh()
    cases = {
        "where's the heatbreak": "heatbreak", "the hot end": "hotend", "heater cartridge": "heater_cartridge",
        "hot end fan": "hotend_fan", "bowden tube": "ptfe_tube", "the blower": "print_fan",
        "heat creep": "heatbreak", "thermal runaway": "thermistor", "idler": "idler_lever",
        # speech-to-text style errors
        "heat brake": "heatbreak", "the thermister": "thermistor", "nozle": "nozzle", "heat sync": "heatsink",
        "idle lever": "idler_lever", "P T F E tube": "ptfe_tube",
    }
    for query, expected in cases.items():
        part = sight.resolve(query)
        assert part and part["id"] == expected, f"{query!r} -> {part and part['id']}, want {expected}"
    assert sight.resolve("multiple things") is None
    assert sight.resolve("") is None


def test_procedures():
    sight, _ = fresh()
    cases = {"my extruder keeps clicking": "clicking", "how do I replace the nozzle": "replace",
             "it's clogged": "clog", "thermal runaway error": "temperature",
             "walk me through replacing the nozzle": "replace"}
    for query, expected in cases.items():
        proc = sight.find_procedure(query)
        assert proc and expected in proc["name"], f"{query!r} -> {proc and proc['name']}"


def test_walk_through_state_and_viewer_events():
    sight, hub = fresh()
    r1 = json.loads(sight.act("walk_through", {"procedure": "clicking extruder"}))
    r2 = json.loads(sight.act("walk_through", {"procedure": "clicking extruder"}))
    r3 = json.loads(sight.act("walk_through", {"procedure": "next"}))
    assert (r1["step"], r2["step"], r3["step"]) == (1, 2, 3)
    back = json.loads(sight.act("walk_through", {"procedure": "clicking extruder", "step": 2}))
    assert back["step"] == 2
    steps = [e for e in hub.events if e["type"] == "step"]
    assert [e["step"] for e in steps] == [1, 2, 3, 2]
    assert steps[0]["part_ids"] == ["hotend", "nozzle"]
    for _ in range(5):
        last = json.loads(sight.act("walk_through", {"procedure": "clicking extruder"}))
    assert last.get("done") is True


def test_preview_is_pure():
    sight, hub = fresh()
    preview = sight.preview("walk_through", {"procedure": "clicking"})
    assert json.loads(preview)["step"] == 1
    assert hub.events == [] and sight._current_procedure is None
    sight.preview("locate", {"query": "nozzle"})
    sight.preview("expand", {"target": "all"})
    assert hub.events == []


def test_locate_and_expand_events():
    sight, hub = fresh()
    result = json.loads(sight.act("locate", {"query": "nozzle"}))
    assert result["part_id"] == "nozzle" and "safety" in result and result["shown_in_3d_view"] is True
    assert {"type": "highlight", "part_id": "nozzle", "mode": "solid"} in hub.events
    sight.act("expand", {"target": "the whole extruder"})
    assert hub.events[-1] == {"type": "explode", "part_id": None}
    sight.act("reset_view", {})
    assert hub.events[-1] == {"type": "reset"}


def test_speculation_part_hit():
    sight, hub = fresh()
    spec = SpeculativeExecutor(sight, hub)
    for partial in ["where's", "where's the", "where's the heat", "where's the heatbreak"]:
        spec.on_partial_turn(partial)
    assert {"type": "highlight", "part_id": "heatbreak", "mode": "ghost"} in hub.events
    spec.on_final_turn("Where's the heatbreak?")
    hit = spec.claim("locate", {"query": "heat break"})
    assert hit is not None and hit.ready and json.loads(hit.result)["part_id"] == "heatbreak"
    assert spec.stats.hits == 1 and spec.stats.misses == 0
    # describe for the same part is also a hit (same payload), and it is consumed
    spec.on_partial_turn("what does the nozzle do")
    assert spec.claim("describe", {"target": "nozzle"}) is not None
    assert spec.claim("describe", {"target": "nozzle"}) is None


def test_speculation_miss_clears_ghost():
    sight, hub = fresh()
    spec = SpeculativeExecutor(sight, hub)
    spec.on_partial_turn("where is the nozzle")
    assert spec.claim("locate", {"query": "thermistor"}) is None
    assert spec.stats.misses == 1
    assert any(e["type"] == "clear_ghost" and e["part_ids"] == ["nozzle"] for e in hub.events)


def test_speculation_procedure():
    sight, hub = fresh()
    spec = SpeculativeExecutor(sight, hub)
    spec.on_partial_turn("my extruder keeps clicking")
    hit = spec.claim("walk_through", {"procedure": "inspect a clicking extruder"})
    assert hit is not None and json.loads(hit.result)["step"] == 1
    # the speculation must not have advanced procedure state
    assert sight._current_procedure is None
    sight.act("walk_through", {"procedure": "inspect a clicking extruder"})
    # mid-procedure "next" is state-dependent: no speculation
    spec.on_partial_turn("okay the extruder is still clicking, next")
    assert spec.claim("walk_through", {"procedure": "clicking"}) is None


def test_tool_budget():
    sight, _ = fresh()
    names = [t["name"] for t in tool_definitions(sight)]
    assert len(names) <= 10, names  # AssemblyAI guidance: <=10 tools per phase
    assert "locate" in names and "walk_through" in names and "get_time" not in names


def test_library_and_switching():
    library = MachineLibrary(ROOT / "manifests")
    for query, expected in {"bike": "Bicycle", "my road bike": "Bicycle", "the 3D printer": "Prusa",
                            "Prusa": "Prusa"}.items():
        assert expected in library.find(query).name, query
    assert library.find("Breville espresso machine") is None
    hub = RecordingHub()
    sight = ModelSight(library.find("printer"), renderer=hub, library=library)
    switched = []
    sight.on_switch = switched.append
    result = json.loads(sight.act("load_machine", {"machine": "bike", "also_asked": "I got a flat"}))
    assert result["loaded"].startswith("Bicycle") and switched and switched[0].name.startswith("Bicycle")
    assert result["already_done_for_their_request"]["walk_through"]["procedure"] == "fix a flat tire"
    assert any(e["type"] == "manifest" and e["name"].startswith("Bicycle") for e in hub.events)
    assert sight.resolve("the pulley wheels")["id"] == "jockey_wheels"
    back = json.loads(sight.act("load_machine", {"machine": "3D printer", "also_asked": "where's the nozzle?"}))
    assert back["already_done_for_their_request"]["locate"]["part_id"] == "nozzle"
    assert "learning new machines is unavailable" in sight.act("load_machine", {"machine": "espresso machine"})


def test_normalize_learned_draft():
    draft = {"name": "Barista Express", "parts": [
        {"id": "Steam Wand", "name": "steam wand", "safety": "None"},
        {"name": "steam wand", "description": "duplicate id"},
        {"id": "portafilter", "name": "Portafilter", "safety": "Hot after brewing."}],
        "procedures": [{"name": "purge", "steps": [{"text": "Open the wand", "parts": ["steam wand", "ghost"]},
                                                     "Close it"]}],
        "vocab": [{"term": "sputter", "maps_to": "steam_wand"}, {"term": "x", "maps_to": "nope"}]}
    data = normalize_draft(draft, "fallback")["manifest"]
    ids = [p["id"] for p in data["parts"]]
    assert ids == ["steam_wand", "portafilter"]  # duplicate name dropped
    assert "safety" not in data["parts"][0] and data["parts"][1]["safety"]
    assert data["procedures"][0]["steps"][0]["parts"] == ["steam_wand"]
    assert data["procedures"][0]["steps"][1] == {"text": "Close it", "parts": []}
    assert data["vocab"] == [{"term": "sputter", "maps_to": "steam_wand"}]


class _FakeReader:
    """A frozen screen reading, so Desktop mode can be tested without a display."""
    available, error = True, ""

    def __init__(self, labels):
        self.reading = Reading("Figma", [TextItem(text, (100 * i, 50, 100 * i + 80, 70), 0.99)
                                         for i, text in enumerate(labels)], time.monotonic())
        self.refreshes = 0

    def request_refresh(self):
        self.refreshes += 1

    def fresh_reading(self, max_age=1.5, timeout=4.0):
        return self.reading


class _FakeOverlay:
    def __init__(self):
        self.sent = []

    def send(self, command):
        self.sent.append(command)


def test_screen_matching_and_actions():
    labels = ["File", "Edit", "Export frame", "Share", "Layers", "Assets", "Prototype", "Settings"]
    assert match_items(_FakeReader(labels).reading.items, "the export button")[0][1].text == "Export frame"
    assert match_items(_FakeReader(labels).reading.items, "setings")[0][1].text == "Settings"   # STT/OCR typo
    assert match_items(_FakeReader(labels).reading.items, "the timeline") == []
    overlay = _FakeOverlay()
    sight = ScreenSight(reader=_FakeReader(labels), overlay=overlay, move_pointer=False)
    found = json.loads(sight.act("locate", {"query": "where's share"}))
    assert found["found"] and found["text"] == "Share" and found["ringed_on_screen"]
    assert overlay.sent[-1]["ring"] == [300, 50, 380, 70] and overlay.sent[-1]["mode"] == "solid"
    missing = json.loads(sight.act("locate", {"query": "timeline"}))
    assert missing["found"] is False and "Layers" in missing["some_visible_text"]
    overview = json.loads(sight.act("describe", {}))
    assert overview["window"] == "Figma" and "Prototype" in overview["visible_text"]
    names = [t["name"] for t in tool_definitions(sight)]
    assert "expand" not in names and "load_machine" not in names and len(names) <= 10


def test_screen_speculation():
    overlay = _FakeOverlay()
    reader = _FakeReader(["File", "Export frame", "Share"])
    sight = ScreenSight(reader=reader, overlay=overlay, move_pointer=False)
    spec = SpeculativeExecutor(sight)
    spec.on_partial_turn("where's the")
    assert reader.refreshes == 1                      # re-read started while still talking
    spec.on_partial_turn("where's the export")
    assert overlay.sent[-1]["mode"] == "ghost" and overlay.sent[-1]["label"] == "Export frame"
    assert spec.claim("locate", {"query": "Export"}) is not None
    spec.on_partial_turn("where's the share button")
    assert spec.claim("locate", {"query": "File"}) is None
    assert overlay.sent[-1] == {"clear": True}


def test_wake_parsing():
    assert NetraEngine._detect_wake("Hey Netra, where is the nozzle?")
    assert NetraEngine._detect_wake("hey nitra")
    assert not NetraEngine._detect_wake("the network is down")
    assert NetraEngine._after_wake("Hey Netra, where is the nozzle?") == "where is the nozzle"
    assert NetraEngine._after_wake("Hey Netra.") is None
    assert NetraEngine._detect_stop("Netra, stop")


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"\033[32mPASS\033[0m {name}")
            except AssertionError as exc:
                failures += 1
                print(f"\033[31mFAIL\033[0m {name}: {exc}")
    sys.exit(1 if failures else 0)
