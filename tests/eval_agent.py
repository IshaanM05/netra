#!/usr/bin/env python3
"""
Text-mode evaluation harness for Netra's Voice Agent API configuration.

Each scenario opens a real Voice Agent API session with the exact session config
the live app uses (agent.build_session), injects user turns as text
(conversation.message + reply.create — no TTS, no audio playback), runs the real
tools against the real manifest with a recording viewer hub, and checks which
tools were called, with what arguments, and what the viewer was told to show.

    python tests/eval_agent.py                 # all scenarios, 3 in parallel
    python tests/eval_agent.py -k clicking     # scenarios whose name contains 'clicking'
    python tests/eval_agent.py --debug -k heat # print every event
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sidebar import config  # noqa: E402
from src.sidebar.agent import build_session  # noqa: E402
from src.sidebar.sight import Manifest, ModelSight  # noqa: E402
from src.sidebar.tools import execute_tool  # noqa: E402
from src.sidebar.viewer import RecordingHub  # noqa: E402

MANIFEST = ROOT / "manifests" / "sample-machine.yaml"


# ---------------------------------------------------------------- expectations

@dataclass
class Expect:
    """What one user turn must produce. tool=None means 'no tool call expected'."""
    tool: str | None
    part: str | None = None          # resolved part id the tool must target
    procedure: str | None = None     # procedure name substring for walk_through
    step: int | None = None          # procedure step the viewer must show after this turn
    any_of: tuple[str, ...] = ()     # alternative acceptable tools
    say: tuple[str, ...] = ()        # substrings (lowercase) the spoken reply must contain (any)


@dataclass
class Scenario:
    name: str
    turns: list[tuple[str, Expect]]


def S(name, *turns):
    return Scenario(name, list(turns))


SCENARIOS: list[Scenario] = [
    # ---- locate: direct and indirect part references
    S("locate heatbreak", ("Where's the heatbreak?", Expect("locate", part="heatbreak"))),
    S("locate nozzle", ("Show me the nozzle.", Expect("locate", part="nozzle", any_of=("point",)))),
    S("locate alias hot end", ("Which part is the hot end?", Expect("locate", part="hotend"))),
    S("locate alias bowden", ("Where does the bowden tube go in?", Expect("locate", part="ptfe_tube"))),
    S("locate thermistor", ("Can you find the thermistor for me?", Expect("locate", part="thermistor"))),
    S("locate idler", ("Where's the idler lever?", Expect("locate", part="idler_lever"))),
    S("locate blower", ("Which one is the blower fan?", Expect("locate", part="print_fan"))),
    S("point motor", ("Point at the extruder motor.", Expect("point", part="extruder_motor", any_of=("locate",)))),
    S("locate runout sensor", ("Where is the filament runout sensor?", Expect("locate", part="filament_sensor"))),
    # ---- describe
    S("describe heatsink", ("What does the heatsink do?", Expect("describe", part="heatsink", any_of=("locate",)))),
    S("describe gears", ("What are the drive gears for?", Expect("describe", part="gearbox", any_of=("locate",)))),
    S("describe overview", ("What am I looking at?", Expect("describe"))),
    S("describe hotend fan", ("Tell me about the hotend fan.", Expect("describe", part="hotend_fan", any_of=("locate",)))),
    # ---- expand / reset
    S("explode all", ("Take the whole thing apart so I can see inside.", Expect("expand"))),
    S("explode hotend", ("Show me an exploded view around the heater block.", Expect("expand", part="hotend"))),
    S("explode then reset",
      ("Show me how it comes apart.", Expect("expand")),
      ("Okay, put it back together.", Expect("reset_view"))),
    # ---- procedures: single and multi-turn with 'next'
    S("clicking procedure 3 steps",
      ("My extruder keeps clicking, can you help?", Expect("walk_through", procedure="clicking", step=1)),
      ("Done, what's next?", Expect("walk_through", procedure="clicking", step=2)),
      ("Okay the spool is fine. Next.", Expect("walk_through", procedure="clicking", step=3))),
    S("cold pull procedure",
      ("How do I do a cold pull?", Expect("walk_through", procedure="cold pull", step=1)),
      ("Got it.", Expect("walk_through", procedure="cold pull", step=2))),
    S("replace nozzle procedure",
      ("Walk me through replacing the nozzle.", Expect("walk_through", procedure="nozzle", step=1))),
    S("temperature error procedure",
      ("The printer says thermal runaway, what do I do?", Expect("walk_through", procedure="temperature", step=1))),
    S("clogged nozzle procedure",
      ("I think my nozzle is clogged.", Expect("walk_through", procedure="clog", step=1))),
    S("procedure then locate mid-way",
      ("My extruder is clicking.", Expect("walk_through", procedure="clicking", step=1)),
      ("Wait, where's the PTFE tube?", Expect("locate", part="ptfe_tube")),
      ("Okay, next step.", Expect("walk_through", procedure="clicking", step=2))),
    # ---- vocab / symptoms
    S("vocab heat creep", ("What part causes heat creep?", Expect("locate", part="heatbreak", any_of=("describe",)))),
    # ---- utility still works
    S("calculate", ("If I print at 12 millimetres per second for 90 seconds, how far is that?", Expect("calculate"))),
    # ---- should NOT call a visual tool
    S("chit chat", ("Thanks, that's really helpful.", Expect(None))),
]


# ---------------------------------------------------------------- runner

INJECT = "instructions"


def inject_payloads(text: str, mode: str) -> list[dict]:
    """How a typed user turn is fed to the agent. conversation.message(role=user) alone does not
    reach the model (verified 2026-09-28), so the default wraps the turn in reply.create instructions."""
    if mode == "system":
        return [{"type": "conversation.message", "role": "system", "content": f'The user says: "{text}"'},
                {"type": "reply.create"}]
    if mode == "both":
        return [{"type": "conversation.message", "role": "user", "content": text},
                {"type": "reply.create", "instructions": f'The user just said: "{text}". Reply to that now.'}]
    return [{"type": "reply.create", "instructions":
             f'The user just said: "{text}". Treat this exactly as if they had spoken it: reply to it '
             "now, following your system prompt and calling tools as usual."}]

@dataclass
class TurnResult:
    user: str
    tools: list[tuple[str, dict]] = field(default_factory=list)
    reply: str = ""
    events: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    ok: bool = False
    why: str = ""


class Session:
    def __init__(self, sight: ModelSight, hub: RecordingHub, debug: bool = False):
        self.sight, self.hub, self.debug = sight, hub, debug
        self.ws = None
        self.turn_state = ""

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {config.API_KEY}"}
        for attempt in range(5):
            self.ws = await websockets.connect(config.AGENT_WS_URL, additional_headers=headers,
                                               max_size=None)
            await self.ws.send(json.dumps({"type": "session.update", "session": build_session(self.sight)}))
            while True:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), 15))
                if msg["type"] == "session.ready":
                    return self
                if msg["type"] == "session.error":
                    await self.ws.close()
                    if msg.get("code") == "at_capacity":
                        await asyncio.sleep(2 + 2 * attempt)
                        break
                    raise RuntimeError(f"session.error: {msg}")
        raise RuntimeError("could not open a session (at capacity)")

    async def __aexit__(self, *exc):
        try:
            await self.ws.send(json.dumps({"type": "session.end"}))
            await self.ws.close()
        except Exception:
            pass

    async def turn(self, text: str, timeout: float = 30) -> TurnResult:
        result = TurnResult(user=text)
        start_events = len(self.hub.events)
        started = time.monotonic()
        for payload in inject_payloads(text, INJECT):
            await self.ws.send(json.dumps(payload))
        pending: list[tuple[str, str]] = []
        outstanding = 0
        deadline = started + timeout
        while time.monotonic() < deadline:
            try:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), deadline - time.monotonic()))
            except asyncio.TimeoutError:
                result.why = "timeout"
                break
            t = msg.get("type")
            if self.debug and t not in ("reply.audio", "transcript.agent.delta"):
                print(f"      {time.monotonic() - started:5.2f}s {t} {json.dumps(msg)[:220]}")
            if t in ("reply.started", "input.speech.started"):
                self.turn_state = t
            elif t == "tool.call":
                args = msg.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result.tools.append((msg["name"], args))
                output = await asyncio.to_thread(execute_tool, msg["name"], args, self.sight)
                pending.append((msg["call_id"], output))
                outstanding += 1
                if self.turn_state == "reply.done":
                    outstanding -= await self._flush(pending)
            elif t == "transcript.agent":
                result.reply = (result.reply + " " + msg.get("text", "")).strip()
            elif t == "reply.done":
                self.turn_state = t
                if msg.get("status") == "interrupted":
                    pending.clear()
                if pending:
                    outstanding -= await self._flush(pending)
                    continue
                # Finished when a reply completes with no tool results still owed to the agent.
                if outstanding == 0:
                    break
                outstanding = 0
        result.seconds = time.monotonic() - started
        result.events = self.hub.events[start_events:]
        return result

    async def _flush(self, pending) -> int:
        sent = 0
        for call_id, output in pending:
            await self.ws.send(json.dumps({"type": "tool.result", "call_id": call_id,
                                           "result": json.dumps({"result": output})}))
            sent += 1
        pending.clear()
        return sent


def check(sight: ModelSight, expect: Expect, res: TurnResult) -> tuple[bool, str]:
    names = [name for name, _ in res.tools]
    if expect.tool is None:
        visual = [n for n in names if n in {"locate", "point", "expand", "walk_through", "describe"}]
        return (not visual, f"unexpected tools {names}" if visual else "")
    allowed = {expect.tool, *expect.any_of}
    calls = [(n, a) for n, a in res.tools if n in allowed]
    if not calls:
        return False, f"expected {expect.tool}, got {names or 'no tool'}"
    if expect.part:
        targets = []
        for _, args in calls:
            part = sight.resolve(str(args.get("query") or args.get("target") or ""))
            targets.append(part and part.get("id"))
        if expect.part not in targets:
            return False, f"expected part {expect.part}, tool targeted {targets} ({calls})"
    if expect.procedure or expect.step:
        steps = [e for e in res.events if e.get("type") == "step"]
        if not steps:
            return False, f"no procedure step shown ({calls})"
        last = steps[-1]
        if expect.procedure and expect.procedure not in str(last.get("procedure", "")).lower():
            return False, f"wrong procedure {last.get('procedure')}"
        if expect.step and last.get("step") != expect.step:
            return False, f"expected step {expect.step}, viewer shows step {last.get('step')} ({calls})"
    if expect.say and not any(s in res.reply.lower() for s in expect.say):
        return False, f"reply missing {expect.say}"
    return True, ""


async def run_scenario(scn: Scenario, sem: asyncio.Semaphore, debug: bool) -> list[TurnResult]:
    async with sem:
        hub = RecordingHub()
        sight = ModelSight(Manifest.load(MANIFEST), renderer=hub)
        results = []
        try:
            async with Session(sight, hub, debug) as session:
                for text, expect in scn.turns:
                    res = await session.turn(text)
                    res.ok, res.why = check(sight, expect, res) if not res.why else (False, res.why)
                    results.append(res)
        except Exception as exc:
            res = TurnResult(user=scn.turns[len(results)][0], why=f"error: {exc}")
            results.append(res)
        mark = "\033[32mPASS\033[0m" if all(r.ok for r in results) and len(results) == len(scn.turns) else "\033[31mFAIL\033[0m"
        print(f"{mark} {scn.name}")
        for r in results:
            tools = ", ".join(f"{n}({json.dumps(a)})" for n, a in r.tools) or "-"
            flag = "  " if r.ok else "✗ "
            print(f"   {flag}{r.user!r} [{r.seconds:.1f}s] → {tools}")
            print(f"      \"{r.reply[:140]}\"" + (f"   <-- {r.why}" if not r.ok else ""))
        return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", default="", help="only scenarios whose name contains this")
    parser.add_argument("-j", type=int, default=3, help="parallel sessions")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--inject", default="instructions", choices=["instructions", "system", "both"])
    parser.add_argument("--repeat", type=int, default=1, help="run each scenario N times (flakiness)")
    args = parser.parse_args()

    global INJECT
    INJECT = args.inject
    scenarios = [s for s in SCENARIOS if args.k.lower() in s.name.lower()] * args.repeat
    sem = asyncio.Semaphore(args.j)
    started = time.monotonic()
    all_results = await asyncio.gather(*(run_scenario(s, sem, args.debug) for s in scenarios))
    turns = [r for rs in all_results for r in rs]
    passed_turns = sum(r.ok for r in turns)
    passed_scn = sum(all(r.ok for r in rs) and len(rs) == len(s.turns) for rs, s in zip(all_results, scenarios))
    latencies = sorted(r.seconds for r in turns if r.ok)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    print(f"\n{'=' * 64}\n  scenarios {passed_scn}/{len(scenarios)}   turns {passed_turns}/{len(turns)}"
          f"   p50 turn {p50:.1f}s   wall {time.monotonic() - started:.0f}s\n{'=' * 64}")
    return 0 if passed_scn == len(scenarios) else 1


if __name__ == "__main__":
    if not config.API_KEY:
        sys.exit("Set ASSEMBLYAI_API_KEY in .env")
    sys.exit(asyncio.run(main()))
