"""
Learn a machine Netra doesn't know yet.

"Netra, I'm working on my Breville Barista Express" → load_machine finds nothing in
the library → MachineLearner runs in the background:

  1. search_live for the machine's manual / parts / troubleshooting pages (Anakin.io)
  2. scrape_live the two best pages in parallel
  3. AssemblyAI LLM Gateway drafts a manifest (parts, aliases, safety, procedures)
  4. the draft is validated and repaired (ids, step→part references, sizes)
  5. saved to manifests/learned/<slug>.yaml, added to the library, viewer switches,
     and the voice agent announces it

Progress is streamed to the viewer as "learning" events.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from .sight import Manifest, MachineLibrary

LLM_GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"
LLM_MODEL = os.environ.get("NETRA_LLM_MODEL", "qwen3.5-4b-32k-fast")

_PROMPT = """You are building a repair-assistant knowledge pack for this machine: {machine}

Use ONLY the source material below (manuals, help pages, search snippets). Do not invent parts or \
procedures that the sources don't support. Output ONE JSON object and nothing else, with this shape:

{{
  "name": "<official machine name>",
  "aliases": ["<short names people say, e.g. 'espresso machine', 'Barista Express'>"],
  "parts": [
    {{"id": "<snake_case>", "name": "<part name>", "aliases": ["<other names>"],
      "description": "<one sentence: what it does / what goes wrong>",
      "safety": "<one short sentence, only if handling it is hazardous>"}}
  ],
  "vocab": [{{"term": "<symptom or jargon word>", "maps_to": "<part id>"}}],
  "procedures": [
    {{"name": "<task, e.g. 'descale the machine'>", "aliases": ["<symptoms or phrasings>"],
      "steps": [{{"text": "<one spoken instruction, <= 25 words>", "parts": ["<part id>"]}}]}}
  ]
}}

Rules: 6 to 12 parts a user can see or touch; 2 to 4 procedures with 3 to 6 steps each; every step's
"parts" must use ids from "parts"; the first step of any procedure involving heat, water pressure,
electricity or blades must be the safety step.

SOURCES
{sources}
"""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "machine"


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_draft(draft: dict, fallback_name: str) -> dict:
    """Repair a model-drafted manifest into something Manifest.load accepts and the tools can use."""
    name = str(draft.get("name") or fallback_name).strip()[:80]
    parts, ids, names = [], set(), set()
    for raw in draft.get("parts") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        if str(raw["name"]).strip().lower() in names:
            continue  # same name = same part
        names.add(str(raw["name"]).strip().lower())
        pid = _slug(str(raw.get("id") or raw["name"]))
        base, n = pid, 2
        while pid in ids:
            pid, n = f"{base}_{n}", n + 1
        ids.add(pid)
        part = {"id": pid, "name": str(raw["name"]).strip()[:60],
                "aliases": [str(a)[:40] for a in (raw.get("aliases") or []) if isinstance(a, str)][:6],
                "description": str(raw.get("description") or "").strip()[:240]}
        safety = str(raw.get("safety") or "").strip()
        if safety.lower().rstrip(".") not in {"", "none", "n/a", "na", "null", "no", "not applicable"}:
            part["safety"] = safety[:200]
        parts.append(part)
    parts = parts[:14]
    by_name = {p["name"].lower(): p["id"] for p in parts}

    def ref(value) -> str | None:
        key = str(value).strip()
        if key in ids:
            return key
        return _slug(key) if _slug(key) in ids else by_name.get(key.lower())

    procedures = []
    for raw in draft.get("procedures") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        steps = []
        for step in raw.get("steps") or []:
            text = step.get("text") if isinstance(step, dict) else step
            if not text:
                continue
            refs = [r for r in (ref(p) for p in (step.get("parts") or [] if isinstance(step, dict) else [])) if r]
            steps.append({"text": str(text).strip()[:220], "parts": list(dict.fromkeys(refs))})
        if steps:
            procedures.append({"name": str(raw["name"]).strip()[:60],
                               "aliases": [str(a)[:50] for a in (raw.get("aliases") or []) if isinstance(a, str)][:6],
                               "steps": steps[:8]})
    vocab = []
    for raw in draft.get("vocab") or []:
        if isinstance(raw, dict) and raw.get("term") and ref(raw.get("maps_to", "")):
            vocab.append({"term": str(raw["term"])[:40], "maps_to": ref(raw["maps_to"])})
    aliases = [str(a)[:40] for a in (draft.get("aliases") or []) if isinstance(a, str)][:8]
    return {"manifest": {"domain": "physical_machine", "name": name, "aliases": aliases,
                         "source": "schematic", "learned": True,
                         "parts": parts, "vocab": vocab[:12], "procedures": procedures[:5]}}


class MachineLearner:
    def __init__(self, library: MachineLibrary, out_dir: Path, hub=None,
                 on_ready: Callable[[Manifest], None] | None = None,
                 on_failed: Callable[[str, str], None] | None = None):
        self.on_failed = on_failed
        self.library = library
        self.out_dir = Path(out_dir)
        self.hub = hub
        self.on_ready = on_ready
        self._busy: str | None = None

    def _progress(self, machine: str, stage: str, detail: str = ""):
        print(f"  \033[1;35m[learn]\033[0m {machine}: {stage} {detail}")
        if self.hub:
            self.hub.broadcast({"type": "learning", "machine": machine, "stage": stage, "detail": detail})

    def start(self, machine: str) -> str:
        """Called from the load_machine tool. Returns immediately; the work runs in the background."""
        if self._busy:
            return json.dumps({"learning": self._busy, "status": "already learning another machine; try again when it's done"})
        self._busy = machine
        threading.Thread(target=self._run_safely, args=(machine,), daemon=True).start()
        return json.dumps({"learning": machine, "status": "started", "eta_seconds": 30,
                           "tell_user": "You're reading the manual and building a pack; it takes under a minute "
                                        "and you'll say when it's ready. Offer to keep helping meanwhile."})

    def _run_safely(self, machine: str):
        try:
            manifest = self.learn(machine)
            self._progress(machine, "ready", f"{len(manifest.parts)} parts · {len(manifest.procedures)} procedures")
            if self.on_ready:
                self.on_ready(manifest)
        except Exception as exc:
            self._progress(machine, "failed", str(exc)[:160])
            if self.on_failed:
                self.on_failed(machine, str(exc)[:160])
        finally:
            self._busy = None

    def learn(self, machine: str) -> Manifest:
        from .tools import _scrape_live, _search_live  # late import: tools imports sight

        started = time.monotonic()
        self._progress(machine, "searching", "looking for the manual and troubleshooting guides")
        found = json.loads(_search_live(f"{machine} parts diagram troubleshooting user manual", self.hub) or "{}")
        results = found.get("results") if isinstance(found, dict) else None
        if not results:
            raise RuntimeError("no web results for that machine")
        snippets = "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)

        self._progress(machine, "reading", ", ".join(r["title"][:40] for r in results[:2]))
        with ThreadPoolExecutor(max_workers=2) as pool:
            pages = list(pool.map(_scrape_live, [r["url"] for r in results[:2]]))
        texts = []
        for raw in pages:
            try:
                texts.append(json.loads(raw)["content"][:9000])
            except (ValueError, KeyError, TypeError):
                continue
        sources = "SEARCH SNIPPETS\n" + snippets + "".join(f"\n\nPAGE {i + 1}\n{t}" for i, t in enumerate(texts))

        self._progress(machine, "drafting", f"building the parts list from {len(texts)} page(s)")
        draft = self._draft(machine, sources)
        data = normalize_draft(draft, machine)
        if len(data["manifest"]["parts"]) < 3:
            raise RuntimeError("the sources didn't describe enough parts")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{_slug(data['manifest']['name'])}.yaml"
        header = (f"# Learned by Netra from the web on {time.strftime('%Y-%m-%d %H:%M')} "
                  f"in {time.monotonic() - started:.0f}s. Sources:\n"
                  + "".join(f"#   {r['url']}\n" for r in results[:3]))
        path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        manifest = Manifest.load(path)
        self.library.add(manifest)
        return manifest

    def _draft(self, machine: str, sources: str) -> dict[str, Any]:
        key = os.environ.get("ASSEMBLYAI_API_KEY", "")
        messages = [{"role": "user", "content": _PROMPT.format(machine=machine, sources=sources[:24000])}]
        for _ in range(2):
            response = httpx.post(LLM_GATEWAY_URL, headers={"Authorization": key},
                                  json={"model": LLM_MODEL, "messages": messages, "max_tokens": 4000},
                                  timeout=90)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"] or ""
            draft = _extract_json(text)
            if draft and draft.get("parts"):
                return draft
            messages += [{"role": "assistant", "content": text[:4000]},
                         {"role": "user", "content": "That was not a single valid JSON object with parts. "
                                                     "Reply with only the JSON object."}]
        raise RuntimeError("the language model did not return a usable parts list")
