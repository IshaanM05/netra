# Netra — 2-minute demo script

**Setup:** laptop with headphones and a USB mic. Terminal: `python run.py --open`. Put the viewer full-screen
on the projector. Keep a second tab with `python tests/rehearse_viewer.py --loop` ready as a fallback: it plays
the same flow without a mic.

| # | Say | What the audience sees | Line to land |
|---|---|---|---|
| 0 | *(intro)* | Extruder slowly rotating | "Fixing hardware means your hands are busy and your eyes are on the machine. Netra is a voice agent that can see what you're working on." |
| 1 | "Hey Netra, where's the heatbreak?" | **Amber ghost** on the heatbreak while you're still saying it, then solid cyan with a label, and the camera flies in | "Watch the amber: that's Netra acting on my *partial* transcript, before I finished the sentence." |
| 2 | "Show me how it all comes apart." | Exploded view | |
| 3 | "My extruder keeps clicking. Walk me through it." | Procedure card 1/5; hotend and nozzle highlighted; Netra says the safety note first | "The steps come from the machine's manifest, not from the model's memory." |
| 4 | "Done. What's next?" | Card 2/5, PTFE tube highlighted | "It keeps its place in the procedure, handles 'go back' and 'repeat', and you can switch problems mid-way." |
| 5 | "Look up the official Prusa guide for a clogged nozzle." | Sources card with the Prusa link; spoken answer names the site | "Live web through Anakin. The search was already running before the agent asked for it." |
| 6 | *(point at the speculation meter)* | Fires / hits / misses / average head start | "Every hit is a tool result that was ready before the model asked for it. Speculation only touches read-only tools, so a wrong guess costs nothing." |
| 7 | "Netra, stop." | Status returns to listening | "One engine, any machine: swap the YAML manifest and GLB model." |

**If something breaks:**
- Agent at capacity: the engine retries automatically. Keep talking over it or switch to the rehearsal tab.
- Mis-heard wake word: say "Netra" clearly, or restart with `--autostart`.
- Wi-Fi down: the viewer needs jsdelivr for three.js, so preload the page before going on stage.

**Numbers to quote** (from `tests/eval_agent.py`): 40/40 scenarios and 53/53 turns pass; tool calls land
about 0.6 s after the request (p50). Quote the speculation head start from the meter during the run.
