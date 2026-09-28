"""
On-screen highlight for Netra Desktop: a glowing ring around a target, drawn with
borderless always-on-top windows. Runs as its own process (tkinter wants its own
main thread) and reads JSON commands from stdin:

    {"ring": [x1, y1, x2, y2], "label": "Export", "mode": "solid" | "ghost"}
    {"clear": true}

    python -m src.sidebar.overlay      # then type commands
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk

COLORS = {"solid": "#22d3ee", "ghost": "#f5a524"}
THICK = 4
PAD = 6
HIDE_AFTER_MS = 7000


class Overlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.edges = [self._window() for _ in range(4)]
        self.tag = self._window()
        self.text = tk.Label(self.tag, font=("Inter", 13, "bold"), fg="#0b0d10", padx=10, pady=4)
        self.text.pack()
        self.commands: queue.Queue = queue.Queue()
        self.hide_job = None
        threading.Thread(target=self._read_stdin, daemon=True).start()
        self.root.after(30, self._poll)

    def _window(self) -> tk.Toplevel:
        w = tk.Toplevel(self.root)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        try:
            w.attributes("-type", "dock")  # keeps it above normal windows on most X11 WMs
        except tk.TclError:
            pass
        w.withdraw()
        return w

    def _read_stdin(self):
        for line in sys.stdin:
            try:
                self.commands.put(json.loads(line))
            except ValueError:
                continue
        self.commands.put({"quit": True})

    def _poll(self):
        while not self.commands.empty():
            cmd = self.commands.get()
            if cmd.get("quit"):
                self.root.destroy()
                return
            if cmd.get("clear"):
                self.hide()
            elif cmd.get("ring"):
                self.show(cmd["ring"], cmd.get("label", ""), cmd.get("mode", "solid"))
        self.root.after(30, self._poll)

    def show(self, box, label: str, mode: str):
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = x1 - PAD, y1 - PAD, x2 + PAD, y2 + PAD
        color = COLORS.get(mode, COLORS["solid"])
        rects = [(x1, y1 - THICK, x2 - x1, THICK), (x1, y2, x2 - x1, THICK),
                 (x1 - THICK, y1 - THICK, THICK, y2 - y1 + 2 * THICK), (x2, y1 - THICK, THICK, y2 - y1 + 2 * THICK)]
        for w, (x, y, width, height) in zip(self.edges, rects):
            w.configure(bg=color)
            w.geometry(f"{max(width, 1)}x{max(height, 1)}+{x}+{y}")
            w.deiconify()
            w.lift()
        if label:
            self.text.configure(text=("predicting · " if mode == "ghost" else "") + label, bg=color)
            self.tag.configure(bg=color)
            self.tag.geometry(f"+{x1}+{max(0, y1 - THICK - 34)}")
            self.tag.deiconify()
            self.tag.lift()
        else:
            self.tag.withdraw()
        if self.hide_job:
            self.root.after_cancel(self.hide_job)
        self.hide_job = self.root.after(HIDE_AFTER_MS, self.hide)

    def hide(self):
        for w in [*self.edges, self.tag]:
            w.withdraw()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the engine; it closes our stdin to stop us
    Overlay().run()
