#!/usr/bin/env python3
"""Codex desktop app hook. Two small circles under each response."""
# two circles with black outlines. click left → green (good). click right → red (bad).
# click a filled one again → goes back to empty (undo). circles never disappear.

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db
from nudge.cli import load_config as _load_config


def _show_buttons(feedback_id, config):
    import tkinter as tk

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-transparent", True)
        root.configure(bg="systemTransparent")
    except tk.TclError:
        root.attributes("-alpha", 0.95)
        root.configure(bg="#2d2d2d")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"80x32+{sw // 2 - 40}+{sh - 70}")

    selected = [None]
    colors = {1: "#22c55e", -1: "#ef4444"}

    def _click(canvas, oval, other_canvas, other_oval, rating):
        if selected[0] == rating:
            canvas.itemconfig(oval, fill="")
            conn = db.connect()
            db.update_feedback_rating(conn, feedback_id, 0)
            conn.close()
            selected[0] = None
            return
        canvas.itemconfig(oval, fill=colors[rating])
        other_canvas.itemconfig(other_oval, fill="")
        conn = db.connect()
        db.update_feedback_rating(conn, feedback_id, rating)
        if db.count_trainable_untrained(conn) >= config.get("batch_min", 16):
            subprocess.Popen(
                [sys.executable, "-m", "nudge.cli", "train"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        conn.close()
        selected[0] = rating

    bg = root.cget("bg")
    frame = tk.Frame(root, bg=bg)
    frame.pack(expand=True)

    c1 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o1 = c1.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c1.pack(side="left", padx=4)

    c2 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o2 = c2.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c2.pack(side="left", padx=4)

    c1.bind("<Button-1>", lambda e: _click(c1, o1, c2, o2, 1))
    c2.bind("<Button-1>", lambda e: _click(c2, o2, c1, o1, -1))

    root.mainloop()


def handle_stop():
    try:
        data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    except (json.JSONDecodeError, EOFError):
        data = {}
    config = _load_config()
    if not config.get("model") or not config.get("panel_enabled", True):
        return

    last_msg = data.get("last_assistant_message", "")
    if not last_msg:
        return

    conn = db.connect()
    fid = db.add_feedback(conn, config["model"], "(codex)", last_msg, 0, source="hook")
    conn.close()

    # NOT daemon — thread must stay alive for tkinter mainloop
    t = threading.Thread(target=_show_buttons, args=(fid, config))
    t.start()


if __name__ == "__main__":
    import threading
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        handle_stop()
