#!/usr/bin/env python3
"""Codex desktop app hook. Two circles under each response + X to close."""
# click left → green (good). click right → red (bad).
# click filled one again → unfills (undo). X closes the circles.
# circles reappear after every response.

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db
from nudge.hooks._common import load_config, maybe_train

SOURCE = "codex"


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
    root.geometry(f"100x32+{sw // 2 - 50}+{sh - 70}")

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
        maybe_train(conn, config)
        conn.close()
        selected[0] = rating

    bg = root.cget("bg")
    frame = tk.Frame(root, bg=bg)
    frame.pack(expand=True)

    c1 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o1 = c1.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c1.pack(side="left", padx=3)

    c2 = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
    o2 = c2.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
    c2.pack(side="left", padx=3)

    # X button to dismiss
    close = tk.Label(frame, text="✕", font=("Arial", 10), fg="gray", bg=bg, cursor="hand2")
    close.pack(side="left", padx=3)
    close.bind("<Button-1>", lambda e: root.destroy())

    c1.bind("<Button-1>", lambda e: _click(c1, o1, c2, o2, 1))
    c2.bind("<Button-1>", lambda e: _click(c2, o2, c1, o1, -1))

    root.mainloop()


def handle_stop():
    try:
        data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
    except (json.JSONDecodeError, EOFError):
        data = {}
    config = load_config()
    if not config.get("model") or not config.get("panel_enabled", True):
        return

    last_msg = data.get("last_assistant_message", "")
    if not last_msg:
        return

    conn = db.connect()
    fid = db.add_feedback(conn, config["model"], "(codex)", last_msg, 0, source=SOURCE)
    conn.close()

    # not daemon — thread must stay alive for tkinter
    t = threading.Thread(target=_show_buttons, args=(fid, config))
    t.start()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        handle_stop()
