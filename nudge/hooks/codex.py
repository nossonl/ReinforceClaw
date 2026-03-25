#!/usr/bin/env python3
"""Codex desktop app hook. Two small circles under each response."""
# two empty circles with black outlines. press one → fills green → records to db → disappears.
# press again on a filled one → unfills and disappears. don't press either → stays, gets skipped.

import json
import sys
import threading
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

    selected = [None]  # track which is filled

    def _click(canvas, oval, rating):
        if selected[0] == rating:
            # already filled — unfill and dismiss
            root.destroy()
            return
        # fill green, record, dismiss
        canvas.itemconfig(oval, fill="#22c55e")
        canvas.update()
        conn = db.connect()
        db.update_feedback_rating(conn, feedback_id, rating)
        if db.count_trainable_untrained(conn) >= config.get("batch_min", 16):
            from nudge.hooks.claude_code import _queue_training
            _queue_training(config)
        conn.close()
        selected[0] = rating
        root.after(300, root.destroy)  # brief flash so you see the green

    bg = root.cget("bg")
    frame = tk.Frame(root, bg=bg)
    frame.pack(expand=True)

    # two empty circles, black outline, no fill
    for rating in (1, -1):
        c = tk.Canvas(frame, width=24, height=24, bg=bg, highlightthickness=0, cursor="hand2")
        oval = c.create_oval(2, 2, 22, 22, fill="", outline="black", width=2)
        c.bind("<Button-1>", lambda e, cv=c, ov=oval, r=rating: _click(cv, ov, r))
        c.pack(side="left", padx=4)

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

    # buttons in a thread so hook doesn't block codex
    t = threading.Thread(target=_show_buttons, args=(fid, config), daemon=True)
    t.start()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        handle_stop()
