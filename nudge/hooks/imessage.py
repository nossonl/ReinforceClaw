# macOS only. Requires Full Disk Access for Terminal/Python to read chat.db.
"""iMessage feedback hook. Polls chat.db, sends rating prompt after bot reply."""

import sqlite3
import subprocess
import time
from nudge import db

from pathlib import Path
CHAT_DB = str(Path.home() / "Library/Messages/chat.db")
RATING_PROMPT = "Reply 1=Good 2=Bad to rate"

_model = "unknown"
_generate_fn = None
# phone → {model, prompt, response} waiting on a rating
_pending = {}


def _osascript_string(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _send(phone, text):
    # applescript to send — phone must be e.164 or email
    script = (
        'tell application "Messages" to send '
        f'"{_osascript_string(text)}" to buddy "{_osascript_string(phone)}" of service "SMS"'
    )
    subprocess.run(["osascript", "-e", script], capture_output=True)


def _last_rowid(conn):
    r = conn.execute("SELECT MAX(ROWID) FROM message").fetchone()
    return r[0] or 0


def _new_messages(conn, since_rowid):
    # grab inbound texts newer than our last seen rowid
    rows = conn.execute("""
        SELECT m.ROWID, h.id AS phone, m.text
        FROM message m
        JOIN handle h ON m.handle_id = h.ROWID
        WHERE m.ROWID > ? AND m.is_from_me = 0 AND m.text IS NOT NULL
        ORDER BY m.ROWID ASC
    """, (since_rowid,)).fetchall()
    return rows


class IMessageFeedbackBot:
    """Poll chat.db, respond, collect 1/2 ratings."""

    def __init__(self, generate_fn=None, model="unknown", poll_interval=3):
        global _model, _generate_fn
        _model = model
        _generate_fn = generate_fn or (lambda t: f"Echo: {t}")
        self.poll_interval = poll_interval

    def run(self):
        # open chat.db read-only so we don't corrupt Messages
        chat_conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
        last = _last_rowid(chat_conn)
        print(f"iMessage bot polling chat.db (last rowid={last})")

        while True:
            time.sleep(self.poll_interval)
            rows = _new_messages(chat_conn, last)
            if not rows:
                continue

            for rowid, phone, text in rows:
                last = max(last, rowid)
                text = text.strip()

                if phone in _pending and text in ("1", "2"):
                    # got a rating
                    pair = _pending.pop(phone)
                    rating = 1 if text == "1" else -1
                    conn = db.connect()
                    db.add_feedback(conn, pair["model"], pair["prompt"],
                                    pair["response"], rating, source="imessage")
                    conn.close()
                    continue

                # new prompt — respond and ask for rating
                resp = _generate_fn(text)
                _send(phone, resp)
                _send(phone, RATING_PROMPT)
                _pending[phone] = {"model": _model, "prompt": text, "response": resp}


if __name__ == "__main__":
    IMessageFeedbackBot().run()
