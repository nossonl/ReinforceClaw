"""WhatsApp feedback via WAHA (self-hosted WhatsApp HTTP API)."""
# WAHA wraps WhatsApp Web behind REST. docker run -p 3000:3000 devlikeapro/waha
# no official business API needed. polls replace buttons (deprecated for non-business).
# user taps poll option → webhook → we record rating. source="whatsapp".

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from nudge import db

WAHA = "http://localhost:3000"
_polls = {}  # poll_msg_id → {model, prompt, response}
_model = "unknown"
_generate_fn = None


def _send(endpoint, payload):
    return requests.post(f"{WAHA}{endpoint}", json=payload, timeout=10)


def _send_text(chat_id, text):
    _send("/api/sendText", {"chatId": chat_id, "text": text})


def _send_poll(chat_id, prompt, response):
    r = _send("/api/sendPoll", {"chatId": chat_id, "poll": {
        "name": "Rate this response:", "options": ["Good", "Bad", "Skip"],
        "multipleAnswers": False}})
    if r.ok:
        mid = r.json().get("key", {}).get("id", "")
        if mid:
            _polls[mid] = {"model": _model, "prompt": prompt, "response": response}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        event = body.get("event", "")
        p = body.get("payload", {})

        if event == "message" and not p.get("fromMe") and p.get("body", "").strip():
            chat_id, text = p["from"], p["body"].strip()
            resp = _generate_fn(text) if _generate_fn else f"Echo: {text}"
            _send_text(chat_id, resp)
            _send_poll(chat_id, text, resp)

        elif event == "poll.vote":
            poll_id = p.get("poll", {}).get("id", "")
            pair = _polls.pop(poll_id, None)
            if pair:
                choice = (p.get("vote", {}).get("selectedOptions") or [None])[0]
                rating = {"Good": 1, "Bad": -1}.get(choice)
                if rating:
                    conn = db.connect()
                    db.add_feedback(conn, pair["model"], pair["prompt"],
                                    pair["response"], rating, source="whatsapp")
                    conn.close()

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a): pass


class WhatsAppFeedbackBot:
    """Minimal bot. Provide generate_fn(prompt) → response string."""
    def __init__(self, generate_fn=None, model="unknown", port=5001, waha_url=None):
        global _model, _generate_fn, WAHA
        _model, _generate_fn = model, generate_fn or (lambda t: f"Echo: {t}")
        if waha_url:
            WAHA = waha_url
        self.port = port

    def run(self):
        requests.post(f"{WAHA}/api/webhook", json={
            "url": f"http://localhost:{self.port}/", "events": ["message", "poll.vote"]},
            timeout=10)
        print(f"WhatsApp bot on :{self.port} (WAHA at {WAHA})")
        HTTPServer(("127.0.0.1", self.port), _Handler).serve_forever()


if __name__ == "__main__":
    WhatsAppFeedbackBot().run()
