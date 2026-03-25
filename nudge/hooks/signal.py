"""Signal feedback via signal-cli-rest-api (bbernhard/signal-cli-rest-api)."""
# no polls in Signal. send a follow-up prompt, wait for "1" or "2" reply.
# docker run -p 8080:8080 bbernhard/signal-cli-rest-api

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from nudge import db

SIGNAL_API = "http://localhost:8080"
ACCOUNT = ""  # your registered Signal number e.g. "+15551234567"
_pending = {}  # sender number → {model, prompt, response}
_model = "unknown"
_generate_fn = None


def _send(number, text):
    # v2 send endpoint
    requests.post(f"{SIGNAL_API}/v2/send", json={
        "message": text, "number": ACCOUNT, "recipients": [number]
    }, timeout=10)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        # signal-cli-rest-api wraps each inbound as an envelope
        envelope = body.get("envelope", {})
        msg = envelope.get("dataMessage", {})
        text = (msg.get("message") or "").strip()
        sender = envelope.get("source", "")

        if not text or not sender:
            self._ok(); return

        # check if this is a rating reply for a pending exchange
        if sender in _pending and text in ("1", "2"):
            pair = _pending.pop(sender)
            rating = 1 if text == "1" else -1
            conn = db.connect()
            db.add_feedback(conn, pair["model"], pair["prompt"],
                            pair["response"], rating, source="signal")
            conn.close()
            _send(sender, "Recorded. Thanks.")
            self._ok(); return

        # normal message → generate + ask for rating
        resp = _generate_fn(text) if _generate_fn else f"Echo: {text}"
        _send(sender, resp)
        _send(sender, "Reply 1=Good, 2=Bad to rate")
        _pending[sender] = {"model": _model, "prompt": text, "response": resp}

        self._ok()

    def _ok(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a): pass


class SignalFeedbackBot:
    """Minimal bot. Provide generate_fn(prompt) → response string."""
    def __init__(self, generate_fn=None, model="unknown", port=5002,
                 signal_url=None, account=None):
        global _model, _generate_fn, SIGNAL_API, ACCOUNT
        _model, _generate_fn = model, generate_fn or (lambda t: f"Echo: {t}")
        if signal_url: SIGNAL_API = signal_url
        if account: ACCOUNT = account
        self.port = port

    def run(self):
        print(f"Signal bot on :{self.port} (signal-cli-rest-api at {SIGNAL_API})")
        HTTPServer(("127.0.0.1", self.port), _Handler).serve_forever()


if __name__ == "__main__":
    SignalFeedbackBot().run()
