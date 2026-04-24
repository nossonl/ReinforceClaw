"""OpenClaw backend. Receives POSTs from the TS plugin, writes to reinforceclaw DB."""
# the TS plugin (openclaw-plugin/) runs inside the gateway and catches messages
# from all 23+ platforms. it sends them here; only rated turns go to SQLite.
# run: python -m reinforceclaw.hooks.openclaw

import json
import hmac
import os
import time
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse
from reinforceclaw import db
from reinforceclaw.hooks._common import RESET_MARK_PATH, maybe_train, pop_pending, queue_training, save_pending

_config = None
_config_mtime = None
_latest_by_session = OrderedDict()  # capped at 1000 entries
_session_lock = RLock()
MAX_SESSIONS = 1000
MAX_PENDING_AGE = 1800
MAX_BODY = 1_048_576  # 1MB
SECRET_HEADER = "X-ReinforceClaw-Secret"


def _cfg():
    global _config, _config_mtime
    from reinforceclaw.cli import load_config
    path = Path.home() / ".reinforceclaw" / "config.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if _config is None or mtime != _config_mtime:
        _config, _config_mtime = load_config(), mtime
    return _config


def _shared_secret():
    return os.environ.get("REINFORCECLAW_OPENCLAW_SECRET") or _cfg().get("openclaw_secret") or None


def _authorized(headers):
    secret = _shared_secret()
    if not secret:
        return False
    got = headers.get(SECRET_HEADER)
    return bool(got) and hmac.compare_digest(got, secret)


def _status_payload(conn):
    counts = db.count(conn)
    ema, _ = db.get_ema(conn)
    adapter = db.latest_adapter(conn)
    return {
        "adapter": f"v{adapter['version']}" if adapter else "none",
        "ratings": counts["total"],
        "good": counts["good"],
        "bad": counts["bad"],
        "untrained": db.count_trainable_untrained(conn),
        "ema": round(ema, 3),
    }


def _send_json(handler, payload, code=200):
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.end_headers()
    handler.wfile.write(json.dumps(payload).encode())


def _send_error(handler, code):
    handler.send_response(code)
    handler.end_headers()


def _session_key(body):
    return str(body.get("sessionKey") or "").strip()


def _local_origin(origin):
    try:
        parsed = urlparse(origin)
        return parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False


def _status_response(self):
    conn = db.connect()
    payload = _status_payload(conn)
    conn.close()
    _send_json(self, payload)


def _prune_sessions():
    now = time.time()
    try:
        reset_at = Path(RESET_MARK_PATH).stat().st_mtime
    except OSError:
        reset_at = 0
    with _session_lock:
        for key, item in list(_latest_by_session.items()):
            ts = item.get("ts", now)
            if ts < reset_at or now - ts > MAX_PENDING_AGE:
                _latest_by_session.pop(key, None)


def _pop_session(session_key):
    with _session_lock:
        item = _latest_by_session.pop(session_key, None)
    if item is not None:
        pop_pending("openclaw", key=item.get("key"), context=session_key)
    else:
        item = pop_pending("openclaw", context=session_key)
    if item and time.time() - float(item.get("ts", time.time())) <= MAX_PENDING_AGE:
        return item
    return None


def _rollout_context(body, prompt, session_key):
    ctx = body.get("context")
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, dict):
        return json.dumps(ctx)
    if isinstance(ctx, list):
        return json.dumps({"messages": ctx})
    return json.dumps({"messages": [{"role": "user", "content": prompt}], "session": session_key, "channel": body.get("channel", "openclaw")})


def _run_command(cmd, session_key):
    _prune_sessions()
    cfg = _cfg()
    conn = db.connect()
    try:
        if cmd in ("good", "bad"):
            item = _pop_session(session_key)
            if item:
                rating = 1 if cmd == "good" else -1
                db.add_feedback(
                    conn, item.get("model") or cfg["model"], item["prompt"], item["response"], rating,
                    context=session_key, source=item.get("channel") or item.get("source", "openclaw"),
                    event_id=item.get("key"), rollout_context=item.get("rollout_context"),
                )
            else:
                return {"ok": False, "message": "No captured response to rate yet."}
            maybe_train(conn, cfg)
            return {"ok": True, "message": f"rated: {cmd}"}
        if cmd == "undo":
            row = db.remove_last(conn, context=session_key)
            return {"ok": bool(row), "message": "removed last rating" if row else "nothing to undo"}
        if cmd == "train":
            queue_training()
            return {"ok": True, "message": "training queued"}
        if cmd == "status":
            return {"ok": True, "message": _status_payload(conn)}
        if cmd == "rollback":
            from reinforceclaw.cli import rollback_adapter
            prev, error = rollback_adapter(conn, cfg)
            if error:
                return {"ok": False, "message": error}
            return {"ok": bool(prev), "message": f"rolled back to v{prev['version']}" if prev else "no previous adapter"}
        if cmd == "reset":
            conn.close()
            conn = None
            from reinforceclaw.cli import reset_state
            try:
                reset_state()
                with _session_lock:
                    _latest_by_session.clear()
            except RuntimeError as exc:
                return {"ok": False, "message": str(exc)}
            return {"ok": True, "message": "reset complete"}
        if cmd in ("on", "off"):
            cfg["panel_enabled"] = cmd == "on"
            from reinforceclaw.cli import save_config
            save_config(cfg)
            return {"ok": True, "message": f"panel {cmd}"}
        return {"ok": False, "message": "unknown command"}
    finally:
        if conn is not None:
            conn.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.request.settimeout(10)

    def do_POST(self):
        if not _authorized(self.headers):
            _send_error(self, 403); return
        origin = self.headers.get("Origin")
        if origin and not _local_origin(origin):
            _send_error(self, 403); return
        length = self.headers.get("Content-Length")
        if not length or not length.isdigit():
            _send_error(self, 411); return
        size = int(length)
        if size <= 0 or size > MAX_BODY:
            _send_error(self, 400); return

        if self.path == "/feedback/status":
            _status_response(self); return

        try:
            raw = self.rfile.read(size)
            if len(raw) != size:
                _send_error(self, 400); return
            body = json.loads(raw)
            if not isinstance(body, dict):
                _send_error(self, 400); return
        except (ValueError, UnicodeDecodeError):
            _send_error(self, 400); return

        if self.path == "/feedback/capture":
            _prune_sessions()
            cfg = _cfg()
            sk = _session_key(body)
            if not sk:
                _send_json(self, {"ok": False, "message": "missing sessionKey"}, 400)
                return
            response = str(body.get("response") or "").strip()
            if not response:
                _send_json(self, {"ok": False, "message": "empty response"}, 400)
                return
            prompt = str(body.get("prompt") or "(openclaw)")
            rollout = _rollout_context(body, prompt, sk)
            if cfg.get("model"):
                # cap session map at 1000
                key = save_pending(
                    "openclaw", cfg["model"], prompt, response,
                    context=sk, channel=body.get("channel", "openclaw"), rollout_context=rollout,
                )
                with _session_lock:
                    if len(_latest_by_session) >= MAX_SESSIONS:
                        _latest_by_session.popitem(last=False)
                    _latest_by_session[sk] = {
                        "key": key,
                        "prompt": prompt,
                        "response": response,
                        "channel": body.get("channel", "openclaw"),
                        "model": cfg["model"],
                        "rollout_context": rollout,
                        "ts": time.time(),
                    }
            _send_json(self, {"ok": True})
            return

        elif self.path == "/feedback/rate":
            _prune_sessions()
            sk = _session_key(body)
            if not sk:
                _send_json(self, {"ok": False, "message": "missing sessionKey"}, 400)
                return
            rating = body.get("rating", 0)
            if isinstance(rating, bool) or not isinstance(rating, int) or rating not in (-1, 0, 1):
                _send_error(self, 400); return
            if rating == 0:
                with _session_lock:
                    _latest_by_session.pop(sk, None)
                pop_pending("openclaw", context=sk)
                _send_json(self, {"ok": True})
                return
            item = _pop_session(sk)
            if not item:
                _send_json(self, {"ok": False, "message": "No captured response to rate yet."}, 400)
                return
            conn = db.connect()
            try:
                cfg = _cfg()
                db.add_feedback(
                    conn, item.get("model") or cfg["model"], item["prompt"], item["response"], rating,
                    context=sk, source=item.get("channel") or item.get("source", "openclaw"),
                    event_id=item.get("key"), rollout_context=item.get("rollout_context"),
                )
                maybe_train(conn, cfg)
            finally:
                conn.close()
            _send_json(self, {"ok": True})
            return
        elif self.path == "/feedback/command":
            sk = _session_key(body)
            if not sk:
                _send_json(self, {"ok": False, "message": "missing sessionKey"}, 400)
                return
            result = _run_command(str(body.get("command", "")).lower(), sk)
            _send_json(self, result, 200 if result.get("ok") else 400)
            return

        _send_json(self, {"ok": False, "message": "unknown path"}, 404)

    def do_GET(self):
        if not _authorized(self.headers):
            _send_error(self, 403); return
        if self.path == "/feedback/status":
            _status_response(self)
        else:
            _send_error(self, 404)

    def log_message(self, *a): pass  # silence per-request logging


def run(port=8420):
    if not _shared_secret():
        raise RuntimeError("reinforceclaw openclaw backend requires openclaw_secret")
    print(f"reinforceclaw openclaw backend on :{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    run()
