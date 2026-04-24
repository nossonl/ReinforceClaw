"""Shared helpers for all hooks. Config, stdin, training, rating."""

import json
import os
import fcntl
import hashlib
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from reinforceclaw import db

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
TRAIN_LOG_PATH = Path.home() / ".reinforceclaw" / "train.log"
TRAIN_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.lock"
TRAIN_QUEUE_LOCK_PATH = Path.home() / ".reinforceclaw" / "train.queue.lock"
TRAIN_RETRY_PATH = Path.home() / ".reinforceclaw" / "train.retry"
PENDING_DIR = Path.home() / ".reinforceclaw" / "pending"
RESET_MARK_PATH = Path.home() / ".reinforceclaw" / "reset.marker"
_CMD_NAMES = ("good", "bad", "undo", "train", "status", "rollback", "reset", "on", "off")
COMMANDS = {f"/{prefix} {cmd}": cmd for prefix in ("rl", "rc", "reinforceclaw") for cmd in _CMD_NAMES}
COMMANDS.update({"/good": "good", "/bad": "bad"})


def load_config():
    from reinforceclaw.cli import load_config as cli_load_config
    return cli_load_config()


def read_stdin():
    try:
        return json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError):
        return {}


def block(reason):
    print(json.dumps({"decision": "block", "reason": reason}))


def _content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return ""


def _recent_transcript_rows(transcript_path, limit=64, max_bytes=262144):
    try:
        path = Path(transcript_path).expanduser()
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            start = max(0, size - max_bytes)
            fh.seek(start)
            text = fh.read().decode("utf-8", "replace")
        lines = text.splitlines()
        if start and lines:
            lines = lines[1:]
        rows = []
        for line in lines[-limit:]:
            if line.strip():
                row = json.loads(line)
                rows.append(row.get("message") if isinstance(row, dict) and isinstance(row.get("message"), dict) else row)
        return rows
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []


def last_msg_from(data, role):
    rows = data.get("messages")
    if isinstance(rows, list):
        for msg in reversed(rows):
            if isinstance(msg, dict) and msg.get("role") == role:
                return _content_text(msg.get("content", ""))

    transcript_path = data.get("transcript_path")
    if not transcript_path:
        return ""
    for msg in reversed(_recent_transcript_rows(transcript_path)):
        if isinstance(msg, dict) and msg.get("role") == role:
            return _content_text(msg.get("content", ""))
    return ""


def _message_list(data, limit=12):
    rows = data.get("messages")
    if isinstance(rows, list):
        source = rows[-limit:]
    else:
        source = []
        transcript_path = data.get("transcript_path")
        if transcript_path:
            source = list(deque(_recent_transcript_rows(transcript_path), maxlen=limit))
    out = []
    for msg in source:
        if not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}:
            continue
        text = _content_text(msg.get("content", "")).strip()
        if text:
            out.append({"role": msg["role"], "content": text})
    return out[-limit:]


def training_context(data, prompt="", response="", limit=12):
    messages = _message_list(data, limit)
    if response and messages and messages[-1]["role"] == "assistant" and messages[-1]["content"].strip() == response.strip():
        messages.pop()
    if prompt and not any(m["role"] == "user" and m["content"].strip() == prompt.strip() for m in messages[-2:]):
        messages.append({"role": "user", "content": prompt})
    payload = {"messages": messages[-limit:]}
    for key in ("cwd", "session_id", "sessionId", "conversation_id", "conversationId", "transcript_path"):
        if data.get(key):
            payload[key] = str(data[key])
    return json.dumps(payload)


def command_from_prompt(data):
    prompt_obj = data.get("prompt", {})
    prompt = prompt_obj.get("content", "") if isinstance(prompt_obj, dict) else str(prompt_obj)
    return COMMANDS.get(" ".join(str(prompt).strip().strip("`*_").lower().split()))


def handle_agent_command(source, data, write=lambda _s: None):
    cmd = command_from_prompt(data)
    if cmd is None:
        sys.exit(0)
    config, context, conn = load_config(), pending_context(data), db.connect()
    reason = f"reinforceclaw: /rl {cmd}"
    try:
        if cmd in ("good", "bad"):
            if not config.get("model"):
                reason = "reinforceclaw not initialized"
                write("Run reinforceclaw init first.\n")
                return block(reason)
            rating = 1 if cmd == "good" else -1
            pending = pop_pending(source, context=context) or pop_pending(source)
            if not pending:
                reason = "reinforceclaw: no captured response"
                write("No captured response to rate.\n")
                return block(reason)
            db.add_feedback(
                conn, pending["model"], pending["prompt"], pending["response"], rating,
                context=pending.get("context") or context, source=source,
                event_id=pending.get("key"), rollout_context=pending.get("rollout_context"),
            )
            write(f"Rated: {'good' if rating == 1 else 'bad'}\n")
            maybe_train(conn, config)
        elif cmd == "undo":
            row = (db.remove_last(conn, source=source, context=context) if context else None) or db.remove_last(conn, source=source)
            write("Removed last rating\n" if row else "Nothing to undo.\n")
        elif cmd == "train":
            queue_training()
            write("Training started in background.\n")
        elif cmd == "status":
            counts, adapter = db.count(conn), db.latest_adapter(conn)
            ema, _ = db.get_ema(conn)
            reason = (f"Adapter: {'v'+str(adapter['version']) if adapter else 'none'} | "
                      f"Ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-) | "
                      f"Untrained: {db.count_trainable_untrained(conn)} | EMA: {ema:.3f}")
            write(reason + "\n")
        elif cmd == "rollback":
            from reinforceclaw.cli import rollback_adapter
            prev, error = rollback_adapter(conn, config)
            reason = error or (f"Rolled back to v{prev['version']}" if prev else "No previous adapter.")
            write(reason + "\n")
        elif cmd == "reset":
            from reinforceclaw.cli import reset_state
            conn.close(); conn = None
            try:
                reset_state()
                reason = "reinforceclaw: /rl reset"
                write("Reset complete.\n")
            except RuntimeError as exc:
                reason = str(exc)
                write(reason + "\n")
        elif cmd in ("on", "off"):
            config["panel_enabled"] = cmd == "on"
            from reinforceclaw.cli import save_config
            save_config(config)
            write(f"Panel {cmd}.\n")
        block(reason)
    finally:
        if conn is not None:
            conn.close()


def pending_context(data):
    for key in ("session_id", "sessionId", "conversation_id", "conversationId", "transcript_path", "cwd"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _safe_name(value):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in value).strip("_") or "pending"


def _pending_prefix(source, context=None):
    safe = _safe_name(source)
    return f"{safe}-{hashlib.sha256(str(context).encode()).hexdigest()[:16]}" if context else safe


def _pending_path(source, context=None):
    return PENDING_DIR / f"{_pending_prefix(source, context)}.json"


def _pending_paths(source, context=None):
    if context:
        return [_pending_path(source, context)]
    if not PENDING_DIR.exists():
        return []
    paths = []
    for path in PENDING_DIR.glob(f"{_pending_prefix(source)}*.json"):
        try:
            paths.append((path.stat().st_mtime, path))
        except OSError:
            pass
    return [path for _, path in sorted(paths, reverse=True)]


def save_pending(source, model, prompt, response, context=None, channel=None, rollout_context=None):
    key = str(time.time_ns())
    db.secure_private_dir(PENDING_DIR.parent)
    db.secure_private_dir(PENDING_DIR)
    path = _pending_path(source, context)
    tmp = path.with_name(f".{path.name}.{key}.tmp")
    payload = {"key": key, "source": source, "model": model, "prompt": prompt, "response": response, "ts": time.time()}
    if context:
        payload["context"] = str(context)
    if channel:
        payload["channel"] = str(channel)
    if rollout_context:
        payload["rollout_context"] = str(rollout_context)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return key


def pop_pending(source, key=None, context=None):
    for path in _pending_paths(source, context):
        claimed = path.with_name(f".{path.name}.{os.getpid()}.claim")
        try:
            path.replace(claimed)
            payload = json.loads(claimed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            claimed.unlink(missing_ok=True)
            continue
        if key and payload.get("key") != key:
            try:
                claimed.replace(path) if not path.exists() else claimed.unlink()
            except OSError:
                pass
            continue
        claimed.unlink(missing_ok=True)
        return payload
    return None


def _train_lock_held() -> bool:
    db.secure_private_dir(TRAIN_LOCK_PATH.parent)
    fd = os.open(TRAIN_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _retry_due():
    try:
        return float(TRAIN_RETRY_PATH.read_text())
    except Exception:
        return 0.0


def _write_retry_due(due):
    db.secure_private_dir(TRAIN_RETRY_PATH.parent)
    fd = os.open(TRAIN_RETRY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(str(due))


def _base_env():
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    env = dict(os.environ)
    for key in tuple(env):
        if key.endswith("_API_KEY") or key in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}:
            env.pop(key, None)
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT_STR}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else PROJECT_ROOT_STR
    )
    return env


def _spawn_train(argv):
    db.secure_private_dir(TRAIN_LOG_PATH.parent)
    fd = os.open(TRAIN_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as log:
        subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            env=_base_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def queue_training(delay_seconds=0):
    """Spawn training in background — never block the hook."""
    db.secure_private_dir(TRAIN_QUEUE_LOCK_PATH.parent)
    fd = os.open(TRAIN_QUEUE_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if _train_lock_held():
            return
        now = time.time()
        if _retry_due() > now:
            return
        _write_retry_due(now + (delay_seconds if delay_seconds > 0 else 30))
        try:
            argv = ([sys.executable, "-m", "reinforceclaw.hooks._common", "retry", str(delay_seconds)]
                    if delay_seconds > 0 else [sys.executable, "-m", "reinforceclaw.cli", "train", "--background"])
            _spawn_train(argv)
        except Exception:
            TRAIN_RETRY_PATH.unlink(missing_ok=True)
            raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def maybe_train(conn, config):
    """Train now if schedule=auto and batch ready. Otherwise the scheduler handles it."""
    schedule = config.get("train_schedule", "auto")
    if schedule != "auto":
        return  # scheduled training handles it, don't train in the hook
    if db.count_trainable_untrained(conn) >= config.get("batch_min", 32):
        queue_training()


def _retry_after(delay_seconds: float):
    time.sleep(max(0.0, delay_seconds))
    TRAIN_RETRY_PATH.unlink(missing_ok=True)
    queue_training()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "retry":
        _retry_after(float(sys.argv[2]))
    elif len(sys.argv) > 1:
        sys.stderr.write("usage: python -m reinforceclaw.hooks._common retry <seconds>\n")
        sys.exit(2)
