#!/usr/bin/env python3
"""Claude Code hooks. Stop hook + /rl command interceptor."""
# stop hook: fires after each assistant turn. records turn, shows panel, queues training.
# prompt hook: intercepts /rl commands before claude sees them.
#
# CRITICAL: hooks run as subprocesses. stdin is piped JSON, NOT a terminal.
# stdout goes back to claude code as the hook response.
# DO NOT print anything to stdout except the final JSON response.
# DO NOT run training inline — it takes minutes and freezes claude code.

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nudge import db


def _load_config():
    p = Path.home() / ".nudge" / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _read_stdin():
    try:
        return json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return {}


def _queue_training():
    """Spawn training in background — never block the hook."""
    subprocess.Popen(
        [sys.executable, "-m", "nudge.cli", "train"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,  # fully detached from hook process
    )


def _trainable_untrained(conn):
    return db.count_trainable_untrained(conn)


def handle_stop():
    """Stop hook — fires after each assistant turn."""
    data = _read_stdin()
    config = _load_config()
    if not config.get("model"):
        return

    # extract last assistant message
    last_msg = data.get("last_assistant_message", "")
    if not last_msg:
        msgs = data.get("messages", [])
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                last_msg = m.get("content", "")
                break
    if not last_msg:
        return

    # extract last user prompt
    prompt = ""
    for m in reversed(data.get("messages", [])):
        if m.get("role") == "user":
            prompt = m.get("content", "")
            break

    conn = db.connect()
    current_id = db.add_feedback(conn, config["model"], prompt, last_msg, 0, source="hook")

    if config.get("panel_enabled", True):
        from nudge.feedback import collect_rating
        rating = collect_rating(timeout_s=10)  # short timeout — don't freeze claude
        if isinstance(rating, int):
            db.update_feedback_rating(conn, current_id, rating)
            if _trainable_untrained(conn) >= config.get("batch_min", 16):
                _queue_training()

    conn.close()
    # stop hooks: stdout is ignored, just exit clean


def handle_prompt():
    """UserPromptSubmit hook — intercept /rl and /good /bad commands."""
    data = _read_stdin()

    # claude code sends prompt as object: {"content": "...", "type": "user"}
    prompt_obj = data.get("prompt", {})
    if isinstance(prompt_obj, dict):
        prompt = prompt_obj.get("content", "").strip()
    else:
        prompt = str(prompt_obj).strip()

    cmds = {
        "/rl good": "good", "/rl bad": "bad", "/rl undo": "undo",
        "/rl train": "train", "/rl status": "status",
        "/rl rollback": "rollback", "/rl reset": "reset",
        "/rl on": "on", "/rl off": "off",
        "/good": "good", "/bad": "bad",
    }

    cmd = None
    for prefix, action in cmds.items():
        if prompt.lower().startswith(prefix):
            cmd = action
            break

    if cmd is None:
        sys.exit(0)  # not ours — passthrough

    config = _load_config()
    conn = db.connect()

    if cmd in ("good", "bad"):
        if not config.get("model"):
            sys.stderr.write("Run nudge init first.\n")
            conn.close()
            print(json.dumps({"result": "block", "reason": "Handled by nudge: not initialized"}))
            return
        rating = 1 if cmd == "good" else -1
        pending = db.latest_pending(conn, source="hook")
        if pending:
            db.update_feedback_rating(conn, pending["id"], rating)
        else:
            # try to get last message from hook data
            last_msg = data.get("last_assistant_message", "(from chat)")
            db.add_feedback(conn, config.get("model", ""), "(from chat)", last_msg, rating, source="hook")
        label = "\033[32mgood\033[0m" if rating == 1 else "\033[31mbad\033[0m"
        sys.stderr.write(f"Rated: {label}\n")
        if _trainable_untrained(conn) >= config.get("batch_min", 16):
            _queue_training()
    elif cmd == "undo":
        r = db.remove_last(conn)
        sys.stderr.write(f"\033[33mRemoved last rating\033[0m\n" if r else "Nothing to undo.\n")
    elif cmd == "train":
        _queue_training()
        sys.stderr.write("Training started in background.\n")
    elif cmd == "status":
        counts = db.count(conn)
        ema, _ = db.get_ema(conn)
        a = db.latest_adapter(conn)
        sys.stderr.write(f"Adapter: {'v'+str(a['version']) if a else 'none'} | "
                         f"Ratings: {counts['total']} ({counts['good']}+ {counts['bad']}-) | "
                         f"Untrained: {_trainable_untrained(conn)} | EMA: {ema:.3f}\n")
    elif cmd == "rollback":
        prev = db.rollback(conn)
        if prev:
            from nudge import trainer
            trainer.hot_swap(config.get("server", "ollama"), prev["path"], config.get("model", ""))
        sys.stderr.write(f"\033[32mRolled back to v{prev['version']}\033[0m\n" if prev
                         else "No previous adapter.\n")
    elif cmd == "reset":
        from nudge.cli import reset_state
        conn.close()
        reset_state()
        sys.stderr.write("\033[32mReset complete.\033[0m\n")
        print(json.dumps({"result": "block", "reason": f"Handled by nudge: /rl {cmd}"}))
        return
    elif cmd == "on":
        config["panel_enabled"] = True
        from nudge.cli import save_config
        save_config(config)
        sys.stderr.write("\033[32mPanel on.\033[0m\n")
    elif cmd == "off":
        config["panel_enabled"] = False
        from nudge.cli import save_config
        save_config(config)
        sys.stderr.write("\033[33mPanel off.\033[0m\n")

    conn.close()

    # block the prompt from reaching claude
    print(json.dumps({"result": "block", "reason": f"Handled by nudge: /rl {cmd}"}))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    {"stop": handle_stop, "prompt": handle_prompt}.get(sys.argv[1], lambda: sys.exit(1))()
