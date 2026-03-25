"""Shared helpers for all hooks. Config, stdin, training, rating."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from nudge import db


def load_config():
    p = Path.home() / ".nudge" / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def read_stdin():
    try:
        return json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return {}


def queue_training():
    """Spawn training in background — never block the hook."""
    subprocess.Popen(
        [sys.executable, "-m", "nudge.cli", "train"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def maybe_train(conn, config):
    """Train now if schedule=auto and batch ready. Otherwise the scheduler handles it."""
    schedule = config.get("train_schedule", "03:00")
    if schedule != "auto":
        return  # scheduled training handles it, don't train in the hook
    if db.count_trainable_untrained(conn) >= config.get("batch_min", 16):
        queue_training()
