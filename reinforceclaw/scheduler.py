"""Training scheduler. Uses launchd on mac, systemd on linux."""
# default: "auto" queues background training when a batch is ready.
# scheduled times like "03:00" are optional for users who prefer a clock window.

import platform
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from . import db

PLIST_PATH = Path.home() / "Library/LaunchAgents/com.reinforceclaw.train.plist"
SYSTEMD_PATH = Path.home() / ".config/systemd/user/reinforceclaw-train.timer"
SYSTEMD_SERVICE = Path.home() / ".config/systemd/user/reinforceclaw-train.service"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_LOG = Path.home() / ".reinforceclaw" / "train.log"
DEFAULT_WINDOW_MINUTES = 180
LAST_ERROR = ""


def _attempt_times(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    hour, minute = _parse_time(schedule)
    start = datetime(2000, 1, 1, hour, minute)
    tries = max(1, int((window_minutes - 1) // 60) + 1)
    return [((start + timedelta(hours=i)).hour, (start + timedelta(hours=i)).minute) for i in range(tries)]


def install(schedule="03:00", window_minutes: int = DEFAULT_WINDOW_MINUTES):
    """Install system scheduler. Returns True on success."""
    db.secure_private_dir(TRAIN_LOG.parent)
    if schedule == "manual":
        return uninstall()
    if schedule == "auto":
        return uninstall()  # no system scheduler needed, hooks handle it

    attempt_times = _attempt_times(schedule, window_minutes)
    if platform.system() == "Darwin":
        return _install_launchd(attempt_times)
    else:
        return _install_systemd(attempt_times)


def uninstall():
    """Remove system scheduler."""
    ok = True
    if PLIST_PATH.exists():
        ok = _run_ok(["launchctl", "unload", str(PLIST_PATH)]) and ok
        PLIST_PATH.unlink()
    if SYSTEMD_PATH.exists():
        ok = _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"]) and ok
        ok = _run_ok(["systemctl", "--user", "stop", "reinforceclaw-train.service"]) and ok
        SYSTEMD_PATH.unlink()
        SYSTEMD_SERVICE.unlink(missing_ok=True)
        ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and ok
    return ok


def _run_ok(cmd):
    global LAST_ERROR
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        LAST_ERROR = (result.stderr or result.stdout or f"{cmd[0]} exited {result.returncode}").strip()
        return False
    except OSError as exc:
        LAST_ERROR = str(exc)
        return False


def _write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _snapshot(*paths):
    return {path: path.read_text() if path.exists() else None for path in paths}


def _restore(snapshot):
    for path, text in snapshot.items():
        if text is None:
            path.unlink(missing_ok=True)
        else:
            _write_text(path, text)


def _systemd_arg(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _parse_time(s):
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("time must be HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time must be HH:MM")
    return hour, minute


def _install_launchd(attempt_times):
    intervals = "\n".join(
        f"""        <dict>
            <key>Hour</key>
            <integer>{hour}</integer>
            <key>Minute</key>
            <integer>{minute}</integer>
        </dict>"""
        for hour, minute in attempt_times
    )
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.reinforceclaw.train</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>reinforceclaw.cli</string>
        <string>train</string>
        <string>--background</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{PROJECT_ROOT}</string>
    </dict>
    <key>StartCalendarInterval</key>
    <array>
{intervals}
    </array>
    <key>StandardOutPath</key>
    <string>{TRAIN_LOG}</string>
    <key>StandardErrorPath</key>
    <string>{TRAIN_LOG}</string>
</dict>
</plist>"""
    db.secure_private_dir(PLIST_PATH.parent)
    before = _snapshot(PLIST_PATH)
    if PLIST_PATH.exists() and not _run_ok(["launchctl", "unload", str(PLIST_PATH)]):
        return False
    _write_text(PLIST_PATH, plist)
    if _run_ok(["launchctl", "load", str(PLIST_PATH)]):
        return True
    _restore(before)
    if before[PLIST_PATH] is not None:
        _run_ok(["launchctl", "load", str(PLIST_PATH)])
    return False


def _install_systemd(attempt_times):
    db.secure_private_dir(SYSTEMD_PATH.parent)
    before = _snapshot(SYSTEMD_SERVICE, SYSTEMD_PATH)
    if SYSTEMD_PATH.exists() and not _run_ok(["systemctl", "--user", "disable", "--now", "reinforceclaw-train.timer"]):
        return False
    # service
    _write_text(SYSTEMD_SERVICE, f"""[Unit]
Description=ReinforceClaw RL training

[Service]
ExecStart={_systemd_arg(sys.executable)} -m reinforceclaw.cli train --background
WorkingDirectory={_systemd_arg(PROJECT_ROOT)}
Environment="PYTHONPATH={PROJECT_ROOT}"
StandardOutput=append:{TRAIN_LOG}
StandardError=append:{TRAIN_LOG}
""")
    calendar = "\n".join(
        f"OnCalendar=*-*-* {hour:02d}:{minute:02d}:00"
        for hour, minute in attempt_times
    )
    _write_text(SYSTEMD_PATH, f"""[Unit]
Description=ReinforceClaw daily training

[Timer]
{calendar}
Persistent=true

[Install]
WantedBy=timers.target
""")
    ok = _run_ok(["systemctl", "--user", "daemon-reload"]) and _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-train.timer"])
    if ok:
        return True
    _restore(before)
    _run_ok(["systemctl", "--user", "daemon-reload"])
    if before[SYSTEMD_PATH] is not None:
        _run_ok(["systemctl", "--user", "enable", "--now", "reinforceclaw-train.timer"])
    return False
