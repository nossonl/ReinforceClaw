"""Training scheduler. Uses launchd on mac, systemd on linux."""
# default: train at 3am if batch is ready
# if machine was off at 3am, trains on next wake/boot
# configurable: "03:00", "manual", "auto" (immediate when batch ready)

import json
import platform
import subprocess
import sys
from pathlib import Path

PLIST_PATH = Path.home() / "Library/LaunchAgents/com.nudge.train.plist"
SYSTEMD_PATH = Path.home() / ".config/systemd/user/nudge-train.timer"
SYSTEMD_SERVICE = Path.home() / ".config/systemd/user/nudge-train.service"


def install(schedule="03:00"):
    """Install system scheduler. Returns True on success."""
    if schedule == "manual":
        uninstall()
        return True
    if schedule == "auto":
        uninstall()  # no system scheduler needed, hooks handle it
        return True

    hour, minute = _parse_time(schedule)
    if platform.system() == "Darwin":
        return _install_launchd(hour, minute)
    else:
        return _install_systemd(hour, minute)


def uninstall():
    """Remove system scheduler."""
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
        PLIST_PATH.unlink()
    if SYSTEMD_PATH.exists():
        subprocess.run(["systemctl", "--user", "disable", "nudge-train.timer"], capture_output=True)
        SYSTEMD_PATH.unlink()
        SYSTEMD_SERVICE.unlink(missing_ok=True)


def _parse_time(s):
    # "03:00" → (3, 0), "14:30" → (14, 30)
    parts = s.split(":")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def _install_launchd(hour, minute):
    # launchd runs missed jobs on next wake — exactly what we want
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nudge.train</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>nudge.cli</string>
        <string>train</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{Path.home() / '.nudge/train.log'}</string>
    <key>StandardErrorPath</key>
    <string>{Path.home() / '.nudge/train.log'}</string>
</dict>
</plist>"""
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(plist)
    r = subprocess.run(["launchctl", "load", str(PLIST_PATH)], capture_output=True)
    return r.returncode == 0


def _install_systemd(hour, minute):
    SYSTEMD_PATH.parent.mkdir(parents=True, exist_ok=True)
    # service
    SYSTEMD_SERVICE.write_text(f"""[Unit]
Description=Nudge RL training

[Service]
ExecStart={sys.executable} -m nudge.cli train
""")
    # timer — Persistent=true means it catches up missed runs
    SYSTEMD_PATH.write_text(f"""[Unit]
Description=Nudge daily training

[Timer]
OnCalendar=*-*-* {hour:02d}:{minute:02d}:00
Persistent=true

[Install]
WantedBy=timers.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    r = subprocess.run(["systemctl", "--user", "enable", "--now", "nudge-train.timer"], capture_output=True)
    return r.returncode == 0
