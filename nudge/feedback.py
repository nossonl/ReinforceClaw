"""Terminal feedback panel. Opens /dev/tty directly so it works inside hook subprocesses."""
# hooks get stdin piped from the parent process (JSON), so sys.stdin is NOT a terminal.
# we bypass that entirely by opening /dev/tty — the real terminal — for keypress reads.
# NO TIMERS. panel stays until you press something or start typing. your call.

import os
import sys
import select
from typing import Optional, Union

_KEYS = {"1": 1, "2": -1, "3": None}  # good, bad, skip


def _open_tty():
    try:
        return os.open("/dev/tty", os.O_RDONLY)
    except OSError:
        return None


def _raw_read(fd: int) -> Optional[str]:
    """One keypress from the real terminal. Waits forever."""
    import termios, tty
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # blocks until a key is pressed. no timeout. take your time.
        ch = os.read(fd, 3).decode("utf-8", errors="ignore")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


PANEL = (
    "\n\033[36m╭─ Rate this response ───────────╮\033[0m\n"
    "\033[36m│\033[0m  Rate good AND bad for best    \033[36m│\033[0m\n"
    "\033[36m│\033[0m  results                       \033[36m│\033[0m\n"
    "\033[36m│\033[0m                                \033[36m│\033[0m\n"
    "\033[36m│\033[0m  \033[32m[1] Good\033[0m  \033[31m[2] Bad\033[0m  \033[90m[3] Skip\033[0m  \033[36m│\033[0m\n"
    "\033[36m│\033[0m  \033[90m[↑] Rate previous response\033[0m   \033[36m│\033[0m\n"
    "\033[36m╰────────────────────────────────╯\033[0m\n"
)

def _clear_panel():
    sys.stderr.write("\033[8A\033[J")
    sys.stderr.flush()


def collect_rating() -> Optional[int]:
    """Show panel, wait for a key. No timer. Returns +1, -1, or None."""
    fd = _open_tty()
    if fd is None:
        return None

    sys.stderr.write(PANEL)
    sys.stderr.flush()
    try:
        key = _raw_read(fd)
        _clear_panel()
        if key is None or key.startswith("\x1b"):
            return None  # escape or arrow = skip
        return _KEYS.get(key)  # 1=good, 2=bad, 3/anything else=skip
    finally:
        os.close(fd)
