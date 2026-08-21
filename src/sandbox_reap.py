"""Reap Docker sandboxes abandoned by a dead server.

SIGKILL and hard client teardowns bypass every exit hook, so a container can
outlive the process that started it. ``atexit`` structurally cannot catch that;
a sweep at the next startup can.

This is host-side housekeeping over the engine's workspace layout, which is why
it lives here and not in ``rlm/``. The exec-protocol fixes that used to share
this module were folded into ``rlm/environments/docker_repl.py`` once the engine
became vendored source — there is nothing left to monkey-patch.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

_LOG = logging.getLogger("rlm-mcp")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:  # exists, owned by another user — must precede OSError
        return True
    except OSError:
        # POSIX raises ProcessLookupError (an OSError subclass) for ESRCH. Windows has no
        # ESRCH mapping: os.kill(pid, 0) goes through OpenProcess, so an unknown or
        # out-of-range pid surfaces as a bare OSError (WinError 87). Catching only
        # ProcessLookupError let that escape and made reap_orphans raise on Windows
        # instead of sweeping. Either way the process is unreachable — treat it as gone.
        return False
    return True


def workspace_root() -> str:
    """Where the vendored env puts sandbox dirs — mirrors docker_repl.setup()."""
    return os.environ.get(
        "RLM_DOCKER_WORKSPACE_DIR", os.path.join(os.getcwd(), ".rlm_workspace"))


def reap_orphans(root: str) -> list[str]:
    """Remove sandboxes whose owning server is gone; return what was reaped.

    SIGKILL and hard client teardowns bypass every exit hook, so a container
    outlives its server: measured 11 of 19 processes exiting with no shutdown
    record, containers alive for hours, workspace dirs left behind. A sweep at
    startup catches what atexit structurally cannot.
    """
    reaped: list[str] = []
    for d in sorted(Path(root).glob("docker_repl_*")):
        marker = d / "owner"
        pid: int | None = None
        container = ""
        if marker.exists():
            try:
                fields = marker.read_text(encoding="utf-8").split()
                pid, container = int(fields[0]), (fields[1] if len(fields) > 1 else "")
            except (ValueError, IndexError, OSError):
                pid = None
        elif time.time() - d.stat().st_mtime < 3600:
            continue  # pre-patch or mid-setup: too young to call an orphan
        if pid is not None and _pid_alive(pid):
            continue
        if container:
            subprocess.run(["docker", "rm", "-f", container],
                           capture_output=True, timeout=60)
        shutil.rmtree(d, ignore_errors=True)
        reaped.append(d.name)
    if reaped:
        _LOG.warning("evt=reaped_orphans count=%d dirs=%s", len(reaped), ",".join(reaped))
    return reaped


def reap_stale_sandboxes() -> None:
    """Sweep once, at startup. A janitor must never break the thing it tidies."""
    try:
        reap_orphans(workspace_root())
    except Exception as exc:
        _LOG.warning("evt=reap_failed err=%r", exc)
