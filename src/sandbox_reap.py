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
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:  # exists, owned by another user — must precede OSError
        return True
    except OSError:
        # POSIX raises ProcessLookupError (an OSError subclass) for ESRCH. Either way
        # the process is unreachable — treat it as gone.
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    """Ask whether a pid exists WITHOUT signalling it.

    os.kill(pid, 0) is not a probe on Windows. signal.CTRL_C_EVENT is 0, so CPython
    routes signal 0 to GenerateConsoleCtrlEvent, which delivers a Ctrl+C to the console
    process group ``pid`` rather than asking whether that pid exists. reap_orphans calls
    this once per owner marker at startup, so on Windows the sweep was firing console
    interrupts at whatever group ids it read out of stale markers — and misreporting
    liveness besides. It surfaced as a KeyboardInterrupt landing in an unrelated test
    thirty tests after the call that queued it.

    OpenProcess + WaitForSingleObject instead: a handle that has not signalled belongs to
    a process that has not exited. GetExitCodeProcess would be shorter and wrong — it
    cannot tell a live process from one that exited with 259 (STILL_ACTIVE).
    """
    import ctypes
    from ctypes import wintypes

    SYNCHRONIZE = 0x0010_0000                    # WaitForSingleObject needs this, not query
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    ERROR_ACCESS_DENIED = 5
    WAIT_TIMEOUT = 0x102

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k32.WaitForSingleObject.restype = wintypes.DWORD
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # Access denied is the Windows spelling of the PermissionError branch above: the
        # process is there, it just is not ours. Anything else means no such pid.
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        return k32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        k32.CloseHandle(handle)


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
