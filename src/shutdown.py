"""Graceful shutdown for the stdio MCP server.

Claude Code was observed sending SIGINT (which the server ignored) then escalating
to SIGTERM, whose default action hard-kills the process before any cleanup — so the
sandbox Docker container was never torn down. We install idempotent SIGTERM/SIGINT
handlers that close the ReplSession container, log a shutdown record, flush, and exit
promptly; an atexit hook covers the clean stdin-EOF path (mcp.run returns, no signal).
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import time
from typing import Callable

from .logsetup import log_event

_shutting_down = False


def _signame(signum) -> str:
    if signum is None:
        return "atexit"
    try:
        return signal.Signals(signum).name
    except Exception:
        return str(signum)


def install_shutdown_hooks(logger: logging.Logger,
                           close_repl: Callable[[], None]) -> None:
    """Register SIGTERM/SIGINT + atexit teardown: close the sandbox container, log a
    shutdown record, and exit. Idempotent — a second signal / atexit won't double-run.
    Must be called on the main thread (signal.signal requirement)."""

    def _teardown(signum=None) -> None:
        global _shutting_down
        if _shutting_down:
            return
        _shutting_down = True
        start = time.monotonic()
        err = None
        try:
            close_repl()
        except Exception as exc:
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
        log_event(logger, "shutdown", signal=_signame(signum),
                  repl_closed=(err is None),
                  dur_ms=round((time.monotonic() - start) * 1000), err=err)
        logging.shutdown()

    def _handler(signum, frame) -> None:
        # Do full cleanup FIRST, then exit hard — avoids SIGINT being swallowed by the
        # anyio loop and re-triggering Claude Code's SIGTERM escalation.
        _teardown(signum)
        os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not on the main thread / unsupported platform — atexit still covers EOF

    atexit.register(_teardown)  # clean stdin-EOF path (no signal delivered)
