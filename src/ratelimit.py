"""Auth-aware retry + a shared throttle for a rate-limited vendor (the Anthropic API).

Standard pattern for "external vendor enforces rate limits — sacrifice speed for
stability": every API call passes through a process-wide throttle (bounded
concurrency + minimum spacing), and a 429 triggers retry-with-backoff.

`retry_and_queue_retries` (sync) and `aretry_and_queue_retries` (async) wrap a
call so that:

  * Every attempt goes through the shared throttle — at most
    ``throttle_max_concurrency`` (default 3) calls in flight, each dispatched
    >= ``throttle_min_interval_s`` (default 1s) after the previous. A single call
    sails through immediately; only under load (large batches) do calls queue
    behind the cap and the spacing, so they fan out by 3 instead of bursting.
  * On HTTP 429 the call waits then retries — OAuth (tight subscription limits)
    waits 5s/10s/15s; API key (higher limits) waits 1s/2s/4s — honoring the
    server's ``Retry-After`` header when it asks for longer — and fails after the
    waits are exhausted. Non-429 errors are never retried.

The async variant applies the same retry policy (without the sync throttle, since
the hot paths here are synchronous).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from contextlib import contextmanager

import anthropic

from .config import load_config
from .logsetup import log_event

_CFG = load_config()
_LOG = logging.getLogger("rlm-mcp")


def _is_rate_limit(exc: BaseException) -> bool:
    # Retryable when: the SDK raised RateLimitError / a 429, OR the CLI transport
    # raised an error flagged is_rate_limit (transport.CliRateLimitError).
    return (
        isinstance(exc, anthropic.RateLimitError)
        or getattr(exc, "status_code", None) == 429
        or bool(getattr(exc, "is_rate_limit", False))
    )


def _retry_after_seconds(exc: BaseException) -> float:
    try:
        ra = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(ra) if ra else 0.0
    except Exception:
        return 0.0


def _waits() -> list[float]:
    """Backoff schedule for the active transport (claude CLI gets the longer waits)."""
    from .auth import resolve_auth_mode  # lazy import to avoid a cycle
    sched = _CFG.oauth_retry_waits if resolve_auth_mode(_CFG) == "oauth" else _CFG.apikey_retry_waits
    return list(sched)


class _Throttle:
    """Bounded concurrency + minimum inter-dispatch spacing (a global rate gate)."""

    def __init__(self, max_concurrency: int, min_interval: float):
        self._sem = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._min = min_interval
        self._last = 0.0

    @contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            with self._lock:
                gap = self._min - (time.monotonic() - self._last)
                if gap > 0:
                    time.sleep(gap)
                self._last = time.monotonic()
            yield
        finally:
            self._sem.release()


_THROTTLE = _Throttle(_CFG.throttle_max_concurrency, _CFG.throttle_min_interval_s)


def retry_and_queue_retries(fn):
    """Sync: route ``fn`` through the shared throttle; retry 429s with auth-aware backoff."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        waits = _waits()
        attempt = 0
        while True:
            with _THROTTLE.slot():
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if not _is_rate_limit(exc) or attempt >= len(waits):
                        raise
                    delay = max(waits[attempt], _retry_after_seconds(exc))
            log_event(_LOG, "retry", attempt=attempt + 1, delay_s=round(delay, 1), reason="rate_limit")
            time.sleep(delay)  # slot released — let queued calls proceed while we wait
            attempt += 1

    return wrapper


def aretry_and_queue_retries(fn):
    """Async: same 429 retry policy (throttle is applied on the sync hot paths)."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        waits = _waits()
        attempt = 0
        while True:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit(exc) or attempt >= len(waits):
                    raise
                delay = max(waits[attempt], _retry_after_seconds(exc))
            log_event(_LOG, "retry", attempt=attempt + 1, delay_s=round(delay, 1),
                      reason="rate_limit", mode="async")
            await asyncio.sleep(delay)
            attempt += 1

    return wrapper
