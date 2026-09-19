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

from . import failures
from .config import load_config
from .logsetup import log_event

#: Module-level ON PURPOSE, and the one place in this package where that is the right
#: answer. This module owns two things, and neither belongs to a caller:
#:
#:  * _THROTTLE is a PROCESS resource. Its whole job is "at most N calls in flight across
#:    this process"; handing each caller its own would multiply the vendor's limit by the
#:    number of callers, which is the opposite of a throttle. It must be shared.
#:  * _CFG supplies only the backoff SCHEDULE (oauth_retry_waits / apikey_retry_waits) and
#:    the throttle's shape. It touches no filesystem path, so unlike the config globals
#:    that were removed elsewhere, a test cannot pollute anything through it -- and the
#:    ceiling-learning that DID write to disk now lives in transport, next to its cfg.
#:
#: The retry decorator also wraps engine methods whose first argument is `self`
#: (better_rlm/auth.py patches AnthropicClient.completion), so cfg cannot be threaded in
#: positionally the way subquery's was.
_CFG = load_config()
_LOG = logging.getLogger("rlm-mcp")


def _is_rate_limit(exc: BaseException) -> bool:
    """Retryable, decided by the one classifier both transports share.

    This was three inline tests (SDK type, status 429, the CLI's duck-typed flag) with no
    5xx branch at all -- so a 529 "overloaded" was retried on the OAuth path, where
    "overloaded" is one of the CLI's markers, and not on the API path. One condition, two
    answers, because the fix landed on the route that reported it. See failures.py.
    """
    return failures.is_rate_limit(exc)


def is_fatal_auth(exc: BaseException) -> bool:
    """Auth itself is dead — a property of the login, not of this one call.

    Retrying cannot help and neither can the next chunk, so batch callers abort on it
    rather than reproducing the identical failure N times.

    Two changes came with the shared classifier. A 403 counts now: ``PermissionDeniedError``
    is not an ``AuthenticationError`` subclass, so an org without access to the configured
    endpoint used to fan out one doomed call per chunk on the API path while the OAuth
    path aborted. And a budget stop no longer counts: it also carries
    ``is_fatal_subcall``, so this returned True for scheduled work: ``subquery`` only
    avoided conflating them by catching ``BudgetStopError`` in an earlier clause.
    """
    return failures.is_auth_dead(exc)


def _retry_after_seconds(exc: BaseException) -> float:
    try:
        ra = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(ra) if ra else 0.0
    except Exception:
        return 0.0


def _waits() -> list[float]:
    """Backoff schedule for the active transport (claude CLI gets the longer waits).

    Never raises. resolve_auth_mode needs a usable transport and raises when there is
    none, but picking a wait schedule is the wrong place to discover that: it turned
    every wrapped call on a machine with no CLI and no usable key into "No transport
    available" raised out of the retry decorator, before the wrapped call itself could
    report the real problem. Unresolvable auth falls back to the shorter apikey waits.
    """
    from .auth import resolve_auth_mode  # lazy import to avoid a cycle
    try:
        oauth = resolve_auth_mode(_CFG) == "oauth"
    except Exception:                # noqa: BLE001 - schedule choice must not fail the call
        oauth = False
    sched = _CFG.oauth_retry_waits if oauth else _CFG.apikey_retry_waits
    return list(sched)


def _next_delay(exc: BaseException, attempt: int, waits: list[float]) -> float | None:
    """Seconds to wait before retrying ``exc``, or None if it must propagate.

    The single home of the retry decision, so the sync and async wrappers below
    cannot drift apart. Ceiling-learning lives in transport, next to the cfg it writes with.
    """
    if not _is_rate_limit(exc) or attempt >= len(waits):
        return None
    return max(waits[attempt], _retry_after_seconds(exc))


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
                    delay = _next_delay(exc, attempt, waits)
                    if delay is None:
                        raise
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
                delay = _next_delay(exc, attempt, waits)
                if delay is None:
                    raise
            log_event(_LOG, "retry", attempt=attempt + 1, delay_s=round(delay, 1),
                      reason="rate_limit", mode="async")
            await asyncio.sleep(delay)
            attempt += 1

    return wrapper
