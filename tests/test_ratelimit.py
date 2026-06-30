import pytest

import src.ratelimit as rl


class _RateLimit(Exception):
    status_code = 429


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Make throttle spacing + backoff instant; keep the REAL _waits() so the
    # auth-aware schedule is exercised (values don't matter when sleep is a no-op).
    monkeypatch.setattr(rl.time, "sleep", lambda *_: None)


def test_retries_then_succeeds():
    n = {"calls": 0}

    @rl.retry_and_queue_retries
    def f():
        n["calls"] += 1
        if n["calls"] < 3:
            raise _RateLimit()
        return "ok"

    assert f() == "ok"
    assert n["calls"] == 3


def test_fails_after_exhausting_retries():
    @rl.retry_and_queue_retries
    def f():
        raise _RateLimit()

    with pytest.raises(_RateLimit):
        f()  # 1 initial attempt + len(waits) retries, then re-raise


def test_non_429_is_not_retried():
    n = {"calls": 0}

    @rl.retry_and_queue_retries
    def f():
        n["calls"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        f()
    assert n["calls"] == 1


def test_waits_are_transport_aware(monkeypatch):
    import src.auth as auth_mod
    # Backoff keys off the RESOLVED transport, not raw env tokens.
    monkeypatch.setattr(auth_mod, "resolve_auth_mode", lambda cfg: "oauth")
    assert rl._waits()[0] == 5  # claude CLI → patient backoff

    monkeypatch.setattr(auth_mod, "resolve_auth_mode", lambda cfg: "apikey")
    assert rl._waits()[0] == 1  # API key → faster backoff


def test_retry_after_header_is_honored(monkeypatch):
    class _Resp:
        headers = {"retry-after": "42"}

    exc = _RateLimit()
    exc.response = _Resp()
    assert rl._retry_after_seconds(exc) == 42.0


def test_cli_rate_limit_marker_is_retried():
    # The CLI transport raises errors flagged is_rate_limit=True (CliRateLimitError);
    # the decorator must treat those as retryable, like a 429.
    class _CliRateLimit(Exception):
        is_rate_limit = True

    n = {"calls": 0}

    @rl.retry_and_queue_retries
    def f():
        n["calls"] += 1
        if n["calls"] < 2:
            raise _CliRateLimit()
        return "ok"

    assert f() == "ok"
    assert n["calls"] == 2
