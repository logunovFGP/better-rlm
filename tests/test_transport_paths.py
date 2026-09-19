"""The transport code paths nothing exercised.

Written BEFORE the shared-classifier refactor, against the code as it stands, so the
refactor has something to break. Every test here passed on the unchanged tree.

What was uncovered, and why it matters:

  * ``CliTransport.acomplete`` -- zero tests. Every stub in the suite defines
    ``async def acomplete(...): raise AssertionError("not used")``, so the async half of
    the OAuth path, which ``rlm_query``'s fan-out actually uses, was never run.
  * ``ApiTransport.complete`` / ``acomplete`` -- no test ever called through them; only
    the lazy client builders were checked.
  * ``_LedgeredTransport._note_if_limit`` -- zero tests, and it decides when a rate limit
    is recorded as evidence about the account's real ceiling.
  * ``ratelimit``'s ``isinstance(anthropic.RateLimitError)`` and ``AuthenticationError``
    branches -- every retry test uses a duck-typed fake, so the SDK branches the API path
    depends on were never taken.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import types

import anthropic
import httpx
import pytest

import better_rlm.ratelimit as rl
import better_rlm.transport as tp
from better_rlm.config import load_config


def _sdk(kind, status: int, body: str = "boom"):
    """A real SDK exception, built the way the SDK builds one.

    Constructed rather than faked on purpose: the point of these cases is the isinstance
    branch, which a SimpleNamespace carrying a status_code cannot reach.
    """
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    if kind is anthropic.APIConnectionError:
        return kind(message=body, request=request)
    return kind(body, response=httpx.Response(status, request=request), body=None)


# --- the async half of the CLI transport -------------------------------------
def _cli(monkeypatch, tmp_path):
    t = tp.CliTransport(dataclasses.replace(load_config(), cli_timeout_s=5))
    monkeypatch.setattr(t, "_neutral_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(t, "_prepare", lambda m, s, model: (["dummy-argv"], "prompt"))
    return t


def _fake_exec(monkeypatch, proc):
    async def spawn(*a, **k):
        return proc
    monkeypatch.setattr(tp.asyncio, "create_subprocess_exec", spawn)


def test_the_async_cli_path_parses_a_successful_spawn(monkeypatch, tmp_path):
    payload = json.dumps({"subtype": "success", "is_error": False, "result": "ANSWER",
                          "usage": {"input_tokens": 5, "output_tokens": 2}})

    class _Proc:
        returncode = 0

        async def communicate(self, _stdin):
            return payload.encode(), b""

    _fake_exec(monkeypatch, _Proc())
    res = asyncio.run(_cli(monkeypatch, tmp_path).acomplete([], None, "m", 16))
    assert res.text == "ANSWER"
    assert (res.input_tokens, res.output_tokens) == (5, 2)


def test_the_async_cli_path_kills_the_child_it_stopped_waiting_for(monkeypatch, tmp_path):
    """asyncio.wait_for cancels the wait and leaves the child running, so this half needs
    a proc.kill() the sync half gets for free from subprocess.run. Nothing exercised it."""
    killed: list[bool] = []

    class _Proc:
        returncode = None

        # Not a coroutine: the stubbed wait_for below never awaits it, and an
        # un-awaited coroutine surfaces as a RuntimeWarning at the next gc.
        def communicate(self, _stdin):
            return None

        def kill(self):
            killed.append(True)

    _fake_exec(monkeypatch, _Proc())

    async def boom(*a, **k):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(tp.asyncio, "wait_for", boom)
    with pytest.raises(tp.CliCompletionError, match="timed out"):
        asyncio.run(_cli(monkeypatch, tmp_path).acomplete([], None, "m", 16))
    assert killed == [True], "the child was left running"


def test_the_async_cli_path_classifies_a_rate_limit_like_the_sync_one(monkeypatch, tmp_path):
    payload = json.dumps({"subtype": "error", "is_error": True,
                          "result": "Rate limit exceeded, please retry"})

    class _Proc:
        returncode = 1

        async def communicate(self, _stdin):
            return payload.encode(), b""

    _fake_exec(monkeypatch, _Proc())
    with pytest.raises(tp.CliRateLimitError):
        asyncio.run(_cli(monkeypatch, tmp_path).acomplete([], None, "m", 16))


# --- the API transport, called through ---------------------------------------
def _sdk_response(text="hi"):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=types.SimpleNamespace(input_tokens=7, output_tokens=3))


class _FakeMessages:
    def __init__(self, calls):
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        return _sdk_response()


def test_the_api_transport_sends_the_system_prompt_and_reads_the_response(monkeypatch):
    calls: list[dict] = []
    t = tp.ApiTransport(load_config())
    monkeypatch.setattr(t, "_sync_client",
                        lambda: types.SimpleNamespace(messages=_FakeMessages(calls)))

    res = t.complete([{"role": "user", "content": "q"}], "SYS", "m", 16)
    assert res.text == "hi" and res.model == "m"
    assert calls[0]["system"] == "SYS", "the system prompt never reached the endpoint"
    assert calls[0]["max_tokens"] == 16


def test_the_api_transport_omits_system_when_there_is_none(monkeypatch):
    calls: list[dict] = []
    t = tp.ApiTransport(load_config())
    monkeypatch.setattr(t, "_sync_client",
                        lambda: types.SimpleNamespace(messages=_FakeMessages(calls)))
    t.complete([{"role": "user", "content": "q"}], None, "m", 16)
    assert "system" not in calls[0], "an empty system key changes the request"


def test_the_async_api_path_returns_the_same_result_as_the_sync_one(monkeypatch):
    calls: list[dict] = []

    class _AsyncMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return _sdk_response()

    t = tp.ApiTransport(load_config())
    monkeypatch.setattr(t, "_async_client",
                        lambda: types.SimpleNamespace(messages=_AsyncMessages()))
    res = asyncio.run(t.acomplete([{"role": "user", "content": "q"}], "SYS", "m", 16))
    assert res.text == "hi" and res.input_tokens == 7
    assert calls[0]["system"] == "SYS"


# --- what counts as evidence about the account ceiling -----------------------
@pytest.mark.parametrize("exc,expected", [
    (tp.CliRateLimitError("slow down"), True),          # duck-typed flag
    (_sdk(anthropic.RateLimitError, 429), True),        # SDK type / status_code
    (RuntimeError("rate limit exceeded"), True),        # bare string, the CLI's only signal
    (RuntimeError("nothing to do with limits"), False),
])
def test_a_rate_limit_is_recorded_as_evidence_about_the_ceiling(monkeypatch, cfg, exc, expected):
    """_note_if_limit had no tests at all, and it is what teaches budget.ceiling where the
    account's real wall is. A false positive raises a floor nobody hit."""
    noted: list[int] = []
    monkeypatch.setattr(tp.budget, "note_limit_hit", lambda c: noted.append(1))

    class _Boom:
        def complete(self, *a):
            raise exc

        async def acomplete(self, *a):    # pragma: no cover - not the path under test
            raise exc

    w = tp._LedgeredTransport(_Boom(), cfg)
    with pytest.raises(type(exc)):
        w.complete([{"role": "user", "content": "q"}], None, "m", 16)
    assert bool(noted) is expected


# --- the SDK branches of the retry policy ------------------------------------
def test_the_sdk_rate_limit_type_is_retried():
    """Every other retry test uses a duck-typed fake carrying status_code=429, so the
    isinstance branch the API path relies on was never taken."""
    assert rl._is_rate_limit(_sdk(anthropic.RateLimitError, 429))


def test_the_sdk_auth_error_is_fatal_to_a_fan_out():
    assert rl.is_fatal_auth(_sdk(anthropic.AuthenticationError, 401))
