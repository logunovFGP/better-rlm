"""One failure, one answer, whichever route produced it.

This is the regression guard the consolidation exists for. Each row below is a failure
CONDITION expressed twice: as the exception the SDK raises on the API path, and as the
error the `claude` CLI produces on the OAuth path. Both columns must classify the same.

They did not, and the disagreements were not theoretical:

  * a 403 aborted the fan-out on OAuth and fanned out one doomed call per chunk on the API;
  * a 529 "overloaded" retried on OAuth and failed immediately on the API;
  * a bare "429" substring decided whether a rate limit was recorded as evidence about the
    account's ceiling, so `5,429,000 tokens` in any message raised a floor nobody hit.
"""

from __future__ import annotations

import anthropic
import httpx
import pytest

from better_rlm import failures
from better_rlm.budget import BudgetStopError
from better_rlm.transport import CliAuthError, CliCompletionError, CliRateLimitError


def _sdk(kind, status: int, body: str = "boom"):
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    if kind is anthropic.APIConnectionError:
        return kind(message=body, request=request)
    return kind(body, response=httpx.Response(status, request=request), body=None)


# (condition, what the SDK raises, what the CLI produces, the action, the shared code)
#
# `code` is None where the two routes CANNOT agree on a name, only on what to do. The
# CLI reports no status and raises one CliRateLimitError for both 429 and 529, so
# "throttled" and "overloaded" are genuinely indistinguishable on that path. Asserting
# one code there would be asserting a distinction the evidence does not carry. The
# ACTION is the contract; the code is the contract wherever both routes can know it.
RETRY, ABORT, PROPAGATE = "retry", "abort", "propagate"

PARITY = [
    ("rate limited",
     _sdk(anthropic.RateLimitError, 429),
     CliRateLimitError("claude CLI rate limited: usage limit reached"),
     RETRY, failures.CODE_RATE_LIMITED),
    ("auth dead",
     _sdk(anthropic.AuthenticationError, 401),
     CliAuthError("claude CLI auth failed: OAuth session expired"),
     ABORT, failures.CODE_KEY_REJECTED),
    ("permission denied",
     _sdk(anthropic.PermissionDeniedError, 403),
     CliAuthError("claude CLI auth failed: invalid api key"),
     ABORT, failures.CODE_KEY_REJECTED),
    ("overloaded",
     _sdk(anthropic.InternalServerError, 529),
     CliRateLimitError("claude CLI rate limited: overloaded"),
     RETRY, None),
    ("ordinary failure",
     _sdk(anthropic.BadRequestError, 400),
     CliCompletionError("claude CLI failed (exit 1): crashed"),
     PROPAGATE, None),
]
_IDS = [row[0] for row in PARITY]


@pytest.mark.parametrize("condition,sdk_exc,cli_exc,action,code", PARITY, ids=_IDS)
def test_both_routes_agree_on_what_to_do(condition, sdk_exc, cli_exc, action, code):
    """The contract. A fix to retry or abort policy must land on both routes at once."""
    for label, exc in (("API", sdk_exc), ("CLI", cli_exc)):
        assert failures.is_rate_limit(exc) is (action == RETRY), f"{label} retry: {condition}"
        assert failures.is_auth_dead(exc) is (action == ABORT), f"{label} abort: {condition}"


@pytest.mark.parametrize("condition,sdk_exc,cli_exc,action,code", PARITY, ids=_IDS)
def test_both_routes_name_it_the_same_where_they_can(condition, sdk_exc, cli_exc, action, code):
    if code is None:
        pytest.skip("the CLI's signal cannot carry this distinction")
    assert failures.classify(sdk_exc) == code, f"API path disagrees on {condition}"
    assert failures.classify(cli_exc) == code, f"CLI path disagrees on {condition}"


# --- the three divergences, each pinned on its own ---------------------------
def test_a_403_aborts_a_fan_out_on_both_routes():
    """PermissionDeniedError is not an AuthenticationError subclass, so is_fatal_auth was
    False: an org without access to the configured endpoint spent one doomed call per
    chunk on the API path, while the OAuth path stopped after the first."""
    from better_rlm.ratelimit import is_fatal_auth

    assert is_fatal_auth(_sdk(anthropic.PermissionDeniedError, 403))


def test_an_overloaded_endpoint_is_retried_on_both_routes():
    """`overloaded` is one of the CLI's markers, so 529 always retried there. The SDK side
    had no 5xx branch at all and gave up on the first attempt."""
    from better_rlm.ratelimit import _is_rate_limit

    assert _is_rate_limit(_sdk(anthropic.InternalServerError, 529))
    assert _is_rate_limit(CliRateLimitError("overloaded"))


def test_a_number_containing_429_is_not_a_rate_limit():
    """This feeds budget.note_limit_hit, which raises the learned ceiling. A bare
    substring test read `5,429,000 tokens` as a wall the account never hit."""
    assert not failures.is_rate_limit(RuntimeError("used 5,429,000 tokens this window"))
    assert failures.is_rate_limit(_sdk(anthropic.RateLimitError, 429)), "a real 429 counts"


def test_a_budget_stop_is_not_an_auth_failure():
    """BudgetStopError carries is_fatal_subcall too, so "auth is dead" and "the budget is
    exhausted" shared one predicate. subquery avoided conflating them only by catching
    BudgetStopError in an earlier clause; any other caller got it wrong."""
    stop = BudgetStopError(spent=100, usable=90, next_call=10)
    assert not failures.is_auth_dead(stop)


def test_an_ordinary_failure_is_neither():
    for exc in (CliCompletionError("the CLI crashed"), ValueError("nonsense")):
        assert not failures.is_rate_limit(exc)
        assert not failures.is_auth_dead(exc)


# --- ordering, which is the design ------------------------------------------
def test_a_flag_is_believed_before_the_sdk_is_consulted(monkeypatch):
    """Flags-first is what lets the CLI's errors classify without the SDK being involved
    at all. Asserted directly: an exception carrying the flag must not need anthropic."""
    import builtins

    real_import = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise AssertionError("the SDK was consulted for a CLI error")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    assert failures.classify(CliRateLimitError("slow down")) == failures.CODE_RATE_LIMITED
    assert failures.classify(CliAuthError("expired")) == failures.CODE_KEY_REJECTED


# --- the two fixes that are about reaching BOTH fan-outs ---------------------
def test_a_dead_api_key_aborts_the_engines_fan_out_too(monkeypatch):
    """The engine's aborts_batch is duck-typed on is_fatal_subcall alone -- correct, it
    must not import a backend -- but the SDK's AuthenticationError carries no such
    attribute, so the contract only ever worked for the CLI. Measured before the fix:
    aborts_batch(AuthenticationError) False, aborts_batch(CliAuthError) True. A dead API
    key stopped OUR batch and not the engine's, so rlm_query issued one doomed call per
    prompt."""
    import better_rlm.transport as tp
    from rlm.utils.exceptions import aborts_batch

    class _Messages:
        def create(self, **kwargs):
            raise _sdk(anthropic.AuthenticationError, 401)

    class _Client:
        messages = _Messages()

    t = tp.ApiTransport.__new__(tp.ApiTransport)
    monkeypatch.setattr(t, "_sync_client", lambda: _Client(), raising=False)

    with pytest.raises(anthropic.AuthenticationError) as caught:
        tp.ApiTransport.complete(t, [], None, "m", 16)
    assert aborts_batch(caught.value), "the engine's fan-out would keep going"
    assert aborts_batch(CliAuthError("expired")), "and the CLI path still does"


def test_the_engine_path_reports_truncation_like_the_sub_query_path():
    """auth.patch_engine's shim returned res.text and dropped res.truncated, so a
    MiniMax rlm_query got silently empty chunk answers for a release after sub_query
    stopped doing that. The note is shared so the two cannot word it differently --
    and calling it here is what catches a typo in the shim's import."""
    from better_rlm.transport import truncation_note

    assert "TRUNCATED" in truncation_note("", 16384)
    assert "reasoning" in truncation_note("", 16384), "an empty answer needs the reason"
    assert "TRUNCATED" in truncation_note("a partial list", 16384)
    assert "16,384" in truncation_note("", 16384), "name the cap that was hit"


def test_a_404_needs_the_model_name_to_mean_model_unknown():
    """Without it, a wrong base URL and an unserved model are the same 404. The safer
    reading wins when the caller cannot say: a bad URL is the commoner setup mistake."""
    exc = _sdk(anthropic.NotFoundError, 404, "model MiniMax-M2.7 does not exist")
    assert failures.classify(exc, model="MiniMax-M2.7") == failures.CODE_MODEL_UNKNOWN
    assert failures.classify(exc) == failures.CODE_URL_WRONG
