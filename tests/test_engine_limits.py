"""The engine's own guardrails, wired instead of re-implemented.

RLM.__init__ ships max_timeout / max_errors and raises typed exceptions that carry
the best answer found so far. Nothing in src/ passed or caught them, so a runaway
rlm_query was bounded only by max_iterations x cli_timeout_s (over three hours) and
a limit surfaced as a raw traceback with the partial work thrown away.
"""

import pathlib

import pytest
from rlm.utils.exceptions import (
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
)
from rlm.utils.token_utils import DEFAULT_CONTEXT_LIMIT, get_context_limit

import src.engine as eng
from src.config import _sub_ctx, load_config


# --- the engine's model table has to know the models we actually run -----------
@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-sonnet-4-6", 200_000),
])
def test_current_claude_models_are_known(model, expected):
    """Before this, every model this fork runs fell through to the 128k default
    while a 2024 id resolved correctly -- so _should_compact would fire at 8x less
    context than the model has."""
    assert get_context_limit(model) == expected
    assert get_context_limit(model) != DEFAULT_CONTEXT_LIMIT or expected == DEFAULT_CONTEXT_LIMIT


def test_unknown_model_still_falls_back_safely():
    assert get_context_limit("some-model-nobody-listed") == DEFAULT_CONTEXT_LIMIT


# --- sub_context_tokens: derived, not hand-maintained -------------------------
def test_sub_context_tokens_is_derived_from_the_sub_model():
    assert _sub_ctx({"sub_context_tokens": 0, "sub_model": "claude-haiku-4-5"}) == 200_000
    # a SMALLER model must derive its own smaller limit, not the cap
    assert _sub_ctx({"sub_context_tokens": 0, "sub_model": "gpt-4"}) == 8_192


def test_derivation_never_raises_the_ceiling_above_the_cap():
    """A model's window is a ceiling, not a target: this server scans oversized input
    rather than stuffing it into the prompt, and a near-window sub-query is worse than
    chunking the same bytes. So derivation only ever corrects DOWNWARD."""
    from src.config import SUB_CONTEXT_CAP

    for big in ("claude-sonnet-5", "gemini-2.5-pro"):
        assert get_context_limit(big) > SUB_CONTEXT_CAP, "test model is not actually big"
        assert _sub_ctx({"sub_context_tokens": 0, "sub_model": big}) == SUB_CONTEXT_CAP


def test_an_explicit_sub_context_tokens_still_wins():
    assert _sub_ctx({"sub_context_tokens": 12_345, "sub_model": "claude-sonnet-5"}) == 12_345


def test_guardrails_are_configured_by_default():
    cfg = load_config()
    assert cfg.query_timeout_s > 0, "an unbounded rlm_query is the thing this fixes"
    assert cfg.query_max_errors > 0


# --- a limit is a deliberate stop, not a crash -------------------------------
class _Boom:
    """Stands in for the RLM object: completion() raises a limit exception."""

    def __init__(self, exc):
        self.exc = exc

    def completion(self, **kwargs):
        raise self.exc


# Three of the five limit types carry partial_answer as a real field; the argument
# order matters (CancellationError's FIRST positional is partial_answer, not a message).
@pytest.mark.parametrize("exc,partial", [
    (TimeoutExceededError(elapsed=9.0, timeout=5.0, partial_answer="found 3 of 5"),
     "found 3 of 5"),
    (ErrorThresholdExceededError(error_count=5, threshold=5, partial_answer="got 2 rows"),
     "got 2 rows"),
    (ErrorThresholdExceededError(error_count=5, threshold=5), ""),
    (CancellationError(partial_answer="half done"), "half done"),
    (CancellationError(), ""),
])
def test_run_query_returns_the_limit_and_keeps_the_partial_answer(monkeypatch, exc, partial):
    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: _Boom(exc))
    res = eng.run_query(load_config(), "ctx", "q?", "root-m", "sub-m")
    assert res["limit"] == type(exc).__name__
    assert res["answer"] == partial, "partial answer lost"
    assert res["limit_detail"]


def test_rlm_query_reports_a_limit_as_a_failed_tool_call(monkeypatch):
    """Burying it under the '## RLM answer' header would log outcome=ok."""
    import src.server as srv

    monkeypatch.setattr(srv, "run_query", lambda *a, **k: {
        "answer": "partial finding", "limit": "TimeoutExceededError",
        "limit_detail": "Timeout exceeded: 1801.0s of 1800.0s limit",
        "execution_time": 0.0, "root_model": "r", "sub_model": "s",
        "usage": [], "cost_usd": None,
    })
    monkeypatch.setattr(srv.STORE, "read_text", lambda ctx: "text")
    out = srv.rlm_query("ctx_x", "question")
    assert out.lstrip().startswith("ERROR"), out[:120]
    assert "TimeoutExceededError" in out
    assert "partial finding" in out, "threw away the work the engine handed back"


def test_iteration_callbacks_are_wired_on_both_exit_paths():
    """on_iteration_start/complete were dead API upstream: declared, documented and
    stored by __init__, never invoked. The loop RETURNS from inside its body when the
    answer is ready, so 'complete' needs a call site there too -- otherwise the final
    iteration is never reported and a one-iteration query emits nothing.

    Verified live: a 3-turn query logged iter=0,1,2 with turns=3.
    """
    src = (pathlib.Path(eng.__file__).parent.parent / "rlm/core/rlm.py").read_text()
    assert "_fire(self.on_iteration_start" in src, "start call site gone"
    assert src.count("_fire(self.on_iteration_complete") == 2, (
        "expected a 'complete' call site on BOTH the answer path and the continue path"
    )
    # and better-rlm actually passes them
    eng_src = pathlib.Path(eng.__file__).read_text()
    assert "on_iteration_complete=_log_iteration" in eng_src
    assert "on_subcall_complete=_log_subcall" in eng_src
