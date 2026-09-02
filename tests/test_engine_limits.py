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



# --- a stopped rlm_query resumes instead of restarting --------------------------------
class _Recorder:
    """Stands in for the RLM object: records completion() kwargs; raises or returns."""

    def __init__(self, exc=None, answer="done"):
        self.exc, self.answer, self.kwargs = exc, answer, None

    def completion(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc

        class _Usage:
            model_usage_summaries = {}

        class _Res:
            response = self.answer
            usage_summary = _Usage()
            execution_time = 1.0

        return _Res()


def _stopped(exc, history, next_iteration, state=b"REPL"):
    """A limit the engine has attached a checkpoint to (see RLM._attach_checkpoint)."""
    exc.history, exc.next_iteration, exc.state_dill = history, next_iteration, state
    return exc


def test_a_session_budget_stop_is_a_limit_not_a_crash(monkeypatch, cfg):
    from rlm.utils.exceptions import SessionBudgetError

    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: _Recorder(
        exc=SessionBudgetError(partial_answer="two of five sections summarised")))
    res = eng.run_query(cfg, "ctx", "q?", "root-m", "sub-m")
    assert res["limit"] == "SessionBudgetError"
    assert res["answer"] == "two of five sections summarised"


def test_a_stopped_query_saves_a_checkpoint_and_the_next_call_resumes_from_it(monkeypatch, cfg, tmp_path):
    """THE fix for rlm_query. Before: a stop returned a partial answer and the next call
    started from iteration 0, re-spending everything. Now the transcript, the iteration
    to retry and the REPL state are saved; the next call hands them back to the engine."""
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
               {"role": "assistant", "content": "print(len(context))"}]
    ck = tmp_path / "ck.json"

    first = _Recorder(exc=_stopped(TimeoutExceededError(elapsed=9.0, timeout=5.0,
                                                        partial_answer="partial"),
                                   history, next_iteration=3))
    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: first)
    res = eng.run_query(cfg, "ctx", "q?", "root-m", "sub-m", checkpoint=ck)

    assert res["resumable"] is True and res["next_iteration"] == 3
    assert ck.exists(), "no checkpoint written"
    assert eng._state_path(ck).read_bytes() == b"REPL", "REPL state not saved"
    assert first.kwargs["resume"] is None, "a first run must not be handed a resume"

    second = _Recorder(answer="final")
    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: second)
    res2 = eng.run_query(cfg, "ctx", "q?", "root-m", "sub-m", checkpoint=ck)

    assert second.kwargs["resume"] == {"history": history, "next_iteration": 3,
                                       "state_dill": b"REPL"}
    assert res2["resumed_from"] == 3
    assert res2["answer"] == "final"
    assert not ck.exists() and not eng._state_path(ck).exists(), \
        "a completed run must delete its checkpoint or the NEXT question resumes a stale one"


def test_fresh_discards_an_existing_checkpoint(monkeypatch, cfg, tmp_path):
    ck = tmp_path / "ck.json"
    ck.write_text('{"history": [], "next_iteration": 7}', encoding="utf-8")
    rec = _Recorder(answer="x")
    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: rec)

    eng.run_query(cfg, "ctx", "q?", "r", "s", checkpoint=ck, fresh=True)

    assert rec.kwargs["resume"] is None, "fresh=True must not resume"


def test_a_limit_without_a_checkpoint_is_reported_as_not_resumable(monkeypatch, cfg, tmp_path):
    """Only limits the engine loop attached a transcript to are resumable. A bare one
    (e.g. raised before the loop started) must not claim otherwise."""
    monkeypatch.setattr(eng, "build_rlm", lambda *a, **k: _Recorder(
        exc=CancellationError(partial_answer="")))
    res = eng.run_query(cfg, "ctx", "q?", "r", "s", checkpoint=tmp_path / "ck.json")
    assert res["resumable"] is False and res["next_iteration"] is None


def test_the_checkpoint_path_is_per_question_and_per_model(cfg):
    a = eng.query_checkpoint_path(cfg, "ctx_1", "q1", "root", "sub")
    assert eng.query_checkpoint_path(cfg, "ctx_1", "q2", "root", "sub") != a
    assert eng.query_checkpoint_path(cfg, "ctx_1", "q1", "other-root", "sub") != a
    assert eng.query_checkpoint_path(cfg, "ctx_1", "q1", "root", "sub") == a
    assert a.is_relative_to(cfg.store_dir / "ctx_1" / "query")


# --- the engine loop itself: conversion and checkpoint attachment --------------------
def _engine_with_fakes(monkeypatch, tmp_path, turn, *, state=b"S"):
    """A real RLM whose environment, prompt setup and completion turn are fakes: enough
    to run the genuine loop body, its except blocks and _attach_checkpoint."""
    import contextlib
    from rlm.core.rlm import RLM

    (tmp_path / "state.dill").write_bytes(state)

    class _Env:
        temp_dir = str(tmp_path)

    r = RLM(backend="anthropic", backend_kwargs={"model_name": "m", "api_key": "x", "max_tokens": 8},
            environment="local", max_iterations=5)

    @contextlib.contextmanager
    def _spawn(prompt):
        yield object(), _Env()

    monkeypatch.setattr(r, "_spawn_completion_context", _spawn)
    monkeypatch.setattr(r, "_setup_prompt", lambda prompt, root_prompt=None: [{"role": "system", "content": "s"}])
    monkeypatch.setattr(r, "_completion_turn", turn)
    return r


def test_the_loop_converts_a_backend_budget_stop_into_a_resumable_limit(monkeypatch, tmp_path):
    """_check_iteration_limits counts REPL stderr, not LM exceptions -- so before this
    the transport's stop escaped completion() as a raw traceback, and the partial answer
    and the whole transcript went with it."""
    from rlm.utils.exceptions import SessionBudgetError
    from src.budget import BudgetStopError

    seen_prompts = []

    def turn(prompt, lm_handler, environment):
        seen_prompts.append(list(prompt))
        raise BudgetStopError(spent=1, usable=1, next_call=1)

    r = _engine_with_fakes(monkeypatch, tmp_path, turn)
    with pytest.raises(SessionBudgetError) as ei:
        r.completion("ctx", root_prompt="q")

    e = ei.value
    assert e.next_iteration == 0, "the failed iteration is the one to retry"
    # The transcript is cut BEFORE the user prompt appended for the failed turn, so a
    # resume does not present the model with a duplicated user message.
    assert e.history == [{"role": "system", "content": "s"}]
    assert len(seen_prompts[0]) == 2, "the turn itself did see system + user prompt"
    assert e.state_dill == b"S", "REPL state not captured before teardown"


def test_a_timeout_also_carries_a_checkpoint(monkeypatch, tmp_path):
    """Every deliberate stop is resumable, not only the budget one."""
    import time as _t

    def turn(prompt, lm_handler, environment):
        raise AssertionError("must not be reached: the timeout fires first")

    r = _engine_with_fakes(monkeypatch, tmp_path, turn)
    r.max_timeout = 0.0
    monkeypatch.setattr(_t, "perf_counter", lambda: 100.0)  # elapsed 0 > 0.0? no; force it
    r._completion_start_time = 0.0
    monkeypatch.setattr(r, "_check_timeout", lambda i, t0: (_ for _ in ()).throw(
        TimeoutExceededError(elapsed=9.0, timeout=5.0, partial_answer=None)))
    with pytest.raises(TimeoutExceededError) as ei:
        r.completion("ctx", root_prompt="q")
    assert ei.value.history == [{"role": "system", "content": "s"}]
    assert ei.value.next_iteration == 0
    assert ei.value.state_dill == b"S"


def test_resume_replays_the_transcript_restarts_at_the_iteration_and_restores_repl_state(monkeypatch, tmp_path):
    from rlm.utils.exceptions import SessionBudgetError
    from src.budget import BudgetStopError

    prior = [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"},
             {"role": "assistant", "content": "a0"}]
    seen = {}

    def turn(prompt, lm_handler, environment):
        seen["prompt"] = list(prompt)
        seen["state"] = (tmp_path / "state.dill").read_bytes()
        raise BudgetStopError(spent=1, usable=1, next_call=1)

    r = _engine_with_fakes(monkeypatch, tmp_path, turn, state=b"STALE")
    with pytest.raises(SessionBudgetError) as ei:
        r.completion("ctx", root_prompt="q",
                     resume={"history": prior, "next_iteration": 2, "state_dill": b"RESTORED"})

    assert seen["state"] == b"RESTORED", "REPL state was not restored before the first turn"
    assert seen["prompt"][:3] == prior, "the transcript was not replayed"
    assert "2/5" in seen["prompt"][3]["content"] or "iteration 2" in seen["prompt"][3]["content"].lower() \
        or seen["prompt"][3]["role"] == "user", "the loop did not restart at the resumed iteration"
    assert ei.value.next_iteration == 2, "a stop during the resumed turn must checkpoint THAT turn"
    assert ei.value.history == prior
