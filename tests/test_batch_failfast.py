"""A dead login is one failure, not N.

Covers the path that let 20 chunks each spawn their own doomed ``claude -p``:
transport has to flag an auth failure distinctly, ratelimit has to classify it as
fatal, and sub_query_batch has to stop fanning out once it has seen one — without
throwing away chunks that fail for their own, local reasons.
"""

import json

import pytest

import src.subquery as sq
from src.ratelimit import is_fatal_auth
from src.transport import (
    CliAuthError,
    CliCompletionError,
    CliRateLimitError,
    _parse_cli_output,
)

# The literal envelope observed on this machine: subtype=success WITH is_error=true,
# so the auth failure arrives dressed as a successful call.
_AUTH_PAYLOAD = json.dumps({
    "type": "result", "subtype": "success", "is_error": True,
    "result": "Failed to authenticate: OAuth session expired and could not be refreshed",
})


@pytest.fixture(autouse=True)
def _pin_auth_discovery(monkeypatch):
    """models.select() resolves the auth mode to map each role onto a subscription
    sibling, so on a machine with neither the claude CLI nor a usable key it raises
    before any stub inside a test can matter - that is every CI runner, and every
    contributor without Claude Code installed. These tests are about probe and batch
    REPORTING, so both are pinned to the oauth path they describe.
    """
    import src.server as srv

    monkeypatch.setattr(srv, "resolve_auth_mode_safe", lambda: "oauth")
    monkeypatch.setattr(srv.models, "select", lambda cfg, role: "claude-haiku-4-5")


def test_auth_failure_parses_to_its_own_error_type():
    with pytest.raises(CliAuthError):
        _parse_cli_output(0, _AUTH_PAYLOAD, "", "claude-haiku-4-5")


def test_only_auth_counts_as_fatal():
    assert is_fatal_auth(CliAuthError("dead login"))
    assert not is_fatal_auth(CliCompletionError("boom"))
    assert not is_fatal_auth(CliRateLimitError("429"))


def _run_batch(monkeypatch, exc, n=10):
    calls: list[str] = []

    def fake_call(model, prompt, max_tokens, system):
        calls.append(prompt)
        raise exc

    monkeypatch.setattr(sq, "_call", fake_call)
    results = sq.sub_query_batch([f"p{i}" for i in range(n)], "m", concurrency=1)
    return calls, results


def test_auth_failure_stops_the_fan_out(monkeypatch):
    calls, results = _run_batch(monkeypatch, CliAuthError("OAuth session expired"))
    assert len(calls) == 1, "chunks after the first re-ran a call already known dead"
    assert len(results) == 10 and all(r.error for r in results)
    assert results[-1].error.startswith("skipped —")


def test_ordinary_failure_does_not_discard_the_other_chunks(monkeypatch):
    calls, _ = _run_batch(monkeypatch, CliCompletionError("just this chunk"))
    assert len(calls) == 10, "a chunk-local failure must not abort the whole batch"


# --- rlm_status(probe=True): the line that separates "configured" from "working" ---
def _login_ok(monkeypatch):
    """Pin the free login check to 'logged in' so these tests exercise the probe
    itself, not whatever auth state the machine running them happens to be in."""
    import src.server as srv

    monkeypatch.setattr(srv.transport, "cli_auth_status",
                        lambda cfg: {"loggedIn": True, "authMethod": "oauth"})
    return srv


def test_probe_reports_failure_instead_of_raising(monkeypatch):
    srv = _login_ok(monkeypatch)
    monkeypatch.setattr(srv, "sub_query",
                        lambda *a, **k: sq.SubResult(0, "", 0, 0, error="OAuth session expired"))
    assert "FAILED — OAuth session expired" in srv._auth_probe_line()


def test_probe_survives_a_transport_that_explodes(monkeypatch):
    srv = _login_ok(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(srv, "sub_query", boom)
    line = srv._auth_probe_line()
    assert line.startswith("\n- auth probe: FAILED — RuntimeError: socket closed")


def test_probe_reports_ok_on_a_live_transport(monkeypatch):
    srv = _login_ok(monkeypatch)
    monkeypatch.setattr(srv, "sub_query", lambda *a, **k: sq.SubResult(0, "ok", 1, 1))
    assert "auth probe: ok" in srv._auth_probe_line()


def test_probe_is_skipped_when_the_free_check_already_knows_it_is_dead(monkeypatch):
    """Paying for a call whose answer is already known is the same waste the batch
    fail-fast exists to stop."""
    import src.server as srv

    monkeypatch.setattr(srv.transport, "cli_auth_status", lambda cfg: {"loggedIn": False})
    called = []
    monkeypatch.setattr(srv, "sub_query", lambda *a, **k: called.append(1))
    assert "SKIPPED" in srv._auth_probe_line()
    assert called == [], "spent a model call it already knew would fail"


def test_status_names_the_fix_when_the_cli_is_not_logged_in(monkeypatch):
    import src.server as srv

    monkeypatch.setattr(srv, "resolve_auth_mode_safe", lambda: "oauth")
    monkeypatch.setattr(srv.transport, "cli_auth_status", lambda cfg: {"loggedIn": False})
    line = srv._cli_login_line()
    assert "NOT LOGGED IN" in line
    assert "claude auth login" in line and "claude setup-token" in line


def test_status_says_nothing_about_cli_login_on_the_sdk_path(monkeypatch):
    import src.server as srv

    monkeypatch.setattr(srv, "resolve_auth_mode_safe", lambda: "apikey")
    assert srv._cli_login_line() == "", "no CLI is involved on the API-key path"


def test_all_chunks_failing_is_reported_as_a_failed_tool_call(monkeypatch):
    """100% failure is not a result with notes. Returning the ordinary success
    string here is how a dead login reads back as 'no findings'."""
    import src.server as srv

    class _Meta:
        chunks = [{"i": 0}, {"i": 1}]

    class _Store:
        def get(self, ctx_id):
            return _Meta()

        def read_chunk(self, ctx_id, i):
            return f"chunk{i}"

    monkeypatch.setattr(srv, "STORE", _Store())
    monkeypatch.setattr(srv, "sub_query_batch", lambda *a, **k: [
        sq.SubResult(0, "", 0, 0, error="OAuth session expired"),
        sq.SubResult(1, "", 0, 0, error="skipped — OAuth session expired"),
    ])
    out = srv.rlm_sub_query_batch("ctx_x", "label everything")
    assert out.lstrip().startswith("ERROR"), out[:200]
    assert "all 2 chunk(s) failed" in out


def test_engine_and_wrapper_share_one_fatal_attribute():
    """The engine aborts its own fan-out on the same flag our transport sets.

    If these ever drift, rlm_query silently goes back to spawning N doomed
    sub-calls while rlm_sub_query_batch correctly stops at one.
    """
    from rlm.utils.exceptions import FATAL_SUBCALL_ATTR, aborts_batch

    assert FATAL_SUBCALL_ATTR == "is_fatal_subcall"
    assert getattr(CliAuthError, FATAL_SUBCALL_ATTR, False) is True
    assert aborts_batch(CliAuthError("dead login"))
    assert not aborts_batch(CliCompletionError("just this prompt"))
    assert not aborts_batch(CliRateLimitError("429"))


def test_lm_handler_batch_stops_on_a_session_fatal_failure():
    """core/lm_handler.py fans out llm_query_batched with concurrency 16.

    This is the fourth copy of the fan-out and the one with the highest concurrency;
    without the guard a dead login produced N identical doomed calls, 16 at a time.
    """
    from rlm.core.comms_utils import LMRequest
    from rlm.core.lm_handler import LMRequestHandler

    calls = []

    class _Client:
        model_name = "m"

        async def acompletion(self, prompt):
            calls.append(prompt)
            raise CliAuthError("OAuth session expired")

        def get_last_usage(self):
            return {}

    class _Handler:
        batch_max_concurrent = 16

        def get_client(self, model=None, depth=0):
            return _Client()

    req = LMRequest(prompts=[f"p{i}" for i in range(40)])
    # bound method off the class: the handler never touches socket state here
    res = LMRequestHandler._handle_batched(None, req, _Handler())

    assert len(calls) <= 16, f"fanned out {len(calls)} doomed calls, expected <= 16"
    assert len(calls) < 40, "no abort happened at all"
    comps = res.chat_completions
    assert len(comps) == 40, "every prompt must still get a result slot"
    assert any("skipped" in (c.error or "") for c in comps), "skips not surfaced"


def test_lm_handler_batch_does_not_abort_on_a_local_failure():
    """A prompt-specific failure must not discard the prompts that would succeed."""
    from rlm.core.comms_utils import LMRequest
    from rlm.core.lm_handler import LMRequestHandler

    calls = []

    class _Client:
        model_name = "m"

        async def acompletion(self, prompt):
            calls.append(prompt)
            raise CliCompletionError("just this prompt")

        def get_last_usage(self):
            return {}

    class _Handler:
        batch_max_concurrent = 16

        def get_client(self, model=None, depth=0):
            return _Client()

    req = LMRequest(prompts=[f"p{i}" for i in range(20)])
    LMRequestHandler._handle_batched(None, req, _Handler())
    assert len(calls) == 20, "a chunk-local failure aborted the batch"
