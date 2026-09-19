"""What ran, and what authenticated it — reported rather than assumed.

Two failure modes this covers. A sandbox container outlives an image rebuild and
nothing detected it (observed: a 3.11 container serving after the image moved to
3.13, with rlm_status printing only the image NAME). And the model that ANSWERS is
not always the one asked for: on OAuth, models.select maps a configured id to its
closest subscription sibling, so the only way to know is to read it back.
"""

import types

import pytest

import better_rlm.server as srv
import better_rlm.subquery as sq
import better_rlm.transport as tp
from better_rlm.sandbox_reap import container_image_status


def _docker(monkeypatch, images_out, inspect_out):
    def run(argv, **kw):
        out = images_out if argv[1] == "images" else inspect_out
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    monkeypatch.setattr("better_rlm.sandbox_reap.subprocess.run", run)


def test_container_on_the_current_image_reads_current(monkeypatch):
    _docker(monkeypatch, "sha256:aaaa111122223333\n", "sha256:aaaa111122223333")
    assert container_image_status("img", "cid").startswith("current")


def test_container_from_an_older_build_is_reported_stale(monkeypatch):
    """The whole point: nothing else in the system notices this."""
    _docker(monkeypatch, "sha256:bbbb444455556666\n", "sha256:aaaa111122223333")
    msg = container_image_status("img", "cid")
    assert msg.startswith("STALE")
    assert "aaaa11112222" in msg and "bbbb44445555" in msg
    assert "reconnect" in msg


def test_no_container_is_not_an_error(monkeypatch):
    assert container_image_status("img", None) == "none created yet"


def test_docker_failure_degrades_to_unknown(monkeypatch):
    def boom(*a, **k):
        raise OSError("no docker")
    monkeypatch.setattr("better_rlm.sandbox_reap.subprocess.run", boom)
    assert container_image_status("img", "cid").startswith("unknown")


# --- auth label ---------------------------------------------------------------
def test_auth_label_names_the_method_and_is_cached(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("better_rlm.auth.resolve_auth_mode", lambda cfg: "oauth")
    calls = []

    def status(cfg):
        calls.append(1)
        return {"loggedIn": True, "authMethod": "oauth_token"}

    monkeypatch.setattr(tp, "cli_auth_status", status)
    cfg = srv.CFG
    assert tp.auth_label(cfg) == "oauth (claude CLI, oauth_token)"
    tp.auth_label(cfg)
    tp.auth_label(cfg)
    assert len(calls) == 1, "auth_label must not pay the 215ms subprocess per call"


def test_auth_label_says_so_when_the_login_is_dead(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("better_rlm.auth.resolve_auth_mode", lambda cfg: "oauth")
    monkeypatch.setattr(tp, "cli_auth_status", lambda cfg: {"loggedIn": False})
    assert "NOT LOGGED IN" in tp.auth_label(srv.CFG)


def test_auth_label_on_the_sdk_path_never_shells_out(monkeypatch):
    monkeypatch.setattr(tp, "_AUTH_LABEL", None)
    monkeypatch.setattr("better_rlm.auth.resolve_auth_mode", lambda cfg: "apikey")
    monkeypatch.setattr(tp, "cli_auth_status",
                        lambda cfg: pytest.fail("no CLI is involved on the API-key path"))
    assert tp.auth_label(srv.CFG) == "apikey (anthropic SDK)"


# --- the model that actually answered ----------------------------------------
def test_sub_result_carries_the_model_the_transport_reported(monkeypatch, cfg):
    monkeypatch.setattr(sq, "_call", lambda *a, **k: ("hi", 1, 2, "claude-haiku-4-5", False))
    assert sq.sub_query(cfg, "p", "asked-for-id").model == "claude-haiku-4-5"
    assert sq.sub_query_batch(cfg, ["a", "b"], "asked-for-id", concurrency=1)[0].model == \
        "claude-haiku-4-5"


# --- a paid call that produced nothing --------------------------------------
def _sdk_response(blocks, stop_reason):
    """The shape the Anthropic SDK returns: typed content blocks plus a stop_reason."""
    content = [types.SimpleNamespace(type=t, text=v, thinking=v) for t, v in blocks]
    return types.SimpleNamespace(
        content=content, stop_reason=stop_reason,
        usage=types.SimpleNamespace(output_tokens=4096, input_tokens=10))


def test_a_reasoning_model_that_ran_out_of_budget_is_reported_truncated():
    """MiniMax-M2.7 emits `thinking` FIRST and it is billed as output. Measured over a
    12.9k-token log chunk: 13,015 output tokens, 99% thinking, 286 chars of answer. At
    the old 4096 cap the budget ran out mid-thought, so NO text block was ever emitted
    and the tool returned a blank answer under a confident header with a token receipt.
    stop_reason is the only thing separating that from having nothing to say."""
    res = tp._result_from_sdk_response(_sdk_response([("thinking", "x" * 900)], "max_tokens"),
                                       "MiniMax-M2.7")
    assert res.text == ""
    assert res.truncated


def test_a_complete_answer_is_not_flagged_and_thinking_is_never_the_answer():
    res = tp._result_from_sdk_response(
        _sdk_response([("thinking", "scratch"), ("text", "the answer")], "end_turn"),
        "MiniMax-M2.7")
    assert res.text == "the answer", "thinking is a scratchpad, not output"
    assert not res.truncated


@pytest.mark.parametrize("answer,expect_empty_wording", [("", True), ("partial list", False)])
def test_the_truncation_note_reaches_the_caller(answer, expect_empty_wording):
    """Rendered, not merely recorded. The defect was visible only at this surface."""
    import better_rlm.batch as batch
    from better_rlm.subquery import SubResult

    note = batch._cut_note(SubResult(0, answer, 10, 4096, truncated=True))
    assert "TRUNCATED at max_tokens" in note
    assert ("spent the whole output budget on reasoning" in note) is expect_empty_wording
    assert str(batch.SUB_MAX_TOKENS) in note.replace(",", ""), "name the cap that was hit"
    assert batch._cut_note(SubResult(0, answer, 10, 20)) == "", "silent when complete"


def test_the_sub_model_budget_clears_a_measured_reasoning_call():
    """13,015 output tokens on one real log chunk. A default below that truncates every
    sub-query over a real log, which is precisely what 4096 did."""
    from better_rlm.subquery import SUB_MAX_TOKENS

    assert SUB_MAX_TOKENS > 13_015


def test_rlm_status_does_not_shadow_the_transport_module(monkeypatch):
    """rlm_status had a LOCAL named `transport`, which shadowed the imported module
    and made transport.auth_label() raise 'str has no attribute auth_label' at
    runtime -- invisible to every unit test that did not call the tool."""
    monkeypatch.setattr(tp, "_AUTH_LABEL", "test-label")
    monkeypatch.setattr(srv, "_container_status", lambda: "test-container")
    out = srv.rlm_status()
    assert not out.lstrip().startswith("ERROR"), out[:200]
    assert "test-label" in out and "test-container" in out
