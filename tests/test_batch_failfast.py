"""A dead login is one failure, not N.

Covers the path that let 20 chunks each spawn their own doomed ``claude -p``:
transport has to flag an auth failure distinctly, ratelimit has to classify it as
fatal, and sub_query_batch has to stop fanning out once it has seen one — without
throwing away chunks that fail for their own, local reasons.
"""

import dataclasses
import json

import pytest

import src.batch as bt
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


def test_the_shared_context_stub_still_matches_the_real_ContextMeta(batch_ctx):
    """The stub must expose every field the batch path reads off a real context.

    This is the drift that cost real time: `chunk_strategy` had always been on
    ContextMeta, but the hand-copied stubs predated the code path that read it — so the
    tools failed with `'_Meta' object has no attribute 'chunk_strategy'` in eight places
    at once. Now there is one stub, and this asserts its surface against the real
    dataclass, so the next field added to that path fails HERE with a clear message
    rather than inside a tool call.
    """
    from src.context_store import ContextMeta

    d, _events = batch_ctx(2)
    stub = d.store.get("ctx_x")
    for field in ("chunks", "chunk_strategy"):
        assert hasattr(stub, field), f"the shared stub is missing {field!r}"
        assert field in ContextMeta.__dataclass_fields__, (
            f"{field!r} is stubbed but no longer exists on ContextMeta - the stub is stale"
        )


def test_auth_failure_parses_to_its_own_error_type():
    with pytest.raises(CliAuthError):
        _parse_cli_output(0, _AUTH_PAYLOAD, "", "claude-haiku-4-5")


def test_only_auth_counts_as_fatal():
    assert is_fatal_auth(CliAuthError("dead login"))
    assert not is_fatal_auth(CliCompletionError("boom"))
    assert not is_fatal_auth(CliRateLimitError("429"))


def _run_batch(monkeypatch, cfg, exc, n=10):
    calls: list[str] = []

    def fake_call(cfg, model, prompt, max_tokens, system):
        calls.append(prompt)
        raise exc

    monkeypatch.setattr(sq, "_call", fake_call)
    results = sq.sub_query_batch(cfg, [f"p{i}" for i in range(n)], "m", concurrency=1)
    return calls, results


def test_auth_failure_stops_the_fan_out(monkeypatch, cfg):
    calls, results = _run_batch(monkeypatch, cfg, CliAuthError("OAuth session expired"))
    assert len(calls) == 1, "chunks after the first re-ran a call already known dead"
    assert len(results) == 10 and all(r.error for r in results)
    assert results[-1].error.startswith("skipped —")


def test_ordinary_failure_does_not_discard_the_other_chunks(monkeypatch, cfg):
    calls, _ = _run_batch(monkeypatch, cfg, CliCompletionError("just this chunk"))
    assert len(calls) == 10, "a chunk-local failure must not abort the whole batch"


# --- lazy prompt builders (Leaf 3 / option D) -------------------------------

def _counting_builder(build_counts: list[int], i: int):
    def _build() -> str:
        build_counts.append(i)
        return f"p{i}"
    return _build


def test_batch_builds_prompts_lazily_one_per_worker(monkeypatch, cfg, batch_ctx):
    calls: list[str] = []
    monkeypatch.setattr(sq, "_call", lambda cfg, model, prompt, max_tokens, system: (
        calls.append(prompt), ("ok", 1, 1, "m"))[1])

    build_counts: list[int] = []
    builders = [_counting_builder(build_counts, i) for i in range(5)]
    assert build_counts == [], "constructing the builders must not build any prompt"

    results = sq.sub_query_batch(cfg, builders, "m", concurrency=2)
    assert sorted(build_counts) == list(range(5)), "every item should be built exactly once"
    assert len(calls) == 5
    assert all(not r.error for r in results)


def test_batch_reports_prompt_build_failure_without_aborting(monkeypatch, cfg, batch_ctx):
    calls: list[str] = []
    monkeypatch.setattr(sq, "_call", lambda cfg, model, prompt, max_tokens, system: (
        calls.append(prompt), ("ok", 1, 1, "m"))[1])

    def boom() -> str:
        raise ValueError("cannot build")

    results = sq.sub_query_batch(cfg, ["p0", boom, "p2"], "m", concurrency=1)
    assert "prompt build failed" in results[1].error
    assert results[0].answer == "ok" and results[2].answer == "ok"
    assert calls == ["p0", "p2"], "the failing builder must not reach _call"


def test_fatal_auth_skip_happens_before_building(monkeypatch, cfg, batch_ctx):
    build_counts: list[int] = []
    builders = [_counting_builder(build_counts, i) for i in range(10)]
    monkeypatch.setattr(sq, "_call", lambda *a: (_ for _ in ()).throw(
        CliAuthError("OAuth session expired")))

    results = sq.sub_query_batch(cfg, builders, "m", concurrency=1)
    assert build_counts == [0], "only the item already in flight when the flag tripped should build"
    assert results[-1].error.startswith("skipped —")


# --- rlm_status(probe=True): the line that separates "configured" from "working" ---
def _login_ok(monkeypatch):
    """Pin the free login check to 'logged in' so these tests exercise the probe
    itself, not whatever auth state the machine running them happens to be in.

    The budget pre-check is pinned for the same reason and it matters more: it reads the
    SERVER's Deps, whose ledger is the operator's real ~/.rlm/usage.jsonl. Left alone,
    every probe assertion below would quietly depend on how much the person running the
    suite had spent in the last five hours.
    """
    import src.server as srv

    monkeypatch.setattr(srv.transport, "cli_auth_status",
                        lambda cfg: {"loggedIn": True, "authMethod": "oauth"})
    monkeypatch.setattr(srv.budget, "check_or_raise", lambda *a, **k: None)
    return srv


def test_probe_reports_failure_instead_of_raising(monkeypatch, batch_ctx):
    srv = _login_ok(monkeypatch)
    monkeypatch.setattr(srv, "sub_query",
                        lambda *a, **k: sq.SubResult(0, "", 0, 0, error="OAuth session expired"))
    assert "FAILED — OAuth session expired" in srv._auth_probe_line()


def test_probe_survives_a_transport_that_explodes(monkeypatch, batch_ctx):
    srv = _login_ok(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(srv, "sub_query", boom)
    line = srv._auth_probe_line()
    assert line.startswith("\n- auth probe: FAILED — RuntimeError: socket closed")


def test_probe_reports_ok_on_a_live_transport(monkeypatch, batch_ctx):
    srv = _login_ok(monkeypatch)
    seen: list[str | None] = []
    monkeypatch.setattr(srv, "sub_query", lambda *a, **k: (
        seen.append(k.get("system")) or sq.SubResult(0, "ok", 1, 1)))
    assert "auth probe: ok" in srv._auth_probe_line()
    # "Reply with exactly: ok" is the same one-line contract the batch needed a system
    # prompt to hold, and the probe truncates the answer to 20 chars either way -- so the
    # padding was invisible here rather than absent.
    assert seen == [bt.TERSE_SYSTEM], "the probe ran under the default persona"


def test_probe_is_skipped_when_the_free_check_already_knows_it_is_dead(monkeypatch, batch_ctx):
    """Paying for a call whose answer is already known is the same waste the batch
    fail-fast exists to stop."""
    import src.server as srv

    monkeypatch.setattr(srv.transport, "cli_auth_status", lambda cfg: {"loggedIn": False})
    called = []
    monkeypatch.setattr(srv, "sub_query", lambda *a, **k: called.append(1))
    assert "SKIPPED" in srv._auth_probe_line()
    assert called == [], "spent a model call it already knew would fail"


def test_probe_blames_the_budget_not_the_login_when_the_floor_refuses_it(monkeypatch,
                                                                        batch_ctx):
    """Past the stop line the transport refuses the probe before dispatch, and the refusal
    was printed as "auth probe: FAILED — session budget stop", sending an operator to
    re-authenticate a login that was fine. rlm_status is what people reach for when
    something looks broken, so it must not misname what is broken."""
    import src.budget as budget
    import src.server as srv

    monkeypatch.setattr(srv.transport, "cli_auth_status",
                        lambda cfg: {"loggedIn": True, "authMethod": "oauth"})

    def _refuse(*a, **k):
        raise budget.BudgetStopError(spent=5_600_000, usable=5_510_000, next_call=16)

    monkeypatch.setattr(srv.budget, "check_or_raise", _refuse)
    called = []
    monkeypatch.setattr(srv, "sub_query", lambda *a, **k: called.append(1))

    line = srv._auth_probe_line()

    assert "SKIPPED" in line and "budget" in line.lower()
    assert "FAILED" not in line, line
    assert called == [], "spent a call the floor had already refused"


def test_status_names_the_fix_when_the_cli_is_not_logged_in(monkeypatch, batch_ctx):
    import src.server as srv

    monkeypatch.setattr(srv, "resolve_auth_mode_safe", lambda: "oauth")
    monkeypatch.setattr(srv.transport, "cli_auth_status", lambda cfg: {"loggedIn": False})
    line = srv._cli_login_line()
    assert "NOT LOGGED IN" in line
    assert "claude auth login" in line and "claude setup-token" in line


def test_status_says_nothing_about_cli_login_on_the_sdk_path(monkeypatch, batch_ctx):
    import src.server as srv

    monkeypatch.setattr(srv, "resolve_auth_mode_safe", lambda: "apikey")
    assert srv._cli_login_line() == "", "no CLI is involved on the API-key path"


def test_all_chunks_failing_is_reported_as_a_failed_tool_call(monkeypatch, batch_ctx):
    """100% failure is not a result with notes. Returning the ordinary success
    string here is how a dead login reads back as 'no findings'."""
    d, events = batch_ctx(2)

    monkeypatch.setattr(bt, "sub_query_batch", lambda *a, **k: [
        sq.SubResult(0, "", 0, 0, error="OAuth session expired"),
        sq.SubResult(1, "", 0, 0, error="skipped — OAuth session expired"),
    ])
    out = bt.run(d, "ctx_x", "label everything")
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


def test_map_start_is_logged_before_the_fan_out(monkeypatch, batch_ctx):
    """A 408-chunk batch runs for tens of minutes. Logging the batch only when it
    RETURNS leaves an in-flight run as N indistinguishable cli_spawn lines with no
    ctx_id and no denominator — indistinguishable from hung. The record has to land
    before sub_query_batch is entered, not after it comes back.
    """
    d, events = batch_ctx(3)


    seen_at_fan_out: list[list[str]] = []

    def fake_batch(cfg, prompts, model, concurrency=1, **kw):
        seen_at_fan_out.append([f.get("phase") for e, f in events if e == "sub_batch"])
        return [sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))]

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_x", "audit every file", max_chunks=2, reduce=False)

    assert seen_at_fan_out == [["map_start"]], "map_start did not land before the fan-out"
    phase, fields = next((e, f) for e, f in events if f.get("phase") == "map_start")
    # ctx_id and the denominator are the two things the cli_spawn lines cannot supply.
    assert fields["ctx_id"] == "ctx_x"
    assert (fields["chunks"], fields["total_chunks"]) == (2, 3)


def test_the_map_fans_out_with_the_output_contract_in_the_system_slot(monkeypatch, batch_ctx):
    """Every sub-model call used to run with NO system prompt at all. `sub_query_batch`
    and both transports have always accepted one and no caller ever passed it, so each
    chunk was answered by the claude CLI's default coding-assistant persona -- which
    explains its work. Measured: 328,453 output tokens on a 33-chunk run capped at 2048,
    4.9x the cap (docs/07 §10). The contract has to travel in the system slot; in the user
    turn it sits ~10K tokens before the generation point, the weakest position it can hold.
    """
    d, _ = batch_ctx(2)
    seen: list[str | None] = []

    def fake_batch(cfg, prompts, model, concurrency=1, **kw):
        seen.append(kw.get("system"))
        return [sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))]

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_x", "audit every file", reduce=False)

    assert seen == [bt.MAP_SYSTEM], "the map fanned out with no output contract"
    # A schema, not a plea: "output exactly X" is the instruction that already failed. The
    # EMPTY case must be shown rather than described -- it is the one a model resists, and
    # an example that is only ever filled teaches that filled is the goal.
    assert '{"findings": []}' in bt.MAP_SYSTEM


def test_a_single_sub_query_also_carries_the_terse_contract(monkeypatch, batch_ctx):
    """rlm_sub_query is the same default-persona exposure as the batch, one call wide --
    and it is uncapped in practice on the OAuth path, so a padded answer costs whatever
    the model felt like emitting. No envelope here: its answer goes straight to a reader.
    """
    d, _ = batch_ctx(2)
    seen: list[str | None] = []
    monkeypatch.setattr(bt, "sub_query", lambda cfg, prompt, model, **kw: (
        seen.append(kw.get("system")) or sq.SubResult(0, "answer", 1, 1)))

    bt.one(d, "ctx_x", "what is in here?", chunk_index=0)

    assert seen == [bt.TERSE_SYSTEM], "the single-query path ran under the default persona"


def test_a_changed_output_contract_does_not_reuse_the_old_contracts_answers(monkeypatch, batch_ctx):
    """The cache key must carry the system prompt the map ACTUALLY SENDS, not just the
    user's prompt. Key without it and every answer cached under the old no-system persona
    stays cache-valid under MAP_SYSTEM -- so a resumed run serves prose answers for a JSON
    contract, and with reduce=True folds prose and envelopes into one synthesis. This is
    the test the three narrower ones miss: they all still pass if `_scan_cache` keys with
    "" while the map sends the contract.
    """
    d, _ = batch_ctx(2)
    fanned: list[int] = []

    def fake_batch(cfg, prompts, model, concurrency=1, **kw):
        fanned.append(len(prompts))
        out = [sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))]
        for r in out:
            kw["on_result"](r)      # persist as production does, or nothing is cached
        return out

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_x", "audit", reduce=False)
    bt.run(d, "ctx_x", "audit", reduce=False)
    assert fanned == [2], "the second run re-asked chunks it had already paid for"

    # Same chunks, same prompt, same model -- only the contract moved. Every answer on
    # disk was produced under the other one and none of it is reusable.
    monkeypatch.setattr(bt, "MAP_SYSTEM", bt.MAP_SYSTEM + "\nAlso: be brief.")
    bt.run(d, "ctx_x", "audit", reduce=False)
    assert fanned == [2, 2], "answers from the OLD output contract were reused under a new one"


def test_the_synthesis_gets_terseness_without_the_envelope(monkeypatch, batch_ctx):
    """`_reduce` exists to produce ONE coherent prose answer, rendered straight to the
    reader. Handing it the map's findings-array contract would destroy that deliverable,
    so only the map -- which fans out N-wide and is read by a program -- gets the schema.
    """
    d, _ = batch_ctx(2)
    monkeypatch.setattr(
        bt, "sub_query_batch", lambda cfg, prompts, model, concurrency=1, **kw: [
            sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))])
    seen: list[str | None] = []

    def fake_reduce(cfg, prompt, model, **kw):
        seen.append(kw.get("system"))
        return sq.SubResult(0, "synthesis", 1, 1)

    monkeypatch.setattr(bt, "sub_query", fake_reduce)
    bt.run(d, "ctx_x", "audit every file", reduce=True)

    assert seen == [bt.TERSE_SYSTEM], "the synthesis ran under the default persona"
    assert "findings" not in bt.TERSE_SYSTEM, "the map's envelope leaked into prose calls"


def _logged_batch(monkeypatch, batch_ctx, reduce, reduce_result=None):
    """Drive rlm_sub_query_batch over a 2-chunk stub context, returning the sub_batch
    records it logged. Each chunk reports 10 in / 3 out, so the totals are checkable."""
    d, events = batch_ctx(2)

    monkeypatch.setattr(bt, "sub_query_batch", lambda cfg, prompts, model, concurrency=1, **kw: [
        sq.SubResult(i, f"finding {i}", 10, 3) for i in range(len(prompts))])
    if reduce_result is not None:
        monkeypatch.setattr(bt, "sub_query", lambda *a, **k: reduce_result)
    bt.run(d, "ctx_x", "audit every file", reduce=reduce)
    return [f for e, f in events if e == "sub_batch"]


def test_a_clean_map_is_logged_with_its_chunk_and_token_counts(monkeypatch, batch_ctx):
    """The batch record used to be emitted only when a chunk ERRORED, so a fully
    successful run left no chunk count and no token totals anywhere — the parent
    tool_call carries neither, leaving N anonymous cli_spawn lines under one rid.
    """
    rec = next(f for f in _logged_batch(monkeypatch, batch_ctx, reduce=False) if f["phase"] == "map")
    assert (rec["chunks"], rec["errors"]) == (2, 0)
    assert (rec["itok"], rec["otok"]) == (20, 6)
    assert rec["err_sample"] is None
    assert rec["ctx_id"] == "ctx_x", "every sub_batch phase carries ctx_id, not just map_start"


def test_rlm_sub_query_batch_passes_builders_not_strings(monkeypatch, batch_ctx):
    """Leaf 3: the server must hand sub_query_batch lazy builders, not a materialized
    list of full prompt strings — that list is ~= the whole context held in RAM for
    the entire batch (the defect this plan fixes)."""
    d, events = batch_ctx(3)

    seen: list = []

    def fake_batch(cfg, prompts, model, concurrency=1, **kw):
        seen.extend(prompts)
        return [sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))]

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_x", "audit every file", reduce=False)

    assert all(callable(p) for p in seen), "prompts must be lazy builders, not strings"
    assert [p() for p in seen] == [
        "audit every file\n\n--- CHUNK 1/3 ---\nchunk0",
        "audit every file\n\n--- CHUNK 2/3 ---\nchunk1",
        "audit every file\n\n--- CHUNK 3/3 ---\nchunk2",
    ]


def test_max_chunks_caps_how_many_prompts_get_built(monkeypatch, batch_ctx):
    """The other half of laziness: with max_chunks set, only the SELECTED chunks may be
    built -- and none of them before the pool runs."""
    builds: list[int] = []
    d, _events = batch_ctx(3, on_read=builds.append)

    seen: list = []

    def fake_batch(cfg, prompts, model, concurrency=1, **kw):
        # The answer cache is keyed by chunk bytes, so run() now reads and hashes each
        # SELECTED chunk before the pool -- one at a time, discarded at once. The memory
        # property this test guards is unchanged and asserted directly: no prompt STRING
        # exists before the pool, and a chunk max_chunks excluded is never read at all.
        assert all(callable(p) for p in prompts), "prompts must reach the pool as lazy builders"
        assert 2 not in builds, "a chunk beyond max_chunks was read"
        seen.extend(prompts)
        return [sq.SubResult(i, f"finding {i}", 1, 1) for i in range(len(prompts))]

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_x", "audit", max_chunks=2, reduce=False)

    assert len(seen) == 2, "max_chunks must cap the builder list, not just the results"
    # Each selected chunk is read twice (cache scan, then the lazy build); the set is what
    # proves max_chunks bounded both passes.
    assert [p() for p in seen] and sorted(set(builds)) == [0, 1], "built a chunk max_chunks excluded"


def test_a_failed_reduce_falls_back_to_the_raw_findings(monkeypatch, batch_ctx):
    """A reduce call that fails must not discard the map work that succeeded.

    The whole point is the fallback: the map spent N model calls, and a failed synthesis
    on top of them is no reason to return nothing. Mutation-checked -- replacing
    ``raw(...)`` with a bare error string fails this test.

    The token assertion here is weaker than it looks and is NOT a guard on the
    itok/otok bookkeeping: a failed sub_query reports 0/0, so billing it anyway is a
    no-op with real inputs (verified by mutation: removing the `if not red.error` guard
    breaks nothing). That guard is defensive, not load-bearing.
    """
    d, events = batch_ctx(2)

    monkeypatch.setattr(bt, "sub_query_batch", lambda cfg, prompts, model, concurrency=1, **kw: [
        sq.SubResult(i, f"finding {i}", 10, 3) for i in range(len(prompts))])
    monkeypatch.setattr(bt, "sub_query",
                        lambda *a, **k: sq.SubResult(0, "", 0, 0, error="reduce blew up"))

    out = bt.run(d, "ctx_x", "audit every file", reduce=True)

    assert "reduce pass failed: reduce blew up" in out
    assert "finding 0" in out and "finding 1" in out, "map findings were discarded"
    assert "tokens: 20 in / 6 out" in out, "a failed reduce must not bill its own tokens"
    rec = next(f for e, f in events if f.get("phase") == "reduce")
    assert (rec["itok"], rec["otok"]) == (20, 6)
    assert rec["reduce_error"] == "reduce blew up"


def test_a_failed_reduce_after_a_deferred_chunk_promises_no_synthesis(monkeypatch, batch_ctx):
    """The partial-run note was appended to the same `note` the raw fallback had already
    closed over, so a reduce that then failed handed back per-chunk findings under the
    line "The synthesis below covers 1 of 2 chunks" — with no synthesis below it."""
    d, _events = batch_ctx(2)

    monkeypatch.setattr(bt, "sub_query_batch", lambda cfg, prompts, model, concurrency=1, **kw: [
        sq.SubResult(0, "finding 0", 10, 3),
        sq.SubResult(1, "", 0, 0, error="deferred — session budget reached"),
    ])
    monkeypatch.setattr(bt, "sub_query",
                        lambda *a, **k: sq.SubResult(0, "", 0, 0, error="reduce blew up"))

    out = bt.run(d, "ctx_x", "audit every file", reduce=True)

    assert "reduce pass failed" in out
    assert "finding 0" in out, "the map findings must survive a failed synthesis"
    assert "synthesis below" not in out, (
        "the raw fallback promised a synthesis that is not in the reply:\n" + out[:400])


def test_a_batch_still_runs_when_only_its_biggest_chunk_will_not_fit(monkeypatch, batch_ctx):
    """The headroom refusal compared the LARGEST remaining call and then announced that
    not even one chunk fitted. Under `files` chunking a 200-token file sits beside a huge
    one, so any headroom between them threw away every chunk that would have fitted. The
    Gate admits what fits and defers the rest; that is the honest stop."""
    import src.budget as budget

    d, _events = batch_ctx(2)

    class _Mixed:
        chunks = [{"i": 0, "est_tokens": 200}, {"i": 1, "est_tokens": 300_000}]
        chunk_strategy = "files"

    d.store.get = lambda ctx_id: _Mixed()
    d = dataclasses.replace(d, cfg=dataclasses.replace(d.cfg,
                                                       session_budget_tokens=1_000_000))
    budget.record(d.cfg, "m", 900_000, 0)        # headroom 50k: fits the small, not the big

    fanned = []
    monkeypatch.setattr(bt, "sub_query_batch",
                        lambda cfg, prompts, model, concurrency=1, **kw: (
                            fanned.append(len(prompts)) or [sq.SubResult(0, "ok", 1, 1)]))

    out = bt.run(d, "ctx_x", "audit every file", reduce=False)

    assert "not enough headroom for even one chunk" not in out, out[:300]
    assert fanned, "the fan-out never ran, so the chunks that fitted were thrown away"


def test_a_reduce_too_big_for_any_window_says_so_instead_of_blaming_the_chunks(batch_ctx):
    """Both refusals used to read "a single chunk can never fit — re-chunk smaller". When
    it is the synthesis that cannot fit, re-chunking makes it WORSE (more answers to fold
    in); the remedy is reduce=False, which costs no chunk answers at all."""
    d, _events = batch_ctx(400, est_tokens=1_000)
    d = dataclasses.replace(d, cfg=dataclasses.replace(d.cfg,
                                                       session_budget_tokens=500_000))

    out = bt.run(d, "ctx_x", "audit every file", reduce=True)

    assert out.lstrip().startswith("ERROR")
    assert "reduce=False" in out, out[:300]
    assert "re-chunk smaller" not in out, "blamed the chunks for the synthesis call"


def test_a_successful_reduce_is_logged_with_the_batch_total(monkeypatch, batch_ctx):
    """A reduce that SUCCEEDS still spends a call and tokens. Logging only the failure
    left those in no record, so the map totals understated every reduce=True batch."""
    recs = _logged_batch(monkeypatch, batch_ctx, reduce=True,
                         reduce_result=sq.SubResult(0, "synthesis", 5, 2))
    rec = next(f for f in recs if f["phase"] == "reduce")
    assert (rec["itok"], rec["otok"]) == (25, 8), "map 20/6 + reduce 5/2"
    assert rec["reduce_error"] is None
    assert rec["ctx_id"] == "ctx_x"


# --- the three report branches the coverage report showed uncovered ----------
# All are "the batch partly worked" paths: the map spent real model calls, so each must
# still hand back the per-chunk findings rather than an error or an empty answer.

def _batch_over(monkeypatch, batch_ctx, results, *, n_chunks=3, reduce=True, sub_context_tokens=None):
    """Drive rlm_sub_query_batch over a stub context whose map returns `results`."""
    d, events = batch_ctx(n_chunks)

    monkeypatch.setattr(bt, "sub_query_batch",
                        lambda cfg, prompts, model, concurrency=1, **kw: results)
    monkeypatch.setattr(bt, "sub_query",
                        lambda *a, **k: sq.SubResult(0, "synthesis", 1, 1))
    if sub_context_tokens is not None:
        # Config is frozen, so build the variant and hand it in -- no global to patch.
        d = dataclasses.replace(
            d, cfg=dataclasses.replace(d.cfg, sub_context_tokens=sub_context_tokens))
    return bt.run(d, "ctx_x", "audit every file", reduce=reduce)


def test_a_partly_failed_map_notes_the_skipped_chunks_and_keeps_the_rest(monkeypatch, batch_ctx):
    """One chunk erroring must not cost the others. The count in the note is what tells a
    reader the answer is partial -- without it a 2-of-3 result reads as complete."""
    out = _batch_over(monkeypatch, batch_ctx, [
        sq.SubResult(0, "finding 0", 10, 3),
        sq.SubResult(1, "", 0, 0, error="rate limited"),
        sq.SubResult(2, "finding 2", 10, 3),
    ], reduce=False)

    assert "1 of 3 chunk(s) errored and were skipped" in out
    assert "finding 0" in out and "finding 2" in out, "successful chunks were discarded"
    assert "[ERROR: rate limited]" in out, "the failed chunk is shown, not hidden"


def test_a_map_that_produced_no_findings_shows_the_raw_report(monkeypatch, batch_ctx):
    """Every chunk answered, but every answer was blank: there is nothing to reduce, and
    spending another model call on nothing is waste. Says so instead of returning empty."""
    out = _batch_over(monkeypatch, batch_ctx, [
        sq.SubResult(0, "   ", 10, 3),
        sq.SubResult(1, "", 10, 3),
    ], n_chunks=2, reduce=True)

    assert "(no findings to reduce)" in out
    assert "map over 2 chunks" in out, "should fall back to the raw map report"


def test_findings_too_large_to_reduce_fall_back_to_raw_instead_of_being_truncated(monkeypatch, batch_ctx):
    """The reduce pass sends every finding in one prompt, so findings bigger than the
    sub-model's window cannot be reduced at all. Returning raw beats a call certain to
    fail with "prompt is too long", and the note names the two ways out."""
    big = "x" * 4000
    out = _batch_over(monkeypatch, batch_ctx, [
        sq.SubResult(0, big, 10, 3),
        sq.SubResult(1, big, 10, 3),
    ], n_chunks=2, reduce=True, sub_context_tokens=100)

    assert "findings too large to reduce in one pass" in out
    assert "fewer/larger chunks" in out and "rlm_query" in out, "must name the way out"
    assert big in out, "the raw findings are still returned"


def test_a_fully_cached_batch_is_still_gated_on_the_synthesis_it_runs(monkeypatch, batch_ctx):
    """A resumed run re-pays for no chunk, but reduce=True still sends every cached answer
    back as one synthesis call. Returning early on "nothing left to map" put that call
    ahead of every budget check — the one call in such a run, and nothing saw it."""
    d, _events = batch_ctx(2)

    def _map(cfg, prompts, model, concurrency=1, **kw):
        """Persists through on_result the way the real pool does — without that the
        second run finds nothing cached and this test proves nothing."""
        out = [sq.SubResult(0, "finding 0", 10, 3), sq.SubResult(1, "finding 1", 10, 3)]
        for r in out:
            if kw.get("on_result") is not None:
                kw["on_result"](r)
        return out

    monkeypatch.setattr(bt, "sub_query_batch", _map)
    reduce_calls = []

    def _synth(*a, **k):
        reduce_calls.append(1)
        return sq.SubResult(0, "synthesis", 1, 1)

    monkeypatch.setattr(bt, "sub_query", _synth)

    first = bt.run(d, "ctx_x", "audit every file", reduce=True)
    assert "synthesis" in first and len(reduce_calls) == 1, "setup: the first run must map"

    # Same prompt, same chunk bytes, so every answer is now on disk and there is no map
    # work left. This ceiling cannot fit the synthesis over those answers.
    tight = dataclasses.replace(d, cfg=dataclasses.replace(
        d.cfg, session_budget_tokens=1_000, budget_stop_fraction=0.95))
    out = bt.run(tight, "ctx_x", "audit every file", reduce=True)

    assert out.startswith("ERROR"), f"a fully cached run spent unchecked: {out[:200]}"
    assert len(reduce_calls) == 1, "the synthesis was issued without passing the budget"
    assert "synthesis over 2 cached chunk(s)" in out, \
        "the refusal must name what was refused, not '0 chunk(s)'"



def test_the_transport_floor_defers_the_batch_instead_of_failing_it(monkeypatch, cfg):
    """If the floor fires under a worker (the Gate normally stops first), the chunk and
    every one after it are DEFERRED -- scheduled work with answers safe on disk -- and
    the fan-out stops. Reporting them as errors would tell the operator to investigate
    a healthy stop, and reissuing the doomed call per chunk is the waste fail-fast exists
    to prevent."""
    from src.budget import BudgetStopError

    calls, results = _run_batch(monkeypatch, cfg, BudgetStopError(spent=1, usable=1, next_call=1))
    assert len(calls) == 1, "the floor was re-hit once per remaining chunk"
    assert len(results) == 10
    assert all(r.error == "deferred — session budget reached" for r in results), \
        [r.error for r in results]


def test_the_gate_reserves_what_a_call_emits_not_the_cap_the_cli_discards(monkeypatch, cfg):
    """The Gate reserves before admitting each chunk. Reserving max_tokens — which the CLI
    never receives, so a call can emit several times it — let the Gate admit work the
    window could not pay for, running past the line it exists to stop short of."""
    import dataclasses
    import src.budget as budget

    cfg = dataclasses.replace(cfg, session_budget_tokens=90_000, budget_stop_fraction=0.95)
    for _ in range(8):                       # measured mean output: 10,000 per call
        budget.record(cfg, "m", 0, 10_000)
    calls: list[str] = []

    def fake_call(cfg_, model, prompt, max_tokens, system):
        calls.append(prompt)
        return "ok", 1, 1, "m"

    monkeypatch.setattr(sq, "_call", fake_call)
    # Spend 80,000 against an 85,500 line. Reserving the cap projects 82,048 and admits;
    # reserving what the path emits projects 90,000 and does not.
    results = sq.sub_query_batch(cfg, ["p0", "p1"], "m", concurrency=1,
                                 max_tokens=2048, gate=budget.Gate(cfg))

    assert calls == [], "the Gate admitted a call it had reserved only the cap for"
    assert all(r.error == "deferred — session budget reached" for r in results), \
        [r.error for r in results]
