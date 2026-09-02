"""The session-window budget: ledger, ceiling, estimate, gate — and resumable results.

These cover the three defects behind one real incident: a 103-chunk batch that spent ~60%
of a 4-hour window, was interrupted, and returned nothing re-runnable. Each test names the
failure it prevents rather than the function it calls.
"""

import dataclasses
import json
import time

import pytest

import src.budget as budget
import src.results as results
from src.config import load_config


@pytest.fixture
def bcfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        store_dir=tmp_path / "contexts",
        budget_ledger=tmp_path / "usage.jsonl",
        budget_state=tmp_path / "budget.json",
        session_window_h=5.0,
        session_budget_tokens=0,
        budget_stop_fraction=0.95,
        subquery_concurrency=3,
    )


# --------------------------- ledger --------------------------- #
def test_spend_counts_only_the_current_rolling_window(bcfg):
    """The vendor limit rolls; spend from six hours ago is not spend now. Counting the
    whole file would report a permanently exhausted budget after one busy day."""
    bcfg.budget_ledger.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    bcfg.budget_ledger.write_text(
        json.dumps({"ts": now - 6 * 3600, "model": "m", "itok": 900_000, "otok": 0}) + "\n"
        + json.dumps({"ts": now - 60, "model": "m", "itok": 1_000, "otok": 500}) + "\n",
        encoding="utf-8")

    s = budget.spent(bcfg)
    assert s.tokens == 1_500, "a record older than the window was counted as current spend"
    assert s.calls == 1


def test_a_corrupt_ledger_line_is_skipped_not_raised(bcfg):
    """The ledger is advisory accounting. A torn line from a killed process must cost one
    record, never break the tool call that is trying to read it."""
    bcfg.budget_ledger.parent.mkdir(parents=True, exist_ok=True)
    bcfg.budget_ledger.write_text(
        json.dumps({"ts": time.time(), "model": "m", "itok": 10, "otok": 5}) + "\n"
        + '{"ts": 123, "itok": tru\n',  # truncated mid-write
        encoding="utf-8")

    assert budget.spent(bcfg).tokens == 15


def test_record_then_spend_round_trips(bcfg):
    budget.record(bcfg, "claude-haiku-4-5", 30_000, 2_048)
    assert budget.spent(bcfg).tokens == 32_048


# --------------------------- ceiling --------------------------- #
def test_an_unknown_ceiling_is_reported_as_unknown_not_unlimited(bcfg):
    """Anthropic exposes no quota endpoint, so "we don't know" is the honest default.
    Rendering it as unlimited is what lets a run walk into the wall."""
    cap, source = budget.ceiling(bcfg)
    assert cap is None and source == "unknown"

    text = budget.render(budget.judge(bcfg, budget.estimate_batch(
        bcfg, [1000] * 3, prompt="p", max_output_tokens=2048, reduce=False)), what="x")
    assert "unknown" in text.lower()
    assert "unlimited" not in text.lower()


def test_hitting_a_usage_limit_teaches_the_ceiling(bcfg):
    budget.record(bcfg, "m", 500_000, 100_000)
    budget.note_limit_hit(bcfg)

    cap, source = budget.ceiling(bcfg)
    assert cap == 600_000 and source == "learned"


def test_a_later_higher_wall_never_raises_a_learned_ceiling(bcfg):
    """A limit hit at 600K proves the ceiling is not 900K. Taking the newest observation
    would let one lucky window inflate the budget and re-authorise the original failure."""
    budget.record(bcfg, "m", 600_000, 0)
    budget.note_limit_hit(bcfg)
    budget.record(bcfg, "m", 300_000, 0)   # window now shows 900K
    budget.note_limit_hit(bcfg)

    assert budget.ceiling(bcfg)[0] == 600_000


def test_a_configured_ceiling_beats_a_learned_one(bcfg):
    budget.record(bcfg, "m", 100, 0)
    budget.note_limit_hit(bcfg)
    cfg = dataclasses.replace(bcfg, session_budget_tokens=2_000_000)

    assert budget.ceiling(cfg) == (2_000_000, "configured")


# --------------------------- estimate --------------------------- #
def test_the_estimate_prices_output_at_the_full_cap(bcfg):
    """Forecasting the answers as short is how a run that "should fit" does not. Output is
    priced at max_output_tokens per call: the estimate must bound the run, not guess it."""
    est = budget.estimate_batch(bcfg, [10_000] * 4, prompt="audit this",
                                max_output_tokens=2048, reduce=False)

    assert est.calls == 4
    assert est.output_tokens == 4 * 2048
    assert est.input_tokens > 40_000        # chunks plus the per-prompt frame


def test_reduce_adds_a_call_that_re_reads_every_map_answer(bcfg):
    """The reduce pass is not free and is not small — it sends every finding back up. A
    forecast that ignored it under-counted the tail of every default batch."""
    no_reduce = budget.estimate_batch(bcfg, [10_000] * 4, prompt="p",
                                      max_output_tokens=2048, reduce=False)
    with_reduce = budget.estimate_batch(bcfg, [10_000] * 4, prompt="p",
                                        max_output_tokens=2048, reduce=True)

    assert with_reduce.calls == no_reduce.calls + 1
    assert with_reduce.input_tokens >= no_reduce.input_tokens + no_reduce.output_tokens


def test_a_resumed_estimate_prices_only_the_gaps(bcfg):
    """Resume leaves GAPS, not a prefix: workers finish out of order, so chunks 0 and 3
    can be done while 1 and 2 are not. Treating `done` as a count would re-price the
    wrong chunks and mis-state the remaining cost."""
    est = budget.estimate_batch(bcfg, [10_000] * 5, prompt="p", max_output_tokens=2048,
                                reduce=False, done={0, 3})

    assert (est.chunks_total, est.chunks_todo, est.chunks_done) == (5, 3, 2)
    assert est.calls == 3


# --------------------------- verdict --------------------------- #
def test_a_run_too_big_for_one_window_is_scheduled_not_refused(bcfg):
    """The point of durable results: an oversized job is several sittings, not a dead end.
    Reporting it as impossible would push the user back to the all-or-nothing run."""
    cfg = dataclasses.replace(bcfg, session_budget_tokens=100_000)
    est = budget.estimate_batch(cfg, [50_000] * 20, prompt="p",
                                max_output_tokens=2048, reduce=False)
    v = budget.judge(cfg, est)

    assert not v.fits
    assert v.possible, "a resumable run must never be reported as impossible"
    assert v.windows_needed > 1
    assert "resumes" in budget.render(v, what="x")


def test_headroom_is_measured_to_the_stop_line_not_the_ceiling(bcfg):
    """95% is the budget. Reporting headroom to 100% hands back 5% the gate will not
    spend, so a run sized against it stops earlier than the estimate promised."""
    cfg = dataclasses.replace(bcfg, session_budget_tokens=100_000, budget_stop_fraction=0.95)
    budget.record(cfg, "m", 50_000, 0)
    v = budget.judge(cfg, budget.estimate_batch(cfg, [10], prompt="p",
                                                max_output_tokens=1, reduce=False))

    assert v.headroom_tokens == 45_000   # 95_000 usable - 50_000 spent


# --------------------------- gate --------------------------- #
def test_the_gate_closes_before_crossing_the_stop_line(bcfg):
    cfg = dataclasses.replace(bcfg, session_budget_tokens=10_000, budget_stop_fraction=0.95)
    gate = budget.Gate(cfg)

    assert gate.allow(9_000) is True
    assert gate.allow(1_000) is False, "the gate admitted a call that crosses the stop line"
    assert gate.closed


def test_a_closed_gate_stays_closed(bcfg):
    """Latching matters: with N concurrent workers a gate that reopened for a small call
    would dribble past the line instead of stopping the batch."""
    cfg = dataclasses.replace(bcfg, session_budget_tokens=10_000)
    gate = budget.Gate(cfg)
    gate.allow(9_600)

    assert gate.allow(1) is False


def test_a_gate_with_no_known_ceiling_never_closes(bcfg):
    """Unknown must not mean zero. Gating on an unknown ceiling would refuse every run on
    a machine that has simply never hit a limit."""
    gate = budget.Gate(bcfg)

    assert gate.limit is None
    assert all(gate.allow(10_000_000) for _ in range(3))


# --------------------------- resumable results --------------------------- #
def test_answers_survive_the_process_that_produced_them(bcfg):
    key = results.key_for("audit", "haiku", "lines", 10)
    results.append(bcfg, "ctx_1", key, results.Saved(3, "finding three", 100, 20, "haiku"))
    results.append(bcfg, "ctx_1", key, results.Saved(7, "finding seven", 100, 20, "haiku"))

    loaded = results.load(bcfg, "ctx_1", key)
    assert set(loaded) == {3, 7}
    assert loaded[3].answer == "finding three"


def test_rechunking_invalidates_cached_answers(bcfg):
    """Chunk 7 under a different strategy or chunk count is different TEXT. Re-using an
    answer across a re-chunk would silently answer about bytes nobody asked about."""
    k_lines = results.key_for("audit", "haiku", "lines", 10)
    results.append(bcfg, "ctx_1", k_lines, results.Saved(7, "old", 1, 1, "haiku"))

    assert results.load(bcfg, "ctx_1", results.key_for("audit", "haiku", "files", 10)) == {}
    assert results.load(bcfg, "ctx_1", results.key_for("audit", "haiku", "lines", 25)) == {}
    assert results.load(bcfg, "ctx_1", results.key_for("other", "haiku", "lines", 10)) == {}
    assert results.load(bcfg, "ctx_1", k_lines)[7].answer == "old"


def test_a_torn_final_line_costs_one_chunk_not_the_whole_resume(bcfg):
    """The exact crash this exists for: killed mid-append. If load() raised, the resume
    file would be unreadable and the run would start from zero — the original failure."""
    key = results.key_for("audit", "haiku", "lines", 3)
    results.append(bcfg, "ctx_1", key, results.Saved(0, "good", 1, 1, "haiku"))
    with open(results.path_for(bcfg, "ctx_1", key), "a", encoding="utf-8") as fh:
        fh.write('{"index": 1, "answer": "half writ')

    loaded = results.load(bcfg, "ctx_1", key)
    assert set(loaded) == {0}


def test_fresh_clears_only_the_matching_question(bcfg):
    k1 = results.key_for("q1", "haiku", "lines", 5)
    k2 = results.key_for("q2", "haiku", "lines", 5)
    results.append(bcfg, "ctx_1", k1, results.Saved(0, "a", 1, 1, "haiku"))
    results.append(bcfg, "ctx_1", k2, results.Saved(0, "b", 1, 1, "haiku"))

    assert results.clear(bcfg, "ctx_1", k1) is True
    assert results.load(bcfg, "ctx_1", k1) == {}
    assert results.load(bcfg, "ctx_1", k2)[0].answer == "b"


# --------------------------- end-to-end resume through the tool --------------------- #
def test_a_second_run_pays_only_for_the_chunks_it_is_missing(monkeypatch, batch_ctx):
    """THE headline fix. The incident was a 30-minute run that returned nothing and would
    have cost the same again. Here the second call must make model calls only for what
    the first did not finish."""
    import src.subquery as sq
    srv, _events = batch_ctx(4)
    asked: list[list[int]] = []

    def fake_batch(prompts, model, concurrency=1, indices=None, on_result=None, **kw):
        asked.append(list(indices))
        out = [sq.SubResult(i, f"finding {i}", 10, 5, model=model) for i in indices]
        for r in out:                     # the real batch persists as answers land
            on_result(r)
        return out

    monkeypatch.setattr(srv, "sub_query_batch", fake_batch)
    srv.rlm_sub_query_batch("ctx_1", "audit every file", reduce=False)
    assert asked == [[0, 1, 2, 3]], "the first run should ask about every chunk"

    out = srv.rlm_sub_query_batch("ctx_1", "audit every file", reduce=False)
    assert len(asked) == 1, "the second run spent model calls on already-answered chunks"
    assert "RESUMED: 4 chunk(s) reused from disk" in out
    assert "finding 2" in out, "a cached answer was dropped from the report"


def test_a_budget_stop_leaves_a_resumable_gap_not_a_lost_run(monkeypatch, batch_ctx):
    """A stopped run must (a) keep what it bought, (b) say it is partial, and (c) resume
    at the gap. Losing any of the three reproduces the original incident."""
    import src.subquery as sq
    srv, _events = batch_ctx(4)
    asked: list[list[int]] = []

    def stop_after_two(prompts, model, concurrency=1, indices=None, on_result=None, **kw):
        asked.append(list(indices))
        out = []
        for pos, i in enumerate(indices):
            if pos < 2:
                r = sq.SubResult(i, f"finding {i}", 10, 5, model=model)
                on_result(r)
            else:
                r = sq.SubResult(i, "", 0, 0, error="deferred — session budget reached")
            out.append(r)
        return out

    monkeypatch.setattr(srv, "sub_query_batch", stop_after_two)
    first = srv.rlm_sub_query_batch("ctx_1", "audit every file", reduce=False)
    assert "STOPPED AT THE BUDGET LINE" in first
    assert "2 chunk(s) deferred" in first
    assert "resumes at chunk 2" in first
    assert not first.lstrip().startswith("ERROR"), "a clean budget stop is not a failure"

    # Resume: only the deferred chunks are re-asked.
    monkeypatch.setattr(srv, "sub_query_batch",
                        lambda prompts, model, concurrency=1, indices=None,
                        on_result=None, **kw: [
                            sq.SubResult(i, f"finding {i}", 10, 5, model=model)
                            for i in indices])
    srv.rlm_sub_query_batch("ctx_1", "audit every file", reduce=False)
    assert asked[0] == [0, 1, 2, 3] and len(asked) == 1


def test_every_completion_reaches_the_ledger_including_the_engine_path(bcfg, monkeypatch):
    """rlm_query's recursive fan-out must be as visible to the budget as our own batch.

    Recording spend in subquery.py only left the single most expensive tool — rlm_query,
    whose sub-calls go through the engine's client (rebound onto get_transport by
    src/auth.py) — spending its whole window budget invisibly, so rlm_estimate would then
    report headroom that had already been consumed. Recording in the transport is what
    makes "all model calls" mean all of them.
    """
    import src.transport as tp

    class _Backend:
        def complete(self, messages, system, model, max_tokens):
            return tp.CompletionResult(text="ok", input_tokens=1, output_tokens=64,
                                       model="claude-haiku-4-5")

        async def acomplete(self, messages, system, model, max_tokens):
            return self.complete(messages, system, model, max_tokens)

    monkeypatch.setattr(tp, "_CACHE", {})
    monkeypatch.setattr(tp, "budget", budget)
    ledgered = tp._LedgeredTransport(_Backend(), bcfg)

    ledgered.complete([{"role": "user", "content": "x" * 4000}], None, "m", 512)

    s = budget.spent(bcfg)
    assert s.calls == 1
    # Input is the LOCAL estimate of the prompt (~4000 chars / 4), not the backend's
    # reported input_tokens=1 -- that under-count is the whole reason this exists.
    assert s.tokens > 1000, f"input was taken from the transport's report: {s.tokens}"


def test_the_ledger_wrapper_does_not_hide_which_backend_was_chosen(bcfg):
    """Wrapping every transport must stay invisible to callers that ask what they got."""
    import src.transport as tp

    class _Backend:
        def complete(self, *a):
            raise AssertionError("not called")

        async def acomplete(self, *a):
            raise AssertionError("not called")

        auth_label = "cli"

    inner = _Backend()
    w = tp._LedgeredTransport(inner, bcfg)
    assert w.inner is inner
    assert w.auth_label == "cli", "attribute forwarding broke; wrapping is not transparent"
