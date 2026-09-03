"""The session-window budget: ledger, ceiling, estimate, gate — and resumable results.

These cover the three defects behind one real incident: a 103-chunk batch that spent ~60%
of a 4-hour window, was interrupted, and returned nothing re-runnable. Each test names the
failure it prevents rather than the function it calls.
"""

import dataclasses
import json
import time

import pytest

import src.batch as bt
import src.budget as budget
import src.results as results
from src.config import load_config
from tests.conftest import FrozenClock


@pytest.fixture
def bcfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        store_dir=tmp_path / "contexts",
        budget_ledger=tmp_path / "usage.jsonl",
        budget_state=tmp_path / "budget.json",
        cache_dir=tmp_path / "cache",
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


def test_hitting_a_usage_limit_is_recorded_as_evidence(bcfg):
    budget.record(bcfg, "m", 500_000, 100_000)
    budget.note_limit_hit(bcfg)

    assert budget.observed_wall(bcfg)[0] == 600_000


def test_a_wall_seen_at_a_higher_local_spend_raises_the_recorded_floor(bcfg):
    """The local ledger sees only OUR spend; other sessions on the account spent the rest,
    so a wall at 600K proves the ceiling is AT LEAST 600K, never that it is exactly that.
    A quieter window gives a tighter floor, so the highest observation is the useful one."""
    budget.record(bcfg, "m", 600_000, 0)
    budget.note_limit_hit(bcfg)
    budget.record(bcfg, "m", 300_000, 0)   # a quieter window: 900K reached before a wall
    budget.note_limit_hit(bcfg)

    assert budget.observed_wall(bcfg)[0] == 900_000


def test_a_hit_usage_limit_never_becomes_the_gate_by_itself(bcfg):
    """The failure this rules out. Treating the observed spend AS the ceiling put the stop
    line at 95% of it — below the spend that produced it — so the next call and every call
    after it was refused. In a large batch a transient 429 that the retry layer then
    handles successfully is routine, which turned one blip into a dead server for the rest
    of the window (and, while the lowest observation won, for the life of the state file).
    """
    budget.record(bcfg, "m", 40_000, 0)
    budget.note_limit_hit(bcfg)                      # a blip, retried successfully

    assert budget.observed_wall(bcfg)[0] == 40_000   # recorded...
    assert budget.ceiling(bcfg) == (None, "unknown")  # ...but not promoted to a gate
    budget.check_or_raise(bcfg, 1_000)               # so the next call still goes


def test_an_observed_wall_is_reported_as_the_number_to_configure(bcfg):
    budget.record(bcfg, "m", 600_000, 0)
    budget.note_limit_hit(bcfg)
    text = budget.render(budget.judge(bcfg, budget.estimate_batch(
        bcfg, [1000] * 3, prompt="p", max_output_tokens=2048, reduce=False)), what="x")

    assert "600,000" in text and "session_budget_tokens" in text


def test_a_configured_ceiling_is_the_only_gate(bcfg):
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


def test_the_reduce_call_counts_as_the_largest_call_when_it_is(bcfg):
    """max_call_tokens answers "can this be sent at all", and the reduce pass reads every
    map answer back in one call. Deriving it from the map chunks alone hid the single
    biggest call in the run from the only check that could refuse it: at 103 chunks the
    synthesis is ~211K tokens against ~32K for the largest chunk."""
    est = budget.estimate_batch(bcfg, [8_000] * 103, prompt="p",
                                max_output_tokens=2048, reduce=True)

    assert est.max_map_call_tokens < 40_000, "a single chunk call should stay small"
    assert est.max_call_tokens > 200_000, (
        "the synthesis call is missing from the largest-call figure: "
        f"max_call={est.max_call_tokens} map_max={est.max_map_call_tokens}")
    assert est.max_call_tokens >= est.max_map_call_tokens


def test_the_smallest_call_is_reported_separately_from_the_largest(bcfg):
    """"Is there room for even one" is a question about the SMALLEST call. Under `files`
    chunking a 200-token file sits beside a 30,000-token one, and answering it with the
    largest refused whole batches that would mostly have fitted."""
    est = budget.estimate_batch(bcfg, [200, 30_000], prompt="p",
                                max_output_tokens=2048, reduce=False)

    assert est.min_call_tokens < est.max_call_tokens
    assert est.min_call_tokens < 5_000


def test_the_reduce_call_is_priced_at_the_cap_it_actually_runs_with(bcfg):
    """BATCH_MAX_TOKENS' own comment: the estimate and the call that spends the tokens
    must use the same number. The synthesis is issued at REDUCE_MAX_TOKENS, so a forecast
    that priced it at the per-chunk cap was not a forecast of this run."""
    from src.batch import BATCH_MAX_TOKENS, REDUCE_MAX_TOKENS

    assert REDUCE_MAX_TOKENS != BATCH_MAX_TOKENS, "otherwise this test proves nothing"
    at_map_cap = budget.estimate_batch(bcfg, [1_000] * 3, prompt="p",
                                       max_output_tokens=BATCH_MAX_TOKENS, reduce=True)
    honest = budget.estimate_batch(bcfg, [1_000] * 3, prompt="p",
                                   max_output_tokens=BATCH_MAX_TOKENS, reduce=True,
                                   reduce_output_tokens=REDUCE_MAX_TOKENS)

    assert honest.output_tokens - at_map_cap.output_tokens == REDUCE_MAX_TOKENS - BATCH_MAX_TOKENS


def test_a_fully_cached_batch_still_forecasts_the_synthesis_it_will_run(bcfg):
    """A resumed run re-pays for no chunk, but with reduce=True it still issues one
    synthesis over the cached answers. Keyed on the REMAINING work, the forecast reported
    zero calls and zero tokens for a run that spends — an estimate of nothing at all."""
    est = budget.estimate_batch(bcfg, [10_000] * 4, prompt="p", max_output_tokens=2048,
                                reduce=True, reduce_output_tokens=4096,
                                done={0, 1, 2, 3})

    assert (est.chunks_todo, est.chunks_done) == (0, 4)
    assert est.calls == 1, "the synthesis over the cached answers was not forecast"
    assert est.total_tokens > 0, "a run that spends was forecast at zero tokens"
    assert est.min_call_tokens == est.max_call_tokens > 0, \
        "with one call left, that call is both the largest and the smallest"


def test_the_synthesis_is_priced_over_every_chunk_not_just_the_uncached_ones(bcfg):
    """The reduce pass reads the CACHED answers back too, so its input does not shrink as
    chunks get answered. Pricing it over `todo` under-counted every resumed run."""
    kw = dict(prompt="p", max_output_tokens=2048, done={0, 1, 2})
    map_only = budget.estimate_batch(bcfg, [10_000] * 4, reduce=False, **kw)
    with_reduce = budget.estimate_batch(bcfg, [10_000] * 4, reduce=True, **kw)

    assert with_reduce.input_tokens - map_only.input_tokens == 2048 * 4, \
        "the synthesis was priced over the 1 remaining chunk, not all 4 answers it reads"


# --------------------------- output is not capped on the CLI path --------------------- #
def test_expected_output_is_the_cap_until_the_ledger_has_something_to_say(bcfg):
    """A cold ledger holds no measurement worth trusting, so the requested cap stands and
    the first run of a fresh install is forecast optimistically. Known, self-correcting."""
    assert budget.expected_output(bcfg, 2048) == 2048

    for _ in range(3):
        budget.record(bcfg, "m", 100, 9_000)
    assert budget.expected_output(bcfg, 2048) == 2048, \
        "three records is not a measurement; the cap must still stand"


def test_expected_output_follows_what_the_cli_actually_emits(bcfg):
    """`claude` accepts no output-token flag, so CliTransport is handed max_tokens and has
    nowhere to put it: the cap is a request, not a bound. Measured on a real 33-chunk batch
    capped at 2048 — 328,453 output tokens, 4.9x the cap it was reserved against."""
    for _ in range(10):
        budget.record(bcfg, "m", 100, 10_000)

    assert budget.expected_output(bcfg, 2048) == 10_000


def test_one_runaway_completion_cannot_make_the_floor_refuse_everything(bcfg):
    """Same trap as a learned ceiling: a reservation derived from an unbounded observation
    stops being an estimate and starts refusing every call. See `ceiling`."""
    for _ in range(8):
        budget.record(bcfg, "m", 0, 5_000_000)

    assert budget.expected_output(bcfg, 2048) == 2048 * 8


def test_the_forecast_prices_output_at_what_the_path_emits_not_the_cap_it_ignores(bcfg):
    """The forecast said a 33-chunk review needed ~5% of headroom; it took ~10%, because
    output was priced at a cap the CLI never receives. A 2x error is the difference between
    "fits" and a stopped run at the sizes this module exists for."""
    thin = budget.estimate_batch(bcfg, [1_000] * 4, prompt="p", max_output_tokens=2048,
                                 reduce=False)
    for _ in range(10):
        budget.record(bcfg, "m", 100, 10_000)
    measured = budget.estimate_batch(bcfg, [1_000] * 4, prompt="p", max_output_tokens=2048,
                                     reduce=False)

    assert thin.output_tokens == 2048 * 4
    assert measured.output_tokens == 10_000 * 4, \
        "the forecast still priced output at a cap this transport cannot enforce"


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


# --------------------------- content-addressed answer cache ------------------------- #
def test_an_answer_is_found_again_from_a_different_context(bcfg):
    """THE reason the cache is keyed by content. The first design stored answers under the
    ctx_id, so re-loading the same file (a new ctx_id) orphaned every answer already paid
    for. The identity of an answer is (chunk bytes, prompt, model) -- which context asked
    is an accident of when you loaded it."""
    k = results.content_key("def test_x(): pass", "audit", "haiku")
    results.cache_put(bcfg, k, results.Saved(7, "finding", 100, 20, "haiku"))

    # a second context holding the same bytes computes the same key -- no ctx_id involved
    k2 = results.content_key("def test_x(): pass", "audit", "haiku")
    hit = results.cache_get(bcfg, k2, index=3)
    assert hit is not None and hit.answer == "finding"
    assert hit.index == 3, "the index is the CALLER's position, not the one stored"


def test_the_key_separates_text_prompt_and_model(bcfg):
    base = results.content_key("same text", "same prompt", "same-model")
    assert results.content_key("other text", "same prompt", "same-model") != base
    assert results.content_key("same text", "other prompt", "same-model") != base
    assert results.content_key("same text", "same prompt", "other-model") != base
    # ...and does NOT depend on chunk position or chunking, by design
    assert results.content_key("same text", "same prompt", "same-model") == base


def test_the_key_separates_output_contracts(bcfg):
    """An answer's key must name everything that produced it. The map used to send no
    system prompt and now sends batch.MAP_SYSTEM, which turns a page of prose into a JSON
    envelope: same chunk, same prompt, same model, DIFFERENT answer. Without `system` in
    the key a resumed run serves the old contract's answers under the new one, and with
    reduce=True merges prose and envelopes into a single synthesis.
    """
    base = results.content_key("same text", "same prompt", "m")
    keyed = results.content_key("same text", "same prompt", "m", bt.MAP_SYSTEM)
    assert keyed != base
    assert keyed == results.content_key("same text", "same prompt", "m", bt.MAP_SYSTEM)
    # The TEXT is hashed, not a version tag, so editing the constant self-invalidates.
    assert results.content_key("same text", "same prompt", "m", bt.MAP_SYSTEM + "!") != keyed
    # The manifest too: one holding answers from two contracts mislabels them as badly.
    assert results.run_key("p", "m", "files", 3, bt.MAP_SYSTEM) != results.run_key(
        "p", "m", "files", 3)


def test_the_estimate_counts_the_system_prompt_it_will_send(bcfg):
    """estimate_batch's contract is that every number in it is the number that will
    actually be sent. MAP_SYSTEM rides on EVERY map call, so omitting it under-counts
    input by its size times the chunk count -- small next to a 300K overshoot, but this
    module exists to stop a forecast drifting from its receipt.
    """
    kw = {"prompt": "audit", "max_output_tokens": 2048, "reduce": False}
    without = budget.estimate_batch(bcfg, [1000] * 10, **kw)
    with_sys = budget.estimate_batch(bcfg, [1000] * 10, system=bt.MAP_SYSTEM, **kw)

    delta = with_sys.input_tokens - without.input_tokens
    assert delta == budget.estimate_tokens(bt.MAP_SYSTEM) * 10
    assert with_sys.output_tokens == without.output_tokens, "output must not move"


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(bcfg):
    """Entries are written atomically so torn files should not exist, but a bad one
    must still cost one re-ask rather than take the batch down."""
    k = results.content_key("t", "p", "m")
    p = results.cache_path(bcfg, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"answer": ', encoding="utf-8")

    assert results.cache_get(bcfg, k, index=0) is None


def test_fresh_deletes_only_the_matching_answer(bcfg):
    k1 = results.content_key("t1", "p", "m")
    k2 = results.content_key("t2", "p", "m")
    results.cache_put(bcfg, k1, results.Saved(0, "a", 1, 1, "m"))
    results.cache_put(bcfg, k2, results.Saved(1, "b", 1, 1, "m"))

    assert results.cache_delete(bcfg, k1) is True
    assert results.cache_get(bcfg, k1, index=0) is None
    assert results.cache_get(bcfg, k2, index=1).answer == "b"


def test_a_cache_hit_refreshes_the_entry_so_lru_keeps_it(bcfg, clock):
    """Eviction is LRU by mtime. A hit must touch the file, or an answer used every day
    is evicted as if it had never been read since it was written."""
    import os
    k = results.content_key("t", "p", "m")
    results.cache_put(bcfg, k, results.Saved(0, "a", 1, 1, "m"), now=clock)
    p = results.cache_path(bcfg, k)
    os.utime(p, (clock.now - 1000, clock.now - 1000))
    before = p.stat().st_mtime

    results.cache_get(bcfg, k, index=0)

    assert p.stat().st_mtime > before, "a hit did not refresh the entry"


def test_the_sweep_evicts_least_recently_used_beyond_the_byte_cap(bcfg, clock):
    """No TTL, on purpose -- an entry stays correct as long as (bytes, prompt, model)
    match. Disk is bounded by bytes, evicting the entries least recently USEFUL first."""
    import dataclasses as dc, os
    cfg = dc.replace(bcfg, cache_sweep_cooldown_s=0)
    keys = [results.content_key(f"t{i}", "p", "m") for i in range(3)]
    for i, k in enumerate(keys):
        results.cache_put(cfg, k, results.Saved(i, "x" * 20, 1, 1, "m"), now=clock)
        os.utime(results.cache_path(cfg, k), (clock.now - 100 + i, clock.now - 100 + i))
    # Cap = room for exactly two entries, measured from a real one rather than guessed:
    # the JSON envelope is ~85 bytes here, and a hand-picked number was simply wrong.
    one = results.cache_path(cfg, keys[0]).stat().st_size
    cfg = dc.replace(cfg, cache_max_bytes=2 * one)
    # entry 0 is the oldest on disk, but it was just USED -- the hit refreshes it
    results.cache_get(cfg, keys[0], index=0)
    os.utime(results.cache_path(cfg, keys[0]), (clock.now, clock.now))

    results.sweep(cfg, now=clock)

    alive = [results.cache_get(cfg, k, index=0) is not None for k in keys]
    assert alive == [True, False, True], (
        "expected the least-recently-used entry (1) evicted and the recently hit one (0) kept")


def test_the_sweep_respects_its_cooldown(bcfg, clock):
    import dataclasses as dc, os
    cfg = dc.replace(bcfg, cache_max_bytes=1, cache_sweep_cooldown_s=300)
    k = results.content_key("t", "p", "m")
    results.cache_put(cfg, k, results.Saved(0, "x" * 50, 1, 1, "m"), now=clock)
    sentinel = cfg.cache_dir / ".sweep"
    sentinel.write_text("recent")
    os.utime(sentinel, (clock.now - 10, clock.now - 10))

    results.sweep(cfg, now=clock)
    assert results.cache_get(cfg, k, index=0) is not None, "swept inside the cooldown"


def test_the_manifest_records_what_a_run_produced_with_its_timestamps(bcfg, clock):
    key = results.run_key("audit", "haiku", "lines", 3)
    results.manifest_append(bcfg, "ctx_1", key, results.Saved(0, "a", 1, 1, "haiku"), now=clock)
    clock.tick(45)
    results.manifest_append(bcfg, "ctx_1", key, results.Saved(1, "b", 1, 1, "haiku"), now=clock)

    lines = [json.loads(l) for l in
             results.manifest_path(bcfg, "ctx_1", key).read_text(encoding="utf-8").splitlines()]
    assert [(r["index"], r["ts"]) for r in lines] == [(0, FrozenClock.START), (1, FrozenClock.START + 45)]


# --------------------------- end-to-end resume through the tool --------------------- #
def test_a_second_run_pays_only_for_the_chunks_it_is_missing(monkeypatch, batch_ctx):
    """THE headline fix. The incident was a 30-minute run that returned nothing and would
    have cost the same again. Here the second call must make model calls only for what
    the first did not finish."""
    import src.subquery as sq
    d, _events = batch_ctx(4)
    asked: list[list[int]] = []

    def fake_batch(cfg, prompts, model, concurrency=1, indices=None, on_result=None, **kw):
        asked.append(list(indices))
        out = [sq.SubResult(i, f"finding {i}", 10, 5, model=model) for i in indices]
        for r in out:                     # the real batch persists as answers land
            on_result(r)
        return out

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_1", "audit every file", reduce=False)
    assert asked == [[0, 1, 2, 3]], "the first run should ask about every chunk"

    out = bt.run(d, "ctx_1", "audit every file", reduce=False)
    assert len(asked) == 1, "the second run spent model calls on already-answered chunks"
    assert "RESUMED: 4 chunk(s) reused from disk" in out
    assert "finding 2" in out, "a cached answer was dropped from the report"


def test_a_budget_stop_leaves_a_resumable_gap_not_a_lost_run(monkeypatch, batch_ctx):
    """A stopped run must (a) keep what it bought, (b) say it is partial, and (c) resume
    at the gap. Losing any of the three reproduces the original incident."""
    import src.subquery as sq
    d, _events = batch_ctx(4)
    asked: list[list[int]] = []

    def stop_after_two(cfg, prompts, model, concurrency=1, indices=None, on_result=None, **kw):
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

    monkeypatch.setattr(bt, "sub_query_batch", stop_after_two)
    first = bt.run(d, "ctx_1", "audit every file", reduce=False)
    assert "STOPPED AT THE BUDGET LINE" in first
    assert "2 chunk(s) deferred" in first
    assert "resumes at chunk 2" in first
    assert not first.lstrip().startswith("ERROR"), "a clean budget stop is not a failure"

    # Resume: only the deferred chunks are re-asked.
    monkeypatch.setattr(bt, "sub_query_batch",
                        lambda cfg, prompts, model, concurrency=1, indices=None,
                        on_result=None, **kw: [
                            sq.SubResult(i, f"finding {i}", 10, 5, model=model)
                            for i in indices])
    bt.run(d, "ctx_1", "audit every file", reduce=False)
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


# --------------------------- frozen clock: arithmetic that was uncoverable ---------- #
def test_the_window_boundary_is_inclusive_at_exactly_one_window_ago(bcfg, clock):
    """A record at EXACTLY now - window: in, or out? Unanswerable against a moving clock,
    which is why the original window test used hour-wide margins and never pinned it.

    ``spent`` asks for records ``>= now - window_s``, so the boundary record counts. That
    is the safe direction: counting a just-expiring call keeps the estimate conservative,
    where dropping it would over-report headroom right at the wall.
    """
    window_s = bcfg.session_window_h * 3600
    budget.record(bcfg, "m", 100, 0, now=lambda: clock.now - window_s)       # exactly on it
    budget.record(bcfg, "m", 7, 0, now=lambda: clock.now - window_s - 0.001)  # a hair past

    s = budget.spent(bcfg, now=clock)
    assert s.tokens == 100, "the boundary record must count and the one past it must not"
    assert s.calls == 1


def test_oldest_expires_in_s_says_when_headroom_next_grows(bcfg, clock):
    """The only answer to "when can I run again", and it had no test. The window rolls
    continuously — it does not reset on a clock boundary — so this is the oldest call's
    timestamp plus one window, measured from now."""
    window_s = bcfg.session_window_h * 3600
    budget.record(bcfg, "m", 10, 0, now=lambda: clock.now - 3600)   # 1h ago
    budget.record(bcfg, "m", 10, 0, now=lambda: clock.now - 600)    # 10m ago

    s = budget.spent(bcfg, now=clock)
    assert s.oldest_expires_in_s == pytest.approx(window_s - 3600)

    clock.tick(3600)   # an hour passes; the oldest is now due to age out
    assert budget.spent(bcfg, now=clock).oldest_expires_in_s == pytest.approx(window_s - 7200)


def test_a_fully_expired_window_reports_no_spend_and_no_expiry(bcfg, clock):
    budget.record(bcfg, "m", 500, 0, now=lambda: clock.now)
    clock.tick(bcfg.session_window_h * 3600 + 1)

    s = budget.spent(bcfg, now=clock)
    assert (s.tokens, s.calls, s.oldest_expires_in_s) == (0, 0, None)


def test_observed_call_latency_is_measured_from_the_ledger_spacing(bcfg, clock):
    """The wall-time forecast comes from this, and it had no test because it is nothing
    BUT clock arithmetic. Ten calls 20s apart at concurrency 3: the span is 180s over 9
    gaps = 20s per slot, times 3 lanes = 60s of work per call.
    """
    cfg = dataclasses.replace(bcfg, subquery_concurrency=3)
    for _ in range(10):
        budget.record(cfg, "m", 1, 1, now=clock)
        clock.tick(20)

    assert budget._observed_call_s(cfg, clock) == pytest.approx(60.0)


def test_too_little_history_falls_back_to_the_default_latency(bcfg, clock):
    """Below the sample threshold the measurement is noise, so the forecast must use the
    documented default rather than a confident number derived from three calls."""
    for _ in range(3):
        budget.record(bcfg, "m", 1, 1, now=clock)
        clock.tick(5)

    assert budget._observed_call_s(bcfg, clock) == budget._DEFAULT_CALL_S


def test_an_implausible_measured_latency_is_rejected(bcfg, clock):
    """A ledger spanning days (an idle server, not a slow one) would forecast a batch at
    weeks. The measurement is only trusted inside a plausible band."""
    for _ in range(10):
        budget.record(bcfg, "m", 1, 1, now=clock)
        clock.tick(6 * 3600)          # six hours between calls

    assert budget._observed_call_s(bcfg, clock) == budget._DEFAULT_CALL_S


def test_the_forecast_uses_the_measured_latency(bcfg, clock):
    """End to end: measured latency feeds the wall-time estimate, so a slow transport
    makes the forecast longer rather than the forecast staying a constant."""
    cfg = dataclasses.replace(bcfg, subquery_concurrency=1)
    for _ in range(10):
        budget.record(cfg, "m", 1, 1, now=clock)
        clock.tick(30)

    est = budget.estimate_batch(cfg, [1000] * 4, prompt="p", max_output_tokens=100,
                                reduce=False, now=clock)
    assert est.seconds == pytest.approx(4 * 30.0)


def test_the_learned_ceiling_records_when_it_was_observed(bcfg, clock):
    """observed_at is what a future reader uses to judge whether a learned ceiling is
    still current. It was written but never asserted."""
    budget.record(bcfg, "m", 400_000, 0, now=clock)
    budget.note_limit_hit(bcfg, now=clock)

    state = json.loads(bcfg.budget_state.read_text(encoding="utf-8"))
    assert state["learned_ceiling_tokens"] == 400_000
    assert state["observed_at"] == clock.now




# --------------------------- (1) content cache, through the tool ------------------- #
def test_a_re_loaded_copy_of_the_same_file_pays_nothing(monkeypatch, batch_ctx):
    """The gap the ctx-keyed design had: same bytes, new ctx_id, full re-spend. Two
    different ctx_ids over identical chunk text must share every answer."""
    import src.subquery as sq
    d, _ = batch_ctx(4)
    asked: list[list[int]] = []

    def fake_batch(cfg, prompts, model, concurrency=1, indices=None, on_result=None, **kw):
        asked.append(list(indices))
        out = [sq.SubResult(i, f"finding {i}", 10, 5, model=model) for i in indices]
        for r in out:
            on_result(r)
        return out

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    bt.run(d, "ctx_first_load", "audit every file", reduce=False)
    out = bt.run(d, "ctx_second_load", "audit every file", reduce=False)

    assert asked == [[0, 1, 2, 3]], "the second context re-asked about identical bytes"
    assert "RESUMED: 4 chunk(s) reused from disk" in out


def test_only_the_changed_chunk_is_re_asked(monkeypatch, batch_ctx):
    """Edit one file in a repo, re-analyse: exactly one call. This is the property that
    makes the cache worth having, and it only holds when chunk boundaries are stable --
    which is why default_strategy prefers `files`."""
    import src.subquery as sq
    asked: list[list[int]] = []

    def fake_batch(cfg, prompts, model, concurrency=1, indices=None, on_result=None, **kw):
        asked.append(list(indices))
        out = [sq.SubResult(i, f"finding {i}", 10, 5, model=model) for i in indices]
        for r in out:
            on_result(r)
        return out

    monkeypatch.setattr(bt, "sub_query_batch", fake_batch)
    d, _ = batch_ctx(4)
    bt.run(d, "ctx_v1", "audit", reduce=False)

    d2, _ = batch_ctx(4, text_for=lambda i: "EDITED" if i == 2 else f"chunk{i}")
    bt.run(d2, "ctx_v2", "audit", reduce=False)

    assert asked == [[0, 1, 2, 3], [2]], "chunks whose bytes did not change were re-asked"


# --------------------------- (2) the content-aware default strategy ------------------ #
def test_a_dir_load_defaults_to_files_chunking(deps, tmp_path):
    class _Meta:
        source_type = "dir"
        content_path = str(tmp_path / "nope")
    assert bt.default_strategy(deps, _Meta()) == "files"


def test_a_bundle_with_file_markers_defaults_to_files_chunking(deps, tmp_path):
    bundle = tmp_path / "bundle.txt"
    bundle.write_text("===== FILE: a.py (12 bytes) =====\nprint(1)\n", encoding="utf-8")

    class _Meta:
        source_type = "file"
        content_path = str(bundle)
    assert bt.default_strategy(deps, _Meta()) == "files"


def test_a_plain_file_keeps_the_configured_default(deps, tmp_path):
    plain = tmp_path / "log.txt"
    plain.write_text("2026-09-02 boot\n2026-09-02 ready\n", encoding="utf-8")

    class _Meta:
        source_type = "file"
        content_path = str(plain)
    assert bt.default_strategy(deps, _Meta()) == deps.cfg.chunk_strategy


# --------------------------- (3) never-possible is judged against the ceiling -------- #
def test_a_chunk_larger_than_the_stop_line_is_impossible_not_waitable(bcfg):
    """The bug: the refusal compared one chunk against REMAINING headroom and said
    "headroom returns in ~Xh". A chunk larger than the ceiling never fits any window;
    saying "wait" is false. The verdict must say so and name the fix (re-chunk)."""
    cfg = dataclasses.replace(bcfg, session_budget_tokens=100_000, budget_stop_fraction=0.95)
    est = budget.estimate_batch(cfg, [200_000], prompt="p", max_output_tokens=2048, reduce=False)
    v = budget.judge(cfg, est)

    assert not v.possible
    assert not v.fits
    text = budget.render(v, what="x")
    assert "Impossible in ANY window" in text
    assert "Re-chunk" in text
    assert "returns" not in text, "must not tell the user to wait for something that never comes"


def test_a_chunk_that_fits_a_fresh_window_but_not_the_remainder_is_waitable(bcfg):
    cfg = dataclasses.replace(bcfg, session_budget_tokens=100_000, budget_stop_fraction=0.95)
    budget.record(cfg, "m", 90_000, 0)   # 5k headroom left
    est = budget.estimate_batch(cfg, [20_000], prompt="p", max_output_tokens=2048, reduce=False)
    v = budget.judge(cfg, est)

    assert v.possible, "a 22k call fits a fresh 95k window -- it is waitable, not impossible"
    assert not v.fits


def test_the_batch_refuses_an_impossible_chunk_without_promising_headroom(monkeypatch, batch_ctx):
    d, _ = batch_ctx(2, est_tokens=10_000_000)
    d = dataclasses.replace(d, cfg=dataclasses.replace(d.cfg, session_budget_tokens=5_800_000))
    called = []
    monkeypatch.setattr(bt, "sub_query_batch", lambda *a, **k: called.append(1) or [])

    out = bt.run(d, "ctx_1", "audit", reduce=False)

    assert out.startswith("ERROR: a single chunk can never fit")
    assert "Headroom returns" not in out
    assert not called, "an impossible batch must not dispatch"


# --------------------------- (4) rlm_query: a ceiling, not an estimate ---------------- #
def test_the_query_ceiling_is_derived_from_config_alone(bcfg):
    q = budget.query_ceiling(bcfg)
    subs = bcfg.max_iterations * bcfg.max_concurrent_subcalls
    assert q.root_calls_max == bcfg.max_iterations
    assert q.sub_calls_max == subs
    assert q.worst_tokens == subs * (bcfg.sub_context_tokens + 2048) + bcfg.max_iterations * bcfg.max_output_tokens
    assert q.timeout_s == bcfg.query_timeout_s
    assert q.timeout_calls == int(bcfg.query_timeout_s / budget._DEFAULT_CALL_S)


def test_the_query_ceiling_says_what_it_is_and_is_not(bcfg):
    text = budget.render_query_ceiling(budget.query_ceiling(bcfg), cap=5_800_000)
    assert "ceiling, not an estimate" in text
    # Both sides of the bound are now stated: the floor stops it, a stop is resumable.
    # The earlier footer said "not gated ... cannot be resumed"; that was true then and
    # is exactly what the transport floor and the engine checkpoint closed.
    assert "stops itself before the wall" in text
    assert "resume" in text
    assert "cannot be resumed" not in text, "the footer still describes the old, unfixed rlm_query"
    assert "x the whole window" in text, "worst case should be stated relative to the window"


def test_the_estimate_tool_appends_the_query_ceiling(monkeypatch, batch_ctx):
    d, _ = batch_ctx(3)
    out = bt.estimate(d, "ctx_1", "audit", reduce=False)
    assert "## Estimate" in out
    assert "rlm_query on this context" in out


# --------------------------- the hard floor under every caller ----------------------- #
def test_the_floor_refuses_exactly_past_the_stop_line_and_not_before(bcfg):
    """`>` not `>=`: a call that lands exactly on the line is allowed. The margin above
    it is what absorbs the concurrency overshoot the docstring admits to."""
    cfg = dataclasses.replace(bcfg, session_budget_tokens=100_000, budget_stop_fraction=0.95)
    budget.record(cfg, "m", 90_000, 0)                 # usable = 95_000

    budget.check_or_raise(cfg, 5_000)                  # exactly on the line: allowed
    with pytest.raises(budget.BudgetStopError) as ei:
        budget.check_or_raise(cfg, 5_001)
    assert ei.value.spent == 90_000 and ei.value.usable == 95_000


def test_the_floor_never_fires_with_an_unknown_ceiling(bcfg):
    """Unknown must not mean zero: a machine that has never hit a limit must not have
    every call refused."""
    budget.check_or_raise(bcfg, 10**9)


def test_a_budget_stop_carries_both_engine_markers():
    """The engine never imports us; it honours two duck-typed attributes. Drop either and
    the behaviour silently degrades -- fan-outs re-issue doomed calls, or the root loop
    lets the stop escape as a traceback."""
    from rlm.utils.exceptions import aborts_batch, stops_run
    e = budget.BudgetStopError(1, 2, 3)
    assert aborts_batch(e), "batched fan-outs must abort on it"
    assert stops_run(e), "the root loop must convert it to a resumable limit"
