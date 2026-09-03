"""Tool-level map-reduce over a chunked context: estimate, run, resume, render.

Every function here takes ``Deps`` as its first argument and reads nothing from module
state. That is the point: this is the logic the tests actually drive, and it used to be
reachable only by patching ``server.CFG`` / ``server.STORE`` — patching that was
conditional on import order, which let a resume test pass by reading its own leftovers
out of the operator's real ``~/.rlm``. See ``deps.py`` for the full account.

``server.py`` keeps the ``@mcp.tool()`` functions as one-line adapters that supply the
process's ``DEPS``. A tool's signature IS the MCP schema, so it cannot carry a ``Deps``
parameter — which makes the adapter boundary the natural composition root and leaves this
module free of globals.

The distinction from ``subquery.py``: that module owns ONE fan-out across the transport
(threads, throttle, retry). This module owns what a batch MEANS — which chunks are still
unanswered, what the remainder will cost, when to stop, and how to report a partial run.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from . import budget, models, results, transport
from .chunking import chunk_text, looks_like_file_bundle
from .config import estimate_tokens
from .deps import Deps
from .logsetup import log_event
from .output import encoded_len
from .subquery import SubResult, sub_query, sub_query_batch

#: Per-chunk output ceiling for a batch map call. Named because the pre-flight estimate
#: and the call that spends the tokens MUST use the same number — an estimate computed
#: against a different cap is not an estimate of this run.
BATCH_MAX_TOKENS = 2048
#: The synthesis pass gets more room than one chunk answer: it speaks for all of them.
#: Named for the same reason, and threaded into estimate_batch for the same reason — it
#: was 4096 at the call site while the forecast priced it at 2048.
REDUCE_MAX_TOKENS = 4096

#: The map's output contract. Every sub-model call used to run with NO system prompt at
#: all -- `sub_query`/`sub_query_batch` and both transports have always accepted one and
#: no caller ever passed it -- so each chunk was answered by the `claude` CLI's default
#: coding-assistant persona, which explains its work. Measured cost of that: 328,453
#: output tokens on a 33-chunk run capped at 2048, 4.9x the cap (docs/07 §10).
#:
#: A schema, not a plea. "Output exactly X" is the instruction that already failed; a
#: model holds an envelope far more tightly than a phrasing rule. Four parts earn their
#: tokens: the envelope; the EMPTY case shown as its own example and named correct, since
#: that is the one a model resists and showing it only filled teaches that filled is the
#: goal; the fence ban, or the same padding returns wearing a json fence; and the
#: precedence clause, because the caller's own "output exactly NO ISSUES" now COMPETES
#: with this, and competing instructions are what produce hedging in the first place.
#:
#: Deliberately generic. This tool maps ANY prompt over chunks -- "summarize each",
#: "extract every IP" -- so the field is `findings`, already this module's word for the
#: per-chunk answers (`_render` builds `findings`, `_reduce` consumes them), and never
#: anything review-specific.
MAP_SYSTEM = (
    "You analyze ONE chunk of a larger document for an automated pipeline. Your "
    "output is parsed by a program, not read by a person.\n"
    "Reply with one raw JSON object and nothing else — no prose around it, no code "
    "fences, no reasoning:\n"
    '{"findings": ["<one self-contained sentence>", "..."]}\n'
    "Nothing to report is the empty list. That is a complete, correct answer, and the "
    "expected one for most chunks:\n"
    '{"findings": []}\n'
    'Example — request: "find hardcoded credentials", chunk contains one:\n'
    '{"findings": ["Line 412 assigns API_KEY to the literal \'sk-live-4f...\'."]}\n'
    "The user's request may describe its own output format. That describes the CONTENT "
    "of a finding string; the JSON envelope above always wins."
)

#: The same obedience contract WITHOUT the envelope, for the three calls whose answer is
#: rendered straight to a reader: the synthesis, a single sub-query, and the auth probe.
#: A findings array there would destroy the deliverable -- `_reduce` exists to produce one
#: coherent prose answer -- so only the map, which fans out N-wide and is read by a
#: program, gets the schema.
TERSE_SYSTEM = (
    "You are answering inside an automated pipeline. Emit only the answer the request "
    "asks for: no preamble, no restating the task, no reasoning, no closing summary. "
    "If the request names an exact string to reply with, reply with that string alone."
)


# --------------------------------------------------------------------------- #
# Chunking default
# --------------------------------------------------------------------------- #
def default_strategy(d: Deps, meta) -> str:
    """The chunk strategy to use when the caller did not pick one.

    ``files`` for anything with file boundaries — a dir load, or a bundle built with the
    documented ``===== FILE:`` separator — and the configured default otherwise. This is
    not a stylistic preference: the answer cache is keyed by chunk bytes, and file
    boundaries are the only ones that survive an edit elsewhere in the corpus. Under
    ``lines``, editing 3 of 1,053 files shifts every later line boundary and nearly every
    cached answer misses; under ``files`` it costs 3 calls. Content-defined chunking at
    file granularity, which is what makes the cache worth having.
    """
    if getattr(meta, "source_type", "") == "dir":
        return "files"
    try:
        with open(meta.content_path, encoding="utf-8", errors="replace") as fh:
            if looks_like_file_bundle(fh.read(8192)):
                return "files"
    except (OSError, AttributeError, TypeError):
        pass
    return d.cfg.chunk_strategy


def _ensure_chunked(d: Deps, ctx_id: str):
    """Chunk the context with the default strategy if nobody has yet. Returns the meta."""
    meta = d.store.get(ctx_id)
    if meta.chunks:
        return meta
    strategy = default_strategy(d, meta)
    text = d.store.read_text(ctx_id)
    chunks = chunk_text(text, strategy, chunk_lines=d.cfg.chunk_lines,
                        chunk_chars=d.cfg.chunk_chars, overlap=d.cfg.chunk_overlap)
    d.store.set_chunks(ctx_id, strategy, [c.as_dict() for c in chunks],
                       text=text)   # reuse this decode; don't make a second copy
    del text   # the full decode has served chunking; don't hold it for the whole batch
    return d.store.get(ctx_id)


# --------------------------------------------------------------------------- #
# Cache scan
# --------------------------------------------------------------------------- #
def _scan_cache(d: Deps, ctx_id: str, sel: list[int], prompt: str, model: str, *,
                fresh: bool = False,
                system: str = "") -> tuple[dict[int, str], dict[int, results.Saved]]:
    """Look every selected chunk up in the answer cache by its digest.

    Returns ``(keys, cached)``: the content key per chunk index (the run needs it to store
    new answers under), and the answers already on disk. ``fresh`` deletes instead of
    looking up — the deliberate way to re-ask.

    ``system`` rides into the key so a cached answer is only reused under the contract
    that produced it; see ``results.content_key``. It must be the same value the map
    actually sends, or the scan reports hits the run cannot use.

    The digests come from ``ContextStore.chunk_digests``, which serves them from the chunk
    meta stamped at chunking time. This scan used to read every selected chunk purely to
    hash it, while the batch then read each un-cached one AGAIN inside its worker — a full
    extra pass over the context before a single call went out.
    """
    keys: dict[int, str] = {}
    cached: dict[int, results.Saved] = {}
    digests = d.store.chunk_digests(ctx_id, sel)
    for i in sel:
        k = results.content_key(digests[i], prompt, model, system)
        keys[i] = k
        if fresh:
            results.cache_delete(d.cfg, k)
            continue
        hit = results.cache_get(d.cfg, k, index=i)
        if hit is not None:
            cached[i] = hit
    return keys, cached


# --------------------------------------------------------------------------- #
# Estimate & budget (no model call)
# --------------------------------------------------------------------------- #
def estimate(d: Deps, ctx_id: str, prompt: str = "", max_chunks: int = 0,
             reduce: bool = True) -> str:
    """Forecast what ``run`` over this context would cost, and judge it against the
    headroom left in the session window. Makes no model call."""
    meta = d.store.get(ctx_id)
    chunks = meta.chunks
    strategy = meta.chunk_strategy or default_strategy(d, meta)
    chunked = bool(chunks)
    if not chunked:
        # Chunk boundaries only — no store mutation, because an estimate must not change
        # the thing it is estimating. A later batch may legitimately chunk differently.
        text = d.store.read_text(ctx_id)
        chunks = [c.as_dict() for c in chunk_text(
            text, strategy, chunk_lines=d.cfg.chunk_lines,
            chunk_chars=d.cfg.chunk_chars, overlap=d.cfg.chunk_overlap)]
        del text
    n = len(chunks)
    sel = list(range(n if max_chunks <= 0 else min(max_chunks, n)))
    sub_model = models.select(d.cfg, models.Role.SUB)
    # The cache scan needs the exact chunk text, which only a chunked context can serve;
    # an unchunked one reports zero hits and says so rather than guess.
    cached = (_scan_cache(d, ctx_id, sel, prompt, sub_model, system=MAP_SYSTEM)[1]
              if chunked else {})
    done_pos = {pos for pos, i in enumerate(sel) if i in cached}
    est = budget.estimate_batch(
        d.cfg, [int(chunks[i].get("est_tokens", 0)) for i in sel], prompt=prompt,
        max_output_tokens=BATCH_MAX_TOKENS, reduce=reduce,
        reduce_output_tokens=REDUCE_MAX_TOKENS, done=done_pos, system=MAP_SYSTEM,
        now=d.clock)
    cap, _src = budget.ceiling(d.cfg)
    body = budget.render(budget.judge(d.cfg, est, now=d.clock),
                         what=f"batch over {ctx_id} ({strategy}, {n} chunks)")
    if not chunked:
        body += ("\n\n_Not chunked yet: cached answers can only be counted once "
                 "rlm_chunk_context has run; the count above assumes none._")
    return d.answer(
        body
        + f"\n\n_Sub-model: {sub_model}. Estimate only — no model call was made._\n\n"
        + budget.render_query_ceiling(budget.query_ceiling(d.cfg, now=d.clock), cap)
    )


def budget_report(d: Deps) -> str:
    """Spend inside the rolling window, the ceiling being gated against, and when
    headroom next grows."""
    cap, source = budget.ceiling(d.cfg)
    s = budget.spent(d.cfg, now=d.clock)
    lines = [
        f"## Session budget — rolling {s.window_h:g}h window",
        "",
        f"- spent by this server: **~{s.tokens:,} tokens** over {s.calls} calls",
    ]
    if cap is None:
        lines += [
            "- ceiling: **unknown**",
            "",
            "No `session_budget_tokens` is set and no usage limit has been hit yet, so runs "
            "are measured but not gated. Set `session_budget_tokens` in config.yaml to gate "
            "from the first run; otherwise the ceiling is learned the first time a usage "
            "limit is hit.",
        ]
    else:
        usable = int(cap * d.cfg.budget_stop_fraction)
        lines += [
            f"- ceiling ({source}): **~{cap:,}**",
            f"- stop line ({round(d.cfg.budget_stop_fraction * 100)}%): ~{usable:,}",
            f"- headroom: **~{max(0, usable - s.tokens):,}**",
        ]
    if s.oldest_expires_in_s:
        lines.append(f"- headroom next grows in ~{budget.fmt_dur(s.oldest_expires_in_s)}")
    lines.append("\n_This server's own spend only — other Claude sessions on the same "
                 "account are invisible here, so headroom is an upper bound._")
    return d.bound("\n".join(lines))


# --------------------------------------------------------------------------- #
# Single sub-query
# --------------------------------------------------------------------------- #
def one(d: Deps, ctx_id: str, prompt: str, chunk_index: int = -1) -> str:
    """One cheap sub-model query over a context, or over one chunk of it."""
    meta = d.store.get(ctx_id)
    if chunk_index >= 0:
        body = d.store.read_chunk(ctx_id, chunk_index)
    else:
        if meta.est_tokens > 0.9 * d.cfg.sub_context_tokens:
            return d.bound(
                f"ERROR: context ~{meta.est_tokens:,} tokens exceeds Haiku's "
                f"{d.cfg.sub_context_tokens:,}-token window. Run rlm_chunk_context then "
                f"rlm_sub_query_batch, or use rlm_query."
            )
        body = d.store.read_text(ctx_id)
    sub_model = models.select(d.cfg, models.Role.SUB)
    res = sub_query(d.cfg, f"{prompt}\n\n--- CONTEXT ---\n{body}", sub_model,
                    system=TERSE_SYSTEM)
    if res.error:
        return d.bound(f"ERROR ({sub_model}): {res.error}")
    return d.answer(
        f"## Sub-query answer ({res.model or sub_model}"
        f" · auth: {transport.auth_label(d.cfg)})\n\n{res.answer}\n\n---\n"
        f"tokens: {res.input_tokens:,} in / {res.output_tokens:,} out"
    )


# --------------------------------------------------------------------------- #
# Map-reduce
# --------------------------------------------------------------------------- #
def _mk_prompt(d: Deps, ctx_id: str, prompt: str, i: int, n: int) -> str:
    return f"{prompt}\n\n--- CHUNK {i + 1}/{n} ---\n{d.store.read_chunk(ctx_id, i)}"


def _persist(d: Deps, ctx_id: str, rkey: str, ckey: str, r: SubResult) -> None:
    """Store one answer the moment it lands: in the content cache (for any future context
    holding these bytes) and in this run's manifest (for the human, and for the over-cap
    reply to point at). The run that motivated all of this died holding 40 finished
    answers it had already paid for."""
    saved = results.Saved(r.index, r.answer, r.input_tokens, r.output_tokens, r.model)
    results.cache_put(d.cfg, ckey, saved, now=d.clock)
    results.manifest_append(d.cfg, ctx_id, rkey, saved, now=d.clock)


def run(d: Deps, ctx_id: str, prompt: str, max_chunks: int = 0, reduce: bool = True,
        fresh: bool = False) -> str:
    """Map ``prompt`` over the chunks of a context, resuming what is already answered and
    stopping at the session-window budget line rather than being killed at it."""
    meta = _ensure_chunked(d, ctx_id)
    n = len(meta.chunks)
    sel = list(range(n if max_chunks <= 0 else min(max_chunks, n)))
    sub_model = models.select(d.cfg, models.Role.SUB)
    strategy = meta.chunk_strategy or d.cfg.chunk_strategy

    # --- what is already answered? ------------------------------------------------
    # Keyed by chunk CONTENT, so a re-loaded file or a re-bundled repo hits on every
    # byte-identical chunk. The manifest is per run and only for the human.
    rkey = results.run_key(prompt, sub_model, strategy, n, MAP_SYSTEM)
    if fresh:
        results.manifest_clear(d.cfg, ctx_id, rkey)
    keys, cached = _scan_cache(d, ctx_id, sel, prompt, sub_model, fresh=fresh,
                               system=MAP_SYSTEM)
    todo = [i for i in sel if i not in cached]

    # --- what will the REMAINDER cost? --------------------------------------------
    sel_tokens = [int(meta.chunks[i].get("est_tokens", 0)) for i in sel]
    done_pos = {pos for pos, i in enumerate(sel) if i in cached}
    est = budget.estimate_batch(d.cfg, sel_tokens, prompt=prompt,
                                max_output_tokens=BATCH_MAX_TOKENS, reduce=reduce,
                                reduce_output_tokens=REDUCE_MAX_TOKENS,
                                done=done_pos, system=MAP_SYSTEM, now=d.clock)
    verdict = budget.judge(d.cfg, est, now=d.clock)

    # A fully resumed run is not a free run: with reduce=True the synthesis still reads
    # every CACHED answer and still costs a call. So the budget gates below run BEFORE the
    # all-cached return — returning first made the one call such a run does make the one
    # call nothing checked.
    what = (f"{len(todo)} chunk(s) of {ctx_id}" if todo else
            f"the synthesis over {len(sel)} cached chunk(s) of {ctx_id}")

    # Two different refusals, because they call for different actions. A chunk larger
    # than the stop line can NEVER be sent — waiting is useless, re-chunk. A chunk larger
    # than what is LEFT can be sent next window — wait, and resume. The first version of
    # this code told both cases to wait.
    if not verdict.possible:
        # Which call is too big decides the remedy, so say which. An oversized chunk needs
        # re-chunking; an oversized synthesis needs dropping, and costs no chunk answers
        # because the map pass runs and persists either way.
        usable = verdict.ceiling_tokens and int(verdict.ceiling_tokens
                                                * d.cfg.budget_stop_fraction)
        blame = ("a single chunk can never fit the session window — re-chunk smaller"
                 if usable and est.max_map_call_tokens > usable else
                 "the reduce pass reads every chunk answer at once and can never fit the "
                 "session window — re-run with reduce=False, or use fewer, larger chunks")
        return d.answer(f"ERROR: {blame}.\n\n" + budget.render(verdict, what=what))
    # The SMALLEST remaining call, not the largest: with `files` chunking one big file
    # sits beside many small ones, and refusing the whole batch because the biggest does
    # not fit — while announcing that not even one chunk fits — threw away the chunks that
    # did. The Gate admits what fits and defers the rest, which is the honest stop.
    if verdict.headroom_tokens is not None and est.min_call_tokens > verdict.headroom_tokens:
        wait = (f" Headroom returns in ~{budget.fmt_dur(verdict.spend.oldest_expires_in_s)}."
                if verdict.spend.oldest_expires_in_s else "")
        lack = ("not enough headroom for even one chunk" if todo else
                "not enough headroom for the synthesis over the cached answers")
        return d.answer(
            f"ERROR: session budget exhausted — {lack}." + wait + "\n\n"
            + budget.render(verdict, what=what)
            + f"\n\n_{len(cached)} chunk(s) already answered are safe on disk; re-run this "
              "tool once the window has rolled and it resumes there._"
        )

    if not todo:
        # Everything asked for is already answered. This is the payoff of persistence:
        # repeating the 30-minute run that returned nothing re-pays for no chunk. It is
        # not free with reduce=True — the synthesis above was budgeted for, and is what
        # the gates just admitted.
        merged = [SubResult(i, cached[i].answer, cached[i].itok, cached[i].otok,
                            model=cached[i].model) for i in sel]
        return _render(d, ctx_id, prompt, merged, sub_model, reduce=reduce, n=n,
                       max_chunks=max_chunks, from_cache=len(sel), deferred=0, key=rkey)

    gate = budget.Gate(d.cfg, now=d.clock)
    # Built lazily, one at a time inside each pool worker (sub_query_batch), not here —
    # holding all N prompt strings at once is ~= the whole context in RAM for the entire
    # multi-minute batch. Leaf 2's seek-based read_chunk is what makes per-worker reads
    # cheap enough that this stays lazy without re-introducing slow reads serially.
    prompts = [partial(_mk_prompt, d, ctx_id, prompt, i, n) for i in todo]
    # BEFORE the map, not only after it: this batch runs for tens of minutes on a large
    # context, and until it returns the only trace is N indistinguishable cli_spawn lines
    # under one rid — no ctx_id, and no denominator to count them against. That is the
    # difference between "377 of 408" and "possibly hung". est_tokens/headroom ride along
    # so the log says what the run was expected to cost.
    log_event(d.log, "sub_batch", phase="map_start", ctx_id=ctx_id,
              chunks=len(prompts), total_chunks=n, cached=len(cached) or None,
              est_tokens=est.total_tokens, est_s=round(est.seconds),
              headroom=verdict.headroom_tokens, ceiling=verdict.ceiling_tokens,
              concurrency=d.cfg.subquery_concurrency)
    fresh_results = sub_query_batch(
        d.cfg, prompts, sub_model, concurrency=d.cfg.subquery_concurrency,
        max_tokens=BATCH_MAX_TOKENS, indices=todo, gate=gate, system=MAP_SYSTEM,
        on_result=lambda r: _persist(d, ctx_id, rkey, keys[r.index], r),
    )
    deferred = sum(1 for r in fresh_results if (r.error or "").startswith("deferred"))
    by_index = {r.index: r for r in fresh_results}
    merged = []
    for i in sel:
        if i in by_index:
            merged.append(by_index[i])
        elif i in cached:
            c = cached[i]
            merged.append(SubResult(i, c.answer, c.itok, c.otok, model=c.model))
        else:
            merged.append(SubResult(i, "", 0, 0, error="not run"))
    return _render(d, ctx_id, prompt, merged, sub_model, reduce=reduce, n=n,
                   max_chunks=max_chunks, from_cache=len(cached), deferred=deferred, key=rkey)


def _render(d: Deps, ctx_id: str, prompt: str, res_list: list[SubResult], sub_model: str, *,
            reduce: bool, n: int, max_chunks: int, from_cache: int, deferred: int,
            key: str = "") -> str:
    """Render a (possibly resumed, possibly budget-stopped) batch.

    Split out of the tool because there are three ways to arrive here — everything already
    cached, a full run, and a run the gate stopped — and all three must produce the SAME
    report shape. Rendering them at three call sites is how a resumed run comes to read as
    though it had done the work.
    """
    itok = sum(r.input_tokens for r in res_list)
    otok = sum(r.output_tokens for r in res_list)
    # Deferral is not failure — it is scheduled work — so it is counted separately
    # everywhere: in the log, in the all-failed check, and in the notes.
    hard_errs = [r for r in res_list if r.error and not r.error.startswith("deferred")]
    # Models as REPORTED BY THE TRANSPORT, not the id we asked for: on OAuth
    # models.select maps a configured id to its closest subscription sibling.
    used = sorted({r.model for r in res_list if r.model}) or [sub_model]
    used_label = ", ".join(used)
    # Unconditional, not only on failure: the parent tool_call record carries neither a
    # chunk count nor token counts, so without this a successful batch is just N
    # indistinguishable cli_spawn lines under one rid. Same rid ties it back.
    log_event(d.log, "sub_batch", phase="map", ctx_id=ctx_id, chunks=len(res_list),
              errors=len(hard_errs), deferred=deferred or None, cached=from_cache or None,
              itok=itok, otok=otok,
              err_sample=hard_errs[0].error if hard_errs else None)
    # Every chunk failed for a REAL reason: that is a failed tool call, not a result with
    # notes. Returning the usual success string here is exactly how a dead login reads
    # back as "no findings". The ERROR prefix is what logsetup maps to outcome=error.
    if hard_errs and len(hard_errs) == len(res_list):
        return d.bound(f"ERROR: all {len(res_list)} chunk(s) failed — {hard_errs[0].error}")

    note = ""
    if max_chunks > 0 and max_chunks < n:
        note += f"\n_NOTE: limited to first {max_chunks} of {n} chunks._"
    if from_cache:
        note += (f"\n_RESUMED: {from_cache} chunk(s) reused from disk — no model call and "
                 "no tokens spent on them._")
    if hard_errs:
        note += f"\n_NOTE: {len(hard_errs)} of {len(res_list)} chunk(s) errored and were skipped._"
    if deferred:
        first = min((r.index for r in res_list
                     if (r.error or "").startswith("deferred")), default=0)
        note += (
            f"\n_**STOPPED AT THE BUDGET LINE**: {deferred} chunk(s) deferred to protect the "
            f"session window. Everything answered so far is on disk. Call this tool again with "
            f"the same ctx_id and prompt once the window has rolled — it resumes at chunk "
            f"{first} and re-pays for nothing._"
        )
    # A deferred chunk is NOT an error — it is work the budget gate postponed — so it
    # loses the ERROR marker a genuine failure keeps. Reading "[ERROR: deferred]" would
    # tell an operator to investigate a healthy stop.
    per_chunk = "\n".join(
        f"### chunk {r.index}\n" + (
            "" if not r.error
            else f"[{r.error}]" if r.error.startswith("deferred")
            else f"[ERROR: {r.error}]"
        ) + ("" if r.error else r.answer.strip())
        for r in res_list
    )

    def _raw(extra: str = "") -> str:
        head = (f"## Batch sub-query — map over {len(res_list)} chunks ({used_label}"
                f" · auth: {transport.auth_label(d.cfg)})\n"
                f"tokens: {itok:,} in / {otok:,} out{d.cost_note(sub_model, itok, otok)}"
                f"{note}{extra}\n\n")
        # Silently truncating here discards findings the caller ALREADY PAID FOR — and
        # with reduce=False they asked for every piece by name. Every answer is already on
        # disk, so hand back the path instead of half the report: 30 chunks of findings is
        # ~240 KB against a 128 KB cap, making this the normal case for a wide batch rather
        # than an edge one. The observed failure was worse still — the over-cap payload was
        # refused by the client outright and the whole 30-call batch read back as an error.
        full = head + per_chunk
        if key and encoded_len(full) > d.cfg.answer_cap_bytes:
            path = results.manifest_path(d.cfg, ctx_id, key)
            shown, kept, room = [], 0, d.cfg.answer_cap_bytes // 3
            for r in res_list:
                block = f"### chunk {r.index}\n" + (r.error or r.answer.strip())
                if kept + len(block) > room:
                    break
                shown.append(block)
                kept += len(block)
            return d.answer(
                head
                + f"_**{len(res_list)} findings ({len(per_chunk):,} bytes) exceed the "
                  f"{d.cfg.answer_cap_bytes:,}-byte reply cap — none are lost.** Every one is "
                  f"on disk, one JSON line per chunk:_\n`{path}`\n\n"
                  "_Read that file, or re-run with `reduce=True` for a synthesis. Showing "
                  f"the first {len(shown)} of {len(res_list)} below._\n\n"
                + "\n".join(shown))
        return d.answer(full)

    # findings = just the successful answers, for the reduce pass
    findings = "\n".join(f"[chunk {r.index}] {r.answer.strip()}"
                         for r in res_list if not r.error and r.answer.strip())
    if not reduce:
        return _raw()
    if not findings:
        return _raw("\n_(no findings to reduce)_")
    if estimate_tokens(findings) > 0.9 * d.cfg.sub_context_tokens:
        return _raw("\n_(findings too large to reduce in one pass — showing raw; use "
                    "fewer/larger chunks, or rlm_query for engine-side reduction)_")
    # A reduce over a partial map IS a partial answer. Saying so is the difference between
    # a synthesis and a synthesis that implies it read the whole document. Bound to a NEW
    # name rather than appended to `note`: `_raw` closed over `note`, so mutating it here
    # made the reduce-failure fallback promise "the synthesis below" above raw findings
    # with no synthesis below them.
    reduce_note = note + (
        f"\n_The synthesis below covers {len(res_list) - deferred} of {n} chunks "
        "— it is PARTIAL._" if deferred else "")
    return _reduce(d, findings, prompt, sub_model, ctx_id=ctx_id, note=reduce_note,
                   n_prompts=len(res_list), n_errors=len(hard_errs),
                   used_label=used_label, itok=itok, otok=otok, raw=_raw)


def _reduce(d: Deps, findings: str, prompt: str, sub_model: str, *, ctx_id: str, note: str,
            n_prompts: int, n_errors: int, used_label: str, itok: int, otok: int,
            raw: Callable[[str], str]) -> str:
    """Fold the per-chunk findings into one synthesis (one more sub-model call), and
    render it. Falls back to ``raw(extra)`` — the caller's per-chunk report — when the
    reduce call fails, so a failed synthesis never discards the findings it was given.

    The parameter list is wide because every item is load-bearing: findings/prompt/
    sub_model drive the call, ctx_id/n_errors the log record, and note/n_prompts/
    used_label/itok/otok the report header. ``itok``/``otok`` are locals here on purpose —
    the reduce call's tokens are added to them for the success header and the log, while
    ``raw`` still renders the caller's map-only totals on the failure path.
    """
    reduce_prompt = (
        "Reduce these independent per-chunk findings from one large document into a "
        "single answer.\n"
        f"Original request: {prompt}\n\n"
        f"Per-chunk findings:\n{findings}\n\n"
        "Synthesize ONE coherent, de-duplicated answer to the original request across "
        "all chunks. Use only what the findings contain; do not invent anything."
    )
    red = sub_query(d.cfg, reduce_prompt, sub_model, max_tokens=REDUCE_MAX_TOKENS,
                    system=TERSE_SYSTEM)
    if not red.error:
        itok += red.input_tokens
        otok += red.output_tokens
    # On success too: a reduce that works still spends a call and tokens, and the map
    # totals would otherwise understate every reduce batch. itok/otok are map+reduce.
    log_event(d.log, "sub_batch", phase="reduce", ctx_id=ctx_id, chunks=n_prompts,
              errors=n_errors, itok=itok, otok=otok, reduce_error=red.error)
    if red.error:
        return raw(f"\n_(reduce pass failed: {red.error}; showing raw findings)_")
    return d.answer(
        f"## Batch sub-query — map+reduce over {n_prompts} chunks ({used_label}"
        f" · auth: {transport.auth_label(d.cfg)})\n"
        f"tokens: {itok:,} in / {otok:,} out{d.cost_note(sub_model, itok, otok)}{note}\n\n"
        f"{red.answer.strip()}"
    )
