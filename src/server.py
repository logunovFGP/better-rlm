#!/usr/bin/env python3
"""RLM MCP Server — Recursive Language Models over oversized contexts.

Fork of eesb99/rlm-mcp, modernized for the current rlms engine:
  * Anthropic models — Sonnet 5 root (Opus 4.8 override), Haiku 4.5 sub-LLM,
    selected via a strategy (src/models.py): under Claude Code OAuth each role
    maps to the closest subscription-supported sibling.
  * Auth reuses Claude Code's OAuth (run `claude setup-token`) — NO API key needed;
    ANTHROPIC_API_KEY is an opt-in fallback only.
  * Docker REPL sandbox by default (host exec only as a documented fallback).
  * External on-disk context store so giant logs/dumps never enter the root context.
  * Bounded tool output — raw-content tools capped tight (~4 KB) so file content never
    floods the root context; synthesis answers (rlm_query/sub_query) bound generously.

Run:  python -m src.server   (stdio transport)
"""

from __future__ import annotations

import os
import re
import shutil
import textwrap
from collections.abc import Callable
from functools import partial
from itertools import islice

from mcp.server.fastmcp import FastMCP

from . import auth, batch, budget, models, sources, transport
from .chunking import STRATEGIES, chunk_text
from .deps import Deps
from .engine import ReplSession, query_checkpoint_path, run_query
from .subquery import sub_query
from .logsetup import log_event, logged_tool, note_startup
# meta_block keeps its old private name: tests reach for srv._meta_block. It and
# skipped_block are pure ContextMeta presentation, moved to output.py to keep this file
# under 800 lines; skipped_block is called from meta_block there, not from here.
from .output import meta_block as _meta_block
from .sandbox_reap import container_image_status
from .shutdown import install_shutdown_hooks

#: THE COMPOSITION ROOT. The one place in this package that resolves real config, opens
#: real log handlers and points at the real store — built once, at import, and passed
#: down explicitly from here. Nothing below reaches for config on its own, which is what
#: lets a test construct its own Deps over tmp_path instead of patching these globals
#: (see deps.py for the pollution bug that motivated it).
#: stdout stays the JSON-RPC channel; logs go to a per-PID file with bounded retention
#: and stderr stays WARNING-only (see logsetup.py).
DEPS = Deps.create()
CFG = DEPS.cfg      # read-only aliases, kept because ~20 call sites below are plain
LOG = DEPS.log      # presentation over these and reading DEPS.cfg everywhere adds noise
STORE = DEPS.store  # without adding a seam — the seam is DEPS itself.
mcp = FastMCP("rlm")

_repl: ReplSession | None = None


def _get_repl() -> ReplSession:
    global _repl
    if _repl is None:
        _repl = ReplSession(CFG)
    return _repl


# Output bounding lives on Deps so batch.py can reach it without importing server.
# These stay as one-liners because the raw-content tools below read better with them.
def _bound(text: str) -> str:
    return DEPS.bound(text)


def _answer(text: str) -> str:
    return DEPS.answer(text)


def _resolve_root_model(model_override: str) -> str:
    """Map the model_override arg to a concrete model, per the documented values:
    '' -> Role.ROOT;  'opus' -> Role.OVERRIDE;  an explicit model id is used as-is
    (but still sibling-mapped under OAuth).
    """
    mo = (model_override or "").strip().lower()
    if mo == "opus":
        return models.select(CFG, models.Role.OVERRIDE)
    if mo:
        return models.map_model(CFG, model_override)
    return models.select(CFG, models.Role.ROOT)


# ============================================================================
# Loading & inspection
# ============================================================================
@mcp.tool()
@logged_tool
def rlm_load_context(source: str, source_type: str = "auto") -> str:
    """Load context from inline text, a file path, or a directory into the
    external store. Use this FIRST for any oversized input (big logs, repo dumps,
    k8s manifest sets) — PREFER it over reading a large file directly (Read/cat)
    when the input is > ~200 KB / ~5,000 lines or would be truncated. The content
    is kept on disk and never returned, so it never enters this conversation's
    context. Returns a ctx_id; then call rlm_query / rlm_sub_query / rlm_exec.

    source_type: auto | text | file | dir (auto detects path vs inline text).
    """
    # A URL would otherwise be stored as ~60 bytes of *text* and reported as a
    # successful load, so every later answer would describe the link, not the data.
    if re.match(r"^[a-z][a-z0-9+.-]*://", source.strip(), re.I):
        return _bound(
            f"ERROR: {source.strip()[:80]} is a URL, not a path — loading it would store "
            "the link text, not its content. Fetch it in the sandbox instead:\n"
            "  1. rlm_exec(\"import requests; "
            "open('/workspace/data','wb').write(requests.get(URL).content)\")\n"
            "  2. rlm_exec(\"print(open('/workspace/owner').read())\") -> 3rd line is the "
            "host path of /workspace\n"
            "  3. rlm_load_file(\"<that path>/data\") -> full chunk / grep / query support\n"
            "No Docker (or sandbox: local)? Fetch on the host — curl also carries auth "
            "headers/cookies:\n"
            "  curl -sL \"<url>\" -o /tmp/data && rlm_load_file(\"/tmp/data\")"
        )
    st = source_type
    if st == "auto":
        if os.path.isdir(source):
            st = "dir"
        elif os.path.isfile(source):
            st = "file"
        else:
            st = "text"
    if st == "dir":
        meta = STORE.load_dir(source)
    elif st == "file":
        meta = STORE.load_file(source, data_type="text")
    else:
        meta = STORE.load_text(source)
    return _bound("## Context loaded\n" + _meta_block(meta))


@mcp.tool()
@logged_tool
def rlm_load_file(path: str, data_type: str = "text") -> str:
    """Load a single file into the external store — use this INSTEAD of Read for any
    file too large to read directly (big logs, dumps, exports). data_type: text | log
    | pdf (pdf needs the optional pypdf extra). Large text/log files are referenced in
    place (no copy). Returns a ctx_id; then call rlm_query / rlm_sub_query / rlm_exec."""
    meta = STORE.load_file(path, data_type=data_type)
    return _bound("## File loaded\n" + _meta_block(meta))


def _sources() -> dict[str, sources.Source]:
    """Re-read the registry on every call: a running server holds its own copy of src/,
    and this file is the one thing operators edit routinely (see sources.py)."""
    return sources.load_sources(CFG.sources_file)


@mcp.tool()
@logged_tool
def rlm_list_sources() -> str:
    """List the named external sources THIS deployment declares — live inputs (cluster
    logs, a metrics or trace backend, a journal, an audit API) that rlm_load_source can
    pull into the store. FREE, no model call. Call this FIRST when the question is about
    a RUNNING SYSTEM rather than a file on disk: the server ships no sources of its own,
    so what exists here is entirely whatever the operator declared."""
    reg = _sources()
    if not reg:
        return _bound(
            "No sources declared — this server ships none by design.\n"
            f"Declare them in `{CFG.sources_file}`: a mapping of name -> "
            "{description, command, timeout_s, max_bytes}, where `command` may contain "
            "`{placeholder}` parameters and `${ENV_VAR}` references. "
            "See README → 'Live sources'."
        )
    lines = [f"## Declared sources ({len(reg)}) — from {CFG.sources_file}", ""]
    for s in reg.values():
        lines.append(f"- `{s.name}` — {s.description or '(no description)'}")
        lines.append(f"  - params: {', '.join(s.params) or 'none'}"
                     f"  |  timeout {s.timeout_s}s, cap {s.max_bytes:,} B"
                     + ("  |  stderr merged into content" if s.merge_stderr else ""))
        # Surface a missing/stale credential HERE, so the user can be asked before a call
        # is attempted rather than after it fails.
        if s.credential_file:
            try:
                sources.check_credential(s)
                state = "ready"
            except ValueError:
                state = "MISSING or STALE — ask the user for a fresh short-lived token"
            lines.append(f"  - credential: {s.credential_file}"
                         + (f" (max age {s.credential_max_age_h:g}h)"
                            if s.credential_max_age_h else "")
                         + f" — {state}")
    # _answer, not _bound: this is an ENUMERATION, not file content. The 4 KB raw cap
    # exists so a giant log cannot flood the root context, but truncating the list of
    # what you may call hides sources from the one tool whose job is to reveal them —
    # measured: 15 sources with useful descriptions already exceed 4 KB.
    return _answer("\n".join(lines))


@mcp.tool()
@logged_tool
def rlm_load_source(name: str, params: dict[str, str] | None = None) -> str:
    """Run a declared external source (see rlm_list_sources) and load its output into the
    store. Use this INSTEAD of shelling out and reading the result: a live log stream or a
    metrics/trace export is exactly the oversized input this server exists for, and its
    content never enters this conversation — only a ctx_id comes back. Then grep / exec /
    chunk / query it like any other context.

    name: a key from rlm_list_sources.
    params: values for that source's placeholders. They are substituted as literal argv
    tokens and the command never goes through a shell, so a value cannot inject a second
    command. Missing or unrecognised parameter names are rejected rather than guessed.

    A partial result (non-zero exit, timeout, size cap) still returns a ctx_id but is
    labelled WITH WARNINGS — a truncated log answers "does X appear?" with a false no."""
    reg = _sources()
    src = reg.get(name)
    if src is None:
        return _bound(
            f"ERROR: unknown source '{name}'. Declared here: {', '.join(reg) or 'none'} "
            f"(registry: {CFG.sources_file}). Call rlm_list_sources."
        )
    try:
        argv = sources.resolve(src, params)
    except ValueError as exc:
        return _bound(f"ERROR: {exc}")
    try:
        run = STORE.load_command(argv, source=f"source:{name}",
                                 timeout_s=src.timeout_s, max_bytes=src.max_bytes,
                                 merge_stderr=src.merge_stderr)
    except OSError as exc:
        return _bound(f"ERROR: could not run source '{name}': {exc}  (argv[0]={argv[0]!r})")
    # argv[0] only, never the rendered argv: a template may expand ${TOKEN} into it.
    log_event(LOG, "source_run", source=name, argv0=argv[0], rc=run.returncode,
              bytes=run.meta.bytes, truncated=run.truncated, timed_out=run.timed_out)

    notes = []
    if run.timed_out:
        notes.append(f"TIMED OUT after {src.timeout_s}s")
    if run.truncated:
        notes.append(f"TRUNCATED at the {src.max_bytes:,}-byte cap")
    if run.returncode != 0:
        notes.append(f"exit code {run.returncode}")
    if run.stderr_tail:
        notes.append(f"stderr tail: {run.stderr_tail[:400]}")
    if run.meta.bytes == 0:
        # Zero bytes is the most dangerous result there is: a dead tunnel, an expired
        # session, a wrong selector and a program that logs to stderr all look exactly
        # like "nothing matched". The stderr cause is the one an operator cannot guess,
        # so name it and its fix rather than only advising suspicion.
        notes.append(
            "ZERO BYTES CAPTURED — do NOT read this as 'nothing happened'. Likely causes, "
            "in order: (1) the command writes its output to STDERR, not stdout, so none of "
            "it was saved — set `merge_stderr: true` on this source"
            + ("" if src.merge_stderr else " (currently OFF)")
            + "; (2) the output was redirected or paged away inside the command itself; "
              "(3) a dead tunnel, expired session, or wrong selector / namespace / time "
              "window. Co-verify with a query that MUST return data before reporting any "
              "negative finding from this context"
        )
    if not run.ok and run.meta.bytes == 0:
        # Nothing came back and the command failed: that is an error, not an empty
        # context someone will later query and get confident nonsense from.
        STORE.drop(run.meta.ctx_id)
        return _bound(f"ERROR: source '{name}' produced no output — " + "; ".join(notes))
    # Header keys off `notes`, not `run.ok`: hitting the size cap exits 0 but still
    # yields a partial context, and that must not read as a clean load.
    body = ("## Source loaded — WITH WARNINGS\n" if notes else "## Source loaded\n") \
        + _meta_block(run.meta)
    if notes:
        body += "\n\n**This context is INCOMPLETE — say so in any answer drawn from it:** " \
                + "; ".join(notes)
    return _bound(body)


@mcp.tool()
@logged_tool
def rlm_inspect_context(ctx_id: str, preview_lines: int = 40) -> str:
    """Return metadata plus a small bounded preview (head lines) for a loaded
    context. Use to sanity-check what was loaded without pulling the content."""
    meta = STORE.get(ctx_id)
    with open(meta.content_path, encoding="utf-8", errors="replace") as fh:
        preview = "".join(islice(fh, preview_lines)).rstrip("\n")
    return _bound(f"## Context {ctx_id}\n{_meta_block(meta)}\n\n### Preview (first {preview_lines} lines)\n```\n{preview}\n```")


@mcp.tool()
@logged_tool
def rlm_chunk_context(ctx_id: str, strategy: str = "", size: int = 0, overlap: int = 0) -> str:
    """Split a loaded context into chunks and record the chunk index on the
    context. strategy: lines | paragraphs | functions | headings | semantic | files
    (default from config). 'size' = lines-per-chunk for the lines strategy.
    Returns chunk count + per-chunk metadata (no content). Chunk defaults stay well
    under the sub-model's context ceiling (config: sub_context_tokens) so sub-queries fit.

    Leave strategy empty for the content-aware default: `files` when the context is a
    dir load or carries `===== FILE:` markers. File boundaries survive edits elsewhere,
    so a re-analysis after changing 3 of 1,000 files re-asks about 3 chunks, not 1,000 --
    under `lines`, one edit shifts every later boundary and nearly nothing is reused."""
    # Empty strategy = the content-aware default: `files` for a dir load or a bundle
    # with FILE markers (boundaries that survive edits, so cached answers keep hitting),
    # else the configured default. See batch.default_strategy for why this matters.
    strategy = strategy or batch.default_strategy(DEPS, STORE.get(ctx_id))
    if strategy not in STRATEGIES:
        return _bound(f"ERROR: unknown strategy '{strategy}'. Choose from {', '.join(STRATEGIES)}.")
    text = STORE.read_text(ctx_id)
    chunks = chunk_text(
        text, strategy,
        chunk_lines=size or CFG.chunk_lines,
        chunk_chars=CFG.chunk_chars,
        overlap=overlap or CFG.chunk_overlap,
    )
    # text=text: hand over the decode we already have, so the byte-offset pass does not
    # read and decode the whole context a second time alongside this copy.
    STORE.set_chunks(ctx_id, strategy, [c.as_dict() for c in chunks], text=text)
    lines = [f"## Chunked {ctx_id} — {len(chunks)} chunks ({strategy})", ""]
    for c in chunks[:25]:
        lbl = f" {c.label}" if c.label else ""
        lines.append(f"- [{c.index}] lines {c.start_line}-{c.end_line}, "
                     f"~{c.est_tokens:,} tok{lbl}")
    if len(chunks) > 25:
        lines.append(f"… (+{len(chunks) - 25} more)")
    return _bound("\n".join(lines))


# ============================================================================
# Deterministic retrieval (no model call, no cost)
# ============================================================================
@mcp.tool()
@logged_tool
def rlm_grep(ctx_id: str, pattern: str, ignore_case: bool = False, max_matches: int = 50) -> str:
    """Search a loaded context for a regex and return matching lines with 1-based line
    numbers — deterministic and FREE (no model call). PREFER this over rlm_sub_query for
    "find / locate / where is X" questions over a huge log or dump. Streams the file so it
    works on multi-GB contexts; stops after max_matches (the result notes if more may exist).

    pattern: a Python regular expression. ignore_case: case-insensitive match.
    """
    try:
        matches, capped = STORE.grep(ctx_id, pattern, ignore_case=ignore_case,
                                     max_matches=max(1, max_matches))
        if not matches:
            return _bound(f"## grep `{pattern}` in {ctx_id}\n\nNo matches.")
        note = f" (stopped at {len(matches)} — more may exist)" if capped else ""
        body = "\n".join(f"{ln}: {txt}" for ln, txt in matches)
        return _bound(
            f"## grep `{pattern}` in {ctx_id} — {len(matches)} match(es){note}\n"
            f"```\n{body}\n```"
        )
    except re.error as exc:  # a bad pattern deserves better than the generic message
        return _bound(f"ERROR: invalid regex '{pattern}': {exc}")


@mcp.tool()
@logged_tool
def rlm_read_chunk(ctx_id: str, chunk_index: int) -> str:
    """Return the raw content of a single chunk (bounded), after rlm_chunk_context.
    Deterministic and FREE (no model call) — use to inspect exactly what a chunk holds,
    e.g. a chunk that rlm_sub_query_batch flagged. Chunk indices are 0-based."""
    meta = STORE.get(ctx_id)
    if not meta.chunks:
        return _bound(f"ERROR: {ctx_id} has not been chunked; call rlm_chunk_context first.")
    if chunk_index < 0 or chunk_index >= len(meta.chunks):
        return _bound(f"ERROR: chunk {chunk_index} out of range (0..{len(meta.chunks) - 1}).")
    ch = meta.chunks[chunk_index]
    lbl = f" — {ch['label']}" if ch.get("label") else ""
    body = STORE.read_chunk(ctx_id, chunk_index)
    return _bound(
        f"## chunk {chunk_index}/{len(meta.chunks) - 1} of {ctx_id}{lbl}\n"
        f"lines {ch['start_line']}-{ch['end_line']}, ~{ch['est_tokens']:,} tok\n"
        f"```\n{body}\n```"
    )


# ============================================================================
# Context lifecycle
# ============================================================================
@mcp.tool()
@logged_tool
def rlm_list_contexts() -> str:
    """List all loaded contexts with size / token / chunk metadata (no content).
    Use to see what's in the store and pick a ctx_id; pair with rlm_drop_context to
    evict ones you no longer need."""
    metas = STORE.list_metas()
    if not metas:
        return _bound("No contexts loaded.")
    lines = [f"## Loaded contexts ({len(metas)})", ""]
    for m in metas:
        lines.append(
            f"- `{m.ctx_id}` — {m.bytes:,} B, ~{m.est_tokens:,} tok, {m.lines:,} lines, "
            f"{len(m.chunks)} chunk(s) — {m.source_type}/{m.data_type} — {m.source}"
        )
    return _bound("\n".join(lines))


@mcp.tool()
@logged_tool
def rlm_drop_context(ctx_id: str) -> str:
    """Evict a loaded context from the store, freeing its metadata and any materialized
    copy. A file loaded in place is NOT deleted — only the store's reference to it. Ids
    are not reusable after dropping."""
    if not STORE.drop(ctx_id):
        return _bound(f"ERROR: unknown context id: {ctx_id}")
    return _bound(f"Dropped `{ctx_id}`.")


# ============================================================================
# Querying
# ============================================================================
@mcp.tool()
@logged_tool
def rlm_estimate(ctx_id: str, prompt: str = "", max_chunks: int = 0, reduce: bool = True) -> str:
    """ESTIMATE BEFORE YOU EXECUTE. Forecast what rlm_sub_query_batch over this context
    would cost — chunks, model calls, tokens, wall time — and judge it against what is
    left in the current session window. Costs nothing: no model call is made.

    Call this BEFORE any batch over a large context. A 103-chunk run silently consumed
    ~60% of a 4-hour window and returned nothing when it was interrupted; this is the
    tool that would have said so in advance.

    Reports how many chunks are ALREADY ANSWERED on disk -- cached by chunk CONTENT, so a
    re-loaded file counts too -- and the forecast covers only the work that remains. A run
    that does not fit this window is not a dead end: start it, let it stop at the budget
    line, and call the batch again next window. Also prints the ceiling for rlm_query on
    this context, which cannot be estimated, only bounded.
    """
    return batch.estimate(DEPS, ctx_id, prompt, max_chunks, reduce)


@mcp.tool()
@logged_tool
def rlm_budget() -> str:
    """Show the session-window token budget: what this server has spent inside the
    rolling window, the ceiling it is gating against (configured, learned, or unknown),
    and when headroom next grows. No model call."""
    return batch.budget_report(DEPS)


@mcp.tool()
@logged_tool
def rlm_query(ctx_id: str, question: str, model_override: str = "", fresh: bool = False) -> str:
    """Answer a question over a loaded context using the FULL recursive RLM loop:
    the root model (Sonnet 5 by default) writes Python in the configured sandbox to
    explore the context and delegates chunk-level work to Haiku 4.5, then returns
    a synthesized answer. This is the headline tool for giant inputs — the
    content stays in the sandbox; only the answer comes back.

    The sandbox is Docker by default but is configurable: under `sandbox: local` /
    RLM_SANDBOX=local the model-written Python runs ON THE HOST with no isolation.
    Call rlm_status to see which is live before trusting this with untrusted input.

    model_override: '' (Sonnet) | 'opus' (Opus 4.8, hardest tasks) | explicit model id.
    Models are resolved by the selection strategy (closest OAuth sibling when on OAuth).

    COST CANNOT BE ESTIMATED, ONLY BOUNDED. The root model decides at run time how many
    sub-calls to make. rlm_estimate prints the ceiling: what config permits (usually
    several times a whole session window) and what query_timeout_s allows in practice.

    GATED AND RESUMABLE. Every model call passes a hard floor at `budget_stop_fraction`
    of the session window, so the run stops itself before the wall. Any stop -- budget,
    timeout, error threshold -- checkpoints the transcript and the sandbox's REPL
    variables; call this tool again with the SAME ctx_id and question and it continues
    from the failed iteration instead of starting over. fresh=True discards a checkpoint.
    Spend is ledgered, so the next estimate sees it. Prefer rlm_exec/rlm_grep (free) and
    rlm_sub_query_batch (estimable) unless the question truly needs multi-hop reasoning.
    """
    root_model = _resolve_root_model(model_override)
    sub_model = models.select(DEPS.cfg, models.Role.SUB)
    text = DEPS.store.read_text(ctx_id)
    ckpt = query_checkpoint_path(DEPS.cfg, ctx_id, question, root_model, sub_model)
    res = run_query(DEPS.cfg, text, question, root_model, sub_model,
                    checkpoint=ckpt, fresh=fresh)
    resumed = (f" · resumed from iteration {res['resumed_from']}"
               if res.get("resumed_from") else "")
    if res.get("limit"):
        # ERROR-prefixed so logsetup records outcome=error; the partial answer still
        # travels, because a stopped run that found something is not a dead loss.
        partial_answer = res["answer"]
        if res.get("resumable"):
            how = (f"\n**Resumable.** The transcript and REPL state are checkpointed. Call "
                   f"rlm_query again with the same ctx_id and question to continue from "
                   f"iteration {res['next_iteration']}"
                   + (" once the window has rolled" if res["limit"] == "SessionBudgetError" else "")
                   + "; fresh=True discards the checkpoint instead.")
        else:
            how = ("\nRaise query_timeout_s / query_max_errors in config.yaml, or narrow the "
                   "question. rlm_exec and rlm_grep need no model call at all.")
        return DEPS.answer(
            f"ERROR: rlm_query stopped on {res['limit']} — {res['limit_detail']}{resumed}\n"
            + (f"\n--- best answer before the limit ---\n{partial_answer}\n"
               if partial_answer else "\nNo partial answer was available.\n")
            + how
        )
    rows = "\n".join(
        f"  - {r['model']}: {r['calls']} calls, "
        f"{r['input_tokens']:,} in / {r['output_tokens']:,} out"
        + (f", ${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "")
        for r in res["usage"]
    )
    total = res["cost_usd"]
    return DEPS.answer(
        f"## RLM answer (root: {res['root_model']}, sub: {res['sub_model']}"
        f" · auth: {transport.auth_label(DEPS.cfg)}{resumed})\n\n{res['answer']}\n\n"
        f"---\n**Model routing / usage:**\n{rows}\n"
        + (f"**Total cost:** ${total:.4f}  |  " if total is not None else "")
        + f"**Time:** {res['execution_time']}s"
    )


@mcp.tool()
@logged_tool
def rlm_sub_query(ctx_id: str, prompt: str, chunk_index: int = -1) -> str:
    """Run a single cheap sub-model query over a context (or one chunk of it).
    Use for targeted questions. If chunk_index < 0 the whole context is used — only do
    that when it fits the sub-model's context window (config: sub_context_tokens),
    else chunk first."""
    return batch.one(DEPS, ctx_id, prompt, chunk_index)


@mcp.tool()
@logged_tool
def rlm_sub_query_batch(ctx_id: str, prompt: str, max_chunks: int = 0, reduce: bool = True,
                        fresh: bool = False) -> str:
    """Map a prompt over the chunks of a context via Haiku 4.5 (concurrent), then by
    default REDUCE the per-chunk findings into one synthesized answer. Map-reduce that
    YOU orchestrate (vs rlm_query, where the engine does it). Auto-chunks with config
    defaults if the context wasn't chunked yet.

    RESUMABLE AND BUDGET-AWARE. Every chunk answer is written to disk the moment it
    lands, so calling this again with the SAME ctx_id and prompt reuses what is already
    answered and pays only for the rest — after a crash, after a budget stop, or in a new
    session tomorrow. The run also stops ITSELF at `budget_stop_fraction` of the session
    window instead of being killed at the wall, reporting the remaining chunks as
    deferred. Call rlm_estimate first to see size, cost and headroom before starting.

    fresh=True: discard the cached answers for this exact prompt+chunking and re-ask.

    reduce=True  (default): one coherent, de-duplicated synthesis across all chunks —
                 best for "summarize / aggregate / what's the overall picture".
    reduce=False: the raw per-chunk findings concatenated — when you want every piece.
    """
    return batch.run(DEPS, ctx_id, prompt, max_chunks, reduce, fresh)



# ============================================================================
# Sandbox REPL — exec & variables
# ============================================================================
@mcp.tool()
@logged_tool
def rlm_exec(code: str, ctx_id: str = "") -> str:
    """Execute Python in the configured sandbox. If ctx_id is given, that context is
    loaded as the REPL variable `context` first. Use for grep/parse/aggregate
    over a loaded context (e.g. count errors, bucket log lines by hour). Output
    is bounded — print only what you need.

    Docker by default; under `sandbox: local` / RLM_SANDBOX=local this code runs ON
    THE HOST with no isolation. rlm_status reports which is live."""
    repl = _get_repl()
    if ctx_id and repl.loaded_ctx != ctx_id:
        repl.load_context(STORE.read_text(ctx_id), ctx_id)
    out, err = repl.execute(code)
    body = f"### stdout\n```\n{out}\n```"
    if err.strip():
        body += f"\n### stderr\n```\n{err}\n```"
    return _bound(body)


@mcp.tool()
@logged_tool
def rlm_status(probe: bool = False) -> str:
    """Report configuration, the active auth + model-selection strategy with the
    RESOLVED models, loaded contexts, and sandbox/Docker availability.

    probe=True additionally spends one tiny sub-model call to prove the transport
    can actually authenticate. Off by default because it costs a call; worth it
    the moment anything returns an auth error, since every line below can be
    correct while the login itself is dead."""
    docker_ok = shutil.which("docker") is not None
    claude_cli = shutil.which(CFG.cli_path)
    creds = auth.auth_status()  # which explicit credential exists (display only)
    try:
        resolved = auth.resolve_auth_mode(CFG)      # "oauth" (CLI) | "apikey" (SDK)
        # NB: not `transport` -- that name is the module imported above, and
        # shadowing it here broke transport.auth_label() at runtime.
        transport_desc = "cli (claude CLI)" if resolved == "oauth" else "api (Anthropic SDK)"
    except Exception as exc:
        resolved, transport_desc = None, f"UNRESOLVED — {exc}"
    mode = resolved or "apikey"
    root, override, sub = (
        models.map_for_mode(mode, models.configured(CFG, r))
        for r in (models.Role.ROOT, models.Role.OVERRIDE, models.Role.SUB)
    )
    ids = STORE.list_ids()
    try:
        src_line = f"{len(_sources())} declared"
    except Exception as exc:   # a broken registry must show HERE, not only on first use
        src_line = f"UNREADABLE — {type(exc).__name__}: {exc}"
    return _bound(
        "## RLM MCP status\n"
        f"- provider: {CFG.provider}   (anthropic is the only supported provider and needs "
        "no key — it uses the claude CLI login; any other raises NotImplementedError at "
        "the first model call, because only this one routes through the session budget)\n"
        f"- mode (configured): {CFG.mode}   (auto | claude-cli | api; RLM_MODE env overrides)\n"
        f"- transport (resolved): {transport_desc}\n"
        f"- claude CLI ({CFG.cli_path}): {'found at ' + claude_cli if claude_cli else 'NOT FOUND on PATH'}"
        f"  | system-prompt mode: {CFG.cli_system_prompt_mode}, safe-mode: {CFG.cli_safe_mode}\n"
        f"- explicit credentials present: {creds}  (claude-cli mode needs NONE — the CLI uses its own login)\n"
        f"- selection policy: {models.policy_name(mode)}\n"
        f"- resolved models → root: {root} | override: {override} | sub: {sub}\n"
        f"- configured models → root: {CFG.root_model} | override: {CFG.root_model_override} | sub: {CFG.sub_model}\n"
        f"- sandbox: {CFG.sandbox} (image: {CFG.sandbox_image})\n"
        f"- sandbox container: {_container_status()}\n"
        f"- auth (resolved): {transport.auth_label(CFG)}\n"
        f"- max_depth: {CFG.max_depth}, max_iterations: {CFG.max_iterations}\n"
        f"- output cap: {CFG.output_cap_bytes} B (raw content) / {CFG.answer_cap_bytes:,} B (answers), "
        f"sub-query concurrency: {CFG.subquery_concurrency}\n"
        f"- docker CLI available: {docker_ok}\n"
        f"- loaded contexts ({len(ids)}): {', '.join(ids[:20]) or 'none'}\n"
        f"- store dir: {CFG.store_dir}\n"
        f"- sources: {src_line}  (registry: {CFG.sources_file})"
        f"{_cli_login_line()}"
        f"{_auth_probe_line() if probe else ''}"
    )


def _container_status() -> str:
    """Image freshness of the live sandbox. Never creates a container just to look.

    Deliberately NOT cached: the defect is the IMAGE moving under an UNCHANGED
    container, so a cache keyed on the container id would never notice the very thing
    this reports. Two docker calls (~200 ms) on a status tool is the right trade.
    """
    if not CFG.use_docker:
        return "n/a (sandbox: local runs on the host)"
    try:
        return container_image_status(CFG.sandbox_image,
                                     _repl.container_id() if _repl else None)
    except Exception as exc:                      # noqa: BLE001
        return f"unknown ({type(exc).__name__})"


def _cli_login_line() -> str:
    """Is the `claude` CLI logged in? FREE (no model call, ~215 ms) and it answers the
    one question every other line here can be right about while the login is dead.

    Always shown, because the failure it catches is invisible otherwise: being signed
    in to Claude Code does not sign in the CLI, and nothing else in this report says so.
    """
    if resolve_auth_mode_safe() != "oauth":
        return ""                                  # SDK path: no CLI login involved
    st = transport.cli_auth_status(CFG)
    if st is None:
        return "\n- cli login: UNKNOWN — could not run `claude auth status`"
    if st.get("loggedIn"):
        return (f"\n- cli login: ok (method: {st.get('authMethod', '?')}, "
                f"provider: {st.get('apiProvider', '?')})")
    return ("\n- cli login: NOT LOGGED IN — every model-backed tool will fail.\n"
            + textwrap.indent(transport.AUTH_REMEDIATION, "    "))


def resolve_auth_mode_safe() -> str:
    try:
        return auth.resolve_auth_mode(CFG)
    except Exception:                              # noqa: BLE001
        return "unknown"


def _auth_probe_line() -> str:
    """One real sub-model call — proves the transport end to end. Skipped when the free
    check above already knows the login is dead: paying for a call whose answer is
    already known is the same waste the batch fail-fast exists to stop."""
    if resolve_auth_mode_safe() == "oauth":
        st = transport.cli_auth_status(CFG)
        if st is not None and not st.get("loggedIn"):
            return "\n- auth probe: SKIPPED — cli login is dead, a probe would only confirm it"
    # The same reasoning one step further: past the stop line the transport refuses the
    # probe before dispatch, and the refusal printed as "auth probe: FAILED — session
    # budget stop", sending an operator to re-authenticate a login that was fine.
    # rlm_status is what people reach for when something looks broken, so it must not
    # misname what is broken.
    try:
        budget.check_or_raise(DEPS.cfg, 16, now=DEPS.clock)
    except budget.BudgetStopError as stop:
        return f"\n- auth probe: SKIPPED — session budget reached, not an auth problem ({stop})"
    try:
        res = sub_query(DEPS.cfg, "Reply with exactly: ok",
                        models.select(DEPS.cfg, models.Role.SUB), max_tokens=16)
    except Exception as exc:                      # noqa: BLE001 - report, never raise
        return f"\n- auth probe: FAILED — {type(exc).__name__}: {exc}"
    if res.error:
        return f"\n- auth probe: FAILED — {res.error}"
    return f"\n- auth probe: ok — sub-model replied {res.answer.strip()[:20]!r}"


def main() -> None:
    # Graceful teardown (close the sandbox container) on SIGTERM/SIGINT + clean EOF.
    install_shutdown_hooks(LOG, lambda: _repl.close() if _repl is not None else None)
    try:
        mode = auth.resolve_auth_mode(CFG)
        # Held until this process serves something — most spares never do.
        note_startup(mode=CFG.mode, transport=mode,
                     root=models.select(CFG, models.Role.ROOT),
                     sub=models.select(CFG, models.Role.SUB), sandbox=CFG.sandbox)
    except Exception as exc:
        # A server that cannot resolve a transport is worth a file even if no tool
        # is ever called: this record is the only trace of why it is useless.
        log_event(LOG, "startup", mode=CFG.mode, transport="unresolved",
                  err=f"{type(exc).__name__}: {exc}")
    mcp.run()


if __name__ == "__main__":
    main()
