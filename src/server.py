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
from functools import partial
from itertools import islice

from mcp.server.fastmcp import FastMCP

from . import auth, models, sources, transport
from .chunking import STRATEGIES, chunk_text
from .config import (
    PROVIDER_KEY_ENV,
    cost_usd,
    estimate_tokens,
    load_config,
)
from .context_store import ContextStore
from .engine import ReplSession, run_query
from .logsetup import configure_logging, log_event, logged_tool, note_startup
from .output import bound_output
from .sandbox_reap import container_image_status
from .shutdown import install_shutdown_hooks
from .subquery import sub_query, sub_query_batch

CFG = load_config()
# stdout is the JSON-RPC channel; detailed logs go to a per-PID file in log_dir with
# bounded retention, and stderr stays WARNING-only (see logsetup.py).
LOG = configure_logging(CFG)
STORE = ContextStore(CFG)
mcp = FastMCP("rlm")

_repl: ReplSession | None = None


def _get_repl() -> ReplSession:
    global _repl
    if _repl is None:
        _repl = ReplSession(CFG)
    return _repl


def _bound(text: str) -> str:
    # Raw-content tools (load/inspect/chunk/grep/read/list/exec/status): keep the tiny cap so
    # raw file content never floods the root context.
    return bound_output(text, CFG.output_cap_bytes)


def _answer(text: str) -> str:
    # Synthesis tools (rlm_query / rlm_sub_query[_batch]): the answer IS the
    # deliverable (already bounded by the model's max_output_tokens), so bound it
    # generously, not at the raw-content cap.
    return bound_output(text, CFG.answer_cap_bytes)


def _cost_note(model: str, itok: int, otok: int) -> str:
    """`  |  cost: $x.xxxx` when report_cost is on, else nothing. Off by default: the
    rate table is Anthropic-only and the CLI path under-counts input tokens, so a
    printed figure would be confidently wrong."""
    if not CFG.report_cost:
        return ""
    return f"  |  cost: ${cost_usd(model, itok, otok):.4f}"


#: Skips that are the POINT of the skip lists, not a surprise (node_modules, .env).
_EXPECTED_SKIPS = frozenset({"skip-dir", "skip-name"})


def _skipped_block(meta) -> str:
    """Report what a dir load did not ingest, loudly ONLY when it is surprising.

    A partial load that reports success is worse than a failure: an answer over the
    remaining files is wrong BY OMISSION and nothing contradicts it. Observed: 173 of
    184 files loaded, the 11 missing ones the exact subject of the question.

    But node_modules being skipped is the intent, not an incident. Screaming INCOMPLETE
    on every JS project trains the reader to skip the line -- and then it is not there
    when 11 source files really do vanish. So policy exclusions get one quiet count and
    the surprising ones get the alarm.
    """
    counts = dict(getattr(meta, "skipped_counts", None) or {})
    if not counts:
        return ""
    expected = {k: v for k, v in counts.items() if k in _EXPECTED_SKIPS}
    surprising = {k: v for k, v in counts.items() if k not in _EXPECTED_SKIPS}
    out = ""
    if expected:
        detail = ", ".join(f"{k} x{v:,}" for k, v in sorted(expected.items()))
        out += f"\n- excluded by policy: {sum(expected.values()):,} ({detail})"
    if surprising:
        n = sum(surprising.values())
        total = meta.file_count + sum(counts.values())
        detail = ", ".join(f"{k} x{v:,}" for k, v in sorted(surprising.items()))
        out += (f"\n- ⚠ **INCOMPLETE — {n:,} readable file(s) were NOT loaded** "
                f"(of {total:,} found; {detail}).\n"
                "  Any answer from this context is wrong by omission if it needed one:\n")
        sample = [e for e in (getattr(meta, "skipped", None) or [])
                  if e.split(":", 1)[0] not in _EXPECTED_SKIPS][:20]
        out += "".join(f"    {e}\n" for e in sample).rstrip("\n")
        if n > len(sample):
            out += f"\n    … and {n - len(sample):,} more"
    return out


def _meta_block(meta) -> str:
    return (
        f"**ctx_id:** `{meta.ctx_id}`\n"
        f"- source: {meta.source}  ({meta.source_type}/{meta.data_type})\n"
        f"- size: {meta.bytes:,} bytes, {meta.lines:,} lines, ~{meta.est_tokens:,} tokens\n"
        f"- files: {meta.file_count}\n"
        f"- sha256: {meta.sha256[:16]}…"
        + _skipped_block(meta)
    )


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
    under the sub-model's context ceiling (config: sub_context_tokens) so sub-queries fit."""
    strategy = strategy or CFG.chunk_strategy
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
def rlm_query(ctx_id: str, question: str, model_override: str = "") -> str:
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
    """
    root_model = _resolve_root_model(model_override)
    sub_model = models.select(CFG, models.Role.SUB)
    text = STORE.read_text(ctx_id)
    res = run_query(CFG, text, question, root_model, sub_model)
    if res.get("limit"):
        # ERROR-prefixed so logsetup records outcome=error; the partial answer still
        # travels, because a stopped run that found something is not a dead loss.
        partial = res["answer"]
        return _answer(
            f"ERROR: rlm_query stopped on {res['limit']} — {res['limit_detail']}\n"
            + (f"\n--- best answer before the limit ---\n{partial}\n" if partial
               else "\nNo partial answer was available.\n")
            + "\nRaise query_timeout_s / query_max_errors in config.yaml, or narrow the "
              "question. rlm_exec and rlm_grep need no model call at all."
        )
    rows = "\n".join(
        f"  - {r['model']}: {r['calls']} calls, "
        f"{r['input_tokens']:,} in / {r['output_tokens']:,} out"
        + (f", ${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "")
        for r in res["usage"]
    )
    total = res["cost_usd"]
    return _answer(
        f"## RLM answer (root: {res['root_model']}, sub: {res['sub_model']}"
        f" · auth: {transport.auth_label(CFG)})\n\n{res['answer']}\n\n"
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
    meta = STORE.get(ctx_id)
    if chunk_index >= 0:
        body = STORE.read_chunk(ctx_id, chunk_index)
    else:
        if meta.est_tokens > 0.9 * CFG.sub_context_tokens:
            return _bound(
                f"ERROR: context ~{meta.est_tokens:,} tokens exceeds Haiku's "
                f"{CFG.sub_context_tokens:,}-token window. Run rlm_chunk_context then "
                f"rlm_sub_query_batch, or use rlm_query."
            )
        body = STORE.read_text(ctx_id)
    sub_model = models.select(CFG, models.Role.SUB)
    res = sub_query(f"{prompt}\n\n--- CONTEXT ---\n{body}", sub_model)
    if res.error:
        return _bound(f"ERROR ({sub_model}): {res.error}")
    return _answer(
        f"## Sub-query answer ({res.model or sub_model}"
        f" · auth: {transport.auth_label(CFG)})\n\n{res.answer}\n\n---\n"
        f"tokens: {res.input_tokens:,} in / {res.output_tokens:,} out"
    )


def _mk_batch_prompt(ctx_id: str, prompt: str, i: int, n: int) -> str:
    return f"{prompt}\n\n--- CHUNK {i + 1}/{n} ---\n{STORE.read_chunk(ctx_id, i)}"


@mcp.tool()
@logged_tool
def rlm_sub_query_batch(ctx_id: str, prompt: str, max_chunks: int = 0, reduce: bool = True) -> str:
    """Map a prompt over the chunks of a context via Haiku 4.5 (concurrent), then by
    default REDUCE the per-chunk findings into one synthesized answer. Map-reduce that
    YOU orchestrate (vs rlm_query, where the engine does it). Auto-chunks with config
    defaults if the context wasn't chunked yet.

    reduce=True  (default): one coherent, de-duplicated synthesis across all chunks —
                 best for "summarize / aggregate / what's the overall picture".
    reduce=False: the raw per-chunk findings concatenated — when you want every piece.
    """
    meta = STORE.get(ctx_id)
    if not meta.chunks:
        text = STORE.read_text(ctx_id)
        chunks = chunk_text(text, CFG.chunk_strategy, chunk_lines=CFG.chunk_lines,
                            chunk_chars=CFG.chunk_chars, overlap=CFG.chunk_overlap)
        STORE.set_chunks(ctx_id, CFG.chunk_strategy, [c.as_dict() for c in chunks],
                         text=text)   # reuse this decode; don't make a second copy
        meta = STORE.get(ctx_id)
        del text   # the full decode has served chunking; don't hold it for the whole batch
    n = len(meta.chunks)
    sel = range(n if max_chunks <= 0 else min(max_chunks, n))
    # Built lazily, one at a time inside each pool worker (sub_query_batch), not here —
    # holding all N prompt strings at once is ~= the whole context in RAM for the entire
    # multi-minute batch. Leaf 2's seek-based read_chunk is what makes per-worker reads
    # cheap enough that this stays lazy without re-introducing slow reads serially.
    prompts = [partial(_mk_batch_prompt, ctx_id, prompt, i, n) for i in sel]
    sub_model = models.select(CFG, models.Role.SUB)
    # BEFORE the map, not only after it: this batch runs for tens of minutes on a
    # large context, and until it returns the only trace is N indistinguishable
    # cli_spawn lines under one rid — no ctx_id, and no denominator to count them
    # against. That is the difference between "377 of 408" and "possibly hung".
    log_event(LOG, "sub_batch", phase="map_start", ctx_id=ctx_id,
              chunks=len(prompts), total_chunks=n,
              concurrency=CFG.subquery_concurrency)
    results = sub_query_batch(prompts, sub_model, concurrency=CFG.subquery_concurrency)
    itok = sum(r.input_tokens for r in results)
    otok = sum(r.output_tokens for r in results)
    errs = [r for r in results if r.error]
    # Models as REPORTED BY THE TRANSPORT, not the id we asked for: on OAuth
    # models.select maps a configured id to its closest subscription sibling.
    used = sorted({r.model for r in results if r.model}) or [sub_model]
    used_label = ", ".join(used)
    # Unconditional, not only on failure: the parent tool_call record carries neither
    # a chunk count nor token counts, so without this a successful batch is just N
    # indistinguishable cli_spawn lines under one rid. Same rid ties it back.
    log_event(LOG, "sub_batch", phase="map", ctx_id=ctx_id, chunks=len(prompts),
              errors=len(errs), itok=itok, otok=otok,
              err_sample=errs[0].error if errs else None)
    # Every chunk failed: that is a failed tool call, not a result with notes.
    # Returning the usual success string here is exactly how a dead login reads
    # back as "no findings". The ERROR prefix is what logsetup maps to outcome=error.
    if errs and len(errs) == len(results):
        return _bound(f"ERROR: all {len(results)} chunk(s) failed — {errs[0].error}")

    note = ""
    if max_chunks > 0 and max_chunks < n:
        note += f"\n_NOTE: limited to first {max_chunks} of {n} chunks._"
    if errs:
        note += f"\n_NOTE: {len(errs)} of {len(prompts)} chunk(s) errored and were skipped._"
    per_chunk = "\n".join(
        f"### chunk {r.index}\n" + (f"[ERROR: {r.error}]" if r.error else r.answer.strip())
        for r in results
    )

    def _raw(extra: str = "") -> str:
        head = (f"## Batch sub-query — map over {len(prompts)} chunks ({used_label}"
                f" · auth: {transport.auth_label(CFG)})\n"
                f"tokens: {itok:,} in / {otok:,} out{_cost_note(sub_model, itok, otok)}"
                f"{note}{extra}\n\n")
        return _answer(head + per_chunk)

    # findings = just the successful answers, for the reduce pass
    findings = "\n".join(f"[chunk {r.index}] {r.answer.strip()}"
                         for r in results if not r.error and r.answer.strip())
    if not reduce:
        return _raw()
    if not findings:
        return _raw("\n_(no findings to reduce)_")
    if estimate_tokens(findings) > 0.9 * CFG.sub_context_tokens:
        return _raw("\n_(findings too large to reduce in one pass — showing raw; use "
                    "fewer/larger chunks, or rlm_query for engine-side reduction)_")

    # REDUCE: fold the per-chunk findings into one synthesis (one more sub-model call).
    reduce_prompt = (
        "Reduce these independent per-chunk findings from one large document into a "
        "single answer.\n"
        f"Original request: {prompt}\n\n"
        f"Per-chunk findings:\n{findings}\n\n"
        "Synthesize ONE coherent, de-duplicated answer to the original request across "
        "all chunks. Use only what the findings contain; do not invent anything."
    )
    red = sub_query(reduce_prompt, sub_model, max_tokens=4096)
    if not red.error:
        itok += red.input_tokens
        otok += red.output_tokens
    # On success too: a reduce that works still spends a call and tokens, and the map
    # totals would otherwise understate every reduce batch. itok/otok are map+reduce.
    log_event(LOG, "sub_batch", phase="reduce", ctx_id=ctx_id, chunks=len(prompts),
              errors=len(errs), itok=itok, otok=otok, reduce_error=red.error)
    if red.error:
        return _raw(f"\n_(reduce pass failed: {red.error}; showing raw findings)_")
    return _answer(
        f"## Batch sub-query — map+reduce over {len(prompts)} chunks ({used_label}"
        f" · auth: {transport.auth_label(CFG)})\n"
        f"tokens: {itok:,} in / {otok:,} out{_cost_note(sub_model, itok, otok)}{note}\n\n"
        f"{red.answer.strip()}"
    )


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
        f"- provider: {CFG.provider}   (anthropic needs no key — it uses the claude CLI login; "
        f"any other needs its {PROVIDER_KEY_ENV.get(CFG.provider, '<PROVIDER>_API_KEY')}; "
        "RLM_PROVIDER env overrides)\n"
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
    try:
        res = sub_query("Reply with exactly: ok",
                        models.select(CFG, models.Role.SUB), max_tokens=16)
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
