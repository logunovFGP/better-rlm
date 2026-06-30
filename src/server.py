#!/usr/bin/env python3
"""RLM MCP Server — Recursive Language Models over oversized contexts.

Fork of eesb99/rlm-mcp, modernized for the current rlms engine:
  * Anthropic models — Sonnet 4.6 root (Opus 4.8 override), Haiku 4.5 sub-LLM,
    selected via a strategy (src/models.py): under Claude Code OAuth each role
    maps to the closest subscription-supported sibling.
  * Auth reuses Claude Code's OAuth (run `claude setup-token`) — NO API key needed;
    ANTHROPIC_API_KEY is an opt-in fallback only.
  * Docker REPL sandbox by default (host exec only as a documented fallback).
  * External on-disk context store so giant logs/dumps never enter the root context.
  * Bounded tool output (~4 KB) — only metadata/findings flow back to the model.

Run:  python -m src.server   (stdio transport)
"""

from __future__ import annotations

import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

from . import auth, models
from .chunking import STRATEGIES, chunk_text
from .config import HAIKU_CONTEXT_TOKENS, cost_usd, load_config
from .context_store import ContextStore
from .engine import ReplSession, run_query
from .output import bound_output
from .subquery import sub_query, sub_query_batch

# Logs go to stderr; stdout is reserved for the MCP JSON-RPC stream.
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("rlm-mcp")

CFG = load_config()
STORE = ContextStore(CFG)
mcp = FastMCP("rlm")

_repl: ReplSession | None = None


def _get_repl() -> ReplSession:
    global _repl
    if _repl is None:
        _repl = ReplSession(CFG)
    return _repl


def _bound(text: str) -> str:
    return bound_output(text, CFG.output_cap_bytes)


def _meta_block(meta) -> str:
    return (
        f"**ctx_id:** `{meta.ctx_id}`\n"
        f"- source: {meta.source}  ({meta.source_type}/{meta.data_type})\n"
        f"- size: {meta.bytes:,} bytes, {meta.lines:,} lines, ~{meta.est_tokens:,} tokens\n"
        f"- files: {meta.file_count}\n"
        f"- sha256: {meta.sha256[:16]}…"
    )


def _resolve_root_model(model_override: str) -> str:
    """Map the model_override arg to a concrete model via the selection strategy.

    '' / default / sonnet / root -> Role.ROOT;  opus / override / hardest -> Role.OVERRIDE;
    an explicit model id is used as-is (but still sibling-mapped under OAuth).
    """
    mo = (model_override or "").strip().lower()
    if mo in ("opus", "override", "hardest", "max"):
        return models.select(CFG, models.Role.OVERRIDE)
    if model_override and mo not in ("", "default", "sonnet", "root"):
        return models.map_model(CFG, model_override)
    return models.select(CFG, models.Role.ROOT)


# ============================================================================
# Loading & inspection
# ============================================================================
@mcp.tool()
def rlm_load_context(source: str, source_type: str = "auto") -> str:
    """Load context from inline text, a file path, or a directory into the
    external store. Use this FIRST for any oversized input (big logs, repo dumps,
    k8s manifest sets) — the content is kept on disk and never returned, so it
    never enters this conversation's context. Returns a ctx_id + metadata.

    source_type: auto | text | file | dir (auto detects path vs inline text).
    """
    try:
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
    except Exception as exc:
        return _bound(f"ERROR loading context: {exc}")


@mcp.tool()
def rlm_load_file(path: str, data_type: str = "text") -> str:
    """Load a single file into the external store. data_type: text | log | pdf
    (pdf needs the optional pypdf extra). Large text/log files are referenced in
    place (no copy). Returns a ctx_id + metadata."""
    try:
        meta = STORE.load_file(path, data_type=data_type)
        return _bound("## File loaded\n" + _meta_block(meta))
    except Exception as exc:
        return _bound(f"ERROR loading file: {exc}")


@mcp.tool()
def rlm_inspect_context(ctx_id: str, preview_lines: int = 40) -> str:
    """Return metadata plus a small bounded preview (head lines) for a loaded
    context. Use to sanity-check what was loaded without pulling the content."""
    try:
        meta = STORE.get(ctx_id)
        head: list[str] = []
        with open(meta.content_path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= preview_lines:
                    break
                head.append(line.rstrip("\n"))
        preview = "\n".join(head)
        return _bound(f"## Context {ctx_id}\n{_meta_block(meta)}\n\n### Preview (first {preview_lines} lines)\n```\n{preview}\n```")
    except Exception as exc:
        return _bound(f"ERROR: {exc}")


@mcp.tool()
def rlm_chunk_context(ctx_id: str, strategy: str = "", size: int = 0, overlap: int = 0) -> str:
    """Split a loaded context into chunks and record the chunk index on the
    context. strategy: lines | paragraphs | functions | headings | semantic | files
    (default from config). 'size' = lines-per-chunk for the lines strategy.
    Returns chunk count + per-chunk metadata (no content). Chunk defaults stay
    well under Haiku's 200K-token ceiling so sub-queries fit."""
    try:
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
        STORE.set_chunks(ctx_id, strategy, [c.as_dict() for c in chunks])
        lines = [f"## Chunked {ctx_id} — {len(chunks)} chunks ({strategy})", ""]
        for c in chunks[:25]:
            lbl = f" {c.label}" if c.label else ""
            lines.append(f"- [{c.index}] lines {c.start_line}-{c.end_line}, "
                         f"~{c.est_tokens:,} tok{lbl}")
        if len(chunks) > 25:
            lines.append(f"… (+{len(chunks) - 25} more)")
        return _bound("\n".join(lines))
    except Exception as exc:
        return _bound(f"ERROR chunking: {exc}")


# ============================================================================
# Querying
# ============================================================================
@mcp.tool()
def rlm_query(ctx_id: str, question: str, model_override: str = "") -> str:
    """Answer a question over a loaded context using the FULL recursive RLM loop:
    the root model (Sonnet 4.6 by default) writes Python in a Docker sandbox to
    explore the context and delegates chunk-level work to Haiku 4.5, then returns
    a synthesized answer. This is the headline tool for giant inputs — the
    content stays in the sandbox; only the answer comes back.

    model_override: '' (Sonnet) | 'opus' (Opus 4.8, hardest tasks) | explicit model id.
    Models are resolved by the selection strategy (closest OAuth sibling when on OAuth).
    """
    try:
        root_model = _resolve_root_model(model_override)
        sub_model = models.select(CFG, models.Role.SUB)
        text = STORE.read_text(ctx_id)
        res = run_query(CFG, text, question, root_model, sub_model)
        rows = "\n".join(
            f"  - {r['model']}: {r['calls']} calls, "
            f"{r['input_tokens']:,} in / {r['output_tokens']:,} out, ${r['cost_usd']:.4f}"
            for r in res["usage"]
        )
        return _bound(
            f"## RLM answer (root: {res['root_model']}, sub: {res['sub_model']})\n\n{res['answer']}\n\n"
            f"---\n**Model routing / usage:**\n{rows}\n"
            f"**Total cost:** ${res['cost_usd']:.4f}  |  **Time:** {res['execution_time']}s"
        )
    except Exception as exc:
        return _bound(f"ERROR in rlm_query: {exc}")


@mcp.tool()
def rlm_sub_query(ctx_id: str, prompt: str, chunk_index: int = -1) -> str:
    """Run a single Haiku 4.5 sub-query over a context (or one chunk of it).
    Use for cheap, targeted questions. If chunk_index < 0 the whole context is
    used — only do that when it fits Haiku's 200K-token window (else chunk first)."""
    try:
        meta = STORE.get(ctx_id)
        if chunk_index >= 0:
            body = STORE.read_chunk(ctx_id, chunk_index)
        else:
            if meta.est_tokens > 0.9 * HAIKU_CONTEXT_TOKENS:
                return _bound(
                    f"ERROR: context ~{meta.est_tokens:,} tokens exceeds Haiku's "
                    f"{HAIKU_CONTEXT_TOKENS:,} window. Run rlm_chunk_context then "
                    f"rlm_sub_query_batch, or use rlm_query."
                )
            body = STORE.read_text(ctx_id)
        sub_model = models.select(CFG, models.Role.SUB)
        res = sub_query(f"{prompt}\n\n--- CONTEXT ---\n{body}", sub_model)
        if res.error:
            return _bound(f"ERROR ({sub_model}): {res.error}")
        return _bound(
            f"## Sub-query answer ({sub_model})\n\n{res.answer}\n\n---\n"
            f"tokens: {res.input_tokens:,} in / {res.output_tokens:,} out"
        )
    except Exception as exc:
        return _bound(f"ERROR in rlm_sub_query: {exc}")


@mcp.tool()
def rlm_sub_query_batch(ctx_id: str, prompt: str, max_chunks: int = 0) -> str:
    """Map a prompt over all chunks of a context via Haiku 4.5 (concurrent), then
    return a bounded aggregation of per-chunk findings + total tokens/cost. This
    is map-reduce that YOU orchestrate (vs rlm_query, where the engine does).
    Auto-chunks with config defaults if the context wasn't chunked yet."""
    try:
        meta = STORE.get(ctx_id)
        if not meta.chunks:
            text = STORE.read_text(ctx_id)
            chunks = chunk_text(text, CFG.chunk_strategy, chunk_lines=CFG.chunk_lines,
                                chunk_chars=CFG.chunk_chars, overlap=CFG.chunk_overlap)
            STORE.set_chunks(ctx_id, CFG.chunk_strategy, [c.as_dict() for c in chunks])
            meta = STORE.get(ctx_id)
        n = len(meta.chunks)
        sel = range(n if max_chunks <= 0 else min(max_chunks, n))
        prompts = [f"{prompt}\n\n--- CHUNK {i + 1}/{n} ---\n{STORE.read_chunk(ctx_id, i)}" for i in sel]
        sub_model = models.select(CFG, models.Role.SUB)
        results = sub_query_batch(prompts, sub_model, concurrency=CFG.subquery_concurrency)
        itok = sum(r.input_tokens for r in results)
        otok = sum(r.output_tokens for r in results)
        cost = cost_usd(sub_model, itok, otok)
        findings = "\n".join(
            f"### chunk {r.index}\n" + (f"[ERROR: {r.error}]" if r.error else r.answer.strip())
            for r in results
        )
        header = (f"## Batch sub-query over {len(prompts)} chunks ({sub_model})\n"
                  f"tokens: {itok:,} in / {otok:,} out  |  cost: ${cost:.4f}\n\n")
        if max_chunks > 0 and max_chunks < n:
            header += f"_NOTE: limited to first {max_chunks} of {n} chunks._\n\n"
        return _bound(header + findings)
    except Exception as exc:
        return _bound(f"ERROR in rlm_sub_query_batch: {exc}")


# ============================================================================
# Sandbox REPL — exec & variables
# ============================================================================
@mcp.tool()
def rlm_exec(code: str, ctx_id: str = "") -> str:
    """Execute Python in the Docker sandbox. If ctx_id is given, that context is
    loaded as the REPL variable `context` first. Use for grep/parse/aggregate
    over a loaded context (e.g. count errors, bucket log lines by hour). Output
    is bounded — print only what you need."""
    try:
        repl = _get_repl()
        if ctx_id and repl.loaded_ctx != ctx_id:
            repl.load_context(STORE.read_text(ctx_id), ctx_id)
        out, err = repl.execute(code)
        body = f"### stdout\n```\n{out}\n```"
        if err.strip():
            body += f"\n### stderr\n```\n{err}\n```"
        return _bound(body)
    except Exception as exc:
        return _bound(f"ERROR in rlm_exec: {exc}")


@mcp.tool()
def rlm_set_variable(name: str, value: str) -> str:
    """Set a variable in the sandbox REPL. `value` is parsed as JSON if possible
    (so dicts/lists/numbers work), else treated as a string."""
    try:
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            parsed = value
        _get_repl().set_var(name, parsed)
        return _bound(f"set `{name}` = {parsed!r}")
    except Exception as exc:
        return _bound(f"ERROR: {exc}")


@mcp.tool()
def rlm_get_variable(name: str) -> str:
    """Return the repr of a variable from the sandbox REPL (bounded)."""
    try:
        return _bound(f"`{name}` = {_get_repl().get_var(name)}")
    except Exception as exc:
        return _bound(f"ERROR: {exc}")


@mcp.tool()
def rlm_set_answer(value: str) -> str:
    """Set the sandbox REPL's final-answer signal (answer['content']/['ready'])."""
    try:
        _get_repl().set_answer(value)
        return _bound("answer set (ready=True)")
    except Exception as exc:
        return _bound(f"ERROR: {exc}")


@mcp.tool()
def rlm_status() -> str:
    """Report configuration, the active auth + model-selection strategy with the
    RESOLVED models, loaded contexts, and sandbox/Docker availability."""
    import shutil
    docker_ok = shutil.which("docker") is not None
    auth_mode = auth.auth_status()
    claude_cli = shutil.which(CFG.cli_path)
    transport = ("cli (claude CLI)" if auth_mode == "oauth"
                 else "api (Anthropic SDK)" if auth_mode == "apikey" else "none")
    strat = models.get_strategy(CFG, auth_mode)
    root = strat.model_for(models.Role.ROOT)
    override = strat.model_for(models.Role.OVERRIDE)
    sub = strat.model_for(models.Role.SUB)
    ids = STORE.list_ids()
    return _bound(
        "## RLM MCP status\n"
        f"- auth: {auth_mode}  (oauth = reusing Claude Code login via `claude setup-token`)\n"
        f"- transport: {transport}  "
        f"(oauth drives the `claude` CLI; apikey uses the Anthropic SDK)\n"
        f"- claude CLI ({CFG.cli_path}): {'found at ' + claude_cli if claude_cli else 'NOT FOUND on PATH'}"
        f"  | system-prompt mode: {CFG.cli_system_prompt_mode}, safe-mode: {CFG.cli_safe_mode}\n"
        f"- selection strategy: {type(strat).__name__}\n"
        f"- resolved models → root: {root} | override: {override} | sub: {sub}\n"
        f"- configured models → root: {CFG.root_model} | override: {CFG.root_model_override} | sub: {CFG.sub_model}\n"
        f"- sandbox: {CFG.sandbox} (image: {CFG.sandbox_image})\n"
        f"- max_depth: {CFG.max_depth}, max_iterations: {CFG.max_iterations}\n"
        f"- output cap: {CFG.output_cap_bytes} bytes, sub-query concurrency: {CFG.subquery_concurrency}\n"
        f"- docker CLI available: {docker_ok}\n"
        f"- loaded contexts ({len(ids)}): {', '.join(ids[:20]) or 'none'}\n"
        f"- store dir: {CFG.store_dir}"
    )


def main() -> None:
    logger.info("Starting RLM MCP server (auth=%s root=%s sub=%s sandbox=%s)",
                auth.auth_status(),
                models.select(CFG, models.Role.ROOT),
                models.select(CFG, models.Role.SUB),
                CFG.sandbox)
    mcp.run()


if __name__ == "__main__":
    main()
