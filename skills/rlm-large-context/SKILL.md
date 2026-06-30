---
name: rlm-large-context
description: Analyze, search, summarize, or answer questions over an oversized input that is too large to read directly — multi-MB/GB logs, large data files, big JSON/CSV exports, repo or directory dumps, k8s manifest sets — by routing it through the `rlm` MCP tools (load into the external store, then query) instead of reading it inline and blowing the context window. Use when a file is larger than ~200 KB or ~5,000 lines, when a Read/cat would be truncated, or when asked "what's in / find X across / summarize / aggregate this huge <log|file|dataset|directory>". NOT for small files (read those directly) or symbol-level code navigation (use grep/git grep).
---

# RLM — reason over oversized contexts

The `rlm` MCP server (Recursive Language Models) answers questions over inputs far
larger than a context window. The content is held in an external on-disk store and a
sandboxed Docker REPL; **only the findings come back** — so a multi-GB log never
enters this conversation. Reach for these tools instead of reading a huge file.

## When to use

Use the `rlm` tools when the input is too big to read directly:
- A log, dump, export, or data file bigger than ~200 KB / ~5,000 lines.
- A `Read`/`cat` that would be truncated, or you already hit "file too large".
- "What's in / find X across / summarize / aggregate this huge <log|file|dataset|dir>".

Do **NOT** use them for:
- Small files — just `Read` them.
- Finding a symbol/definition in a code repo — `grep`/`git grep` is faster and cheaper.
  RLM is for *oversized single contexts*, not repository navigation.

## Workflow

1. **Load** (never read the big file directly):
   - `rlm_load_file(path)` — a single file.
   - `rlm_load_context(source, source_type=auto|text|file|dir)` — inline text, a file, or a directory.
   - Returns a `ctx_id` + metadata (bytes, lines, est_tokens). The content stays on disk.
2. **Ask** — the headline path:
   - `rlm_query(ctx_id, question)` — the root model writes Python in the sandbox to
     explore the context and delegates chunk work to a cheap sub-model, then returns a
     synthesized answer. Best for "analyze / aggregate / what's the pattern" questions.
     Runs a real multi-turn loop (tens of seconds).
   - `rlm_query(ctx_id, question, model_override="opus")` — for the hardest reasoning.
3. **Or go targeted / faster:**
   - `rlm_inspect_context(ctx_id)` — metadata + a small head preview (sanity-check what loaded).
   - `rlm_chunk_context(ctx_id)` then `rlm_sub_query_batch(ctx_id, prompt)` — map a prompt
     over every chunk (map-reduce) for per-section findings.
   - `rlm_sub_query(ctx_id, prompt)` — one cheap sub-model pass (seconds) for a single question.
   - `rlm_exec(code, ctx_id)` — run your own Python over the content in the sandbox; the
     content is the `context` variable. Best for exact counts / greps / aggregations.
4. **Check setup:** `rlm_status` — shows the active mode, resolved models, and Docker availability.

## Examples

- *"Find the most frequent error and its peak hour in this 800 MB app log."*
  → `rlm_load_file("/var/log/app.log")` → `rlm_query(ctx_id, "Most frequent ERROR type, and the hour with the most errors?")`
- *"Summarize what these 400 k8s manifests configure and flag misconfigurations."*
  → `rlm_load_context("/path/manifests", source_type="dir")` → `rlm_query(ctx_id, "Summarize the workloads and flag security misconfigurations.")`
- *"Exact count of lines containing OOMKilled in this huge log."*
  → `rlm_load_file(path)` → `rlm_exec("print(sum('OOMKilled' in l for l in context.splitlines()))", ctx_id)`

## Notes
- The content never enters this chat — only the bounded (~4 KB) answer/findings return.
- `rlm_query` is multi-turn (tens of seconds). For a quick single question prefer
  `rlm_sub_query` or `rlm_exec`.
- Defaults: Sonnet 5 root, Opus 4.8 override, Haiku 4.5 sub. Auth reuses your Claude Code
  login via the `claude` CLI (no API key); `rlm_status` shows the resolved transport.
