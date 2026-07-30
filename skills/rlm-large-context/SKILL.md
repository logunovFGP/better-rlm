---
name: rlm-large-context
description: Reason over an input too large to read directly — multi-MB/GB logs, big JSON/CSV/data exports, PDFs, repo or directory dumps, k8s manifest sets, document corpora — by routing it through the `rlm` MCP tools (load into the external store, then grep/exec/query) instead of reading it inline and blowing the context window. Use when a file is larger than ~200 KB or ~5,000 lines, when a Read/cat would be truncated, and ESPECIALLY for questions whose work grows with the input: "count/label/classify EVERY entry", "aggregate across the whole log", "which pairs contradict each other", "cross-reference all N records", "what's the overall picture across these 1000 documents", "summarize this whole directory or repo". Also for "find X across this huge <log|dump|dataset|dir>". NOT for small files (read those directly) or for locating one symbol or definition in a code repo (grep/git grep is faster and free).
---

# RLM — reason over oversized contexts

The `rlm` MCP server holds content in an external on-disk store plus a sandboxed Docker
REPL and **returns only the findings** — a multi-GB log never enters this conversation.
Reach for these tools instead of reading a huge file.

## Route by the shape of the question, not the size of the file

Size decides *whether* to load. The question's complexity decides *which* tool.

| Question shape | Tool | Cost |
|---|---|---|
| "Where is X / does X appear" — one lookup | `rlm_grep(ctx_id, pattern)` | free, no model call |
| Exact counts, sums, buckets, parsing | `rlm_exec(code, ctx_id)` | free, no model call |
| "Label / classify / judge **every** entry"; aggregate over the whole input | `rlm_chunk_context` → `rlm_sub_query_batch` | one cheap call per chunk |
| One targeted semantic question, input under ~200K tokens | `rlm_sub_query(ctx_id, prompt)` | one cheap call |
| Cross-referencing, contradicting pairs, multi-hop over a corpus | `rlm_query(ctx_id, question)` | recursive loop, tens of seconds |
| Hardest reasoning | `rlm_query(..., model_override="opus")` | most expensive |

Start at the top. `rlm_grep` and `rlm_exec` answer more questions than expected and spend
no tokens on the content.

## What this is uniquely good at

From the RLM paper (Zhang, Kraska & Khattab, MIT CSAIL — *Recursive Language Models*),
which measured a median **+26% over context compaction** and **+13% over Claude Code**
across four long-context tasks at comparable cost:

- **Dense aggregation** — every element must be examined (label all, count by category,
  find the peak). Compaction fails here by design: it "presumes that some details that
  appear early in the prompt can safely be forgotten." Measured +28% over the base model.
- **Pairwise / cross-referencing** — work grows quadratically ("which entries conflict?").
  Unaided frontier models scored **≤0.1%** where the recursive path reached 58%.
- **Deep research over a corpus** — multi-hop across ~1000 documents (6–11M tokens).
- **Whole-repo or whole-directory understanding** — reasoning that spans many files at once
  (evaluated up to 4.2M tokens).

## What it is *not* for

- Small files — just `Read` them.
- Finding one symbol or definition in a repo — `grep`/`git grep` is faster and free. The
  paper's own result: single-needle tasks degrade to plain keyword heuristics, so paying for
  recursion buys nothing.
- **Code repositories specifically:** prefer `rlm_exec` over `rlm_query`. On the paper's
  code-repository benchmark the *no-sub-calling* variant beat every recursive variant, and
  an ordinary coding agent with context offloading scored higher still. Load the directory,
  then drive it with your own Python.

## Workflow

1. **Load** — `rlm_load_file(path, data_type=text|log|pdf)` for one file, or
   `rlm_load_context(source, source_type=auto|text|file|dir)` for inline text, a file, or a
   whole directory. Returns a `ctx_id`; the content stays on disk.
2. **Sanity-check** (optional) — `rlm_inspect_context(ctx_id)` for metadata plus a head preview.
3. **Route** — pick from the table above.
4. **Housekeeping** — `rlm_list_contexts` to see what is loaded, `rlm_drop_context(ctx_id)`
   to evict one, `rlm_read_chunk(ctx_id, i)` to read exactly what a flagged chunk holds,
   `rlm_status` for mode, resolved models and Docker availability.

## Examples

- *"Most frequent error and its peak hour in this 800 MB log."*
  → `rlm_load_file(path)` → `rlm_exec` to bucket by hour — free, no model needed.
- *"Label every one of these 40k support tickets by root cause, then total them."*
  → `rlm_load_file(path)` → `rlm_chunk_context` → `rlm_sub_query_batch(ctx_id, "label each…")`
- *"Which of these 900 config entries contradict each other?"*
  → `rlm_load_context(dir)` → `rlm_query(ctx_id, "find contradicting pairs")`
- *"What do these 400 k8s manifests configure, and what is misconfigured?"*
  → `rlm_load_context(path, source_type="dir")` → `rlm_query(ctx_id, "summarize workloads, flag security misconfigurations")`
- *"Does OOMKilled appear in this huge log, and where?"*
  → `rlm_load_file(path)` → `rlm_grep(ctx_id, "OOMKilled")` — free.

## Notes

- Content never enters this chat — only the bounded findings (~4 KB for raw content, a
  larger bound for synthesized answers).
- `rlm_query` is a real multi-turn loop (tens of seconds) with a long-tailed cost
  distribution. For a single question prefer `rlm_sub_query`; for exact answers `rlm_exec`.
- Defaults: Sonnet 5 root, Opus 4.8 override, Haiku 4.5 sub. Auth reuses your Claude Code
  login via the `claude` CLI — no API key. Reported input-token counts on that path
  under-count the piped prompt, so treat its cost figures as a floor.
