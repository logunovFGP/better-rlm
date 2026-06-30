# Phase 5 — Validation transcript

Harness: `scripts/validate.py` (reproducible). The **no-key** portions run the
load/inspect/chunk/exec path end-to-end in the Docker sandbox; the **LLM-routed**
tools (`rlm_query`, `rlm_sub_query[_batch]`) are wired and listed but require
`ANTHROPIC_API_KEY`, which was **unset** in this environment.

## Registration (acceptance: `claude mcp list` shows it; CLI connects)
```
$ claude mcp add -s user rlm -- bash /Users/workpc/PROJECTS/tests/rlm-mcp/run_server.sh
Added stdio MCP server rlm … to user config (File modified: ~/.claude.json)
$ claude mcp list
rlm: bash /Users/workpc/PROJECTS/tests/rlm-mcp/run_server.sh - ✔ Connected
```
User scope → surfaces in **both** the CLI and the desktop app (they share `~/.claude`).
All 12 tools enumerated by `mcp.list_tools()`.

## (a) Large log — >1M tokens, no root-context overflow
Synthesized log: **5.3 MB, 80,000 lines**.
- `rlm_load_file(big.log, "log")` → referenced in place (no copy), metadata returned.
- `rlm_chunk_context(lines, 5000)` → **16 chunks × ~86,800 tok ≈ 1.39M tokens** (each chunk ≪ Haiku 200K).
- `rlm_exec(...)` over the loaded `context` **inside the Docker sandbox** returned ~300 bytes:
  ```
  total_lines : 80000
  ERROR lines : 8051
  errors by hour : {'00':321, … '14':381 (peak), … '23':347}
  top error msgs : [('timeout waiting for upstream',1390), ('db deadlock detected',1386), ('nil pointer dereference',1346)]
  ```
  → 1.39M tokens loaded and aggregated; only a tiny bounded summary crossed back into the caller's context.

## (b) K8s manifests — misconfiguration surface
`rlm_load_context(<dir>, "dir")` (3 files) → `rlm_exec` scan:
```
k8s misconfig hits: [('privileged: true',1), ('hostNetwork: true',1), (':latest',2), ('runAsUser: 0',1)]
files scanned: 3
```

## (c) Repo dump — cross-file structure
`rlm_load_context(<repo>, "dir")` (2 files) → `rlm_exec`:
```
functions defined : ['handler', 'make_user']
cross-file imports: [('models', 'make_user')]
```

## Model routing (auth-gated — ready, requires Claude Code OAuth token)
```
rlm_query(ctx, "Summarize the dominant error pattern and its peak hour")   # Sonnet 5 root + Haiku 4.5 sub (Docker)
rlm_query(ctx, "…", model_override="opus")                                  # Opus 4.8 root for the hardest tasks
rlm_sub_query_batch(ctx, "List error signatures in this chunk")             # Haiku 4.5 map-reduce, concurrency 6
```
Routing is proven at runtime by the per-model `usage` table the tools return
(engine `UsageSummary.model_usage_summaries` is keyed by model id), and cost is
computed from token counts × verified rates (Sonnet $3/$15, Opus $5/$25, Haiku $1/$5 per MTok).

## OAuth → `claude` CLI transport (verified live 2026-06-30)
Under OAuth the server drives the official `claude` CLI instead of the HTTP API (Strategy in
`src/transport.py`). Verified end-to-end with a `claude setup-token` token in `.env`, **no API key**:

- **CLI mechanics / OAuth headless:** `claude -p --output-format json --model claude-haiku-4-5
  --safe-mode --tools "" --no-session-persistence` → `{"subtype":"success","result":"OK",
  "usage":{"input_tokens":…,"output_tokens":…},"total_cost_usd":…}` — parser matches.
- **Premium model over OAuth (the old HTTP 429 case):** same call with `--model claude-sonnet-4-6`
  → `success`, `claude-sonnet-4-6` in `modelUsage`, **no 429** — in **both** `--system-prompt`
  (replace) and `--append-system-prompt` modes. Default is `replace` (cleaner RLM prompt; premium
  access confirmed).
- **Full `rlm_query`** (root=Sonnet via CLI, Docker REPL): on a synthetic log with a planted answer
  →  *"The single most frequent ERROR type is OutOfMemory, occurring exactly 60 times."* — correct,
  via **2 real Sonnet turns** (multi-turn transcript flattening works), `usage` keyed
  `claude-sonnet-4-6`, ~17 s.
- **`rlm_sub_query`** (Haiku via CLI) → returns cleanly; sub-model routing proven.
- `rlm_status` → `auth: oauth`, `transport: cli (claude CLI)`, `system-prompt mode: replace`.

Note: under the CLI transport, `usage.input_tokens` excludes cache-read tokens, so reported input
tokens/cost are lower bounds; **per-model routing attribution is exact** (which model ran each role).

## How to reproduce
```
cd rlm-mcp && .venv/bin/python scripts/validate.py /tmp/rlm-val      # pure-logic + store/chunking
# then, with a `claude setup-token` token in .env (no API key), exercise rlm_query /
# rlm_sub_query_batch from a Claude session — OAuth drives the claude CLI transport.
```
