# Phase 2 — Wrapper audit (base + reference impls)

## Base wrapper we forked: `eesb99/rlm-mcp` (last commit 2026-01-26)
One file, `src/server.py` (341 lines), `FastMCP` + 5 task tools.

| Aspect | Base wrapper (as-is) | What it needs / what we did |
|---|---|---|
| Engine call | `RLM(...).completion(prompt)` — **already the current class API** | kept |
| Backend | `backend="litellm"`, `other_backends=["litellm"]` | **BROKEN vs rlms 0.1.3** (`litellm` removed from `get_client`) → `anthropic` |
| Models | OpenRouter `x-ai/grok-code-fast-1` / `openai/gpt-4o-mini` | → Sonnet 4.6 root (Opus 4.8 override) / Haiku 4.5 sub |
| API key | `OPENROUTER_API_KEY` | → `ANTHROPIC_API_KEY` |
| Sandbox | `environment="local"` (host exec; README admits "no additional sandboxing") | → `environment="docker"` (custom `rlm-sandbox` image) |
| MCP SDK | `mcp>=1.0.0` (unpinned) | → pinned `mcp==1.28.1` |
| Engine install | README: `git clone … && pip install -e .` | → `pip install rlms==0.1.3` (now on PyPI) |
| Tool surface | `rlm_execute / rlm_analyze / rlm_code / rlm_decompose / rlm_status` | expanded to 12 (load/inspect/chunk/query/sub_query[_batch]/exec/vars/answer/status) |
| Oversized inputs | context passed as an **inline string arg** → flows through the caller's context | **external on-disk store** + bounded ~4 KB output |

## Reference impls (cloned as patterns, not dependencies — all stale)

| Repo | Last commit | Stack / SDK | Models | Sandbox | Best idea borrowed |
|---|---|---|---|---|---|
| `win10ogod/RLM-MCP` | 2026-01-05 | TypeScript, `@modelcontextprotocol/sdk` | claude-3-5-haiku/sonnet, gpt-4o, **llama3/ollama** | client-LLM (host sampling) | client-side LLM + local/Ollama option (we keep Anthropic per brief, but Ollama noted as a zero-cost sub option) |
| `richardwhiteii/rlm` | 2026-01-22 | Python, `mcp`, Claude Agent SDK + Ollama | **claude-haiku-4-5** (but uses invalid date-suffixed IDs `…-20250514/-20251101`) | — | **CLAUDE.md auto-trigger** + Haiku sub for cheap chunk work |
| `MuhammadIndar/MCP-RLM` | 2026-01-25 | Python, FastMCP | claude-3.5-sonnet, gpt-4o, ollama, qwen | — | **documented Claude Desktop config**; planner/worker split |
| `delonsp/rlm-mcp-server` | 2026-06-06 (most recent, 56 files) | Python, `modelcontextprotocol` | claude-code/mem, gpt-4o | Docker REPL | richest surface: **`rlm_load_file` / `load_s3` / `process_pdf` / `search_code` / async tasks / var pinning** — inspired our load/chunk/exec/var tools |

## Decisions recorded
- **Tool naming & set**: adopt the brief's baseline (`rlm_load_context/load_file/inspect/chunk/query/sub_query/sub_query_batch/exec/set_variable/get_variable/set_answer`) + `rlm_status`; two layers — engine-orchestrated `rlm_query` and caller-orchestrated primitives.
- **Chunking defaults per content type**: `lines` default (2000 lines / 120 KB cap ≈ 30k tok, well under Haiku 200K); `files` for dir loads (one chunk per file marker); `functions`/`headings` for code/markdown; `paragraphs`/`semantic` for prose.
- **Ollama sub option**: not enabled (brief mandates Anthropic Haiku for sub); documented as a future swap (engine supports `vllm`/`openrouter` backends).
- **`sub_query_batch` concurrency**: default 6 (config `subquery_concurrency`); engine `rlm_query_batched` parallelism default 4.
