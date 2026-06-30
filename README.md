# rlm-mcp — Recursive Language Models as MCP tools

An MCP server that exposes **RLM (Recursive Language Models)** to Claude Code (CLI **and**
desktop). RLM's win is **oversized single contexts** — multi-hundred-MB/GB logs, large repo
dumps, big k8s manifest sets — by keeping the content as an external variable in a sandboxed
Python REPL and recursively sub-querying it, instead of pushing it into the prompt.

> Fork of [`eesb99/rlm-mcp`](https://github.com/eesb99/rlm-mcp), modernized for the current
> engine [`alexzhang13/rlm`](https://github.com/alexzhang13/rlm) (`rlms`). Core RLM by
> **Zhang, Kraska, Khattab** (MIT CSAIL). See [NOTICE](NOTICE).

**This is a supplement, not a replacement** for Claude Code's native tools. Reach for it when
an input is too big to read directly; for normal feature work and small files, native tools win.

## Auth — reuses your Claude Code login, zero setup
A transport **Strategy** (`src/transport.py`) decides *how* each model call is made, selected by
**`mode`** (`config.yaml` or the `RLM_MODE` env var):

- **`claude-cli`** — the server **drives the official `claude` CLI** (`claude -p`) for every
  completion; it does **not** call the HTTP API. The CLI authenticates from your **existing Claude
  Code login** (keychain), so there's **nothing to set up**: no API key, no `setup-token`, no token
  in `.env` — and no premium-model gating to work around. (A `CLAUDE_CODE_OAUTH_TOKEN` in the env is
  still honored, e.g. for a headless box with no keychain.)
- **`api`** — calls go over the Anthropic SDK using `ANTHROPIC_API_KEY`.
- **`auto`** (default) — prefer the `claude` CLI when it's installed (reuse the Claude Code login,
  zero setup); otherwise fall back to `ANTHROPIC_API_KEY`.

The function interface is identical in every mode; only the transport swaps, and both run behind the
same throttle + auth-aware retry. If no transport is available the server **fails fast** with a clear
message. `rlm_status` shows the configured `mode` and the resolved `transport`.

## What changed vs the upstream wrapper
- Auth: OpenRouter/`OPENROUTER_API_KEY` → **reuse your Claude Code login via the `claude` CLI** (no key, no `setup-token`); `mode: api` + `ANTHROPIC_API_KEY` optional.
- Backend `litellm` → **`anthropic`** (litellm was removed from the engine; the old wrapper errors against `rlms 0.1.3`).
- Models grok/gpt-4o-mini → **Sonnet 5 root** (Opus 4.8 override) + **Haiku 4.5 sub**.
- `environment="local"` (host exec) → **Docker sandbox by default**.
- Inline-string context → **external on-disk store** + bounded tool output (raw content ~4 KB; synthesis answers bound generously).
- Pinned `rlms==0.1.3`, `mcp==1.28.1`.

## Model selection (strategy)
Role→model mapping lives in one place — `src/models.py` (a strategy pattern), not hardcoded across the code.
- **API key:** each role uses its configured model verbatim (root `claude-sonnet-5`, override `claude-opus-4-8`, sub `claude-haiku-4-5`).
- **Claude Code OAuth:** each role maps to the closest **subscription-supported sibling**. Verified by a live probe: current 4.x IDs work as-is; `claude-fable-5` is mapped to `claude-opus-4-8` (the API's own guidance), and deprecated dated IDs map to their current equivalents.
`rlm_status` prints both the configured and the resolved models for the active auth mode.

## Tools (12)
`rlm_load_context` · `rlm_load_file` · `rlm_inspect_context` · `rlm_chunk_context` ·
`rlm_query` (full recursive: Sonnet root + Haiku sub in Docker) · `rlm_sub_query` ·
`rlm_sub_query_batch` (Haiku map-reduce) · `rlm_exec` (Python in the sandbox) ·
`rlm_set_variable` · `rlm_get_variable` · `rlm_set_answer` · `rlm_status`.

## Prerequisites
- macOS/Linux, **Python 3.12 or 3.13** (the engine has no 3.14 wheels), `uv` recommended.
- **Docker** running (for the default sandbox). Apple Silicon supported.
- A **Claude Code login / subscription** — just be logged into the `claude` CLI (run `claude` once). No API key, no `setup-token` required.

## Install
```bash
cd rlm-mcp
./install.sh                       # venv + pinned deps + builds the rlm-sandbox image
# Auth: if you're already logged into the `claude` CLI, you're done — nothing to set up (mode=auto).
# Headless box with no keychain? `claude setup-token` then `./scripts/set-token.sh` to store the token.
# Prefer the API instead? put ANTHROPIC_API_KEY in .env and set mode: api (or RLM_MODE=api).
```

## Register with Claude Code (CLI + desktop share ~/.claude)
```bash
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
claude mcp list        # expect:  rlm: … ✔ Connected
```
Equivalent JSON (`~/.claude.json` → `mcpServers`):
```json
{ "mcpServers": { "rlm": { "command": "bash", "args": ["/ABS/PATH/rlm-mcp/run_server.sh"] } } }
```
A single user-scoped registration surfaces in both the CLI and the desktop app. After adding,
restart/reopen a session; `rlm_status` confirms config and shows the resolved `transport`. To pin
the mode at registration (no file editing), add `-e RLM_MODE=claude-cli` (or `api`) to `claude mcp add`.

## Making Claude reach for RLM automatically
`install.sh` installs a user skill **`rlm-large-context`** (symlinked into `~/.claude/skills/`,
shared by CLI + desktop — `skills/rlm-large-context/SKILL.md` in this repo). Its description triggers
on oversized-input intents ("what's in / find X across / summarize this huge log|dump|dataset"), so
Claude reaches for the `rlm` tools without being told. **Restart the session after install** so the
skill loads; you can also invoke it manually with `/rlm-large-context`.

Prefer a global rule too? Add to `~/.claude/CLAUDE.md`:
```md
## Oversized inputs → RLM
When a file is larger than ~200 KB or ~5,000 lines (logs, dumps, manifest sets), do NOT read it
directly. Use the `rlm` MCP server: `rlm_load_file`/`rlm_load_context` then `rlm_query` (or
`rlm_chunk_context` + `rlm_sub_query_batch` for map-reduce). The content stays in the sandbox;
only findings come back.
```

## Configuration
- **`mode`** (`config.yaml`, or the `RLM_MODE` env var which wins) — `auto` (default) | `claude-cli`
  | `api`. `auto` reuses your Claude Code login via the CLI; **no credentials file needed**.
- `.env` — usually **empty**. Optional `CLAUDE_CODE_OAUTH_TOKEN` (headless box, no keychain) or
  `ANTHROPIC_API_KEY` (for `mode: api`). Credentials stay host-side; they never enter the container.
- `config.yaml` — `mode`, models, `max_depth`/`max_iterations`, `sandbox` (`docker`|`local`),
  `sandbox_image`, `sandbox_timeout_s`, concurrency, `output_cap_bytes` (raw) / `answer_cap_bytes` (synthesis), `cli_*` knobs, chunk defaults, dirs.

Model routing & cost (per MTok): Sonnet 5 $3/$15 (default root; rate cloned from 4.6 pending published pricing), Sonnet 4.6 $3/$15, Opus 4.8 $5/$25, Haiku 4.5 $1/$5.
`rlm_query`/`rlm_sub_query*` return a per-model usage table so you can see exactly what ran on Haiku vs Sonnet.

## Rate-limit handling (throttle + auth-aware retry)
External vendors enforce rate limits; this server **sacrifices speed for stability** via
`retry_and_queue_retries` (`src/ratelimit.py`). Every model call — engine root, engine
sub, and standalone sub-queries — passes through one process-wide gate, regardless of transport
(`claude` CLI on OAuth, Anthropic SDK on API key):
- **Throttle:** at most `throttle_max_concurrency` (3) calls in flight, each dispatched
  `>= throttle_min_interval_s` (1s) after the previous. A single call is instant; large
  batches queue and fan out by 3 instead of bursting.
- **Retry on limits (auth-aware):** OAuth waits `5,10,15`s (tight subscription limits); API key
  waits `1,2,4`s. Both an HTTP `429` (SDK path) and a rate/usage-limit failure from the `claude`
  CLI (`CliRateLimitError`) trigger the retry; the SDK's `Retry-After` is honored as a floor.
  Fails after the waits are exhausted; non-limit errors are never retried.
Tune via `config.yaml`: `throttle_max_concurrency`, `throttle_min_interval_s`,
`oauth_retry_waits`, `apikey_retry_waits`. (SDK-side retries are disabled so this is the
single source of retry policy.)

## Security
- Python from the model runs in the **Docker sandbox** by default; credentials never enter the
  container (sub-LLM calls proxy back to the host). Setting `sandbox: local` runs on the host —
  **only do this on trusted inputs**; you accept the risk of executing model-written Python directly.
- The OAuth transport spawns the `claude` CLI **on the host** with `--safe-mode` (no hooks/
  CLAUDE.md/skills/MCP, so it can't recurse into this server) and `--tools ""` (text-only — RLM
  runs its own sandbox), in a neutral empty cwd, with `ANTHROPIC_API_KEY` scrubbed from its env so
  the subscription path is used. The CLI authenticates from your own Claude Code login; the token
  never enters the container.
- **Do not point `rlm_load_context` at directories containing credentials.** `load_dir` skips
  `.git`, `.env`, common key files, and binaries, but treat it as best-effort.
- Loaded context stays local; nothing is sent anywhere except Anthropic via your Claude auth.
- No `--dangerously-skip-permissions` anywhere (and never `--bare`, which would force API-key auth).
- **Subscription-OAuth note:** the `claude` CLI draws on your Claude subscription's quota/limits.
  A high-volume RLM run can fire many Haiku sub-queries; a rate/usage-limit failure is retried with
  backoff and then surfaced as a clear error (fail-fast) — retry later, or set `ANTHROPIC_API_KEY`
  (Console pay-as-you-go has separate, higher limits) for bulk/concurrent use.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `No transport available` | neither the `claude` CLI nor `ANTHROPIC_API_KEY` is usable. Log into the `claude` CLI (run `claude` once — recommended, zero setup), or set `ANTHROPIC_API_KEY` + `mode: api`. `rlm_status` shows what's detected. |
| `mode=api but ANTHROPIC_API_KEY is not set` / `mode=claude-cli but the claude CLI was not found` | you pinned a `mode` whose dependency is missing — install/log into the CLI, set the key, or switch back to `mode: auto`. |
| Sonnet/Opus `429` on OAuth (historical) | This was the old **HTTP-OAuth** failure: subscription tokens gated premium models to *Claude-Code-shaped* requests. The current OAuth transport drives the `claude` CLI instead, which is inherently Claude-Code-shaped — so **the gate no longer applies** (verified: Sonnet served over the CLI in both `--system-prompt` and `--append-system-prompt` modes). No identity-block hack needed. The grounding directive (`src/engine.py`) stays — it stops the agent identity from making the model *simulate* the REPL. |
| OAuth rate/usage limit | Your **Claude subscription** quota is saturated — often because an interactive Claude Code session is using the same subscription. The CLI failure is retried with backoff, then surfaced. Run RLM when that session is idle/closed, or set `ANTHROPIC_API_KEY` (separate, higher limits) for bulk/concurrent use. |
| `Failed to start container` / docker errors | start Docker Desktop; run `./install.sh` to build `rlm-sandbox`; or set `sandbox: local` |
| `Unknown backend: litellm` | you're on the unpatched upstream — this fork uses `anthropic` |
| no 3.14 wheels | create the venv with Python 3.12/3.13 |
| `rlm_query` slow on OAuth | each engine turn spawns a fresh `claude` CLI (~3–4 s cold start), so multi-turn queries are slower than the HTTP/SDK path — the deliberate speed-for-stability trade. Plus a one-time container start. Use `ANTHROPIC_API_KEY` (SDK path) if latency matters more than reusing your subscription. |
| `claude: command not found` (OAuth) | the `claude` CLI isn't on the server's PATH; install Claude Code, or set `cli_path` in `config.yaml` to its absolute path (e.g. `/opt/homebrew/bin/claude`). `rlm_status` shows whether it's found. |

## Acknowledgments / citation
RLM by Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL). Base wrapper by eesb99.
```bibtex
@article{zhang2025rlm, title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar}, year={2025}}
```
MIT License — see [LICENSE](LICENSE).
