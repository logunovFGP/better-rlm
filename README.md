# better-rlm

**Give Claude Code a 4 GB log file and ask it a question.**

better-rlm is an MCP server that lets Claude answer questions about inputs far larger than any
context window — multi-hundred-MB logs, whole-repo dumps, giant JSON exports, sprawling k8s
manifest sets — by *never putting them in the prompt*.

The content is loaded into a sandboxed Python REPL as an ordinary variable. Claude then explores
it with code and recursive sub-queries, and only the findings come back. Cost and latency scale
with **what the answer needs**, not with the size of the file.

```
You:     "Which service caused the 03:14 cascade?"  →  2.3 GB of logs
Claude:  grep, chunk, sub-query, correlate  (inside the sandbox)
You get: the answer + a per-model token/cost table
```

> **This is engineering, not research.** Recursive Language Models are the work of
> **Alex L. Zhang, Tim Kraska, and Omar Khattab (MIT CSAIL)** — see [their paper][paper] and
> [reference implementation][rlm]. They proved the idea. This project does the unglamorous part:
> making it install in one command, authenticate with no API key, run sandboxed by default, survive
> rate limits, and clean up after itself — so the technique is usable by anyone with a Claude Code
> subscription rather than only by people willing to wire it up themselves.

---

## Why this fork exists

better-rlm began as a fork of [`eesb99/rlm-mcp`][upstream], which no longer runs against the current
engine. Everything below is the gap between a working demo and something you'd leave installed.

| | Upstream wrapper | **better-rlm** |
|---|---|---|
| **Runs at all** | Errors on `rlms 0.1.3` — the `litellm` backend was removed from the engine | Pinned to `rlms==0.1.3` / `mcp==1.28.1` on the `anthropic` backend |
| **Setup cost** | OpenRouter account + `OPENROUTER_API_KEY` | **Nothing.** Reuses the Claude Code login you already have |
| **Model-written Python** | `environment="local"` — executed **on your host** | **Docker sandbox by default**, credentials never enter the container |
| **Context handling** | Passed as an inline string | External on-disk store; tool output bounded so a big result can't blow up the session |
| **Models** | grok / gpt-4o-mini | Sonnet 5 root · Haiku 4.5 sub · Opus 4.8 override, resolved per auth mode |
| **Rate limits** | Unhandled | Process-wide throttle + auth-aware retry, so batch runs degrade instead of failing |
| **Disk** | Grows | Log sweep capped at 20 files / 50 MB / 7 days; orphaned sandboxes reaped at startup |
| **Shutdown** | Hard kill | SIGTERM/SIGINT tears down the container and logs the exit |
| **Discovery** | You remember to use it | Ships a skill so Claude reaches for it on oversized-input intents |
| **Platforms** | macOS/Linux | macOS · Linux · **Windows** (native `install.ps1`, no WSL required) |

### The zero-setup part is the point

Every other RLM wrapper asks for an API key before it does anything. better-rlm's default
transport **drives the official `claude` CLI**, which authenticates from your existing Claude Code
login. No API key, no `setup-token`, no token in `.env`, no premium-model gating to work around.

If you're logged into `claude`, installation is finished when `install.sh` exits.

---

## Getting Started

Pick **your** platform and follow it top to bottom. Each section is complete and self-contained —
there is nothing to cross-reference from the other two.

- [macOS](#macos) · [Linux](#linux) · [Windows](#windows)

---

### macOS

**1. Prerequisites**

```bash
brew install python@3.12 uv        # uv optional; installer falls back to venv + pip
brew install --cask docker         # then LAUNCH Docker Desktop (Apple Silicon fine)
claude                             # run once and log in — this is your auth
```

Python 3.11–3.13 all work; `install.sh` pins **3.12**. The engine ships no 3.14 wheels.
No API key, no token, nothing in `.env` — the login above is the whole auth story.

**2. Install**

```bash
git clone https://github.com/logunovFGP/better-rlm && cd better-rlm
./install.sh
```

This creates `.venv_sh`, installs pinned deps, builds the `rlm-sandbox` Docker image, creates
`.env` (empty — only needed for `mode: api`), and symlinks the `rlm-large-context` skill into
`~/.claude/skills/`. If Docker isn't running it prints a warning and continues; jump to step 5.

**3. Register the server**

```bash
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
claude mcp list                    # 'rlm' should appear
```

Or run `./install.sh --register` in step 2 to do both at once. Registration is opt-in either way:
it writes `~/.claude.json` — global state, outside this checkout — and `claude mcp add` exits
non-zero on a name that already exists. Re-running with an existing `rlm` reports it and prints the
remove/re-add pair, rather than failing or silently hijacking another checkout's registration.

**4. Verify**

Restart your Claude session (so both the server and the skill load), then ask Claude for
`rlm_status`. It reports the resolved transport, the models actually selected, the sandbox mode,
and whether Docker was found. Then point it at something enormous.

**5. No Docker? (optional)**

Only `rlm_exec` and `rlm_query` need the sandbox; the other eleven tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- bash "$(pwd)/run_server.sh"
```

**This executes model-written Python directly on your Mac**, with your environment in scope. Use it
for trusted inputs only — see [Security](#security).

**macOS notes**

- The venv is `.venv_sh`, deliberately separate from Windows' `.venv_windows`, so a shared
  checkout can't cross-clobber interpreters. `install.sh` rebuilds it from scratch each run.
- OAuth reuses your Claude Code login from the **macOS keychain**. Nothing to configure.
- Re-running `./install.sh` is safe and idempotent.

---

### Linux

**1. Prerequisites**

```bash
sudo apt install -y python3.12 python3.12-venv git      # or dnf/pacman equivalent
curl -LsSf https://astral.sh/uv/install.sh | sh         # optional
sudo apt install -y docker.io && sudo systemctl start docker
sudo usermod -aG docker "$USER"                         # then log out/in so it applies
claude                                                  # run once and log in — this is your auth
```

Python 3.11–3.13 all work; `install.sh` pins **3.12**. The engine ships no 3.14 wheels.
No API key, no token, nothing in `.env` — the login above is the whole auth story.

**2. Install**

```bash
git clone https://github.com/logunovFGP/better-rlm && cd better-rlm
./install.sh
```

This creates `.venv_sh`, installs pinned deps, builds the `rlm-sandbox` Docker image, creates
`.env` (empty — only needed for `mode: api`), and symlinks the `rlm-large-context` skill into
`~/.claude/skills/`. If Docker isn't running it prints a warning and continues; jump to step 5.

**3. Register the server**

```bash
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
claude mcp list                    # 'rlm' should appear
```

Or run `./install.sh --register` in step 2 to do both at once. Registration is opt-in either way:
it writes `~/.claude.json` — global state, outside this checkout — and `claude mcp add` exits
non-zero on a name that already exists. Re-running with an existing `rlm` reports it and prints the
remove/re-add pair, rather than failing or silently hijacking another checkout's registration.

**4. Verify**

Restart your Claude session (so both the server and the skill load), then ask Claude for
`rlm_status`. It reports the resolved transport, the models actually selected, the sandbox mode,
and whether Docker was found. Then point it at something enormous.

**5. No Docker? (optional)**

Only `rlm_exec` and `rlm_query` need the sandbox; the other eleven tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- bash "$(pwd)/run_server.sh"
```

**This executes model-written Python directly on your host**, with your environment in scope. Use it
for trusted inputs only — see [Security](#security).

**Linux notes**

- If `docker info` needs `sudo`, the image build step will warn and skip. Fix the `docker` group
  membership (above) and re-run `./install.sh`.
- The venv is `.venv_sh`, deliberately separate from Windows' `.venv_windows`, so a WSL-shared
  checkout can't cross-clobber interpreters. `install.sh` rebuilds it from scratch each run.
- OAuth reuses your Claude Code login from `~/.claude/.credentials.json`. On a headless box with no
  login, set `CLAUDE_CODE_OAUTH_TOKEN` in `.env` instead.

---

### Windows

Native — **no WSL required**. Works in both Windows PowerShell 5.1 and PowerShell 7 (`pwsh`).

**1. Prerequisites**

```powershell
winget install Python.Python.3.13   # 3.11-3.13 all work; no 3.14 wheels exist
winget install astral-sh.uv         # optional; installer falls back to venv + pip
winget install Docker.DockerDesktop # then LAUNCH Docker Desktop and wait for "Engine running"
claude                              # run once and log in — this is your auth
```

`install.ps1` defaults to **3.13**; pass `-PythonVersion 3.12` to pin an older one. No API key, no
token, nothing in `.env` — the login above is the whole auth story.

**2. Install**

```powershell
git clone https://github.com/logunovFGP/better-rlm
cd better-rlm
.\install.ps1 -Register
```

`-Register` does the install **and** registers the server in one shot. This creates
`.venv_windows`, installs pinned deps, builds the `rlm-sandbox` Docker image, creates `.env`
(empty — only needed for `mode: api`), and links the `rlm-large-context` skill into
`%USERPROFILE%\.claude\skills`. If Docker isn't running it warns and continues; jump to step 5.

Re-running is cheap and safe: the venv is only rebuilt when `pyproject.toml`, `uv.lock`, or
`-PythonVersion` actually change (fingerprinted in `.venv_windows\.rlm-deps-sha256`). Otherwise it
reports `Dependencies unchanged` and leaves the venv alone — so a running `rlm` server, which holds
the interpreter open, is not in the way.

When dependencies *have* changed and a server is running, the rebuild cannot delete the venv. The
script names the processes holding it and stops there; add **`-Force`** to have it stop them for
you:

```powershell
.\install.ps1 -Force
```

Other flags: `-Sandbox local` (skip the Docker image build entirely) · `-SkipDocker` ·
`-SkipSkill` · `-WhatIf` (dry run) · `-Verbose`.

**3. Register the server** — skip if you used `-Register`

Run `.\install.ps1` without `-Register` and it prints the exact command for your path. It looks
like this:

```powershell
claude mcp add -s user rlm -- cmd /c "C:\path\to\better-rlm\run_server.cmd"
claude mcp list                     # 'rlm' should appear
```

Note the launcher is **`run_server.cmd`**, not `run_server.sh`, and it must be invoked through
`cmd /c`.

**4. Verify**

Restart your Claude session (so both the server and the skill load), then ask Claude for
`rlm_status`. It reports the resolved transport, the models actually selected, the sandbox mode,
and whether Docker was found. Then point it at something enormous.

**5. No Docker? (optional)**

Only `rlm_exec` and `rlm_query` need the sandbox; the other eleven tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```powershell
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- cmd /c "C:\path\to\better-rlm\run_server.cmd"
```

**This executes model-written Python directly on your PC**, with your environment in scope. Use it
for trusted inputs only — see [Security](#security).

**Windows notes**

- The venv is **`.venv_windows`**, deliberately separate from the POSIX `.venv_sh`, so a
  WSL-shared checkout can't cross-clobber interpreters.
- `run_server.cmd` sets `PYTHONUTF8=1` on purpose: the Linux sandbox guest reads host-written
  files as UTF-8 while Windows defaults to cp1252, so without it a non-ASCII context fails to
  write.
- The skill is linked as a **directory junction**. If a real (non-link) folder already exists at
  `%USERPROFILE%\.claude\skills\rlm-large-context`, the installer leaves it alone — delete it and
  re-run to get the link.
- OAuth reuses your Claude Code login from the **Windows credential store** — no keychain, no
  `setup-token`, nothing to configure.
- Paths with spaces are fine, but keep the quotes in the `claude mcp add` command above.
- **Run `claude mcp add` from PowerShell, not Git Bash.** Git Bash rewrites the `/c` in `cmd /c`
  into a Windows path (`C:/`), so the server registers with a mangled command and never starts.
- `-Register` is safe to re-run: an existing `rlm` is reported with the remove/re-add pair instead
  of aborting the install.

---

## When to use it

**This is a supplement, not a replacement** for Claude Code's native tools. That honesty is the
feature: for normal work and small files, `Read` and `Grep` win on every axis. Reach for
better-rlm when the input is genuinely too big to read — roughly **>200 KB or >5,000 lines** — or
when a naive read would truncate and quietly cost you the answer.

Good fits: incident logs, whole-repo dumps for architecture questions, large CSV/JSON exports,
manifest sets, anything where "search it, don't read it" is the right instinct.

---

## Install a skill so Claude reaches for it on its own

Both installers add a user skill, **`rlm-large-context`** — `install.sh` symlinks it into
`~/.claude/skills/`, `install.ps1` junctions it into `%USERPROFILE%\.claude\skills` — shared by the
CLI and desktop app. It is linked rather than copied, so editing `skills/<name>/SKILL.md` takes
effect with no reinstall. Its description triggers on oversized-input intents ("what's in / find X
across / summarize this huge log|dump|dataset"), so Claude routes to the `rlm` tools without being
told. Restart the session after install; `/rlm-large-context` invokes it manually.

Prefer an explicit rule as well? Add to `~/.claude/CLAUDE.md`:

```md
## Oversized inputs → RLM
When a file is larger than ~200 KB or ~5,000 lines (logs, dumps, manifest sets), do NOT read it
directly. Use the `rlm` MCP server: `rlm_load_file`/`rlm_load_context` then `rlm_query` (or
`rlm_chunk_context` + `rlm_sub_query_batch` for map-reduce). The content stays in the sandbox;
only findings come back.
```

---

## The 13 tools

**Load & inspect** — `rlm_load_context` · `rlm_load_file` · `rlm_inspect_context` · `rlm_chunk_context`
**Deterministic retrieval — free, no model call** — `rlm_grep` · `rlm_read_chunk`
**Lifecycle** — `rlm_list_contexts` · `rlm_drop_context`
**Model-backed** — `rlm_query` (full recursive: Sonnet root + Haiku sub in Docker) · `rlm_sub_query` · `rlm_sub_query_batch` (Haiku map-reduce)
**Sandbox & status** — `rlm_exec` (Python in the sandbox) · `rlm_status`

Two of those cost nothing. `rlm_grep` and `rlm_read_chunk` are pure retrieval with no model call,
so narrowing a 2 GB file down to the interesting 40 KB is free — you only pay once you ask a
question about it. Sandbox variables are set and read through `rlm_exec` itself
(`name = value`, `print(repr(name))`); there are no separate variable tools.

---

## Auth — three modes, one interface

A transport **Strategy** (`src/transport.py`) decides *how* each model call is made, selected by
**`mode`** (`config.yaml` or the `RLM_MODE` env var):

- **`claude-cli`** — drives the official `claude` CLI (`claude -p`) for every completion; it does
  **not** call the HTTP API. Authenticates from your existing Claude Code login (keychain), so
  there is nothing to set up. A `CLAUDE_CODE_OAUTH_TOKEN` in the env is still honored, e.g. for a
  headless box with no keychain.
- **`api`** — calls go over the Anthropic SDK using `ANTHROPIC_API_KEY`.
- **`auto`** (default) — prefer the `claude` CLI when installed; otherwise fall back to
  `ANTHROPIC_API_KEY`.

The function interface is identical in every mode; only the transport swaps, and all of them run
behind the same throttle and auth-aware retry. If no transport is available the server **fails
fast** with a clear message rather than half-working.

### Model selection

Role→model mapping lives in one place — `src/models.py` — not hardcoded across the codebase.

- **API key:** each role uses its configured model verbatim (root `claude-sonnet-5`, override
  `claude-opus-4-8`, sub `claude-haiku-4-5`).
- **Claude Code OAuth:** each role maps to the closest **subscription-supported sibling**. Verified
  by live probe: current 4.x IDs work as-is; `claude-fable-5` maps to `claude-opus-4-8` (the API's
  own guidance) and deprecated dated IDs map to their current equivalents.

`rlm_status` prints both configured and resolved models for the active auth mode.

### Providers — one vendor locally, any vendor remotely

`provider` (`config.yaml` or `RLM_PROVIDER`) selects the vendor: **`anthropic`** (default) ·
`gemini` · `openai` · `azure_openai` · `portkey`.

- **`anthropic` is the only provider needing no API key** — it authenticates through the local
  `claude` CLI login. That's why it's the default and why local runs stay zero-setup.
- **Every other provider needs its key** (`GEMINI_API_KEY`, `OPENAI_API_KEY`, …) plus that vendor's
  model IDs in `root_model`/`sub_model`. A missing key is a loud error at resolve time, not a
  failed call halfway through.
- **This exists for remote deploys.** A container has no keychain for the `claude` CLI to read, so
  `anthropic` + OAuth cannot work there; a keyed provider can.

No per-vendor code is maintained here: the pinned `rlms` engine already ships clients for all of
the above, so a non-Anthropic provider reuses *its* client through one adapter
(`transport.EngineClientTransport`) and gains only our shared throttle and 429 retry. Anthropic
keeps two dedicated transports (`CliTransport`, `ApiTransport`) precisely because it's the odd one
out with a keyless path. Mixing vendors per role (Claude root + a cheaper sub elsewhere) is
possible — the engine takes `other_backends` separately — but isn't wired to config yet.

---

## Built to be left running

**Rate-limit handling.** This server **sacrifices speed for stability**. Every model call — engine
root, engine sub, standalone sub-queries — passes through one process-wide gate
(`src/ratelimit.py`), regardless of transport:

- **Throttle:** at most `throttle_max_concurrency` (3) calls in flight, each dispatched
  `>= throttle_min_interval_s` (1s) after the previous. A single call is instant; large batches
  queue and fan out by 3 instead of bursting into a limit.
- **Retry, auth-aware:** OAuth waits `5,10,15`s (tight subscription limits); API key waits `1,2,4`s.
  Both an HTTP `429` (SDK path) and a rate/usage-limit failure from the `claude` CLI
  (`CliRateLimitError`) trigger it, and the SDK's `Retry-After` is honored as a floor. Fails after
  the waits are exhausted; non-limit errors are never retried. SDK-side retries are disabled so
  this is the single source of retry policy.

**Bounded disk — never accumulates.** Structured logs go to a per-PID file
`~/.rlm/logs/rlm-mcp-<date>-<pid>.log` (logfmt: `ts=… pid=… lvl=… evt=… k=v`). stdout stays the
JSON-RPC channel; **stderr is WARNING-only** so healthy runs don't spam Claude Code's error-tagged
MCP log. A race-safe startup sweep caps `~/.rlm/logs` to **≤20 files AND ≤50 MB AND ≤7 days**
across all processes, so many short-lived session servers can't fill your disk. Per-file rotation
is 2 MB × 3 backups.

Events: `startup`, `tool_call` (rid, args summary, duration, outcome), `rlm_query` (root/sub model,
turns, `max_iter_hit`, tokens, cost, answer bytes, truncated), `cli_spawn` (model, duration, exit),
`retry`, `shutdown`.

**Graceful shutdown.** SIGTERM/SIGINT — and a clean stdin EOF — tear down the sandbox container and
log a `shutdown` record before exiting.

**Cost visibility, off by default.** `rlm_query`/`rlm_sub_query*` can return a per-model usage table
so you see exactly what ran on Haiku versus Sonnet. It's opt-in (`report_cost: false`) because a
figure you can't fully trust is worse than no figure.

---

## Security

- Model-written Python runs in the **Docker sandbox** by default; credentials never enter the
  container (sub-LLM calls proxy back to the host). Setting `sandbox: local` runs it on your host —
  **only for trusted inputs**; you're accepting execution of model-written code.
- The OAuth transport spawns the `claude` CLI **on the host** with `--safe-mode` (no hooks, no
  CLAUDE.md, no skills, no MCP — it can't recurse into this server) and `--tools ""` (text-only;
  RLM runs its own sandbox), in a neutral empty cwd, with `ANTHROPIC_API_KEY` scrubbed from its env
  so the subscription path is used. The token never enters the container.
- **Don't point `rlm_load_context` at directories containing credentials.** `load_dir` skips
  `.git`, `.env`, common key files, and binaries, but treat that as best-effort, not a guarantee.
- Loaded context stays local; nothing is sent anywhere except your configured provider.
- No `--dangerously-skip-permissions` anywhere, and never `--bare` (which would force API-key auth).
- **Subscription-OAuth note:** the `claude` CLI draws on your Claude subscription's quota. A
  high-volume run can fire many Haiku sub-queries; a limit failure is retried with backoff and then
  surfaced as a clear error. Retry later, or set `ANTHROPIC_API_KEY` (Console pay-as-you-go has
  separate, higher limits) for bulk use.

---

## Configuration

- **`mode`** (`config.yaml`, or `RLM_MODE` which wins) — `auto` (default) | `claude-cli` | `api`.
- **`sandbox`** (`config.yaml`, or `RLM_SANDBOX` which wins) — `docker` (default) | `local`.
  Invalid values are rejected rather than degraded to host exec.
- `.env` — usually **empty**. Optional `CLAUDE_CODE_OAUTH_TOKEN` (headless, no keychain) or
  `ANTHROPIC_API_KEY` (for `mode: api`). Credentials stay host-side.
- `config.yaml` — `mode`, `provider`, models, `max_depth`/`max_iterations`, `sandbox`
  (`docker`|`local`), `sandbox_image`, `sandbox_timeout_s`, concurrency, `output_cap_bytes` (raw) /
  `answer_cap_bytes` (synthesis), `report_cost`, `cli_*` knobs, chunk defaults, dirs, and the
  logging/throttle keys named above.

Registering with Claude Code, in JSON (`~/.claude.json` → `mcpServers`):

```json
{ "mcpServers": { "rlm": { "command": "bash", "args": ["/ABS/PATH/better-rlm/run_server.sh"] } } }
```

A single user-scoped registration surfaces in both the CLI and the desktop app. To pin the mode at
registration without editing files, add `-e RLM_MODE=claude-cli` (or `api`) to `claude mcp add`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No transport available` | Neither the `claude` CLI nor `ANTHROPIC_API_KEY` is usable. Log into the CLI (run `claude` once — zero setup), or set `ANTHROPIC_API_KEY` + `mode: api`. `rlm_status` shows what's detected. |
| `mode=api but ANTHROPIC_API_KEY is not set` / `mode=claude-cli but the claude CLI was not found` | You pinned a `mode` whose dependency is missing — install/log into the CLI, set the key, or return to `mode: auto`. |
| Sonnet/Opus `429` on OAuth (historical) | The old **HTTP-OAuth** failure: subscription tokens gated premium models to Claude-Code-shaped requests. The current transport drives the `claude` CLI, which is inherently Claude-Code-shaped, so the gate no longer applies (verified in both `--system-prompt` and `--append-system-prompt` modes). |
| OAuth rate/usage limit | Your Claude subscription quota is saturated — often because an interactive session is using the same subscription. Run when that session is idle, or use `ANTHROPIC_API_KEY`. |
| `Failed to start container` / docker errors | Start Docker Desktop; run `./install.sh` to build `rlm-sandbox`; or set `sandbox: local`. |
| `'.venv_windows' must be rebuilt - but it is in use by: PID ...` | Dependencies changed and a running `rlm` server holds the interpreter. Stop Claude Code (or disconnect `rlm` via `/mcp`) and re-run, or run `.\install.ps1 -Force` to stop those processes automatically. Unchanged dependencies never hit this — the venv is reused. |
| `Unknown backend: litellm` | You're on the unpatched upstream — this fork uses `anthropic`. |
| No 3.14 wheels | Create the venv with Python 3.12 or 3.13. |
| `rlm_query` slow on OAuth | Each engine turn spawns a fresh `claude` CLI (~3–4 s cold start), so multi-turn queries are slower than the SDK path — the deliberate speed-for-stability trade, plus a one-time container start. Use `ANTHROPIC_API_KEY` if latency matters more than reusing your subscription. |
| `claude: command not found` (OAuth) | The CLI isn't on the server's PATH; install Claude Code, or set `cli_path` in `config.yaml` to its absolute path. `rlm_status` shows whether it's found. |

---

## Credits

Recursive Language Models are the work of **Alex L. Zhang, Tim Kraska, and Omar Khattab**
(MIT CSAIL) — [paper][paper], [engine][rlm]. The base MCP wrapper structure came from
[`eesb99/rlm-mcp`][upstream]. This project is the engineering layer on top of both; see
[NOTICE](NOTICE) for the full attribution chain.

```bibtex
@article{zhang2025rlm, title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar}, year={2025}}
```

MIT License — see [LICENSE](LICENSE). Contributions welcome: issues and PRs are open.

[paper]: https://arxiv.org/abs/2512.24601
[rlm]: https://github.com/alexzhang13/rlm
[upstream]: https://github.com/eesb99/rlm-mcp
