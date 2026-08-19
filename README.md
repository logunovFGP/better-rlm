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

Two ways in:

- **[As a Claude Code plugin](#install-as-a-claude-code-plugin)** — two commands, same on every OS.
  No cloning, no venv to manage, and the skill comes with it.
- **From a checkout** — the platform installers below. Choose this if you want to *edit* better-rlm,
  or you'd rather not depend on `uv`.

Both end up running the same server. The plugin is the shorter path; the checkout is the one the
rest of this README's paths (`config.yaml`, `.env`, `install.ps1`) assume.

---

### Install as a Claude Code plugin

**Prerequisite:** [`uv`](https://docs.astral.sh/uv/) on `PATH` — it builds and caches the Python
environment on first launch, so nothing else needs installing. Plus `claude`, logged in.

```bash
/plugin marketplace add logunovFGP/better-rlm
/plugin install better-rlm@better-rlm
```

Restart Claude Code. You get the `rlm` MCP server and the `rlm-large-context` skill together —
`/plugin` owns both, so **don't also run `install.sh`/`install.ps1`**; two copies of the same skill
name and two `rlm` registrations only compete with each other.

**One more step for the default sandbox.** The plugin does not build the Docker image, and
`sandbox: docker` is the default — so `rlm_exec` and `rlm_query` fail until the image exists. Build
it once from the plugin directory (`/plugin` shows the path; it is the `${CLAUDE_PLUGIN_ROOT}` the
server is launched with):

```bash
docker build -t rlm-sandbox -f docker/Dockerfile.sandbox docker/
```

Everything else — grep, chunking, loading, inspection — works without it. If you accept
[the documented risk](#security) of running model-written Python on your host, set
`RLM_SANDBOX=local` in the plugin's MCP server env instead of building the image.

To configure it, edit `config.yaml` **inside the plugin directory**; it is read relative to the
server's own root, not your project. Updating the plugin replaces that directory, so keep durable
settings in your own notes.

---

### Install from a checkout

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

Only `rlm_exec` and `rlm_query` need the sandbox; the other thirteen tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- bash "$(pwd)/run_server.sh"
```

To go back once Docker is healthy, drop the env var — re-running `./install.sh` will tell you the
registration no longer matches the sandbox and print exactly this:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
```

Either way, restart Claude Code afterwards: the mode is read once at server start, so there is no
per-call override, and `rlm_status` shows which mode is actually live.

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

Only `rlm_exec` and `rlm_query` need the sandbox; the other thirteen tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- bash "$(pwd)/run_server.sh"
```

To go back once Docker is healthy, drop the env var — re-running `./install.sh` will tell you the
registration no longer matches the sandbox and print exactly this:

```bash
claude mcp remove -s user rlm
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
```

Either way, restart Claude Code afterwards: the mode is read once at server start, so there is no
per-call override, and `rlm_status` shows which mode is actually live.

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

**It asks rather than gives up.** When something needs a decision, the installer prompts instead of
printing a warning and moving on:

| Situation | Choices |
|---|---|
| Docker installed but not running | Retry (after you start it) · run with local sandbox · skip the image build |
| Venv held by a running `rlm` server, deps changed | Stop those processes and rebuild · cancel |
| `rlm` not registered | Register now · not now |
| `rlm` registered to another checkout | Re-point here · leave it |
| Skill link points at another checkout | Re-point here · leave it |

Choosing "local sandbox" also registers with `RLM_SANDBOX=local`, so the server actually honours it.

Flags pre-answer prompts for unattended use — `-Force` (stop venv holders), `-Register`,
`-Sandbox local`, `-SkipDocker`, `-SkipSkill` — and **`-NonInteractive`** takes the safe default for
every question (never registers, never kills, never re-points). Prompting is skipped automatically
under `-WhatIf`, a redirected pipeline, or CI, so nothing can hang waiting on a question.

Other flags: `-WhatIf` (dry run) · `-Verbose` · `-PythonVersion 3.12`.

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

Only `rlm_exec` and `rlm_query` need the sandbox; the other thirteen tools — including the free
`rlm_grep` and `rlm_read_chunk` — never touch it. To run those two without Docker:

```powershell
claude mcp remove -s user rlm
claude mcp add -s user rlm -e RLM_SANDBOX=local -- cmd /c "C:\path\to\better-rlm\run_server.cmd"
```

To go back once Docker is healthy, just re-run `.\install.ps1` and choose **`[R]` Retry** at the
Docker prompt — it notices the registration still carries `RLM_SANDBOX=local` and offers to drop it.
Manually, that is:

```powershell
claude mcp remove -s user rlm
claude mcp add -s user rlm -- cmd /c "C:\path\to\better-rlm\run_server.cmd"
```

Either way, restart Claude Code afterwards: the mode is read once at server start, so there is no
per-call override, and `rlm_status` shows which mode is actually live.

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

## Uninstall

```bash
./uninstall.sh --dry-run     # show what would go, change nothing
./uninstall.sh               # macOS / Linux
```

```powershell
.\uninstall.ps1 -WhatIf      # show what would go, change nothing
.\uninstall.ps1              # Windows
```

Reverses the installer for **this checkout only**, and is safe to re-run. It removes the `rlm`
registration, the skill link, `core.hooksPath`, the virtualenv and build artefacts.

Three things it deliberately leaves alone:

- **A registration or skill link owned by another checkout.** Both installers refuse to hijack
  those, which is what lets several checkouts coexist — so an unguarded `claude mcp remove` here
  would uninstall whichever checkout currently owns the name. Each one is compared against this
  directory first and reported instead when it does not match.
- **`~/.rlm`** — your loaded contexts and logs. Shared by every checkout and the only copy of
  that data, so it needs `--purge-data` / `-PurgeData`.
- **The `rlm-sandbox` image**, shared the same way: `--image` / `-Image`.

A `.env` you edited is also kept, because it may hold `CLAUDE_CODE_OAUTH_TOKEN` or an API key;
an untouched copy (byte-identical to `.env.example`) is removed. On Windows, a running server
holds `.venv_windows` open — pass `-Force` to stop it, or close Claude Code first.

Other gitignored working state is left to `git clean -xdf`, which does it better than a
hand-maintained list. To finish, delete the directory.

> If you installed via `/plugin`, uninstall through `/plugin` instead — these scripts only know
> about a checkout install.

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

Installed as a plugin, the skill ships inside it — `/plugin` loads `skills/rlm-large-context`
directly and there is nothing to link. From a checkout, both installers add it as a *user* skill —
`install.sh` symlinks it into
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

## The 15 tools

**Load & inspect** — `rlm_load_context` · `rlm_load_file` · `rlm_load_source` · `rlm_inspect_context` · `rlm_chunk_context`
**Deterministic retrieval — free, no model call** — `rlm_grep` · `rlm_read_chunk` · `rlm_list_sources`
**Lifecycle** — `rlm_list_contexts` · `rlm_drop_context`
**Model-backed** — `rlm_query` (full recursive: Sonnet root + Haiku sub, model-written Python in the sandbox) · `rlm_sub_query` · `rlm_sub_query_batch` (Haiku map-reduce)
**Sandbox & status** — `rlm_exec` (Python in the sandbox) · `rlm_status`

Only `rlm_exec` and `rlm_query` execute *model-written* code, so only those two depend on the
sandbox — the other thirteen behave identically whether it is Docker or `local`. Both tools state
which mode they are in, and `rlm_status` is authoritative: **`sandbox: local` means that code runs
on your host, unisolated.** `rlm_load_source` also runs a process on the host, but never
model-written and never through a shell: only a command an operator declared, with parameters
substituted as literal argv tokens — see [Live sources](#live-sources) below.

Two of those cost nothing. `rlm_grep` and `rlm_read_chunk` are pure retrieval with no model call,
so narrowing a 2 GB file down to the interesting 40 KB is free — you only pay once you ask a
question about it. Sandbox variables are set and read through `rlm_exec` itself
(`name = value`, `print(repr(name))`); there are no separate variable tools.

---

## Live sources

The biggest inputs usually are not files — they are what a system is emitting right now. An hour
of pod logs across a namespace, a metrics range query, a trace export, a journal, an audit feed:
all far past a context window, and all exactly what this server is for. Before this existed the
only route was "shell out, redirect to a temp file, load the file", which leaves an unbounded
uncleaned intermediate on disk and gives the agent nothing to discover.

**This server ships no sources.** It has no vendor knowledge, no endpoints, no credentials and no
built-in registry, and it stays inert until an operator opts in. What your infrastructure is stays
yours: the registry lives at `sources_file` (default `~/.rlm/sources.yaml`), **outside the repo**,
so a site's clusters and endpoints never land in a checkout or a diff.

```yaml
# ~/.rlm/sources.yaml — every value below is an example, nothing here is built in
workload-logs:
  description: Logs for one workload in the current cluster
  command: kubectl logs -n {namespace} -l app={app} --since={since} --tail=-1
  timeout_s: 120
  max_bytes: 268435456

metrics-range:
  description: Range query against the metrics backend
  command: curl -sS -H "Authorization: Bearer ${METRICS_TOKEN}"
           "${METRICS_URL}/api/v1/query_range?query={query}&start={start}&end={end}"

boot-journal:
  description: This host's journal since last boot
  command: journalctl -b --no-pager -o short-iso
```

Then, from the agent:

```
rlm_list_sources()                                         # free — what exists here?
rlm_load_source("workload-logs",
                {"namespace": "prod", "app": "api", "since": "1h"})   # -> ctx_id
rlm_grep(ctx_id, "OOMKilled")                              # free
```

The output never passes through the conversation — only a `ctx_id` comes back, and from there it
is an ordinary context: `rlm_grep`, `rlm_exec`, `rlm_chunk_context` + `rlm_sub_query_batch`,
`rlm_query`.

**How a template is executed.** `command` is split with `shlex` once, at load time, and always run
with `shell=False`. Parameters are substituted into the already-split argv tokens, so a value can
never introduce a shell metacharacter, a pipe or a second command — `app: "web; rm -rf /"` becomes
one literal argv token. `${VAR}` in the *template* expands from the server's environment, which is
where a token belongs: not in the file and not in the conversation. Parameter values are never
expanded, so a value containing `$HOME` cannot read the environment back out. `{name}` is a
parameter and `${VAR}` is an environment reference; they do not collide. A command needing a pipe
should be a wrapper script you register instead — and note that registering `sh -c "…{param}…"`
deliberately re-opens the shell you were being protected from.

**Partial results are labelled, not hidden.** A source that exits non-zero, overruns `timeout_s`,
or hits `max_bytes` still returns its `ctx_id`, but marked *WITH WARNINGS* — a truncated log
answers "does X appear?" with a confident, wrong **no**. A command that fails *and* produces
nothing is an error, not an empty context. **Exit 0 with no output is flagged too**: a dead
tunnel, a lapsed session and a wrong selector all look identical to "nothing matched", and that
ambiguity is how an empty result gets reported as a finding. Both bounds kill the process, so a `--follow` source
terminates instead of running forever or filling the disk.

The registry is re-read on every call, so adding a source needs no server reconnect. `rlm_status`
reports how many are declared and surfaces a malformed file there rather than on first use.

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
- **Named sources run on the host, and only what an operator wrote.** `rlm_load_source` executes a
  command from `~/.rlm/sources.yaml` with `shell=False`; parameters land as literal argv tokens, so
  the model cannot inject a second command or reach a source that was never declared. Nothing ships
  declared, so a fresh install can run nothing. Keep tokens in the environment (`${VAR}` in the
  template) rather than in the file — only `argv[0]` is logged, never the rendered command.
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
- **`sources_file`** (`config.yaml`) — default `~/.rlm/sources.yaml`, the named-source registry.
  Outside the repo on purpose; absent by default, so no source exists until you declare one. See
  [Live sources](#live-sources).
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
