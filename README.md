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
| **Runs at all** | Errors on `rlms 0.1.3` — the `litellm` backend was removed from the engine | Engine vendored at `./rlm` (from `v0.1.3`) on the `anthropic` backend, `mcp==1.28.1` |
| **Setup cost** | OpenRouter account + `OPENROUTER_API_KEY` | **Nothing.** Reuses the Claude Code login you already have |
| **Model-written Python** | `environment="local"` — executed **on your host** | **Docker sandbox by default**, credentials never enter the container |
| **Context handling** | Passed as an inline string | External on-disk store; tool output bounded so a big result can't blow up the session |
| **Models** | grok / gpt-4o-mini | Sonnet 5 root · Haiku 4.5 sub · Opus 4.8 override, resolved per auth mode |
| **Rate limits** | Unhandled | Process-wide throttle + auth-aware retry, so batch runs degrade instead of failing |
| **Disk** | Grows | Log sweep capped at 20 files / 50 MB / 7 days; orphaned sandboxes reaped at startup |
| **Shutdown** | Hard kill | SIGTERM/SIGINT tears down the container and logs the exit |
| **Discovery** | You remember to use it | Skill for explicit use, plus an opt-in `--hook` that routes oversized reads automatically |
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

**Configuring it.** The [operator TUI](#operator-tui--configure-mode-and-models-from-the-cli) ships
with the plugin and works there — run it from the plugin directory (`/plugin` shows the path):

```bash
./run_tui.sh                      # interactive
./run_tui.sh --one-shot /status   # print the resolved config and exit
```

The plugin never runs `install.sh`, so there is no `.venv_sh`; the script falls back to `uv run`,
the same way the plugin launches the server. Editing `config.yaml` in that directory by hand works
too — it is read relative to the server's own root, not your project.

**Updating the plugin replaces that directory**, so config changes made either way are lost on
update. Keep durable settings in your own notes, or use a checkout if you change them often.

What you cannot change anywhere: the **provider**. Anthropic is the only one supported — see
[Providers](#providers--one-protocol-several-endpoints) for why that is a correctness guard rather than
a preference. The TUI configures the transport mode and the three models.

---

### Install from PyPI

```bash
pip install better-rlm          # or: uv tool install better-rlm
claude mcp add -s user rlm -- better-rlm server
```

That is the whole setup for the model-backed tools. `better-rlm server` runs the
MCP server in-process, so there is no script path to point at and no checkout to
keep around.

```bash
better-rlm where     # which config/env files are in use, and which mode you're in
better-rlm auth      # prints the `claude setup-token` flow
better-rlm           # the config TUI
```

#### Installing from a local folder, for testing the packaged shape

The same package installs straight from a clone, which is how you exercise the
**wheel** shape without publishing anything:

```bash
uv pip install .        # or: pip install .   — wheel shape, ~/.rlm paths
uv pip install -e .     # editable: the console script points back at the clone
```

The two are not interchangeable for testing. `better_rlm.config.IS_CHECKOUT` asks
whether a `pyproject.toml` sits next to the package, so a plain (non-editable)
install into site-packages is the only way to reach the code path a PyPI user gets:

```
$ better-rlm where
mode:    installed (pip)
config:  ~/.rlm/config.yaml   (absent -- baked-in defaults apply)
```

Run it from **outside** the clone. Inside it, the local `better_rlm/` wins on
`sys.path` and you are testing the checkout again without noticing.

**What a pip install does not give you**, and why you might still want the checkout:

| | pip install | checkout |
|---|---|---|
| `config.yaml`, `.env` | `~/.rlm/` | repo root, diffable in git |
| Docker sandbox image | not built — use `sandbox: local` in config | `install.sh` builds `rlm-sandbox` |
| `rlm-large-context` skill | not linked | symlinked into `~/.claude/skills` |
| oversized-read hook | unavailable | `./install.sh --hook` |
| `better-rlm install` | refuses, tells you to clone | re-runs the installer |

`sandbox: local` runs generated code **on your host** rather than in a container — see
[Security](#security) before choosing it. If you want the Docker sandbox, the skill, or the
read hook, install from a checkout instead.

The package ships the vendored engine as top-level `rlm`, the same import name upstream's
`rlms` distribution uses. Installing both into one environment collides; use separate
virtualenvs.

Maintainers: see [Releasing](#releasing) for how a version is cut.

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

Python 3.12–3.14 all work; `install.sh` pins **3.12**.
No API key, no token, nothing in `.env` — the login above is the whole auth story.

**2. Install**

```bash
git clone https://github.com/logunovFGP/better-rlm && cd better-rlm
./install.sh
```

This creates `.venv_sh`, installs pinned deps, builds the `rlm-sandbox` Docker image, creates
`.env` (empty — only needed for `mode: api`), symlinks the `rlm-large-context` skill into
`~/.claude/skills/`, and links the `better-rlm` command into `~/.local/bin`. If Docker isn't
running it prints a warning and continues; jump to step 5.

> **macOS puts `~/.local/bin` on nobody's PATH.** Stock `/etc/paths` is `/usr/local/bin`,
> `/usr/bin`, `/bin`, `/usr/sbin`, `/sbin` — no `~/.local/bin`, unlike most Linux distros.
> `install.sh` warns when it links into a directory your PATH doesn't contain. To fix it for
> zsh (the macOS default shell since Catalina):
>
> ```bash
> echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh
> better-rlm where        # confirms which checkout and which config file it resolves to
> ```
>
> Prefer not to touch PATH? `./run_tui.sh` and `./run_server.sh` in the checkout do the same
> jobs and always have.

**Don't need the Docker sandbox, the skill or the read hook?** `pip install better-rlm` gives
you the server and the command with no clone — see [Install from PyPI](#install-from-pypi).

**3. Register the server**

```bash
claude mcp add -s user rlm -- bash "$(pwd)/run_server.sh"
claude mcp list                    # 'rlm' should appear
```

If you installed from PyPI instead of cloning, the launcher is the command itself — there is no
script path to point at:

```bash
claude mcp add -s user rlm -- better-rlm server
```

Or run `./install.sh --register` in step 2 to do both at once. Registration is opt-in either way:
it writes `~/.claude.json` — global state, outside this checkout — and `claude mcp add` exits
non-zero on a name that already exists. Re-running with an existing `rlm` reports it and prints the
remove/re-add pair, rather than failing or silently hijacking another checkout's registration.

**4. Verify**

Restart your Claude session (so both the server and the skill load), then ask Claude for
`rlm_status`. It reports the resolved transport, the models actually selected, the sandbox mode,
and whether Docker was found. Then point it at something enormous.

**5. Configure (optional)**

Defaults work. To change the transport mode, provider or models without hand-editing
`config.yaml`:

```bash
better-rlm                         # interactive picker
better-rlm --one-shot /status      # print the resolved config and exit
better-rlm --one-shot /mode-help   # compare the proxy and host transports
better-rlm auth                    # long-lived token, for a server you leave running
```

`better-rlm` edits `config.yaml` only. The MCP server reads it at startup, so a change takes
effect on the next `claude mcp restart rlm` — never mid-session.

**6. No Docker? (optional)**

Only `rlm_exec` and `rlm_query` need the sandbox; the other fifteen tools — including the free
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
- `better-rlm` is one global name: the last checkout whose installer ran owns it. `install.sh`
  says so when it re-points the link, `uninstall.sh` refuses to remove a link another checkout
  owns, and `better-rlm where` tells you which one is live.
- Apple Silicon needs no special handling — the sandbox image builds `linux/arm64` natively and
  the wheels are pure Python.

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

Python 3.12–3.14 all work; `install.sh` pins **3.12**.
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

Only `rlm_exec` and `rlm_query` need the sandbox; the other fifteen tools — including the free
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
winget install Python.Python.3.13   # 3.12-3.14 all work
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

Only `rlm_exec` and `rlm_query` need the sandbox; the other fifteen tools — including the free
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
effect with no reinstall. Restart the session after install; `/rlm-large-context` invokes it.

**The skill cannot fire on file size, and its description is not the reason.** Skill selection
matches a description against *conversation text*. This skill's real trigger is a property of the
*data* — "the file is larger than ~200 KB". You type "analyze this log"; nothing in that sentence
says 2 GB, so the model cannot evaluate the condition until it has already called `Read`, by which
point the read was truncated and the context is spent. No rewrite closes that: it would have to
enumerate every phrasing of "read a file that happens to be large". Expect to name the skill.

For automatic routing, install the hook:

```bash
./install.sh --hook
```

It adds a `PreToolUse` hook that checks the size *after* a `Read` call is formed and *before* it
runs — the one moment the predicate is knowable — and redirects anything over 200 KB to
`rlm_load_file`. Deterministic where description-matching is probabilistic.

It fails open, and never blocks without an alternative. A normal `Read` still happens when the
`rlm` server is not registered, for images/PDFs/archives the context store cannot serve, and for a
bounded read — one passing an explicit `limit`, which is the correct *end* of the RLM workflow
after `rlm_grep` returns line numbers.

**It covers the `Read` tool only.** Files read through Bash (`cat`, `head`, `sed -n`) are not seen,
and some setups instruct the model to prefer those. Matching `Bash` too would mean parsing a path
out of an arbitrary command line, with real false-positive risk; that is a follow-up, not a
promise. Opt-in because it writes `~/.claude/settings.json` and changes `Read` behaviour in every
project on the machine. `./uninstall.sh` removes it. POSIX only for now — `install.ps1` has no
`-Hook` equivalent yet.

Prefer an explicit rule as well? Add to `~/.claude/CLAUDE.md`:

```md
## Oversized inputs → RLM
When a file is larger than ~200 KB or ~5,000 lines (logs, dumps, manifest sets), do NOT read it
directly. Use the `rlm` MCP server: `rlm_load_file`/`rlm_load_context` then `rlm_query` (or
`rlm_chunk_context` + `rlm_sub_query_batch` for map-reduce). The content stays in the sandbox;
only findings come back.
```

---

## The 17 tools

**Load & inspect** — `rlm_load_context` · `rlm_load_file` · `rlm_load_source` · `rlm_inspect_context` · `rlm_chunk_context`
**Deterministic retrieval — free, no model call** — `rlm_grep` · `rlm_read_chunk` · `rlm_list_sources`
**Lifecycle** — `rlm_list_contexts` · `rlm_drop_context`
**Model-backed** — `rlm_query` (full recursive: Sonnet root + Haiku sub, model-written Python in the sandbox) · `rlm_sub_query` · `rlm_sub_query_batch` (Haiku map-reduce)
**Sandbox & status** — `rlm_exec` (Python in the sandbox) · `rlm_status`
**Cost & forecast** — `rlm_estimate` (forecast a `rlm_sub_query_batch` before running it) · `rlm_budget` (what this server has spent inside the session window)

Only `rlm_exec` and `rlm_query` execute *model-written* code, so only those two depend on the
sandbox — the other fifteen behave identically whether it is Docker or `local`. Both tools state
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

container-logs:
  description: One container's logs
  command: docker logs --timestamps --since {since} {container}
  merge_stderr: true       # see below — without this a postgres container loads as empty
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

**`merge_stderr` when the program logs to stderr.** By default stderr is kept as a
diagnostic tail, not loaded — for a well-behaved command it is the error channel, and
merging it would bury a failure message inside the data. But plenty of programs write their
*logs* there by convention: postgres does, so `docker logs` on a postgres container puts
every line on stderr and the context loads **empty**. The usual answer, `2>&1`, is shell
syntax that does not exist here by design, so it is a per-source flag instead. With it on
there is no separate tail — a failure message arrives interleaved with the data.

**Credentials: a file the server never opens.** A source needing a token declares
`credential_file` instead of expecting an exported variable, with an optional
`credential_max_age_h`:

```yaml
metrics-range:
  command: curl -sS --fail-with-body -K ${HOME}/.rlm/secrets/metrics.curlrc
           "${METRICS_URL}/api/v1/query_range?query={query}"
  credential_file: ~/.rlm/secrets/metrics.curlrc
  credential_max_age_h: 4
```

Only `exists()` and `st_mtime` are consulted — the file is **never read**, so a token cannot
reach the conversation, a log, or a tool result through this server. `curl -K` reads it
itself at request time, so rotating the file rotates the credential with no restart. A
missing or stale file refuses the call and returns an instruction aimed at the *user*, who
is the only party that should be minting tokens. Because the age limit is enforced locally
it holds whatever expiry the upstream service is willing to issue: a backend that only hands
out 30-day tokens still gets a 4-hour one here. `rlm_list_sources` reports each declared
credential as `ready` or `MISSING or STALE`, so the gap surfaces before a call is attempted
rather than as a puzzling 403 afterwards.

An unset `${VAR}` in a template is refused for the same reason. `os.path.expandvars` leaves
an unknown name as the *literal* text `${METRICS_TOKEN}`, which would otherwise be sent to
the far end as if it were a token — a local misconfiguration surfacing as a remote auth
error. Only the braced form is checked, so `awk '{print $1}'` and `grep 'x$'` stay usable.

**Partial results are labelled, not hidden.** A source that exits non-zero, overruns `timeout_s`,
or hits `max_bytes` still returns its `ctx_id`, but marked *WITH WARNINGS* — a truncated log
answers "does X appear?" with a confident, wrong **no**. A command that fails *and* produces
nothing is an error, not an empty context.

**Zero bytes always carries static advice**, whatever the exit code, because it is the result
most likely to be misread as a finding. The note names the causes in likelihood order —
output on stderr and therefore captured nowhere (with `merge_stderr` and its current state
called out), output redirected or paged away inside the command, then a dead tunnel, lapsed
session or wrong selector/namespace/window — and says to co-verify with a query that must
return data before reporting any negative conclusion. Both bounds kill the process, so a `--follow` source
terminates instead of running forever or filling the disk.

The registry is re-read on every call, so adding a source needs no server reconnect. `rlm_status`
reports how many are declared and surfaces a malformed file there rather than on first use.

---

## Auth — three modes, one interface

A transport **Strategy** (`better_rlm/transport.py`) decides *how* each model call is made, selected by
**`mode`** (`config.yaml` or the `RLM_MODE` env var):

- **`claude-cli`** — drives the official `claude` CLI (`claude -p`) for every completion; it does
  **not** call the HTTP API. Authenticates from the **`claude` CLI's own login** (keychain), so
  usually there is nothing to set up. A `CLAUDE_CODE_OAUTH_TOKEN` in the env is still honored, e.g.
  for a headless box with no keychain.
- **`api`** — calls go over the Anthropic SDK using `ANTHROPIC_API_KEY`.
- **`auto`** (default) — prefer the `claude` CLI when installed; otherwise fall back to
  `ANTHROPIC_API_KEY`.

The function interface is identical in every mode; only the transport swaps, and all of them run
behind the same throttle and auth-aware retry. If no transport is available the server **fails
fast** with a clear message rather than half-working.

#### Being signed in to Claude Code is NOT the same as the CLI being logged in

This is the one auth trap worth knowing, because every other line of `rlm_status` can be
correct while it bites. Claude Code (and the desktop app) hold their **own** credential for the
host session. A nested `claude -p` cannot borrow it — measured with the delegation env both
stripped and left fully intact, identical `OAuth session expired` either way. The CLI needs a
login of its own, and when its token expires a headless refresh cannot complete interactive
OAuth, so it simply stays dead.

`rlm_status` now answers this directly, free and with no model call — it shells out to
`claude auth status --json` (~215 ms):

```
- cli login: NOT LOGGED IN — every model-backed tool will fail.
    1. `claude auth login`   — interactive; refreshes the CLI login
    2. `claude setup-token`  — long-lived token; put it in CLAUDE_CODE_OAUTH_TOKEN
    3. ANTHROPIC_API_KEY + `mode: api`
```

Option 2 is the durable one for a server you leave running: a long-lived token does not
depend on an interactive refresh that a background process cannot perform. The same
remediation is attached to every auth failure, so a failed `rlm_sub_query_batch` tells you how
to fix it instead of only that it broke. With the login dead, `probe=True` skips its call
rather than spending one to confirm what the free check already established.

#### Where the token has to live

**`export CLAUDE_CODE_OAUTH_TOKEN=…` in your shell does not reach the server.** The MCP
server is launched by Claude Code with its own environment, not from your login shell — so a
token exported in a terminal is invisible to it. It goes in **`.env` at the repo root**, which
`better_rlm/config.py` loads at startup and `.gitignore` already covers:

```bash
./install.sh --auth        # or: .\install.ps1 -Auth
```

That runs `claude setup-token`, then asks for the token at a **hidden prompt** and writes it
to `.env` itself (0600, replacing any empty slot). The value is never echoed, never passed as
an argument, and never enters shell history — the installer reports only its byte count. It
deliberately does *not* capture `setup-token`'s stdout: the browser flow prints there too, so
capturing it would hide the UI you have to interact with.

If the token is already exported in the shell you run the installer from, it is copied into
`.env` for you with no prompt and no second `setup-token` — that export alone would never
have reached the server.

On POSIX `.env` is written `0600`. On Windows it inherits the repo directory's ACL and the
installer does **not** tighten it: `Get-Acl`/`Set-Acl` is Windows-only API that cannot be
exercised on this project's CI, and shipping an unverified credential-permissions path is
worse than naming the exposure. To lock it down yourself:

```powershell
icacls .env /inheritance:r /grant:r "$env:USERNAME:(R,W)"
```

To set it without the value touching your shell history — from the shell where you just
exported it:

```bash
python3 -c 'import os,pathlib; p=pathlib.Path(".env"); \
  ls=[l for l in p.read_text().splitlines() if not l.startswith("CLAUDE_CODE_OAUTH_TOKEN=")]; \
  p.write_text("\n".join(ls+[f"CLAUDE_CODE_OAUTH_TOKEN={os.environ[\'CLAUDE_CODE_OAUTH_TOKEN\']}"])+"\n")'
```

`claude setup-token` prints the token exactly once, to your terminal. `install.sh` never
captures it — piping it anywhere would put a year-long credential into the installer log.
Note that typing `export CLAUDE_CODE_OAUTH_TOKEN=<token>` writes it to your shell history in
plaintext; scrub that line, and rotate by re-running `claude setup-token` if it leaked.

Verify it took, without printing it:

```bash
grep -c '^CLAUDE_CODE_OAUTH_TOKEN=.\+' .env    # 1 = set
```

Then restart the Claude session and check `rlm_status` — the `cli login:` line should go
green, or `- CLAUDE_CODE_OAUTH_TOKEN is set in .env`.

### Model selection

Role→model mapping lives in one place — `better_rlm/models.py` — not hardcoded across the codebase.

- **API key:** each role uses its configured model verbatim (root `claude-sonnet-5`, override
  `claude-opus-4-8`, sub `claude-haiku-4-5`).
- **Claude Code OAuth:** each role maps to the closest **subscription-supported sibling**. Verified
  by live probe: current 4.x IDs work as-is; `claude-fable-5` maps to `claude-opus-4-8` (the API's
  own guidance) and deprecated dated IDs map to their current equivalents.

`rlm_status` prints both configured and resolved models for the active auth mode.

### Providers — one protocol, several endpoints

`provider` names the **wire protocol**, not the vendor, and `anthropic` is the only
value that makes model calls. That is a correctness guard, not a preference: the
session-window ledger, the 95% floor and ceiling-learning live in
`transport._LedgeredTransport`, which is only in the stack when the Anthropic client
is. A Gemini or OpenAI client would spend past a budget that refused nothing, and
half-gated spending is worse than none because the estimate then reports headroom
already consumed. `auth.require_anthropic` refuses them at every door.

**That does not mean Anthropic is the only endpoint.** Plenty of services speak the
Anthropic messages format. Pointing the client at one keeps every safety property,
because the client - and so the ledger - is unchanged; only the URL moves.

```bash
better-rlm --one-shot /endpoint     # Anthropic | MiniMax | custom base URL
```

The picker writes `base_url` to `config.yaml`. `RLM_BASE_URL` overrides it at
registration, the same way `RLM_MODE` does.

#### Worked example: MiniMax

```yaml
# config.yaml
provider: anthropic                       # protocol, not vendor - leave it
mode: api                                 # required; see the warning below
base_url: https://api.minimax.io/anthropic
root_model: MiniMax-M2.7
root_model_override: MiniMax-M3
sub_model: MiniMax-M2.7-highspeed
sub_context_tokens: 204800
```

```bash
# .env
MINIMAX_API_KEY=<your MiniMax key>
```

**Keys are stored per provider**, the way cline-2 keeps a settings entry per provider
id. `anthropic` reads `ANTHROPIC_API_KEY`, `minimax` reads `MINIMAX_API_KEY`, a custom
endpoint reads `RLM_API_KEY`. Two consequences worth knowing:

- Configuring a second provider does not destroy the first one's key. You can switch
  back and forth without re-entering either.
- A provider's key is **never** read for another. A missing `MINIMAX_API_KEY` fails as
  missing rather than quietly sending your Anthropic key to MiniMax.

**`mode` must be `api`.** `auto` and `claude-cli` spawn the `claude` CLI, which talks
to Anthropic whatever `base_url` says. `/status` flags that combination as `IGNORED`
rather than letting it look configured.

Two things are wrong on a non-Anthropic endpoint, both by omission rather than
breakage:

- **Cost reporting.** `COST_PER_MTOK` carries Anthropic rates only. `report_cost`
  defaults to `false`; leave it off.
- **Context derivation.** The engine's model table does not know these ids and returns
  its 128000 default, which under-uses a larger window rather than overflowing it. Set
  `sub_context_tokens` explicitly, as above.

Nothing in the test suite exercises a live non-Anthropic endpoint. The wiring is
tested; the vendor's behaviour is not.

## Built to be left running

**Rate-limit handling.** This server **sacrifices speed for stability**. Every model call — engine
root, engine sub, standalone sub-queries — passes through one process-wide gate
(`better_rlm/ratelimit.py`), regardless of transport:

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

### Operator TUI — configure mode and models from the CLI

The MCP server runs unattended, but configuring it (changing the root
model, comparing `claude-cli` vs `api`) used to mean hand-editing
`config.yaml`. The bundled TUI is a thin Python REPL that
offers that same surface via slash commands — no extra dependency (it's
`rich.prompt`, already a dep) and no extra process (it's just
`python -m better_rlm.cli`).

`better-rlm` with no arguments opens the TUI. It shows the configuration and offers
what to do about it - no flags, no commands to memorise.

```
better-rlm

  ┌ better-rlm ──────────────────────────┐
  │ mode:      api                       │
  │ provider:  MiniMax                   │
  │ api key:   MINIMAX_API_KEY=MISSING   │
  └──────────────────────────────────────┘

  1  Supply your MiniMax key   MINIMAX_API_KEY is not set - every model-backed tool fails
  2  Run guided setup          provider, transport, credentials, connection test, models
  3  Change mode               currently api
  4  Change provider           currently MiniMax
  5  Change models             root MiniMax-M3, sub MiniMax-M2.7
  6  Test the connection       one tiny model call that proves auth works
  7  Run the verify gate       the test suite
  s  Slash commands            type any /command directly
  q  Quit
```

On a **first run** — no credential for the configured provider — it shows the
onboarding screen instead of the menu: a welcome, and one card per provider.

```
                      Welcome to better-rlm
             Connect a model provider to get started.

  ╭──────────────────────────────────────────────────────────╮
  │ ✦  Use your Claude Code subscription   →                 │
  │    Reuse the `claude` CLI login — no API key             │
  ╰──────────────────────────────────────────────────────────╯
  ╭──────────────────────────────────────────────────────────╮
  │ ⚙  MiniMax                                               │
  │    Anthropic-compatible endpoint                         │
  ╰──────────────────────────────────────────────────────────╯

            ↑/↓ navigate, Enter to select, Esc to exit
```

Two questions, then the credential. First the vendor, then — only when there is a
real choice — how to reach it:

| | transports |
|---|---|
| **Claude** | OAuth (proxy, your `claude` CLI login) or API key (host) |
| **MiniMax** | API key only — so it says so and skips the question |

Keys echo **masked** — `sk••••••••••••yz`, first and last two characters
— so you can tell a good paste from an empty clipboard without the key landing in
your scrollback.

On a terminal every list is a **live picker**: arrow keys move a highlight, typing
filters, Enter selects, Esc cancels — cline-2's dialog interaction
(`components/dialogs/*.tsx`), with its `searchable-list` scoring and windowing ported
whole. Off a terminal — a pipe, CI, `--one-shot` — the same lists render as numbered
prompts, so scripts keep working.

**Anything currently broken leads.** The menu is built from the live state, the way
cline-2's `getMainMenuOptions()` filters its rows: a missing key, an endpoint the mode
is ignoring, or a signed-out CLI appears as row 1 with the reason spelled out. A
healthy config shows none of them. Every action returns to the menu, so one session
fixes several things.

Option 2 runs the guided flow end to end, each step narrowing the next:

```
provider  ->  transport  ->  credentials  ->  connection test  ->  models x3
```

Ported from cline-2's onboarding machine (`apps/cli/src/tui/views/onboarding/`), in its
order. The transport screen is skipped when a provider offers one way in - MiniMax is
API-key only - and it says so rather than showing a question with a single answer. The
credential step branches the way cline's `runProviderChange` does: a local-CLI status
screen when the `claude` CLI holds the credential, a form with the endpoint and the key
side by side otherwise.

Two steps are **not** cline's. It runs a connectivity test before offering models, so a
wrong URL or a mistyped key is caught on the screen that produced it instead of at the
first real query; on failure it offers retry, edit, or write-anyway, so being offline
never traps you in an unconfigurable install. And it asks for all three models - the
orchestrator, the hardest-task override, and the per-chunk worker - each with what that
model is actually for, because they are three different jobs with three different bills.

The model list is the **configured endpoint's own**: choose MiniMax and you are offered
MiniMax ids with their context windows and per-Mtok prices, never Claude ids.

Nothing is written to `config.yaml` until the last step, so cancelling at any screen
leaves it untouched.

**It mounts itself when it finishes.** The last step registers this install as Claude
Code's `rlm` MCP server (`-s user`), so finishing setup and being mounted are the same
act. A registration already pointing here is left alone; one pointing at a *different*
install is replaced and the change is reported, because that install reads a different
`config.yaml`. Claude Code only for now. If the `claude` CLI is not on PATH it says so
and prints the command to run by hand.

`/setup` re-runs it. The individual steps stay available as `/mode`, `/provider`,
`/model`, so changing one thing does not mean walking the whole flow.

API keys are read hidden and written straight to `.env` at mode 0600; the value is
never echoed and never logged, only its length and a short digest.

Launch it from the checkout root:

```bash
./run_tui.sh                                # interactive
./run_tui.sh --one-shot /status             # one command, exit (CI-friendly)
./run_tui.sh --one-shot /mode-help          # show the host/proxy comparison
```

On Windows use `run_tui.cmd`, the analog of `run_server.cmd` above — it resolves
`.venv_windows\Scripts\python.exe` and sets `PYTHONUTF8=1` so the picker's
box-drawing characters render:

```bat
run_tui.cmd
run_tui.cmd --one-shot /status
```

Slash commands exposed by the TUI:

| Command | Effect |
|---|---|
| `/setup` | guided configuration: provider, transport, credentials, connection test, models |
| `/help` | list every command |
| `/status` | show current mode / provider / model / cli login state |
| `/mode-help` | side-by-side comparison of `auto` vs `claude-cli` vs `api` (host/proxy terminology from cline-2's mode picker) |
| `/mode` | open the mode picker; writes `mode:` back to `config.yaml` |
| `/provider` | pick the provider for the current mode, then supply its credential |
| `/model`, `/override`, `/sub` | open the corresponding model picker (curated list + custom id) |
| `/test` | run `uv run --extra dev pytest -q` — the same gate the pre-push hook runs |
| `/test-config` | focused pytest on `tests/test_config.py`, `tests/test_auth.py`, `tests/test_transport.py` |
| `/auth-probe` | check the configured endpoint can be reached (free on the CLI path) |
| `/quit` (or `/exit`) | exit |

Same "proxy" vs "host" terminology as
[cline-2's mode picker][cline-mode-picker]: `claude-cli` is the
**proxy** path (spawns the `claude` CLI binary, reuses your existing
Claude Code login), `api` is the **host** path (talks to the model
endpoint directly, requires an API key). `auto` is the "pick the best
one at launch" option that prefers the proxy and falls back to the
host path.

There is no provider picker. `auth.require_anthropic` refuses every
provider but `anthropic` at every door that leads to a model call — only
its transport passes through the session-window ledger, so the others
would spend ungated. `/status` shows the configured provider and flags
it when it is not `anthropic`; changing it is a deliberate hand edit.

`--one-shot` exits with the command's status (pytest's rc for `/test`
and `/test-config`, `2` for an unknown command, else `0`), so CI can
gate on it.

The picker writes changes back to `config.yaml` (not `.env` — `.env`
is still the installer's territory, see `install.sh --auth`). A running
MCP server holds `better_rlm/` from startup, so a picker write is not live
until the server reconnects — `/status` reminds you when this matters.

[cline-mode-picker]: https://github.com/cline/cline/blob/main/cli/src/tui/components/dialogs/mode-picker.tsx

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
| Python too old or too new | better-rlm needs 3.11-3.14. `uv venv --python 3.13` fetches one. |
| `rlm_query` slow on OAuth | Each engine turn spawns a fresh `claude` CLI (~3–4 s cold start), so multi-turn queries are slower than the SDK path — the deliberate speed-for-stability trade, plus a one-time container start. Use `ANTHROPIC_API_KEY` if latency matters more than reusing your subscription. |
| `claude: command not found` (OAuth) | The CLI isn't on the server's PATH; install Claude Code, or set `cli_path` in `config.yaml` to its absolute path. `rlm_status` shows whether it's found. |

---

## Releasing

Pushing a version tag publishes. `git push origin v0.4.0` runs the suite on ubuntu and
windows, builds, creates the GitHub Release, and uploads to PyPI.

```bash
echo 0.4.0 > VERSION && uv run python scripts/sync_version.py
git commit -am "chore: 0.4.0" && git push origin main
git tag v0.4.0 && git push origin v0.4.0      # this publishes
```

`VERSION` is the single source of truth. `pyproject.toml` reads it through setuptools
dynamic metadata and `scripts/sync_version.py` propagates it to `plugin.json`, so the tag,
the wheel and the plugin manifest cannot claim different versions. The workflow refuses a
malformed version, a version `VERSION` does not declare, and a version already released.

The manual path still exists and adds a dry run: *Actions* → *Release* → *Run workflow*, or

```bash
gh workflow run release.yml -f version=0.4.0 -f dry_run=true   # verify + build, publish nothing
```

**A merge to `main` never publishes.** There is no branch trigger and no schedule; `push`
filters on `tags` alone, and `tests/test_release.py::test_a_merge_can_never_publish`
rejects a `branches` key or a tag pattern looser than `vMAJOR.MINOR.PATCH`. That guard
matters because a PyPI version number is burned once and cannot be reclaimed, even after a
yank - so publishing requires a human to name the version by pushing a tag.

Uploads use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI verifies
the workflow's OIDC identity and mints a short-lived credential, so no API token is stored
in this repository. The publish job holds `id-token: write` and nothing else, runs last
because a GitHub Release is reversible and PyPI is not, and ships the artifacts the build
job produced rather than rebuilding, so both indexes serve identical files.

The two-platform gate is not ceremony: every defect found while merging #2, #4 and #5 was
Windows-only and green on Linux.

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
