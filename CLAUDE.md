# rlm-mcp

MCP server exposing Recursive Language Models over oversized contexts. Fork of
`eesb99/rlm-mcp`. See `README.md` for the tool surface and `docs/` for the
architecture notes.

## Branching

This repo uses trunk-based development. Read `TRUNK-BASED-PATTERNS.md` before
any code change. Config: `trunk-based.json` (repo root). Worktrees and long-lived
branches are blocked by a PreToolUse guard.

Verify command: `uv run --extra dev pytest -q` — `python` is not on PATH here, so it
must go through uv, and `--extra dev` is what pulls in pytest (it lives in the `dev`
extra, so a plain `uv run` resolves the project without it). Same command the
pre-push hook runs: `scripts/githooks/pre-push`, wired up by the installers via
`core.hooksPath`.

## Patching the vendored engine

The engine is **vendored source at `./rlm`**, not a dependency — copied from
`alexzhang13/rlm` at `v0.1.3`, provenance and the upstream-merge recipe in
`rlm/UPSTREAM.md`. Engine fixes are ordinary edits there; `import rlm` resolves to
this copy (`rlms` is no longer installed).

One import-time patch remains, and it is not a workaround: `better_rlm/auth.py:patch_engine`
rebinds the engine's `AnthropicClient` so completions route through our transport,
throttle and retry. That is dependency injection — folding it into `rlm/` would make
the engine import from `better_rlm/`, inverting the dependency and breaking the engine's
standalone use. It stays. It is idempotent, guarded by an `_rlmmcp_patched` flag.

The Docker REPL exec-protocol fixes that used to live in `better_rlm/sandbox_patch.py` are
now the engine's own code (`rlm/environments/docker_repl.py`): the hardened result
marker, the capped `locals` echo, the atomic `state.dill` write, the state-load
warning, UTF-8 host writes, and a real `timeout_s`. `tests/test_sandbox.py::
test_template_still_carries_every_hardening` is the regression guard that replaced
the old fail-loudly string match — an upstream merge that reverts the template fails
there. What is left of that module is `better_rlm/sandbox_reap.py`, host-side housekeeping
that sweeps sandboxes abandoned by a dead server.

**Fix engine defects in `rlm/` directly. Do not add a new monkey-patch.**

## Running server, files on disk

Python imports `better_rlm/` once at startup, so editing files changes nothing for an
already-running MCP server until it is reconnected. Never assume a fix is live
because it is on disk.

## Operator TUI (configure mode and models from the CLI)

The operator TUI (`better_rlm/tui.py`, entry `python -m better_rlm.cli` or `./run_tui.sh`)
mirrors cline-2's mode/model picker (`describe.py` is the data-only twin of
cline-2's `describeMode()`). It writes to `config.yaml` only -- never to
`.env` -- and is purely additive: the MCP server's runtime is unaffected.

It does **not** offer a provider picker, and must not grow one back.
`auth.require_anthropic` raises on every provider but `anthropic` at every
door that leads to a model call, because only its transport passes through
the session-window ledger; a picker writing `provider: openai` would produce
a config.yaml the server refuses. `/status` shows the value and flags a bad
one. `config.PROVIDER_KEY_ENV` was deleted in `f855b9c` for the same reason
-- do not re-add it, here or in `describe.py`.

Same "proxy vs host" terminology as cline-2: `claude-cli` is the **proxy**
path (spawns the `claude` CLI), `api` is the **host** path (talks to the
model endpoint directly), `auto` picks between them per launch. See the
TUI's `/mode-help` for the full comparison.

### Design boundary: TUI edits config; MCP server reads it

The TUI edits `config.yaml`; the next server start reads it. Two properties
hold unconditionally and are not up for trade:

  * diffs of `config.yaml` show every config change that ever shipped --
    there is no `~/.config/better-rlm/` store and must not be one;
  * the explicit `claude mcp restart rlm` is the only restart surface --
    no implicit "alias does the right thing" surprise.

**This section used to ban a console-script entry point outright**, to protect a
third property: that multiple checkouts coexist on one machine. Publishing to
PyPI breaks that property for the global name, deliberately. What replaces the
ban:

  * **the global `better-rlm` name resolves to one install** -- the wheel's, or
    whichever checkout linked it last. The per-checkout scripts
    (`./run_tui.sh`, `./run_server.sh`) stay the way to address a *specific*
    checkout, and each MCP registration still carries its own absolute path, so
    coexistence survives everywhere except the bare word;
  * **`better-rlm` dispatches, it never reimplements.** `better_rlm/cli.py` maps
    a subcommand onto the thing that already does that job. A subcommand growing
    its own copy of installer or server logic is the actual regression here --
    two implementations of auth, drifting apart.

### Two shapes: checkout and wheel

`better_rlm.config.IS_CHECKOUT` is the single probe (is there a `pyproject.toml`
next to the package?). Everything that reads a repo-root file must go through
`config_file()` / `env_file()` rather than `PKG_ROOT / "..."`, or it silently
reads nothing in a wheel and falls back to baked-in defaults with no explanation.

| | checkout | `pip install better-rlm` |
|---|---|---|
| `config.yaml` | repo root | `~/.rlm/config.yaml` |
| `.env` | repo root | `~/.rlm/.env` |
| `better-rlm server` | `run_server.sh` | in-process (`server.main()`) |
| `better-rlm auth` | `install.sh --auth` | prints the `claude setup-token` flow |
| `better-rlm install` | `install.sh` | refuses, names the clone command |

`~/.rlm` is **not** a new store -- contexts, logs, cache, the budget ledger and
the source registry already live there (`_DEFAULTS` in `config.py`). Config and
credentials join the directory the tool already owns. A pip install has no
Docker sandbox image either; `sandbox: local` is the fallback.

`claude mcp add rlm -- better-rlm server` is the wheel's registration form. It
is why `server` must not depend on a shell script.

### Publishing

`better-rlm` on PyPI. One version number is burned per release and can never be
reused, even after a yank -- so `release.yml` stays **`workflow_dispatch` only**
(`tests/test_release.py` fails on any other trigger), and `publish` runs last,
after the GitHub release, because that one is reversible and PyPI is not.

Publishing uses **Trusted Publishing (OIDC)**: no PyPI token exists in this
repository's secrets. The `publish` job holds exactly `id-token: write`, and
ships the artifacts the `release` job built rather than rebuilding, so GitHub
and PyPI serve byte-identical files.

The vendored engine ships as top-level `rlm` because that is the name its own 66
absolute imports use and `rlm/UPSTREAM.md`'s merge recipe depends on the tree
matching upstream. Upstream publishes the same import name (as the `rlms`
distribution), so installing both into one environment collides. Do not "fix"
this by renaming the tree -- it would break every future upstream merge.
