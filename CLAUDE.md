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

One import-time patch remains, and it is not a workaround: `src/auth.py:patch_engine`
rebinds the engine's `AnthropicClient` so completions route through our transport,
throttle and retry. That is dependency injection — folding it into `rlm/` would make
the engine import from `src/`, inverting the dependency and breaking the engine's
standalone use. It stays. It is idempotent, guarded by an `_rlmmcp_patched` flag.

The Docker REPL exec-protocol fixes that used to live in `src/sandbox_patch.py` are
now the engine's own code (`rlm/environments/docker_repl.py`): the hardened result
marker, the capped `locals` echo, the atomic `state.dill` write, the state-load
warning, UTF-8 host writes, and a real `timeout_s`. `tests/test_sandbox.py::
test_template_still_carries_every_hardening` is the regression guard that replaced
the old fail-loudly string match — an upstream merge that reverts the template fails
there. What is left of that module is `src/sandbox_reap.py`, host-side housekeeping
that sweeps sandboxes abandoned by a dead server.

**Fix engine defects in `rlm/` directly. Do not add a new monkey-patch.**

## Running server, files on disk

Python imports `src/` once at startup, so editing files changes nothing for an
already-running MCP server until it is reconnected. Never assume a fix is live
because it is on disk.

## Operator TUI (configure mode and models from the CLI)

The operator TUI (`src/tui.py`, entry `python -m src.cli` or `./run_tui.sh`)
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

The MCP wiring (`claude mcp add rlm -- bash /abs/path/run_server.sh`) is a
**fixed pointer** to this checkout. It starts `python -m src.server`, which
reads `./config.yaml` relative to its own location. The TUI edits
`config.yaml`; the next server start reads it. Two properties hold
unconditionally and are not up for trade:

  * diffs of `config.yaml` show every config change that ever shipped --
    there is no `~/.config/better-rlm/` store and must not be one;
  * the explicit `claude mcp restart rlm` is the only restart surface --
    no implicit "alias does the right thing" surprise.

**`better-rlm` on PATH: what it costs.** This section previously banned a
console-script entry point outright, to protect a third property -- that
multiple checkouts coexist on one machine. `[project.scripts]` plus the
`~/.local/bin` link in `install.sh` breaks that property *for the global name
only*, and deliberately: an operator signing in for the first time should not
have to know where the checkout lives.

The rule that replaces the ban:

  * **the global name resolves to one checkout** -- whichever installer ran
    last. `install.sh` reports the re-point instead of doing it silently, and
    `uninstall.sh` refuses to remove a link owned by a different checkout;
  * **the per-checkout scripts stay the addressable path.** `./install.sh`,
    `./run_tui.sh` and `./run_server.sh` are how you reach a *specific*
    checkout, and the MCP registration still carries its own absolute path --
    so coexistence survives everywhere except the bare word `better-rlm`;
  * **`better-rlm` dispatches, it never reimplements.** `src/cli.py` maps a
    subcommand onto the script that already does that job. A subcommand that
    grows its own copy of installer or server logic is the actual regression
    here -- two implementations of auth, drifting apart.

A global config store is still banned. So is a `better-rlm` that does work of
its own rather than handing off to a launcher.
