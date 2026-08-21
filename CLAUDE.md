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
