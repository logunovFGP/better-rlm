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

The `rlms` dependency is pinned and its internals are patched from `src/`, never
in `site-packages` — see `src/auth.py:patch_engine` (transport/auth) and
`src/sandbox_patch.py:patch_sandbox` (Docker REPL exec protocol). Both are
idempotent, guarded by an `_rlmmcp_patched` flag on the target module. A patch
that string-matches vendored source must fail loudly when the match is gone, so a
version bump surfaces as an error instead of a silent no-op.

## Running server, files on disk

Python imports `src/` once at startup, so editing files changes nothing for an
already-running MCP server until it is reconnected. Never assume a fix is live
because it is on disk.
