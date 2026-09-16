#!/usr/bin/env bash
# Launch the better-rlm operator TUI (configure transport mode and models, run
# tests via slash commands). Does NOT start the MCP server -- that one is
# run_server.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Two shapes ship this file, and they build different virtualenvs.
#
#   checkout  install.sh creates .venv_sh -- the same interpreter run_server.sh
#             uses, so the TUI and the server share one resolved graph.
#   plugin    /plugin never runs install.sh. The server is launched with
#             `uv run --directory ${CLAUDE_PLUGIN_ROOT}`, which resolves into
#             .venv. Pointing this script at .venv_sh there failed with a bare
#             "no such file or directory" and left the TUI looking unavailable
#             when it was shipped and working all along.
#
# Prefer the installer venv, fall back to uv with the plugin's own invocation
# (--extra pdf included so uv reuses that resolution instead of building a
# second one).
if [ -x "$DIR/.venv_sh/bin/python" ]; then
  exec "$DIR/.venv_sh/bin/python" -m better_rlm.cli "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --directory "$DIR" --extra pdf python -m better_rlm.cli "$@"
fi
echo "run_tui.sh: no interpreter found." >&2
echo "  checkout: run ./install.sh first (creates .venv_sh)" >&2
echo "  plugin:   install uv (https://docs.astral.sh/uv/) -- it is already the" >&2
echo "            prerequisite for the plugin's MCP server" >&2
exit 1
