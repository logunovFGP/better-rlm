#!/usr/bin/env bash
# Launch the better-rlm operator TUI (configure mode / provider / model,
# run tests via slash commands). Mirrors run_server.sh's venv convention.
# Does NOT start the MCP server -- that one is run_server.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# POSIX venv, same name the installer uses (.venv_sh) so the TUI and the
# MCP server share one resolved dependency graph.
exec "$DIR/.venv_sh/bin/python" -m src.cli "$@"
