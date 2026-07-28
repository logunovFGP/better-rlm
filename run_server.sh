#!/usr/bin/env bash
# Launch the RLM MCP server (stdio transport). Forked from eesb99/rlm-mcp.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# Match run_server.cmd: host-written files must be UTF-8 for the Linux guest.
# Already the default here; set explicitly so both launchers behave identically.
export PYTHONUTF8=1
# POSIX-only venv (.venv_sh) — kept separate from the Windows .venv_windows so a
# WSL-shared checkout doesn't cross-clobber interpreters.
exec "$DIR/.venv_sh/bin/python" -m src.server
