#!/usr/bin/env bash
# Launch the RLM MCP server (stdio transport). Forked from eesb99/rlm-mcp.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec "$DIR/.venv/bin/python" -m src.server
