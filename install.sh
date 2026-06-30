#!/usr/bin/env bash
# One-shot setup: venv + pinned deps + Docker sandbox image, then print the
# `claude mcp add` command. Idempotent.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "==> Python env (.venv, 3.12) + dependencies"
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -e ".[dev,pdf]"
else
  python3 -m venv .venv
  .venv/bin/pip install -U pip
  .venv/bin/pip install -e ".[dev,pdf]"
fi

echo "==> Docker sandbox image (rlm-sandbox)"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -t rlm-sandbox -f docker/Dockerfile.sandbox docker/
else
  echo "  WARNING: docker unavailable. Set 'sandbox: local' in config.yaml to run"
  echo "           on the host (less safe — see README Security)."
fi

echo "==> .env"
[ -f .env ] || { cp .env.example .env; echo "  created .env — add your ANTHROPIC_API_KEY"; }

cat <<MSG

==> Register with Claude Code (CLI + desktop share ~/.claude, so one user-scoped
    registration surfaces in both):

  claude mcp add -s user rlm -- bash "$DIR/run_server.sh"
  claude mcp list            # confirm 'rlm' is listed

Then set ANTHROPIC_API_KEY in $DIR/.env before running live queries.
MSG
