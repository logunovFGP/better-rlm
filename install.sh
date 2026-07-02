#!/usr/bin/env bash
# One-shot setup: venv + pinned deps + Docker sandbox image, then print the
# `claude mcp add` command. Idempotent.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "==> Python env (.venv_sh, 3.12) + dependencies"
# Always rebuild the venv fresh. Reusing an existing venv proved flaky on a
# WSL-shared checkout; a clean create is reliable and deterministic. Per-platform
# name (.venv_sh) keeps it separate from the Windows .venv_windows. (On POSIX an
# open file survives rm, so a running server is unaffected until its next start.)
rm -rf .venv_sh
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv_sh
  uv pip install --python .venv_sh/bin/python -e ".[dev,pdf]"
else
  python3 -m venv .venv_sh
  .venv_sh/bin/pip install -U pip
  .venv_sh/bin/pip install -e ".[dev,pdf]"
fi

echo "==> Docker sandbox image (rlm-sandbox)"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -t rlm-sandbox -f docker/Dockerfile.sandbox docker/
else
  echo "  WARNING: docker unavailable. Set 'sandbox: local' in config.yaml to run"
  echo "           on the host (less safe — see README Security)."
fi

echo "==> .env (optional — mode=auto reuses your Claude Code login, no key needed)"
[ -f .env ] || { cp .env.example .env; echo "  created .env (only needed for mode: api — add ANTHROPIC_API_KEY there)"; }

echo "==> Skill (rlm-large-context) — makes Claude reach for RLM on oversized inputs"
mkdir -p "$HOME/.claude/skills"
LINK="$HOME/.claude/skills/rlm-large-context"
if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
  echo "  WARNING: $LINK exists and is not a symlink — leaving it. Copy skills/rlm-large-context there manually."
else
  ln -sfn "$DIR/skills/rlm-large-context" "$LINK"
  echo "  linked $LINK -> $DIR/skills/rlm-large-context  (shared by CLI + desktop)"
fi

cat <<MSG

==> Register with Claude Code (CLI + desktop share ~/.claude, so one user-scoped
    registration surfaces in both):

  claude mcp add -s user rlm -- bash "$DIR/run_server.sh"
  claude mcp list            # confirm 'rlm' is listed

Auth: mode=auto reuses your Claude Code login via the \`claude\` CLI — no API key needed.
      (SDK path instead: put ANTHROPIC_API_KEY in $DIR/.env and set mode: api, or
      register with: claude mcp add -s user rlm -e RLM_MODE=api -- bash "$DIR/run_server.sh")

Restart/reopen your Claude session so the 'rlm' server AND the 'rlm-large-context'
skill load. Then ask about any oversized log/dump — Claude will reach for RLM.
MSG
