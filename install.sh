#!/usr/bin/env bash
# One-shot setup: venv + pinned deps + Docker sandbox image, then print the
# `claude mcp add` command. Idempotent.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# --register mirrors install.ps1's -Register so both platforms offer the same opt-in.
# Registration stays OPT-IN on purpose: it writes ~/.claude.json (global state, outside
# this checkout), and `claude mcp add` exits 1 on a name that already exists — so doing
# it by default would break the idempotency this script promises above.
REGISTER=0
for arg in "$@"; do
  case "$arg" in
    --register) REGISTER=1 ;;
    -h|--help) echo "usage: ./install.sh [--register]"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

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
  # Non-fatal: a registry timeout on the base image must not abort setup under `set -e`
  # and skip the .env and skill steps below, which need no Docker at all. Observed:
  # "load metadata for python:3.11-slim: DeadlineExceeded" left the skill un-linked.
  docker build -t rlm-sandbox -f docker/Dockerfile.sandbox docker/ || {
    echo "  WARNING: image build failed (registry timeout / offline?)."
    if docker image inspect rlm-sandbox >/dev/null 2>&1; then
      echo "           Existing rlm-sandbox image kept — the sandbox still works."
    else
      echo "           No rlm-sandbox image present: rlm_exec/rlm_query will fail."
    fi
    echo "           Re-run ./install.sh when the network recovers."
  }
else
  echo "  WARNING: docker unavailable. Set 'sandbox: local' in config.yaml to run"
  echo "           on the host (less safe — see README Security)."
fi

echo "==> .env (optional — mode=auto reuses your Claude Code login, no key needed)"
[ -f .env ] || { cp .env.example .env; echo "  created .env (only needed for mode: api — add ANTHROPIC_API_KEY there)"; }

echo "==> Skills — make Claude reach for RLM on oversized inputs"
# Symlinked, not copied: editing skills/<name>/SKILL.md takes effect immediately with no
# reinstall. Loops over skills/*/ so adding a skill needs no change here.
mkdir -p "$HOME/.claude/skills"
for SRC in "$DIR"/skills/*/; do
  SRC="${SRC%/}"
  NAME="$(basename "$SRC")"
  LINK="$HOME/.claude/skills/$NAME"
  if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
    echo "  WARNING: $LINK exists and is not a symlink — leaving it. Copy skills/$NAME there manually."
  else
    ln -sfn "$SRC" "$LINK"
    echo "  linked $LINK -> $SRC  (shared by CLI + desktop)"
  fi
done

if [ "$REGISTER" -eq 1 ]; then
  echo "==> Register with Claude Code (--register)"
  if ! command -v claude >/dev/null 2>&1; then
    echo "  WARNING: claude CLI not on PATH — cannot auto-register. Run manually:"
    echo "    claude mcp add -s user rlm -- bash \"$DIR/run_server.sh\""
  elif claude mcp get rlm >/dev/null 2>&1; then
    # Keep --register re-runnable: an existing 'rlm' may point at a DIFFERENT checkout,
    # so report it instead of failing (`claude mcp add` would exit 1) or hijacking it.
    echo "  'rlm' is already registered — left as-is. To point it at THIS checkout:"
    echo "    claude mcp remove -s user rlm"
    echo "    claude mcp add -s user rlm -- bash \"$DIR/run_server.sh\""
  else
    claude mcp add -s user rlm -- bash "$DIR/run_server.sh"
    echo "  Registered. Restart Claude Code to load the server and skill."
  fi
fi

cat <<MSG

==> Register with Claude Code (CLI + desktop share ~/.claude, so one user-scoped
    registration surfaces in both). Skip if you used --register:

  claude mcp add -s user rlm -- bash "$DIR/run_server.sh"
  claude mcp list            # confirm 'rlm' is listed

Auth: mode=auto reuses your Claude Code login via the \`claude\` CLI — no API key needed.
      (SDK path instead: put ANTHROPIC_API_KEY in $DIR/.env and set mode: api, or
      register with: claude mcp add -s user rlm -e RLM_MODE=api -- bash "$DIR/run_server.sh")

Restart/reopen your Claude session so the 'rlm' server AND the 'rlm-large-context'
skill load. Then ask about any oversized log/dump — Claude will reach for RLM.
MSG
