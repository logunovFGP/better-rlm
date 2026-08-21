#!/usr/bin/env bash
# Reverse ./install.sh. Idempotent, and never removes anything owned by a DIFFERENT checkout.
#
# The hard part of uninstalling is not deletion, it is ownership. install.sh writes three
# things outside this directory — the `rlm` MCP registration, the ~/.claude/skills links,
# and (indirectly) the shared rlm-sandbox image — and it deliberately refuses to hijack any
# of them when they already belong to another checkout, which is what lets several checkouts
# coexist. An uninstaller that just ran `claude mcp remove -s user rlm` would break whichever
# checkout happens to own the registration right now. So every global artefact is checked
# against THIS directory before it is touched, and reported instead when it does not match.
#
# Tiers, because "uninstall" must not mean "lose data":
#   default        reverse what install.sh created, for this checkout only
#   --purge-data   also delete ~/.rlm (loaded contexts + logs, SHARED by every checkout)
#   --image        also delete the rlm-sandbox Docker image (SHARED by every checkout)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PURGE_DATA=0
REMOVE_IMAGE=0
DRY=0
for arg in "$@"; do
  case "$arg" in
    --purge-data) PURGE_DATA=1 ;;
    --image)      REMOVE_IMAGE=1 ;;
    -n|--dry-run) DRY=1 ;;
    -h|--help)
      cat <<'USAGE'
usage: ./uninstall.sh [--purge-data] [--image] [--dry-run]

Reverses ./install.sh for THIS checkout. Safe to re-run.

  --purge-data  also delete ~/.rlm (loaded contexts and logs — shared by every
                checkout, and the only copy of that data)
  --image       also delete the rlm-sandbox Docker image (shared by every checkout)
  --dry-run     print what would happen, change nothing

Left alone on purpose: a registration or skill link owned by another checkout, a
.env you edited, and anything git can remove for you (see the closing note).
USAGE
      exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# Single choke point for every mutation so --dry-run cannot miss one.
run() {
  if [ "$DRY" -eq 1 ]; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi
}

# Past-tense confirmation, suppressed under --dry-run: the `[dry-run] <cmd>` line above
# already says what would run, and printing "removed X" when nothing was removed is a lie.
# Warnings and "nothing to do" lines are unconditional — they are true in both modes.
did() { [ "$DRY" -eq 1 ] || printf '  %s\n' "$1"; }

# 1) MCP registration ---------------------------------------------------------
# First, not last: while the registration stands, Claude Code can relaunch the server at
# any moment, and a server started after we delete .venv_sh below fails on a missing
# interpreter instead of simply being gone.
echo "==> MCP registration ('rlm', user scope)"
if ! command -v claude >/dev/null 2>&1; then
  echo "  claude CLI not on PATH — cannot check. If 'rlm' is registered, remove it with:"
  echo "    claude mcp remove -s user rlm"
else
  REG="$(claude mcp get rlm 2>/dev/null)" || REG=""
  if [ -z "$REG" ]; then
    echo "  not registered — nothing to do"
  elif printf '%s' "$REG" | grep -qF "$DIR/run_server.sh"; then
    run claude mcp remove -s user rlm
    did "removed (was pointing at this checkout)"
  else
    echo "  WARNING: 'rlm' is registered to a DIFFERENT checkout — left as-is."
    echo "           Removing it here would uninstall that one. To see which:"
    echo "             claude mcp get rlm"
  fi
fi

# 2) Skill links --------------------------------------------------------------
# Loops skills/*/ for the same reason install.sh does: adding a skill needs no edit here.
echo "==> Skill links (~/.claude/skills)"
for SRC in "$DIR"/skills/*/; do
  SRC="${SRC%/}"
  NAME="$(basename "$SRC")"
  LINK="$HOME/.claude/skills/$NAME"
  if [ -L "$LINK" ]; then
    # -L before -e: a dangling symlink still needs removing, and -e is false for one.
    TARGET="$(readlink "$LINK")"
    if [ "$TARGET" = "$SRC" ]; then
      run rm -f "$LINK"
      did "removed $LINK"
    else
      echo "  WARNING: $LINK points at another checkout ($TARGET) — left as-is."
    fi
  elif [ -e "$LINK" ]; then
    echo "  WARNING: $LINK is a real directory, not a link — left as-is (not ours to delete)."
  else
    echo "  $NAME not linked — nothing to do"
  fi
done

# 3) Verify gate --------------------------------------------------------------
echo "==> Verify gate (core.hooksPath)"
if ! git -C "$DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "  not a git checkout — skipped"
else
  CUR="$(git -C "$DIR" config --local --get core.hooksPath 2>/dev/null)" || CUR=""
  if [ "$CUR" = "scripts/githooks" ]; then
    run git -C "$DIR" config --local --unset core.hooksPath
    did "unset core.hooksPath ('git push' no longer runs the verify gate)"
  elif [ -z "$CUR" ]; then
    echo "  not set — nothing to do"
  else
    echo "  core.hooksPath is '$CUR', not ours — left as-is"
  fi
fi

# 4) Python env + build artefacts --------------------------------------------
# .venv goes too: the pre-push gate and `uv run` recreate it on demand, so removing it
# costs nothing and leaving it behind contradicts "uninstalled".
echo "==> Python env + build artefacts"
for P in .venv_sh .venv rlm_mcp.egg-info; do
  if [ -e "$P" ]; then
    run rm -rf "$P"
    did "removed $P"
  else
    echo "  $P absent — nothing to do"
  fi
done

# 5) .env ---------------------------------------------------------------------
# install.sh creates this by copying .env.example. Untouched, it holds no secret and can go.
# Edited, it may hold CLAUDE_CODE_OAUTH_TOKEN or an API key — deleting that silently would
# destroy a credential the operator pasted, so it is reported and kept instead.
echo "==> .env"
if [ ! -f .env ]; then
  echo "  absent — nothing to do"
elif cmp -s .env .env.example; then
  run rm -f .env
  did "removed (byte-identical to .env.example — no secrets in it)"
else
  echo "  WARNING: .env differs from .env.example — kept, it may hold a token or API key."
  echo "           Delete it yourself once you have saved anything you need:"
  echo "             rm $DIR/.env"
fi

# 6) Loaded contexts + logs (opt-in) -----------------------------------------
echo "==> Store dir (~/.rlm)"
STORE="$HOME/.rlm"
if [ ! -d "$STORE" ]; then
  echo "  absent — nothing to do"
elif [ "$PURGE_DATA" -eq 1 ]; then
  run rm -rf "$STORE"
  did "removed $STORE (--purge-data)"
else
  N="$(find "$STORE/contexts" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')"
  echo "  kept $STORE ($N contexts) — SHARED by every checkout, and the only copy of that"
  echo "  data. Delete with: ./uninstall.sh --purge-data"
fi

# 7) Sandbox image (opt-in) --------------------------------------------------
echo "==> Docker sandbox image (rlm-sandbox)"
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "  docker unavailable — skipped"
elif ! docker image inspect rlm-sandbox >/dev/null 2>&1; then
  echo "  no rlm-sandbox image — nothing to do"
elif [ "$REMOVE_IMAGE" -eq 1 ]; then
  # Non-fatal: another checkout's container may still reference the image, and that must
  # not abort the run under `set -e` after everything above already succeeded.
  run docker image rm rlm-sandbox \
    || echo "  WARNING: could not remove (in use?) — try: docker image rm -f rlm-sandbox"
else
  echo "  kept — SHARED by every checkout. Delete with: ./uninstall.sh --image"
fi

echo
if [ "$DRY" -eq 1 ]; then
  echo "Dry run — nothing was changed. Re-run without --dry-run to apply."
else
  # "anything it owned", not "no longer registered": a registration or link belonging to
  # another checkout is deliberately left standing above, and claiming otherwise here
  # would contradict the warning the operator just read.
  echo "Done. Anything this checkout owned is removed; its files are still here."
fi

cat <<MSG

Not handled on purpose: other gitignored working state (.rlm_workspace/, __pycache__,
.pytest_cache). git does that better than a hand-rolled list that would drift:
    git -C "$DIR" clean -xdn      # preview
    git -C "$DIR" clean -xdf      # delete

To remove it entirely, delete the directory:
    rm -rf "$DIR"
MSG
