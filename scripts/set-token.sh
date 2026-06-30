#!/usr/bin/env bash
# Store a Claude Code OAuth token (from `claude setup-token`) into rlm-mcp/.env
# without echoing it to the terminal or shell history.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "1) Run:  claude setup-token   (in another shell) and copy the token it prints."
read -rsp "2) Paste CLAUDE_CODE_OAUTH_TOKEN here (input hidden): " TOK
echo
TOK="$(printf '%s' "$TOK" | tr -d '[:space:]')"
if [ -z "$TOK" ] || [[ "$TOK" == \<* ]]; then
  echo "No valid token entered (got empty or a placeholder). Aborted." >&2
  exit 1
fi
printf 'CLAUDE_CODE_OAUTH_TOKEN=%s\n' "$TOK" > "$DIR/.env"
echo "Wrote $DIR/.env (token not displayed). Next: /mcp → rlm → Reconnect, then run rlm_status."
