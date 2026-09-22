#!/usr/bin/env bash
# Send ONE small/medium text file to the pod via base64 (safe against PTY canonical-mode
# line editing / wrapping that can corrupt a raw heredoc for longer or wide-character files).
# Usage: sendfile.sh <local-file> <remote-path>
set -euo pipefail
LOCAL="$1"; REMOTE="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
SHA="$(sha256sum "$LOCAL" | cut -d' ' -f1)"
B64="$(mktemp)"; base64 -w 100 "$LOCAL" > "$B64"
PART="$(mktemp)"
{
  printf "mkdir -p \"\$(dirname '%s')\"\n" "$REMOTE"
  printf "cat > '%s.b64' <<'__B64__'\n" "$REMOTE"
  cat "$B64"
  printf "__B64__\n"
  printf "base64 -d '%s.b64' > '%s' && rm '%s.b64'\n" "$REMOTE" "$REMOTE" "$REMOTE"
  printf "echo \"remote sha: \$(sha256sum '%s' | cut -d' ' -f1)\"\n" "$REMOTE"
  printf 'echo "local  sha: %s"\n' "$SHA"
} > "$PART"
MCB_POD_TIMEOUT="${MCB_POD_TIMEOUT:-300}" bash "$HERE/podsh.sh" -f "$PART"
rm -f "$B64" "$PART"
