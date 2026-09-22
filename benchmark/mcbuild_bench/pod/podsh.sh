#!/usr/bin/env bash
# Run a script on the RunPod pod through the PROXY ssh (ssh.runpod.io), which only offers an
# interactive PTY shell (no remote command, no scp). The script is typed into that shell as a
# heredoc, executed with bash, and every output line is prefixed with "@@ " so the PTY's command
# echo can be filtered out. Usage:
#   podsh.sh -f script.sh            run a local script file on the pod
#   podsh.sh 'cmd1' 'cmd2' ...       run the given lines
#   env: MCB_POD_SSH (required, e.g. <podid>-<hash>@ssh.runpod.io), MCB_POD_KEY (default ~/.ssh/id_ed25519),
#        MCB_POD_TIMEOUT seconds (default 300)
set -uo pipefail
SSH_TARGET="${MCB_POD_SSH:?set MCB_POD_SSH}"
KEY="${MCB_POD_KEY:-$HOME/.ssh/id_ed25519}"
TIMEOUT="${MCB_POD_TIMEOUT:-300}"
TAG="@@ "
if [ "${1:-}" = "-f" ]; then BODY="$(cat "$2")"; else BODY="$(printf '%s\n' "$@")"; fi
{
  printf 'export PS1="" PATH="$HOME/.local/bin:$PATH"; stty -echo 2>/dev/null\n'
  printf 'cat > /tmp/mcb_cmd.sh <<'"'"'__MCB_EOF__'"'"'\n%s\n__MCB_EOF__\n' "$BODY"
  printf 'bash /tmp/mcb_cmd.sh 2>&1 | sed -u "s/^/%s/"; echo "%sEXIT=${PIPESTATUS[0]}"\n' "$TAG" "$TAG"
  printf 'exit\n'
} | timeout "$TIMEOUT" ssh -tt -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -i "$KEY" "$SSH_TARGET" 2>&1 \
  | tr -d '\r' | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
  | grep -o '@@ .*' | sed 's/^@@ //' || true
