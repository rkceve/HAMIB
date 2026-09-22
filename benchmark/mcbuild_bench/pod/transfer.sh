#!/usr/bin/env bash
# Type a small tarball into the pod through the proxy ssh (no scp available) and unpack it.
# Usage: transfer.sh <local.tgz> <remote-dir>      (env MCB_POD_SSH / MCB_POD_KEY as in podsh.sh)
# The tarball is base64-encoded (76-char lines), sent in chunks as heredocs, then decoded and
# verified by sha256 on the pod before extraction.
set -euo pipefail
TGZ="$1"; RDIR="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PODSH="$HERE/podsh.sh"
B64="$(mktemp)"; base64 -w 76 "$TGZ" > "$B64"
SHA="$(sha256sum "$TGZ" | cut -d' ' -f1)"
LINES=$(wc -l < "$B64"); CHUNK=${MCB_CHUNK_LINES:-6000}   # 6000 lines x 76 chars ≈ 450 KB per ssh session
echo "sending $(wc -c < "$TGZ") bytes as $LINES base64 lines in chunks of $CHUNK"
MCB_POD_TIMEOUT=120 bash "$PODSH" "mkdir -p '$RDIR' && rm -f '$RDIR/snapshot.b64' && echo reset-ok"
i=0; n=0
while [ $i -lt "$LINES" ]; do
  n=$((n+1)); PART="$(mktemp)"
  { printf "cat >> '%s/snapshot.b64' <<'__B64__'\n" "$RDIR"; sed -n "$((i+1)),$((i+CHUNK))p" "$B64"; printf '__B64__\necho chunk-%d-ok $(wc -l < %s/snapshot.b64)\n' "$n" "$RDIR"; } > "$PART"
  MCB_POD_TIMEOUT=900 bash "$PODSH" -f "$PART"; rm -f "$PART"
  i=$((i+CHUNK))
done
MCB_POD_TIMEOUT=300 bash "$PODSH" \
  "cd '$RDIR' && base64 -d snapshot.b64 > snapshot.tgz && echo \"remote sha: \$(sha256sum snapshot.tgz | cut -d' ' -f1)\" && echo 'local  sha: $SHA'" \
  "cd '$RDIR' && [ \"\$(sha256sum snapshot.tgz | cut -d' ' -f1)\" = '$SHA' ] && tar xzf snapshot.tgz && rm snapshot.b64 && echo EXTRACTED && ls" \
  "cat '$RDIR'/cms-prototype/SNAPSHOT_REF 2>/dev/null"
rm -f "$B64"
