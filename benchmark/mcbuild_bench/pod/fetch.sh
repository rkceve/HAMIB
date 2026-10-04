#!/usr/bin/env bash
# Download files from the pod through the PROXY ssh (PTY only, no scp): the pod packs the given
# paths into a gzip tarball, prints its sha256 and size, then streams it as base64 lines, which
# podsh.sh captures. The local side decodes and refuses the archive unless the checksum matches.
# Usage:  fetch.sh <remote-base-dir> <local.tgz> <path> [<path> ...]
#         (paths are relative to <remote-base-dir>; the archive keeps them relative)
# env:    MCB_POD_SSH (required), MCB_POD_KEY, MCB_POD_TIMEOUT (default 900 s)
set -uo pipefail
REMOTE_DIR="${1:?remote base dir}"; LOCAL="${2:?local .tgz}"; shift 2
[ $# -ge 1 ] || { echo "no paths given" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
export MCB_POD_TIMEOUT="${MCB_POD_TIMEOUT:-900}"
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
bash "$HERE/podsh.sh" "cd '$REMOTE_DIR' || exit 3" \
  "tar czf /tmp/mcb_fetch.tgz $(printf "'%s' " "$@") || exit 4" \
  'echo "SHA $(sha256sum /tmp/mcb_fetch.tgz | cut -d" " -f1) SIZE $(stat -c %s /tmp/mcb_fetch.tgz)"' \
  'echo B64_BEGIN; base64 -w 76 /tmp/mcb_fetch.tgz; echo B64_END; rm -f /tmp/mcb_fetch.tgz' > "$TMP"
want_sha="$(grep -m1 '^SHA ' "$TMP" | awk '{print $2}')"
want_size="$(grep -m1 '^SHA ' "$TMP" | awk '{print $4}')"
[ -n "$want_sha" ] || { echo "fetch failed: no checksum line from the pod"; grep -E 'EXIT=' "$TMP"; exit 5; }
sed -n '/^B64_BEGIN$/,/^B64_END$/p' "$TMP" | grep -vE '^B64_(BEGIN|END)$' | tr -d ' \r' | base64 -d > "$LOCAL" 2>/dev/null
got_sha="$(sha256sum "$LOCAL" | cut -d' ' -f1)"; got_size="$(stat -c %s "$LOCAL")"
if [ "$got_sha" = "$want_sha" ] && [ "$got_size" = "$want_size" ]; then
  echo "fetched $LOCAL  bytes=$got_size  sha256=$got_sha  (verified)"
else
  echo "CHECKSUM MISMATCH: pod sha=$want_sha size=$want_size / local sha=$got_sha size=$got_size"; exit 6
fi
