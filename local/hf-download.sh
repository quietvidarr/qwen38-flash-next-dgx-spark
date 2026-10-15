#!/bin/bash
# Resumable, verified download of an HF repo at a pinned revision with curl (8 parallel, no hf_xet).
# usage: [INCLUDE=regex] [REF=branch] local/hf-download.sh REPO REVISION
#   INCLUDE: only paths matching this extended regex; REF: the refs/ name written on success (default main)
# Files go into the HF cache (HF_HUB_CACHE, default ~/.cache/huggingface/hub) as blobs/<sha256 for LFS | git oid>
# plus snapshots/<rev>/<path> links; LFS files are checked against their sha256. Needs curl, jq, flock.
set -u -o pipefail
REPO=$1; REV=$2
exec 7>"${TMPDIR:-/tmp}/hf-download-${REPO//\//--}.lock"; flock -n 7 || { echo "already running"; exit 1; }
M="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}/models--${REPO//\//--}"
mkdir -p "$M/blobs" "$M/snapshots/$REV" "$M/refs"
LIST=$(mktemp)
curl -sf "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" |
  jq -r '.[]|select(.type=="file")|"\(.lfs.oid // .oid) \(.size) \(if .lfs then "lfs" else "git" end) \(.path)"' |
  grep -E " ${INCLUDE:-.*}\$" > "$LIST" ||
  { echo "could not list $REPO@$REV"; exit 1; }
fetch() {
  local oid=$1 size=$2 kind=$3 path=$4 blob=$M/blobs/$1 part=$M/blobs/$1.curl.part
  if [ -e "$blob" ] && [ "$(stat -L -c %s "$blob")" = "$size" ]; then echo "have $path"; else
    for i in $(seq 1 500); do
      [ -f "$part" ] && [ "$(stat -c %s "$part")" -ge "$size" ] && break
      curl -sSL --retry 5 --connect-timeout 20 --speed-limit 20000 --speed-time 60 -C - \
        "https://huggingface.co/$REPO/resolve/$REV/$path" -o "$part" || sleep 5
    done
    if [ "$kind" = git ] || [ "$(sha256sum "$part" | cut -d' ' -f1)" = "$oid" ]; then mv "$part" "$blob"; echo "OK $path"
    else echo "BAD CHECKSUM $path (removed, rerun)"; rm -f "$part"; return 1; fi
  fi
  local link=$M/snapshots/$REV/$path up=../..
  mkdir -p "$(dirname "$link")"
  local d=$path; while [[ $d == */* ]]; do d=${d#*/}; up=../$up; done
  ln -sfn "$up/blobs/$oid" "$link"
}
export -f fetch; export M REV REPO
for round in 1 2 3 4 5; do
  xargs -P 8 -L 1 bash -c 'fetch $0 $1 $2 $3' < "$LIST" && { rc=0; break; }
  rc=$?; echo "=== round $round had failures, retrying"
done
[ "$rc" = 0 ] && printf %s "$REV" > "$M/refs/${REF:-main}"
rm -f "$LIST"
echo "=== download done rc=$rc $(date)"
exit "$rc"
