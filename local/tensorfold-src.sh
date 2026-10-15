#!/bin/bash
# Fetch our TensorFold fork at the pinned commit (local/pins.env: TENSORFOLD_FORK, TENSORFOLD_FORK_REV) into
# TENSORFOLD_SRC (default .cache/TensorFold-spark) and print that directory. Reuses a clone that has the commit.
#   local/tensorfold-src.sh [REV]
set -euo pipefail
source "$(dirname "$0")/config.sh"
rev="${1:-$TENSORFOLD_FORK_REV}"; src="$TENSORFOLD_SRC"
if ! git -C "$src" cat-file -e "$rev^{commit}" 2>/dev/null; then
  [ -d "$src/.git" ] || { mkdir -p "$src"; git init -q "$src"; }
  GIT_TERMINAL_PROMPT=0 git -C "$src" fetch -q "$TENSORFOLD_FORK" "$rev" >&2 || {
    log "cannot fetch $TENSORFOLD_FORK @ $rev (network?)," >&2
    log "or clone it yourself: git clone $TENSORFOLD_FORK $src" >&2; exit 1; }
  git -C "$src" cat-file -e "$rev^{commit}" || { log "$rev is not in $TENSORFOLD_FORK" >&2; exit 1; }
  git -C "$src" update-ref "refs/pinned/$rev" "$rev"          # keep it past a gc
fi
echo "$src"
