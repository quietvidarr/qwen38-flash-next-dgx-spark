#!/bin/bash
# Build our image: upstream's image (built by upstream's scripts/build_tensorfold.sh if missing: needs the
# network) with TensorFold from our fork at the pinned commit (local/tensorfold-src.sh: needs the network once).
# The docker build itself is offline. Prints the image tag on its last line.
#   local/build.sh          build what is missing
#   local/build.sh --force  rebuild our layer even if the tag exists
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$RECIPE_DIR"
IMG="$(TF_LOCAL_IMAGE= local_image)"     # always the tag of the pinned inputs (TF_LOCAL_IMAGE only pins what start.sh runs)
if ! docker image inspect "$UPSTREAM_IMAGE" >/dev/null 2>&1; then
  log "upstream image $UPSTREAM_IMAGE missing: building it with scripts/build_tensorfold.sh"
  for i in 1 2 3 4 5; do bash scripts/build_tensorfold.sh && break; log "build attempt $i failed (network?), retrying"; sleep 30; done
  docker image inspect "$UPSTREAM_IMAGE" >/dev/null
fi
if [ "${1:-}" != --force ] && docker image inspect "$IMG" >/dev/null 2>&1; then
  log "have $IMG"; echo "$IMG"; exit 0
fi
SRC="$(local/tensorfold-src.sh)"
ctx=$(mktemp -d); trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/local" "$ctx/tensorfold"; cp local/Dockerfile local/serve_local.py "$ctx/local/"
git -C "$SRC" archive "$TENSORFOLD_FORK_REV" | tar -x -C "$ctx/tensorfold"
echo "$TENSORFOLD_FORK_REV" > "$ctx/tensorfold/.fork-rev"
docker build --network none --build-arg BASE="$UPSTREAM_IMAGE" --build-arg TENSORFOLD_FORK_REV="$TENSORFOLD_FORK_REV" \
  -f "$ctx/local/Dockerfile" -t "$IMG" "$ctx"
# the patched runtime imports, and our serve wrapper finds upstream's serve.py
docker run --rm --network none --entrypoint python3 "$IMG" -c '
import sys; sys.path.insert(0, "/opt/recipe")
import serve, inspect
from tensorfold.families.qwen4_exp.cuda import exl3_pack, forward
assert hasattr(exl3_pack, "NGRAM_READ") and "rows=None if ahead is None" in inspect.getsource(forward.stage)
assert callable(serve.engine_options) and serve.MODEL_ID
print("patched runtime OK", serve.engine_options())'
log "built $IMG"
echo "$IMG"
