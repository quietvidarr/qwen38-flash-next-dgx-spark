# Our settings for this recipe (sourced by local/*.sh). Defaults here; put overrides in ./.env
# (KEY=value lines, git-ignored), never edit upstream files for settings. See LOCAL_CHANGES.md.
#
# Pins come from upstream's own files, so an upstream update moves them with it:
#   weights + vision tower: manifests/exl3-405.json, manifests/vision-bf16.json
#   upstream image tag and TensorFold revision: scripts/supervise.py (IMAGE) and docker/Dockerfile
# local/check-update.sh verifies those patterns still exist before an update is applied.

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RECIPE_DIR/local/pins.env"                 # TensorFold fork + pinned commit (ours)
[ -f "$RECIPE_DIR/.env" ] && { set -a; source "$RECIPE_DIR/.env"; set +a; }

: "${CONTAINER_NAME:=Qwen3.8-Flash-Next-EXL3-4.05bpw-TensorFold}"
: "${SERVED_NAME:=$CONTAINER_NAME}"           # the API model id (upstream: qwen38-flash-next-tf405)
: "${HOST:=127.0.0.1}"                         # the router reaches models on 127.0.0.1:8001
: "${PORT:=8001}"
: "${KEEP_CONTAINER:=1}"                       # stop.sh keeps the stopped container (no auto-start: restart=no)

# Engine options (upstream hardcodes them in scripts/serve.py; local/serve_local.py overrides from here)
: "${STREAMS:=8}"                              # requests decoded together = conversations that keep a cache slot
                                               # (upstream 2; vision needs >= 2)
: "${TF_FLASH_KEEP:=16}"                        # kept prompt ends (resume points) for all slots (patch 0005; upstream 8)
: "${KV_DTYPE:=bf16}"                          # bf16 (exact, upstream) | int8 | int4
: "${PREFETCH:=0}"                             # 1: read + lock the 36 GiB n-gram table into RAM at start
                                               # 0: leave it on the SSD; our pread reader fetches rows (patch 0001)
: "${TENSORFOLD_MEMORY_RESERVE_GIB:=7}"       # upstream's value
: "${TF_EXL3_NGRAM_READ:=pread}"               # pread (patch 0001) | mmap (TensorFold's original page faults)
: "${TF_EXL3_NGRAM_READERS:=128}"            # n-gram reader threads (native reader, patch 0003)
: "${TF_EXL3_PROMPT_ROWS:=4096}"               # prompt piece rows (patch 0002; upstream 2048): bigger = fewer expert
                                               # weight reads a token (faster prefill), ~1.5 GB more scratch
: "${TF_FLASH_HC12:=1}"                         # patch 0017: hyper-connection matrices from the lossless 12-bit store
: "${DEFAULT_TOP_K:=20}"                        # sampling defaults for fields a request omits (empty = upstream:
: "${DEFAULT_TEMPERATURE:=}"                    # greedy, top_k off). top_k 20 = the model's generation_config;
: "${DEFAULT_TOP_P:=}"                          # with top_k off, sampled decode drops to ~10 tok/s (CPU sort)
: "${DEFAULT_MIN_P:=}"
: "${TF_PROFILE:=0}"                            # 1: kernel profiler on demand (see local/serve_local.py)
: "${TF_PROFILE_TRACE:=0}"                      # 1: the profiler also writes a kernel timeline (.json)
: "${MEMORY_LIMIT:=112g}"                      # upstream's container cap (no swap)

# Chat template override (TensorFold TF_CHAT_TEMPLATE): stock template + sticky reasoning notes, so an effort switch
# mid-conversation keeps the earlier prompt (router: reasoning_prefix / reasoning_changes). Empty = the checkpoint's.
: "${CHAT_TEMPLATE=$RECIPE_DIR/local/templates/chat_template.jinja}"

# Prompt cache on disk (patch 0006). Empty PROMPT_CACHE_DIR turns it off.
: "${PROMPT_CACHE_DIR:=${XDG_CACHE_HOME:-$HOME/.cache}/tensorfold-prompt-cache}"   # shared by every TensorFold model; budgets are global
: "${TF_POINTS_LAST:=all}"                      # states at EVERY message start (patch 0010; 3: the last 3; 0: upstream)
: "${TF_POINTS_GAP:=256}"                       # with all: starts before the last 3 at least this far apart (256 = every one; larger only keeps fewer old points, no prefill cost)
: "${TF_PROMPT_CACHE_GIB:=100}"                 # normal points: this many GiB, least recently used out first
: "${TF_PROMPT_CACHE_TTL_H:=72}"                # ... and each deleted this many hours after it was saved
: "${TF_PROMPT_CACHE_PINNED_GIB:=50}"           # pinned points (prompt heads many conversations start with)
: "${TF_PROMPT_CACHE_PIN_AFTER:=3}"             # pinned once this many different continuations resumed from it
: "${TF_PROMPT_CACHE_QUEUE_GIB:=4}"            # host copies waiting for the disk writer; past it points wait on the GPU (0010)
: "${TF_PROMPT_CACHE_EXIT_S:=20}"              # a stopping server waits this long for the writer (stop.sh gives 30 s)

# Derived from upstream files
manifest_field() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$RECIPE_DIR/manifests/$1" "$2"; }
MODEL_REPO="$(manifest_field exl3-405.json repo)"
MODEL_REVISION="$(manifest_field exl3-405.json revision)"
MODEL_BRANCH="$(manifest_field exl3-405.json branch)"
VISION_REPO="$(manifest_field vision-bf16.json repo)"
VISION_REVISION="$(manifest_field vision-bf16.json revision)"
HF_HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
MODEL_CACHE="$HF_HUB/models--${MODEL_REPO//\//--}"
MODEL_DIR="$MODEL_CACHE/snapshots/$MODEL_REVISION"
VISION_FILE="$HF_HUB/models--${VISION_REPO//\//--}/snapshots/$VISION_REVISION/vision_tower_bf16.safetensors"
UPSTREAM_IMAGE="$(sed -n 's/^IMAGE = "\(.*\)"$/\1/p' "$RECIPE_DIR/scripts/supervise.py")"
TENSORFOLD_REV="$(grep -o 'fetch --depth 1 origin [0-9a-f]\{40\}' "$RECIPE_DIR/docker/Dockerfile" | awk '{print $NF}')"

TENSORFOLD_SRC="${TENSORFOLD_SRC:-$RECIPE_DIR/.cache/TensorFold-spark}"   # clone of the fork (local/tensorfold-src.sh)

# Our image: upstream's image with TensorFold from our fork at TENSORFOLD_FORK_REV. Its tag names both inputs, so a
# change to either (upstream pin or fork pin) asks for a rebuild (local/build.sh).
local_image() {
  # TF_LOCAL_IMAGE in .env pins an earlier build (rollback); empty = built from the pinned fork commit
  [ -n "${TF_LOCAL_IMAGE:-}" ] && { echo "$TF_LOCAL_IMAGE"; return; }
  local h
  h=$( (cat "$RECIPE_DIR"/local/Dockerfile "$RECIPE_DIR"/local/serve_local.py; echo "$TENSORFOLD_FORK_REV") | sha256sum | cut -c1-12)
  echo "qwen38-flash-next-tensorfold-local:${UPSTREAM_IMAGE##*:}-$h"
}
log() { printf '[exl3-tf] %s\n' "$*"; }
