#!/bin/bash
# Download the pinned weights and vision tower into the Hugging Face cache (local/hf-download.sh:
# resumable curl, 8 parallel), then verify every file against upstream's manifests.
# Used by the router's weights update (models.json "weights_update").
set -euo pipefail
source "$(dirname "$0")/config.sh"
DL="$RECIPE_DIR/local/hf-download.sh"
REF="$MODEL_BRANCH" "$DL" "$MODEL_REPO" "$MODEL_REVISION"
INCLUDE='vision_tower_bf16\.safetensors' "$DL" "$VISION_REPO" "$VISION_REVISION"
python3 "$RECIPE_DIR/local/verify_weights.py"
