#!/bin/bash
# Start the model for the router (models.json "start"). Reuses the kept container when its image and
# settings are unchanged; otherwise recreates it. Waits until it serves, or exits non-zero with the log.
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$RECIPE_DIR"
IMG="$(local_image)"
docker image inspect "$IMG" >/dev/null 2>&1 || { log "image $IMG missing: run local/build.sh"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] && [ -f "$VISION_FILE" ] || { log "weights missing: run local/download.sh"; exit 1; }
TOWER="$(readlink -f "$VISION_FILE")"
mkdir -p runtime-cache/{home,torch-extensions,triton}

ENVS=(STREAMS KV_DTYPE PREFETCH TENSORFOLD_MEMORY_RESERVE_GIB TF_EXL3_NGRAM_READ TF_EXL3_NGRAM_READERS TF_EXL3_PROMPT_ROWS TF_FLASH_HC12 TF_FLASH_KEEP SERVED_NAME
      DEFAULT_TEMPERATURE DEFAULT_TOP_K DEFAULT_TOP_P DEFAULT_MIN_P TF_PROFILE TF_PROFILE_TRACE PORT TF_POINTS_LAST TF_POINTS_GAP)
args=(--name "$CONTAINER_NAME" --label qwen.recipe=tensorfold-exl3-405-local --restart no
      --gpus all --network host --memory "$MEMORY_LIMIT" --memory-swap "$MEMORY_LIMIT" --shm-size 2g
      --read-only --tmpfs /tmp:rw,noexec,nosuid,size=1g --user "$(id -u):$(id -g)"
      -v "$MODEL_CACHE:/hf/model:ro" -v "$TOWER:/vision/vision_tower_bf16.safetensors:ro"
      -v "$RECIPE_DIR/runtime-cache:/cache:rw"
      -e HOME=/cache/home -e TORCH_EXTENSIONS_DIR=/cache/torch-extensions -e TRITON_CACHE_DIR=/cache/triton
      -e MAX_JOBS=1 -e TENSORFOLD_NO_UPDATE_CHECK=1)
for v in "${ENVS[@]}"; do args+=(-e "$v=${!v}"); done
if [ -n "$PROMPT_CACHE_DIR" ]; then              # patch 0006: kept prompt points on disk
  mkdir -p "$PROMPT_CACHE_DIR"
  args+=(-v "$PROMPT_CACHE_DIR:/prompt-cache:rw" -e TF_PROMPT_CACHE_DIR=/prompt-cache
         -e "TF_PROMPT_CACHE_KEY=$IMG|$MODEL_REPO@$MODEL_REVISION|$KV_DTYPE")
  for v in TF_PROMPT_CACHE_GIB TF_PROMPT_CACHE_TTL_H TF_PROMPT_CACHE_PINNED_GIB TF_PROMPT_CACHE_PIN_AFTER \
           TF_PROMPT_CACHE_QUEUE_GIB TF_PROMPT_CACHE_EXIT_S; do
    args+=(-e "$v=${!v}"); done
fi
args+=("$IMG" --model "/hf/model/snapshots/$MODEL_REVISION" --vision-weights /vision/vision_tower_bf16.safetensors
       --host "$HOST" --port "$PORT")
want=$(printf '%s\n' "${args[@]}" | sha256sum | cut -c1-16)

have=$(docker inspect -f '{{index .Config.Labels "qwen.local.config"}}' "$CONTAINER_NAME" 2>/dev/null || true)
if [ -n "$have" ] && [ "$have" = "$want" ]; then
  log "starting kept container $CONTAINER_NAME"
  docker start "$CONTAINER_NAME" >/dev/null
else
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    owner=$(docker inspect -f '{{index .Config.Labels "qwen.recipe"}}' "$CONTAINER_NAME")
    [ "$owner" = tensorfold-exl3-405-local ] || { log "a container named $CONTAINER_NAME exists that is not ours"; exit 1; }
    log "settings or image changed: recreating $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
  log "creating $CONTAINER_NAME from $IMG (streams $STREAMS, kv $KV_DTYPE, prefetch $PREFETCH, n-gram $TF_EXL3_NGRAM_READ)"
  docker run -d --label "qwen.local.config=$want" "${args[@]}" >/dev/null
fi

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$SERVED_NAME\""; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]; then
    log "container exited:"; docker logs --tail 40 "$CONTAINER_NAME" 2>&1; exit 1
  fi
  (( $(date +%s) - t0 > ${READY_TIMEOUT:-2400} )) && { log "not ready after ${READY_TIMEOUT:-2400}s"; exit 1; }
  sleep 2
done
log "ready on $HOST:$PORT as $SERVED_NAME after $(( $(date +%s) - t0 ))s"
docker logs "$CONTAINER_NAME" 2>&1 | grep -E "serve_local|sequence|streams|n-gram|GiB" | tail -6 || true
