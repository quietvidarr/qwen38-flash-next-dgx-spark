#!/bin/bash
# Stop the model (models.json "stop"). KEEP_CONTAINER=1 (default) keeps the stopped container.
set -euo pipefail
source "$(dirname "$0")/config.sh"
docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || { log "no container $CONTAINER_NAME"; exit 0; }
docker stop -t 30 "$CONTAINER_NAME" >/dev/null || true
if [ "$KEEP_CONTAINER" = 1 ]; then log "stopped $CONTAINER_NAME (kept)"
else docker rm "$CONTAINER_NAME" >/dev/null; log "stopped and removed $CONTAINER_NAME"; fi
