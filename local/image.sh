#!/bin/bash
# Image helper for the model router (models.json image_cmd / image_update). The image is built here, not
# pulled: "update available" means our image for the current recipe + patches does not exist yet.
#   local/image.sh ref      print "TAG TAG" (the tag local/start.sh will run)
#   local/image.sh update   build it (local/build.sh)
set -euo pipefail
source "$(dirname "$0")/config.sh"
case "${1:-ref}" in
  ref) img="$(local_image)"; echo "$img $img" ;;
  update) exec "$RECIPE_DIR/local/build.sh" ;;
  *) echo "usage: $0 ref|update"; exit 2 ;;
esac
