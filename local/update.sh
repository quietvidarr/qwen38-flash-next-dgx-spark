#!/bin/bash
# Safe update: check first (local/check-update.sh), and only if compatible replay our commits onto upstream
# (git rebase), rebuild the image if its inputs changed, and say whether the weights must be downloaded.
#   local/update.sh            check + update the recipe + build the image
#   local/update.sh --check    only the check (same as local/check-update.sh)
# The running model is not restarted; the router picks up the new image at its next start.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
"$HERE/local/check-update.sh"; rc=$?
[ "${1:-}" = --check ] && exit $rc
[ $rc = 0 ] || { echo "update NOT applied (check exit $rc)"; exit $rc; }
UP=$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main)
[ "$(git rev-list --count "$(git merge-base HEAD "$UP")..$UP")" = 0 ] && exit 0
OLD=$(git rev-parse HEAD)
git tag -f "pre-update-$(date +%Y%m%d-%H%M%S)" "$OLD" >/dev/null
if ! git rebase --autostash "$UP"; then
  git rebase --abort; echo "rebase failed although the check passed: left unchanged"; exit 3
fi
echo "== recipe updated: $(git rev-parse --short "$OLD") -> $(git rev-parse --short HEAD) (old state tagged pre-update-*)"
git diff --quiet "$OLD" HEAD -- manifests/ || echo "== weights pins changed: run local/download.sh before the next start"
echo "== building the image for the updated recipe"
"$HERE/local/build.sh" || { echo "image build failed: the previous image is still there; roll back with: git reset --hard $OLD"; exit 1; }
echo "== done. Next model start uses it. Then re-measure (LOCAL_CHANGES.md, 'After an update')."
