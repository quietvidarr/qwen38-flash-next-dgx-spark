#!/bin/bash
# Pre-update compatibility check: is the newest upstream recipe compatible with our changes?
# Read-only: the live checkout, image and container are not touched (it works in a temporary worktree).
#
#   local/check-update.sh [UPSTREAM_REF]     (default origin/main after a fetch)
#
# Exit 0 = up to date or compatible (safe for local/update.sh), 3 = incompatible (do not update; see the
# reasons, then adapt local/ or the fork as described in LOCAL_CHANGES.md), 1 = the check itself failed.
# What it checks is the list of "contact points" in LOCAL_CHANGES.md; keep both in sync.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
bad=(); warn=()
fail() { bad+=("$1"); echo "  INCOMPATIBLE: $1"; }
note() { warn+=("$1"); echo "  note: $1"; }

if [ -z "${1:-}" ]; then git fetch -q origin || { echo "fetch failed (network?)"; exit 1; }; fi
UP="${1:-$(git rev-parse --abbrev-ref origin/HEAD 2>/dev/null || echo origin/main)}"
BASE=$(git merge-base HEAD "$UP")
NEW=$(git rev-list --count "$BASE..$UP")
echo "== upstream $UP: $NEW new commit(s) since our base $(git rev-parse --short "$BASE")"
[ "$NEW" = 0 ] && { echo "== up to date"; exit 0; }
git log --oneline "$BASE..$UP" | sed 's/^/   /'
git diff --stat "$BASE" "$UP" | sed 's/^/   /'

WT=$(mktemp -d /tmp/exl3-check-XXXX)
cleanup() { git worktree remove --force "$WT" >/dev/null 2>&1; rm -rf "$WT"; }
trap cleanup EXIT
git worktree add -q --detach "$WT" HEAD

echo "== 1. our recipe commits replay onto upstream (git rebase in a scratch worktree)"
if git -C "$WT" -c user.name=check -c user.email=check@local rebase -q "$UP" >/dev/null 2>&1; then
  echo "  ok: $(git -C "$WT" rev-list --count "$UP..HEAD") local commit(s) replay cleanly"
else
  files=$(git -C "$WT" diff --name-only --diff-filter=U | tr '\n' ' ')
  git -C "$WT" rebase --abort >/dev/null 2>&1
  fail "our commits conflict with upstream in: ${files:-?}"
  git -C "$WT" checkout -q --detach "$UP"          # still check the contact points on upstream's tree
fi
T="$WT"

echo "== 2. contact points in upstream files"
S="$T/scripts/serve.py"
grep -q '^def engine_options():' "$S" || fail "scripts/serve.py: no engine_options() (local/serve_local.py overrides it)"
for k in streams prefetch kv_dtype; do grep -q "$k=" "$S" || fail "scripts/serve.py: engine_options has no '$k'"; done
grep -q '^MODEL_ID = ' "$S" || fail "scripts/serve.py: no MODEL_ID global (serve_local.py sets our served name)"
grep -q 'engine_options()' <(sed -n '/^def main/,$p' "$S") || fail "scripts/serve.py: main() no longer calls engine_options()"
grep -q 'from tensorfold.cuda.server import .*App' <(sed -n '/^def main/,$p' "$S") || fail "scripts/serve.py: main() no longer imports App from tensorfold.cuda.server (serve_local.py wraps it for sampling defaults)"
grep -q 'sampling=' <(sed -n '/^def main/,$p' "$S") || fail "scripts/serve.py: main() no longer passes sampling= to App (serve_local.py fills DEFAULT_* there)"
grep -q 'COPY scripts/serve.py /opt/recipe/serve.py' "$T/docker/Dockerfile" || fail "docker/Dockerfile: serve.py not at /opt/recipe/serve.py"
grep -q '^IMAGE = "' "$T/scripts/supervise.py" || fail "scripts/supervise.py: no IMAGE = \"...\" (local/config.sh reads the image tag there)"
NEWREV=$(grep -o 'fetch --depth 1 origin [0-9a-f]\{40\}' "$T/docker/Dockerfile" | awk '{print $NF}')
[ -n "$NEWREV" ] || fail "docker/Dockerfile: TensorFold pin pattern 'fetch --depth 1 origin <sha>' not found"
grep -q '/opt/TensorFold' "$T/docker/Dockerfile" || fail "docker/Dockerfile: TensorFold no longer in /opt/TensorFold (local/Dockerfile replaces it there)"
for f in exl3-405.json vision-bf16.json; do
  [ -f "$T/manifests/$f" ] || fail "manifests/$f is gone (local/config.sh takes the weights pins from it)"
done
for k in repo revision branch; do
  python3 -c "import json,sys; json.load(open(sys.argv[1]))[sys.argv[2]]" "$T/manifests/exl3-405.json" $k 2>/dev/null || fail "manifests/exl3-405.json: no '$k'"
done
if ! git diff --quiet "$BASE" "$UP" -- scripts/supervise.py; then
  note "scripts/supervise.py changed: compare its docker_command() with local/start.sh (mounts, env, limits):"
  git diff "$BASE" "$UP" -- scripts/supervise.py | grep '^[+-][^+-]' | head -30 | sed 's/^/     /'
fi
for f in scripts/serve.py docker/Dockerfile docker/requirements.txt; do
  git diff --quiet "$BASE" "$UP" -- $f || note "$f changed upstream (image rebuild needed: local/build.sh)"
done
if ! git diff --quiet "$BASE" "$UP" -- manifests/; then
  note "weights pins changed: run local/download.sh (~100 GB) before starting:"
  git diff "$BASE" "$UP" -- manifests/ | grep -E '^[+-] *"(repo|revision|branch)"' | sed 's/^/     /'
fi

echo "== 3. our TensorFold fork builds on upstream's TensorFold tree"
if [ -n "$NEWREV" ]; then
  source local/config.sh
  SRC="$(local/tensorfold-src.sh)" || { echo "  could not fetch the fork (network?)"; exit 1; }
  if git -C "$SRC" merge-base --is-ancestor "$NEWREV" "$TENSORFOLD_FORK_REV" 2>/dev/null; then
    echo "  ok: fork ${TENSORFOLD_FORK_REV:0:8} is based on TensorFold ${NEWREV:0:8}"
  else fail "upstream moves TensorFold to $NEWREV: rebase the fork onto it, then pin it in local/pins.env"; fi
  # upstream's own TensorFold patches: the fork carries vision-prefix-cache.patch as a commit
  for p in $(grep -o 'git -C /opt/TensorFold apply /opt/recipe/patches/[^ ]*' "$T/docker/Dockerfile" | awk '{print $NF}' | sed 's#/opt/recipe/##' | sort -u); do
    [ "$p" = patches/vision-prefix-cache.patch ] && git diff --quiet "$BASE" "$UP" -- "$p" && continue
    fail "upstream's $p is new or changed: port it into the fork"
  done
fi

echo
if [ ${#bad[@]} -gt 0 ]; then
  echo "== RESULT: INCOMPATIBLE (${#bad[@]} problem(s)) — do not update. Fix per LOCAL_CHANGES.md, then re-run."
  exit 3
fi
echo "== RESULT: COMPATIBLE${warn:+ with ${#warn[@]} note(s)} — local/update.sh can apply it."
exit 0
