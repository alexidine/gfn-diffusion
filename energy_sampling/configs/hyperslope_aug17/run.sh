#!/usr/bin/env bash
# Run the hyperslope_aug17 ladder, one arm at a time, on the single local GPU.
#
#   bash configs/hyperslope_aug17/run.sh              # all arms, INDEX order
#   bash configs/hyperslope_aug17/run.sh lr8e5 hl28   # named arms only
#
# THE GPU PRE-FLIGHT GUARD IS DELIBERATELY LEFT ARMED. train.py::require_free_gpu
# refuses to start beside another train.py, and two training runs on this card
# have BSOD'd the machine three times (train.py's own __main__ comment,
# 2026-08-11/12). GFN_GPU_GUARD=0 is for CPU-only config loading, NEVER here.
#
# Every arm carries checkpoint_read_only: true, so nothing this script runs can
# overwrite WARM_qm9_mle3k.pt -- which every arm loads and which is not in git.
#
# An arm that aborts (FrozenTrainingState on a non-finite streak) is EXPECTED on
# the hot rungs and is a result, not a failure: the loop continues to the next
# arm and the exit code is recorded in the log.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="C:/Users/mikem/venvs/csd_mxt_gfn/Scripts/python.exe"
export PYTHONPATH="C:\\Users\\mikem\\Projects\\mxt_gfn\\mxtaltools;C:\\Users\\mikem\\Projects\\mxt_gfn\\gfn_diffusion"

mkdir -p "$HERE/logs"
cd "$ROOT" || exit 1

if [ "$#" -gt 0 ]; then
  ARMS=("$@")
else
  ARMS=($(tail -n +2 "$HERE/INDEX.tsv" | cut -f1))
fi

echo "hyperslope_aug17: ${#ARMS[@]} arms -> ${ARMS[*]}"
echo "start $(date '+%H:%M:%S')"

for arm in "${ARMS[@]}"; do
  cfg="configs/hyperslope_aug17/${arm}.yaml"
  log="$HERE/logs/${arm}.log"
  if [ ! -f "$cfg" ]; then
    echo "SKIP $arm -- no such config: $cfg"; continue
  fi
  echo "=== $arm  $(date '+%H:%M:%S')  -> $log"
  "$PY" train.py --config "$cfg" > "$log" 2>&1
  rc=$?
  last=$(grep -aoE "UNRECOVERABLE.*|Traceback" "$log" | head -1)
  echo "    exit=$rc  $(grep -ac 'Non-finite gradient' "$log") non-finite steps  ${last:-ok}"
done

echo "done $(date '+%H:%M:%S')"
