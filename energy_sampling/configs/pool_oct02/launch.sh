#!/bin/bash
# pool_oct02: submit the polish array, then the merge array once every polish task has ended (afterany: a failed slice
# leaves its basins out of the merge rather than blocking it). Run from the gfn-diffusion checkout on the cluster.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${HERE}/joblogs"
P=$(sbatch --parsable "${HERE}/submit_polish.sbatch")
P=${P%%;*}
M=$(sbatch --parsable --dependency=afterany:${P} "${HERE}/submit_merge.sbatch")
echo "polish array ${P}; merge array ${M%%;*} (after ${P}); logs in ${HERE}/joblogs"
