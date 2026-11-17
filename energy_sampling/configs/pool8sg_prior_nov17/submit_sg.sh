#!/bin/bash
# Chain one group's prior build: setup -> shard array -> merge. usage: bash submit_sg.sh 14 [19 61 ...]
# Run from this folder on the cluster, after that group's search battery has banded every task.
set -e
cd "$(dirname "$0")"
for G in "$@"; do
    S=$(sbatch --parsable setup_sg${G}.sbatch)
    A=$(sbatch --parsable --dependency=afterok:${S} shard_sg${G}.sbatch)
    M=$(sbatch --parsable --dependency=afterok:${A} merge_sg${G}.sbatch)
    echo "sg${G}: setup ${S} -> shards ${A} -> merge ${M}"
done
