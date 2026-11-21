#!/bin/bash
# Chain the pool8_kick_nov20 build: setup -> shard array -> merge + checks. Run from anywhere on the cluster.
set -e
cd "$(dirname "$0")"
S=$(sbatch --parsable setup.sbatch)
A=$(sbatch --parsable --dependency=afterok:${S} shard.sbatch)
M=$(sbatch --parsable --dependency=afterok:${A} merge.sbatch)
echo "pool8_kick_nov20: setup ${S} -> shards ${A} -> merge + checks ${M}"
