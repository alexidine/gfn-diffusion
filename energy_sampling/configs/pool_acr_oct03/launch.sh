#!/bin/bash
# pool_acr_oct03: the acridine (P2_1/c, Z'=1, MACE acr_newmodel) leg of the pooled prior build, chained by job
# dependencies. Run from anywhere on the cluster.
#
#   bash launch.sh all     premerge -> export -> walk (4 shards)
#   bash launch.sh prep    premerge -> export                   (writes <SYS>/anchors.pt)
#   bash launch.sh flood   the walk only                        (needs anchors.pt; resubmitting resumes every shard)
#
# SYS = /scratch/mk8347/data/crystal_datasets/pooled_oct02/acridine_mace. The assembly into the shipped prior file
# (C2 relabelling, de-dupe, normaliser images) is a later stage, not submitted here.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${HERE}/joblogs"
STAGE=${1:?usage: launch.sh all|prep|flood}
sub() { local j; j=$(sbatch --parsable "$@"); echo "${j%%;*}"; }
live=$(squeue -h -u "${USER}" -n pa_premerge,pa_export,pa_flood -o %i) || { echo "squeue failed" >&2; exit 1; }
if [ -n "${live}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "jobs still queued or running ($(echo ${live} | tr '\n' ' ')); not resubmitting (FORCE=1)" >&2; exit 1
fi
DEP=""
case "${STAGE}" in
    prep|all)
        PRE=$(sub "${HERE}/submit_premerge.sbatch")
        EXP=$(sub --dependency=afterok:${PRE} "${HERE}/submit_export.sbatch")
        DEP="--dependency=afterok:${EXP}"
        echo "premerge ${PRE} -> export ${EXP}" ;;
    flood) ;;
    *) echo "unknown stage ${STAGE}" >&2; exit 1 ;;
esac
if [ "${STAGE}" = "flood" ] || [ "${STAGE}" = "all" ]; then
    FL=$(sub ${DEP} "${HERE}/submit_flood.sbatch")
    echo "walk ${FL} (4 shards)"
fi
echo "logs: ${HERE}/joblogs; premerge progress: /scratch/mk8347/data/crystal_datasets/pooled_oct02/acridine_mace/pool/curate.log"
