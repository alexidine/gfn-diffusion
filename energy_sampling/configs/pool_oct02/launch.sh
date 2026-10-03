#!/bin/bash
# pool_oct02: submit stages for named systems, chained by job dependencies. Run from anywhere on the cluster.
#
#   bash launch.sh prep  <system> ...   premerge (where the pool was not merged locally) -> polish -> merge
#   bash launch.sh flood <system> ...   flood -> assemble          (needs the system's anchors_polished.pt)
#   bash launch.sh all   <system> ...   prep, then flood -> assemble once the merge has succeeded
#   bash launch.sh merge <system> ...   the merge alone (safe to repeat)
#
# Systems: the names in SYSTEMS.tsv (mipcas_elj mipcas_uma nehzor_elj nehzor_uma). POLISH_TASKS=<array spec> limits the
# polish to those INDEX.tsv tasks (e.g. POLISH_TASKS=1 for the one MIPCAS eLJ block that failed on 2026-10-02).
# Every stage resumes or repeats safely; a polish task that is still running must not be resubmitted.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${HERE}/joblogs"
STAGE=${1:?usage: launch.sh prep|flood|all|merge <system> ...}
shift
if [ $# -eq 0 ]; then echo "name at least one system" >&2; exit 1; fi

ids() {  # comma-separated task ids of the given systems in a TSV (column 2 = system name)
    local file=$1; shift
    local out="" name
    for name in "$@"; do
        local got
        got=$(awk -F'\t' -v s="${name}" 'NR > 1 && $2 == s {printf "%s%s", sep, $1; sep=","}' "${file}")
        if [ -z "${got}" ]; then echo "unknown system ${name} in ${file}" >&2; exit 1; fi
        out="${out:+${out},}${got}"
    done
    echo "${out}"
}
sub() { local j; j=$(sbatch --parsable "$@"); echo "${j%%;*}"; }

SYS=$(ids "${HERE}/SYSTEMS.tsv" "$@")
MERGE=""
if [ "${STAGE}" = "prep" ] || [ "${STAGE}" = "all" ]; then
    PRE=$(sub --array="${SYS}" "${HERE}/submit_premerge.sbatch")
    POL=$(sub --array="${POLISH_TASKS:-$(ids "${HERE}/INDEX.tsv" "$@")}" --dependency=afterok:${PRE} "${HERE}/submit_polish.sbatch")
    MERGE=$(sub --array="${SYS}" --dependency=afterany:${POL} "${HERE}/submit_merge.sbatch")
    echo "premerge ${PRE} -> polish ${POL} -> merge ${MERGE}"
elif [ "${STAGE}" = "merge" ]; then
    MERGE=$(sub --array="${SYS}" "${HERE}/submit_merge.sbatch")
    echo "merge ${MERGE}"
fi
if [ "${STAGE}" = "flood" ] || [ "${STAGE}" = "all" ]; then
    DEP=""
    if [ -n "${MERGE}" ]; then DEP="--dependency=afterok:${MERGE}"; fi
    FL=$(sub --array="$(ids "${HERE}/FLOOD.tsv" "$@")" ${DEP} "${HERE}/submit_flood.sbatch")
    AS=$(sub --array="${SYS}" --dependency=afterany:${FL} "${HERE}/submit_assemble.sbatch")
    echo "flood ${FL} -> assemble ${AS}"
fi
echo "logs: ${HERE}/joblogs"
