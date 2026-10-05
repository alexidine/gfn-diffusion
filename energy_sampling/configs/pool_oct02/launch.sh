#!/bin/bash
# pool_oct02: submit stages for named systems, chained by job dependencies. Run from anywhere on the cluster.
#
#   bash launch.sh prep   <system> ...  premerge (where the pool was not merged on the dev box) -> export (anchors.pt)
#   bash launch.sh flood  <system> ...  flood -> assemble          (needs the system's anchors.pt)
#   bash launch.sh all    <system> ...  prep, then flood -> assemble once the export has succeeded
#   bash launch.sh finish <system> ...  export -> flood -> assemble (the premerge already ran: <system>/pool/registry.pt)
#   bash launch.sh assemble <system> ... assemble alone, from the anchors.pt and flood shards already on disk
#   bash launch.sh polish <system> ...  the converged-minima census: polish -> merge (not part of the prior build)
#
# Systems: the names in SYSTEMS.tsv (mipcas_elj mipcas_uma nehzor_elj nehzor_uma). POLISH_TASKS=<array spec> limits a
# polish to those INDEX.tsv tasks. Every stage resumes or repeats safely; a task that is still running must not be
# resubmitted.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "${HERE}/joblogs"
STAGE=${1:?usage: launch.sh prep|flood|all|finish|assemble|polish <system> ...}
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
EXPORT=""
case "${STAGE}" in
    prep|all)
        PRE=$(sub --array="${SYS}" "${HERE}/submit_premerge.sbatch")
        EXPORT=$(sub --array="${SYS}" --dependency=afterok:${PRE} "${HERE}/submit_export.sbatch")
        echo "premerge ${PRE} -> export ${EXPORT}" ;;
    finish)
        EXPORT=$(sub --array="${SYS}" "${HERE}/submit_export.sbatch")
        echo "export ${EXPORT}" ;;
    polish)
        POL=$(sub --array="${POLISH_TASKS:-$(ids "${HERE}/INDEX.tsv" "$@")}" "${HERE}/submit_polish.sbatch")
        MERGE=$(sub --array="${SYS}" --dependency=afterany:${POL} "${HERE}/submit_merge.sbatch")
        echo "polish ${POL} -> merge ${MERGE}" ;;
    assemble)
        AS=$(sub --array="${SYS}" "${HERE}/submit_assemble.sbatch")
        echo "assemble ${AS}" ;;
    flood) ;;
    *) echo "unknown stage ${STAGE}" >&2; exit 1 ;;
esac
if [ "${STAGE}" = "flood" ] || [ "${STAGE}" = "all" ] || [ "${STAGE}" = "finish" ]; then
    DEP=""
    if [ -n "${EXPORT}" ]; then DEP="--dependency=afterok:${EXPORT}"; fi
    FL=$(sub --array="$(ids "${HERE}/FLOOD.tsv" "$@")" ${DEP} "${HERE}/submit_flood.sbatch")
    AS=$(sub --array="${SYS}" --dependency=afterany:${FL} "${HERE}/submit_assemble.sbatch")
    echo "flood ${FL} -> assemble ${AS}"
fi
echo "logs: ${HERE}/joblogs"
