#!/bin/bash
# Submits inn_cap_oct26's four stage-1 jobs, one A100 each (arms_s, arms_m, arms_l1, arms_l2), then the two stage-2 jobs,
# each to start once its stage-1 job has ended (--dependency=afterany): s2a (f1_s forward / pooled) after job s, s2b (f1_l
# forward) after job l1. A stage-2 job refuses to train unless its own stage-1 arm finished (final.pt beside its best.pt),
# so a sibling arm's failure in the same stage-1 job does not cancel it.
set -e
cd "$(dirname "$0")"
for f in arms_s.tsv arms_m.tsv arms_l1.tsv arms_l2.tsv arms_s2a.tsv arms_s2b.tsv common.args common_s2.args submit_inn_cap26.sbatch; do
    [ -f "$f" ] || { echo "FATAL: missing $f -- git pull gfn-diffusion" >&2; exit 1; }
done
declare -A JOB
for g in s m l1 l2; do
    id=$(sbatch --parsable --export=ALL,ARMS_TSV=arms_${g}.tsv,COMMON=common.args submit_inn_cap26.sbatch)
    JOB[$g]=${id%%;*}
    echo "stage 1 ${g}: job ${JOB[$g]}"
done
for pair in s2a:s s2b:l1; do
    j=${pair%%:*}; d=${pair##*:}
    id=$(sbatch --parsable --dependency=afterany:${JOB[$d]} \
        --export=ALL,ARMS_TSV=arms_${j}.tsv,COMMON=common_s2.args submit_inn_cap26.sbatch)
    echo "stage 2 ${j}: job ${id%%;*} (after ${JOB[$d]})"
done
