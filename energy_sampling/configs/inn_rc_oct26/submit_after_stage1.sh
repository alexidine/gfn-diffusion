#!/bin/bash
# Submits inn_rc_oct26's two jobs, each to start after its inn_rc_oct25 stage-1 job (job name innrc25) ends OK
# (--dependency=afterok, --kill-on-invalid-dep=yes: a stage-1 job that fails cancels its stage-2 job). A stage-1 job
# that has already left the queue gives no dependency: that stage-2 job is submitted to start when scheduled, and its
# own check refuses to train if the stage-1 best.pt is missing. Which stage-1 job is which: its .info file, which
# lists its arms file (only rcall_big's mentions rcall_big).
cd "$(dirname "$0")"
INFO=../inn_rc_oct25/joblogs
A=""; B=""
for j in $(squeue -h -u "$USER" -n innrc25 -o %i); do
    if [ ! -f "${INFO}/job_${j}.info" ]; then
        echo "FATAL: stage-1 job ${j} has no ${INFO}/job_${j}.info yet (not started?): cannot tell its arm" >&2; exit 1
    fi
    if grep -q rcall_big "${INFO}/job_${j}.info"; then B=${j}; else A=${j}; fi
done
echo "stage-1 jobs still in the queue: rcall ${A:-none}, rcall_big ${B:-none}"
DEP_A=""; [ -n "${A}" ] && DEP_A="--dependency=afterok:${A} --kill-on-invalid-dep=yes"
DEP_B=""; [ -n "${B}" ] && DEP_B="--dependency=afterok:${B} --kill-on-invalid-dep=yes"
sbatch ${DEP_A} submit_inn_rc_oct26.sbatch
sbatch ${DEP_B} --export=ALL,ARMS_TSV=arms_big.tsv submit_inn_rc_oct26.sbatch
