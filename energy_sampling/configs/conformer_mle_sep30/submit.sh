#!/bin/bash
# conformer_mle_sep30: submit one rung's build (CPU) and its training (GPU, afterok on the build), or with
# `one` ONE job that builds the rung when it is missing and then trains (no dependency).
#     bash configs/conformer_mle_sep30/submit.sh <rung> [one]      rungs: pilot r20k
# Later legs: sbatch configs/conformer_mle_sep30/train_<rung>.sbatch (it resumes the arm's own _running.pt).
# WRITTEN BY make.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${1:-}
MODE=${2:-two}
case " pilot r20k " in *" ${R} "*) ;; *) echo "usage: bash $0 <rung> [one], rung one of: pilot r20k" >&2; exit 2;; esac
case "${MODE}" in one|two) ;; *) echo "usage: bash $0 <rung> [one]" >&2; exit 2;; esac
if [ "${R}" = "pilot" ] && ls /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/*mle30_pilot_*_running.pt >/dev/null 2>&1; then
    echo "the pilot arm already has checkpoints, and a new pilot would RESUME them; to re-pilot:" >&2
    echo "    rm /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/*mle30_pilot_* ; rm -r /scratch/mk8347/data/conformer_datasets/rungs_sep30/pilot" >&2
    exit 1
fi
if [ "${MODE}" = "one" ]; then
    T=$(sbatch --parsable configs/conformer_mle_sep30/run_${R}.sbatch); T=${T%%;*}
    echo "rung ${R}: ONE job ${T} (builds the rung if it is missing, then trains)"
    exit 0
fi
B=$(sbatch --parsable configs/conformer_mle_sep30/build_${R}.sbatch); B=${B%%;*}
# --kill-on-invalid-dep: a failed build cancels the training job instead of leaving it pending forever
T=$(sbatch --parsable --dependency=afterok:${B} --kill-on-invalid-dep=yes configs/conformer_mle_sep30/train_${R}.sbatch); T=${T%%;*}
echo "rung ${R}: build ${B} -> train ${T} (afterok:${B})"
