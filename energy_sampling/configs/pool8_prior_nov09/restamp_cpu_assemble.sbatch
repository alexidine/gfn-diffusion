#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=restampA
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# CPU restamp, step 2 of 2 (restamp_cpu_parts.sbatch's header): checks that the 24 parts cover every row under one
# energy and writes <stem>_switched.pt (data_processing/restamp_prior.py --assemble). Same IN / FORCES / SPACE_GROUPS
# as step 1. Loads the whole source (the trainer's ~2.9 GB + 2x the file) plus the parts' columns.
module purge

PARTS=24
IN=${IN:?set IN, the prior file to restamp}
OUT=${IN%.pt}_switched.pt
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
SETS="--set energy_config.crystal_coordinates=fd"
if [ -n "${SPACE_GROUPS}" ]; then SETS="${SETS} --set space_groups=${SPACE_GROUPS}"; fi
if [ "${FORCES:-0}" = 1 ]; then SETS="${SETS} --forces"; fi
if [ -e "${OUT}" ]; then echo "FATAL: ${OUT} exists" >&2; exit 1; fi

# singularity directly, not srun: on some nodes (partition 'all', e.g. ga017) Slurm's environment carries
# SLURM_CPUS_PER_TASK != SLURM_TRES_PER_TASK and srun refuses to start (restampP 23723658_6..23, 2026-11-19);
# a one-task job needs no job step
singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -u data_processing/restamp_prior.py ${IN} ${OUT} --config configs/mk_dev.yaml ${SETS} --device cpu \
            --assemble ${PARTS}
        ls -la ${OUT}
    "
