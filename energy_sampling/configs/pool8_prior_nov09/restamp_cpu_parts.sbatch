#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=restampP
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-23

# CPU restamp of a pre-noised eLJ prior under the switched trainer eLJ, step 1 of 2 (data_processing/restamp_prior.py
# --part): task k re-scores rows k*n/24 .. (k+1)*n/24 and writes only the new columns to <stem>_switched.pt.parts/.
# Step 2, restamp_cpu_assemble.sbatch (afterok), writes <stem>_switched.pt. The source is not touched.
#   IN=/scratch/.../pool8_p1_train_k4_metric.pt FORCES=1 sbatch configs/pool8_prior_nov09/restamp_cpu_parts.sbatch
# Optional: SPACE_GROUPS='[14]' for a file built in another group; FORCES=1 adds the per-row force column. A task whose
# part exists does nothing, so resubmitting finishes what is left. Local measurement (2026-11-19, the 2.04M-row K 0
# file): ~1.6 ms per row on CPU, 9 GB peak per task (the source memory-mapped; its per-row Python lists are not).
module purge

PARTS=24
IN=${IN:?set IN, the prior file to restamp}
OUT=${IN%.pt}_switched.pt
K=${SLURM_ARRAY_TASK_ID:?array task id}
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
SETS="--set energy_config.crystal_coordinates=fd"
if [ -n "${SPACE_GROUPS}" ]; then SETS="${SETS} --set space_groups=${SPACE_GROUPS}"; fi
if [ "${FORCES:-0}" = 1 ]; then SETS="${SETS} --forces"; fi
if [ ! -s "${IN}" ]; then echo "FATAL: ${IN} missing" >&2; exit 1; fi
echo "restamp part ${K}/${PARTS} of ${IN}; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"

srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e
        source /ext3/env.sh
        # not ${WORKDIR}: data_processing/utils.py would shadow energy_sampling/utils.py (build_pool8_prior.sbatch)
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -u data_processing/restamp_prior.py ${IN} ${OUT} --config configs/mk_dev.yaml ${SETS} --device cpu \
            --part ${K}/${PARTS}
    "
