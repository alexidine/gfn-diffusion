#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=restamp
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# Re-score a pre-noised eLJ prior under the switched trainer eLJ (owner 2026-11-19: lj_envelope 1.0 is the default;
# data_processing/restamp_prior.py) and write it beside the source as <stem>_switched.pt. The source is not touched.
#   IN=/scratch/.../pool8_p1_train_k4_metric.pt sbatch configs/pool8_prior_nov09/restamp_prior.sbatch
# Optional: SPACE_GROUPS='[14]' for a file built in another group (default: mk_dev's); FORCES=1 adds the per-row
# force column (build_noised_prior --forces) to a file without one, as the pool8sg groups carry. One scoring pass over every row
# on the GPU, after a load of the whole file into RAM (the trainer's ~2.9 GB + 2x the file).
module purge

IN=${IN:?set IN, the prior file to restamp}
OUT=${IN%.pt}_switched.pt
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
SETS="--set energy_config.crystal_coordinates=fd"
[ -n "${SPACE_GROUPS}" ] && SETS="${SETS} --set space_groups=${SPACE_GROUPS}"
[ "${FORCES:-0}" = 1 ] && SETS="${SETS} --forces"
[ -s "${IN}" ] || { echo "FATAL: ${IN} missing" >&2; exit 1; }
[ -e "${OUT}" ] && { echo "FATAL: ${OUT} exists" >&2; exit 1; }
echo "restamp ${IN} -> ${OUT}; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"

srun singularity exec --nv \
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
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        python -u data_processing/restamp_prior.py ${IN} ${OUT} --config configs/mk_dev.yaml ${SETS} --device cuda
    "
