#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=kicksetup
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8_kick_nov20, step 1 of 3 (README.md): the training band-set list of the current P-1 K=4 prior's build, its
# log_hi (kept) and log_lo = log_hi - 0.3, and its held-out descent groups (prior_kick_compare.py units), all read
# from the current prior, which is not written. CPU only. Skips what exists.
module purge
set -e

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
SRC=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09
CURRENT=${SRC}/priors/pool8_p1_train_k4_metric_switched.pt
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_kick_nov20/priors
SHARDS=${OUT}/shards_k4
mkdir -p ${SHARDS}

NDONE=$(ls ${SRC}/band32/done 2>/dev/null | wc -l)
if [ "${NDONE}" != "2899" ]; then echo "FATAL: ${SRC}/band32/done holds ${NDONE} of 2899 tasks" >&2; exit 1; fi
if [ ! -s ${CURRENT} ]; then echo "FATAL: ${CURRENT} (the prior this build copies) is missing" >&2; exit 1; fi
# the current prior's build list, copied; the band folder's listing must still equal it
[ -s ${OUT}/train_list.txt ] || cp ${SRC}/priors/train_list.txt ${OUT}/train_list.txt
if ! cmp -s ${OUT}/train_list.txt <(ls -d ${SRC}/band32/train/pool8_c*_band); then
    echo "FATAL: ${SRC}/band32/train no longer lists what ${SRC}/priors/train_list.txt (the current prior's build) does" >&2
    exit 1
fi
echo "band sets: $(wc -l < ${OUT}/train_list.txt) train"
echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse --short=12 HEAD)"
if [ -s ${SHARDS}/log_hi.txt ] && [ -s ${SHARDS}/log_lo.txt ] && [ -s ${OUT}/heldout_units.pt ]; then
    echo "log_lo $(cat ${SHARDS}/log_lo.txt) log_hi $(cat ${SHARDS}/log_hi.txt), held-out units (kept)"; exit 0
fi

# singularity directly, not srun: on some nodes (partition 'all') srun refuses to start (restampP 23723658, 2026-11-19)
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
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
        python -u data_processing/prior_kick_compare.py radii ${CURRENT} ${SHARDS}/log_lo.txt ${SHARDS}/log_hi.txt \
            --expect_log_hi -0.375 --width 0.3
        python -u data_processing/prior_kick_compare.py units ${CURRENT} ${OUT}/heldout_units.pt
    "
