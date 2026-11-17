#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8calib
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8 sharded prior build, step 1 of 3 (README in this folder's shard_pool8_prior.sbatch header): the list files and
# ONE kick radius for every shard. A shard calibrating its own log_hi would give each shard another kernel, and
# merge_noised_priors.py refuses files whose log_hi differ. Builds the K 0 file SHARDS/calib.pt (no children) over every
# 29th training band set (~100 sets, spread over the list), with the shards' settings, and writes its calibrated log_hi to
# SHARDS/log_hi.txt. CPU only. Skips what exists.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BASE=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09
BAND=${BASE}/band32
OUT=${BASE}/priors
SHARDS=${OUT}/shards_k4
TSF=0.36358
mkdir -p ${SHARDS}

NDONE=$(ls ${BAND}/done 2>/dev/null | wc -l)
if [ "${NDONE}" != "2899" ]; then echo "FATAL: ${BAND}/done holds ${NDONE} of 2899 tasks" >&2; exit 1; fi
ls -d ${BAND}/train/pool8_c*_band > ${OUT}/train_list.txt
ls -d ${BAND}/heldout_random/pool8_c*_band ${BAND}/heldout_scaffold/pool8_c*_band > ${OUT}/heldout_list.txt
awk 'NR % 29 == 1' ${OUT}/train_list.txt > ${SHARDS}/calib_list.txt
echo "band sets: $(wc -l < ${OUT}/train_list.txt) train, $(wc -l < ${OUT}/heldout_list.txt) held out; calibration on $(wc -l < ${SHARDS}/calib_list.txt)"
echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse --short=12 HEAD)"

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
        [ -s ${SHARDS}/calib.pt ] || python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --thermal_scaling_factor ${TSF} --device cpu \
            --basins ${SHARDS}/calib_list.txt --k 0 --child_kernel metric --out ${SHARDS}/calib.pt
        python -c \"import sys, torch; print(torch.load(sys.argv[1], weights_only=False, map_location='cpu')['noised_build']['log_hi'])\" \
            ${SHARDS}/calib.pt | tail -n 1 > ${SHARDS}/log_hi.txt.tmp
        mv ${SHARDS}/log_hi.txt.tmp ${SHARDS}/log_hi.txt
        echo log_hi \$(cat ${SHARDS}/log_hi.txt)
    "
