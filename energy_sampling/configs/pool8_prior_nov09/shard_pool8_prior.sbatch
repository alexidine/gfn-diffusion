#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8shard
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-99

# pool8 SHARDED prior build (owner 2026-11-11: "Yes, shard it"; the single 32-core CPU build never left the queue).
# The same files as build_pool8_prior.sbatch, in three chained jobs:
#   1. calib_pool8_prior.sbatch  the list files and one kick radius (SHARDS/log_hi.txt) for every shard
#   2. this array                task i builds training band sets i*SETS+1 .. (i+1)*SETS of train_list.txt (whole
#                                molecules: one band set = one search task's molecules) into SHARDS/shard_<i>.pt: K 4
#                                metric children, FD chart, eLJ scale 0.36358, the calibrated log_hi, seed 1000 + i
#   3. merge_pool8_prior.sbatch  data_processing/merge_noised_priors.py -> the training prior, then the conditions and
#                                test files, under the names configs/pool8_mle_nov10 reads
# A shard differs from the same molecules in one build only by its random draws (the held-out descent groups, the
# kicks); anchors, energies, kernel and identity are the same (checked locally 2026-11-11 on task 0 split in two).
# A task whose shard exists does nothing, so resubmitting finishes what is left.
# Local rates (2026-11-09/11): anchor load ~2 ms per anchor, scoring ~2.7 ms per row on CPU, builder peak RAM ~3.2 GB
# + ~1.8 KB per row; a shard is ~20k anchors, ~100k rows.
module purge

K=4
SETS=29
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/priors
SHARDS=${OUT}/shards_k4
TSF=0.36358
I=${SLURM_ARRAY_TASK_ID:?array task id}
TAG=$(printf %03d ${I})
SHARD=${SHARDS}/shard_${TAG}.pt
if [ -s ${SHARD} ]; then echo "shard ${TAG}: exists"; exit 0; fi
LOGHI=$(cat ${SHARDS}/log_hi.txt 2>/dev/null)
if ! [[ "${LOGHI}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then echo "FATAL: ${SHARDS}/log_hi.txt is '${LOGHI}' (run calib_pool8_prior.sbatch)" >&2; exit 1; fi
sed -n "$(( I * SETS + 1 )),$(( (I + 1) * SETS ))p" ${OUT}/train_list.txt > ${SHARDS}/list_${TAG}.txt
NSETS=$(wc -l < ${SHARDS}/list_${TAG}.txt)
echo "shard ${TAG}: ${NSETS} band sets, K ${K}, log_hi ${LOGHI}, seed $(( 1000 + I )); gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
if [ "${NSETS}" = "0" ]; then echo "shard ${TAG}: no band sets (the list holds $(wc -l < ${OUT}/train_list.txt))"; exit 0; fi

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
        python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --thermal_scaling_factor ${TSF} --device cpu \
            --basins ${SHARDS}/list_${TAG}.txt --k ${K} --child_kernel metric --log_hi ${LOGHI} --seed $(( 1000 + I )) \
            --out ${SHARD}
    "
