#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sg@SG@shard
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-@LASTSHARD@

# pool8sg_prior_nov17, space group @SG@, step 2 of 3 (README.md): task i builds training band sets i*SETS+1 ..
# (i+1)*SETS of train_list.txt (whole molecules: one band set = one search task's molecules) into SHARDS/shard_<i>.pt:
# K @K@ metric children, FD chart, space_groups [@SG@], eLJ scale @TSF@, the shared log_hi, seed 1000 + i. The recipe of
# configs/pool8_prior_nov09/shard_pool8_prior.sbatch with the space group set. A task whose shard exists does nothing,
# so resubmitting finishes what is left. CPU only. Rendered by make.py from shard_template.sbatch.
module purge

K=@K@
SETS=@SETS@
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
OUT=@OUT@
SHARDS=${OUT}/shards_k${K}
TSF=@TSF@
I=${SLURM_ARRAY_TASK_ID:?array task id}
TAG=$(printf %03d ${I})
SHARD=${SHARDS}/shard_${TAG}.pt
if [ -s ${SHARD} ]; then echo "shard ${TAG}: exists"; exit 0; fi
LOGHI=$(cat ${SHARDS}/log_hi.txt 2>/dev/null)
if ! [[ "${LOGHI}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then echo "FATAL: ${SHARDS}/log_hi.txt is '${LOGHI}' (run setup_sg@SG@.sbatch)" >&2; exit 1; fi
sed -n "$(( I * SETS + 1 )),$(( (I + 1) * SETS ))p" ${OUT}/train_list.txt > ${SHARDS}/list_${TAG}.txt
NSETS=$(wc -l < ${SHARDS}/list_${TAG}.txt)
echo "shard ${TAG}: ${NSETS} band sets, sg @SG@, K ${K}, log_hi ${LOGHI}, seed $(( 1000 + I )); gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
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
        # not ${WORKDIR}: data_processing/utils.py would shadow energy_sampling/utils.py
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --set 'space_groups=[@SG@]' --thermal_scaling_factor ${TSF} \
            --device cpu --basins ${SHARDS}/list_${TAG}.txt --k ${K} --child_kernel metric --log_hi ${LOGHI} --forces \
            --seed $(( 1000 + I )) --out ${SHARD}.tmp.pt
        mv ${SHARD}.tmp.pt ${SHARD}
    "
