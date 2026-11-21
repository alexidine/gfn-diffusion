#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=kickshard
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-99

# pool8_kick_nov20, step 2 of 3 (README.md): task i builds training band sets i*SETS+1 .. (i+1)*SETS of train_list.txt
# into SHARDS/shard_<i>.pt: K 4 metric children at 10^U(log_lo, log_hi) Angstrom (setup.sbatch), FD chart, P-1, eLJ
# scale 0.36358, the switched eLJ (mk_dev), forces, the current prior's held-out descent groups (--heldout_units),
# seed 1000 + i. The recipe of configs/pool8sg_prior_nov17/shard_template.sbatch. A task whose shard exists does
# nothing, so resubmitting finishes what is left. CPU only.
module purge

K=4
SETS=29
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_kick_nov20/priors
SHARDS=${OUT}/shards_k${K}
TSF=0.36358
I=${SLURM_ARRAY_TASK_ID:?array task id}
TAG=$(printf %03d ${I})
SHARD=${SHARDS}/shard_${TAG}.pt
if [ -s ${SHARD} ]; then echo "shard ${TAG}: exists"; exit 0; fi
LOGHI=$(cat ${SHARDS}/log_hi.txt 2>/dev/null)
LOGLO=$(cat ${SHARDS}/log_lo.txt 2>/dev/null)
for V in "${LOGHI}" "${LOGLO}"; do
    if ! [[ "${V}" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then echo "FATAL: log_lo '${LOGLO}' log_hi '${LOGHI}' (run setup.sbatch)" >&2; exit 1; fi
done
if [ ! -s ${OUT}/heldout_units.pt ]; then echo "FATAL: ${OUT}/heldout_units.pt is missing (run setup.sbatch)" >&2; exit 1; fi
sed -n "$(( I * SETS + 1 )),$(( (I + 1) * SETS ))p" ${OUT}/train_list.txt > ${SHARDS}/list_${TAG}.txt
NSETS=$(wc -l < ${SHARDS}/list_${TAG}.txt)
echo "shard ${TAG}: ${NSETS} band sets, K ${K}, 10^U(${LOGLO}, ${LOGHI}) A, seed $(( 1000 + I )); gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
if [ "${NSETS}" = "0" ]; then echo "shard ${TAG}: no band sets (the list holds $(wc -l < ${OUT}/train_list.txt))"; exit 0; fi

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
        # not ${WORKDIR}: data_processing/utils.py would shadow energy_sampling/utils.py
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --thermal_scaling_factor ${TSF} \
            --device cpu --basins ${SHARDS}/list_${TAG}.txt --k ${K} --child_kernel metric \
            --log_lo ${LOGLO} --log_hi ${LOGHI} --heldout_units ${OUT}/heldout_units.pt --forces \
            --seed $(( 1000 + I )) --out ${SHARD}.tmp.pt
        mv ${SHARD}.tmp.pt ${SHARD}
    "
