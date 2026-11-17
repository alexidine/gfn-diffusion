#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8merge
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8 sharded prior build, step 3 of 3 (shard_pool8_prior.sbatch header): every shard -> the files
# configs/pool8_mle_nov10 reads, in OUT:
#   pool8_p1_train_k4_metric.pt    data_processing/merge_noised_priors.py over SHARDS/shard_000..099.pt (refuses shards
#                                  that differ in kernel, identity or eLJ scale, or that share a molecule)
#   pool8_p1_heldout_anchors.pt    the held-out molecules' anchors, K 0 (kept if it exists: an earlier build wrote it)
#   pool8_p1_train_conditions.pt   molecule_split_files.py conditions
#   pool8_p1_test_w5.pt            molecule_split_files.py heldout, W 5, 8 per molecule, at most 500 molecules
# Refuses unless every shard of the array exists. Skips a file that exists. CPU only.
module purge

K=4
NSHARD=100
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/priors
SHARDS=${OUT}/shards_k4
TSF=0.36358
PRIOR=pool8_p1_train_k${K}_metric.pt
HELDOUT=pool8_p1_heldout_anchors.pt

: > ${SHARDS}/shards.txt
for I in $(seq 0 $(( NSHARD - 1 ))); do
    F=${SHARDS}/shard_$(printf %03d ${I}).pt
    if [ -s ${SHARDS}/list_$(printf %03d ${I}).txt ] && [ ! -s ${F} ]; then echo "FATAL: ${F} is missing" >&2; exit 1; fi
    [ -s ${F} ] && echo ${F} >> ${SHARDS}/shards.txt
done
NSETS=$(cat ${SHARDS}/list_*.txt | wc -l)
if [ "${NSETS}" != "$(wc -l < ${OUT}/train_list.txt)" ]; then
    echo "FATAL: the shard lists hold ${NSETS} band sets, train_list.txt $(wc -l < ${OUT}/train_list.txt)" >&2; exit 1
fi
echo "merging $(wc -l < ${SHARDS}/shards.txt) shards over ${NSETS} band sets; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
( while sleep 120; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used " $3 " GB, available " $7 " GB"}')"; done ) > ${SHARDS}/merge_${SLURM_JOB_ID}_ram.txt &
RAM_PID=$!
trap "kill ${RAM_PID} 2>/dev/null" EXIT TERM

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
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        [ -s ${OUT}/${PRIOR} ] || python -u data_processing/merge_noised_priors.py ${OUT}/${PRIOR} --list ${SHARDS}/shards.txt
        [ -s ${OUT}/${HELDOUT} ] || python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --thermal_scaling_factor ${TSF} --device cpu \
            --basins ${OUT}/heldout_list.txt --k 0 --heldout_frac 0 --out ${OUT}/${HELDOUT}
        [ -s ${OUT}/pool8_p1_train_conditions.pt ] || python -u data_processing/molecule_split_files.py conditions \
            ${OUT}/${PRIOR} ${OUT}/pool8_p1_train_conditions.pt
        [ -s ${OUT}/pool8_p1_test_w5.pt ] || python -u data_processing/molecule_split_files.py heldout \
            ${OUT}/${HELDOUT} ${OUT}/pool8_p1_test 5 --per_molecule 8 --max_molecules 500
        ls -la ${OUT}
    "
