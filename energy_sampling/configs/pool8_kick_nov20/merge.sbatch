#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=kickmerge
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8_kick_nov20, step 3 of 3 (README.md): every shard -> OUT/PRIOR (data_processing/merge_noised_priors.py; refuses
# shards that differ in kernel, identity or eLJ scale, or that share a molecule), then the three checks against the
# current prior (data_processing/prior_kick_compare.py compare, to OUT/compare.txt): anchor sets per split, child
# excess over parent, rows kept at load at 10 and 6 kT. The job FAILS when the anchor sets differ. CPU only.
module purge

K=4
NSHARD=100
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
CURRENT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/priors/pool8_p1_train_k4_metric_switched.pt
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_kick_nov20/priors
SHARDS=${OUT}/shards_k${K}
PRIOR=pool8_p1_train_k4_metric_loglo-0.675.pt

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
if [ -s ${OUT}/${PRIOR} ]; then
    for F in $(cat ${SHARDS}/shards.txt); do
        if [ ${F} -nt ${OUT}/${PRIOR} ]; then echo "FATAL: ${OUT}/${PRIOR} is older than ${F}; delete it and resubmit" >&2; exit 1; fi
    done
fi
echo "merging $(wc -l < ${SHARDS}/shards.txt) shards over ${NSETS} band sets; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
( while sleep 120; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used " $3 " GB, available " $7 " GB"}')"; done ) > ${SHARDS}/merge_${SLURM_JOB_ID}_ram.txt &
RAM_PID=$!
trap "kill ${RAM_PID} 2>/dev/null" EXIT TERM

# singularity directly, not srun: on some nodes (partition 'all') srun refuses to start (restampP 23723658, 2026-11-19)
singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e -o pipefail
        source /ext3/env.sh
        # not ${WORKDIR}: data_processing/utils.py would shadow energy_sampling/utils.py
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        [ -s ${OUT}/${PRIOR} ] || python -u data_processing/merge_noised_priors.py ${OUT}/${PRIOR} --list ${SHARDS}/shards.txt
        python -u data_processing/prior_kick_compare.py compare ${CURRENT} ${OUT}/${PRIOR} --window_kT 10 6 \
            | tee ${OUT}/compare.txt
        ls -la ${OUT}
    "
