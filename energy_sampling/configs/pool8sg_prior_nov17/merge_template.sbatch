#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sg@SG@merge
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8sg_prior_nov17, space group @SG@, step 3 of 3 (README.md): every shard -> the four files a training run reads, in OUT:
#   @PRIOR@        data_processing/merge_noised_priors.py over SHARDS/shard_*.pt (refuses shards that differ in kernel,
#                  identity or eLJ scale, or that share a molecule)
#   @HELDOUT@      the held-out molecules' anchors, K 0
#   @CONDITIONS@   molecule_split_files.py conditions
#   @TEST@         molecule_split_files.py heldout, W 5, 8 per molecule, at most 500 molecules
# Refuses unless every listed shard exists. An output that exists is kept only if it is newer than every input it is
# built from; an older one is a leftover of an earlier build and stops the job (delete it to rebuild). CPU only.
# Rendered by make.py from merge_template.sbatch.
module purge

K=@K@
NSHARD=@NSHARD@
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
OUT=@OUT@
SHARDS=${OUT}/shards_k${K}
TSF=@TSF@
PRIOR=@PRIOR@
HELDOUT=@HELDOUT@
CONDITIONS=@CONDITIONS@
TEST=@TEST@

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
# stale(OUTPUT, INPUT...): 0 = build it (missing), 1 = keep it (newer than every input); exits on an older output
stale() {
    local f=$1; shift
    [ -s "${f}" ] || return 0
    for src in "$@"; do
        if [ "${src}" -nt "${f}" ]; then
            echo "FATAL: ${f} is older than its input ${src}: a leftover of an earlier build. Delete it and resubmit." >&2
            exit 1
        fi
    done
    echo "keeping ${f} (newer than its inputs)"
    return 1
}
BUILD_PRIOR=0; BUILD_HELDOUT=0; BUILD_COND=0; BUILD_TEST=0
stale ${OUT}/${PRIOR} $(cat ${SHARDS}/shards.txt) && BUILD_PRIOR=1
stale ${OUT}/${HELDOUT} ${OUT}/heldout_list.txt && BUILD_HELDOUT=1
stale ${OUT}/${CONDITIONS} ${OUT}/${PRIOR} $(cat ${SHARDS}/shards.txt) && BUILD_COND=1
stale ${OUT}/${TEST} ${OUT}/${HELDOUT} ${OUT}/heldout_list.txt && BUILD_TEST=1
echo "merging $(wc -l < ${SHARDS}/shards.txt) shards over ${NSETS} band sets (build: prior ${BUILD_PRIOR} heldout ${BUILD_HELDOUT} conditions ${BUILD_COND} test ${BUILD_TEST}); gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
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
        # not ${WORKDIR}: data_processing/utils.py would shadow energy_sampling/utils.py
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        if [ ${BUILD_PRIOR} = 1 ]; then
            python -u data_processing/merge_noised_priors.py ${OUT}/${PRIOR} --list ${SHARDS}/shards.txt
        fi
        if [ ${BUILD_HELDOUT} = 1 ]; then
            python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
                --set energy_config.crystal_coordinates=fd --set 'space_groups=[@SG@]' --thermal_scaling_factor ${TSF} \
                --device cpu --basins ${OUT}/heldout_list.txt --k 0 --heldout_frac 0 --out ${OUT}/${HELDOUT}
        fi
        if [ ${BUILD_COND} = 1 ]; then
            python -u data_processing/molecule_split_files.py conditions ${OUT}/${PRIOR} ${OUT}/${CONDITIONS}
        fi
        if [ ${BUILD_TEST} = 1 ]; then
            python -u data_processing/molecule_split_files.py heldout ${OUT}/${HELDOUT} ${OUT}/@TEST_STEM@ 5 \
                --per_molecule 8 --max_molecules 500
        fi
        ls -la ${OUT}
    "
