#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sg@SG@setup
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8sg_prior_nov17, space group @SG@, step 1 of 3 (README.md): the band-set lists and the kick radius. No calibration:
# every group uses the log_hi of the pool8 P-1 K=4 prior (read from its noised_build record, memory-mapped), because
# merge_noised_priors.py refuses files whose log_hi differ and the groups will be merged for multi-group training.
# CPU only. Rendered by make.py from setup_template.sbatch.
module purge
set -e

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BAND=@BAND@
OUT=@OUT@
SHARDS=${OUT}/shards_k@K@
P1_PRIOR=@P1_PRIOR@
mkdir -p ${SHARDS}

NDONE=$(ls ${BAND}/done 2>/dev/null | wc -l)
if [ "${NDONE}" != "@NTASKS@" ]; then echo "FATAL: ${BAND}/done holds ${NDONE} of @NTASKS@ tasks" >&2; exit 1; fi
# the lists are replaced only when their content changes, so a resubmitted chain does not make the merge step's
# outputs look older than their inputs
ls -d ${BAND}/train/pool8sg@SG@_c*_band > ${OUT}/train_list.txt.new
ls -d ${BAND}/heldout_random/pool8sg@SG@_c*_band ${BAND}/heldout_scaffold/pool8sg@SG@_c*_band > ${OUT}/heldout_list.txt.new
for L in train_list heldout_list; do
    if cmp -s ${OUT}/${L}.txt.new ${OUT}/${L}.txt; then rm ${OUT}/${L}.txt.new; else mv ${OUT}/${L}.txt.new ${OUT}/${L}.txt; fi
done
echo "band sets: $(wc -l < ${OUT}/train_list.txt) train, $(wc -l < ${OUT}/heldout_list.txt) held out"
echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse --short=12 HEAD)"
if [ -s ${SHARDS}/log_hi.txt ]; then echo "log_hi $(cat ${SHARDS}/log_hi.txt) (kept)"; exit 0; fi
if [ ! -s ${P1_PRIOR} ]; then echo "FATAL: ${P1_PRIOR} (the P-1 prior whose log_hi every group shares) is missing" >&2; exit 1; fi

# singularity directly, not srun: on some nodes (partition 'all', e.g. ga017) Slurm's environment carries
# SLURM_CPUS_PER_TASK != SLURM_TRES_PER_TASK and srun refuses to start (restampP 23723658_6..23, 2026-11-19);
# a one-task job needs no job step
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
        python -c \"import sys, torch; print(torch.load(sys.argv[1], weights_only=False, map_location='cpu', mmap=True)['noised_build']['log_hi'])\" \
            ${P1_PRIOR} | tail -n 1 > ${SHARDS}/log_hi.txt.tmp
        mv ${SHARDS}/log_hi.txt.tmp ${SHARDS}/log_hi.txt
        echo log_hi \$(cat ${SHARDS}/log_hi.txt) from ${P1_PRIOR}
    "
