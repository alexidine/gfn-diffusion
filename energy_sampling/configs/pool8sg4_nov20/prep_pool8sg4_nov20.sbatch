#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sg4merge
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# pool8sg4_nov20 (README.md), CPU only: the one training prior a four-group run reads. Checks that the merge will accept
# the four K=4 priors (check_priors.py, memory-mapped, seconds), then concatenates them with
# data_processing/merge_noised_priors.py, P-1 FIRST (the trainer's dead-row probe reads the first rows against
# space_groups[0] = 2). Refuses an existing output (the merge never overwrites). RAM: four ~10M-row files at once; the
# one-group merges were given 96 GB, so this asks 256 GB and logs `free -g` every 2 minutes beside the output.
module purge

DATA=/scratch/mk8347/data/crystal_datasets/conditional
INPUTS=(
    ${DATA}/pool8_nov09/priors/pool8_p1_train_k4_metric_switched.pt
    ${DATA}/pool8sg_nov15/priors/sg14/pool8_sg14_train_k4_metric.pt
    ${DATA}/pool8sg_nov15/priors/sg19/pool8_sg19_train_k4_metric.pt
    ${DATA}/pool8sg_nov15/priors/sg61/pool8_sg61_train_k4_metric.pt
)
OUTDIR=${DATA}/pool8sg4_nov20/priors
OUT=${OUTDIR}/pool8_4sg_train_k4_metric.pt
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling

for f in "${INPUTS[@]}"; do [ -s "${f}" ] || { echo "FATAL: ${f} is missing or empty" >&2; exit 1; }; done
[ -e "${OUT}" ] && { echo "FATAL: ${OUT} exists (the merge never overwrites; delete it to rebuild)" >&2; exit 1; }
mkdir -p ${OUTDIR}
echo "merge -> ${OUT}; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"
( while sleep 120; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used " $3 " GB, available " $7 " GB"}')"; done ) > ${OUTDIR}/merge_${SLURM_JOB_ID}_ram.txt &
RAM_PID=$!
trap "kill ${RAM_PID} 2>/dev/null" EXIT TERM

# singularity directly, not srun (srun refuses on partition-'all' nodes; configs/pool8sg_prior_nov17 379dab02)
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
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        python -u configs/pool8sg4_nov20/check_priors.py ${INPUTS[*]}
        python -u data_processing/merge_noised_priors.py ${OUT} ${INPUTS[*]}
        ls -la ${OUTDIR}
    "
