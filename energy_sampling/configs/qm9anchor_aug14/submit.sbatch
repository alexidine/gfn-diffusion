#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-5
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=qm9anchor_aug14

# qm9anchor_aug14 :: 3 hyper bandwidths x 2 loss-beta settings.
# Arms are NAMED, not numbered, so the array index selects a row of INDEX.tsv rather than
# a filename -- renaming arms to integers would lose which arm is which in wandb.
module purge

# ---- paths ----
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/qm9anchor_aug14

# ---- arm selection: line N+2 of INDEX.tsv (line 1 is the header) ----
ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then
    echo "no arm at INDEX.tsv row for array id ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"

# ---- run ----
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u train.py --config ${CONFIG}
    "
