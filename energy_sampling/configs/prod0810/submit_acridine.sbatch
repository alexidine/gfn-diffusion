#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END
#SBATCH --array=4-6
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=prod0810_acridine

# prod0810 :: acridine
# index -> arm mapping is in configs/prod0810/INDEX.tsv
module purge

# ---- paths ----
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling

# ---- config selection ----
CONFIG=${WORKDIR}/configs/prod0810/${SLURM_ARRAY_TASK_ID}.yaml

# ---- run ----
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python train.py --config ${CONFIG}
    "
