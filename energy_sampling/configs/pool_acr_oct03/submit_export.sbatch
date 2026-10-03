#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_export
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A.out

# pool_acr_oct03 export: the pooled registry's basins within WINDOW kT, in the chart the trainer reads, re-scored by
# MACE the trainer's way (data_processing/pool_anchors_planar.py: exact for planar acridine), as <SYS>/anchors.pt
# (prior layout): the anchor rows of the prior and the starts of the walk.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/pool_acr_oct03
PYLIBS=/scratch/mk8347/pylibs
NAME=acridine_mace
SYS=/scratch/mk8347/data/crystal_datasets/pooled_oct02/${NAME}
CAMP=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_zp1_sep30
MOL=/scratch/mk8347/data/crystal_datasets/acridine/acr_newmodel_conformer.pt
MODEL=/scratch/mk8347/data/acr_newmodel.model
WINDOW=${WINDOW:-5}

POOL=${SYS}/pool
if [ ! -f "${POOL}/registry.pt" ]; then echo "FATAL: ${POOL}/registry.pt missing (the premerge stage writes it)" >&2; exit 1; fi
echo "JOB ${SLURM_JOB_ID} export ${NAME} (window ${WINDOW} kT) start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m data_processing.pool_anchors_planar ${POOL} ${SYS}/anchors.pt --mol ${MOL} --mlip_path ${MODEL}             --window ${WINDOW} --device cuda
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
