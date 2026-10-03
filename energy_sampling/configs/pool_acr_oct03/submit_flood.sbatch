#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_flood
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A_%a.out

# pool_acr_oct03 walk: one task = one of N_SHARDS shards of the capped-MC walk (data_processing/capped_mc.py) from
# <SYS>/anchors.pt under MACE acr_newmodel, with the arguments of FLOOD_ARGS (one line; the pool_oct02 protocol of
# GFN 5fc53fe2), into <SYS>/flood/shard_<k>. Resubmitting continues every shard from its state.pt (--resume).
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
N_SHARDS=4   # = the --array range above
CUT=0.134    # the campaign's atomwise identity cut: starts are thinned at it

SHARD=${SLURM_ARRAY_TASK_ID:?array task id}
ARGS=$(head -1 ${ARMS}/FLOOD_ARGS)
SEEDS=${SYS}/anchors.pt
OUT=${SYS}/flood/shard_${SHARD}
if [ ! -f "${SEEDS}" ]; then echo "FATAL: ${SEEDS} missing (the export stage writes it)" >&2; exit 1; fi
if [ -z "${ARGS}" ]; then echo "FATAL: ${ARMS}/FLOOD_ARGS is empty" >&2; exit 1; fi
mkdir -p ${OUT}
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SHARD} walk ${NAME} shard ${SHARD}/${N_SHARDS} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        export GPU_MEM_FRACTION=0.9
        python -u data_processing/capped_mc.py --prior ${SEEDS} --energy_function mace --mlip_path ${MODEL}             --out ${OUT} --n_shards ${N_SHARDS} --shard ${SHARD} --rdf_dcut ${CUT} ${ARGS} --resume
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SHARD} status=${RC} end=$(date -Is)"
exit ${RC}
