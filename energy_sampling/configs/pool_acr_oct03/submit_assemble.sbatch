#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_assemble
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A.out

# pool_acr_oct03 assemble: anchors.pt + the walk's shards -> the shipped prior file
# <SYS>/acridine_mace_pooled_oct03_prior.pt (data_processing/pool_assemble_mace.py): anchors outside the trainer's latent
# box left out, latent de-dupe at DEDUPE (walk states widened until the file fits TARGET_ROWS), the 8 normaliser images
# of every kept row, and a re-scoring check by MACE. No relabelling of the molecule's own symmetry (owner 2026-10-05).
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
DEDUPE=${DEDUPE:-0.01}
TARGET_ROWS=${TARGET_ROWS:-400000}
PC_MIN=${PC_MIN:-0.55}   # packing coefficient kept: the range the trainer's density penalty leaves unpenalised
PC_MAX=${PC_MAX:-0.95}
OUTTAG=${OUTTAG:-}    # appended to the output name, e.g. _pc062

OUT=${SYS}/${NAME}_pooled_oct03${OUTTAG}_prior.pt
if [ ! -f "${SYS}/anchors.pt" ]; then echo "FATAL: ${SYS}/anchors.pt missing (the export stage writes it)" >&2; exit 1; fi
if [ -z "$(ls ${SYS}/flood/shard_*/shard_*.pt 2>/dev/null | head -1)" ]; then echo "FATAL: no walk shards under ${SYS}/flood" >&2; exit 1; fi
echo "JOB ${SLURM_JOB_ID} assemble ${NAME} (dedupe ${DEDUPE}, target rows ${TARGET_ROWS}) start=$(date -Is)"

srun singularity exec --nv     --overlay ${OVERLAY}:ro     --bind ${PROJECT_ROOT}:${PROJECT_ROOT}     --bind /scratch/mk8347/data:/scratch/mk8347/data     --pwd ${WORKDIR}     ${IMAGE}     /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m data_processing.pool_assemble_mace ${SYS}/anchors.pt ${SYS}/flood ${OUT} --sg 14 --mol ${MOL} \
            --mlip_path ${MODEL} --dedupe ${DEDUPE} --target-rows ${TARGET_ROWS} --pc-min ${PC_MIN} --pc-max ${PC_MAX}
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
