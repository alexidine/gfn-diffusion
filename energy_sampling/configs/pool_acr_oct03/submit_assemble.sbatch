#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_assemble
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A.out

# pool_acr_oct03 assemble: anchors.pt + the walk's shards -> the shipped prior file (data_processing/pool_assemble_mace.py):
# states outside the trainer's latent box and states the trainer's density penalty touches left out, latent de-dupe
# (anchors at DEDUPE; walk states at WALK_RADIUS, the 1 kT latent kick under acr_newmodel -- a physical scale, no row
# budget: owner 2026-10-05), the 8 normaliser images of every kept state, a re-scoring check by MACE. The file is as
# large as that makes it (about 3 million rows expected); E_MAX_KT=<x> leaves out states more than x kT above the
# lowest, the way to make it smaller. Written as <SYS>/acridine_mace_pooled_oct03_pc055_prior.pt and copied into the priors
# directory the training arms read. The unfiltered file of job 19219651 (acridine_mace_pooled_oct03_prior.pt, no
# density filter, walk radius 0.284) is NOT a training input.
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
WALK_RADIUS=${WALK_RADIUS:-0.031}
E_MAX_KT=${E_MAX_KT:-}
PRIORS=/scratch/mk8347/data/crystal_datasets/conditional/priors

OUT=${SYS}/${NAME}_pooled_oct03_pc055_prior.pt
if [ ! -f "${SYS}/anchors.pt" ]; then echo "FATAL: ${SYS}/anchors.pt missing (the export stage writes it)" >&2; exit 1; fi
if [ -z "$(ls ${SYS}/flood/shard_*/shard_*.pt 2>/dev/null | head -1)" ]; then echo "FATAL: no walk shards under ${SYS}/flood" >&2; exit 1; fi
EMAX_ARG=""
if [ -n "${E_MAX_KT}" ]; then EMAX_ARG="--e-max-kT ${E_MAX_KT}"; fi
echo "JOB ${SLURM_JOB_ID} assemble ${NAME} (dedupe ${DEDUPE}, walk radius ${WALK_RADIUS}, energy ceiling ${E_MAX_KT:-none}) start=$(date -Is)"

srun singularity exec --nv --overlay ${OVERLAY}:ro --bind ${PROJECT_ROOT}:${PROJECT_ROOT} --bind /scratch/mk8347/data:/scratch/mk8347/data --pwd ${WORKDIR} ${IMAGE}     /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m data_processing.pool_assemble_mace ${SYS}/anchors.pt ${SYS}/flood ${OUT} --sg 14 --mol ${MOL} \
            --mlip_path ${MODEL} --dedupe ${DEDUPE} --walk-radius ${WALK_RADIUS} ${EMAX_ARG}
    "
RC=$?
if [ ${RC} -eq 0 ]; then
    cp "${OUT}" "${PRIORS}/" && echo "copied to ${PRIORS}/$(basename ${OUT}): $(stat -c %s ${PRIORS}/$(basename ${OUT})) bytes" || RC=1
fi
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
