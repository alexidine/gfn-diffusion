#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_assemble
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A.out

# pool_acr_oct03 assemble: data_processing/pool_assemble_mace.py on anchors.pt and every walk shard ->
# <SYS>/acridine_mace_pooled_oct05_prior.pt and its .summary.json, then both copied into ${PRIORS}. It follows
# pool_oct02's assemble (GFN 54ea7e36): candidates under CEILING kT above the lowest anchor, thinned in the trainer
# latent at ONE radius for every pair, RADIUS = the latent kick that raises a relaxed basin by a median 1 kT under
# acr_newmodel (0.0312, measured 2026-09-30); no row budget; out-of-box and density-penalised states left out; the 8
# normaliser images; a sample of rows re-scored by MACE. Earlier acridine files (acridine_mace_pooled_oct03_prior.pt,
# no density filter; ..._pc055_prior.pt, thinned to a row budget on the dev box) are NOT training inputs.
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
RADIUS=${RADIUS:-0.0312}
CEILING=${CEILING:-10}
PRIORS=/scratch/mk8347/data/crystal_datasets/conditional/priors

OUT=${SYS}/${NAME}_pooled_oct05_prior.pt
if [ ! -f "${SYS}/anchors.pt" ]; then echo "FATAL: ${SYS}/anchors.pt missing (the export stage writes it)" >&2; exit 1; fi
if [ -z "$(ls ${SYS}/flood/shard_*/shard_*.pt 2>/dev/null | head -1)" ]; then echo "FATAL: no walk shards under ${SYS}/flood" >&2; exit 1; fi
echo "JOB ${SLURM_JOB_ID} assemble ${NAME} radius ${RADIUS} ceiling ${CEILING} kT start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m data_processing.pool_assemble_mace ${SYS}/anchors.pt ${SYS}/flood ${OUT} --sg 14 --mol ${MOL} \
            --mlip_path ${MODEL} --radius ${RADIUS} --ceiling_kT ${CEILING} || exit 1
        cp -v ${OUT} ${OUT}.summary.json ${PRIORS}/ || exit 1
        cat ${OUT}.summary.json
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
