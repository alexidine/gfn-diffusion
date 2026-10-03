#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=po_assemble
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

# pool_oct02 assemble: one task = one system of SYSTEMS.tsv: data_processing/pool_assemble.py on anchors_polished.pt
# and every flood shard -> <system>/<system>_pooled_oct02_prior.pt (anchors + flood, latent de-dupe, normaliser images;
# 'prior' and 'equalized_prior' hold the same rows; a sample of rows re-scored). Environment: DEDUPE (default 0.01, the
# latent radius), TARGET_ROWS (default 400000: the flood radius widens until the file holds at most that many rows).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/pool_oct02
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/pooled_oct02
UMA=/scratch/mk8347/models/uma/esen_s.pt

ROW=$((SLURM_ARRAY_TASK_ID + 2))
read -r NAME SG KEY TSF MOL PRE <<< "$(awk -F'\t' -v n=${ROW} 'NR==n {print $2, $3, $4, $5, $6, $7}' ${ARMS}/SYSTEMS.tsv)"
if [ -z "${NAME}" ]; then echo "no system at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi

if [ ! -d "${DATA}/${NAME}/flood" ]; then echo "${NAME}: no flood yet; nothing to assemble"; exit 0; fi
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} assemble ${NAME} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m data_processing.pool_assemble ${DATA}/${NAME}/anchors_polished.pt ${DATA}/${NAME}/flood \
            ${DATA}/${NAME}/${NAME}_pooled_oct02_prior.pt --sg ${SG} --key ${KEY} --mol ${MOL} --mlip_path ${UMA} \
            --dedupe ${DEDUPE:-0.01} --target-rows ${TARGET_ROWS:-400000}
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} status=${RC} end=$(date -Is)"
exit ${RC}
