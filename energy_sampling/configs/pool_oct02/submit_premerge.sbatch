#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=po_premerge
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

# pool_oct02 premerge (CPU): one task = one system of SYSTEMS.tsv (line 1 is the header). For a system whose premerge
# column is 1: a coordinator curate pass over the uploaded pool (<system>/pool: coord.yaml from this directory, shards
# uploaded from the dev box), then pool_anchors starts -> <system>/starts.pt (one row per basin within 10 kT, in the
# chart the trainer reads). A system with premerge 0 already has its starts.pt and exits at once.
# DO NOT EDIT --array BY HAND: make.py rewrites it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/pool_oct02
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/pooled_oct02
PRIORS=/scratch/mk8347/data/crystal_datasets/conditional/priors
UMA=/scratch/mk8347/models/uma/esen_s.pt

ROW=$((SLURM_ARRAY_TASK_ID + 2))
read -r NAME SG KEY TSF MOL PRE WINDOW RADIUS CEILING <<< "$(awk -F'\t' -v n=${ROW} 'NR==n {print $2, $3, $4, $5, $6, $7, $8, $9, $10}' ${ARMS}/SYSTEMS.tsv)"
if [ -z "${NAME}" ]; then echo "no system at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi

if [ "${PRE}" != "1" ]; then echo "${NAME}: starts.pt was built locally; nothing to do"; exit 0; fi
POOL=${DATA}/${NAME}/pool
if [ ! -d "${POOL}/shards" ]; then echo "FATAL: ${POOL}/shards missing (upload the pool)" >&2; exit 1; fi
cp "${ARMS}/${NAME}/pool_coord.yaml" "${POOL}/coord.yaml"
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} premerge ${NAME}: $(ls ${POOL}/shards | wc -l) shard directories start=$(date -Is)"

srun singularity exec  \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m mxtaltools.crystal_search.coordinator ${POOL} > ${POOL}/curate.log 2>&1 || { tail -30 ${POOL}/curate.log; exit 1; }
        tail -25 ${POOL}/curate.log
        CUDA_VISIBLE_DEVICES=-1 python -u -m data_processing.pool_anchors starts ${POOL} ${DATA}/${NAME}/starts.pt \
            --mol ${MOL} --sg ${SG} --key ${KEY} --window 10
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} status=${RC} end=$(date -Is)"
exit ${RC}
