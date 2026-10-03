#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-14
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=po_flood
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

# pool_oct02 flood: one task = one row of FLOOD.tsv (line 1 is the header) = one shard of one system's capped-MC flood
# (data_processing/capped_mc.py) from <system>/anchors.pt, with the row's arguments, into
# <system>/flood/shard_<k>. Resubmitting continues every shard from its state.pt (--resume).
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
NAME=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${ARMS}/FLOOD.tsv)
SHARD=$(awk -F'\t' -v n=${ROW} 'NR==n {print $3}' ${ARMS}/FLOOD.tsv)
NSH=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${ARMS}/FLOOD.tsv)
KEY=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${ARMS}/FLOOD.tsv)
CUT=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${ARMS}/FLOOD.tsv)
ARGS=$(awk -F'\t' -v n=${ROW} 'NR==n {print $7}' ${ARMS}/FLOOD.tsv)
if [ -z "${NAME}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
SEEDS=${DATA}/${NAME}/anchors.pt
OUT=${DATA}/${NAME}/flood/shard_${SHARD}
if [ ! -f "${SEEDS}" ]; then echo "FATAL: ${SEEDS} missing (the export stage writes it)" >&2; exit 1; fi
mkdir -p ${OUT}
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} flood ${NAME} shard ${SHARD}/${NSH} (${KEY}) start=$(date -Is)"

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
        export GPU_MEM_FRACTION=0.9
        python -u data_processing/capped_mc.py --prior ${SEEDS} --energy_function ${KEY} --mlip_path ${UMA} \
            --out ${OUT} --n_shards ${NSH} --shard ${SHARD} --rdf_dcut ${CUT} ${ARGS} --resume
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} status=${RC} end=$(date -Is)"
exit ${RC}
