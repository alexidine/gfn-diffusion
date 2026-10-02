#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=po_merge
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

# pool_oct02 merge: one task = one system (the order of make.py's SYSTEMS): a coordinator curate pass over the polish
# campaign (every polish shard, RDF leader clustering at the identity cut), then pool_anchors export -> the lowest state
# of every basin within 12 kT, in the trainer's chart, as <system>/anchors_polished.pt (prior layout; the flood's seeds).
# The GPU is used only by the export's trainer-way re-scoring (UMA). launch.sh submits this after every polish task.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/pool_oct02
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/pooled_oct02
UMA=/scratch/mk8347/models/uma/esen_s.pt

# the first INDEX row of each system, in order of appearance
ROW=$(awk -F'\t' 'NR > 1 && !seen[$2]++ {print NR}' ${ARMS}/INDEX.tsv | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "${ROW}" ]; then echo "no system at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
read -r NAME SG KEY TSF MOL <<< "$(awk -F'\t' -v n=${ROW} 'NR==n {print $2, $7, $8, $9, $10}' ${ARMS}/INDEX.tsv)"
CAMP=${DATA}/${NAME}/polish
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} merge ${NAME} (sg ${SG}, ${KEY}) start=$(date -Is)"
ls ${CAMP}/shards/*/ | wc -l

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u -m mxtaltools.crystal_search.coordinator ${CAMP} || exit 1
        python -u -m data_processing.pool_anchors export ${CAMP} ${DATA}/${NAME}/anchors_polished.pt \
            --mol ${MOL} --sg ${SG} --key ${KEY} --tsf ${TSF} --window 12 --mlip_path ${UMA}
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} status=${RC} end=$(date -Is)"
exit ${RC}
