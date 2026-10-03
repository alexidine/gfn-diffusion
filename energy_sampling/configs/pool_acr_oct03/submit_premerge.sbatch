#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pa_premerge
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_acr_oct03/joblogs/%x_%A.out

# pool_acr_oct03 premerge (CPU): one coordinator curate pass over the acridine pool -- every end state of the
# acr_zp1_sep30 campaign within 10 kT (its shards, linked, not copied), RDF leader clustering at the campaign's
# atomwise cut -- into <SYS>/pool/registry.pt. Rerunning continues: shards already ingested are skipped.
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

POOL=${SYS}/pool
mkdir -p "${POOL}" || exit 1
if [ ! -d "${CAMP}/shards" ]; then echo "FATAL: ${CAMP}/shards missing" >&2; exit 1; fi
[ -e "${POOL}/shards" ] || ln -s "${CAMP}/shards" "${POOL}/shards"
cp "${ARMS}/pool_coord.yaml" "${POOL}/coord.yaml"
echo "JOB ${SLURM_JOB_ID} premerge ${NAME}: $(ls ${POOL}/shards | wc -l) shard directories start=$(date -Is)"

srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
        python -u -m mxtaltools.crystal_search.coordinator ${POOL} > ${POOL}/curate.log 2>&1 || { tail -30 ${POOL}/curate.log; exit 1; }
        tail -25 ${POOL}/curate.log
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
