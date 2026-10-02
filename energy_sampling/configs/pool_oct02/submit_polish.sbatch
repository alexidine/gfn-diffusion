#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --signal=USR1@900
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-5
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=po_polish
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

# pool_oct02 polish: one task = one row of INDEX.tsv (line 1 is the header) = one slice (mol_seed) of one system's
# pooled anchors, relaxed again by run_search (init_sample_method data) into that system's polish campaign directory
# (shards, stream 'polish'). DO NOT EDIT --array BY HAND: make.py rewrites it. Resubmitting resumes every task.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
ARMS=${PROJECT_ROOT}/gfn-diffusion/energy_sampling/configs/pool_oct02
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/pooled_oct02

ROW=$((SLURM_ARRAY_TASK_ID + 2))
NAME=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${ARMS}/INDEX.tsv)
SUB=$(awk -F'\t' -v n=${ROW} 'NR==n {print $3}' ${ARMS}/INDEX.tsv)
NSTART=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${ARMS}/INDEX.tsv)
if [ -z "${NAME}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CAMP=${DATA}/${NAME}/polish
RUN=polish_${SUB}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
if [ ! -f "${DATA}/${NAME}/starts.pt" ]; then echo "FATAL: ${DATA}/${NAME}/starts.pt missing (upload it)" >&2; exit 1; fi

mkdir -p ${CAMP}/runs
for f in coord.yaml polish.yaml; do  # frozen on the first task: a later pull cannot change a running polish
    if [ -f "${CAMP}/${f}" ]; then
        cmp -s "${ARMS}/${NAME}/${f}" "${CAMP}/${f}" || { echo "FATAL: ${CAMP}/${f} differs from ${ARMS}/${NAME}/${f}" >&2; exit 1; }
    else
        cp "${ARMS}/${NAME}/${f}" "${CAMP}/${f}.tmp.$$" && mv -n "${CAMP}/${f}.tmp.$$" "${CAMP}/${f}"; rm -f "${CAMP}/${f}.tmp.$$"
    fi
done

echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${NAME} slice ${SUB} of ${NSTART} starts, code=${COMMIT} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${CAMP}/runs \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH
        python - <<'PY' || exit 1
import yaml
cfg = yaml.safe_load(open('${CAMP}/polish.yaml'))
cfg['run_name'] = '${RUN}'
cfg['mol_seed'] = int('${SUB}')
cfg['opt_seed'] = int(cfg['opt_seed']) + int('${SUB}') * 15_000_000
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open('${CAMP}/runs/${RUN}.yaml', 'w'))
PY
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CAMP}/runs/${RUN}.yaml"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} status=${RC} end=$(date -Is)"
exit ${RC}
