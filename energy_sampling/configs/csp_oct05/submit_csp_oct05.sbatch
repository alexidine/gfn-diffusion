#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-19
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=csp05
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/csp_oct05/joblogs/%x_%A_%a.out

# csp_oct05: the mock structure-prediction evaluator (eval/cond_panel/csp.py collect) on qm9full_sep30 checkpoints.
# Task = row of INDEX.tsv (line 1 is the header): one collect call. Columns: tag, out, arm, step, sampler, n_train,
# n_test, collect_args. The checkpoint is the arm's _step<step>.pt archive; the run config is the arm's yaml in
# configs/qm9full_sep30. Pools are written under ${RESULTS}/<out>/pools/<sampler>/, one file per molecule; a file on
# disk is skipped, so resubmitting a task continues it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${WORKDIR}/configs/csp_oct05
LOGS=${BATTERY}/joblogs
CKPTS=${WORKDIR}/checkpoints
RESULTS=${PROJECT_ROOT}/csp_oct05
SEARCH=${PROJECT_ROOT}/MXtalTools/configs/crystal_searches/qm9_full_sep29/tasks/0.yaml
PYLIBS=/scratch/mk8347/pylibs
mkdir -p ${LOGS} ${RESULTS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${BATTERY}/INDEX.tsv
col() { awk -F'\t' -v n=${ROW} -v c=$1 'NR==n {print $c}' ${INDEX}; }
TAG=$(col 1); OUT=$(col 2); ARM=$(col 3); STEP=$(col 4); NAME=$(col 5); NTRAIN=$(col 6); NTEST=$(col 7); ARGS=$(col 8)
if [ -z "${TAG}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi

CONFIG=${WORKDIR}/configs/qm9full_sep30/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "FATAL: missing config ${CONFIG}" >&2; exit 1; fi
if [ ! -f "${SEARCH}" ]; then echo "FATAL: missing search config ${SEARCH} -- git pull MXtalTools" >&2; exit 1; fi
if [ ! -f "${PROJECT_ROOT}/MXtalTools/mxtaltools/crystal_search/standardize.py" ]; then
    echo "FATAL: MXtalTools lacks crystal_search/standardize.py -- git pull MXtalTools" >&2; exit 1
fi
if [ ! -f "${WORKDIR}/eval/cond_panel/csp.py" ]; then echo "FATAL: missing eval/cond_panel/csp.py -- git pull gfn-diffusion" >&2; exit 1; fi
if [ ! -d "${PYLIBS}/spglib" ]; then echo "FATAL: ${PYLIBS} holds no spglib (the cell standardisation imports it)" >&2; exit 1; fi
NA=$(ls ${CKPTS}/*${ARM}_*_step${STEP}.pt 2>/dev/null | grep -v '_buffers.pt$' | wc -l)
if [ "${NA}" -ne 1 ]; then
    echo "FATAL: ${NA} matches for *${ARM}_*_step${STEP}.pt in ${CKPTS} (need exactly 1)" >&2; exit 1
fi
CK=$(ls ${CKPTS}/*${ARM}_*_step${STEP}.pt | grep -v '_buffers.pt$')
J=${LOGS}/${TAG}_${SLURM_JOB_ID}
echo "array ${SLURM_ARRAY_TASK_ID} -> ${TAG}: ${NAME}=$(basename ${CK}) -> ${RESULTS}/${OUT}; ${NTRAIN} training + ${NTEST} held-out molecules; ${ARGS}"
{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -c 'import spglib, yaml; from eval.cond_panel import csp, relax, sampler; from data_processing.pool_anchors import Chart; from mxtaltools.crystal_search.standardize import standardize_cells' || { echo 'FATAL: an import the evaluator needs failed in this environment' >&2; exit 1; }
        python -u -m eval.cond_panel.csp collect --model ${NAME}=${CK} --config ${CONFIG} --search-config ${SEARCH} \
            --out ${RESULTS}/${OUT} --n-train ${NTRAIN} --n-test ${NTEST} ${ARGS}
    " 2>&1 | tee ${J}.log
exit ${PIPESTATUS[0]}
