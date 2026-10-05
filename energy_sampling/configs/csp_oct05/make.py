"""csp_oct05: writes INDEX.tsv (one `eval/cond_panel/csp.py collect` call per row) and submit_csp_oct05.sbatch beside
this file.

    cd energy_sampling && python configs/csp_oct05/make.py

Rows: the 24-molecule panel (12 training + 12 held-out, the evaluator's seed 0) at seven archived steps of
qf30_fwdF_lr5 and one of qf30_upb_lr5, each as 256 plain draws and as the density modes of 32,768 draws; the random
initialiser and the unconditional prior beside the step-100,000 rows; 4,096 starts per molecule for the random
initialiser and for step 100,000 (csp24x); and a 100-molecule panel at step 100,000 (csp100)."""
import os

root = os.path.dirname(os.path.abspath(__file__))
PLAIN = '--starts 256 --batch-rows 1024'
MODES = '--cluster-from 32768 --starts 32 --batch-rows 256'
rows = [('p24_s100k_plain', 'csp24', 'qf30_fwdF_lr5', 100000, 's100k', 12, 12, f'--random --prior {PLAIN}'),
        ('p24_s100k_modes', 'csp24', 'qf30_fwdF_lr5', 100000, 's100k', 12, 12, f'--prior {MODES}')]
for step in (10000, 20000, 30000, 40000, 50000, 70000):
    name = f's{step // 1000:03d}k'
    rows += [(f'p24_{name}_plain', 'csp24', 'qf30_fwdF_lr5', step, name, 12, 12, PLAIN),
             (f'p24_{name}_modes', 'csp24', 'qf30_fwdF_lr5', step, name, 12, 12, MODES)]
rows += [('p24_upb070k_plain', 'csp24', 'qf30_upb_lr5', 70000, 'upb070k', 12, 12, PLAIN),
         ('p24_upb070k_modes', 'csp24', 'qf30_upb_lr5', 70000, 'upb070k', 12, 12, MODES),
         ('p24x_random', 'csp24x', 'qf30_fwdF_lr5', 100000, 's100k', 12, 12, '--skip-models --random --starts 4096 --batch-rows 1024'),
         ('p24x_s100k', 'csp24x', 'qf30_fwdF_lr5', 100000, 's100k', 12, 12, '--starts 4096 --batch-rows 1024'),
         ('p100_s100k_modes', 'csp100', 'qf30_fwdF_lr5', 100000, 's100k', 50, 50, f'--prior {MODES}'),
         ('p100_s100k_plain', 'csp100', 'qf30_fwdF_lr5', 100000, 's100k', 50, 50, '--random --prior --starts 64 --batch-rows 1024')]
with open(os.path.join(root, 'INDEX.tsv'), 'w', newline='\n') as fh:
    fh.write('\t'.join(['tag', 'out', 'arm', 'step', 'sampler', 'n_train', 'n_test', 'collect_args']) + '\n')
    for r in rows:
        fh.write('\t'.join(str(x) for x in r) + '\n')

sbatch = r'''#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-__LAST__
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
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u -m eval.cond_panel.csp collect --model ${NAME}=${CK} --config ${CONFIG} --search-config ${SEARCH} \
            --out ${RESULTS}/${OUT} --n-train ${NTRAIN} --n-test ${NTEST} ${ARGS}
    " 2>&1 | tee ${J}.log
exit ${PIPESTATUS[0]}
'''.replace('__LAST__', str(len(rows) - 1))
with open(os.path.join(root, 'submit_csp_oct05.sbatch'), 'w', newline='\n') as fh:
    fh.write(sbatch)
print(f'{len(rows)} tasks written to {root}')
for i, r in enumerate(rows):
    print(i, *r, sep='  ')
