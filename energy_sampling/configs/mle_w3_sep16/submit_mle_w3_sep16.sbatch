#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mlew3
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mle_w3_sep16/joblogs/%x_%A_%a.out

# mle_w3_sep16: phase-1 MLE on the W3 P2_1/c priors, weights-only from the mle09 best-MLE checkpoints.
# Arm = row of INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/mle_w3_sep16
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
mkdir -p ${LOGS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${ARMS}/INDEX.tsv)
PRIOR=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${ARMS}/INDEX.tsv)
PRIOR_BYTES=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION: total_reloads is not checkpointed.
if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# THE PENALTY THESE PRIORS WERE FILTERED AGAINST lives in MXtalTools (a71efd6b).
if ! grep -q 'MONO_CLASS' ${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py; then
    echo "FATAL: MXtalTools checkout lacks MONO_CLASS (commit a71efd6b, the monoclinic class walls) -- git pull MXtalTools" >&2
    exit 1
fi

# THE PRIOR MUST BE THE WHOLE FILE: a partial upload has the right name and the wrong size.
HAVE=$(stat -c %s ${DATA}/${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES} -- upload unfinished or a different file" >&2
    exit 1
fi

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif continue_from_checkpoint`,
# so checkpoint_name ALWAYS wins. An arm with its own _running.pt blanks it and auto-resumes
# (full load: optimizer state and step carry; load_weights_only is read only on the
# checkpoint_name branch). A first launch seeds weights-only from the mle09 arm's _best.pt,
# and REFUSES an absent or ambiguous match.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|null|" -e "s|CONTINUE_PLACEHOLDER|true|" ${CONFIG} > ${RESOLVED}
else
    N=$(ls ${CKPTS}/*${SRC}_*_best.pt 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "FATAL: no seed matches *${SRC}_*_best.pt in ${CKPTS}" >&2; exit 1
    fi
    if [ "${N}" -gt 1 ]; then
        echo "FATAL: ${N} ambiguous matches for *${SRC}_*_best.pt:" >&2
        ls ${CKPTS}/*${SRC}_*_best.pt >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_best.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK}) (mtime $(stat -c %y ${CK}))"
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" -e "s|CONTINUE_PLACEHOLDER|false|" ${CONFIG} > ${RESOLVED}
fi
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|CONTINUE_PLACEHOLDER' ${RESOLVED}; then
    echo "FATAL: placeholder left in ${RESOLVED}" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# The out-of-process occupancy record -- the number the scheduler cancels on.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
smi_epilogue() {
    kill ${SMI_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${J}_sacct.txt 2>&1
}
trap smi_epilogue EXIT TERM

# THE LAUNCH LINE IS COPIED, NOT REWRITTEN: PYTHONPATH must hold the PARENT of
# energy_sampling or every arm dies at import; both --bind lines are required.
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u train.py --config ${RESOLVED}
    " 2>&1 | tee ${J}.trainlog

if grep -q 'UNRECOVERABLE' ${J}.trainlog 2>/dev/null; then
    echo "arm ${ARM} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${CKPTS}/${ARM}.dead
fi
