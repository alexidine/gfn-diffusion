#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=p12
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_sep12/joblogs/%x_%A_%a.out

# prod_sep12: phase 2 from the mle09 best-MLE checkpoints. Arm = row of
# INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py
# rewrites it. Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/prod_sep12
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $2}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION. total_reloads is not checkpointed,
# so a leg that aborted UNRECOVERABLE would hand the next one a fresh rewind
# budget and the same poisoned _running.pt.
if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- an arm with its
# own _running.pt continues it; only a first launch seeds from the mle09 arm.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    CK=${OWN}
else
    # THE SEED IS THE mle09 ARM'S _best.pt: its bwd/mle record, hardlinked off
    # `running` at the 50-step save cadence. mle09's stage is terminal, so no
    # phase1_exit.pt exists for these arms. REFUSES AN AMBIGUOUS MATCH.
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
fi

# THE PRIOR MODEL. Not in the checkpoint; only train_prior's snapshot_prior
# writes it, and protocol.begin() returns early on a resumed step, so a
# continued leg must be handed its own. On the first leg the stub writes it.
OWNPRIOR=$(ls -t ${CKPTS}/*${ARM}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${OWNPRIOR}" ]; then
    PM=$(basename ${OWNPRIOR}); echo "  prior model <- ${PM}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" ${CONFIG} > ${RESOLVED}
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|PRIOR_MODEL_PLACEHOLDER' ${RESOLVED}; then
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
