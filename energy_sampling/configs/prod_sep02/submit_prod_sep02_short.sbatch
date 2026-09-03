#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-11
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=p02short
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_sep02/joblogs/%x_%A_%a.out

# prod_sep02 2-day chained arms. Arm = row of INDEX_short.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/prod_sep02
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX_short.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $2}' ${ARMS}/INDEX_short.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# A DEAD ARM MUST NOT EAT ITS REMAINING LEGS. total_reloads is not checkpointed,
# so a leg that aborted UNRECOVERABLE hands the next leg a fresh rewind budget,
# the same poisoned _running.pt, and an identical re-detonation. This is a static
# LR scan, so some arms are MEANT to be too hot and cannot self-recover.
if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- without this an
# extended arm would silently restart from the phase-1 exit and discard its own
# progress.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    CK=${OWN}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the same file or it is not a comparison.
    N=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "FATAL: no phase-1 exit matches *${SRC}_*_phase1_exit.pt" >&2; exit 1
    fi
    if [ "${N}" -gt 1 ]; then
        echo "FATAL: ${N} ambiguous matches for *${SRC}_*_phase1_exit.pt:" >&2
        ls ${CKPTS}/*${SRC}_*_phase1_exit.pt >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK})"
fi

# THE PRIOR MODEL, resolved the same way and for the same reason. `prior_model`
# is not in the checkpoint and only train_prior's snapshot_prior creates it, but
# protocol.begin() returns early when step_ind != 0 -- so a resumed leg would
# come up with no prior sampler and bwd_sampling_mode: prior would silently
# degrade to an anchors-only buffer across 95% of the branch budget. On leg 1
# there is nothing to find and `null` is correct: the stub runs and writes it.
OWNPRIOR=$(ls -t ${CKPTS}/*${ARM}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${OWNPRIOR}" ]; then
    PM=$(basename ${OWNPRIOR}); echo "  prior model <- ${PM}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" ${CONFIG} > ${RESOLVED}

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# THE OCCUPANCY RECORD THAT MATTERS. The in-process sensor disagrees with this
# by a batch-dependent, sign-flipping error, and THIS is the number the scheduler
# cancels on -- 16795765_0 was killed mid-run at 15.19 s/it on 2026-09-02 with no
# way to tell whether it had fallen into the 38-49% band, because this sidecar
# was missing. Read it with:
#   tail -n 400 <arm>_smi.csv | awk -F, '$3+0==$3 {s+=$3;n++} END {print s/n}'
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
smi_epilogue() {
    kill ${SMI_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${J}_sacct.txt 2>&1
}
trap smi_epilogue EXIT TERM

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
    echo "arm ${ARM} exhausted its rewind budget -- sentinel set, later legs will skip"
    touch ${CKPTS}/${ARM}.dead
fi
