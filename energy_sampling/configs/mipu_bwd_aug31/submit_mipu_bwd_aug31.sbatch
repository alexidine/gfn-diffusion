#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mb31
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mipu_bwd_aug31/joblogs/%x_%A_%a.out

# mipu_bwd_aug31 -- MIPCAS UMA, bwd floor x learning rate, 8 arms.
# Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/mipu_bwd_aug31
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

# REQUEUE SAFETY. 24 h reaches ~4600 steps at ~18.7 s/step against a 15000-step
# budget, so these arms are EXPECTED to hit the wall repeatedly. train.py's
# loader is `if checkpoint_name ... elif continue_from_checkpoint`, so
# checkpoint_name ALWAYS wins -- without this an extended arm would silently
# restart from the phase-1 exit and discard its own progress. Resubmit this
# same sbatch to extend.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  REQUEUE: resuming own $(basename ${OWN})"
    CK=${OWN}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the SAME file or it is not a comparison.
    N=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "FATAL: no phase-1 exit matches *${SRC}_*_phase1_exit.pt in ${CKPTS}" >&2
        exit 1
    fi
    if [ "${N}" -gt 1 ]; then
        echo "FATAL: ${N} checkpoints match *${SRC}_*_phase1_exit.pt -- ambiguous, refusing:" >&2
        ls ${CKPTS}/*${SRC}_*_phase1_exit.pt >&2
        exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  seed <- $(basename ${CK})"
fi
sed "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" ${CONFIG} > ${RESOLVED}

{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# the out-of-process occupancy record -- the in-process sensor disagrees with it
# by a batch-dependent, sign-flipping error, and THIS is the number the
# scheduler cancels on.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!

epilogue() {
    kill ${SMI_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${J}_sacct.txt 2>&1
}
trap epilogue EXIT TERM

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
    "
