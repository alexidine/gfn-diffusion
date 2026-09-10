#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mle09
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mle_sep09/joblogs/%x_%A_%a.out

# mle_sep09: phase 1 continued on the rebuilt priors. Arm = row of INDEX.tsv
# (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/mle_sep09
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $4}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# REQUEUE SAFETY. These arms have no exit gate and run to the 48 h wall, so every
# one of them is EXPECTED to be resubmitted. train.py's loader is
# `if checkpoint_name ... elif continue_from_checkpoint`, so checkpoint_name
# ALWAYS wins -- left pointing at the seed, a requeued arm would silently restart
# from its ancestor's phase-1 exit and throw away everything this wave bought.
# On a requeue we therefore BLANK checkpoint_name and continue our own run:
# that path takes load_full, so optimizer state and the step count carry, and
# load_weights_only is read only on the checkpoint_name branch.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  REQUEUE: continuing own $(basename ${OWN})"
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|null|" -e "s|CONTINUE_PLACEHOLDER|true|" ${CONFIG} > ${RESOLVED}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: an arm that seeds
    # from a file nobody named is not a result. phase1_exit ONLY -- never a live
    # _running.pt, which array tasks would glob at different steps.
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
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" -e "s|CONTINUE_PLACEHOLDER|false|" ${CONFIG} > ${RESOLVED}
fi

# A placeholder that survived substitution would reach train.py as a literal
# filename and fail deep in the loader; catch it here, where the message is about
# the substitution that did not happen.
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|CONTINUE_PLACEHOLDER' ${RESOLVED}; then
    echo "FATAL: placeholder left in ${RESOLVED}" >&2
    grep -n 'WARM_CHECKPOINT_PLACEHOLDER\|CONTINUE_PLACEHOLDER' ${RESOLVED} >&2
    exit 1
fi

{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# The out-of-process occupancy record -- this is the number the scheduler cancels
# on, and the in-process sensor reads several points low on the MLIP routes.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI=$!

# THE LAUNCH LINE IS COPIED FROM configs/prod_t100, NOT REWRITTEN. Four things
# in it are load-bearing and none are obvious:
#   PYTHONPATH  train.py does `from energy_sampling...`, so the PARENT of this
#               directory must be on the path. Running `python train.py` from
#               inside energy_sampling/ puts only energy_sampling/ on sys.path,
#               and every arm dies at import with ModuleNotFoundError.
#   --bind      the container sees neither the project tree nor
#               /scratch/mk8347/data without them. The seed glob above runs
#               OUTSIDE the container and passes regardless, so this fails late
#               and looks unrelated to the launch line.
#   --pwd       sets the working directory inside the container.
#   -u          unbuffered, so a joblog tail shows progress rather than nothing.
srun singularity exec --nv     --overlay ${OVERLAY}:ro     --bind ${PROJECT_ROOT}:${PROJECT_ROOT}     --bind /scratch/mk8347/data:/scratch/mk8347/data     --pwd ${WORKDIR}     ${IMAGE}     /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u train.py --config ${RESOLVED}
    "
RC=$?
kill ${SMI} 2>/dev/null
exit ${RC}
