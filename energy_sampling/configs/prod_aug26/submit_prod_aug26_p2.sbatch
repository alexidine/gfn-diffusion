#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=20-29
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=prod26p2
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_aug26/joblogs/%x_%A_%a.out

# prod_aug26 PHASE-2 fan. Arm = row of INDEX.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/prod_aug26
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $2}' ${ARMS}/INDEX.tsv)
TAG=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $3}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

# WARM-START RESOLUTION. The checkpoint filename embeds a problem hash the
# generator does not recompute; resolve by run-name glob, newest first, into a
# per-job COPY of the config (the repo config keeps its placeholder). LOUD on
# a miss: a fan arm silently cold-starting would corrupt the whole comparison.
J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  REQUEUE: resuming own $(basename ${OWN})"
    CK=${OWN}
else
    N=$(ls ${CKPTS}/*${SRC}_*_${TAG}.pt 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "FATAL: no warm checkpoint matches *${SRC}_*_${TAG}.pt in ${CKPTS}" >&2
        exit 1
    fi
    if [ "${N}" -gt 1 ]; then
        echo "FATAL: ${N} checkpoints match *${SRC}_*_${TAG}.pt -- ambiguous, refusing:" >&2
        ls ${CKPTS}/*${SRC}_*_${TAG}.pt >&2
        exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_${TAG}.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  warm <- $(basename ${CK})"
fi
sed "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" ${CONFIG} > ${RESOLVED}

{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

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
