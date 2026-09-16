#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=dose16
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/dose_sep16/joblogs/%x_%A_%a.out

# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used. Arm = row of
# INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/dose_sep16
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

if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"; exit 0
fi

# RESUME OWN, else SEED FROM ONE FROZEN ARCHIVE SHARED BY THE WHOLE BATTERY.
# The parent keeps training while these queue, so its live running.pt would
# hand arms launched hours apart different seeds. The first arm to launch picks
# the parent's newest _step<N>.pt archive (immutable, with its own frozen
# buffer sidecar) and records it; later arms read the record.
SEED_FILE=${LOGS}/SEED.txt
FLK_SEED=${WORKDIR}/configs/flk_sep14/joblogs/SEED.txt
if [ ! -s ${SEED_FILE} ] && [ -s ${FLK_SEED} ]; then cp ${FLK_SEED} ${SEED_FILE}; echo "  seed inherited from flk_sep14: $(cat ${SEED_FILE})"; fi
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"; CK=${OWN}
else
    if [ ! -s ${SEED_FILE} ]; then
        NEWEST=$(ls -t ${CKPTS}/*${SRC}_*_step*[0-9].pt 2>/dev/null | head -1)
        if [ -z "${NEWEST}" ]; then
            echo "FATAL: no archive matches *${SRC}_*_step<N>.pt in ${CKPTS}" >&2; exit 1
        fi
        echo "$(basename ${NEWEST})" > ${SEED_FILE}
        echo "  seed chosen by this arm: $(basename ${NEWEST})"
    fi
    CK=${CKPTS}/$(cat ${SEED_FILE})
    if [ ! -f "${CK}" ]; then echo "FATAL: recorded seed ${CK} is missing" >&2; exit 1; fi
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK}) (mtime $(stat -c %y ${CK}))"
fi

# THE PRIOR MODEL: own if a leg wrote one, else the PARENT'S (its stub wrote it).
OWNPRIOR=$(ls -t ${CKPTS}/*${ARM}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${OWNPRIOR}" ]; then PM=$(basename ${OWNPRIOR})
else
    NP=$(ls ${CKPTS}/*${SRC}_*_prior.pt 2>/dev/null | wc -l)
    if [ "${NP}" -ne 1 ]; then
        echo "FATAL: ${NP} matches for *${SRC}_*_prior.pt (need exactly 1)" >&2; exit 1
    fi
    PM=$(basename $(ls ${CKPTS}/*${SRC}_*_prior.pt))
fi
echo "  prior model <- ${PM}"

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" ${CONFIG} > ${RESOLVED}
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|PRIOR_MODEL_PLACEHOLDER' ${RESOLVED}; then
    echo "FATAL: placeholder left in ${RESOLVED}" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

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
    touch ${CKPTS}/${ARM}.dead
fi
