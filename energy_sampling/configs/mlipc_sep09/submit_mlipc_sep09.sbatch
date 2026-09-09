#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mlipc09
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/mlipc_sep09/joblogs/%x_%A_%a.out

# Four MLIP execution settings on one every-step UMA arm. See make.py.
#
# A100 ON PURPOSE, unlike the compile-rollout probe: the reference numbers this is
# read against (p02_mipu_lr0p0625, energy 39.3% of 41.8 h) were measured on an A100,
# and MLIP cost per sample is what we are comparing. One hour, one GPU, so it should
# still backfill. Drop to --gres=gpu:1 only if the pool will not take it, and then
# compare arms to each other rather than to the p02 baseline.
#
# 64G not 48G: `ckoff` removes MLIP activation checkpointing, and the init-time prior
# re-analysis runs with the graph LIVE. That pass is where this route has OOM'd
# before. A GPU OOM there is a legitimate result; a HOST OOM would just be noise.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/mlipc_sep09
LOGS=${ARMS}/joblogs
CKPTS=${WORKDIR}/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/mlipc_${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/mlipc_${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# Every arm seeds from the SAME phase-1 exit, refusing an ambiguous match: the four
# cells differ by the MLIP setting or they differ by nothing worth measuring.
N=$(ls ${CKPTS}/*pt100_mipu_lr4p0_*_phase1_exit.pt 2>/dev/null | wc -l)
if [ "${N}" -ne 1 ]; then
    echo "FATAL: need exactly one pt100_mipu_lr4p0 phase-1 exit, found ${N}" >&2
    ls ${CKPTS}/*pt100_mipu_lr4p0_*_phase1_exit.pt >&2; exit 1
fi
CK=$(ls ${CKPTS}/*pt100_mipu_lr4p0_*_phase1_exit.pt)
echo "arm ${ARM}  SEED $(basename ${CK})"

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|null|" ${CONFIG} > ${RESOLVED}

{ nvidia-smi -L; scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"; } > ${J}.info 2>&1

# occupancy, because a faster MLIP REMOVES GPU-busy time and mipu already sits at
# 62-66% against a ~54% kill line
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null' EXIT TERM

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
