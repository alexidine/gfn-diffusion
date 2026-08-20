#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-4
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=clu19_mem
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/cluster_aug19/joblogs/%x_%A_%a.out

# cluster_aug19 :: mem. Arm = row of INDEX_mem.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index; a short
# range drops tail arms with no error at all.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/cluster_aug19
LOGS=${ARMS}/joblogs

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX_mem.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# ---- one-shot environment record. MIG turns the in-process sensor off SILENTLY
# and permanently; the UUID is what proves both samplers read the same card.
{ nvidia-smi -L
  nvidia-smi --query-gpu=mig.mode.current,uuid,name,memory.total,driver_version --format=csv
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# ---- concurrent samplers spanning the WHOLE job, so the denominator matches the
# scheduler's. FILTER TO OUR GPU INDEX when reading these: sample output shows
# 4-GPU nodes at wildly different utilizations, and a co-tenant is the cluster's
# version of the confound that muddied the local sizer comparison.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
stdbuf -oL nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv,nounits \
    -l 30 > ${J}_apps.csv &
APPS_PID=$!
if command -v dcgmi >/dev/null 2>&1; then
    # SM_ACTIVE (field 1002): the only genuinely DIFFERENT instrument here.
    # utilization.gpu is an any-kernel-resident duty cycle, blind to how MUCH of
    # the GPU a kernel uses -- and F-048 notes these files have never been read.
    stdbuf -oL dcgmi dmon -e 1002 -d 10000 > ${J}_dcgm.txt 2>&1 &
    DCGM_PID=$!
else
    echo "dcgmi not available" > ${J}_dcgm.txt
    DCGM_PID=""
fi

# ---- epilogue that survives scancel: on a cancelled job THIS is the record.
epilogue() {
    kill ${SMI_PID} ${APPS_PID} ${DCGM_PID} 2>/dev/null
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
        # PER-ARM ENV, one KEY=VAL per line in <arm>.env. SINGLE quotes only in
        # this block, comments included: it is interpolated inside the srun bash
        # -c string, so a double quote truncates the command and train.py never
        # starts, with no wandb run and a file that still passes bash -n.
        if [ -f ${ARMS}/${ARM}.env ]; then
            set -a; . ${ARMS}/${ARM}.env; set +a
            echo 'arm env:'; cat ${ARMS}/${ARM}.env
        fi
        python -u train.py --config ${CONFIG}
    "
