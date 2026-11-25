#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=224G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool4_nov24
#SBATCH --array=0-1
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool4_nov24/joblogs/%x_%A_%a.out

# pool4_nov24 (make.py, from submit_template.sbatch): one arm per array index: smoke_tb_m4_onpol tb_m4_onpol (settings in each arm yaml).
# Submit: sbatch --array=0-1. Each arm's prior, conditions and test file are read from
# its own yaml and checked here; an arm naming a source checkpoint stops here if the file is missing.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/pool4_nov24
GROUP_SH=${WORKDIR}/configs/cap_size_nov01/run_group.sh
LOGS=${ARMS}/joblogs
mkdir -p ${LOGS}

ARM_LIST=(smoke_tb_m4_onpol tb_m4_onpol)
GROUP="${ARM_LIST[${SLURM_ARRAY_TASK_ID:?array task id}]}"
CKDIR=${WORKDIR}/checkpoints

if [ ! -s ${GROUP_SH} ]; then echo "FATAL: ${GROUP_SH} is missing" >&2; exit 1; fi
if ! grep -q _molecule_table ${WORKDIR}/models/nf_policy.py || ! grep -q apply_compact_store ${WORKDIR}/checkpointing.py; then
    echo "FATAL: ${WORKDIR} lacks the nov05 molecule table / compact sidecar -- git pull gfn-diffusion" >&2; exit 1
fi
for ARM in ${GROUP}; do
    if [ ! -f ${ARMS}/${ARM}.yaml ]; then echo "FATAL: missing config ${ARMS}/${ARM}.yaml" >&2; exit 1; fi
    for KEY in prior_path molecules_path test_molecules_path; do
        f=$(awk -v k="${KEY}:" '$1 == k {print $2}' ${ARMS}/${ARM}.yaml)
        if [ -z "${f}" ] || [ ! -s "${f}" ]; then echo "FATAL: ${ARM} ${KEY} '${f}' is missing or empty" >&2; exit 1; fi
    done
    SRC=$(awk '/^checkpoint_name:/{print $2}' ${ARMS}/${ARM}.yaml)
    if [ "$(awk '/^continue_from_checkpoint:/{print $2}' ${ARMS}/${ARM}.yaml)" = "true" ] && [ -n "${SRC}" ] && [ "${SRC}" != "null" ]; then
        echo "FATAL: ${ARM} resumes but still names checkpoint_name ${SRC}: it would restart from those weights (make.py --resume)" >&2; exit 1
    fi
    if [ "$(awk '/^continue_from_checkpoint:/{print $2}' ${ARMS}/${ARM}.yaml)" != "true" ]; then
        RUN0=$(awk '/^run_name:/{print $2}' ${ARMS}/${ARM}.yaml)
        if ls ${CKDIR}/pool4_nov24_${RUN0}_*_running.pt > /dev/null 2>&1; then
            echo "FATAL: ${ARM} starts fresh, but ${CKDIR} already holds a running checkpoint of pool4_nov24_${RUN0}: resume it (make.py --resume) or rename the run" >&2; exit 1
        fi
    fi
    if [ "$(awk '/^continue_from_checkpoint:/{print $2}' ${ARMS}/${ARM}.yaml)" = "true" ]; then
        RUN=$(awk '/^run_name:/{print $2}' ${ARMS}/${ARM}.yaml)
        if ! ls ${CKDIR}/pool4_nov24_${RUN}_*_running.pt > /dev/null 2>&1; then
            echo "FATAL: ${ARM} resumes run pool4_nov24_${RUN}, but ${CKDIR} has no running checkpoint for it" >&2; exit 1
        fi
    elif [ -n "${SRC}" ] && [ "${SRC}" != "null" ] && [ ! -s ${CKDIR}/${SRC} ]; then
        echo "FATAL: ${ARM} loads weights from ${CKDIR}/${SRC}, which is missing" >&2; exit 1
    fi
    cp ${ARMS}/${ARM}.yaml ${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
done

J=${LOGS}/pool4_nov24_${SLURM_JOB_ID}
{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
  echo "arms: ${GROUP}"
} > ${J}.info 2>&1

stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
( while sleep 60; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used " $3 " GB, available " $7 " GB"}')"; done ) > ${J}_ram.txt &
RAM_PID=$!
epilogue() {
    kill ${SMI_PID} ${RAM_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList,Reason,Comment%64 > ${J}_sacct.txt 2>&1
}
trap epilogue EXIT TERM

set -o pipefail   # the job status is the container's, not tee's
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /tmp:/tmp \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:${WORKDIR}:\$PYTHONPATH
        export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        bash ${GROUP_SH} ${LOGS} ${SLURM_JOB_ID} ${J}.share ${GROUP}
    " 2>&1 | tee ${J}.trainlog

for ARM in ${GROUP}; do
    echo "===== ${ARM}: last lines"
    tail -n 25 ${LOGS}/${ARM}_${SLURM_JOB_ID}.trainlog 2>/dev/null
done
