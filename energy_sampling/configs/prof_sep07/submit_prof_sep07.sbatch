#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --array=0-1
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=prof07
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prof_sep07/joblogs/%x_%A_%a.out

# PROFILE, 3 hours, two arms of the prod_sep02 shape with the torch.profiler window
# armed in steady state. See make.py. Arm = row of INDEX.tsv (line 1 is the header).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/prof_sep07
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

# SEED from the same phase-1 exit the battery used; refuse an ambiguous match.
N=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt 2>/dev/null | wc -l)
if [ "${N}" -eq 0 ]; then echo "FATAL: no phase-1 exit matches *${SRC}_*_phase1_exit.pt" >&2; exit 1; fi
if [ "${N}" -gt 1 ]; then
    echo "FATAL: ${N} ambiguous matches for *${SRC}_*_phase1_exit.pt:" >&2
    ls ${CKPTS}/*${SRC}_*_phase1_exit.pt >&2; exit 1
fi
CK=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt)
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK})"

# fresh run_name, so leg 1 writes the prior model itself
sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|null|" ${CONFIG} > ${RESOLVED}

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  scontrol show node ${SLURM_NODELIST}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

# THE OCCUPANCY RECORD THAT MATTERS, same sidecar as the battery.
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
smi_epilogue() {
    kill ${SMI_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${J}_sacct.txt 2>&1
    # WHO ELSE WAS ON THE NODE. This is the question the wandb stream could not
    # answer for prod_sep02 (a 48-CPU CPU-only job on partition `all` landed on
    # ga036 the minute mipu_lr0p125 slowed); record it beside the log every time.
    sacct -N ${SLURM_NODELIST} -a -X -S "$(date -d '-4 hours' +%FT%T)" \
        --format=JobID,JobName%20,User,Partition,Start,End,NCPUS,ReqMem,State \
        > ${J}_cotenants.txt 2>&1
    # THE PROFILE, beside the log and echoed into the .out so it can be pasted.
    for f in ${WORKDIR}/profiling_results/${ARM}_step*; do
        [ -e "$f" ] && cp "$f" ${LOGS}/ && echo "profile artifact: $(basename $f)"
    done
    for t in ${LOGS}/${ARM}_step*_table.txt; do
        [ -e "$t" ] && { echo "===== ${t}"; cat "$t"; }
    done
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
