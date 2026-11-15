#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --partition=a100_chemistry,l40s_public
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8big
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/@TAG@/joblogs/%x_%j.out

# @TAG@ (make.py, from submit_template.sbatch): one unaugmented conditional MLE run on the pool8 P-1 noised set, a
# larger NF model, usage-targeted batch growth (batch_util_target). Any GPU type on the listed partitions; one run
# per card (configs/cap_size_nov01/run_group.sh, solo). Fresh launch, MLE only.
@GROUPS_DOC@
# Read: eval_test/mle and eval_test/mle_gap (between molecules), eval_bwd/mle_heldout_gap (within), against base/base_s2.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/@TAG@
GROUP_SH=${WORKDIR}/configs/cap_size_nov01/run_group.sh
LOGS=${ARMS}/joblogs
DATA=@DATA@
mkdir -p ${LOGS}

GROUP="@CASES@"

FILES=(@FILES@)
BYTES=(@BYTES@)
for i in "${!FILES[@]}"; do
    HAVE=$(stat -c %s ${DATA}/${FILES[$i]} 2>/dev/null || echo 0)
    if [ "${BYTES[$i]}" = "-1" ]; then   # built on the cluster: any complete (non-empty) file
        if [ "${HAVE}" = "0" ]; then echo "FATAL: ${DATA}/${FILES[$i]} is missing or empty" >&2; exit 1; fi
    elif [ "${HAVE}" != "${BYTES[$i]}" ]; then
        echo "FATAL: ${DATA}/${FILES[$i]} is ${HAVE} bytes, expected ${BYTES[$i]}" >&2; exit 1
    fi
done
if [ ! -s ${GROUP_SH} ]; then echo "FATAL: ${GROUP_SH} is missing" >&2; exit 1; fi
if ! grep -q _molecule_table ${WORKDIR}/models/nf_policy.py || ! grep -q apply_compact_store ${WORKDIR}/checkpointing.py; then
    echo "FATAL: ${WORKDIR} lacks the nov05 molecule table / compact sidecar -- git pull gfn-diffusion" >&2; exit 1
fi
for ARM in ${GROUP}; do
    if [ ! -f ${ARMS}/${ARM}.yaml ]; then echo "FATAL: missing config ${ARMS}/${ARM}.yaml" >&2; exit 1; fi
    cp ${ARMS}/${ARM}.yaml ${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
done

J=${LOGS}/@TAG@_${SLURM_JOB_ID}
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
