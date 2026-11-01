#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --array=0-1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=capnov01
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/cap_size_nov01/joblogs/%x_%A_%a.out

# cap_size_nov01 (make.py): conditional NF MLE on the rc_oct22 anchors, crystals per molecule x model size.
#   array 0: cap_nov01_w768_k2  cap_nov01_w768_k4  cap_nov01_w768_k10    (W768 L8 H12, the local zo2jwtz4 model)
#   array 1: cap_nov01_w1024_k2 cap_nov01_w1024_k4 cap_nov01_w1024_k10   (W1024 L12 H16)
# Three arms on one A100 (80 GB) under MPS, cuda_memory_fraction 0.30 each. Fresh launches only (no resume).
# Read: eval_test/mle against eval_bwd/mle_mol_crystals (eval_test/mle_gap), eval_fwd/eval_test jensen_z,
# Excess Energy Nats P50, data/mle_passes; compare arms at equal passes.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/cap_size_nov01
LOGS=${ARMS}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors/rc_oct22
mkdir -p ${LOGS}

case "${SLURM_ARRAY_TASK_ID}" in
    0) GROUP="cap_nov01_w768_k2 cap_nov01_w768_k4 cap_nov01_w768_k10" ;;
    1) GROUP="cap_nov01_w1024_k2 cap_nov01_w1024_k4 cap_nov01_w1024_k10" ;;
    *) echo "FATAL: no group for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

FILES=(rcoct22i1_prior.pt rcoct22i1_conditions.pt rcoct22i1_qm9rc_test_conditions.pt)
BYTES=(13156215839 1357117492 13259723)
for i in "${!FILES[@]}"; do
    HAVE=$(stat -c %s ${DATA}/${FILES[$i]} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${BYTES[$i]}" ]; then
        echo "FATAL: ${DATA}/${FILES[$i]} is ${HAVE} bytes, expected ${BYTES[$i]}" >&2; exit 1
    fi
done
SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py
if grep -q 'MXT_NIGGLI_TRICLINIC is retired' ${SYM}; then
    NIG_EXPORT=""
elif grep -q "os.environ.get('MXT_NIGGLI_TRICLINIC'" ${SYM}; then
    NIG_EXPORT="export MXT_NIGGLI_TRICLINIC=1;"
else
    echo "FATAL: ${SYM} has no triclinic Niggli penalty -- git pull MXtalTools" >&2; exit 1
fi
for F in run_group.sh thin_prior.py; do
    if [ ! -f ${ARMS}/${F} ]; then echo "FATAL: ${ARMS}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
for ARM in ${GROUP}; do
    if [ ! -f ${ARMS}/${ARM}.yaml ]; then echo "FATAL: missing config ${ARMS}/${ARM}.yaml" >&2; exit 1; fi
    cp ${ARMS}/${ARM}.yaml ${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
done

J=${LOGS}/capnov01_${SLURM_JOB_ID}
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
# inside one container: the cap-2 and cap-4 priors (written once; the second job finds them), then the three arms
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
        ${NIG_EXPORT}
        export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1
        CUDA_VISIBLE_DEVICES=-1 python -u ${ARMS}/thin_prior.py ${DATA}/rcoct22i1_prior.pt ${DATA}/caps_nov01 || { echo 'FATAL: thin_prior.py failed' >&2; exit 1; }
        bash ${ARMS}/run_group.sh ${LOGS} ${SLURM_JOB_ID} ${J}.share ${GROUP}
    " 2>&1 | tee ${J}.trainlog

for ARM in ${GROUP}; do
    echo "===== ${ARM}: last lines"
    tail -n 25 ${LOGS}/${ARM}_${SLURM_JOB_ID}.trainlog 2>/dev/null
done
