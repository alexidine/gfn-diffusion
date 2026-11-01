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
#SBATCH --job-name=p2lvl
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/p2lvl_nov01/joblogs/%x_%A_%a.out

# p2lvl_nov01 (make.py): phase 2 from cap_size_nov01's w768 cap-2 step-6000 weights, by pooled_center.
#   array 0: p2lvl_nov01_pooledvg  p2lvl_nov01_tbhead
#   array 1: p2lvl_nov01_tbpool    p2lvl_nov01_pooledvg_s2
# Two arms on one A100 (80 GB) under MPS, cuda_memory_fraction 0.45 each. Fresh weights-only launches (no resume).
# Read first: pooled/pooled_mixed_frac 1, cond_draw/eligible_both >= 1250, Batch Size 2500, lr_ctrl rewinds 0;
# then eval_fwd/jensen_z, Excess Energy Nats P50 beside eval_bwd/mle and eval_test/mle, pooled/*_avg, fwd/tb_resid.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/p2lvl_nov01
GROUP_SH=${WORKDIR}/configs/cap_size_nov01/run_group.sh
LOGS=${ARMS}/joblogs
CKPTS=${WORKDIR}/checkpoints
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors/rc_oct22
SOURCE=${CKPTS}/dev_cap_nov01_w768_k2_elj-rcoct22i1_cap2_prior-T6.9-d7ef7b_step6000.pt
mkdir -p ${LOGS}

case "${SLURM_ARRAY_TASK_ID}" in
    0) GROUP="p2lvl_nov01_pooledvg p2lvl_nov01_tbhead" ;;
    1) GROUP="p2lvl_nov01_tbpool p2lvl_nov01_pooledvg_s2" ;;
    *) echo "FATAL: no group for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

FILES=(rcoct22i1_conditions.pt rcoct22i1_qm9rc_test_conditions.pt)
BYTES=(1357117492 13259723)
for i in "${!FILES[@]}"; do
    HAVE=$(stat -c %s ${DATA}/${FILES[$i]} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${BYTES[$i]}" ]; then
        echo "FATAL: ${DATA}/${FILES[$i]} is ${HAVE} bytes, expected ${BYTES[$i]}" >&2; exit 1
    fi
done
for F in ${DATA}/caps_nov01/rcoct22i1_cap2_prior.pt ${SOURCE} ${GROUP_SH}; do
    if [ ! -s ${F} ]; then echo "FATAL: ${F} is missing or empty" >&2; exit 1; fi
done
SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py
if grep -q 'MXT_NIGGLI_TRICLINIC is retired' ${SYM}; then
    NIG_EXPORT=""
elif grep -q "os.environ.get('MXT_NIGGLI_TRICLINIC'" ${SYM}; then
    NIG_EXPORT="export MXT_NIGGLI_TRICLINIC=1;"
else
    echo "FATAL: ${SYM} has no triclinic Niggli penalty -- git pull MXtalTools" >&2; exit 1
fi
if ! grep -q POOLED_CENTERS ${WORKDIR}/protocol.py || ! grep -q _prior_static_flag ${WORKDIR}/train.py; then
    echo "FATAL: ${WORKDIR} lacks pooled_center / buffers.prior_buffer.static -- git pull gfn-diffusion" >&2; exit 1
fi
for ARM in ${GROUP}; do
    if [ ! -f ${ARMS}/${ARM}.yaml ]; then echo "FATAL: missing config ${ARMS}/${ARM}.yaml" >&2; exit 1; fi
    cp ${ARMS}/${ARM}.yaml ${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
done

J=${LOGS}/p2lvl_${SLURM_JOB_ID}
{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
  echo "arms: ${GROUP}"
  echo "source: $(ls -l ${SOURCE})"
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
        ${NIG_EXPORT}
        export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1
        bash ${GROUP_SH} ${LOGS} ${SLURM_JOB_ID} ${J}.share ${GROUP}
    " 2>&1 | tee ${J}.trainlog

for ARM in ${GROUP}; do
    echo "===== ${ARM}: last lines"
    tail -n 25 ${LOGS}/${ARM}_${SLURM_JOB_ID}.trainlog 2>/dev/null
done
