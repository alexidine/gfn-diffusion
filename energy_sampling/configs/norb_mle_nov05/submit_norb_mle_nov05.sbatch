#!/bin/bash
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --array=0-5
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=norbmle
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/norb_mle_nov05/joblogs/%x_%A_%a.out

# norb_mle_nov05 (make.py, from submit_template.sbatch): conditional MLE knob sweep on the norbornane P-1 set, 24 arms,
# 4 per A100 under MPS (configs/cap_size_nov01/run_group.sh). Fresh launches, MLE only, 20k steps each.
#   array 0: base (the base recipe); base_s2 (the base, another seed (the noise floor)); lr_x01 (final LR 0.1 x seed_lr (1.25e-5)); lr_x04 (final LR 0.4 x seed_lr (5e-5))
#   array 1: lr_x08 (final LR 0.8 x seed_lr (1e-4)); wd_1e4 (Adam weight decay 1e-4 (L2, every group)); wd_1e3 (Adam weight decay 1e-3); batch_250 (batch 250 (accumulated to the same 1000-row update))
#   array 2: pooled_mol (molecule as pooled vectors only, no cross-attention to atoms); atom_d64 (atom encoder node width 64 (base 128)); atom_d256 (atom encoder node width 256); atom_c1 (atom encoder 1 message-passing round (base 3))
#   array 3: atom_c5 (atom encoder 5 rounds); window_15 (data: minima within 15 kT (more minima per molecule; base 10)); tf_w128 (transformer width 128 (base 256)); tf_w512 (transformer width 512)
#   array 4: tf_l2 (transformer 2 layers (base 4)); tf_l8 (transformer 8 layers); norm_rms (RMS norm in the transformer (base layer norm)); ctx_256 (condition context width 256 (base 128))
#   array 5: rows_typed (typed FD rows without the prefix physics features); bins_16 (spline 16 bins (base 32)); bins_64 (spline 64 bins); lr_nohold (final LR 0.2 x seed_lr from step 0, no 3k-step warm hold)
# Read: eval_test/mle and eval_test/mle_gap (between molecules), eval_bwd/mle_heldout_gap (within), against base/base_s2.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/norb_mle_nov05
GROUP_SH=${WORKDIR}/configs/cap_size_nov01/run_group.sh
LOGS=${ARMS}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/norbornane_nov04/priors
mkdir -p ${LOGS}

case "${SLURM_ARRAY_TASK_ID}" in
    0) GROUP="base base_s2 lr_x01 lr_x04" ;;
    1) GROUP="lr_x08 wd_1e4 wd_1e3 batch_250" ;;
    2) GROUP="pooled_mol atom_d64 atom_d256 atom_c1" ;;
    3) GROUP="atom_c5 window_15 tf_w128 tf_w512" ;;
    4) GROUP="tf_l2 tf_l8 norm_rms ctx_256" ;;
    5) GROUP="rows_typed bins_16 bins_64 lr_nohold" ;;
    *) echo "FATAL: no group for array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

FILES=(norbornane_p1_train_k100_metric.pt norbornane_p1_train_conditions.pt norbornane_p1_test_w10.pt norbornane_p1_train_k100_metric.pt.fdchart-80b2454bf1fe9638.pt)
BYTES=(2328032453 332153 4378129 46210165)
for i in "${!FILES[@]}"; do
    HAVE=$(stat -c %s ${DATA}/${FILES[$i]} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${BYTES[$i]}" ]; then
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

J=${LOGS}/norb_mle_nov05_${SLURM_JOB_ID}
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
