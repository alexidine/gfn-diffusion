#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=compprobe
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prof_sep07/joblogs/%x_%j.out

# THE DECIDING MEASUREMENT: does compiling the step body actually fuse the
# elementwise SDE math? The full-batch run OOM'd before it could say -- compiling
# `_fwd_step` defeats gradient checkpointing, because AOTAutograd saves the compiled
# region's activations and the outer torch.utils.checkpoint cannot discard them, so
# all T=100 timesteps are retained (64-75 GiB of an 80 GiB card).
#
# BATCH 128 SIDESTEPS THAT ENTIRELY, and costs nothing in fidelity: kernel launch
# count is a property of the traced GRAPH, not the data. Measured identical at batch
# 4000 (ELJ) and 1600 (UMA): 116,510 both times. So a batch that fits answers the
# only open question.
#
# TWO PASSES, and the second is the control. traj_checkpoint TRUE is production
# shape and the configuration that OOM'd; FALSE removes the checkpoint/compile fight
# so we see what the fusion does on its own. Comparing the two also prices what
# gradient checkpointing itself costs in launches (it re-runs the forward, so eager
# should show roughly a third fewer without it).
#
# --gres=gpu:1 not gpu:a100:1 ON PURPOSE: launch counts are hardware-independent, so
# this runs anywhere and does not wait on the contended a100_chemistry pool. Pass
# --partition=l40s_public (or h200_public) on the sbatch line to pick a freer one.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
CKPTS=${WORKDIR}/checkpoints

MIP=$(ls ${CKPTS}/*pt100_mip_lr4p0_*_phase1_exit.pt 2>/dev/null | head -1)
if [ -z "${MIP}" ]; then echo "FATAL: no mip phase-1 exit checkpoint" >&2; exit 1; fi
echo "NODE: $(hostname)   PARTITION: ${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "checkpoint: $(basename ${MIP})"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        echo '=============================== batch 128, traj_checkpoint TRUE (production shape)'
        python -u -m bench.compile_rollout --checkpoint ${MIP} --modes eager,auto,step \
            --batch 128 --T 100 --reps 3 --traj-checkpoint true \
            --out bench/results/probe_ckpt_on_\${SLURM_JOB_ID}.json
        echo
        echo '=============================== batch 128, traj_checkpoint FALSE (control)'
        python -u -m bench.compile_rollout --checkpoint ${MIP} --modes eager,step \
            --batch 128 --T 100 --reps 3 --traj-checkpoint false \
            --out bench/results/probe_ckpt_off_\${SLURM_JOB_ID}.json
    "
