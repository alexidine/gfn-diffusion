#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=compbench
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prof_sep07/joblogs/%x_%j.out

# THE FAST TEST: does compiling the fused per-timestep kernels actually remove
# compiled-region entries, and does it pay? bench/compile_rollout.py builds only the
# policy -- no MLIP, no data, no optimizer -- so this answers in minutes what a
# training arm needs a 3 h wall for.
#
# DELIBERATELY SMALL AND SHORT so it backfills instead of queueing behind the
# batteries: 25 minutes, 4 CPUs, 32G, one GPU. The interactive version of this kept
# waiting on QOSGrpGRES behind the submitter's own training arms.
#
# Runs BOTH routes. The host chunk is route-independent (utilization x step_time is
# constant on every arm), so if ELJ and UMA disagree about which mode wins, the
# mechanism is not what we think it is.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
CKPTS=${WORKDIR}/checkpoints
LOGS=${WORKDIR}/configs/prof_sep07/joblogs
mkdir -p ${LOGS}

MIP=$(ls ${CKPTS}/*pt100_mip_lr4p0_*_phase1_exit.pt 2>/dev/null | head -1)
MIPU=$(ls ${CKPTS}/*pt100_mipu_lr4p0_*_phase1_exit.pt 2>/dev/null | head -1)
if [ -z "${MIP}" ] || [ -z "${MIPU}" ]; then
    echo "FATAL: could not resolve both phase-1 exit checkpoints" >&2
    ls ${CKPTS}/*phase1_exit.pt >&2; exit 1
fi
echo "ELJ  checkpoint: $(basename ${MIP})"
echo "UMA  checkpoint: $(basename ${MIPU})"
nvidia-smi -L

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        echo '================================================== ELJ (mip), batch 4000'
        python -u -m bench.compile_rollout --checkpoint ${MIP}  --modes eager,auto,step \
            --batch 4000 --T 100 --reps 5 --out bench/results/compile_rollout_mip.json
        echo
        echo '================================================== UMA (mipu), batch 1600'
        python -u -m bench.compile_rollout --checkpoint ${MIPU} --modes eager,auto,step \
            --batch 1600 --T 100 --reps 5 --out bench/results/compile_rollout_mipu.json
    "
