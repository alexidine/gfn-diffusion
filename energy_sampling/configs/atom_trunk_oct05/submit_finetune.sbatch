#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=atft
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/atom_trunk_oct05/joblogs/%x_%A_%a.out

# atom_trunk_oct05, continuation: the two-stage trunk of submit_pretrain.sbatch (at_c2_w32.pt, 40,000 steps) is
# continued with pretrain_atom_trunk.py --init, same data, target and bridge multipliers, learning rate 1e-4 (twice
# the rate that run ended on).
#   task 0  at_c2_ft       unchanged sampling of trajectory positions (the T + 1 grid)
#   task 1  at_c2_ft_late  half of the training states drawn on t/T in [0.9, 1]
# Both score the checkpoint before any update and then every EVAL_EVERY steps, at ten trajectory positions of which
# five lie in [0.9, 1); the tables add the energy above the stored crystal and the force and its error in units of
# one step's noise standard deviation.
# Environment: BATCH (default 256), STEPS (default 60000), EVAL_EVERY (default 5000), LR (default 1e-4).
# No resume: a job that hits the wall clock keeps its last evaluated checkpoint.
# Smoke: STEPS=200 EVAL_EVERY=100 sbatch --time=00:30:00 --array=1 <this file>.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
OUT=${WORKDIR}/checkpoints/atom_trunk_oct05
INIT=${OUT}/at_c2_w32.pt

case ${SLURM_ARRAY_TASK_ID} in
    0) TAG=at_c2_ft; LATE=0 ;;
    1) TAG=at_c2_ft_late; LATE=0.5 ;;
    *) echo "no arm at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

for f in ${DATA}/qm9full_prior.pt ${DATA}/qm9full_test_conditions.pt ${INIT}; do
    if [ ! -s "${f}" ]; then echo "FATAL: ${f} is missing or empty" >&2; exit 1; fi
done
if ! grep -q -- "--init" ${WORKDIR}/pretrain_atom_trunk.py; then
    echo "FATAL: ${WORKDIR}/pretrain_atom_trunk.py has no --init option (pull gfn-diffusion)" >&2; exit 1
fi
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${TAG} start=$(date -Is) batch=${BATCH:-256} steps=${STEPS:-60000} late_share=${LATE}"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u pretrain_atom_trunk.py --arm trunk --out ${OUT} --tag ${TAG} --init ${INIT} \
            --conditions ${DATA}/qm9full_prior.pt --split-key prior \
            --test-conditions ${DATA}/qm9full_test_conditions.pt \
            --target full --feature-cutoff 5 --num-convs 2 --message-dim 32 --noise-scales 1,2,4 \
            --late-share ${LATE} --late-from 0.9 --eval-positions 1,0.99,0.98,0.96,0.94,0.9,0.8,0.5,0.2,0 \
            --lr ${LR:-1e-4} --batch ${BATCH:-256} --steps ${STEPS:-60000} --eval-every ${EVAL_EVERY:-5000} \
            --eval-molecules 1024 --device cuda --vram-frac 0.95
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${TAG} status=${RC} end=$(date -Is)"
exit ${RC}
