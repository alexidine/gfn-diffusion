#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-1
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=atrunk
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/atom_trunk_oct05/joblogs/%x_%A_%a.out

# atom_trunk_oct05: offline pre-training of the atom trunk (models/atom_trunk.py) with pretrain_atom_trunk.py.
# No trainer and no rollout. One task = one trunk, fitted to the 10 A compressed-eLJ energy and its force on the
# latent, from 5 A pair features, on every kept crystal of every full-QM9 training molecule (qm9full_prior.pt,
# key 'prior') walked back along the reference bridge at 1x, 2x and 4x its variance; scored on the run's held-out
# molecules (qm9full_test_conditions.pt). Energy units and the bridge are the conditional run's (kT 6.9, lj_coeff
# 1.0, T 50, t_scale 0.05: the script's defaults).
#   task 0  at_c2_w32  two message-passing stages, 32-wide messages
#   task 1  at_c3_w32  three stages, 32-wide messages
# Environment: BATCH (default 256), STEPS (default 40000), EVAL_EVERY (default 5000). A checkpoint and the
# evaluation json are written at every evaluation; there is no resume, so a job that hits the wall clock keeps
# its last evaluated checkpoint. Smoke: STEPS=200 EVAL_EVERY=100 sbatch --array=0 <this file>.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
OUT=${WORKDIR}/checkpoints/atom_trunk_oct05

case ${SLURM_ARRAY_TASK_ID} in
    0) TAG=at_c2_w32; CONVS=2 ;;
    1) TAG=at_c3_w32; CONVS=3 ;;
    *) echo "no arm at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

for f in qm9full_prior.pt qm9full_test_conditions.pt; do
    if [ ! -s "${DATA}/${f}" ]; then echo "FATAL: ${DATA}/${f} is missing or empty" >&2; exit 1; fi
done
for f in ${WORKDIR}/pretrain_atom_trunk.py ${WORKDIR}/models/atom_trunk.py ${MXT_ROOT}/mxtaltools/crystal_building/image_pairs.py; do
    if [ ! -s "${f}" ]; then echo "FATAL: ${f} is missing (pull BOTH repositories)" >&2; exit 1; fi
done
mkdir -p ${OUT}
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${TAG} start=$(date -Is) batch=${BATCH:-256} steps=${STEPS:-40000}"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u pretrain_atom_trunk.py --arm trunk --out ${OUT} --tag ${TAG} \
            --conditions ${DATA}/qm9full_prior.pt --split-key prior \
            --test-conditions ${DATA}/qm9full_test_conditions.pt \
            --target full --feature-cutoff 5 --num-convs ${CONVS} --message-dim 32 --noise-scales 1,2,4 \
            --batch ${BATCH:-256} --steps ${STEPS:-40000} --eval-every ${EVAL_EVERY:-5000} --eval-molecules 1024 \
            --device cuda --vram-frac 0.95
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${TAG} status=${RC} end=$(date -Is)"
exit ${RC}
