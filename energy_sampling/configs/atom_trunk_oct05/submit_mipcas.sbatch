#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=atmip
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/atom_trunk_oct05/joblogs/%x_%j.out

# atom_trunk_oct05, MIPCAS: an atom trunk for the unconditional MIPCAS eLJ runs (mle_pool_oct05's mlepl_mip_lr2:
# temperature 2.5, lj_coeff 1.0, 100 trajectory steps, t_scale 0.05). The QM9 trunks were fitted at temperature 6.9,
# and the target is compressed per atom in kT, so they do not serve another temperature. One run, two stages and
# 32-wide messages as at_c2_w32, on the pooled MIPCAS prior walked back along the reference bridge at 1x, 2x and
# 4x its variance, a quarter of the states drawn on t/T in [0.9, 1]. One molecule: the held-out split is 8192
# CRYSTALS of the same file (--holdout-rows), 1024 of them scored.
# Environment: BATCH (default 256), STEPS (default 60000), EVAL_EVERY (default 5000), LR (default 1e-3).
# No resume: a job that hits the wall clock keeps its last evaluated checkpoint.
# Smoke: STEPS=200 EVAL_EVERY=100 sbatch --time=00:30:00 <this file>.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
OUT=${WORKDIR}/checkpoints/atom_trunk_oct05
PRIOR=${DATA}/mipcas_elj_pooled_oct05_prior.pt
TAG=at_mip_elj_t25

if [ ! -s "${PRIOR}" ]; then echo "FATAL: ${PRIOR} is missing or empty" >&2; exit 1; fi
if ! grep -q -- "--holdout-rows" ${WORKDIR}/pretrain_atom_trunk.py; then
    echo "FATAL: ${WORKDIR}/pretrain_atom_trunk.py has no --holdout-rows option (pull gfn-diffusion)" >&2; exit 1
fi
mkdir -p ${OUT}
echo "JOB ${SLURM_JOB_ID} ${TAG} start=$(date -Is) batch=${BATCH:-256} steps=${STEPS:-60000}"

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
            --conditions ${PRIOR} --split-key prior --holdout-rows 8192 --eval-molecules 1024 \
            --temperature 2.5 --lj-coeff 1.0 --T 100 --t-scale 0.05 \
            --target full --feature-cutoff 5 --num-convs 2 --message-dim 32 --noise-scales 1,2,4 \
            --late-share 0.25 --late-from 0.9 --eval-positions 1,0.99,0.98,0.96,0.94,0.9,0.8,0.5,0.2,0 \
            --lr ${LR:-1e-3} --batch ${BATCH:-256} --steps ${STEPS:-60000} --eval-every ${EVAL_EVERY:-5000} \
            --device cuda --vram-frac 0.95
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} ${TAG} status=${RC} end=$(date -Is)"
exit ${RC}
