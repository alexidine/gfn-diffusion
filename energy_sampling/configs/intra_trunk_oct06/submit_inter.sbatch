#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=st06
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/intra_trunk_oct06/joblogs/%x_%j.out

# intra_trunk_oct06, second stage: the crystal energy trunk STACKED on the pre-trained intra trunk
# (models/stacked_trunk.py, pretrain_atom_trunk.py --intra).
#
# The intra trunk of submit_pretrain.sbatch (it06_mmff.pt) is loaded frozen; its per-atom states are each atom's
# input beside the element, and the fitted stages pass messages over intermolecular pairs only. Target, data,
# bridge states and tables are atom_trunk_oct05's (the compressed eLJ energy inside 10 A, in kT at temperature
# 6.9, on bridge states of qm9full_prior at 1x, 2x and 4x the reference variance, held-out molecules from
# qm9full_test_conditions).
#
# The schedule is the one that produced at_c2_ft_late, in one job, two runs:
#   st06_c2_w32      40,000 steps at batch 256, rate 1e-3: two stages, 32-wide messages (as at_c2_w32)
#   st06_c2_ft_late  continued from it for 60,000 steps at rate 1e-4, half of the states on t/T in [0.9, 1]
# Read st06_c2_ft_late's last table against at_c2_ft_late's (force cosine 0.9996 to 0.9998 on held-out molecules,
# error 0.09 to 0.13 noise standard deviations a step): the stack is accepted if it is no worse.
#
# RUN AFTER submit_pretrain.sbatch HAS WRITTEN it06_mmff.pt; this job refuses to start without it.
# No resume. A job that hits the wall clock keeps each run's last evaluated checkpoint; ONLY=late runs the
# second run alone from an existing st06_c2_w32.pt.
# Environment: INTRA (default it06_mmff), BATCH (256), STEPS_A (40000), STEPS_B (60000), EVAL_EVERY (5000), ONLY.
# Smoke: SMOKE=1 sbatch --time=01:00:00 <this file>   (200 + 200 steps, writes st06_smoke_*)
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
OUT=${WORKDIR}/checkpoints/intra_trunk_oct06
INTRA=${OUT}/${INTRA:-it06_mmff}.pt
TAG_A=st06_c2_w32
TAG_B=st06_c2_ft_late
STEPS_A=${STEPS_A:-40000}
STEPS_B=${STEPS_B:-60000}
EVAL_EVERY=${EVAL_EVERY:-5000}
if [ -n "${SMOKE}" ]; then TAG_A=st06_smoke_w32; TAG_B=st06_smoke_ft_late; STEPS_A=200; STEPS_B=200; EVAL_EVERY=100; fi

for F in ${DATA}/qm9full_prior.pt ${DATA}/qm9full_test_conditions.pt; do
    if [ ! -s "${F}" ]; then echo "FATAL: ${F} is missing or empty" >&2; exit 1; fi
done
if [ ! -s "${INTRA}" ]; then
    echo "FATAL: ${INTRA} is missing -- the intra trunk (submit_pretrain.sbatch) has not written its checkpoint" >&2; exit 1
fi
if ! grep -q -- "--intra" ${WORKDIR}/pretrain_atom_trunk.py; then
    echo "FATAL: ${WORKDIR}/pretrain_atom_trunk.py has no --intra option (pull gfn-diffusion)" >&2; exit 1
fi
if ! grep -q 'max_pairs' ${MXT_ROOT}/mxtaltools/crystal_building/image_pairs.py; then
    echo "FATAL: MXtalTools' image_pairs.py is out of date -- git pull MXtalTools" >&2; exit 1
fi
if [ "${ONLY}" = "late" ] && [ ! -s "${OUT}/${TAG_A}.pt" ]; then
    echo "FATAL: ONLY=late needs ${OUT}/${TAG_A}.pt" >&2; exit 1
fi
mkdir -p ${OUT}
echo "JOB ${SLURM_JOB_ID} ${TAG_A} then ${TAG_B} on ${INTRA} start=$(date -Is) batch=${BATCH:-256}"

COMMON="--arm trunk --out ${OUT} --intra ${INTRA} \
    --conditions ${DATA}/qm9full_prior.pt --split-key prior --test-conditions ${DATA}/qm9full_test_conditions.pt \
    --target full --feature-cutoff 5 --num-convs 2 --message-dim 32 --noise-scales 1,2,4 \
    --eval-positions 1,0.99,0.98,0.96,0.94,0.9,0.8,0.5,0.2,0 \
    --batch ${BATCH:-256} --eval-every ${EVAL_EVERY} --eval-molecules 1024 --device cuda --vram-frac 0.95"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        if [ \"${ONLY}\" != \"late\" ]; then
            python -u pretrain_atom_trunk.py ${COMMON} --tag ${TAG_A} --lr 1e-3 --steps ${STEPS_A} || exit 1
        fi
        python -u pretrain_atom_trunk.py ${COMMON} --tag ${TAG_B} --init ${OUT}/${TAG_A}.pt \
            --late-share 0.5 --late-from 0.9 --lr 1e-4 --steps ${STEPS_B}
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} ${TAG_B} status=${RC} end=$(date -Is)"
exit ${RC}
