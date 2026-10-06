#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=it06
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/intra_trunk_oct06/joblogs/%x_%j.out

# intra_trunk_oct06: pre-train the intra trunk (models/intra_trunk.py, pretrain_intra_trunk.py).
#
# One molecule's atoms as a typed point cloud, every atom reading every other; per-atom energies whose sum is
# fitted to MMFF94's intramolecular energy and whose gradient is fitted to its Cartesian force, on the QM9
# molecules of the crystal set (build_intra_energy_set.py: up to two relaxed conformers a molecule, each with
# copies displaced by 0.02, 0.05, 0.1 and 0.2 A; 128,714 molecules, 1,050,960 geometries). The crystal work's
# held-out molecules are held out here too. Units are the usual ones of a fitted potential: a reference energy
# per element, one scale from the force RMS, both stored in the checkpoint.
#
# One run, it06_mmff: three stages, node width 128, message width 64, 60,000 steps at batch 1024, rate 5e-4 with
# a cosine to 1e-5, averaged weights. Each evaluation prints one table: force and energy error by displacement
# width for held-out and training molecules, and the energy difference between two conformers.
#
# RESUBMIT THIS FILE to continue past the wall: the run resumes its own it06_mmff_running.pt, and only under
# the arguments it was written with (a longer schedule is a new TAG, not STEPS=).
# Environment: TAG (default it06_mmff), STEPS (60000), BATCH (1024), LR (5e-4), EVAL_EVERY (5000).
# Smoke: TAG=it06_smoke STEPS=400 EVAL_EVERY=200 sbatch --time=00:30:00 <this file>
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
SET=/scratch/mk8347/data/crystal_datasets/conditional/priors/qm9full_intra_mmff94.pt
SET_BYTES=521855317
OUT=${WORKDIR}/checkpoints/intra_trunk_oct06
TAG=${TAG:-it06_mmff}

HAVE=$(stat -c %s ${SET} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${SET_BYTES}" ]; then
    echo "FATAL: ${SET} is ${HAVE} bytes, expected ${SET_BYTES} -- upload unfinished or a different file" >&2; exit 1
fi
for F in pretrain_intra_trunk.py models/intra_trunk.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
mkdir -p ${OUT}
echo "JOB ${SLURM_JOB_ID} ${TAG} start=$(date -Is) steps=${STEPS:-60000} batch=${BATCH:-1024}"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u pretrain_intra_trunk.py --set ${SET} --out ${OUT} --tag ${TAG} \
            --steps ${STEPS:-60000} --batch ${BATCH:-1024} --lr ${LR:-5e-4} --eval-every ${EVAL_EVERY:-5000} \
            --device cuda
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} ${TAG} status=${RC} end=$(date -Is)"
exit ${RC}
