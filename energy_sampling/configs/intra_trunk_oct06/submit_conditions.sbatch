#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=tc06
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/intra_trunk_oct06/joblogs/%x_%j.out

# intra_trunk_oct06, third stage: the molecule condition read off the intra trunk (build_trunk_conditions.py).
#
# Writes copies of the full-QM9 prior and condition files whose `embedding` is the pooled per-atom states of the
# intra trunk inside the stacked trunk ([mean, sum / 16], width 256) instead of the Mo3ENet vector, into
# ${DATA}/trunkcond_st06/ under the same file names. A run takes the trunk's condition by pointing prior_path,
# molecules_path and test_molecules_path there and setting embedding_conditioning_dim: 256.
#
# RUN AFTER submit_inter.sbatch HAS WRITTEN st06_c2_ft_late.pt (TRUNK= names another stacked checkpoint). The
# intra trunk is frozen inside every stacked checkpoint, so st06_c2_w32.pt gives the same conditions.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
TRUNK=${WORKDIR}/checkpoints/intra_trunk_oct06/${TRUNK:-st06_c2_ft_late}.pt
OUT=${DATA}/trunkcond_st06
FILES="${DATA}/qm9full_prior.pt ${DATA}/qm9full_conditions.pt ${DATA}/qm9full_test_conditions.pt"

for F in ${FILES} ${TRUNK}; do
    if [ ! -s "${F}" ]; then echo "FATAL: ${F} is missing or empty" >&2; exit 1; fi
done
if [ ! -f ${WORKDIR}/build_trunk_conditions.py ]; then
    echo "FATAL: ${WORKDIR}/build_trunk_conditions.py is missing -- git pull gfn-diffusion" >&2; exit 1
fi
mkdir -p ${OUT}
echo "JOB ${SLURM_JOB_ID} conditions from ${TRUNK} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PROJECT_ROOT}/gfn-diffusion:${PYLIBS}:\$PYTHONPATH
        python -u build_trunk_conditions.py --trunk ${TRUNK} --out-dir ${OUT} --device cuda ${FILES}
    "
RC=$?
echo "JOB ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
