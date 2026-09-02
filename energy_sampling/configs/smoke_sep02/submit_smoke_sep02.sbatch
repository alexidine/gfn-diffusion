#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=smoke02
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/smoke_sep02/joblogs/%x_%A_%a.out

# SMOKE, 20 minutes. One arm per system, carrying the production battery's shape.
# Everything it exists to catch -- an energy_config key the committed constructor
# rejects, a missing module, an unresolvable checkpoint glob, a config the schema
# refuses, a joblogs dir SLURM cannot create -- lands within seconds, before the
# MLIP initialises. So a wall shorter than a UMA prior scan still answers the
# question. Arm = row of INDEX.tsv (line 1 is the header).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/smoke_sep02
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $2}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# A SMOKE ALWAYS SEEDS. It deliberately does NOT glob its own _running.pt: a
# second smoke must re-test the seed path, not silently resume the first one.
N=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt 2>/dev/null | wc -l)
if [ "${N}" -eq 0 ]; then
    echo "FATAL: no phase-1 exit matches *${SRC}_*_phase1_exit.pt in ${CKPTS}" >&2
    ls ${CKPTS}/ | head -40 >&2
    exit 1
fi
if [ "${N}" -gt 1 ]; then
    echo "FATAL: ${N} checkpoints match *${SRC}_*_phase1_exit.pt -- ambiguous:" >&2
    ls ${CKPTS}/*${SRC}_*_phase1_exit.pt >&2
    exit 1
fi
CK=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt)
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  seed <- $(basename ${CK})"
sed "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" ${CONFIG} > ${RESOLVED}

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u train.py --config ${RESOLVED}
    "
