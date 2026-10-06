#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=ct06
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/conformer_trunk_oct06/joblogs/%x_%A_%a.out

# conformer_trunk_oct06: pre-train the atoms-visible conformer energy model (pretrain_conformer_trunk.py,
# models/conformer_trunk.py) on the r20k condition set, level full, MMFF.
#
# The model reads each atom's element and bond counts and each atom pair's distance and bonds-apart, in the
# geometry the state places, and returns per-atom energies; its gradient with respect to the state is its
# energy differentiated through the build. It is fitted to the force-field part of the potential the r20k
# run samples from, under that run's clip, minus each molecule's potential at its reference conformer.
# Condition set, held-out molecules, held-out stored conformers, the three thermal test sets and the energy
# table are energy_model_oct03's (models/energy_probe.py), so an arm here reads against an em03 arm at the
# same step: same data, same target on those test sets, same schedule (300,000 steps at batch 1024, cosine
# 3e-4 -> 3e-6), a different model.
#
# Arm = row of INDEX.tsv (line 1 is the header):
#   0 ct06_e      energy alone, thermal states alone: em03's protocol with this model
#   1 ct06_ef     the gradient fitted as well
#   2 ct06_ef_br  the gradient fitted, and half the training states drawn on the bridge from the reference
#                 conformer to a stored conformer at 1x, 2x and 4x the sampler's reference noise
# Contrasts: 0 v em03 = what seeing atoms buys; 0 v 1 = what fitting the gradient buys; 1 v 2 = what bridge
# states cost on the thermal sets and buy on the bridge sets (lines F of the log). Every arm scores all sets.
#
# RESUBMIT THIS FILE to continue an arm past the wall: each arm resumes its own <arm>_running.pt.
# One arm: sbatch --array=<row> configs/conformer_trunk_oct06/train_conformer_trunk.sbatch
# Smoke: STEPS=400 EVAL_EVERY=200 sbatch --time=00:40:00 --array=2 <this file>   (writes <arm>_smoke_*)
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/conformer_trunk_oct06
LOGS=${ARMS}/joblogs
OUT=${WORKDIR}/checkpoints/conformer_trunk_oct06
RUNG=/scratch/mk8347/data/conformer_datasets/rungs_sep30/r20k
CONFIG=${WORKDIR}/configs/conformer_mle_sep30/mle30_r20k.yaml
mkdir -p ${LOGS} ${OUT}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX.tsv
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${INDEX})
FLAGS=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${INDEX})
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
STEPS=${STEPS:-300000}
EVAL_EVERY=${EVAL_EVERY:-5000}
WANDB="--wandb-project 'GFN Energy'"
if [ "${STEPS}" != "300000" ]; then ARM=${ARM}_smoke; WANDB=""; fi
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# RUNG GUARD: the set the r20k run trains on, built by configs/conformer_mle_sep30/build_r20k.sbatch
for F in manifest.json conditions_train.pt prior_train.pt; do
    if [ ! -s ${RUNG}/${F} ]; then echo "FATAL: ${RUNG}/${F} is missing" >&2; exit 1; fi
done
if [ ! -f ${CONFIG} ]; then echo "FATAL: missing ${CONFIG}" >&2; exit 1; fi
for F in pretrain_conformer_trunk.py models/conformer_trunk.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
RDKIT=$(grep -o '"rdkit": *"[^"]*"' ${RUNG}/manifest.json | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
if [ -z "${RDKIT}" ]; then echo "FATAL: ${RUNG}/manifest.json names no RDKit version" >&2; exit 1; fi

# ONE JOB PER ARM. A second job on an arm resumes the same checkpoint, then both overwrite it and log
# to one wandb run. This job exits if another running job of this name holds the same array row.
ME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
OTHER=$(squeue -h -u ${USER} --name=${SLURM_JOB_NAME} -t RUNNING -o "%K %i" 2>/dev/null \
    | awk -v k=${SLURM_ARRAY_TASK_ID} -v me=${ME} '$1==k && $2!=me {print $2}' | head -1)
if [ -n "${OTHER}" ]; then
    echo "arm ${ARM} is already running as job ${OTHER}; this job exits without starting a second copy. Resubmit once that one has ended." >&2
    exit 0
fi

if [ -f ${OUT}/${ARM}_running.pt ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: ${OUT}/${ARM}_running.pt"
else
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH"
fi
echo "  flags: ${FLAGS}  steps: ${STEPS}"

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:${WORKDIR}:\$PYTHONPATH
        python -c \"import rdkit; from rdkit.Chem import AllChem; assert rdkit.__version__ == '${RDKIT}', 'RDKit %s; the condition set was built under ${RDKIT}' % rdkit.__version__; assert hasattr(AllChem, 'MMFFGetMoleculeForceField'), 'RDKit without MMFF'; import mxtaltools.conformers.builder; import models.conformer_trunk\" || exit 1
        python -u pretrain_conformer_trunk.py --rung ${RUNG} --config ${CONFIG} --out-dir ${OUT} --name ${ARM} \
            ${FLAGS} --batch 1024 --steps ${STEPS} --lr 3e-4 --lr-final 3e-6 --warmup 2000 \
            --noise-range 0.3 2.0 --heldout-frac 0.10 --conformer-frac 0.10 \
            --eval-samples 20000 --eval-every ${EVAL_EVERY} --save-every ${EVAL_EVERY} ${WANDB}
    " 2>&1 | tee -a ${J}.trainlog
exit ${PIPESTATUS[0]}
