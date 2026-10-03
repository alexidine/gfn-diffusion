#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-4
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=em03
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/energy_model_oct03/joblogs/%x_%A_%a.out

# energy_model_oct03: can the sampler's own policy network regress E(x | c)? (models/energy_probe.py)
#
# Every arm trains the forward policy's token network, one scalar per coordinate summed over the
# molecule, on the potential the r20k run samples from (its arm config, clip included), minus each
# molecule's potential at its reference conformer. Data: the r20k condition set's stored prior
# rows, each displaced afresh every step by its coordinates' thermal widths times a multiplier
# log-uniform over 0.3-2.0. 10% of the molecules (by parent skeleton) and 10% of the training
# molecules' stored conformers are held out. 300,000 steps at batch 1024, cosine 3e-4 -> 3e-6.
#
# Arm = row of INDEX.tsv (line 1 is the header):
#   0 em03_frozen       the stored embeddings (the frozen encoder the sampler runs on); token width 128, 2 mixing blocks
#   1 em03_frozen_big   the same embeddings; token width 256, 4 mixing blocks
#   2 em03_e2e_pre      the shipped encoder run inside the step and trained with the rest
#   3 em03_e2e_scratch  the same encoder shape, stereo features 2, random weights, trained with the rest
#   4 em03_e2e_big      encoder width 256 / 6 layers from random weights; token width 256, 4 mixing blocks
# Contrasts: 0 v 2 v 3 = what training the encoder buys at fixed size; 0 v 1 and 3 v 4 = what size buys.
#
# RESUBMIT THIS FILE to continue an arm past the wall: each arm resumes its own <arm>_running.pt.
# One arm: sbatch --array=<row> configs/energy_model_oct03/train_energy_model.sbatch
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/energy_model_oct03
LOGS=${ARMS}/joblogs
OUT=${WORKDIR}/checkpoints/energy_model_oct03
RUNG=/scratch/mk8347/data/conformer_datasets/rungs_sep30/r20k
CONFIG=${WORKDIR}/configs/conformer_mle_sep30/mle30_r20k.yaml
mkdir -p ${LOGS} ${OUT}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX.tsv
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${INDEX})
FLAGS=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${INDEX})
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# RUNG GUARD: the set the r20k run trains on, built by configs/conformer_mle_sep30/build_r20k.sbatch
for F in manifest.json conditions_train.pt prior_train.pt; do
    if [ ! -s ${RUNG}/${F} ]; then echo "FATAL: ${RUNG}/${F} is missing" >&2; exit 1; fi
done
if [ ! -f ${CONFIG} ]; then echo "FATAL: missing ${CONFIG}" >&2; exit 1; fi
RDKIT=$(grep -o '"rdkit": *"[^"]*"' ${RUNG}/manifest.json | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
if [ -z "${RDKIT}" ]; then echo "FATAL: ${RUNG}/manifest.json names no RDKit version" >&2; exit 1; fi

if [ -f ${OUT}/${ARM}_running.pt ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: ${OUT}/${ARM}_running.pt"
else
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH"
fi
echo "  flags: ${FLAGS}"

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
        python -c \"import rdkit; from rdkit.Chem import AllChem; assert rdkit.__version__ == '${RDKIT}', 'RDKit %s; the condition set was built under ${RDKIT}' % rdkit.__version__; assert hasattr(AllChem, 'MMFFGetMoleculeForceField'), 'RDKit without MMFF'; import mxtaltools.conformers.builder\" || exit 1
        python -u -m models.energy_probe --rung ${RUNG} --config ${CONFIG} --out-dir ${OUT} --name ${ARM} \
            ${FLAGS} --batch 1024 --steps 300000 --lr 3e-4 --lr-final 3e-6 --warmup 2000 \
            --noise-range 0.3 2.0 --heldout-frac 0.10 --conformer-frac 0.10 \
            --eval-samples 20000 --eval-every 5000 --save-every 5000 --wandb-project 'GFN Energy'
    " 2>&1 | tee -a ${J}.trainlog
exit ${PIPESTATUS[0]}
