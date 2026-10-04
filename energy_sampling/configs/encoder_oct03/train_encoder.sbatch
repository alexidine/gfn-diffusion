#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=enc03
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/encoder_oct03/joblogs/%x_%A_%a.out

# encoder_oct03: the molecule encoder trained on ALL of QM9 (models/encoder_probe.py).
#
# The shipped encoder saw 20,000 molecules for 60,000 steps at stereo features 1. Every arm here:
# 110,000 training molecules and 5,000 held out by parent skeleton, read from the committed
# configs/conformer_db_sep29/qm9_index.tsv.gz; stereo features 2 (r/s at pseudo-asymmetric
# centres, an E/Z bond column, the two E/Z probes); message passing + attention + shortest-path
# bias; 400,000 steps at 256 molecules, rate 3e-4 falling on a cosine to 3e-6 after 2,000 warm-up
# steps; the best held-out state is the one saved and scored.
#
# Arm = row of INDEX.tsv (line 1 is the header):
#   0 enc03_w128     width 128, 4 layers: the shipped shape, so its embeddings fit the current policy input
#   1 enc03_w192     width 192, 4 layers
#   2 enc03_w256l6   width 256, 6 layers
#
# Writes, per arm, under checkpoints/encoder_oct03/<arm>/:
#   mp+attn+spd_n110000_s0.pt        the encoder (models/encoder_cache.load_encoder reads it)
#   mp+attn+spd_n110000_s0_state.pt  the training state the arm resumes from
#   <arm>.json                       the probe table and the loss curves
# RESUBMIT THIS FILE to continue an arm past the wall.
# One arm: sbatch --array=<row> configs/encoder_oct03/train_encoder.sbatch
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/encoder_oct03
LOGS=${ARMS}/joblogs
SOURCE=configs/conformer_db_sep29/qm9_index.tsv.gz
NCPU=${SLURM_CPUS_PER_TASK:-8}
mkdir -p ${LOGS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX.tsv
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${INDEX})
HIDDEN=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${INDEX})
LAYERS=$(awk -F'\t' -v n=${ROW} 'NR==n {print $3}' ${INDEX})
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
OUT=${WORKDIR}/checkpoints/encoder_oct03/${ARM}
J=${LOGS}/${ARM}_${SLURM_JOB_ID}
mkdir -p ${OUT}

if [ "$(sha256sum ${WORKDIR}/${SOURCE} | cut -d' ' -f1)" != "7c88ed2437c11efe03747d66ff5a97259ea0c2338cad9343b5825b687c9125c0" ]; then
    echo "FATAL: ${WORKDIR}/${SOURCE} is not the committed QM9 index (git pull)" >&2; exit 1
fi
# ONE JOB PER ARM. A second job on an arm resumes the same state file and both then overwrite it.
# This job exits if another running job of this name holds the same array row.
ME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
OTHER=$(squeue -h -u ${USER} --name=${SLURM_JOB_NAME} -t RUNNING -o "%K %i" 2>/dev/null \
    | awk -v k=${SLURM_ARRAY_TASK_ID} -v me=${ME} '$1==k && $2!=me {print $2}' | head -1)
if [ -n "${OTHER}" ]; then
    echo "arm ${ARM} is already running as job ${OTHER}; this job exits without starting a second copy. Resubmit once that one has ended." >&2
    exit 0
fi

if [ -f ${OUT}/mp+attn+spd_n110000_s0_state.pt ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM} (width ${HIDDEN}, ${LAYERS} layers)  RESUME"
else
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM} (width ${HIDDEN}, ${LAYERS} layers)  FRESH"
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:${WORKDIR}:\$PYTHONPATH
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        python -u -m models.encoder_probe --device cuda --arms mp+attn+spd --sizes 110000 --n-test 5000 \
            --seeds 0 --stereo-features 2 --hidden ${HIDDEN} --layers ${LAYERS} --no-match-params \
            --steps 400000 --batch-mols 256 --lr 3e-4 --lr-schedule cosine --lr-final 3e-6 --warmup-steps 2000 \
            --source ${SOURCE} --build-workers ${NCPU} --resume --save-every 5000 --verbose \
            --save-dir ${OUT} --out ${OUT}/${ARM}.json
    " 2>&1 | tee -a ${J}.log
exit ${PIPESTATUS[0]}
