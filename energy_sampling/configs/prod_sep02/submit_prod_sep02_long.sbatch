#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=p02long
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/prod_sep02/joblogs/%x_%A_%a.out

# prod_sep02 7-day single-leg arms. Arm = row of INDEX_long.tsv (line 1 is the header).
# DO NOT EDIT --array BY HAND: make.py rewrites it to match the index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/prod_sep02
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
mkdir -p ${LOGS}

ARM=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' ${ARMS}/INDEX_long.tsv)
SRC=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $2}' ${ARMS}/INDEX_long.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# RESUME OR SEED. train.py's loader is `if checkpoint_name ... elif
# continue_from_checkpoint`, so checkpoint_name ALWAYS wins -- without this an
# extended arm would silently restart from the phase-1 exit and discard its own
# progress.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    CK=${OWN}
else
    # REFUSES AN AMBIGUOUS MATCH rather than taking the newest: every arm must
    # provably seed from the same file or it is not a comparison.
    N=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "FATAL: no phase-1 exit matches *${SRC}_*_phase1_exit.pt" >&2; exit 1
    fi
    if [ "${N}" -gt 1 ]; then
        echo "FATAL: ${N} ambiguous matches for *${SRC}_*_phase1_exit.pt:" >&2
        ls ${CKPTS}/*${SRC}_*_phase1_exit.pt >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_phase1_exit.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK})"
fi

# THE PRIOR MODEL, resolved the same way and for the same reason. `prior_model`
# is not in the checkpoint and only train_prior's snapshot_prior creates it, but
# protocol.begin() returns early when step_ind != 0 -- so a resumed leg would
# come up with no prior sampler and bwd_sampling_mode: prior would silently
# degrade to an anchors-only buffer across 95% of the branch budget. On leg 1
# there is nothing to find and `null` is correct: the stub runs and writes it.
OWNPRIOR=$(ls -t ${CKPTS}/*${ARM}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${OWNPRIOR}" ]; then
    PM=$(basename ${OWNPRIOR}); echo "  prior model <- ${PM}"
else
    PM=null; echo "  prior model <- null (leg 1 writes it)"
fi

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" ${CONFIG} > ${RESOLVED}

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
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        python -u train.py --config ${RESOLVED}
    " 2>&1 | tee ${J}.trainlog
