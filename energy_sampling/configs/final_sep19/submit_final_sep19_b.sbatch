#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=fin19
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/final_sep19/joblogs/%x_%A_%a.out

# final_sep19 leg b: the switch into rare rollouts (N=10, share pin30) from leg A archives; 2x2 on P_B kept frozen x stored force.
# Arm = row of INDEX_b.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/final_sep19
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
mkdir -p ${LOGS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX_b.tsv
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${INDEX})
SRC=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${INDEX})
PRIOR=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${INDEX})
PRIOR_BYTES=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${INDEX})
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

# A DEAD ARM MUST NOT EAT ITS RESUBMISSION.
if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

# THE WALLS. The W3 / Niggli priors are filtered against the monoclinic class walls and the Niggli
# triclinic penalty; older MXtalTools reads MXT_NIGGLI_TRICLINIC, newer refuses the variable.
SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py
if ! grep -q 'MONO_CLASS' ${SYM}; then
    echo "FATAL: ${SYM} lacks MONO_CLASS (the monoclinic class walls) -- git pull MXtalTools" >&2; exit 1
fi
if grep -q 'MXT_NIGGLI_TRICLINIC is retired' ${SYM}; then
    NIG_EXPORT=""
elif grep -q "os.environ.get('MXT_NIGGLI_TRICLINIC'" ${SYM}; then
    NIG_EXPORT="export MXT_NIGGLI_TRICLINIC=1;"
else
    echo "FATAL: ${SYM} has no triclinic Niggli penalty -- git pull MXtalTools" >&2; exit 1
fi
HAVE=$(stat -c %s ${DATA}/${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES}" >&2; exit 1
fi

# RESUME OR SEED. checkpoint_name always wins in train.py: an arm with its own _running.pt continues it
# (full load); only a first launch seeds. REFUSES AN AMBIGUOUS MATCH.
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    CK=${OWN}
else
    # THE SEED IS LEG A'S NEWEST STEP ARCHIVE (archive_period 5000; its frozen _stepN_buffers.pt sidecar
    # pairs with it), never its live _running.pt, which the still-running leg-A job may be rewriting.
    CK=$(ls -t ${CKPTS}/*${SRC}_*_step[0-9]*.pt 2>/dev/null | grep -v '_buffers.pt$' | head -1)
    if [ -z "${CK}" ]; then
        echo "FATAL: leg A arm ${SRC} has no step archive yet in ${CKPTS} (needs archive_period steps)" >&2; exit 1
    fi
    if [ ! -f "${CK%.pt}_buffers.pt" ]; then
        echo "FATAL: ${CK} has no frozen buffers sidecar beside it" >&2; exit 1
    fi
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SWITCH from: $(basename ${CK}) (mtime $(stat -c %y ${CK}))"
fi

# THE PRIOR MODEL: written by the stub's snapshot_prior on leg A's first launch; every later launch is
# handed one (its own if it has one, else its source arm's).
OWNPRIOR=$(ls -t ${CKPTS}/*${ARM}_*_prior.pt 2>/dev/null | head -1)
SRCPRIOR=$(ls -t ${CKPTS}/*${SRC}_*_prior.pt 2>/dev/null | head -1)
if [ -n "${OWNPRIOR}" ]; then
    PM=$(basename ${OWNPRIOR}); echo "  prior model <- ${PM} (own)"
elif [ -n "${SRCPRIOR}" ]; then
    PM=$(basename ${SRCPRIOR}); echo "  prior model <- ${PM} (source arm)"
else
    PM=null; echo "  prior model <- null (the stub writes it)"
fi

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" ${CONFIG} > ${RESOLVED}
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|PRIOR_MODEL_PLACEHOLDER' ${RESOLVED}; then
    echo "FATAL: placeholder left in ${RESOLVED}" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
smi_epilogue() {
    kill ${SMI_PID} 2>/dev/null
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 \
        > ${J}_sacct.txt 2>&1
}
trap smi_epilogue EXIT TERM

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        ${NIG_EXPORT}
        python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1
        python -u train.py --config ${RESOLVED}
    " 2>&1 | tee ${J}.trainlog

if grep -q 'UNRECOVERABLE' ${J}.trainlog 2>/dev/null; then
    echo "arm ${ARM} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${CKPTS}/${ARM}.dead
fi
