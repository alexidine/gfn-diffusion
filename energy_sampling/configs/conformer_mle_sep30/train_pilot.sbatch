#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mle30
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/conformer_mle_sep30/joblogs/%x_%A_%a.out

# conformer_mle_sep30 leg a: PILOT: 1000 training steps (epochs) on the pilot rung.
# Arm = row of INDEX_a.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/conformer_mle_sep30
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
DATA=/scratch/mk8347/data/conformer_datasets/rungs_sep30
mkdir -p ${LOGS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX_a.tsv
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${INDEX})
SRC=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${INDEX})
SET=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${INDEX})
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

# RUNG GUARD. The rung was built ON THE CLUSTER by build_${SET}.sbatch (the dependency submit.sh
# sets). Every file the run and the offline eval read must be there; the RDKit version the
# conformer guard below asserts is the one the rung's manifest records, since the run re-embeds
# every member and ConformerModeller refuses reference conformers another RDKit embedded. The
# fitted prior is committed: its sha256 is HEAD's. NIG_EXPORT is the crystal template's
# MXtalTools export, empty on this route.
NIG_EXPORT=""
RUNG=${DATA}/${SET}
for F in manifest.json conditions_train.pt conditions_heldout.pt prior_train.pt conditions_train.references.pt conditions_heldout.references.pt; do
    if [ ! -s ${RUNG}/${F} ]; then echo "FATAL: ${RUNG}/${F} is missing; run build_${SET}.sbatch first" >&2; exit 1; fi
done
RDKIT=$(grep -o '"rdkit": *"[^"]*"' ${RUNG}/manifest.json | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
if [ -z "${RDKIT}" ]; then echo "FATAL: ${RUNG}/manifest.json names no RDKit version" >&2; exit 1; fi
if [ "$(sha256sum ${WORKDIR}/conformer_prior_v2.pt | cut -d' ' -f1)" != "e6c2953f2f874f83f5fab84793d1339288f6efd26a3bc157d2c8251f0ddb6153" ]; then
    echo "FATAL: ${WORKDIR}/conformer_prior_v2.pt is not HEAD's (sha256 e6c2953f2f874f83f5fab84793d1339288f6efd26a3bc157d2c8251f0ddb6153); git pull" >&2; exit 1
fi

# RESUME OR SEED. checkpoint_name always wins in train.py: an arm with its own _running.pt continues it
# (full load); only a first launch seeds. REFUSES AN AMBIGUOUS MATCH.
# CK_STEP=<N> (env) seeds this leg from the arm's OWN _step<N>.pt archive and its frozen _step<N>_buffers.pt
# instead of the live _running.pt -- the way back from a poisoned leg (cancel it first). The archive must exist.
# A WEIGHTS-ONLY ARM (its yaml carries load_weights_only: WEIGHTS_ONLY_PLACEHOLDER) takes the weights alone from its
# seed on a FIRST launch and resumes itself in full afterwards: a resubmission that loaded weights only would restart
# the arm at step 0 with empty buffers. A yaml without the placeholder is untouched by the substitution.
WO=false
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${CK_STEP:-}" ]; then
    NA=$(ls ${CKPTS}/*${ARM}_*_step${CK_STEP}.pt 2>/dev/null | grep -v '_buffers.pt$' | wc -l)
    if [ "${NA}" -ne 1 ]; then
        echo "FATAL: ${NA} matches for *${ARM}_*_step${CK_STEP}.pt in ${CKPTS} (need exactly 1)" >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${ARM}_*_step${CK_STEP}.pt | grep -v '_buffers.pt$')
    if [ ! -f "${CK%.pt}_buffers.pt" ]; then
        echo "FATAL: ${CK} has no frozen buffers sidecar beside it" >&2; exit 1
    fi
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESTART FROM ARCHIVE: $(basename ${CK}) (CK_STEP=${CK_STEP})"
elif [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    CK=${OWN}
else
    # FRESH: no checkpoint of this arm exists, so the rung starts from scratch and trains its own
    # phase 1 first. `basename null` is `null`, so the resolved checkpoint_name is YAML null.
    CK=null
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH (no checkpoint)"
    WO=true
fi

# THE PRIOR MODEL IS NEVER HANDED TO A LEG. These arms churn the prior buffer from noised ANCHORS
# (buffers.prior_buffer.source: anchors), under which the prior model is never used -- and loading one is
# actively harmful: train.py's init-time grow_prior_buffer() fills the prior buffer with min_size (10k)
# prior-MODEL samples whenever a prior model is loaded and the buffer is under max_size, bypassing the
# anchors-only source (p20_acr_n5 leg 2, 2026-09-23: anchor fraction 1.0 -> 0.2, bwd/under_coverage 2.83 -> 3.49,
# backward and replay errors rising). Leg 1 always ran with null; every leg does.
PM=null; echo "  prior model <- null (anchors-only churn; never loaded)"

sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" \
    -e "s|PRIOR_MODEL_PLACEHOLDER|${PM}|" \
    -e "s|WEIGHTS_ONLY_PLACEHOLDER|${WO}|" ${CONFIG} > ${RESOLVED}
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|PRIOR_MODEL_PLACEHOLDER\|WEIGHTS_ONLY_PLACEHOLDER' ${RESOLVED}; then
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
        python -c \"import rdkit; from rdkit.Chem import AllChem; assert rdkit.__version__ == '${RDKIT}', 'RDKit %s; the condition sets were built under ${RDKIT}' % rdkit.__version__; assert hasattr(AllChem, 'MMFFGetMoleculeForceField'), 'RDKit without MMFF'; import mxtaltools.conformers.builder\" || exit 1
        python -u conformer_modeller.py --config ${RESOLVED}
    " 2>&1 | tee ${J}.trainlog

if grep -q 'UNRECOVERABLE' ${J}.trainlog 2>/dev/null; then
    echo "arm ${ARM} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${CKPTS}/${ARM}.dead
fi
