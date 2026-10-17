#!/bin/bash
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=inn17
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_oct17/joblogs/%x_%A_%a.out

# inn_oct17: a benchmark of the exact-density autoregressive flow (inn_modeller.py) against the GFN fine-tune
# pft_mip_ft on MIPCAS eLJ at that run's target (box and reduction penalties 1000, T 2.5, pooled oct05 prior).
# Each array task runs one arm in two phases:
#   1. likelihood on the stored prior rows, held out by crystal, until its step count or its clock; best.pt is the
#      checkpoint with the best held-out likelihood;
#   2. the energy stage from that best.pt, on a 360-minute clock: the TB residual on detached fresh draws and on
#      stored rows, the stored rows scored against a lagged copy of the draws' level.
# Both phases run the occupancy batch sizer (--util_target 0.75): the rows per update grow while the card is under
# 75% busy (the likelihood batch; in the energy stage the stored-row batch, which costs no energy call). The fresh-draw
# batch, and so the energy-call count, is fixed at 2000.
#
#   index  arm              what it is
#   0      sharp1_s0        one block in the sharpness-rule order, one update per energy call
#   1      sharp1_s1        arm 0 with seed 1 (its held-out split is drawn with that seed too)
#   2      sharp2_s0        two blocks: the sharpness-rule order, then its reverse; 20,000 likelihood steps (about
#                           arm 0's compute). Its phase-1 eval_fwd/* readings are bf16 draws of a stacked flow: read them
#                           with eval_fwd/round_trip_p99 (best.pt and phase 2, in fp32, are unaffected)
#   3      sharp1_upb4_s0   arm 0 with each scored batch reused for 4 updates
#
# EVERY ARM'S FLAGS ARE READ WHEN ITS TASK STARTS, from configs/inn_oct17/arms.tsv (per arm) and common.args (shared):
# edit those files on the cluster to change a queued arm in place. Outputs: artifacts/inn_proto/inn17_<arm>_mle/ and
# artifacts/inn_proto/inn17_<arm>/ (metrics.jsonl, best.pt, latest.pt, mle_exit.pt, final.pt), wandb runs of the same
# names. THERE IS NO RESUME: a resubmitted arm starts again and overwrites them.
# DRY_RUN=1 SLURM_ARRAY_TASK_ID=<n> bash submit_inn_oct17.sbatch prints an arm's two commands and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_oct17}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
PRIOR=mipcas_elj_pooled_oct05_prior.pt
PRIOR_BYTES=616093331

ROW=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i' ${BATTERY}/arms.tsv)
if [ -z "${ROW}" ]; then echo "FATAL: no arm at array index '${SLURM_ARRAY_TASK_ID}' in ${BATTERY}/arms.tsv" >&2; exit 1; fi
ARM=$(echo "${ROW}" | cut -f2); MODEL=$(echo "${ROW}" | cut -f3); MLE_ARM=$(echo "${ROW}" | cut -f4); ENERGY_ARM=$(echo "${ROW}" | cut -f5)
BOTH=$(awk -F'\t' '$1 == "BOTH" {print $2}' ${BATTERY}/common.args)
MLE=$(awk -F'\t' '$1 == "MLE" {print $2}' ${BATTERY}/common.args)
ENERGY=$(awk -F'\t' '$1 == "ENERGY" {print $2}' ${BATTERY}/common.args)
for V in ARM MODEL BOTH MLE ENERGY; do
    if [ -z "${!V}" ]; then echo "FATAL: ${V} is empty (arms.tsv / common.args unreadable?)" >&2; exit 1; fi
done
BEST=artifacts/inn_proto/inn17_${ARM}_mle/best.pt
ARGS1="${BOTH} ${MODEL} ${MLE} ${MLE_ARM} --run_name ${ARM}_mle"
ARGS2="${BOTH} ${MODEL} ${ENERGY} ${ENERGY_ARM} --run_name ${ARM} --init_from ${BEST}"
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"
echo "phase 1: python -u inn_modeller.py ${ARGS1}"
echo "phase 2: python -u inn_modeller.py ${ARGS2}"
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${LOGS}
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# THE WALLS: as inn_oct12 and the GFN batteries' job scripts.
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
for F in inn_modeller.py energies/ar_transformer.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
  echo "arm row: ${ROW}"
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
        python -c \"import inn_modeller\" || { echo 'FATAL: inn_modeller does not import in this container (a missing module, or gfn-diffusion / MXtalTools not pulled)' >&2; exit 1; }
        python -u inn_modeller.py ${ARGS1} || { echo 'FATAL: phase 1 (likelihood) failed' >&2; exit 1; }
        if [ ! -f ${BEST} ]; then echo 'FATAL: phase 1 left no ${BEST}' >&2; exit 1; fi
        python -u inn_modeller.py ${ARGS2}
    " 2>&1 | tee ${J}.trainlog
