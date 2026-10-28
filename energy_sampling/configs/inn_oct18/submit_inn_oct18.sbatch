#!/bin/bash
#SBATCH --time=0-14:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-2
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=inn18
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_oct18/joblogs/%x_%A_%a.out

# inn_oct18: three follow-ups to inn_oct17 (the exact-density autoregressive flow, inn_modeller.py, on MIPCAS eLJ at
# box and reduction penalties 1000, T 2.5, the pooled oct05 prior), on the same trainer code as inn_oct17.
# Each array task runs one arm: stage 1 (likelihood on the stored prior rows, held out by crystal; best.pt = best
# held-out likelihood) unless the arm names a start checkpoint, then stage 2 (the energy stage) from that best.pt.
#
#   index  arm                start of stage 2                                what differs from inn17 sharp1_nz_s0
#   0      nz_knee30          inn17_sharp1_nz_s0_mle/best.pt (no stage 1)     fresh-branch Huber knee 30 (--beta 30)
#   1      nz_long            inn17_sharp1_nz_s0_mle/best.pt (no stage 1)     stage 2 on a 700-minute clock, not 360
#   2      nz2r_s0            its own stage 1                                 stage-1 noise of norm 2 x 0.0244
# Arms 0 and 1 start from the stage-1 checkpoint of inn17 arm 3 (sharp1_nz_s0), whose own stage 2 is their control.
#
# EVERY ARM'S FLAGS ARE READ WHEN ITS TASK STARTS, from configs/inn_oct18/arms.tsv (per arm) and common.args (shared):
# edit those files on the cluster to change a queued arm in place. Outputs: artifacts/inn_proto/inn18_<arm>_mle/ (stage
# 1, arms that run it) and artifacts/inn_proto/inn18_<arm>/, wandb runs of the same names. THERE IS NO RESUME.
# DRY_RUN=1 SLURM_ARRAY_TASK_ID=<n> bash submit_inn_oct18.sbatch prints an arm's commands and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_oct18}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
PRIOR=mipcas_elj_pooled_oct05_prior.pt
PRIOR_BYTES=616093331

ROW=$(awk -F'\t' -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i' ${BATTERY}/arms.tsv)
if [ -z "${ROW}" ]; then echo "FATAL: no arm at array index '${SLURM_ARRAY_TASK_ID}' in ${BATTERY}/arms.tsv" >&2; exit 1; fi
NF=$(echo "${ROW}" | awk -F'\t' '{print NF}')
if [ "${NF}" != "6" ]; then echo "FATAL: arm row has ${NF} tab-separated fields, expected 6: ${ROW}" >&2; exit 1; fi
ARM=$(echo "${ROW}" | cut -f2); MODEL=$(echo "${ROW}" | cut -f3); MLE_ARM=$(echo "${ROW}" | cut -f4)
ENERGY_ARM=$(echo "${ROW}" | cut -f5); START=$(echo "${ROW}" | cut -f6)
BOTH=$(awk -F'\t' '$1 == "BOTH" {print $2}' ${BATTERY}/common.args)
MLE=$(awk -F'\t' '$1 == "MLE" {print $2}' ${BATTERY}/common.args)
ENERGY=$(awk -F'\t' '$1 == "ENERGY" {print $2}' ${BATTERY}/common.args)
for V in ARM MODEL ENERGY_ARM START BOTH MLE ENERGY; do
    if [ -z "${!V}" ]; then echo "FATAL: ${V} is empty (arms.tsv / common.args unreadable?)" >&2; exit 1; fi
done
if [ "${START}" = "-" ]; then
    RUN_MLE=1; BEST=artifacts/inn_proto/inn18_${ARM}_mle/best.pt
    if [ "${MLE_ARM}" = "-" ]; then echo "FATAL: arm ${ARM} runs stage 1 but its stage-1 column is '-'" >&2; exit 1; fi
else
    RUN_MLE=0; BEST=${START}
fi
[ "${MLE_ARM}" = "-" ] && MLE_ARM=""
[ "${ENERGY_ARM}" = "-" ] && ENERGY_ARM=""
ARGS1="${BOTH} ${MODEL} ${MLE} ${MLE_ARM} --run_name ${ARM}_mle"
ARGS2="${BOTH} ${MODEL} ${ENERGY} ${ENERGY_ARM} --run_name ${ARM} --init_from ${BEST}"
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"
if [ "${RUN_MLE}" = "1" ]; then echo "stage 1: python -u inn_modeller.py ${ARGS1}"; else echo "stage 1: none (start ${BEST})"; fi
echo "stage 2: python -u inn_modeller.py ${ARGS2}"
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${LOGS}
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# THE WALLS: as inn_oct17.
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
if [ "${RUN_MLE}" = "0" ] && [ ! -f ${WORKDIR}/${BEST} ]; then
    echo "FATAL: start checkpoint ${WORKDIR}/${BEST} is missing" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
  echo "arm row: ${ROW}"
  if [ "${RUN_MLE}" = "0" ]; then echo "start checkpoint: ${BEST} $(sha256sum ${WORKDIR}/${BEST} | cut -c1-16)"; fi
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
        if [ ${RUN_MLE} = 1 ]; then
            python -u inn_modeller.py ${ARGS1} || { echo 'FATAL: stage 1 (likelihood) failed' >&2; exit 1; }
        fi
        if [ ! -f ${BEST} ]; then echo 'FATAL: no stage-2 start ${BEST}' >&2; exit 1; fi
        python -u inn_modeller.py ${ARGS2}
    " 2>&1 | tee ${J}.trainlog
