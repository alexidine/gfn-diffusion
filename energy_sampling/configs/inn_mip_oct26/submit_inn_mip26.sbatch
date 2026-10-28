#!/bin/bash
#SBATCH --time=0-13:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=innmip26
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_mip_oct26/joblogs/%x_%j.out

# inn_mip_oct26: a design battery of the exact-density autoregressive flow (inn_modeller.py) on MIPCAS eLJ, the recipe
# of inn_mps_oct21 (common.args) with one change per arm. Each job packs its arms onto ONE A100 under NVIDIA MPS:
#
#   phase 1  one stage-1 run (10,000 steps) per stage-1 configuration its arms name (stage1.tsv), all at once, as
#            <key>_mle_<group>; each run's share of the card is 0.9 x its configuration's memory weight / the sum
#   phase 2  every arm of the job at once on a 480-minute clock (common.args), from its configuration's stage-1
#            best.pt, shares weighted the same way; an arm whose stage 1 wrote no final.pt is skipped
#
# Jobs (arms_<group>.tsv): p1 level games, p2 losses, p3 reuse and replay (all from the base configuration, which each
# of these jobs trains for itself under its own name, with a base arm of its own: base, base_p2, base_p3), a1 floor,
# forces, bins, bounded rows, a2 blocks, order, knot derivatives, MADE, a3 transformer size and normalisation (each
# from its own stage 1). One seed per arm; base_s1 (p1) differs from base in its data and draw streams only (model
# building reseeds torch, so the initial weights are base's). Submit all six with  bash submit_all.sh, or one with
#   sbatch --export=ALL,ARMS_TSV=arms_<group>.tsv submit_inn_mip26.sbatch
#
# Read: eval_fwd/jensen_z (ELBO) and eval_fwd/emp_z against updates, energy calls (energy/samples_total; the forces
# arm's path term adds 512 per update that it does not count) and wall time (within a job only); emp_z - jensen_z and
# logw_std; the Excess Energy tail (P99.9, above 30 kT); eval_bwd held-out row NLL; collapses (jensen_z drops,
# TB-error spikes).
#
# MPS: the job starts nvidia-cuda-mps-control on the node before the container and the processes connect through
# CUDA_MPS_PIPE_DIRECTORY; if it does not start, the processes time-share the card (stated in the .info file). A card
# in Exclusive_Process mode without MPS cannot host several processes, and the job stops before training.
# Flags are read when the job STARTS from common.args, stage1.tsv and the arms file (edit in place while queued).
# Outputs: artifacts/inn_proto/inn_mip26_<run>/, wandb runs of the same names, joblogs/<run>_<jobid>.trainlog.
# THERE IS NO RESUME. DRY_RUN=1 BATTERY_DIR=$(pwd) ARMS_TSV=arms_p1.tsv bash submit_inn_mip26.sbatch prints every
# command and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_mip_oct26}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
PRIOR=mipcas_elj_pooled_oct05_prior.pt
PRIOR_BYTES=616093331
TAG=inn_mip26

ARMS_FILE=${BATTERY}/${ARMS_TSV:-arms_p1.tsv}
if [ ! -f "${ARMS_FILE}" ]; then echo "FATAL: no arms file ${ARMS_FILE}" >&2; exit 1; fi
G=$(basename "${ARMS_FILE}" .tsv); G=${G#arms_}

field() { awk -F'\t' -v k="$1" '$1 == k {print $2}' ${BATTERY}/common.args; }
BOTH=$(field BOTH); MLE=$(field MLE); ENERGY=$(field ENERGY)
for V in BOTH MLE ENERGY; do
    if [ -z "${!V}" ]; then echo "FATAL: ${V} is empty (common.args unreadable?)" >&2; exit 1; fi
done

# stage-1 configurations: key, flags of both stages, flags of stage 2 only, memory weight
declare -A S1_FLAGS S1_S2 S1_WT
while IFS=$'\t' read -r KEY FL S2 WT DESC; do
    case "${KEY}" in ''|\#*) continue ;; esac
    if [ -z "${FL}" ] || [ -z "${S2}" ] || [ -z "${WT}" ]; then echo "FATAL: malformed stage1.tsv row: ${KEY}" >&2; exit 1; fi
    [ "${FL}" = "-" ] && FL=""; [ "${S2}" = "-" ] && S2=""
    S1_FLAGS[${KEY}]="${FL}"; S1_S2[${KEY}]="${S2}"; S1_WT[${KEY}]="${WT}"
done < ${BATTERY}/stage1.tsv

# the arms: index, arm, stage-1 key, stage-2 flags, description
ARMS=(); KEYS=()
declare -A ARM_KEY ARM_FLAGS
while IFS=$'\t' read -r IDX ARM KEY FLAGS DESC; do
    case "${IDX}" in ''|\#*) continue ;; esac
    if [ -z "${ARM}" ] || [ -z "${KEY}" ] || [ -z "${FLAGS}" ]; then
        echo "FATAL: malformed ${ARMS_FILE} row (need 5 tab-separated fields): ${IDX} ${ARM} ${KEY} ${FLAGS}" >&2; exit 1
    fi
    if [ -z "${S1_FLAGS[${KEY}]+set}" ]; then echo "FATAL: arm ${ARM}: no stage-1 configuration '${KEY}' in stage1.tsv" >&2; exit 1; fi
    [ "${FLAGS}" = "-" ] && FLAGS=""
    ARMS+=("${ARM}"); ARM_KEY[${ARM}]=${KEY}; ARM_FLAGS[${ARM}]="${FLAGS}"
    case " ${KEYS[*]} " in *" ${KEY} "*) ;; *) KEYS+=("${KEY}") ;; esac
done < ${ARMS_FILE}
if [ ${#ARMS[@]} -eq 0 ]; then echo "FATAL: no arms in ${ARMS_FILE}" >&2; exit 1; fi
# each run's share of the card: 0.9 x its configuration's memory weight / the sum over the runs of its phase
W1=0; for K in "${KEYS[@]}"; do W1=$(awk -v a=${W1} -v b=${S1_WT[${K}]} 'BEGIN {print a + b}'); done
W2=0; for A in "${ARMS[@]}"; do W2=$(awk -v a=${W2} -v b=${S1_WT[${ARM_KEY[${A}]}]} 'BEGIN {print a + b}'); done
share() { awk -v w=$1 -v t=$2 'BEGIN {printf "%.3f", 0.9 * w / t}'; }

declare -A MLE_CMD BEST DONE
for K in "${KEYS[@]}"; do
    MLE_CMD[${K}]="${BOTH} ${MLE} ${S1_FLAGS[${K}]} --cuda_memory_fraction $(share ${S1_WT[${K}]} ${W1}) --run_name ${K}_mle_${G}"
    BEST[${K}]=artifacts/inn_proto/${TAG}_${K}_mle_${G}/best.pt
    DONE[${K}]=artifacts/inn_proto/${TAG}_${K}_mle_${G}/final.pt   # written only when stage 1 ends; best.pt exists from step 0
done
declare -A S2_CMD
for A in "${ARMS[@]}"; do
    K=${ARM_KEY[${A}]}
    S2_CMD[${A}]="${BOTH} ${ENERGY} ${S1_FLAGS[${K}]} ${S1_S2[${K}]} ${ARM_FLAGS[${A}]} --cuda_memory_fraction $(share ${S1_WT[${K}]} ${W2}) --run_name ${A} --init_from ${BEST[${K}]}"
done
echo "job ${G}: ${#KEYS[@]} stage-1 runs (weights sum ${W1}), then ${#ARMS[@]} arms (weights sum ${W2}); shares are 0.9 x weight / sum"
for K in "${KEYS[@]}"; do echo "stage 1 ${K}: python -u inn_modeller.py ${MLE_CMD[${K}]}"; done
for A in "${ARMS[@]}"; do echo "stage 2 ${A}: python -u inn_modeller.py ${S2_CMD[${A}]}"; done
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${LOGS}
J=${LOGS}/job_${SLURM_JOB_ID}

# THE WALLS: as inn_mps_oct21.
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
for F in inn_modeller.py energies/ar_transformer.py energies/prior_flow.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
if ! grep -q '^NORMS = ' ${WORKDIR}/energies/ar_transformer.py || ! grep -q 'def resolve_auto_options' ${WORKDIR}/inn_modeller.py; then
    echo "FATAL: gfn-diffusion predates --tf_norm or the option resolution -- git pull gfn-diffusion" >&2; exit 1
fi

# MPS, on the node, before the container; the processes inherit the pipe directory through the environment
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_${SLURM_JOB_ID}
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${SLURM_JOB_ID}
mkdir -p ${CUDA_MPS_PIPE_DIRECTORY} ${CUDA_MPS_LOG_DIRECTORY}
MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null | head -1)
if command -v nvidia-cuda-mps-control >/dev/null 2>&1 && nvidia-cuda-mps-control -d; then
    MPS=on
else
    MPS=off
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    case "${MODE}" in
        *Exclusive_Process*) echo "FATAL: compute mode ${MODE} and MPS did not start: the card cannot host several processes" >&2; exit 1 ;;
    esac
fi

{ nvidia-smi -L
  echo "compute mode: ${MODE}   MPS: ${MPS}"
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
  cat ${ARMS_FILE} ${BATTERY}/stage1.tsv ${BATTERY}/common.args
} > ${J}.info 2>&1

stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks_throttle_reasons.active,power.draw,temperature.gpu \
    --format=csv,nounits -l 10 > ${J}_smi.csv &
SMI_PID=$!
epilogue() {
    kill ${SMI_PID} 2>/dev/null
    if [ "${MPS}" = "on" ]; then echo quit | nvidia-cuda-mps-control 2>/dev/null; fi
    sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,NodeList,Reason,Comment%64 > ${J}_sacct.txt 2>&1
}
trap epilogue EXIT TERM

# one container; inside it, each phase launches its runs in the background and waits for every one
INNER=${J}_inner.sh
{
    echo 'source /ext3/env.sh'
    echo "export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH"
    echo "${NIG_EXPORT}"
    echo 'export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1'
    echo "python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1"
    echo "python -c \"import inn_modeller\" || { echo 'FATAL: inn_modeller does not import in this container' >&2; exit 1; }"
    echo 'FAIL=0; PIDS=(); NAMES=()'
    for K in "${KEYS[@]}"; do
        echo "python -u inn_modeller.py ${MLE_CMD[${K}]} > ${LOGS}/${K}_mle_${G}_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${K}_mle_${G})"
    done
    echo 'for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || { echo "stage 1 ${NAMES[$i]} FAILED (see its .trainlog)" >&2; FAIL=1; }; done'
    echo 'PIDS=(); NAMES=()'
    for A in "${ARMS[@]}"; do
        # an arm whose stage 1 did not finish (no final.pt) is skipped; the others run
        echo "if [ -f ${DONE[${ARM_KEY[${A}]}]} ]; then python -u inn_modeller.py ${S2_CMD[${A}]} > ${LOGS}/${A}_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${A}); else echo 'arm ${A} SKIPPED: stage 1 wrote no ${DONE[${ARM_KEY[${A}]}]}' >&2; FAIL=1; fi"
    done
    echo 'for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} && echo "arm ${NAMES[$i]} done" || { echo "arm ${NAMES[$i]} FAILED (see its .trainlog)" >&2; FAIL=1; }; done'
    echo 'exit ${FAIL}'
} > ${INNER}

set -o pipefail   # the job's exit status is the inner script's (a failed run), not tee's
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /tmp:/tmp \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash ${INNER} 2>&1 | tee ${J}.trainlog
