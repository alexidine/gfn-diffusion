#!/bin/bash
#SBATCH --time=1-08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=innmps21
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_mps_oct21/joblogs/%x_%j.out

# inn_mps_oct21: the exact-density autoregressive flow (inn_modeller.py, MIPCAS eLJ at box and reduction penalties 1000,
# T 2.5, the pooled oct05 prior) trained toward convergence, six stage-2 arms packed onto ONE A100 under NVIDIA MPS.
#
# Recipe: the candidate default of the 20 Oct battery (docs/results/spline-flow-bounded-rows-2026-10-20.md): bounded
# linear rows, tied knot derivatives, bin-height floor 1e-7, error-filtered replay at 0.1, one transformer block of 32 bins
# in the order 8.11.10.9.7.6.5.4.3.2.1.0. Changes from that battery: the occupancy sizer is off (--util_target 0) with 4096
# stored rows per update (its gradient was near noise-free at 8192), evals on 10k draws, and stage 2 is scheduled on a
# 26-hour clock (--budget_minutes 1560) so every arm finishes its learning-rate decay inside the leg whatever its speed.
#
#   phase 1  two stage-1 runs at once (seeds s0, s1), cap 0.3 of the card each      -> inn_mps21_s0_mle, inn_mps21_s1_mle
#   phase 2  six stage-2 arms at once, cap 0.12 each (local peak 4.1 GB per arm)     -> inn_mps21_<arm>
#     0 base       the candidate default, from s0
#     1 seed1      the candidate default, from s1 (seed 1, same held-out split)
#     2 lr_half    peak rate 1.5e-4
#     3 lr_double  peak rate 6e-4
#     4 b1000      1000 fresh draws per update (more steps per energy call)
#     5 rep02      replay weight 0.2
#
# MPS: the job starts nvidia-cuda-mps-control on the node before the container and the processes connect through
# CUDA_MPS_PIPE_DIRECTORY; if it does not start, the processes time-share the card (stated in the .info file). A card
# in Exclusive_Process mode without MPS cannot host two processes, and the job stops before training.
# Flags are read when the job STARTS from configs/inn_mps_oct21/arms.tsv and common.args (edit in place while queued).
# Outputs: artifacts/inn_proto/inn_mps21_<run>/, wandb runs of the same names, joblogs/<arm>_<jobid>.trainlog.
# THERE IS NO RESUME. DRY_RUN=1 bash submit_inn_mps_oct21.sbatch prints every command and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_mps_oct21}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
PRIOR=mipcas_elj_pooled_oct05_prior.pt
PRIOR_BYTES=616093331
TAG=inn_mps21

field() { awk -F'\t' -v k="$1" '$1 == k {print $2}' ${BATTERY}/common.args; }
BOTH=$(field BOTH); MLE=$(field MLE); ENERGY=$(field ENERGY)
for V in BOTH MLE ENERGY; do
    if [ -z "${!V}" ]; then echo "FATAL: ${V} is empty (common.args unreadable?)" >&2; exit 1; fi
done
seedflags() { local s; s=$(field "SEED_$1"); if [ -z "${s}" ]; then echo "FATAL: no SEED_$1 line in common.args" >&2; exit 1; fi
              [ "${s}" = "-" ] && s=""; echo "${s}"; }

# the arms: index, arm, seed tag, stage-2 flags
ARMS=(); SEEDS_USED=()
declare -A ARM_SEED ARM_FLAGS
while IFS=$'\t' read -r IDX ARM SEED FLAGS; do
    case "${IDX}" in ''|\#*) continue ;; esac
    if [ -z "${ARM}" ] || [ -z "${SEED}" ] || [ -z "${FLAGS}" ]; then
        echo "FATAL: malformed arms.tsv row (need 4 tab-separated fields): ${IDX} ${ARM} ${SEED} ${FLAGS}" >&2; exit 1
    fi
    [ "${FLAGS}" = "-" ] && FLAGS=""
    ARMS+=("${ARM}"); ARM_SEED[${ARM}]=${SEED}; ARM_FLAGS[${ARM}]="${FLAGS}"
    case " ${SEEDS_USED[*]} " in *" ${SEED} "*) ;; *) SEEDS_USED+=("${SEED}") ;; esac
done < ${BATTERY}/arms.tsv
if [ ${#ARMS[@]} -eq 0 ]; then echo "FATAL: no arms in ${BATTERY}/arms.tsv" >&2; exit 1; fi

declare -A MLE_CMD BEST
for S in "${SEEDS_USED[@]}"; do
    SF=$(seedflags ${S}) || exit 1
    MLE_CMD[${S}]="${BOTH} ${MLE} ${SF} --run_name ${S}_mle"
    BEST[${S}]=artifacts/inn_proto/${TAG}_${S}_mle/best.pt
done
declare -A S2_CMD
for A in "${ARMS[@]}"; do
    S=${ARM_SEED[${A}]}; SF=$(seedflags ${S}) || exit 1
    S2_CMD[${A}]="${BOTH} ${ENERGY} ${SF} ${ARM_FLAGS[${A}]} --run_name ${A} --init_from ${BEST[${S}]}"
done
for S in "${SEEDS_USED[@]}"; do echo "stage 1 ${S}: python -u inn_modeller.py ${MLE_CMD[${S}]}"; done
for A in "${ARMS[@]}"; do echo "stage 2 ${A}: python -u inn_modeller.py ${S2_CMD[${A}]}"; done
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${LOGS}
J=${LOGS}/job_${SLURM_JOB_ID}

# THE WALLS: as inn_oct18f.
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
if ! grep -q 'def decode' ${WORKDIR}/energies/ar_transformer.py || ! grep -q 'def resolve_auto_options' ${WORKDIR}/inn_modeller.py; then
    echo "FATAL: gfn-diffusion predates the cached sampler or the option resolution -- git pull gfn-diffusion" >&2; exit 1
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
  cat ${BATTERY}/arms.tsv ${BATTERY}/common.args
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
    for S in "${SEEDS_USED[@]}"; do
        echo "python -u inn_modeller.py ${MLE_CMD[${S}]} > ${LOGS}/${S}_mle_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${S}_mle)"
    done
    echo 'for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} || { echo "FATAL: stage 1 ${NAMES[$i]} failed" >&2; FAIL=1; }; done'
    echo '[ ${FAIL} = 0 ] || exit 1'
    for S in "${SEEDS_USED[@]}"; do
        echo "[ -f ${BEST[${S}]} ] || { echo 'FATAL: no stage-1 checkpoint ${BEST[${S}]}' >&2; exit 1; }"
    done
    echo 'PIDS=(); NAMES=()'
    for A in "${ARMS[@]}"; do
        echo "python -u inn_modeller.py ${S2_CMD[${A}]} > ${LOGS}/${A}_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${A})"
    done
    echo 'for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} && echo "arm ${NAMES[$i]} done" || { echo "arm ${NAMES[$i]} FAILED (see its .trainlog)" >&2; FAIL=1; }; done'
    echo 'exit ${FAIL}'
} > ${INNER}

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /tmp:/tmp \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash ${INNER} 2>&1 | tee ${J}.trainlog
