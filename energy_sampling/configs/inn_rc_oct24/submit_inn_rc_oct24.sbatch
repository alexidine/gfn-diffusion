#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=innrc24
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_rc_oct24/joblogs/%x_%j.out

# inn_rc_oct24: the conditional exact-density flow (inn_modeller.py), stage 1 (likelihood) only, on the rc_oct22
# random-conformer priors (eLJ, T 6.9, space group 2, Z' 1; build_rc_prior.py, installment 1), three arms packed onto ONE
# A100 under NVIDIA MPS:
#
#   0 rcq        qm9rc only: the qm9full SMILES with one random conformer each in place of the DFT geometry
#   1 rcall      all five families of installment 1 (qm9rc, csdrc, zincrc, gdbrc, pubrc)
#   2 rcall_big  as rcall, with the wider and deeper conditioner (384 x 6 x 8)
#
# Every arm is scored on the same held-out set, the qm9rc held-out molecules (the qm9full prior's held-out SMILES, with
# random-conformer geometries), so rcq against rcall isolates the training molecules. The recipe is inn_atoms_oct21's
# atoms arm (atom-token condition, 200k steps, batch 2048, one row per molecule).
#
# Read: eval_test/mle and eval_test/mle_gap (held-out minus train likelihood) against steps for each arm; the held-out
# excess energy; training-molecule draws' Excess Energy Nats P50.
#
# MPS: the job starts nvidia-cuda-mps-control on the node before the container and the processes connect through
# CUDA_MPS_PIPE_DIRECTORY; if it does not start, the processes time-share the card (stated in the .info file). A card
# in Exclusive_Process mode without MPS cannot host two processes, and the job stops before training.
# Flags are read when the job STARTS from configs/inn_rc_oct24/arms.tsv and common.args (edit in place while queued).
# Outputs: artifacts/inn_proto/inn_rc24_<arm>/, wandb runs of the same names, joblogs/<arm>_<jobid>.trainlog.
# THERE IS NO RESUME. DRY_RUN=1 bash submit_inn_rc_oct24.sbatch prints every command and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_rc_oct24}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors/rc_oct22
# the five files and their sizes in bytes (the uploads of D:\crystal_datasets\conditional\priorsc_oct22)
FILES=(rcoct22i1_qm9rc_prior.pt rcoct22i1_qm9rc_conditions.pt rcoct22i1_qm9rc_test_conditions.pt rcoct22i1_prior.pt rcoct22i1_conditions.pt)
BYTES=(2207012889 228547666 13259723 13156215839 1357117492)

ALL=$(awk -F'\t' '$1 == "ALL" {print $2}' ${BATTERY}/common.args)
if [ -z "${ALL}" ]; then echo "FATAL: no ALL line in ${BATTERY}/common.args" >&2; exit 1; fi

# the arms: index, arm, flags, memory share
ARMS=()
declare -A ARM_CMD
TOTAL_SHARE=0
while IFS=$'\t' read -r IDX ARM FLAGS SHARE; do
    case "${IDX}" in ''|\#*) continue ;; esac
    if [ -z "${ARM}" ] || [ -z "${FLAGS}" ] || [ -z "${SHARE}" ]; then
        echo "FATAL: malformed arms.tsv row (need 4 tab-separated fields): ${IDX} ${ARM} ${FLAGS} ${SHARE}" >&2; exit 1
    fi
    [ "${FLAGS}" = "-" ] && FLAGS=""
    ARMS+=("${ARM}")
    ARM_CMD[${ARM}]="${ALL} ${FLAGS} --cuda_memory_fraction ${SHARE} --run_name ${ARM}"
    TOTAL_SHARE=$(awk -v a="${TOTAL_SHARE}" -v b="${SHARE}" 'BEGIN {print a + b}')
done < ${BATTERY}/arms.tsv
if [ ${#ARMS[@]} -eq 0 ]; then echo "FATAL: no arms in ${BATTERY}/arms.tsv" >&2; exit 1; fi
if awk -v t="${TOTAL_SHARE}" 'BEGIN {exit !(t > 0.95)}'; then
    echo "FATAL: the arms' memory shares sum to ${TOTAL_SHARE} of the card (> 0.95)" >&2; exit 1
fi
for A in "${ARMS[@]}"; do echo "${A}: python -u inn_modeller.py ${ARM_CMD[${A}]}"; done
echo "memory shares sum to ${TOTAL_SHARE}"
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
for i in "${!FILES[@]}"; do
    HAVE=$(stat -c %s ${DATA}/${FILES[$i]} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${BYTES[$i]}" ]; then
        echo "FATAL: ${DATA}/${FILES[$i]} is ${HAVE} bytes, expected ${BYTES[$i]}" >&2; exit 1
    fi
done
for F in inn_modeller.py energies/ar_transformer.py energies/prior_flow.py tests/models/test_inn_atom_tokens.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
if ! grep -q 'class AtomEncoder' ${WORKDIR}/inn_modeller.py || ! grep -q 'def shared_context' ${WORKDIR}/inn_modeller.py; then
    echo "FATAL: gfn-diffusion predates the atom-token condition -- git pull gfn-diffusion" >&2; exit 1
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

# one container: the atom-token tests on the CPU first (the encoder's batch independence in train and eval mode, causality,
# the cached decode with context against the d-pass inverse, with this checkout's MXtalTools), then every arm at once
INNER=${J}_inner.sh
{
    echo 'source /ext3/env.sh'
    echo "export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH"
    echo "${NIG_EXPORT}"
    echo 'export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1'
    echo "python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1"
    echo "python -c \"import inn_modeller\" || { echo 'FATAL: inn_modeller does not import in this container' >&2; exit 1; }"
    echo "CUDA_VISIBLE_DEVICES=-1 python tests/models/test_inn_atom_tokens.py > ${LOGS}/preflight_${SLURM_JOB_ID}.log 2>&1 || { echo 'FATAL: the atom-token tests fail in this container (see preflight log)' >&2; exit 1; }"
    echo 'FAIL=0; PIDS=(); NAMES=()'
    for A in "${ARMS[@]}"; do
        echo "python -u inn_modeller.py ${ARM_CMD[${A}]} > ${LOGS}/${A}_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${A})"
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
