#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=inncap26
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_cap_oct26/joblogs/%x_%j.out

# inn_cap_oct26: a capacity survey of the conditional exact-density flow (inn_modeller.py, atom-token condition), stage 1
# (likelihood) only, on the qm9rc prior (rc_oct22; eLJ, T 6.9, space group 2, Z' 1). Two axes:
#   training molecules  --molecule_fraction 0.01 / 0.1 / 1 (1,105 / 11,452 / 114,817 molecules; nested; the same
#                       subsets as the local runs rc_qm9f001_oct26 / rc_qm9f01_oct26)
#   model size          S = transformer 256 x 4 with the atom encoder 128 x 3 (the rc recipe, 6.0M parameters);
#                       M = transformer 384 x 6 (rcall_big's); L = transformer 512 x 8 with the encoder 256 x 5;
#                       f1_enc = S with the encoder 256 x 5 (the encoder alone)
# Every arm runs 50,000 steps on one cosine (matched data exposure) and is scored on its own training molecules
# (eval_fwd) and the 6,652 held-out molecules (eval_test). Jobs: s (arms_s.tsv: the S arms and f1_enc), m (M), l1 (f1_l alone)
# and l2 (f001_l, f01_l), one A100 each, the arms of a job at once under NVIDIA MPS, each at 0.9 x its memory weight / the job's sum.
#
# Read: Excess Energy Nats P50/P90 and Reasonable Sample Fraction on training molecules against eval_test's, eval_fwd/
# jensen_z, eval_bwd/mle (the clean training rows, the anchors -- not the noised rows stage 1 fits), E/noised_* (the
# draws' energy marginal against the buffer's at start-up), eval_bwd/mle_heldout_rows, eval_test/mle, against steps, across fractions at one
# size and across sizes at one fraction. Locally (one block, size S) the 1% subset reached a training-molecule excess
# P50 of 6.0 kT at 10k steps and the full set plateaued at 32.6 kT (20k steps).
#
# Prior construction (both stages): the prior buffer (--stored_source buffer): the training rows as anchors, a Gaussian
# kick calibrated to 6 kT mean excess over the anchor, admitted under Emin(c) + ramp_floor, 4 rows per training
# condition, churned at every eval. Stage 2 (COMMON=common_s2.args, arms_s2a.tsv: f1_s forward / pooled level;
# arms_s2b.tsv: f1_l forward) runs from a full-set arm's best.pt on the same construction; submit_all.sh chains
# s2a after job s and s2b after job l1 (--dependency=afterok).
#
# Flags are read when the job STARTS from common.args (or the file COMMON names) and the arms file (edit in place
# while queued). Submit all six
# with  bash submit_all.sh, or one with  sbatch --export=ALL,ARMS_TSV=arms_<s|m|l1|l2>.tsv submit_inn_cap26.sbatch
# Outputs: artifacts/inn_proto/inn_cap27_<arm>/ (stage 2: inn_cap27s2_<arm>/), wandb runs of the same names, joblogs/<arm>_<jobid>.trainlog.
# THERE IS NO RESUME. DRY_RUN=1 BATTERY_DIR=$(pwd) ARMS_TSV=arms_s.tsv bash submit_inn_cap26.sbatch prints every
# command and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BATTERY=${BATTERY_DIR:-${WORKDIR}/configs/inn_cap_oct26}  # BATTERY_DIR: a local copy, for DRY_RUN only
LOGS=${BATTERY}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors/rc_oct22
# the three files the arms read and their sizes in bytes (the uploads of D:/crystal_datasets/conditional/priors/rc_oct22)
FILES=(rcoct22i1_qm9rc_test_conditions.pt rcoct22i1_qm9rc_prior.pt rcoct22i1_qm9rc_conditions.pt)
BYTES=(13259723 2207012889 228547666)

ARMS_FILE=${BATTERY}/${ARMS_TSV:-arms_s.tsv}
if [ ! -f "${ARMS_FILE}" ]; then echo "FATAL: no arms file ${ARMS_FILE}" >&2; exit 1; fi
COMMON_FILE=${BATTERY}/${COMMON:-common.args}
if [ ! -f "${COMMON_FILE}" ]; then echo "FATAL: no common flags file ${COMMON_FILE}" >&2; exit 1; fi
ALL=$(awk -F'\t' '$1 == "ALL" {print $2}' ${COMMON_FILE})
if [ -z "${ALL}" ]; then echo "FATAL: no ALL line in ${COMMON_FILE}" >&2; exit 1; fi

# the arms: index, arm, flags, memory weight, description; each arm's share of the card is 0.9 x weight / the sum
ARMS=()
declare -A ARM_FLAGS ARM_WT
W=0
while IFS=$'\t' read -r IDX ARM FLAGS WT DESC; do
    case "${IDX}" in ''|\#*) continue ;; esac
    if [ -z "${ARM}" ] || [ -z "${FLAGS}" ] || [ -z "${WT}" ]; then
        echo "FATAL: malformed ${ARMS_FILE} row (need 5 tab-separated fields): ${IDX} ${ARM} ${FLAGS} ${WT}" >&2; exit 1
    fi
    [ "${FLAGS}" = "-" ] && FLAGS=""
    ARMS+=("${ARM}"); ARM_FLAGS[${ARM}]="${FLAGS}"; ARM_WT[${ARM}]=${WT}
    W=$(awk -v a="${W}" -v b="${WT}" 'BEGIN {print a + b}')
done < ${ARMS_FILE}
if [ ${#ARMS[@]} -eq 0 ]; then echo "FATAL: no arms in ${ARMS_FILE}" >&2; exit 1; fi
declare -A ARM_CMD
TOTAL_SHARE=0
for A in "${ARMS[@]}"; do
    SHARE=$(awk -v w=${ARM_WT[${A}]} -v t=${W} 'BEGIN {printf "%.3f", 0.9 * w / t}')
    ARM_CMD[${A}]="${ALL} ${ARM_FLAGS[${A}]} --cuda_memory_fraction ${SHARE} --run_name ${A}"
    TOTAL_SHARE=$(awk -v a="${TOTAL_SHARE}" -v b="${SHARE}" 'BEGIN {print a + b}')
done
if awk -v t="${TOTAL_SHARE}" 'BEGIN {exit !(t > 0.95)}'; then
    echo "FATAL: the arms' memory shares sum to ${TOTAL_SHARE} of the card (> 0.95)" >&2; exit 1
fi
for A in "${ARMS[@]}"; do echo "${A}: python -u inn_modeller.py ${ARM_CMD[${A}]}"; done
echo "memory shares sum to ${TOTAL_SHARE}"
for A in "${ARMS[@]}"; do   # the common file must suit the arms: stage 1 has no --init_from, stage 2 has one
    MS=$(echo "${ARM_CMD[${A}]}" | grep -o -- '--mle_steps [0-9]*' | tail -1 | cut -d' ' -f2)
    CK=$(echo "${ARM_CMD[${A}]}" | grep -o -- '--init_from [^ ]*' | tail -1 | cut -d' ' -f2)
    if [ -n "${CK}" ] && [ "${MS}" != "0" ]; then
        echo "FATAL: arm ${A} has --init_from and a stage 1 (--mle_steps ${MS}): wrong COMMON for ${ARMS_FILE}?" >&2; exit 1
    fi
    if [ -z "${CK}" ] && [ "${MS}" = "0" ]; then
        echo "FATAL: arm ${A} has no stage 1 (--mle_steps 0) and no --init_from: wrong COMMON for ${ARMS_FILE}?" >&2; exit 1
    fi
done
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
for F in inn_modeller.py energies/ar_transformer.py energies/prior_flow.py tests/models/test_inn_atom_tokens.py data_processing/molecule_subset.py; do
    if [ ! -f ${WORKDIR}/${F} ]; then echo "FATAL: ${WORKDIR}/${F} is missing -- git pull gfn-diffusion" >&2; exit 1; fi
done
if ! grep -q 'class AtomEncoder' ${WORKDIR}/inn_modeller.py || ! grep -q 'def shared_context' ${WORKDIR}/inn_modeller.py; then
    echo "FATAL: gfn-diffusion predates the atom-token condition -- git pull gfn-diffusion" >&2; exit 1
fi
if ! grep -q "add_argument('--molecule_fraction'" ${WORKDIR}/inn_modeller.py || ! grep -q 'def subset_training_molecules' ${WORKDIR}/train.py; then
    echo "FATAL: gfn-diffusion predates --molecule_fraction -- git pull gfn-diffusion" >&2; exit 1
fi
if grep -q -- '--stored_source buffer' ${COMMON_FILE} ${ARMS_FILE} && ! grep -q 'def pbuf_ref_energy' ${WORKDIR}/inn_modeller.py; then
    echo "FATAL: the arms read --stored_source buffer in stage 1, which this gfn-diffusion lacks or predates (pbuf_ref_energy) -- git pull gfn-diffusion" >&2; exit 1
fi
for A in "${ARMS[@]}"; do   # a stage-2 arm: its stage-1 run finished (final.pt) and left its best.pt
    CK=$(echo "${ARM_CMD[${A}]}" | grep -o -- '--init_from [^ ]*' | tail -1 | cut -d' ' -f2)
    if [ -n "${CK}" ] && { [ ! -s "${CK}" ] || [ ! -s "$(dirname "${CK}")/final.pt" ]; }; then
        echo "FATAL: arm ${A}: its stage-1 run has no best.pt or did not finish (no final.pt beside '${CK}')" >&2; exit 1
    fi
done
if ! grep -q '^SG_LOOKUP_TABLES = ' ${WORKDIR}/energies/molecular_crystal.py; then
    echo "FATAL: gfn-diffusion predates the start-up scoring fix (merge_batches and SG_LOOKUP_TABLES, c1fb45a4) -- git pull gfn-diffusion" >&2; exit 1
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
  cat ${ARMS_FILE} ${COMMON_FILE}
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

# one container: the atom-token tests on the CPU and the pinned molecule draw first, then every arm of the job at once
INNER=${J}_inner.sh
{
    echo 'source /ext3/env.sh'
    echo "export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH"
    echo "${NIG_EXPORT}"
    echo 'export GFN_ALLOW_GPU_SHARING=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1'
    echo "python -c \"import mxtaltools.common.sym_utils as s; assert getattr(s, 'NIGGLI_TRICLINIC', True), 'Niggli triclinic penalty is OFF'\" || exit 1"
    echo "python -c \"import inn_modeller\" || { echo 'FATAL: inn_modeller does not import in this container' >&2; exit 1; }"
    echo "python -c \"import inn_modeller as m; s, _ = m.calibrate_gauss_sigma(lambda x: (600 * x * x, 0.05), 6.0, 0.01); assert abs(s - 0.1) < 1e-3\" || { echo 'FATAL: the gauss-tile calibration check fails in this container' >&2; exit 1; }"
    echo "CUDA_VISIBLE_DEVICES=-1 python tests/models/test_inn_atom_tokens.py > ${LOGS}/preflight_${SLURM_JOB_ID}.log 2>&1 || { echo 'FATAL: the atom-token tests fail in this container (see preflight log)' >&2; exit 1; }"
    echo "python -c \"from data_processing.molecule_subset import molecule_u as u; assert u('mol_a') == 0.43388624636742484, u('mol_a')\" || { echo 'FATAL: data_processing.molecule_subset is missing or draws other subsets in this container' >&2; exit 1; }"
    echo 'FAIL=0; PIDS=(); NAMES=()'
    for A in "${ARMS[@]}"; do
        echo "python -u inn_modeller.py ${ARM_CMD[${A}]} > ${LOGS}/${A}_${SLURM_JOB_ID}.trainlog 2>&1 & PIDS+=(\$!); NAMES+=(${A})"
    done
    echo 'for i in "${!PIDS[@]}"; do wait ${PIDS[$i]} && echo "arm ${NAMES[$i]} done" || { echo "arm ${NAMES[$i]} FAILED (see its .trainlog)" >&2; FAIL=1; }; done'
    echo 'exit ${FAIL}'
} > ${INNER}

set -o pipefail   # the job's exit status is the inner script's (a failed arm), not tee's
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /tmp:/tmp \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash ${INNER} 2>&1 | tee ${J}.trainlog
