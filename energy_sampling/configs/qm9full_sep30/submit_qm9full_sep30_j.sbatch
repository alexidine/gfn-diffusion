#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-6
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=qf30j
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/qm9full_sep30/joblogs/%x_%A_%a.out

# qm9full_sep30 leg j: conditional MLE from scratch at a pinned batch, the force term in P_F crossed with the state
# encoder that reads atoms (make.py, LEG J). ONE JOB RUNS THE ARMS OF ONE LINE OF GROUPS_j.tsv ON ONE GPU:
# array index n = line n + 1 of that file (no header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue a group past the wall: each arm resumes its own _running.pt.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/qm9full_sep30
LOGS=${ARMS}/joblogs
CKPTS=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
mkdir -p ${LOGS}

GROUP=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' ${ARMS}/GROUPS_j.tsv)
if [ -z "${GROUP}" ]; then echo "no group at line ${SLURM_ARRAY_TASK_ID} of GROUPS_j.tsv" >&2; exit 1; fi

# THE NIGGLI PENALTY. Older MXtalTools reads MXT_NIGGLI_TRICLINIC; newer makes Niggli the only rule
# and refuses the variable. Anything else cannot score these priors.
SYM=${PROJECT_ROOT}/MXtalTools/mxtaltools/common/sym_utils.py
if grep -q 'MXT_NIGGLI_TRICLINIC is retired' ${SYM}; then
    NIG_EXPORT=""
elif grep -q "os.environ.get('MXT_NIGGLI_TRICLINIC'" ${SYM}; then
    NIG_EXPORT="export MXT_NIGGLI_TRICLINIC=1;"
else
    echo "FATAL: ${SYM} has no triclinic Niggli penalty -- git pull MXtalTools" >&2; exit 1
fi
if [ ! -f ${ARMS}/run_group.sh ]; then echo "FATAL: ${ARMS}/run_group.sh is missing -- git pull gfn-diffusion" >&2; exit 1; fi

# EACH ARM: its checks, and its config with the start resolved. A fresh first launch; a resubmission resumes
# the arm's own _running.pt (checkpoint_name is null in every row).
LIVE=""
for ARM in ${GROUP}; do
    LINE=$(awk -F'\t' -v a=${ARM} '$1==a' ${ARMS}/INDEX_j.tsv)
    if [ -z "${LINE}" ]; then echo "FATAL: ${ARM} is not a row of INDEX_j.tsv" >&2; exit 1; fi
    PRIOR=$(echo "${LINE}" | cut -f5)
    PRIOR_BYTES=$(echo "${LINE}" | cut -f6)
    CONFIG=${ARMS}/${ARM}.yaml
    if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi
    if [ -f ${CKPTS}/${ARM}.dead ]; then
        echo "arm ${ARM} aborted UNRECOVERABLE on an earlier launch -- skipping"
        continue
    fi
    HAVE=$(stat -c %s ${DATA}/${PRIOR} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
        echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES} -- upload unfinished or a different file" >&2
        exit 1
    fi
    # THE TRUNK (rows whose state encoder reads atoms, or with a force term) and the code that reads it.
    if grep -q -E '^  (state_atoms: [1-9]|force_drift_fwd: [0-9])' ${CONFIG}; then
        TRUNK=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/intra_trunk_oct06/st06_c2_ft_late.pt
        if [ ! -s "${TRUNK}" ]; then
            echo "FATAL: ${TRUNK} is missing -- the stacked trunk (configs/intra_trunk_oct06/submit_inter.sbatch)" >&2; exit 1
        fi
        if ! grep -q 'def state_info' ${WORKDIR}/models/crystal_force.py; then
            echo "FATAL: models/crystal_force.py has no state features -- git pull gfn-diffusion" >&2; exit 1
        fi
        if ! grep -q 'max_pairs' ${PROJECT_ROOT}/MXtalTools/mxtaltools/crystal_building/image_pairs.py; then
            echo "FATAL: MXtalTools' image_pairs.py has no pair cap -- git pull MXtalTools" >&2; exit 1
        fi
    fi
    # THE FORCE TERM: the gate inside the cap (a gate that starts at zero has no gradient without it).
    if grep -q '^  force_drift_fwd: [0-9]' ${CONFIG}; then
        if ! grep -q 'mean_sq = self._live_only' ${WORKDIR}/models/gfn.py; then
            echo "FATAL: models/gfn.py caps the force step before its gate -- git pull gfn-diffusion" >&2; exit 1
        fi
    fi
    RESOLVED=${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
    OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
    if [ -n "${OWN}" ]; then
        echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
        sed -e "s|CONTINUE_PLACEHOLDER|true|" ${CONFIG} > ${RESOLVED}
    else
        echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH"
        sed -e "s|CONTINUE_PLACEHOLDER|false|" ${CONFIG} > ${RESOLVED}
    fi
    if grep -q 'CONTINUE_PLACEHOLDER' ${RESOLVED}; then
        echo "FATAL: placeholder left in ${RESOLVED}" >&2; exit 1
    fi
    LIVE="${LIVE} ${ARM}"
done
N_LIVE=$(echo ${LIVE} | wc -w)
if [ "${N_LIVE}" -eq 0 ]; then echo "every arm of this group is marked dead -- nothing to run"; exit 0; fi

# MEMORY: an arm alone keeps its config's cuda_memory_fraction; arms sharing the card split 0.94 of it.
if [ "${N_LIVE}" -gt 1 ]; then
    FRAC=$(awk -v n=${N_LIVE} 'BEGIN {printf "%.3f", 0.94 / n}')
    for ARM in ${LIVE}; do
        RESOLVED=${LOGS}/${ARM}_${SLURM_JOB_ID}.yaml
        if [ "$(grep -c '^cuda_memory_fraction: ' ${RESOLVED})" != "1" ]; then
            echo "FATAL: ${RESOLVED} does not have exactly one cuda_memory_fraction line" >&2; exit 1
        fi
        sed -i "s|^cuda_memory_fraction: .*|cuda_memory_fraction: ${FRAC}|" ${RESOLVED}
        echo "arm ${ARM}: cuda_memory_fraction ${FRAC} (${N_LIVE} arms on this card)"
    done
fi

J=${LOGS}/qf30j_${SLURM_JOB_ID}
{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "arms:${LIVE}"
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
        python -c \"from models.crystal_force import CrystalDriftForce, TrunkForce\" || exit 1
        bash ${ARMS}/run_group.sh ${LOGS} ${SLURM_JOB_ID} ${J}.share ${LIVE}
    "

for ARM in ${LIVE}; do
    LOG=${LOGS}/${ARM}_${SLURM_JOB_ID}.trainlog
    echo "===== ${ARM}: last lines of ${LOG}"
    tail -n 25 ${LOG} 2>/dev/null
    if grep -q 'UNRECOVERABLE' ${LOG} 2>/dev/null; then
        echo "arm ${ARM} exhausted its rewind budget -- sentinel set, resubmissions will skip"
        touch ${CKPTS}/${ARM}.dead
    fi
done
