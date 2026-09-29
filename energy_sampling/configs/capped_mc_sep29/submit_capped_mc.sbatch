#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-15
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=cmc
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/capped_mc_sep29/joblogs/%x_%A_%a.out

# capped_mc_sep29: capped Metropolis walkers (data_processing/capped_mc.py) from each MLIP prior's search minima, on the
# training energy (UMA / MACE), for the thermal-band prior rebuild. One task = one (system, shard) row of INDEX.tsv
# (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue every shard from its state.pt (--resume).
# PILOT=1 sbatch ... runs a small version (64 seeds, 20 steps) into <name>_pilot/ to measure throughput first.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/capped_mc_sep29
LOGS=${ARMS}/joblogs
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors
OUTROOT=/scratch/mk8347/data/crystal_datasets/capped_mc
mkdir -p ${LOGS} ${OUTROOT}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
INDEX=${ARMS}/INDEX.tsv
NAME=$(awk -F'\t' -v n=${ROW} 'NR==n {print $2}' ${INDEX})
SHARD=$(awk -F'\t' -v n=${ROW} 'NR==n {print $3}' ${INDEX})
NSH=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${INDEX})
PRIOR=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${INDEX})
PRIOR_BYTES=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${INDEX})
EF=$(awk -F'\t' -v n=${ROW} 'NR==n {print $7}' ${INDEX})
MLIP=$(awk -F'\t' -v n=${ROW} 'NR==n {print $8}' ${INDEX})
if [ -z "${NAME}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi

HAVE=$(stat -c %s ${DATA}/${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES}" >&2; exit 1
fi
if [ ! -f "${MLIP}" ]; then echo "FATAL: MLIP checkpoint ${MLIP} missing" >&2; exit 1; fi

if [ -n "${PILOT:-}" ]; then
    OUT=${OUTROOT}/${NAME}_pilot/shard_${SHARD}
    EXTRA="--max_seeds 64 --steps 20 --adapt_end 12 --cov_updates 4,8 --shard_every 10 --log_every 5"
else
    OUT=${OUTROOT}/${NAME}/shard_${SHARD}
    EXTRA=""
fi
mkdir -p ${OUT}
J=${LOGS}/${NAME}_s${SHARD}_${SLURM_JOB_ID}
echo "array ${SLURM_ARRAY_TASK_ID} -> ${NAME} shard ${SHARD}/${NSH} (${EF}) -> ${OUT} ${PILOT:+[PILOT]}"

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export GPU_MEM_FRACTION=0.9
        python -u data_processing/capped_mc.py --prior ${DATA}/${PRIOR} --energy_function ${EF} --mlip_path ${MLIP} \
            --out ${OUT} --n_shards ${NSH} --shard ${SHARD} --resume ${EXTRA}
    " 2>&1 | tee ${J}.log
