#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-399
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=cdb
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/conformer_db_sep29/joblogs/%x_%A_%a.out

# conformer_db_sep29: the offline QM9 conformer database (build_conformer_database.py), CPU only, one
# shard per array task (shard = SLURM_ARRAY_TASK_ID of 400). WRITTEN BY make.py: do
# not edit by hand, rerun make.py. Resubmit this same file to continue every shard from its
# file; the builder refuses a shard file written under another header.
# PILOT=1 sbatch --array=0-3 --time=01:00:00 configs/conformer_db_sep29/submit_db.sbatch runs
# --max-molecules 5 per shard into /scratch/mk8347/data/conformer_datasets/qm9_db_sep29_pilot.
# Report: python build_conformer_database.py summarize <output dir>
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/conformer_db_sep29
LOGS=${ARMS}/joblogs
N_SHARDS=400
POOL_NEED=0
PRIOR=${WORKDIR}/conformer_prior_v2.pt
PRIOR_BYTES=319791
PRIOR_SHA256=e6c2953f2f874f83f5fab84793d1339288f6efd26a3bc157d2c8251f0ddb6153
INDEX=${ARMS}/qm9_index.tsv.gz
INDEX_SHA256=7c88ed2437c11efe03747d66ff5a97259ea0c2338cad9343b5825b687c9125c0
mkdir -p ${LOGS}

SHARD=${SLURM_ARRAY_TASK_ID}
if [ -z "${SHARD}" ]; then echo "FATAL: no SLURM_ARRAY_TASK_ID; submit this file as an array" >&2; exit 1; fi

HAVE=$(stat -c %s ${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: the prior at ${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES} (sha256 ${PRIOR_SHA256}); upload conformer_prior_v2.pt there" >&2; exit 1
fi
HAVE=$(sha256sum ${PRIOR} | cut -d' ' -f1)
if [ "${HAVE}" != "${PRIOR_SHA256}" ]; then
    echo "FATAL: the prior at ${PRIOR} has sha256 ${HAVE}, expected ${PRIOR_SHA256}" >&2; exit 1
fi
if [ ! -f "${INDEX}" ]; then echo "FATAL: the QM9 index table ${INDEX} is missing (pull the repository)" >&2; exit 1; fi
HAVE=$(sha256sum ${INDEX} | cut -d' ' -f1)
if [ "${HAVE}" != "${INDEX_SHA256}" ]; then
    echo "FATAL: ${INDEX} has sha256 ${HAVE}, expected ${INDEX_SHA256}" >&2; exit 1
fi

if [ -n "${PILOT:-}" ]; then
    OUT=/scratch/mk8347/data/conformer_datasets/qm9_db_sep29_pilot
    EXTRA="--max-molecules 5"
else
    OUT=/scratch/mk8347/data/conformer_datasets/qm9_db_sep29
    EXTRA=""
fi
mkdir -p ${OUT}
J=${LOGS}/shard${SHARD}_${SLURM_JOB_ID}
echo "array ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}/${N_SHARDS} -> ${OUT} ${PILOT:+[PILOT]}"

{ scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
} > ${J}.info 2>&1

srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        python -u build_conformer_database.py build --source ${INDEX} --prior ${PRIOR} \
            --pool-need ${POOL_NEED} --config configs/conformer_mk.yaml --out-dir ${OUT} \
            --shard ${SHARD} --n-shards ${N_SHARDS} --threads 1 ${EXTRA}
    " 2>&1 | tee ${J}.log
