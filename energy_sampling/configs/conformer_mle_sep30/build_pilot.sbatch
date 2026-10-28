#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=31G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mle30b_pilot
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/conformer_mle_sep30/joblogs/%x_%j.out

# conformer_mle_sep30 BUILD, rung pilot: 5 train / 2 held-out molecules cut from the
# QM9 conformer database at 16 rows per condition, with both reference tables, into
# /scratch/mk8347/data/conformer_datasets/rungs_sep30/pilot. CPU only, IN THE TRAINING CONTAINER. Members are built from the database's stored
# reference conformers, and the training run builds each member from the conditions file's.
# WRITTEN BY make.py: do not edit by hand. Rerunning skips a set that has its manifest and a
# table already written (the reference builder resumes from its .parts directory).
# Projected: 0.0 CPU-hours; both builders run one worker per CPU of the job
# (SLURM_CPUS_PER_TASK, 16 as submitted here). run_pilot.sbatch runs this file as a script.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
LOGS=${WORKDIR}/configs/conformer_mle_sep30/joblogs
DB=/scratch/mk8347/data/conformer_datasets/qm9_db_sep29
OUT=/scratch/mk8347/data/conformer_datasets/rungs_sep30/pilot
NCPU=${SLURM_CPUS_PER_TASK:-16}
mkdir -p ${LOGS}
J=${LOGS}/build_pilot_${SLURM_JOB_ID}

# INPUT GUARD: the database, and the committed files the builders read, as make.py saw them at HEAD
NSH=$(ls ${DB}/shard_*_of_0400.pt 2>/dev/null | wc -l)
if [ "${NSH}" != "400" ]; then echo "FATAL: ${DB} holds ${NSH} shard files, expected 400" >&2; exit 1; fi
check() {  # path bytes sha256
    local HAVE
    HAVE=$(stat -c %s ${WORKDIR}/$1 2>/dev/null || echo 0)
    if [ "${HAVE}" != "$2" ]; then echo "FATAL: ${WORKDIR}/$1 is ${HAVE} bytes, expected $2 (git pull)" >&2; exit 1; fi
    if [ "$(sha256sum ${WORKDIR}/$1 | cut -d' ' -f1)" != "$3" ]; then echo "FATAL: ${WORKDIR}/$1 is not sha256 $3 (git pull)" >&2; exit 1; fi
}
check configs/conformer_db_sep29/qm9_index.tsv.gz 936687 7c88ed2437c11efe03747d66ff5a97259ea0c2338cad9343b5825b687c9125c0
check 'models/results/encoder_ckpt/mp+attn+spd_n20000_s0.pt' 3251767 0e5d57ca187d680c0f9ca7de108b6ac537ae75be81f980fd544da543ca880e2e
check conformer_prior_v2.pt 319791 e6c2953f2f874f83f5fab84793d1339288f6efd26a3bc157d2c8251f0ddb6153
mkdir -p ${OUT}

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
        if [ -f ${OUT}/manifest.json ]; then
            echo 'SET: ${OUT}/manifest.json exists -- the set is built, skipping build_conformer_set.py'
        else
            python -u build_conformer_set.py --out-dir ${OUT} --source configs/conformer_db_sep29/qm9_index.tsv.gz \
                --config configs/conformer_mk.yaml --n-train 5 --n-heldout 2 \
                --database ${DB} --rows-per-condition-cap 16 --threads 1 \
                --workers ${NCPU} || exit 1
        fi
        for SIDE in train heldout; do
            TABLE=${OUT}/conditions_\${SIDE}.references.pt
            if [ -f \${TABLE} ]; then echo \"REFERENCES: \${TABLE} exists, skipping\"; continue; fi
            python -u build_conformer_references.py --conditions ${OUT}/conditions_\${SIDE}.pt \
                --config configs/conformer_mk.yaml --database ${DB} --workers ${NCPU} --threads 1
            RC=\$?
            # 1 = the table is written and some conditions failed (a condition the database cannot
            # floor, recorded in the table's failures); anything else is a failed build
            if [ \${RC} -ne 0 ] && { [ \${RC} -ne 1 ] || [ ! -f \${TABLE} ]; }; then
                echo \"FATAL: build_conformer_references.py exited \${RC} on \${SIDE}\" >&2; exit 1
            fi
            [ \${RC} -eq 1 ] && echo \"REFERENCES: \${SIDE} table written WITH FAILURES (see above)\"
        done
        python -c \"import json; m = json.load(open('${OUT}/manifest.json')); print('BUILT', m['rungs'], 'rdkit', m['versions']['rdkit'], 'prior rows', m['prior'].get('rows'))\"
    " 2>&1 | tee ${J}.log
exit ${PIPESTATUS[0]}
