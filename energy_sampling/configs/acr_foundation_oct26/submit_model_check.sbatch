#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acrmodels
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/acr_foundation_oct26/joblogs/%x_%j.out

# acr_foundation_oct26, the model check: acr_model_check.py on the ten H-bearing acridine CIFs for each candidate energy
# model, one process per model, one after another; a model that fails (missing file, a fairchem too old for its
# checkpoint, out of memory) is reported and the next one runs. Then summarize.py prints one table relative to Form II.
# The job first prints the container's torch / fairchem / mace / ase versions (does UMA 1.2 load here?).
# MACE-OFF24 is fetched from GitHub (github.com/ACEsuit/mace-off, owner-approved 2026-10-26) when it is not yet in
# /scratch/mk8347/models/, and kept only at its published byte count. UMA 1.2 is skipped until it is there.
# Outputs: ${OUT}/<label>.json and <label>.log, summary.txt; this job's log in joblogs/.
# DRY_RUN=1 bash submit_model_check.sbatch prints the model list and stops.
module purge 2>/dev/null

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
HERE=${WORKDIR}/configs/acr_foundation_oct26
OUT=/scratch/mk8347/data/crystal_datasets/acridine/model_check_oct26
CIFS=/scratch/mk8347/data/crystal_datasets/acridine
M=/scratch/mk8347/models
OFF=${M}/MACE-OFF24_medium.model
OFF_BYTES=18375569

# label, kind, checkpoint
MODELS="mace_off24_m mace ${OFF}
uma_s_1p1 uma ${M}/uma/uma-s-1p1.pt
uma_m_1p1 uma ${M}/uma/uma-m-1p1.pt
uma_s_1p2p1 uma ${M}/uma/uma-s-1p2p1.pt
mace_acr_old mace /scratch/mk8347/data/acr_112025_mh1_stagetwo.model
mace_acr_new mace /scratch/mk8347/data/acr_newmodel.model"

echo "${MODELS}" | while read -r L K P; do
    [ -f "${P}" ] && echo "model ${L} (${K}): ${P} $(stat -c %s ${P}) bytes" || echo "model ${L} (${K}): ${P} MISSING"
done
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${OUT}
if [ "$(stat -c %s ${OFF} 2>/dev/null || echo 0)" != "${OFF_BYTES}" ]; then
    curl -sL -o ${OFF}.part "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model" \
        && [ "$(stat -c %s ${OFF}.part)" = "${OFF_BYTES}" ] && mv ${OFF}.part ${OFF} \
        || { rm -f ${OFF}.part; echo "MACE-OFF24 not fetched: it will be skipped"; }
fi

INNER=${HERE}/joblogs/inner_${SLURM_JOB_ID}.sh
{
    echo 'source /ext3/env.sh'
    echo 'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8'
    echo "python -c \"import torch, ase; print('torch', torch.__version__, 'ase', ase.__version__)\""
    echo "python -c \"import fairchem.core, importlib.metadata as m; print('fairchem-core', m.version('fairchem-core'))\" || echo 'fairchem not importable'"
    echo "python -c \"import importlib.metadata as m; print('mace-torch', m.version('mace-torch'))\" || echo 'mace not importable'"
    echo "${MODELS}" | while read -r L K P; do
        echo "if [ -f ${P} ]; then python -u ${HERE}/acr_model_check.py ${K} ${P} ${L} --cif_dir ${CIFS} --out_dir ${OUT} > ${OUT}/${L}.log 2>&1 && echo '${L}: done' || echo '${L}: FAILED (see ${OUT}/${L}.log)'; else echo '${L}: skipped, no checkpoint'; fi"
    done
    echo "python ${HERE}/summarize.py ${OUT} | tee ${OUT}/summary.txt"
} > ${INNER}

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${HERE} \
    ${IMAGE} \
    /bin/bash ${INNER}
