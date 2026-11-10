#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8priorcpu
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out
# CPU-ONLY VARIANT of build_pool8_prior.sbatch (2026-11-10): the same steps and files, no GPU. On a GPU the job
# idles the A100 for ~1 h loading the training anchors (CPU-only MolCrystalData construction), and Torch's
# low-GPU-utilisation policy cancelled two builds at ~2 h (22491177, 22543962). A CPU job is outside that policy.
# Estimate 8-15 h. K is required at submit:  K=4 sbatch configs/pool8_prior_nov09/build_pool8_prior_cpu.sbatch

# pool8_nov09 prior: the files configs/pool8_mle_nov10 reads, built from the band sets of MXtalTools
# configs/crystal_searches/pool8_nov09 (band_pool8_nov09.sbatch -> BAND/<split>/pool8_c*_band, 32 picks per molecule).
#   1. list files, one per split (heldout = heldout_random + heldout_scaffold)
#   2. HELDOUT: the held-out molecules' anchors, scored, no children, none held out within (K 0, --heldout_frac 0)
#   3. PRIOR:   the training molecules' anchors + K children each, FD chart, metric kick, held-out crystals of training
#               molecules by descent group (--heldout_frac 0.05, the builder's default)
#   4. CONDITIONS and TEST (data_processing/molecule_split_files.py)
# eLJ scale 0.36358, the MIPCAS and norbornane priors' (the search kT 6.86 raw eLJ x 0.36358 = 2.49 kJ/mol ~ T 2.5).
# K is required at submit:  K=4 sbatch configs/pool8_prior_nov09/build_pool8_prior.sbatch
# A step whose output exists is skipped, so resubmitting finishes what is left.
# Local check (task 0, 2026-11-09): 1.1 KB per row on disk; load 2.5 ms per anchor (one CPU core), child scoring
# ~0.8 ms per row on a shared GPU, builder peak RAM ~3.2 GB + ~1.8 KB per row: K 4 ~5 h, K 10 ~8 h projected.
module purge

K=${K:?set K, children per anchor: K=4 sbatch ...}
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BASE=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09
BAND=${BASE}/band32
OUT=${BASE}/priors
TSF=0.36358
PRIOR=pool8_p1_train_k${K}_metric.pt
HELDOUT=pool8_p1_heldout_anchors.pt
mkdir -p ${OUT}

NDONE=$(ls ${BAND}/done 2>/dev/null | wc -l)
if [ "${NDONE}" != "2899" ]; then echo "FATAL: ${BAND}/done holds ${NDONE} of 2899 tasks" >&2; exit 1; fi
ls -d ${BAND}/train/pool8_c*_band > ${OUT}/train_list.txt
ls -d ${BAND}/heldout_random/pool8_c*_band ${BAND}/heldout_scaffold/pool8_c*_band > ${OUT}/heldout_list.txt
echo "band sets: $(wc -l < ${OUT}/train_list.txt) train, $(wc -l < ${OUT}/heldout_list.txt) held out; K ${K}"
echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse --short=12 HEAD)"
( while sleep 120; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used " $3 " GB, available " $7 " GB"}')"; done ) > ${OUT}/build_k${K}_${SLURM_JOB_ID}_ram.txt &
RAM_PID=$!
trap "kill ${RAM_PID} 2>/dev/null" EXIT TERM

srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e
        source /ext3/env.sh
        # not ${WORKDIR}: the data_processing scripts put it first themselves only when it is absent; on PYTHONPATH it
        # sits behind the script's folder and data_processing/utils.py shadows energy_sampling/utils.py
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 PYTHONUNBUFFERED=1
        B='python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml
           --set energy_config.crystal_coordinates=fd --thermal_scaling_factor ${TSF} --device cpu'
        [ -s ${OUT}/${HELDOUT} ] || \$B --basins ${OUT}/heldout_list.txt --k 0 --heldout_frac 0 --out ${OUT}/${HELDOUT}
        [ -s ${OUT}/${PRIOR} ] || \$B --basins ${OUT}/train_list.txt --k ${K} --child_kernel metric --out ${OUT}/${PRIOR}
        [ -s ${OUT}/pool8_p1_train_conditions.pt ] || python -u data_processing/molecule_split_files.py conditions \
            ${OUT}/${PRIOR} ${OUT}/pool8_p1_train_conditions.pt
        [ -s ${OUT}/pool8_p1_test_w5.pt ] || python -u data_processing/molecule_split_files.py heldout \
            ${OUT}/${HELDOUT} ${OUT}/pool8_p1_test 5 --per_molecule 8 --max_molecules 500
        ls -la ${OUT}
    "
