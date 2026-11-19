#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=test4uni
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# The held-out test file of a run over P-1, P2_1/c, P2_12_12_1 and Pbca, drawn uniformly over the groups (owner
# 2026-11-20): data_processing/molecule_split_files.py heldout_groups, the same 125 held-out molecules (seed 0) in every
# group, at most 8 crystals each within 5 kT of the molecule's floor in that group; <= 4,000 rows, inside the eval's
# whole-set scoring. First rebuilds P-1's held-out anchors under the switched eLJ with the current builder (the file
# beside the P-1 prior was scored on the hard cutoff by a builder that left stale atom pointers on K 0 files). CPU only.
# A step whose output exists is skipped.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
DATA=/scratch/mk8347/data/crystal_datasets/conditional
P1=${DATA}/pool8_nov09/priors
SG=${DATA}/pool8sg_nov15/priors
OUTDIR=${DATA}/pool8sg4_nov20/priors
P1_HELDOUT=${P1}/pool8_p1_heldout_anchors_switched.pt
TEST=${OUTDIR}/pool8_4sg_test_w5.pt
mkdir -p ${OUTDIR}
for f in ${P1}/heldout_list.txt ${SG}/sg14/pool8_sg14_heldout_anchors.pt ${SG}/sg19/pool8_sg19_heldout_anchors.pt \
         ${SG}/sg61/pool8_sg61_heldout_anchors.pt; do
    if [ ! -s ${f} ]; then echo "FATAL: ${f} missing" >&2; exit 1; fi
done
echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)"

# singularity directly, not srun (restamp_cpu_parts.sbatch: srun refuses on some partition-'all' nodes)
singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e
        source /ext3/env.sh
        export PYTHONPATH=${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
        [ -s ${P1_HELDOUT} ] || python -u data_processing/build_noised_prior.py --config configs/mk_dev.yaml \
            --set energy_config.crystal_coordinates=fd --set 'space_groups=[2]' --thermal_scaling_factor 0.36358 \
            --device cpu --basins ${P1}/heldout_list.txt --k 0 --heldout_frac 0 --log_hi -0.375 --out ${P1_HELDOUT}
        [ -s ${TEST} ] || python -u data_processing/molecule_split_files.py heldout_groups ${TEST} 5 \
            ${P1_HELDOUT} ${SG}/sg14/pool8_sg14_heldout_anchors.pt ${SG}/sg19/pool8_sg19_heldout_anchors.pt \
            ${SG}/sg61/pool8_sg61_heldout_anchors.pt --molecules 125 --per_molecule 8 --seed 0
        ls -la ${TEST}
    "
