#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acrprior
#SBATCH --array=0-4
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out

# acr_nov25 priors (make.py): one array task per model, in MODELS order. Each makes
#   0. BAND        the model's band set from its search trajectories (MXtalTools
#                  configs/crystal_searches/acr_mlip_nov19/band.py: 10 kT, 0.1 A net, C2 images, recorded energies, no
#                  MLIP call), unless BAND exists; then turns it into
#   1. PRIOR       acr_<model>_k0.pt: every anchor at the energy its search recorded (--stored_energies: the trainer's
#                  analysis computes the other terms, no MLIP call; 256 anchors re-scored through the MLIP as the check),
#                  no forces (owner 2026-11-25: not needed); no kicked children (--k 0: MLE draws the live physical
#                  kernel), each anchor written again as its own radius-0 child (--self_children: the TB stage's backward rows), held-out crystals by descent group
#                  (the builder's default 0.05; a state and its C2 image share their group). The build refuses when a
#                  re-scored check anchor misses its recorded energy by more than ETOL.
#   2. CONDITIONS  acr_<model>_k0_conditions.pt (data_processing/molecule_split_files.py conditions).
# The run config is the model's arm (configs/acr_nov25/<model>.yaml): its energy function, MLIP and D3 settings.
# A step whose output exists is skipped, so resubmitting finishes what is left.
module purge

MODELS=(macecspl maceoff24m umas1p1 umas1p2 esens)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:?array task id}]}
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
BAND=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_mlip_nov19/${MODEL}_band_10kT_0.1A_c2
OUT=/scratch/mk8347/data/crystal_datasets/acridine/priors/acr_nov25
PRIOR=acr_${MODEL}_k0.pt
case ${MODEL} in   # the container's fairchem 2.13 cannot build UMA 1.2; it has no torch-dftd (MACE-CSP's D3)
    macecspl) PYEXTRA=/scratch/mk8347/pylibs_dftd ;;
    umas1p2)  PYEXTRA=/scratch/mk8347/pylibs_fc216 ;;
    *)        PYEXTRA= ;;
esac
# --energy_tol (kJ/mol): the build refuses when a re-scored check anchor misses the search's recorded energy by more.
# MACE-CSP re-scores to 0.0014 at most (2,502-row test, dev box). The UMA-s 1.1 trainer energy differs from the search's
# by a median 0.02 and up to ~1.1 (2,986-row test, states and C2 images alike; cause not pinned down), so the UMA family
# takes 2.0; the bar is there to catch a wrong crystal, which misses by far more.
case ${MODEL} in
    umas1p1|umas1p2|esens) ETOL=2.0 ;;
    *)                     ETOL=0.05 ;;
esac
mkdir -p ${OUT}
SEARCH=$(dirname ${BAND})/${MODEL}_search_traj.pt
if [ ! -s ${BAND}/meta.json ] && [ ! -s ${SEARCH} ]; then echo "FATAL: no band set ${BAND} and no trajectories ${SEARCH}" >&2; exit 1; fi
if [ ! -s ${WORKDIR}/configs/acr_nov25/${MODEL}.yaml ]; then echo "FATAL: no config ${MODEL}.yaml (git pull)" >&2; exit 1; fi
echo "acr_nov25 prior ${MODEL}: band ${BAND}; gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse --short=12 HEAD)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse --short=12 HEAD)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347:/scratch/mk8347 \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        set -e
        source /ext3/env.sh
        export PYTHONPATH=${PYEXTRA:+${PYEXTRA}:}${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion:\$PYTHONPATH
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
        [ -s ${BAND}/meta.json ] || python -u ${PROJECT_ROOT}/MXtalTools/configs/crystal_searches/acr_mlip_nov19/band.py ${MODEL}             --device cuda
        [ -s ${OUT}/${PRIOR} ] || python -u data_processing/build_noised_prior.py --config configs/acr_nov25/${MODEL}.yaml \
            --basins ${BAND} --k 0 --self_children --stored_energies --child_kernel metric --device cuda --chunk 4096 --energy_tol ${ETOL} --out ${OUT}/${PRIOR}
        [ -s ${OUT}/acr_${MODEL}_k0_conditions.pt ] || python -u data_processing/molecule_split_files.py conditions \
            ${OUT}/${PRIOR} ${OUT}/acr_${MODEL}_k0_conditions.pt
        ls -la ${OUT}
    "
