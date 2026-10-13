#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=inn12
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_oct12/joblogs/%x_%A_%a.out

# inn_oct12: the exact-density flow sampler (inn_modeller.py, a prototype) in place of the SDE policy and P_B, one
# seed per arm. Arms 0-4 are conditional QM9 at the settings of the GFN run qf30_frc_ctl (eLJ, T 6.9, per-molecule
# energy reference); arms 5-7 are MIPCAS eLJ at the settings of p20_mip_n5 (T 2.5). Every arm trains its own
# likelihood stage from the same seed, then the energy stage.
#
#   index  arm                  what it adds
#   0      qm9_base             30M-parameter flow; energy stage = TB residual on detached draws and on stored rows
#   1      qm9_path             arm 0 + the pathwise energy-gradient term (differentiable draws, the energy's force)
#   2      qm9_path_noise       arm 1 + stored rows displaced and scored again every step
#   3      qm9_path_noise_temp  arm 2 + temperature as a flow input, trained over T 6.9 to 27.6, read at 6.9
#   4      qm9_path_noise_iw    arm 2 + importance-weighted likelihood of the flow's own draws, 8 per molecule
#   5      mip_base             3.4M-parameter flow, 30,000 energy steps
#   6      mip_path             arm 5 + the pathwise term at weight 0.1
#   7      mip_temp             arm 5 + temperature as a flow input, trained over T 2.5 to 10, read at 2.5
#
# QM9 arms end their energy stage on the clock, 40 h after the loop starts, with the rate schedule stretched over that
# time: the arms compare at equal wall time, not equal steps. MIPCAS arms run a fixed step count and finish early.
# The base config is the checked-out configs/mk_dev.yaml; the route settings are the flags below. The run uses no
# checkpoint and writes none: outputs are artifacts/inn_proto/inn12_<arm>/ (metrics.jsonl, latest.pt, mle_exit.pt,
# final.pt) and wandb runs inn12_<arm>. THERE IS NO RESUME: resubmitting an arm starts it again and overwrites those.
# DRY_RUN=1 SLURM_ARRAY_TASK_ID=<n> bash submit_inn_oct12.sbatch prints an arm's command and stops.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
LOGS=${WORKDIR}/configs/inn_oct12/joblogs
CKPTS=${WORKDIR}/checkpoints
DATA=/scratch/mk8347/data/crystal_datasets/conditional/priors

QM9_ROUTE="--prior_path ${DATA}/qm9full_prior.pt --molecules_path ${DATA}/qm9full_conditions.pt --test_molecules_path ${DATA}/qm9full_test_conditions.pt --set embedding_conditioning=true --set energy_config.temperature=6.9 --set energy_config.reward_range=null --set energy_config.energy_reference=seed_min"
QM9_FLOW="--n_blocks 8 --n_bins 24 --hidden 1024 --hidden_layers 3 --context_dim 256 --context_hidden 1024"
QM9_STAGES="--mle_steps 20000 --mle_batch 8192 --lr_mle 5e-4 --mle_noise_log_range -2.5 -1.5 --batch 4000 --lr_energy 1e-4 --lr_energy_final 1e-5 --energy_warmup 200 --beta_data 10 --budget_minutes 2400 --eval_every 1000 --figs_every 5"
# name:bytes. The prior's size is the one the GFN batteries check (their INDEX files) and is enforced. The
# conditions files must exist; their sizes are the dev box's copies, and a difference is printed, not refused.
QM9_FILES="qm9full_prior.pt:2433322986"
QM9_NOTED="qm9full_conditions.pt:250619181 qm9full_test_conditions.pt:13219376"
MIP_ROUTE="--prior_path ${DATA}/mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2.pt"
MIP_FLOW="--n_blocks 6 --n_bins 16 --hidden 512 --hidden_layers 2"
MIP_STAGES="--mle_steps 30000 --mle_batch 8192 --lr_mle 1e-3 --energy_steps 30000 --batch 2000 --lr_energy 3e-4 --lr_energy_final 1e-5 --energy_warmup 500 --beta_data 80 --eval_every 1000 --figs_every 5"
MIP_FILES="mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2.pt:165617107"
MIP_NOTED=""
PATH_QM9="--w_path 1 --path_batch 1024"
NOISE="--data_noise_log_range -3 -1.5"

case "${SLURM_ARRAY_TASK_ID}" in
    0) ARM=qm9_base;            SYS=QM9; EXTRA="" ;;
    1) ARM=qm9_path;            SYS=QM9; EXTRA="${PATH_QM9}" ;;
    2) ARM=qm9_path_noise;      SYS=QM9; EXTRA="${PATH_QM9} ${NOISE}" ;;
    3) ARM=qm9_path_noise_temp; SYS=QM9; EXTRA="${PATH_QM9} ${NOISE} --t_factor 4 --t_target_frac 0.25" ;;
    4) ARM=qm9_path_noise_iw;   SYS=QM9; EXTRA="${PATH_QM9} ${NOISE} --w_iw 5 --iw_k 8" ;;
    5) ARM=mip_base;            SYS=MIP; EXTRA="" ;;
    6) ARM=mip_path;            SYS=MIP; EXTRA="--w_path 0.1 --path_batch 512" ;;
    7) ARM=mip_temp;            SYS=MIP; EXTRA="--t_factor 4 --t_target_frac 0.25" ;;
    *) echo "FATAL: no arm at array index '${SLURM_ARRAY_TASK_ID}'" >&2; exit 1 ;;
esac
ROUTE_VAR=${SYS}_ROUTE; FLOW_VAR=${SYS}_FLOW; STAGES_VAR=${SYS}_STAGES; FILES_VAR=${SYS}_FILES; NOTED_VAR=${SYS}_NOTED
# max_minutes is the hard stop behind the 40 h budget; the wall above is 48 h, which leaves the prior scan at
# startup (about 15 min on QM9, twice that with temperature conditioning) and the last eval their room.
ARGS="--config configs/mk_dev.yaml --tag inn12 --run_name ${ARM} --set checkpoints_dir=${CKPTS} --set gpu_util_sample_period_s=60 --cuda_memory_fraction 0.97 --max_minutes 2640 --oom_retries 3 --eval_samples 10000 --log_every 50 ${!ROUTE_VAR} ${!FLOW_VAR} ${!STAGES_VAR} ${EXTRA}"
echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}"
echo "python -u inn_modeller.py ${ARGS}"
if [ -n "${DRY_RUN:-}" ]; then exit 0; fi

mkdir -p ${LOGS}
J=${LOGS}/${ARM}_${SLURM_JOB_ID}

# THE WALLS. The Niggli priors are filtered against the monoclinic class walls and the Niggli triclinic penalty;
# older MXtalTools reads MXT_NIGGLI_TRICLINIC, newer refuses the variable. As the GFN batteries' job scripts.
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
for ENTRY in ${!FILES_VAR}; do
    FILE=${ENTRY%%:*}; BYTES=${ENTRY##*:}
    HAVE=$(stat -c %s ${DATA}/${FILE} 2>/dev/null || echo 0)
    if [ "${HAVE}" != "${BYTES}" ]; then
        echo "FATAL: ${DATA}/${FILE} is ${HAVE} bytes, expected ${BYTES}" >&2; exit 1
    fi
done
for ENTRY in ${!NOTED_VAR}; do
    FILE=${ENTRY%%:*}; BYTES=${ENTRY##*:}
    if [ ! -f ${DATA}/${FILE} ]; then echo "FATAL: ${DATA}/${FILE} is missing" >&2; exit 1; fi
    HAVE=$(stat -c %s ${DATA}/${FILE})
    if [ "${HAVE}" != "${BYTES}" ]; then echo "NOTE: ${DATA}/${FILE} is ${HAVE} bytes; the dev box copy is ${BYTES}"; fi
done
if [ ! -f ${WORKDIR}/inn_modeller.py ]; then
    echo "FATAL: ${WORKDIR}/inn_modeller.py is missing -- git pull gfn-diffusion" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
  echo "gfn-diffusion $(git -C ${PROJECT_ROOT}/gfn-diffusion rev-parse HEAD 2>&1)  MXtalTools $(git -C ${PROJECT_ROOT}/MXtalTools rev-parse HEAD 2>&1)"
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
        python -c \"import inn_modeller\" || { echo 'FATAL: inn_modeller does not import in this container (a missing module, or gfn-diffusion / MXtalTools not pulled)' >&2; exit 1; }
        python -u inn_modeller.py ${ARGS}
    " 2>&1 | tee ${J}.trainlog
