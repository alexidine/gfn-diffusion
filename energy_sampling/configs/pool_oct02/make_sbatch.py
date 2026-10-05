"""pool_oct02: writes the six job scripts (submit_premerge / export / polish / merge / flood / assemble .sbatch) from one shared
header and container call, so the stages cannot drift apart. Run before make.py (which sets their --array lines).

    python configs/pool_oct02/make_sbatch.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = '''#!/bin/bash
#SBATCH --time={time}
{gres}#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --tasks-per-node=1
{signal}#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name={job}
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool_oct02/joblogs/%x_%A_%a.out

{doc}
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${{PROJECT_ROOT}}/MXtalTools
WORKDIR=${{PROJECT_ROOT}}/gfn-diffusion/energy_sampling
ARMS=${{WORKDIR}}/configs/pool_oct02
PYLIBS=/scratch/mk8347/pylibs
DATA=/scratch/mk8347/data/crystal_datasets/pooled_oct02
PRIORS=/scratch/mk8347/data/crystal_datasets/conditional/priors
UMA=/scratch/mk8347/models/uma/esen_s.pt
'''
SYSROW = '''
ROW=$((SLURM_ARRAY_TASK_ID + 2))
read -r NAME SG KEY TSF MOL PRE WINDOW RADIUS CEILING <<< "$(awk -F'\\t' -v n=${ROW} 'NR==n {print $2, $3, $4, $5, $6, $7, $8, $9, $10}' ${ARMS}/SYSTEMS.tsv)"
if [ -z "${NAME}" ]; then echo "no system at task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
'''
RUN = '''
srun singularity exec {nv} \\
    --overlay ${{OVERLAY}}:ro \\
    --bind ${{PROJECT_ROOT}}:${{PROJECT_ROOT}} \\
    --bind /scratch/mk8347/data:/scratch/mk8347/data \\
    --bind /scratch/mk8347/models:/scratch/mk8347/models \\
    --pwd {pwd} \\
    ${{IMAGE}} \\
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${{MXT_ROOT}}:${{PROJECT_ROOT}}/gfn-diffusion:${{PYLIBS}}:\\$PYTHONPATH
{body}"
RC=$?
echo "JOB ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} status=${{RC}} end=$(date -Is)"
exit ${{RC}}
'''
GPU = '#SBATCH --gres=gpu:1\n'


def w(name, s):
    open(os.path.join(HERE, name), 'w', newline='\n').write(s)


# ---------------------------------------------------------------- premerge
w('submit_premerge.sbatch', HEAD.format(
    time='16:00:00', gres='', mem='96G', cpus=8, signal='', job='po_premerge', doc=
    '''# pool_oct02 premerge (CPU): one task = one system of SYSTEMS.tsv (line 1 is the header). For a system whose premerge
# column is 1: a coordinator curate pass over the uploaded pool (<system>/pool: coord.yaml from this directory, shards
# uploaded from the dev box), then pool_anchors starts -> <system>/starts.pt (one row per basin within 10 kT, in the
# chart the trainer reads). A system with premerge 0 already has its starts.pt and exits at once.
# DO NOT EDIT --array BY HAND: make.py rewrites it.''') + SYSROW + '''
if [ "${PRE}" != "1" ]; then echo "${NAME}: starts.pt was built locally; nothing to do"; exit 0; fi
POOL=${DATA}/${NAME}/pool
if [ ! -d "${POOL}/shards" ]; then echo "FATAL: ${POOL}/shards missing (upload the pool)" >&2; exit 1; fi
cp "${ARMS}/${NAME}/pool_coord.yaml" "${POOL}/coord.yaml"
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} premerge ${NAME}: $(ls ${POOL}/shards | wc -l) shard directories start=$(date -Is)"
''' + RUN.format(nv='', pwd='${WORKDIR}', body='''        python -u -m mxtaltools.crystal_search.coordinator ${POOL} > ${POOL}/curate.log 2>&1 || { tail -30 ${POOL}/curate.log; exit 1; }
        tail -25 ${POOL}/curate.log
        CUDA_VISIBLE_DEVICES=-1 python -u -m data_processing.pool_anchors starts ${POOL} ${DATA}/${NAME}/starts.pt \\
            --mol ${MOL} --sg ${SG} --key ${KEY} --window 10
    '''))

# ---------------------------------------------------------------- export
w('submit_export.sbatch', HEAD.format(
    time='03:00:00', gres=GPU, mem='64G', cpus=8, signal='', job='po_export', doc=
    '''# pool_oct02 export: one task = one system of SYSTEMS.tsv: pool_anchors export of the POOLED registry (<system>/pool)
# -> the lowest state of every basin within the system window (kT above the lowest basin), in the chart the trainer
# reads, as <system>/anchors.pt (prior layout): the anchor rows of the prior and the starts of the flood. A system merged
# on the dev box has its anchors.pt uploaded; the task then only checks that the file is there. The GPU is used only by
# the re-scoring (UMA).''') + SYSROW + '''
OUT=${DATA}/${NAME}/anchors.pt
if [ "${PRE}" != "1" ]; then
    if [ -f "${OUT}" ]; then echo "${NAME}: ${OUT} was uploaded ($(stat -c %s ${OUT}) bytes)"; exit 0; fi
    echo "FATAL: ${OUT} missing (upload it from the dev box)" >&2; exit 1
fi
POOL=${DATA}/${NAME}/pool
if [ ! -f "${POOL}/registry.pt" ]; then echo "FATAL: ${POOL}/registry.pt missing (the premerge stage writes it)" >&2; exit 1; fi
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} export ${NAME} (sg ${SG}, ${KEY}, window ${WINDOW} kT) start=$(date -Is)"
''' + RUN.format(nv='--nv', pwd='${WORKDIR}', body='''        python -u -m data_processing.pool_anchors export ${POOL} ${OUT} \\
            --mol ${MOL} --sg ${SG} --key ${KEY} --tsf ${TSF} --window ${WINDOW} --mlip_path ${UMA}
    '''))

# ---------------------------------------------------------------- polish
w('submit_polish.sbatch', HEAD.format(
    time='08:00:00', gres=GPU, mem='32G', cpus=4, signal='#SBATCH --signal=USR1@900\n', job='po_polish', doc=
    '''# pool_oct02 polish: one task = one row of INDEX.tsv (line 1 is the header) = one block (mol_seed) of one system's
# starts.pt, relaxed again by run_search (init_sample_method data) into that system's polish campaign directory (shards,
# stream 'polish'). The block size is ceil(rows of starts.pt / tasks of the system), read from the file here.
# DO NOT EDIT --array BY HAND: make.py rewrites it. Resubmitting resumes every task.''') + '''
ROW=$((SLURM_ARRAY_TASK_ID + 2))
read -r NAME SUB NSUB <<< "$(awk -F'\\t' -v n=${ROW} 'NR==n {print $2, $3, $4}' ${ARMS}/INDEX.tsv)"
if [ -z "${NAME}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CAMP=${DATA}/${NAME}/polish
RUN=polish_${SUB}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
if [ ! -f "${DATA}/${NAME}/starts.pt" ]; then echo "FATAL: ${DATA}/${NAME}/starts.pt missing" >&2; exit 1; fi

mkdir -p ${CAMP}/runs
for f in coord.yaml polish.yaml; do  # frozen by the first task: a later pull cannot change a running polish
    if [ -f "${CAMP}/${f}" ]; then
        cmp -s "${ARMS}/${NAME}/${f}" "${CAMP}/${f}" || { echo "FATAL: ${CAMP}/${f} differs from ${ARMS}/${NAME}/${f}" >&2; exit 1; }
    else
        cp "${ARMS}/${NAME}/${f}" "${CAMP}/${f}.tmp.$$" && mv -n "${CAMP}/${f}.tmp.$$" "${CAMP}/${f}"; rm -f "${CAMP}/${f}.tmp.$$"
    fi
done

echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${NAME} block ${SUB} of ${NSUB}, code=${COMMIT} start=$(date -Is)"
''' + RUN.format(nv='--nv', pwd='${CAMP}/runs', body='''        python - <<'PY' || exit 1
import math, torch, yaml
cfg = yaml.safe_load(open('${CAMP}/polish.yaml'))
n = torch.load(cfg['dataset_path'], weights_only=False).num_graphs
cfg['num_samples'] = math.ceil(n / int('${NSUB}'))
cfg['run_name'] = '${RUN}'
cfg['mol_seed'] = int('${SUB}')
cfg['opt_seed'] = int(cfg['opt_seed']) + int('${SUB}') * 15_000_000
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open('${CAMP}/runs/${RUN}.yaml', 'w'))
print(n, 'starts, block of', cfg['num_samples'])
PY
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CAMP}/runs/${RUN}.yaml'''))

# ---------------------------------------------------------------- merge
w('submit_merge.sbatch', HEAD.format(
    time='04:00:00', gres=GPU, mem='64G', cpus=8, signal='', job='po_merge', doc=
    '''# pool_oct02 merge: one task = one system of SYSTEMS.tsv: a coordinator curate pass over the polish campaign (every
# polish shard, RDF leader clustering at the identity cut), then pool_anchors export -> the lowest state of every basin
# within 12 kT, in the chart the trainer reads, as <system>/anchors_polished.pt (prior layout; the seeds of the flood).
# The GPU is used only by the export's re-scoring (UMA). Safe to repeat: new shards are ingested, the export rewritten.''') + SYSROW + '''
CAMP=${DATA}/${NAME}/polish
if [ ! -d "${CAMP}/shards" ]; then echo "${NAME}: no polish shards yet; nothing to merge"; exit 0; fi
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} merge ${NAME} (sg ${SG}, ${KEY}): $(find ${CAMP}/shards -name '*.pt' | wc -l) shards start=$(date -Is)"
''' + RUN.format(nv='--nv', pwd='${WORKDIR}', body='''        python -u -m mxtaltools.crystal_search.coordinator ${CAMP} || exit 1
        python -u -m data_processing.pool_anchors export ${CAMP} ${DATA}/${NAME}/anchors_polished.pt \\
            --mol ${MOL} --sg ${SG} --key ${KEY} --tsf ${TSF} --window 12 --mlip_path ${UMA}
    '''))

# ---------------------------------------------------------------- flood
w('submit_flood.sbatch', HEAD.format(
    time='12:00:00', gres=GPU, mem='48G', cpus=8, signal='', job='po_flood', doc=
    '''# pool_oct02 flood: one task = one row of FLOOD.tsv (line 1 is the header) = one shard of one system's capped-MC flood
# (data_processing/capped_mc.py) from <system>/anchors.pt, with the row's arguments, into
# <system>/flood/shard_<k>. Resubmitting continues every shard from its state.pt (--resume).''') + '''
ROW=$((SLURM_ARRAY_TASK_ID + 2))
NAME=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $2}' ${ARMS}/FLOOD.tsv)
SHARD=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $3}' ${ARMS}/FLOOD.tsv)
NSH=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $4}' ${ARMS}/FLOOD.tsv)
KEY=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $5}' ${ARMS}/FLOOD.tsv)
CUT=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $6}' ${ARMS}/FLOOD.tsv)
ARGS=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $7}' ${ARMS}/FLOOD.tsv)
if [ -z "${NAME}" ]; then echo "no task at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
SEEDS=${DATA}/${NAME}/anchors.pt
OUT=${DATA}/${NAME}/flood/shard_${SHARD}
if [ ! -f "${SEEDS}" ]; then echo "FATAL: ${SEEDS} missing (the export stage writes it)" >&2; exit 1; fi
mkdir -p ${OUT}
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} flood ${NAME} shard ${SHARD}/${NSH} (${KEY}) start=$(date -Is)"
''' + RUN.format(nv='--nv', pwd='${WORKDIR}', body='''        export GPU_MEM_FRACTION=0.9
        python -u data_processing/capped_mc.py --prior ${SEEDS} --energy_function ${KEY} --mlip_path ${UMA} \\
            --out ${OUT} --n_shards ${NSH} --shard ${SHARD} --rdf_dcut ${CUT} ${ARGS} --resume
    '''))

# ---------------------------------------------------------------- assemble
w('submit_assemble.sbatch', HEAD.format(
    time='06:00:00', gres=GPU, mem='128G', cpus=8, signal='', job='po_assemble', doc=
    '''# pool_oct02 assemble: one task = one system of SYSTEMS.tsv: data_processing/pool_assemble.py on anchors.pt
# and every flood shard -> <system>/<system>_pooled_oct05_prior.pt and its .summary.json (anchors + flood states under
# the system's ceiling, thinned in the trainer latent at the system's radius, normaliser images; 'prior' and
# 'equalized_prior' hold the same rows; a sample of rows re-scored), then both files copied into ${PRIORS}. The radius
# and the ceiling are the SYSTEMS.tsv columns (make.py: RADIUS_MULT x KICK, CEILING); there is no row budget.''') + SYSROW + '''
if [ ! -d "${DATA}/${NAME}/flood" ]; then echo "${NAME}: no flood yet; nothing to assemble"; exit 0; fi
if [ -z "${RADIUS}" ] || [ -z "${CEILING}" ]; then echo "FATAL: no radius/ceiling for ${NAME} in SYSTEMS.tsv" >&2; exit 1; fi
OUT=${DATA}/${NAME}/${NAME}_pooled_oct05_prior.pt
echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} assemble ${NAME} radius ${RADIUS} ceiling ${CEILING} kT start=$(date -Is)"
''' + RUN.format(nv='--nv', pwd='${WORKDIR}', body='''        python -u -m data_processing.pool_assemble ${DATA}/${NAME}/anchors.pt ${DATA}/${NAME}/flood ${OUT} \\
            --sg ${SG} --key ${KEY} --mol ${MOL} --mlip_path ${UMA} --radius ${RADIUS} --ceiling_kT ${CEILING} || exit 1
        cp -v ${OUT} ${OUT}.summary.json ${PRIORS}/ || exit 1
        cat ${OUT}.summary.json
    '''))
print('wrote 6 sbatch files')
