"""mle_nig_sep17 -- phase 1 (bwd MLE) on the Niggli-reduced P-1 priors (`*_niggli_v2.pt`),
warm (weights-only from the mle09 best-MLE checkpoint) and fresh, per problem.

    python configs/mle_nig_sep17/make.py                # refuses on a dirty tree
    python configs/mle_nig_sep17/make.py --allow-dirty  # local validation only

Everything except the start is the mle_w3_sep16 arm shape (see that make.py): same rate,
burn-in, batch policy, eval cadence, archives, terminal MLE stage, identity alignment.

    mip   mipcas sg2 Z'=1  elj  (200k)
    mipu  mipcas sg2 Z'=1  uma  (f047, 200k)

THE PENALTY IS CODE. The priors hold Niggli cells, so the run must score triclinic cells
with tri_niggli_reduction_penalty. The sbatch sets MXT_NIGGLI_TRICLINIC=1 when the cluster's
MXtalTools reads that variable, leaves it unset when that checkout has made Niggli the only
rule, and asserts sym_utils.NIGGLI_TRICLINIC inside the container before training.

A fresh arm has no seed: checkpoint_name null, fresh weights at step 0.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('w3make', HERE.parent / 'mle_w3_sep16' / 'make.py')
w3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3)

TAG = 'mlenig'
BATTERY = 'mle_nig_sep17'
PRIOR_SUFFIX = '_niggli_v2.pt'

FAM = {
    'mip':  dict(seed='mle_sep09/mle09_mip_lr4p0.yaml',  src='mle09_mip_lr4p0',  prior_bytes=165_617_107),
    'mipu': dict(seed='mle_sep09/mle09_mipu_lr4p0.yaml', src='mle09_mipu_lr4p0', prior_bytes=232_846_061),
}
STARTS = ('fresh',)


def nig_path(path):
    assert path.startswith(w3.CLUSTER_DATA + '/') and path.endswith('.pt') and '_niggli' not in path, path
    return path[:-len('.pt')] + PRIOR_SUFFIX


def arm_name(fam, start):
    return f"{TAG}_{fam}_{start}_lr{w3.SCALE:g}"


def build_arm(base, fam, start):
    spec = FAM[fam]
    seed = w3.load(spec['seed'])
    name = arm_name(fam, start)
    # the W3 arm shape, pointed at this family's seed arm
    w3.FAM[fam] = spec
    cfg, dropped = w3.deltas(copy.deepcopy(base), fam, name)
    cfg['tag'] = TAG
    prior = nig_path(seed['prior_path'])
    cfg['prior_path'] = prior
    cfg['molecules_path'] = prior
    if start == 'fresh':
        cfg['checkpoint_name'] = None
        cfg['load_weights_only'] = False
        cfg['warm_start_ignore_problem_keys'] = None
    return name, cfg, dropped


def check(cfg, name, fam, start):
    spec = FAM[fam]
    seed = w3.load(spec['seed'])
    mine, theirs = w3.problem_def(cfg), w3.problem_def(seed)
    moved = sorted(k for k in set(mine) | set(theirs) if k != 'prior_path' and mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity moved from the mle09 seed on {moved}'
    assert cfg['prior_path'] == cfg['molecules_path'] == nig_path(seed['prior_path']), name
    assert cfg['space_groups'] == [2], name
    assert cfg['continue_from_checkpoint'] == w3.CONT_PLACEHOLDER, name
    if start == 'warm':
        assert cfg['checkpoint_name'] == w3.CK_PLACEHOLDER and cfg['load_weights_only'] is True, name
        assert cfg['warm_start_ignore_problem_keys'] == ['prior_path'], name
    else:
        assert cfg['checkpoint_name'] is None and cfg['warm_start_ignore_problem_keys'] is None, name
    st = cfg['protocols'][w3.PROTOCOL]['stages']
    assert len(st) == 1 and 'exit' not in st[0] and st[0]['train_mode'] == 'bwd', name
    assert cfg['integrator']['T'] == w3.SHIP_T == cfg['eval_T'], name
    assert cfg['lr_control']['fixed_scale'] == w3.SCALE, name
    w3._scan_local_paths(cfg, name)


SBATCH = r"""#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-__LAST__
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=__TAG__
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/__BATTERY__/joblogs/%x_%A_%a.out

# __BATTERY__: phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.
# Arm = row of INDEX.tsv (line 1 is the header). DO NOT EDIT --array BY HAND: make.py rewrites it.
# Resubmit this same file to continue an arm past the wall.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/__BATTERY__
LOGS=${ARMS}/joblogs
CKPTS=__CKPTS__
DATA=__DATA__
mkdir -p ${LOGS}

ROW=$((SLURM_ARRAY_TASK_ID + 2))
ARM=$(awk -F'\t' -v n=${ROW} 'NR==n {print $1}' ${ARMS}/INDEX.tsv)
SRC=$(awk -F'\t' -v n=${ROW} 'NR==n {print $4}' ${ARMS}/INDEX.tsv)
PRIOR=$(awk -F'\t' -v n=${ROW} 'NR==n {print $5}' ${ARMS}/INDEX.tsv)
PRIOR_BYTES=$(awk -F'\t' -v n=${ROW} 'NR==n {print $6}' ${ARMS}/INDEX.tsv)
if [ -z "${ARM}" ]; then echo "no arm at row ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; fi
CONFIG=${ARMS}/${ARM}.yaml
if [ ! -f "${CONFIG}" ]; then echo "missing config ${CONFIG}" >&2; exit 1; fi

J=${LOGS}/${ARM}_${SLURM_JOB_ID}
RESOLVED=${J}.yaml

if [ -f ${CKPTS}/${ARM}.dead ]; then
    echo "arm ${ARM} aborted UNRECOVERABLE on an earlier leg -- skipping"
    exit 0
fi

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

HAVE=$(stat -c %s ${DATA}/${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES} -- upload unfinished or a different file" >&2
    exit 1
fi

# RESUME OR START. checkpoint_name always wins in train.py, so a requeue blanks it and auto-resumes
# its own _running.pt. A first launch seeds weights-only from the mle09 _best.pt (warm) or from
# nothing (fresh, SRC "-").
OWN=$(ls -t ${CKPTS}/*${ARM}_*_running.pt 2>/dev/null | head -1)
if [ -n "${OWN}" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  RESUME: $(basename ${OWN})"
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|null|" -e "s|CONTINUE_PLACEHOLDER|true|" ${CONFIG} > ${RESOLVED}
elif [ "${SRC}" = "-" ]; then
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  FRESH"
    sed -e "s|CONTINUE_PLACEHOLDER|false|" ${CONFIG} > ${RESOLVED}
else
    N=$(ls ${CKPTS}/*${SRC}_*_best.pt 2>/dev/null | wc -l)
    if [ "${N}" -ne 1 ]; then
        echo "FATAL: ${N} matches for *${SRC}_*_best.pt in ${CKPTS} (need exactly 1)" >&2
        ls ${CKPTS}/*${SRC}_*_best.pt >&2; exit 1
    fi
    CK=$(ls ${CKPTS}/*${SRC}_*_best.pt)
    echo "array ${SLURM_ARRAY_TASK_ID} -> arm ${ARM}  SEED: $(basename ${CK}) (mtime $(stat -c %y ${CK}))"
    sed -e "s|WARM_CHECKPOINT_PLACEHOLDER|$(basename ${CK})|" -e "s|CONTINUE_PLACEHOLDER|false|" ${CONFIG} > ${RESOLVED}
fi
if grep -q 'WARM_CHECKPOINT_PLACEHOLDER\|CONTINUE_PLACEHOLDER' ${RESOLVED}; then
    echo "FATAL: placeholder left in ${RESOLVED}" >&2; exit 1
fi

{ nvidia-smi -L
  scontrol show job ${SLURM_JOB_ID}
  echo "nodelist: ${SLURM_NODELIST}  host: $(hostname)"
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
        python -u train.py --config ${RESOLVED}
    " 2>&1 | tee ${J}.trainlog

if grep -q 'UNRECOVERABLE' ${J}.trainlog 2>/dev/null; then
    echo "arm ${ARM} exhausted its rewind budget -- sentinel set, resubmissions will skip"
    touch ${CKPTS}/${ARM}.dead
fi
"""


def main(argv):
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    base = yaml.safe_load(w3.MK_DEV.read_text(encoding='utf-8'))
    arms = {}
    for fam in FAM:
        for start in STARTS:
            name, cfg, dropped = build_arm(base, fam, start)
            check(cfg, name, fam, start)
            w3.load_check(cfg, name)
            arms[name] = (cfg, fam, start, dropped)

    for fam, spec in FAM.items():
        local = w3.LOCAL_DATA / nig_path(w3.load(spec['seed'])['prior_path']).rsplit('/', 1)[1]
        if local.exists():
            assert local.stat().st_size == spec['prior_bytes'], f'{fam}: {local} size != PRIOR_BYTES'

    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    for name, (cfg, *_rest) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        for name, (cfg, fam, start, _d) in arms.items():
            src = FAM[fam]['src'] if start == 'warm' else '-'
            f.write(f"{name}\t{fam}\t{start}\t{src}\t{cfg['prior_path'].rsplit('/', 1)[1]}\t{FAM[fam]['prior_bytes']}\n")
    sb = (SBATCH.replace('__LAST__', str(len(arms) - 1)).replace('__TAG__', TAG).replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    for i, (name, (cfg, fam, start, dropped)) in enumerate(arms.items()):
        print(f"[{i}] {name:<24} {start:<5} prior={cfg['prior_path'].rsplit('/', 1)[1]} dropped={dropped or '-'}")


if __name__ == '__main__':
    main(sys.argv[1:])
