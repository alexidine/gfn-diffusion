r"""conformer_mle_sep30 -- converge the conformer MLE warm-up (stage `train_prior` of protocol
`conformer_conditional_tb`) on rungs cut from the QM9 conformer database ON THE CLUSTER.

    python configs/conformer_mle_sep30/make.py

ONE JOB PER RUNG (run_<rung>.sbatch), or TWO chained by submit.sh:
  * build_<rung>.sbatch (CPU): build_conformer_set.py --database DB --rows-per-condition-cap 16
    --workers <the job's CPUs> cuts the rung (train + held-out conditions, the database prior)
    into RUNGS_ROOT/<rung>, then build_conformer_references.py --database DB writes both
    reference tables (the per-condition floors the offline eval reads). Every member is built
    from the database's stored reference conformer (no embedding, the stereo lock's thermal
    check skipped on the database's record of its pass), and the run builds each member from the
    reference the conditions file stores, so the rung no longer depends on which RDKit embeds
    it; it still runs in the training container, whose RDKit types the MMFF94 terms and whose
    version the training job asserts against the rung's manifest. Rerunning it skips what is
    already on disk (a set with a manifest, a written table).
  * train_<rung>.sbatch (GPU): configs/final_sep19/make.py::SBATCH through
    configs/conformer_cond/make.py::conformer_template (the RESUME / FRESH / CK_STEP block, the
    dead-arm sentinel, the RDKit / MMFF import guard, the nvidia-smi and sacct sidecars), pinned
    to its row of INDEX_a.tsv, with the data guard replaced by RUNG_GUARD (the rung's files
    present, the RDKit version read from its manifest, the committed prior's sha256). A first
    launch is FRESH; RESUBMITTING THE SAME FILE continues the arm from its own _running.pt.
  * run_<rung>.sbatch (GPU, ONE JOB, no dependency): train_<rung>.sbatch with ONE_JOB_GUARD in
    place of RUNG_GUARD -- when a file of the rung is missing it runs build_<rung>.sbatch as a
    script inside the job (under the lock directory RUNGS_ROOT/<rung>.building, which a second
    job refuses rather than waits on), then trains. It asks for the build's CPUs and the larger
    of the two memory requests, and the build's time comes out of the training wall.
    NOT YET RUN ON THE CLUSTER.

    bash configs/conformer_mle_sep30/submit.sh pilot       # build -> train, afterok
    bash configs/conformer_mle_sep30/submit.sh r20k
    bash configs/conformer_mle_sep30/submit.sh r20k one    # ONE job: build if missing, then train
    sbatch configs/conformer_mle_sep30/train_r20k.sbatch   # every later leg (or run_r20k.sbatch)

EACH ARM is the committed configs/conformer_mk.yaml (git HEAD, never the working tree) with only
  run_name mle30_<rung>; molecules_path / test_molecules_path / prior_path = the rung's
  conditions_train.pt / conditions_heldout.pt / prior_train.pt; prior_dataset_noise 'thermal';
  train_prior max_steps TRAIN_PRIOR_MAX_STEPS (its level gate cannot fire on a large rung --
  a perfect sampler reads w1r/worst ~13 against the bar 10 under the eval's uniform condition
  draw -- so MLE runs until the owner stops it); epochs EPOCHS; checkpoints_dir CLUSTER_CKPTS;
  checkpoint_name WARM_CHECKPOINT_PLACEHOLDER, which the sbatch resolves (null FRESH, the arm's
  own _running.pt on RESUME).
The pilot arm is the same on the PILOT rung, with epochs PILOT_EPOCHS.

REFUSES: an override key the committed config does not carry; a drive-letter or backslash path in
anything it writes (a local path passes every local check and kills every arm on-cluster); a
template span it substitutes that is not there exactly once; an arm the loader, the invariants or
the stage protocol reject (configs/final_sep19/make.py::load_check).

PROJECTIONS (printed and written to INDEX_a.tsv; WORKING ASSUMPTIONS from the local 2000/200
database rung of 2026-09-30 and the local 450/50 builds of 2026-10-01, not cluster
measurements): conditions and prior rows per molecule, the set builder's and reference builder's
CPU seconds, the builder's peak host memory, and the training footprint model (train_footprint):
compact rows at configs/conformer_cond/make.py::row_bytes, the conditions table at its
TABLE_BYTES_PER_CONDITION_MAX per train condition, the prior dataset held twice (as built and
thermal-noised), the anchor buffer at its resolved capacity (max of its max_size and the prior
rows it is seeded with), the prior buffer at its max_size.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import pathlib
import re
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                      # energy_sampling/configs
ES = ROOT.parent                        # energy_sampling/
REPO = ES.parent                        # gfn_diffusion/
BATTERY = HERE.name
TAG = 'mle30'
LEG = 'a'

#: THE RUNGS: (name, train molecules, held-out molecules). Owner decision 2026-09-30: one
#: rung of 20,000 / 2,000. Any mix is one edit here.
RUNGS = [('r20k', 20_000, 2_000)]
#: the pilot: a tiny build and PILOT_EPOCHS training steps, through the same two jobs
PILOT = ('pilot', 5, 2)
PILOT_EPOCHS = 1000                     # two evals (eval_period 500), one figure pass (figs_period 1000)

ROWS_PER_CONDITION_CAP = 16
TRAIN_PRIOR_MAX_STEPS = 200_000
EPOCHS = 1_000_000
CANONICAL = 'energy_sampling/configs/conformer_mk.yaml'
PROTOCOL = 'conformer_conditional_tb'
STAGES = ['train_prior', 'tb_conditioning']

PROJECT_ROOT = '/scratch/mk8347/projects/gfn_cond'
WORKDIR = f'{PROJECT_ROOT}/gfn-diffusion/energy_sampling'
DATABASE = '/scratch/mk8347/data/conformer_datasets/qm9_db_sep29'
DATABASE_SHARDS = 400
RUNGS_ROOT = '/scratch/mk8347/data/conformer_datasets/rungs_sep30'
SOURCE_REL = 'configs/conformer_db_sep29/qm9_index.tsv.gz'     # the QM9 (dataset_index, smiles) table
ENCODER_REL = 'models/results/encoder_ckpt/mp+attn+spd_n20000_s0.pt'   # models/encoder_cache.DEFAULT_CKPT
PRIOR_REL = 'conformer_prior_v2.pt'     # energy_config.internal_prior_path

# --- the build job (CPU)
BUILD_CPUS = 16                         # both builders run one worker process per CPU of the job
SET_THREADS = 1                         # torch threads per set-builder worker
#: THE SET BUILDER, members from the database's stored references and the thermal check skipped
#: on its recorded pass (local 450/50 build, 2026-10-01, 24-thread desktop; the same build embedding
#: every member and running every thermal check walked at 0.55 s per molecule):
#:   SET_FIXED_S         what does not scale with the rung -- the split plan over the whole source
#:                       table and two reads of the 400-shard database
#:   SET_WALK_S          CPU seconds per kept molecule of the walk, which --workers divides
#:                       (35 s serial against about 4 s on 8 workers for 450 molecules)
#:   SET_SERIAL_S        seconds per kept molecule after the walk, in the parent process alone
#:                       (layout, files, the database prior and their verification)
#: CLUSTER_SLOWDOWN scales all three: the r20k build of 2026-09-30 ran at 1.04 s per molecule on
#: the cluster against 0.68 s locally for the same code.
SET_FIXED_S = 90.0
SET_WALK_S = 0.08
SET_SERIAL_S = 0.06
CLUSTER_SLOWDOWN = 1.5
#: CPU seconds per condition of the database-floor reference build: locally 803 s over 3,353
REFS_S_PER_CONDITION = 0.24
#: conditions per molecule (stereoisomers): locally 3,486 / 2,000 train and 355 / 200 held-out
CONDITIONS_PER_MOLECULE = 1.75
#: prior rows per train molecule at cap 16: locally 22,760 / 2,000
ROWS_PER_MOLECULE = 11.4
#: the set builder's peak host memory, GB = BUILD_MEM_BASE_GB + BUILD_MEM_PER_1K_MOL_GB x
#: molecules / 1000 (train + held-out) + BUILD_MEM_PER_WORKER_GB x workers. Base: the 3.0 GB peak
#: of the local 450/50 build, most of it the whole database's reference table (218,473 references
#: and 204,253 recorded refusals, read whatever the rung's size). Per molecule: 1.75 conditions at
#: about 0.3 MB (a member's 0.22 MB of resident memory as the trainer holds it, and its 0.07 MB
#: condition graph) -- an ESTIMATE, not measured at rung size; the 3.15 GB per 1,000 molecules of
#: 2026-09-30 was each member's cached batched force field, which build_member now releases.
#: Per worker: (8.9 - 3.0) / 8 GB of the 8-worker build's process tree.
BUILD_MEM_BASE_GB = 3.0
BUILD_MEM_PER_1K_MOL_GB = 0.6
BUILD_MEM_PER_WORKER_GB = 0.75
BUILD_WALL_MARGIN = 4.0
BUILD_MEM_MARGIN = 2.0

# --- the training job (GPU)
TRAIN_WALL = '1-00:00:00'
TRAIN_GRES = 'gpu:a100:1'               # the pattern of every recent GPU battery
TRAIN_MEM = {'pilot': '48G'}            # host memory; the rest from host_gb below
TRAIN_MEM_DEFAULT_GB = 48
#: host --mem = TRAIN_MEM_MARGIN x (projected host GB + HOST_BASE_GB for the interpreter, torch and
#: the CUDA context), rounded up to 16 GB, at least TRAIN_MEM_DEFAULT_GB
TRAIN_MEM_MARGIN = 2.5
HOST_BASE_GB = 6.0
PILOT_WALL = '01:00:00'
#: Host memory of the training process per train condition, KB: its member object (253 KB,
#: the 247 KiB per member the 2026-09-30 compaction brief records; not re-measured since
#: stored-reference members) and the float64 conditions file while it is read (73 KB,
#: 72,626 B per condition on the 3,486-condition rung). WORKING ASSUMPTIONS.
MEMBER_HOST_KB = 253.0
COND_FILE_KB = 73.0
CARDS_GB = {'a100-40': 40, 'l40s-48': 48, 'a100-80': 80}

_spec = importlib.util.spec_from_file_location('cc_make', ROOT / 'conformer_cond' / 'make.py')
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)
fin = cc.fin
PLACEHOLDER = cc.PLACEHOLDER
CLUSTER_CKPTS = cc.CLUSTER_CKPTS
#: a drive letter not preceded by a letter or digit (so a URL scheme does not match)
DRIVE = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]')


# ----------------------------------------------------------------------------- git

def git_blob(rel: str) -> bytes:
    """A committed file's BYTES at HEAD (binary-safe). Raises -- never the working tree."""
    out = subprocess.run(['git', 'show', f'HEAD:{rel}'], capture_output=True, cwd=str(REPO))
    if out.returncode != 0:
        raise SystemExit(f'REFUSING: git cannot produce HEAD:{rel} ({out.stderr.decode().strip()})')
    return out.stdout


def committed_sha256(rel_to_es: str):
    """(bytes, sha256) of a file committed at HEAD, relative to energy_sampling/."""
    blob = git_blob('energy_sampling/' + rel_to_es)
    return len(blob), hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------- arms

def _node(cfg, dotted):
    node = cfg
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise SystemExit(f'REFUSING: the override {dotted!r} names no key of the committed '
                             f'configs/conformer_mk.yaml (missing at {part!r})')
        node = node[part]
    return node


def set_key(cfg, dotted, value):
    """Set an EXISTING dotted key; refuses a key the committed config does not carry."""
    _node(cfg, dotted)
    parent, _, leaf = dotted.rpartition('.')
    (_node(cfg, parent) if parent else cfg)[leaf] = value


def rung_dir(rung):
    return f'{RUNGS_ROOT}/{rung}'


def build_arm(base, rung, pilot=False):
    """The committed config with this battery's overrides, each on a key that exists."""
    cfg = copy.deepcopy(base)
    d = rung_dir(rung)
    overrides = {
        'run_name': f'{TAG}_{rung}',
        'molecules_path': f'{d}/conditions_train.pt',
        'test_molecules_path': f'{d}/conditions_heldout.pt',
        'prior_path': f'{d}/prior_train.pt',
        'prior_dataset_noise': 'thermal',
        'epochs': PILOT_EPOCHS if pilot else EPOCHS,
        'checkpoints_dir': CLUSTER_CKPTS,
        'checkpoint_name': PLACEHOLDER,
    }
    for k, v in overrides.items():
        set_key(cfg, k, v)
    tp = cc.stage(cfg, 'train_prior')
    if 'max_steps' not in tp:
        raise SystemExit('REFUSING: the committed train_prior stage carries no max_steps')
    tp['max_steps'] = TRAIN_PRIOR_MAX_STEPS
    return cfg


def check_arm(cfg, base, name, pilot):
    """What the arm must be, re-asserted on the finished dict."""
    assert cfg['protocol'] == PROTOCOL, (name, cfg['protocol'])
    assert [s['name'] for s in cfg['protocols'][PROTOCOL]['stages']] == STAGES, name
    assert cfg['continue_from_checkpoint'] is False and cfg['load_weights_only'] is False, name
    assert cfg['prior_model_name'] is None, name
    assert cfg['archive_buffers'] is True, name          # CK_STEP needs the frozen sidecar
    assert cfg['buffers']['anchor_buffer']['tile'] == 'thermal', name
    assert int(cc.stage(cfg, 'train_prior')['max_steps']) < int(cfg['epochs']) or pilot, name
    # nothing else moved: every flattened leaf equals the base's but the overrides
    want = {'run_name', 'molecules_path', 'test_molecules_path', 'prior_path',
            'prior_dataset_noise', 'epochs', 'checkpoints_dir', 'checkpoint_name'}
    fa, fb = cc.flat({k: v for k, v in cfg.items() if k != 'protocols'}), \
        cc.flat({k: v for k, v in base.items() if k != 'protocols'})
    moved = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    assert set(moved) <= want, f'{name}: moved beyond the overrides: {sorted(set(moved) - want)}'
    pa, pb = copy.deepcopy(cfg['protocols']), copy.deepcopy(base['protocols'])
    for p in (pa, pb):
        [s for s in p[PROTOCOL]['stages'] if s['name'] == 'train_prior'][0].pop('max_steps')
    assert pa == pb, f'{name}: the protocols block moved beyond train_prior max_steps'
    try:
        fin.w3._scan_local_paths(cfg, name)
    except AssertionError as e:
        raise SystemExit(f'REFUSING {e}') from None
    fin.load_check(cfg, name, STAGES)


# ----------------------------------------------------------------------------- projections

def build_seconds(n_mol, workers=BUILD_CPUS):
    """(projected wall seconds, CPU seconds) of a build of n_mol molecules on ``workers`` CPUs,
    before any margin: the set builder's fixed part, its walk over the workers, its serial tail,
    and the reference builder over the workers."""
    n_cond = CONDITIONS_PER_MOLECULE * n_mol
    walk_s, serial_s = SET_WALK_S * n_mol, SET_FIXED_S + SET_SERIAL_S * n_mol
    refs_s = REFS_S_PER_CONDITION * n_cond
    return (CLUSTER_SLOWDOWN * (serial_s + (walk_s + refs_s) / workers),
            CLUSTER_SLOWDOWN * (serial_s + walk_s + refs_s))


def build_cost(n_mol):
    """(wall 'HH:MM:SS', mem 'NG', CPU-hours) for a build of n_mol molecules (train + held-out)."""
    wall, cpu = build_seconds(n_mol)
    wall_s = BUILD_WALL_MARGIN * wall + 1800
    hours = min(47, max(1, math.ceil(wall_s / 3600)))
    mem = BUILD_MEM_MARGIN * (BUILD_MEM_BASE_GB + BUILD_MEM_PER_1K_MOL_GB * n_mol / 1000
                              + BUILD_MEM_PER_WORKER_GB * BUILD_CPUS)
    return f'{hours:02d}:00:00', f'{max(8, math.ceil(mem))}G', cpu / 3600


def train_footprint(n_train, cfg):
    """(prior rows, train conditions, GB: GPU resident, GPU warm-up peak, host, one buffer
    sidecar) for an arm `cfg` on n_train molecules, from the compact-store model of
    configs/conformer_cond/make.py (row_bytes at K_MAX, TABLE_BYTES_PER_CONDITION_MAX,
    COMPACT_HOST_BYTES_PER_ROW).

    GPU resident: the conditions table, the prior dataset (the file's rows; twice under
    prior_dataset_noise 'thermal'), the anchor buffer at its resolved capacity (max of its
    max_size and the prior rows, which seed it whole) and the prior buffer at min(rows,
    max_size). The peak adds one chunk of COMPACT_CHUNK_ROWS materialised graphs. Host: the
    members and the conditions file being read (MEMBER_HOST_KB, COND_FILE_KB per condition)
    and the rows' host columns. The sidecar holds the prior and anchor buffers (replay is
    empty under train_prior)."""
    rows = ROWS_PER_MOLECULE * n_train
    conds = CONDITIONS_PER_MOLECULE * n_train
    per = cc.row_bytes(cfg)
    b = cfg['buffers']
    anchor_rows = cc.anchor_capacity(cfg, seed_rows=rows)
    prior_rows = min(rows, int(b['prior_buffer']['max_size']))
    table = conds * cc.TABLE_BYTES_PER_CONDITION_MAX
    buffers = anchor_rows * per['anchor'] + prior_rows * per['prior']
    resident = (table + rows * per['prior_sample'] + buffers) / 1e9
    peak = resident + COMPACT_CHUNK_ROWS * cc.TABLE_BYTES_PER_CONDITION_MAX / 1e9
    host = (conds * (MEMBER_HOST_KB + COND_FILE_KB) * 1e3
            + (2 * rows + anchor_rows + prior_rows) * cc.COMPACT_HOST_BYTES_PER_ROW) / 1e9
    sidecar = (buffers + (anchor_rows + prior_rows) * cc.COMPACT_HOST_BYTES_PER_ROW) / 1e9
    return rows, conds, resident, peak, host, sidecar


#: rows of graphs a compact store materialises at a time (buffer.py::COMPACT_CHUNK_ROWS)
COMPACT_CHUNK_ROWS = 4096


# ----------------------------------------------------------------------------- sbatch

BUILD_TEMPLATE = r'''#!/bin/bash
#SBATCH --time=@@WALL@@
#SBATCH --mem=@@MEM@@
#SBATCH --cpus-per-task=@@CPUS@@
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=@@TAG@@b_@@RUNG@@
#SBATCH --output=@@WORKDIR@@/configs/@@BATTERY@@/joblogs/%x_%j.out

# @@BATTERY@@ BUILD, rung @@RUNG@@: @@N_TRAIN@@ train / @@N_HELDOUT@@ held-out molecules cut from the
# QM9 conformer database at @@CAP@@ rows per condition, with both reference tables, into
# @@OUT@@. CPU only, IN THE TRAINING CONTAINER. Members are built from the database's stored
# reference conformers, and the training run builds each member from the conditions file's.
# WRITTEN BY make.py: do not edit by hand. Rerunning skips a set that has its manifest and a
# table already written (the reference builder resumes from its .parts directory).
# Projected: @@CPU_H@@ CPU-hours; both builders run one worker per CPU of the job
# (SLURM_CPUS_PER_TASK, @@CPUS@@ as submitted here). run_@@RUNG@@.sbatch runs this file as a script.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=@@PROJECT_ROOT@@
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
LOGS=${WORKDIR}/configs/@@BATTERY@@/joblogs
DB=@@DATABASE@@
OUT=@@OUT@@
NCPU=${SLURM_CPUS_PER_TASK:-@@CPUS@@}
mkdir -p ${LOGS}
J=${LOGS}/build_@@RUNG@@_${SLURM_JOB_ID}

# INPUT GUARD: the database, and the committed files the builders read, as make.py saw them at HEAD
NSH=$(ls ${DB}/shard_*_of_@@SHARDS_PAD@@.pt 2>/dev/null | wc -l)
if [ "${NSH}" != "@@SHARDS@@" ]; then echo "FATAL: ${DB} holds ${NSH} shard files, expected @@SHARDS@@" >&2; exit 1; fi
check() {  # path bytes sha256
    local HAVE
    HAVE=$(stat -c %s ${WORKDIR}/$1 2>/dev/null || echo 0)
    if [ "${HAVE}" != "$2" ]; then echo "FATAL: ${WORKDIR}/$1 is ${HAVE} bytes, expected $2 (git pull)" >&2; exit 1; fi
    if [ "$(sha256sum ${WORKDIR}/$1 | cut -d' ' -f1)" != "$3" ]; then echo "FATAL: ${WORKDIR}/$1 is not sha256 $3 (git pull)" >&2; exit 1; fi
}
check @@SOURCE_REL@@ @@SOURCE_BYTES@@ @@SOURCE_SHA@@
check '@@ENCODER_REL@@' @@ENCODER_BYTES@@ @@ENCODER_SHA@@
check @@PRIOR_REL@@ @@PRIOR_BYTES@@ @@PRIOR_SHA@@
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
            python -u build_conformer_set.py --out-dir ${OUT} --source @@SOURCE_REL@@ \
                --config configs/conformer_mk.yaml --n-train @@N_TRAIN@@ --n-heldout @@N_HELDOUT@@ \
                --database ${DB} --rows-per-condition-cap @@CAP@@ --threads @@SET_THREADS@@ \
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
'''

#: the train template's index read of its rdkit column, replaced by the read of the rung
_SET_COLUMN = r"""SET=$(awk -F'\t' -v n=${{ROW}} 'NR==n {{print $5}}' ${{INDEX}})
"""
_RUNG_FILES = ('manifest.json conditions_train.pt conditions_heldout.pt prior_train.pt '
               'conditions_train.references.pt conditions_heldout.references.pt')
_GUARD_HEAD = r"""# RUNG GUARD. The rung is built ON THE CLUSTER by build_${{SET}}.sbatch. Every file the run and
# the offline eval read must be there; the RDKit version the conformer guard below asserts is the
# one the rung's manifest records (the run builds each member from the reference conformer the
# conditions file stores, and types its MMFF94 terms with this RDKit). The fitted prior is
# committed: its sha256 is HEAD's. NIG_EXPORT is the crystal template's MXtalTools export, empty
# on this route.
NIG_EXPORT=""
RUNG=${{DATA}}/${{SET}}
"""
_GUARD_REFUSE = r"""for F in @@RUNG_FILES@@; do
    if [ ! -s ${{RUNG}}/${{F}} ]; then echo "FATAL: ${{RUNG}}/${{F}} is missing; run build_${{SET}}.sbatch first" >&2; exit 1; fi
done
"""
#: ONE JOB: build the rung here when a file of it is missing. NO WAIT: a second job that finds
#: the lock directory exits at once (the builder's own job is running, or one died building --
#: the message says how to clear it), so nothing can pend on a build that never finishes.
_GUARD_BUILD = r"""# ONE JOB: a rung with a file missing is built here, by build_${{SET}}.sbatch run as a script
# (its #SBATCH lines are comments; it uses this job's CPUs), before training.
MISSING=""
for F in @@RUNG_FILES@@; do
    [ -s ${{RUNG}}/${{F}} ] || MISSING="${{MISSING}} ${{F}}"
done
if [ -n "${{MISSING}}" ]; then
    echo "rung ${{SET}}: missing${{MISSING}} -- building it in this job"
    mkdir -p ${{DATA}}
    if ! mkdir ${{RUNG}}.building 2>/dev/null; then
        echo "FATAL: ${{RUNG}}.building exists: another job is building this rung, or one died building it (then: rmdir ${{RUNG}}.building)" >&2; exit 1
    fi
    bash ${{ARMS}}/build_${{SET}}.sbatch; BUILD_RC=$?
    rmdir ${{RUNG}}.building
    if [ ${{BUILD_RC}} -ne 0 ]; then echo "FATAL: the build of rung ${{SET}} exited ${{BUILD_RC}}" >&2; exit 1; fi
    for F in @@RUNG_FILES@@; do
        if [ ! -s ${{RUNG}}/${{F}} ]; then echo "FATAL: the build left ${{RUNG}}/${{F}} missing" >&2; exit 1; fi
    done
fi
"""
_GUARD_TAIL = r"""RDKIT=$(grep -o '"rdkit": *"[^"]*"' ${{RUNG}}/manifest.json | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
if [ -z "${{RDKIT}}" ]; then echo "FATAL: ${{RUNG}}/manifest.json names no RDKit version" >&2; exit 1; fi
if [ "$(sha256sum ${{WORKDIR}}/@@PRIOR_REL@@ | cut -d' ' -f1)" != "@@PRIOR_SHA@@" ]; then
    echo "FATAL: ${{WORKDIR}}/@@PRIOR_REL@@ is not HEAD's (sha256 @@PRIOR_SHA@@); git pull" >&2; exit 1
fi
"""
RUNG_GUARD = _GUARD_HEAD + _GUARD_REFUSE + _GUARD_TAIL
ONE_JOB_GUARD = _GUARD_HEAD + _GUARD_BUILD + _GUARD_TAIL


def fill(text, values, where):
    for k, v in values.items():
        text = text.replace(f'@@{k}@@', str(v))
    left = sorted(set(re.findall(r'@@([A-Z_0-9]+)@@', text)))
    if left:
        raise SystemExit(f'{where}: unfilled placeholder(s) {left}')
    return text


def refuse_local(text, where):
    """A drive-letter path passes every local check (the file exists on the dev box) and kills
    the job on-cluster. Backslashes are shell syntax in the sbatch files; in an arm's config
    they are refused by w3._scan_local_paths (check_arm)."""
    bad = [ln for ln in text.splitlines() if DRIVE.search(ln)]
    if bad:
        raise SystemExit(f'{where}: local path(s) would ship to the cluster:\n  '
                         + '\n  '.join(bad))


def train_sbatch(row, rung, wall, mem, prior_sha, pilot=False, one_job=False):
    """The training job; ``one_job`` builds the rung first when it is missing (ONE_JOB_GUARD)
    and asks for the build's CPUs."""
    text = cc.conformer_template(fin.SBATCH)
    guard = fill(ONE_JOB_GUARD if one_job else RUNG_GUARD,
                 {'PRIOR_REL': PRIOR_REL, 'PRIOR_SHA': prior_sha, 'RUNG_FILES': _RUNG_FILES}, 'guard')
    text = cc._replace_once(text, cc.DATA_GUARD, guard, 'conformer data guard')
    if one_job:
        text = cc._replace_once(text, '#SBATCH --cpus-per-task=8',
                                f'#SBATCH --cpus-per-task={max(8, BUILD_CPUS)}', 'cpus line')
    text = cc._replace_once(text, cc._RDKIT_COLUMN, _SET_COLUMN, 'rdkit index column')
    text = cc._replace_once(text, '#SBATCH --array=0-{last}', f'#SBATCH --array={row}', 'array line')
    text = cc._replace_once(text, '#SBATCH --gres=gpu:a100:1', f'#SBATCH --gres={TRAIN_GRES}', 'gres line')
    text = cc._replace_once(text, '#SBATCH --mem=48G', f'#SBATCH --mem={mem}', 'mem line')
    return text.format(
        wall=wall, tag=TAG, battery=BATTERY, ckpts=CLUSTER_CKPTS, data=RUNGS_ROOT, leg=LEG,
        seed_block=cc.SEED_FRESH, last=row,
        what=(('ONE JOB (builds the rung first when it is missing). ' if one_job else '')
              + (f'PILOT: {PILOT_EPOCHS} training steps (epochs) on the pilot rung.' if pilot else
                 f'MLE warm-up (train_prior, max_steps {TRAIN_PRIOR_MAX_STEPS:,}) on rung {rung}, '
                 f'FRESH on the first launch; resubmit this file to continue.')))


SUBMIT = r'''#!/bin/bash
# @@BATTERY@@: submit one rung's build (CPU) and its training (GPU, afterok on the build), or with
# `one` ONE job that builds the rung when it is missing and then trains (no dependency).
#     bash configs/@@BATTERY@@/submit.sh <rung> [one]      rungs: @@RUNGS@@
# Later legs: sbatch configs/@@BATTERY@@/train_<rung>.sbatch (it resumes the arm's own _running.pt).
# WRITTEN BY make.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${1:-}
MODE=${2:-two}
case " @@RUNGS@@ " in *" ${R} "*) ;; *) echo "usage: bash $0 <rung> [one], rung one of: @@RUNGS@@" >&2; exit 2;; esac
case "${MODE}" in one|two) ;; *) echo "usage: bash $0 <rung> [one]" >&2; exit 2;; esac
if [ "${R}" = "pilot" ] && ls @@CKPTS@@/*@@TAG@@_pilot_*_running.pt >/dev/null 2>&1; then
    echo "the pilot arm already has checkpoints, and a new pilot would RESUME them; to re-pilot:" >&2
    echo "    rm @@CKPTS@@/*@@TAG@@_pilot_* ; rm -r @@RUNGS_ROOT@@/pilot" >&2
    exit 1
fi
if [ "${MODE}" = "one" ]; then
    T=$(sbatch --parsable configs/@@BATTERY@@/run_${R}.sbatch); T=${T%%;*}
    echo "rung ${R}: ONE job ${T} (builds the rung if it is missing, then trains)"
    exit 0
fi
B=$(sbatch --parsable configs/@@BATTERY@@/build_${R}.sbatch); B=${B%%;*}
# --kill-on-invalid-dep: a failed build cancels the training job instead of leaving it pending forever
T=$(sbatch --parsable --dependency=afterok:${B} --kill-on-invalid-dep=yes configs/@@BATTERY@@/train_${R}.sbatch); T=${T%%;*}
echo "rung ${R}: build ${B} -> train ${T} (afterok:${B})"
'''


def main(argv=None):
    names = [r[0] for r in RUNGS]
    if len(set(names + [PILOT[0]])) != len(names) + 1:
        raise SystemExit(f'REFUSING: rung names repeat: {names} + {PILOT[0]}')
    for a in names + [PILOT[0]]:
        for c in names + [PILOT[0]]:
            if a != c and c.endswith(a):
                raise SystemExit(f'REFUSING: rung {a} is a suffix of {c}; the sbatch glob *{TAG}_{a}_* '
                                 f'would match the other arm\'s checkpoints')
    dirty = cc.dirty_warnings()
    if dirty:
        print('WARNING: uncommitted state (the base is read from HEAD; the cluster runs HEAD):\n  '
              + '\n  '.join(dirty))
    base = yaml.safe_load(git_blob(CANONICAL).decode('utf-8'))
    source_bytes, source_sha = committed_sha256(SOURCE_REL)
    encoder_bytes, encoder_sha = committed_sha256(ENCODER_REL)
    prior_bytes, prior_sha = committed_sha256(PRIOR_REL)
    if base['energy_config']['internal_prior_path'] != PRIOR_REL:
        raise SystemExit(f"REFUSING: the committed internal_prior_path is "
                         f"{base['energy_config']['internal_prior_path']!r}, not {PRIOR_REL!r}")
    pb_cap = int(base['buffers']['prior_buffer']['max_size'])
    frac = float(base['cuda_memory_fraction'])
    archive_period = int(base['archive_period'])

    rows, written = [], {}
    all_rungs = [PILOT] + list(RUNGS)
    for i, (rung, n_train, n_heldout) in enumerate(all_rungs):
        pilot = rung == PILOT[0]
        name = f'{TAG}_{rung}'
        cfg = build_arm(base, rung, pilot=pilot)
        check_arm(cfg, base, name, pilot)
        b_wall, b_mem, cpu_h = build_cost(n_train + n_heldout)
        if pilot:
            b_wall = '01:00:00'
        n_rows, n_cond, resident, peak, host, sidecar = train_footprint(n_train, cfg)
        n_archives = (PILOT_EPOCHS if pilot else TRAIN_PRIOR_MAX_STEPS) // archive_period
        t_mem = TRAIN_MEM.get(rung) or f'{max(TRAIN_MEM_DEFAULT_GB, 16 * math.ceil(TRAIN_MEM_MARGIN * (host + HOST_BASE_GB) / 16))}G'
        t_wall = PILOT_WALL if pilot else TRAIN_WALL
        values = {'WALL': b_wall, 'MEM': b_mem, 'CPUS': BUILD_CPUS, 'TAG': TAG, 'RUNG': rung,
                  'WORKDIR': WORKDIR, 'BATTERY': BATTERY, 'N_TRAIN': n_train, 'N_HELDOUT': n_heldout,
                  'CAP': ROWS_PER_CONDITION_CAP, 'OUT': rung_dir(rung), 'CPU_H': f'{cpu_h:.1f}',
                  'PROJECT_ROOT': PROJECT_ROOT, 'DATABASE': DATABASE, 'SHARDS': DATABASE_SHARDS,
                  'SHARDS_PAD': f'{DATABASE_SHARDS:04d}', 'SOURCE_REL': SOURCE_REL,
                  'SOURCE_BYTES': source_bytes, 'SOURCE_SHA': source_sha, 'ENCODER_REL': ENCODER_REL,
                  'ENCODER_BYTES': encoder_bytes, 'ENCODER_SHA': encoder_sha,
                  'PRIOR_REL': PRIOR_REL, 'PRIOR_BYTES': prior_bytes, 'PRIOR_SHA': prior_sha,
                  'SET_THREADS': SET_THREADS}
        written[f'build_{rung}.sbatch'] = fill(BUILD_TEMPLATE, values, f'build_{rung}.sbatch')
        written[f'train_{rung}.sbatch'] = train_sbatch(i, rung, t_wall, t_mem, prior_sha, pilot=pilot)
        # ONE JOB: the larger memory request of the two; the build's wall comes out of the
        # training wall, and the pilot's is the two added
        one_mem = f'{max(int(t_mem[:-1]), int(b_mem[:-1]))}G'
        one_wall = '02:00:00' if pilot else t_wall
        written[f'run_{rung}.sbatch'] = train_sbatch(i, rung, one_wall, one_mem, prior_sha,
                                                     pilot=pilot, one_job=True)
        written[f'{name}.yaml'] = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(written[f'{name}.yaml']) == cfg, name
        rows.append([name, rung, 'fresh', '-', rung, str(n_train), str(n_heldout), b_wall, b_mem,
                     f'{cpu_h:.1f}', TRAIN_GRES, t_mem, f'{n_rows:,.0f}', f'{n_cond:,.0f}',
                     f'{resident:.1f}', f'{peak:.1f}', f'{host:.1f}', f'{sidecar:.1f}',
                     f'{sidecar * (n_archives + 1):,.0f}'])
    header = ['arm', 'rung', 'start', 'warm_src', 'set', 'n_train_molecules', 'n_heldout_molecules',
              'build_wall', 'build_mem', 'build_cpu_hours', 'train_gres', 'train_host_mem',
              'proj_prior_rows', 'proj_train_conditions', 'proj_gpu_resident_GB',
              'proj_gpu_warmup_peak_GB', 'proj_host_GB', 'proj_sidecar_GB',
              'proj_disk_GB_at_max_steps']
    written[f'INDEX_{LEG}.tsv'] = '\n'.join('\t'.join(r) for r in [header] + rows) + '\n'
    written['submit.sh'] = fill(SUBMIT, {'BATTERY': BATTERY, 'TAG': TAG, 'CKPTS': CLUSTER_CKPTS,
                                         'RUNGS_ROOT': RUNGS_ROOT,
                                         'RUNGS': ' '.join(r[0] for r in all_rungs)}, 'submit.sh')
    for fname, text in written.items():
        refuse_local(text, fname)

    for stale in list(HERE.glob('*.sbatch')) + list(HERE.glob(f'{TAG}_*.yaml')):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for fname, text in written.items():
        (HERE / fname).write_text(text, encoding='utf-8', newline='\n')

    print(f'wrote {len(written)} files into {HERE}')
    print(f'\nProjected cost per rung. Build: CPU job, {BUILD_CPUS} CPUs, one builder process per CPU; wall = '
          f'{BUILD_WALL_MARGIN:g}x the projected wall time ({CLUSTER_SLOWDOWN:g}x the local rates) + 30 min, '
          f'memory = {BUILD_MEM_MARGIN:g}x the projected peak. Training: '
          f'GPU memory from the compact-store model (train_footprint: '
          f'{cc.TABLE_BYTES_PER_CONDITION_MAX:,} B per train condition for the conditions table; '
          f'the prior dataset as built and noised; the anchor buffer at max(its max_size, the prior '
          f'rows); the prior buffer at up to {pb_cap:,} rows), before the model and its '
          f'activations; host from {MEMBER_HOST_KB:g} + {COND_FILE_KB:g} KB per train condition. Working assumptions from the local 2000/200 rung, not cluster measurements.')
    print(f"{'rung':<7}{'train/held-out mol':>20}{'build wall':>11}{'build mem':>10}{'build CPU-h':>12}"
          f"{'prior rows':>12}{'train cond':>11}{'GPU resident GB':>16}{'GPU peak GB':>12}{'host GB':>9}"
          f"{'train --mem':>12}{'sidecar GB':>11}{'disk GB':>9}")
    for r in rows:
        print(f'{r[1]:<7}{r[5] + "/" + r[6]:>20}{r[7]:>11}{r[8]:>10}{r[9]:>12}{r[12]:>12}{r[13]:>11}'
              f'{r[14]:>16}{r[15]:>12}{r[16]:>9}{r[11]:>12}{r[17]:>11}{r[18]:>9}')
    print(f'sidecar GB = one buffer sidecar (prior buffer + the anchor buffer, compact rows), '
          f'rewritten every eval_period ({base["eval_period"]}) steps; disk GB = the rolling one plus one '
          f'frozen copy per archive (archive_period {archive_period}, archive_buffers '
          f'{base["archive_buffers"]}) up to train_prior max_steps (pilot: its epochs).')
    print(f'\ncuda_memory_fraction {frac:g} (committed, not overridden) caps the process at: '
          + ', '.join(f'{k} {v * frac:.0f} GB' for k, v in CARDS_GB.items()))
    print(f'every training leg requests {TRAIN_GRES}, {TRAIN_WALL}; the cluster must hold '
          f'{DATABASE} ({DATABASE_SHARDS} shards) and a pull of HEAD.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
