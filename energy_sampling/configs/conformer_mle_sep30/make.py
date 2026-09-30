r"""conformer_mle_sep30 -- converge the conformer MLE warm-up (stage `train_prior` of protocol
`conformer_conditional_tb`) on rungs cut from the QM9 conformer database ON THE CLUSTER.

    python configs/conformer_mle_sep30/make.py

TWO JOBS PER RUNG, chained by submit.sh:
  * build_<rung>.sbatch (CPU): build_conformer_set.py --database DB --rows-per-condition-cap 16
    cuts the rung (train + held-out conditions, the database prior) into RUNGS_ROOT/<rung>, then
    build_conformer_references.py --database DB writes both reference tables (the per-condition
    floors the offline eval reads). IT MUST RUN ON THE CLUSTER, in the training container: the
    run re-embeds every member with its own RDKit, and ConformerModeller refuses a file whose
    stored reference conformers another RDKit embedded (_refuse_reference_mismatch). Rerunning
    it skips what is already on disk (a set with a manifest, a written table).
  * train_<rung>.sbatch (GPU): configs/final_sep19/make.py::SBATCH through
    configs/conformer_cond/make.py::conformer_template (the RESUME / FRESH / CK_STEP block, the
    dead-arm sentinel, the RDKit / MMFF import guard, the nvidia-smi and sacct sidecars), pinned
    to its row of INDEX_a.tsv, with the data guard replaced by RUNG_GUARD (the rung's files
    present, the RDKit version read from its manifest, the committed prior's sha256). A first
    launch is FRESH; RESUBMITTING THE SAME FILE continues the arm from its own _running.pt.

    bash configs/conformer_mle_sep30/submit.sh pilot    # build -> train, afterok
    bash configs/conformer_mle_sep30/submit.sh r20k
    sbatch configs/conformer_mle_sep30/train_r20k.sbatch   # every later leg

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
database rung of 2026-09-30, not cluster measurements): conditions and prior rows per molecule,
the set builder's and reference builder's CPU seconds, the builder's peak host memory, and the
training footprint model -- ROW_KB per prior row, COND_KB per train condition, the prior
dataset held once as loaded, once cloned into the anchor buffer (whole: anchor thinning is off),
and once more transiently while the prior buffer seed is cut to its max_size.
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
BUILD_CPUS = 4                          # the set builder is one process (2 torch threads); the references use them all
SET_THREADS = 2
#: CPU seconds per molecule of the set builder, walk refusals included: the owner's 0.71 s per
#: key; locally 1503 s for 2,200 kept molecules (0.68 s)
SET_S_PER_MOLECULE = 0.71
#: CPU seconds per condition of the database-floor reference build: locally 803 s over 3,353
REFS_S_PER_CONDITION = 0.24
#: conditions per molecule (stereoisomers): locally 3,486 / 2,000 train and 355 / 200 held-out
CONDITIONS_PER_MOLECULE = 1.75
#: prior rows per train molecule at cap 16: locally 22,760 / 2,000
ROWS_PER_MOLECULE = 11.4
#: the set builder's peak host memory, GB = BUILD_MEM_BASE_GB + BUILD_MEM_PER_1K_MOL_GB x
#: molecules / 1000 (train + held-out): the peak resident set of local builds of 110 and 440
#: molecules (1.41 and 2.45 GB, 2026-09-30), extrapolated linearly. The same builds ran at 1.9 to
#: 4.6 s per molecule on a shared, busy CPU, hence the wall margin.
BUILD_MEM_BASE_GB = 1.06
BUILD_MEM_PER_1K_MOL_GB = 3.15
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
#: the owner's training footprint model: KB per prior row in memory, KB per train condition
ROW_KB = 35.6
COND_KB = 35.4
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

def build_cost(n_mol):
    """(wall 'HH:MM:SS', mem 'NG', CPU-hours) for a build of n_mol molecules (train + held-out)."""
    n_cond = CONDITIONS_PER_MOLECULE * n_mol
    set_s = SET_S_PER_MOLECULE * n_mol
    refs_s = REFS_S_PER_CONDITION * n_cond
    wall_s = BUILD_WALL_MARGIN * (set_s + refs_s / BUILD_CPUS) + 1800
    hours = min(47, max(1, math.ceil(wall_s / 3600)))
    mem = BUILD_MEM_MARGIN * (BUILD_MEM_BASE_GB + BUILD_MEM_PER_1K_MOL_GB * n_mol / 1000)
    return f'{hours:02d}:00:00', f'{max(8, math.ceil(mem))}G', (set_s + refs_s) / 3600


def train_footprint(n_train, prior_buffer_cap):
    """(prior rows, train conditions, GB: GPU resident, GPU warm-up peak, host, one buffer
    sidecar). The sidecar holds the prior and anchor buffers (replay is empty under train_prior)
    at ROW_KB a row."""
    rows = ROWS_PER_MOLECULE * n_train
    conds = CONDITIONS_PER_MOLECULE * n_train
    one = rows * ROW_KB * 1e3 / 1e9
    cond_gb = conds * COND_KB * 1e3 / 1e9
    pb = min(rows, prior_buffer_cap) * ROW_KB * 1e3 / 1e9
    resident = one + one + pb + cond_gb       # dataset, anchor buffer (whole), prior buffer, conditions
    peak = resident + one                     # + the prior-buffer seed's transient whole clone
    host = one + one + cond_gb                # the loaded file and its thermal-noised clone, the conditions
    sidecar = pb + one
    return rows, conds, resident, peak, host, sidecar


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
# @@OUT@@. CPU only, IN THE TRAINING CONTAINER: the training run re-embeds every member with
# this RDKit and refuses a file whose reference conformers another RDKit embedded.
# WRITTEN BY make.py: do not edit by hand. Rerunning skips a set that has its manifest and a
# table already written (the reference builder resumes from its .parts directory).
# Projected: @@CPU_H@@ CPU-hours (set builder single-process, references on @@CPUS@@ workers).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=@@PROJECT_ROOT@@
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
LOGS=${WORKDIR}/configs/@@BATTERY@@/joblogs
DB=@@DATABASE@@
OUT=@@OUT@@
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
                --database ${DB} --rows-per-condition-cap @@CAP@@ --threads @@SET_THREADS@@ || exit 1
        fi
        for SIDE in train heldout; do
            TABLE=${OUT}/conditions_\${SIDE}.references.pt
            if [ -f \${TABLE} ]; then echo \"REFERENCES: \${TABLE} exists, skipping\"; continue; fi
            python -u build_conformer_references.py --conditions ${OUT}/conditions_\${SIDE}.pt \
                --config configs/conformer_mk.yaml --database ${DB} --workers @@CPUS@@ --threads 1
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
RUNG_GUARD = r"""# RUNG GUARD. The rung was built ON THE CLUSTER by build_${{SET}}.sbatch (the dependency submit.sh
# sets). Every file the run and the offline eval read must be there; the RDKit version the
# conformer guard below asserts is the one the rung's manifest records, since the run re-embeds
# every member and ConformerModeller refuses reference conformers another RDKit embedded. The
# fitted prior is committed: its sha256 is HEAD's. NIG_EXPORT is the crystal template's
# MXtalTools export, empty on this route.
NIG_EXPORT=""
RUNG=${{DATA}}/${{SET}}
for F in manifest.json conditions_train.pt conditions_heldout.pt prior_train.pt conditions_train.references.pt conditions_heldout.references.pt; do
    if [ ! -s ${{RUNG}}/${{F}} ]; then echo "FATAL: ${{RUNG}}/${{F}} is missing; run build_${{SET}}.sbatch first" >&2; exit 1; fi
done
RDKIT=$(grep -o '"rdkit": *"[^"]*"' ${{RUNG}}/manifest.json | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
if [ -z "${{RDKIT}}" ]; then echo "FATAL: ${{RUNG}}/manifest.json names no RDKit version" >&2; exit 1; fi
if [ "$(sha256sum ${{WORKDIR}}/@@PRIOR_REL@@ | cut -d' ' -f1)" != "@@PRIOR_SHA@@" ]; then
    echo "FATAL: ${{WORKDIR}}/@@PRIOR_REL@@ is not HEAD's (sha256 @@PRIOR_SHA@@); git pull" >&2; exit 1
fi
"""


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


def train_sbatch(row, rung, wall, mem, prior_sha, pilot=False):
    text = cc.conformer_template(fin.SBATCH)
    text = cc._replace_once(text, cc.DATA_GUARD, fill(RUNG_GUARD, {'PRIOR_REL': PRIOR_REL,
                                                                   'PRIOR_SHA': prior_sha}, 'guard'),
                            'conformer data guard')
    text = cc._replace_once(text, cc._RDKIT_COLUMN, _SET_COLUMN, 'rdkit index column')
    text = cc._replace_once(text, '#SBATCH --array=0-{last}', f'#SBATCH --array={row}', 'array line')
    text = cc._replace_once(text, '#SBATCH --gres=gpu:a100:1', f'#SBATCH --gres={TRAIN_GRES}', 'gres line')
    text = cc._replace_once(text, '#SBATCH --mem=48G', f'#SBATCH --mem={mem}', 'mem line')
    return text.format(
        wall=wall, tag=TAG, battery=BATTERY, ckpts=CLUSTER_CKPTS, data=RUNGS_ROOT, leg=LEG,
        seed_block=cc.SEED_FRESH, last=row,
        what=(f'PILOT: {PILOT_EPOCHS} training steps (epochs) on the pilot rung.' if pilot else
              f'MLE warm-up (train_prior, max_steps {TRAIN_PRIOR_MAX_STEPS:,}) on rung {rung}, '
              f'FRESH on the first launch; resubmit this file to continue.'))


SUBMIT = r'''#!/bin/bash
# @@BATTERY@@: submit one rung's build (CPU) and its training (GPU, afterok on the build).
#     bash configs/@@BATTERY@@/submit.sh <rung>        rungs: @@RUNGS@@
# Later legs: sbatch configs/@@BATTERY@@/train_<rung>.sbatch (it resumes the arm's own _running.pt).
# WRITTEN BY make.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${1:-}
case " @@RUNGS@@ " in *" ${R} "*) ;; *) echo "usage: bash $0 <rung>, rung one of: @@RUNGS@@" >&2; exit 2;; esac
if [ "${R}" = "pilot" ] && ls @@CKPTS@@/*@@TAG@@_pilot_*_running.pt >/dev/null 2>&1; then
    echo "the pilot arm already has checkpoints, and a new pilot would RESUME them; to re-pilot:" >&2
    echo "    rm @@CKPTS@@/*@@TAG@@_pilot_* ; rm -r @@RUNGS_ROOT@@/pilot" >&2
    exit 1
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
            b_wall, b_mem = '01:00:00', '8G'
        n_rows, n_cond, resident, peak, host, sidecar = train_footprint(n_train, pb_cap)
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
    print(f'\nProjected cost per rung. Build: CPU job, {BUILD_CPUS} CPUs; wall = {BUILD_WALL_MARGIN:g}x the '
          f'projected CPU time + 30 min, memory = {BUILD_MEM_MARGIN:g}x the projected peak. Training: '
          f'GPU memory from the footprint model ({ROW_KB} KB per prior row, {COND_KB} KB per train '
          f'condition; the prior dataset once as loaded, once in the anchor buffer, once transiently '
          f'while the prior buffer seed is cut to {pb_cap:,} rows), before the model and its '
          f'activations. Working assumptions from the local 2000/200 rung, not cluster measurements.')
    print(f"{'rung':<7}{'train/held-out mol':>20}{'build wall':>11}{'build mem':>10}{'build CPU-h':>12}"
          f"{'prior rows':>12}{'train cond':>11}{'GPU resident GB':>16}{'GPU peak GB':>12}{'host GB':>9}"
          f"{'train --mem':>12}{'sidecar GB':>11}{'disk GB':>9}")
    for r in rows:
        print(f'{r[1]:<7}{r[5] + "/" + r[6]:>20}{r[7]:>11}{r[8]:>10}{r[9]:>12}{r[12]:>12}{r[13]:>11}'
              f'{r[14]:>16}{r[15]:>12}{r[16]:>9}{r[11]:>12}{r[17]:>11}{r[18]:>9}')
    print(f'sidecar GB = one buffer sidecar (prior buffer + the whole anchor buffer at {ROW_KB} KB a row), '
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
