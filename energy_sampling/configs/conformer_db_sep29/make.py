"""conformer_db_sep29: the offline QM9 conformer database (build_conformer_database.py) as a
CPU-only SLURM array, one task per shard. Writes submit_db.sbatch, whole, from TEMPLATE below.

    python configs/conformer_db_sep29/make.py [--n-shards 400]

--pool-need is POOL_NEED, 0: no encoder pool is excluded, the universe is all of QM9 (owner
decision 2026-09-29; build_conformer_set.py's default), passed explicitly so the sbatch says it.
Derived here from the local tree and written into the sbatch, which checks each on-cluster
before any task runs:
  * the prior's size and sha256, from the local conformer_prior_v2.pt (the committed config's
    energy_config.internal_prior_path). The prior is not in git: the owner places it by hand at
    ${WORKDIR}/conformer_prior_v2.pt, and the sbatch refuses any other size or sha256;
  * the sha256 of qm9_index.tsv.gz beside this file (133,728 rows `dataset_index, smiles` of
    qm9_dataset.pt), which the sbatch requires present and unchanged.

REFUSES: a local configs/conformer_mk.yaml that differs from HEAD (the cluster reads the
committed one), and any drive-letter path in what it writes -- a local path passes every local
check, because the file exists on the dev box, and kills every task on-cluster.

Output: OUTROOT (one shard file per task); PILOT=1 at sbatch runs --max-molecules PILOT_MOLECULES
per shard into PILOT_OUT. Resubmitting the same file continues each shard from its file;
build_conformer_database.py refuses a shard file written under another header.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ES = HERE.parents[1]                                     # energy_sampling/
BATTERY = HERE.name
N_SHARDS = 400
PROJECT_ROOT = '/scratch/mk8347/projects/gfn_cond'
WORKDIR = f'{PROJECT_ROOT}/gfn-diffusion/energy_sampling'
OUTROOT = '/scratch/mk8347/data/conformer_datasets/qm9_db_sep29'
PILOT_OUT = '/scratch/mk8347/data/conformer_datasets/qm9_db_sep29_pilot'
PILOT_MOLECULES = 5
PRIOR_NAME = 'conformer_prior_v2.pt'                    # at ${WORKDIR}/, placed by hand
INDEX_NAME = 'qm9_index.tsv.gz'
#: leading QM9 SMILES excluded as the encoder's pool: none (owner decision 2026-09-29)
POOL_NEED = 0
CONFIG_REL = 'configs/conformer_mk.yaml'
WALL = '06:00:00'
MEM = '4G'
#: files a task imports or reads that must be committed and pushed (checked, reported)
SHIPPED = ('build_conformer_database.py', 'build_conformer_set.py', CONFIG_REL,
           f'configs/{BATTERY}/{INDEX_NAME}', f'configs/{BATTERY}/submit_db.sbatch')
#: a drive letter not preceded by a letter or digit (so a URL scheme does not match)
DRIVE = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]')

TEMPLATE = r'''#!/bin/bash
#SBATCH --time=@@WALL@@
#SBATCH --mem=@@MEM@@
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-@@LAST@@
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=cdb
#SBATCH --output=@@WORKDIR@@/configs/@@BATTERY@@/joblogs/%x_%A_%a.out

# @@BATTERY@@: the offline QM9 conformer database (build_conformer_database.py), CPU only, one
# shard per array task (shard = SLURM_ARRAY_TASK_ID of @@N_SHARDS@@). WRITTEN BY make.py: do
# not edit by hand, rerun make.py. Resubmit this same file to continue every shard from its
# file; the builder refuses a shard file written under another header.
# PILOT=1 sbatch --array=0-3 --time=01:00:00 configs/@@BATTERY@@/submit_db.sbatch runs
# --max-molecules @@PILOT_MOLECULES@@ per shard into @@PILOT_OUT@@.
# Report: python build_conformer_database.py summarize <output dir>
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=@@PROJECT_ROOT@@
WORKDIR=${PROJECT_ROOT}/gfn-diffusion/energy_sampling
ARMS=${WORKDIR}/configs/@@BATTERY@@
LOGS=${ARMS}/joblogs
N_SHARDS=@@N_SHARDS@@
POOL_NEED=@@POOL_NEED@@
PRIOR=${WORKDIR}/@@PRIOR_NAME@@
PRIOR_BYTES=@@PRIOR_BYTES@@
PRIOR_SHA256=@@PRIOR_SHA256@@
INDEX=${ARMS}/@@INDEX_NAME@@
INDEX_SHA256=@@INDEX_SHA256@@
mkdir -p ${LOGS}

SHARD=${SLURM_ARRAY_TASK_ID}
if [ -z "${SHARD}" ]; then echo "FATAL: no SLURM_ARRAY_TASK_ID; submit this file as an array" >&2; exit 1; fi

HAVE=$(stat -c %s ${PRIOR} 2>/dev/null || echo 0)
if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: the prior at ${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES} (sha256 ${PRIOR_SHA256}); upload @@PRIOR_NAME@@ there" >&2; exit 1
fi
HAVE=$(sha256sum ${PRIOR} | cut -d' ' -f1)
if [ "${HAVE}" != "${PRIOR_SHA256}" ]; then
    echo "FATAL: the prior at ${PRIOR} has sha256 ${HAVE}, expected ${PRIOR_SHA256}" >&2; exit 1
fi
if [ ! -f "${INDEX}" ]; then echo "FATAL: the QM9 index table ${INDEX} is missing (pull the repository)" >&2; exit 1; fi
HAVE=$(sha256sum ${INDEX} | cut -d' ' -f1)
if [ "${HAVE}" != "${INDEX_SHA256}" ]; then
    echo "FATAL: ${INDEX} has sha256 ${HAVE}, expected ${INDEX_SHA256}" >&2; exit 1
fi

if [ -n "${PILOT:-}" ]; then
    OUT=@@PILOT_OUT@@
    EXTRA="--max-molecules @@PILOT_MOLECULES@@"
else
    OUT=@@OUTROOT@@
    EXTRA=""
fi
mkdir -p ${OUT}
J=${LOGS}/shard${SHARD}_${SLURM_JOB_ID}
echo "array ${SLURM_ARRAY_TASK_ID} -> shard ${SHARD}/${N_SHARDS} -> ${OUT} ${PILOT:+[PILOT]}"

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
        python -u build_conformer_database.py build --source ${INDEX} --prior ${PRIOR} \
            --pool-need ${POOL_NEED} --config @@CONFIG_REL@@ --out-dir ${OUT} \
            --shard ${SHARD} --n-shards ${N_SHARDS} --threads 1 ${EXTRA}
    " 2>&1 | tee ${J}.log
'''


def assert_no_local_paths(text: str, where: str):
    """A drive-letter path passes every local check (the file exists on the dev box) and
    kills the job on-cluster."""
    bad = [ln for ln in text.splitlines() if DRIVE.search(ln)]
    if bad:
        raise SystemExit(f'{where}: local path(s) would ship to the cluster:\n  '
                         + '\n  '.join(bad))


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def render(values: dict) -> str:
    """TEMPLATE with every @@NAME@@ filled; refuses a placeholder left over or a local path."""
    text = TEMPLATE
    for k, v in values.items():
        text = text.replace(f'@@{k}@@', str(v))
    left = sorted(set(re.findall(r'@@([A-Z_]+)@@', text)))
    if left:
        raise SystemExit(f'unfilled placeholder(s) in the sbatch: {left}')
    assert_no_local_paths(text, 'submit_db.sbatch')
    return text


def uncommitted(paths) -> list:
    """The paths (relative to energy_sampling/) git reports as modified or untracked; [] when
    git is unavailable."""
    try:
        r = subprocess.run(['git', '-C', str(ES), 'status', '--porcelain', '--', *paths],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln[3:] for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else []


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n-shards', type=int, default=N_SHARDS)
    args = ap.parse_args(argv)
    if args.n_shards < 1:
        raise SystemExit('--n-shards must be >= 1')
    sys.path.insert(0, str(ES))
    import yaml

    import build_conformer_set as bcs

    config = ES / CONFIG_REL
    head = bcs._provenance_git(config).get('config_matches_head')
    if head is False:
        raise SystemExit(f'REFUSING: {config} differs from HEAD; the cluster reads the committed '
                         f'file, so the database would be built under another energy')
    if head is None:
        print(f'WARNING: could not compare {config} with HEAD (no git?)')
    with open(config, 'r', encoding='utf-8') as f:
        prior_rel = (yaml.safe_load(f).get('energy_config') or {}).get('internal_prior_path')
    prior = Path(prior_rel) if prior_rel and Path(prior_rel).is_absolute() else ES / str(prior_rel)
    if prior.name != PRIOR_NAME or not prior.is_file():
        raise SystemExit(f'REFUSING: the config names the prior {prior_rel!r}; expected a local '
                         f'{PRIOR_NAME}, whose size and sha256 the cluster copy is checked against')
    index = HERE / INDEX_NAME
    if not index.is_file():
        raise SystemExit(f'REFUSING: {index} is missing')
    values = {'WALL': WALL, 'MEM': MEM, 'LAST': args.n_shards - 1, 'N_SHARDS': args.n_shards,
              'WORKDIR': WORKDIR, 'PROJECT_ROOT': PROJECT_ROOT, 'BATTERY': BATTERY,
              'POOL_NEED': POOL_NEED, 'PRIOR_NAME': PRIOR_NAME, 'PRIOR_BYTES': prior.stat().st_size,
              'PRIOR_SHA256': sha256_of(prior), 'INDEX_NAME': INDEX_NAME,
              'INDEX_SHA256': sha256_of(index), 'OUTROOT': OUTROOT, 'PILOT_OUT': PILOT_OUT,
              'PILOT_MOLECULES': PILOT_MOLECULES, 'CONFIG_REL': CONFIG_REL}
    text = render(values)
    (HERE / 'submit_db.sbatch').write_text(text, encoding='utf-8', newline='\n')
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').touch()           # SLURM opens --output before the script can mkdir
    print(f"wrote {HERE / 'submit_db.sbatch'}: --array=0-{args.n_shards - 1}, pool need {POOL_NEED}, "
          f"prior {values['PRIOR_BYTES']} B sha256 "
          f"{values['PRIOR_SHA256']}, index sha256 {values['INDEX_SHA256']}")
    pending = uncommitted(SHIPPED)
    if pending:
        print('WARNING: a task reads these, and they are not committed as they are here; commit '
              'and push both repositories before submitting:\n  ' + '\n  '.join(pending))
    return values


if __name__ == '__main__':
    main()
