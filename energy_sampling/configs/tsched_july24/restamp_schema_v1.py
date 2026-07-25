"""
Back-fill `schema_version: 1` into the stored problem_def of pre-stamp
checkpoints, so the tsched_july24 battery can FULL-resume from the
stab_july21c parent snapshots under the 2026-07-24+ code.

WHY THIS IS LEGITIMATE (and not a compatibility bypass): the failed load's
mismatch report printed exactly ONE differing field -- schema_version
stored='<missing>' vs current=1 -- meaning every identity field
(energy_function, energy_config, prior_path, space_groups, z_primes,
mol/temp/vec_cond) already matches the v1 schema. The parent postdates the
vec_cond addition; it merely predates the stamp itself. Back-filling the
stamp asserts nothing that isn't already true of the stored def.

SAFE BY CONSTRUCTION: assert_problem_match / find_shared_prior /
restore_buffers all compare the FULL (normalized) dict. Adding
schema_version can only remove that one line from the diff -- any real
field mismatch in a stamped file still blocks the load exactly as before.
So blanket-stamping a glob cannot create a false pass.

THREE FILE CLASSES need the stamp (three separate def comparisons):
  - the *_phase1_exit.pt checkpoint itself       (assert_problem_match)
  - its *_buffers.pt sidecars, frozen + rolling  (restore_buffers -- the
    anchor/prior buffer content anchor_seed feeds on)
  - *_prior.pt shared-prior snapshots            (find_shared_prior -- else
    reuse_prior misses and phase 1 re-runs instead of being skipped)
All share the parent run stem, so one glob covers them.

Run ON THE CLUSTER from the energy_sampling directory (sidecars unpickle
PyG batches, so the training env/codebase must be importable):

    python configs/tsched_july24/restamp_schema_v1.py \
        '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/checkpoints/stab_july21c_elj_h512x4_T60_lr5.0e-5_*.pt'

Each modified file is first copied to <name>.pre_schema_v1.bak (skipped if
the backup already exists). Already-stamped files are left untouched.
"""

import glob
import hashlib
import json
import os
import shutil
import sys

import torch

SCHEMA_VERSION = 1

# problem_hash, inlined from utils.py so this script has no repo imports
def problem_hash(problem_def, n_chars=6):
    canonical = json.dumps(problem_def, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:n_chars]


def stamp(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        return 'skipped (not a dict payload)'
    pd = payload.get('problem_def')
    if not isinstance(pd, dict):
        return 'skipped (no problem_def)'
    if 'schema_version' in pd:
        return f"already stamped (v{pd['schema_version']})"

    pd['schema_version'] = SCHEMA_VERSION
    # keep the stored fingerprint consistent with the def it fingerprints
    # (only a legacy fallback guard -- def comparison always wins when present)
    if 'problem_hash' in payload:
        payload['problem_hash'] = problem_hash(pd)

    backup = path + '.pre_schema_v1.bak'
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return 'stamped v1'


if __name__ == '__main__':
    patterns = sys.argv[1:]
    if not patterns:
        sys.exit('usage: restamp_schema_v1.py <glob> [<glob> ...]   (quote the globs)')
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    paths = [p for p in paths if p.endswith('.pt')]
    if not paths:
        sys.exit(f'no .pt files matched: {patterns}')
    width = max(len(os.path.basename(p)) for p in paths)
    for p in paths:
        try:
            result = stamp(p)
        except Exception as e:
            result = f'ERROR: {e}'
        print(f'{os.path.basename(p):<{width}}  {result}')
