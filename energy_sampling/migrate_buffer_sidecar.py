"""Bring a buffer sidecar (`<run>_buffers.pt`) up to the current format version.

    python migrate_buffer_sidecar.py <run>_buffers.pt [--lj-coeff X] [--out PATH] [--dry-run]

WHY THIS EXISTS. On 2026-09-02 the eLJ calibration coefficient moved ONTO the
data (`stamp_lj_coeff`, applied inside `compute_eLJ_energy`), so a stored `.elj`
is calibrated and every row carries a per-graph `lj_coeff`. Sidecars written
before that carry a RAW `.elj` and no stamp; the runtime REFUSES to load them
(CrystalBuffer._refuse_unknown_currency) because stamping a raw row as
calibrated is a silent 2.62x error on mipcas, and nothing downstream could
tell. This script is the one sanctioned way to make such a sidecar loadable:
it stamps the rows AND rescales the raw column in the same step, then marks
the dict with the format version so the runtime knows it has been done.

WHAT IT DOES, per buffer (prior / replay / anchor), via
buffer.migrate_legacy_lj_coeff:
  already format version 2      -> skipped, untouched
  rows already stamped          -> stamp verified against --lj-coeff, dict
                                   marked version 2, nothing rescaled
                                   (a post-relocation, pre-version sidecar)
  rows unstamped, elj route     -> `.elj` and `y` multiplied by --lj-coeff,
                                   rows stamped
  rows unstamped, other routes  -> rows stamped 1.0, nothing rescaled
Anchor `reward` / `energy` are composite totals the old consumer had already
scaled and are never touched. Partially stamped or mixed rows are refused.

THE COEFFICIENT. On the elj route it is the prior dataset's
`thermal_scaling_factor` (mipcas 0.3635836825, nehzor 0.1555787474); on
uma/mace it is 1.0 by construction (train.py refuses anything else). Pass it
with --lj-coeff, or omit it and it is read from the prior .pt named in the
sidecar's own problem_def -- the two are cross-checked when both are available.

The original is kept beside the result as `<name>.pre_lj_migration.bak`
(never picked up by Checkpointer.sidecar_candidates, so it cannot be restored
by accident).
"""
import argparse
import os
import shutil
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
for p in (_here, os.path.dirname(_here)):
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.buffer import (  # noqa: E402
    BUFFER_FORMAT_VERSION, BufferCurrencyError, migrate_legacy_lj_coeff, row_lj_coeff)
from energy_sampling.utils import atomic_save  # noqa: E402

BUFFER_KEYS = ('prior_buffer', 'replay_buffer', 'anchor_buffer')


def coefficient_from_prior(prior_path: str, energy_key: str):
    """The coefficient a run on `prior_path` would calibrate to (train.py
    init_prior_dataset): thermal_scaling_factor on elj, 1.0 elsewhere."""
    if energy_key != 'elj':
        return 1.0
    prior_data = torch.load(prior_path, map_location='cpu', weights_only=False)
    if 'thermal_scaling_factor' not in prior_data:
        return None
    return float(prior_data['thermal_scaling_factor'])


def _column_stats(buf):
    batch = buf['batch']
    stamp = row_lj_coeff(batch)
    elj = getattr(batch, 'elj', None)
    return {
        'rows': int(batch.num_graphs),
        'format_version': int(buf.get('format_version', 1) or 1),
        'stamp': None if stamp is None else sorted(set(stamp.tolist()))[:4],
        'elj_mean': None if elj is None else float(elj.float().mean()),
        'y_mean': None if buf.get('y') is None else float(buf['y'].float().mean()),
    }


def migrate_sidecar_state(state: dict, lj_coeff: float = None, energy_key: str = None):
    """Migrate every buffer in a loaded sidecar dict. Returns (new_state, report).

    `report` maps buffer name -> one of 'skipped (already version N)',
    'absent', or the migrate_legacy_lj_coeff `lj_migration` string, plus the
    before/after column stats. Raises BufferCurrencyError / ValueError rather
    than guessing, exactly as the per-buffer function does.
    """
    problem_def = state.get('problem_def') or {}
    if energy_key is None:
        energy_key = problem_def.get('energy_function')
    if energy_key is None:
        raise ValueError('sidecar problem_def names no energy_function; pass --energy-function')
    if lj_coeff is None:
        prior_path = problem_def.get('prior_path')
        if energy_key != 'elj':
            lj_coeff = 1.0
        elif prior_path and os.path.exists(prior_path):
            lj_coeff = coefficient_from_prior(prior_path, energy_key)
        if lj_coeff is None:
            raise ValueError(
                f'cannot derive lj_coeff: elj route and no readable prior at '
                f'{prior_path!r}; pass --lj-coeff explicitly')
    out = dict(state)
    report = {}
    for key in BUFFER_KEYS:
        buf = state.get(key)
        if buf is None:
            report[key] = {'action': 'absent'}
            continue
        before = _column_stats(buf)
        if before['format_version'] >= BUFFER_FORMAT_VERSION:
            report[key] = {'action': f"skipped (already version {before['format_version']})",
                           'before': before}
            continue
        migrated = migrate_legacy_lj_coeff(buf, lj_coeff, energy_key)
        out[key] = migrated
        report[key] = {'action': migrated['lj_migration'], 'before': before,
                       'after': _column_stats(migrated)}
    out['lj_coeff'] = float(lj_coeff)
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sidecar', help='<run>_buffers.pt to migrate')
    ap.add_argument('--lj-coeff', type=float, default=None,
                    help='the run\'s calibrated coefficient; read from the sidecar\'s '
                         'problem_def prior_path when omitted')
    ap.add_argument('--energy-function', default=None,
                    help='override the sidecar problem_def energy_function')
    ap.add_argument('--out', default=None,
                    help='write here instead of in place (no .bak is made then)')
    ap.add_argument('--dry-run', action='store_true', help='report only, write nothing')
    args = ap.parse_args(argv)

    state = torch.load(args.sidecar, map_location='cpu', weights_only=False)
    problem_def = state.get('problem_def') or {}
    energy_key = args.energy_function or problem_def.get('energy_function')
    derived = None
    if energy_key is not None:
        prior_path = problem_def.get('prior_path')
        if prior_path and os.path.exists(prior_path):
            derived = coefficient_from_prior(prior_path, energy_key)
    if args.lj_coeff is not None and derived is not None \
            and abs(args.lj_coeff - derived) > 1e-6 * max(abs(derived), 1.0):
        raise SystemExit(
            f'--lj-coeff {args.lj_coeff!r} disagrees with the prior named in the sidecar '
            f'({problem_def.get("prior_path")}: {derived!r}). One of them is the wrong '
            f'calibration for these rows; refusing.')
    coeff = args.lj_coeff if args.lj_coeff is not None else derived

    try:
        new_state, report = migrate_sidecar_state(state, coeff, energy_key)
    except (BufferCurrencyError, ValueError) as e:
        raise SystemExit(f'REFUSED: {e}')

    print(f'{args.sidecar}: energy_function={energy_key} lj_coeff={new_state["lj_coeff"]!r} '
          f'(saved at step {state.get("step_ind")})')
    for key in BUFFER_KEYS:
        r = report[key]
        line = f'  {key:14s} {r["action"]}'
        if 'before' in r:
            b = r['before']
            line += (f" | before: rows {b['rows']} v{b['format_version']} stamp {b['stamp']} "
                     f"elj_mean {b['elj_mean']} y_mean {b['y_mean']}")
        if 'after' in r:
            a = r['after']
            line += f" | after: stamp {a['stamp']} elj_mean {a['elj_mean']} y_mean {a['y_mean']}"
        print(line)
    changed = [k for k in BUFFER_KEYS if 'after' in report[k] or
               report[k]['action'].startswith('stamped')]
    if not changed:
        print('nothing to do: every buffer already carries its currency')
        return 0
    if args.dry_run:
        print('dry run: nothing written')
        return 0
    if args.out is None:
        backup = args.sidecar + '.pre_lj_migration.bak'
        if os.path.exists(backup):
            raise SystemExit(f'{backup} already exists -- this sidecar was migrated once; '
                             f'refusing to overwrite the only pre-migration copy')
        shutil.copyfile(args.sidecar, backup)
        print(f'original kept at {backup}')
        target = args.sidecar
    else:
        target = args.out
    atomic_save(new_state, target)
    print(f'wrote {target}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
