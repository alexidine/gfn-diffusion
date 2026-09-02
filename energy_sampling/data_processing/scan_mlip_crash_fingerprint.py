"""
Did a crashed MLIP forward ever reach this run's ONLINE, EVAL or BUFFER energies?

WHY A SCAN RATHER THAN A LOG GREP. Before the 2026-08-30 fix, a non-OOM MLIP failure
substituted an all-ZEROS energy and reported nothing: the flag never reached a metric,
so wandb cannot answer this and the only trace was a stdout line that may be long gone.
But the fabricated value is still SITTING IN THE STATE, because a lattice energy is
`(pot/(sym_mult*z_prime) - gas) * 96.485` and zeroing either leg moves it by the whole
other leg -- order 20,000 kJ/mol against a physical -55 to -60. That is two orders of
magnitude outside the distribution, so it does not need statistics to see.

WHAT IT READS, and why these four. Each is a place a fabricated energy comes to rest:

  * condition_log_z.best_energy -- THE PRIMARY DETECTOR. A per-condition running
    MINIMUM that is never reset, so it integrates every crash the run ever had, and a
    spuriously LOW value (the gas-leg direction) is exactly what a minimum preserves.
    It survives buffer eviction, which none of the others do.
  * prior_buffer.y / replay_buffer.y / anchor_buffer.energy -- rows admitted while
    holding a fabricated energy. These turn over, so absence here is weaker evidence
    than absence in best_energy.

SCALE-FREE BY NECESSITY. elj energies run -400..-650, uma lattice energies -55..-60,
and the toys differ again, so a fixed threshold would be wrong for at least one route.
The test is instead "how far outside its OWN distribution", using a median/IQR robust
band -- a crash value is ~100x out, so the multiple is not delicate.

WHAT A CLEAN RESULT DOES AND DOES NOT MEAN. Clean means no fabricated energy is
resident in the state that was checkpointed. It does NOT prove no crash occurred: a
poisoned row can be evicted from a buffer, and a run whose crash predates its last
best_energy reset would not show. Treat it as strong evidence, not proof -- and note
that MLIP routes only are scanned, since analytic energies cannot reach this path.

USAGE
    python -m data_processing.scan_mlip_crash_fingerprint <checkpoint_dir> [--all-routes]
"""
import argparse
import glob
import os

import torch

#: A value this many robust-sigma outside the median is reported. A crash lands ~100x
#: out, so this is deliberately loose: it should surface junk for a human to dismiss
#: rather than silently set a bar that a real crash could sit under.
ROBUST_SIGMA = 20.0
MLIP_ROUTES = ('uma', 'mace')


def _robust_band(v):
    med = v.median()
    # 1.349 sigma per IQR for a normal; only used to set an order of magnitude
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    sigma = float((q3 - q1) / 1.349)
    if sigma <= 0:
        sigma = float(v.abs().median()) or 1.0
    return float(med), sigma


#: Below this there is no distribution to test against, so the robust band is
#: meaningless. Such a field is REPORTED as un-judgeable rather than dropped -- an
#: UNCONDITIONAL run has library_size 1, so best_energy is a single scalar, and
#: silently skipping it would remove the PRIMARY detector from every such run while
#: the summary still read "no outliers". That is the false-clear this tool exists to
#: avoid, and it bit the first version of it.
MIN_FOR_BAND = 8


def _scan_tensor(v):
    """(n, min, max, n_outliers) with n_outliers None when the field is too small to
    judge distributionally. None only when there is genuinely no data."""
    if not torch.is_tensor(v) or v.numel() == 0:
        return None
    v = v.flatten().float()
    v = v[torch.isfinite(v)]          # +inf = unvisited condition, NaN = post-fix crash
    if v.numel() == 0:
        return None
    if v.numel() < MIN_FOR_BAND:
        return int(v.numel()), float(v.min()), float(v.max()), None
    med, sigma = _robust_band(v)
    out = int(((v - med).abs() > ROBUST_SIGMA * sigma).sum())
    return int(v.numel()), float(v.min()), float(v.max()), out


def scan_file(path, all_routes=False):
    try:
        d = torch.load(path, weights_only=False, map_location='cpu')
    except Exception as e:
        return [(os.path.basename(path), 'UNREADABLE', type(e).__name__, 0, 0, 0)]
    if not isinstance(d, dict):
        return []

    pdef = d.get('problem_def') or {}
    route = pdef.get('energy_function', '?')
    if not all_routes and route not in MLIP_ROUTES:
        return []   # analytic energies cannot reach the MLIP crash path

    targets = []
    clz = d.get('condition_log_z')
    if isinstance(clz, dict):
        targets.append(('best_energy', clz.get('best_energy')))
    for buf, key in (('prior_buffer', 'y'), ('replay_buffer', 'y'),
                     ('anchor_buffer', 'energy')):
        b = d.get(buf)
        if isinstance(b, dict):
            targets.append((f'{buf}.{key}', b.get(key)))

    rows = []
    for name, tensor in targets:
        got = _scan_tensor(tensor)
        if got:
            n, lo, hi, bad = got
            rows.append((os.path.basename(path), route, name, n, lo, hi, bad))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory')
    ap.add_argument('--all-routes', action='store_true',
                    help='also scan elj/toy runs, which cannot reach this path '
                         '(useful only to calibrate what normal looks like)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.directory, '*.pt')))
    print(f'{len(files)} checkpoint(s) in {args.directory}\n')
    print(f'{"file":<48} {"route":<6} {"field":<22} {"n":>7} '
          f'{"min":>12} {"max":>12} {"outliers":>9}')
    scanned = flagged = unjudged = 0
    for f in files:
        for row in scan_file(f, args.all_routes):
            name, route, field, n, lo, hi, bad = row
            scanned += 1
            if bad is None:
                unjudged += 1
                cell, mark = '   n/a', '  <== EYEBALL (too few to test)'
            else:
                flagged += bool(bad)
                cell = f'{bad:>9}'
                mark = '  <== CHECK' if bad else ''
            print(f'{name[:48]:<48} {route:<6} {field:<22} {n:>7} '
                  f'{lo:>12.2f} {hi:>12.2f} {cell}{mark}')

    print()
    if not scanned:
        print('NOTHING SCANNED. No checkpoint carried an MLIP-route problem_def with '
              'readable energies -- this is not a clean result, it is an empty one. '
              'Pass --all-routes to confirm the files parse at all.')
    elif flagged:
        print(f'{flagged} field(s) flagged. A crashed leg lands ~20,000 kJ/mol out; '
              f'anything smaller is more likely ordinary high-energy junk. Compare the '
              f'extreme against the bulk before concluding.')
    else:
        print(f'{scanned} field(s) scanned, no outliers. No fabricated energy is '
              f'resident in this state. NOT proof no crash occurred -- buffer rows '
              f'turn over; best_energy is the durable one.')
    if unjudged:
        print()
        print(f'{unjudged} field(s) were too small for a distributional test and are '
              f'marked EYEBALL above -- read their min/max against the bulk yourself. '
              f'An UNCONDITIONAL run puts best_energy here (library_size 1), i.e. the '
              f'PRIMARY detector, so do not skip them.')


if __name__ == '__main__':
    main()
