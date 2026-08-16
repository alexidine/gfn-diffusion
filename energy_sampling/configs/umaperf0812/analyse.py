"""umaperf0812 -- read the arms back off wandb and answer the two questions.

    python configs/umaperf0812/analyse.py

Q1 (the A/B): does the MLIP work raise GPU occupancy and cut step time, with the
   batch PINNED so nothing else differs?
Q2 (the controller): does the batch controller do anything alarming in a real loop?

PHASE 2 ONLY. Phase 1 is bwd/dataset MLE and never calls the energy, so including it
would dilute exactly the effect being measured -- the mistake that made me claim
"utilization rises with batch" earlier from phase-confounded numbers.
"""
import numpy as np
import wandb

TAG = 'umaperf0812'
PROJECT = 'mkilgour/GFN Energy'
KEYS = ['energy/frac_of_step', 'energy/ms_per_sample', 'gpu/util_recent',
        'train_step_time', 'samples_per_sec', 'Batch Size', 'phase', 'z_cal/p']


def series(run, key, phase2_only=True):
    try:
        hist = run.history(keys=[key], samples=4000, pandas=False)
    except Exception:
        return []
    rows = [(x['_step'], x[key]) for x in hist if x.get(key) is not None]
    if not phase2_only or not rows:
        return [v for _, v in rows]
    try:
        ph = run.history(keys=['phase'], samples=4000, pandas=False)
        ph = {x['_step']: x['phase'] for x in ph if x.get('phase') is not None}
    except Exception:
        return [v for _, v in rows]
    if not ph:
        return [v for _, v in rows]
    steps = sorted(ph)
    vals = [ph[s] for s in steps]

    def phase_at(q):
        i = np.searchsorted(steps, q, side='right') - 1
        return vals[max(i, 0)]

    return [v for s, v in rows if phase_at(s) >= 2]


def summarise(run):
    out = {}
    for k in KEYS:
        v = series(run, k)
        out[k] = (float(np.median(v)) if v else float('nan'), len(v))
    return out


api = wandb.Api(timeout=120)
# NEWEST per name. `{r.name: r for r in ...}` keeps the LAST iterated, which under
# order='-created_at' is the OLDEST -- so the first attempt at this reported a
# crashed 0-step run as `b_optimised` and produced an all-nan comparison. There are
# several runs per name here because the smoke took four attempts.
runs = {}
for r in api.runs(PROJECT, filters={'config.tag': TAG}, order='-created_at'):
    runs.setdefault(r.name, r)
runs = {k: v for k, v in runs.items() if (v.summary.get('_step') or 0) > 0}
if not runs:
    raise SystemExit(f'no runs found with tag {TAG} -- has anything run yet?')

print(f"{'arm':<22} {'state':<9} {'steps':>6} {'frac_of_step':>13} {'ms/sample':>10} "
      f"{'util%':>7} {'step_t':>8} {'batch':>7}")
rows = {}
for name, r in sorted(runs.items()):
    s = summarise(r)
    rows[name] = s
    print(f"{name[:22]:<22} {r.state:<9} {r.summary.get('_step', '?'):>6} "
          f"{s['energy/frac_of_step'][0]:>13.3f} {s['energy/ms_per_sample'][0]:>10.3f} "
          f"{s['gpu/util_recent'][0]:>7.1f} {s['train_step_time'][0]:>8.2f} "
          f"{s['Batch Size'][0]:>7.0f}")

a = next((k for k in rows if 'a_baseline' in k), None)
b = next((k for k in rows if 'b_optimised' in k), None)
if a and b:
    print("\n--- Q1: the controlled A/B (batch pinned, phase 2 only) ---")
    for key, label, better in [('train_step_time', 'step time', 'lower'),
                               ('energy/frac_of_step', 'energy share of step', 'lower'),
                               ('energy/ms_per_sample', 'energy ms/sample', 'lower'),
                               ('gpu/util_recent', 'GPU utilization', 'higher'),
                               ('samples_per_sec', 'samples/sec', 'higher')]:
        va, vb = rows[a][key][0], rows[b][key][0]
        if not (np.isfinite(va) and np.isfinite(vb)):
            print(f"  {label:<24} n/a"); continue
        delta = (vb / va - 1) * 100 if va else float('nan')
        arrow = 'better' if ((delta < 0) == (better == 'lower')) else 'WORSE'
        print(f"  {label:<24} {va:>9.3f} -> {vb:>9.3f}  ({delta:+.1f}%, {arrow})")
    print("\n  NB the absolute numbers do not transfer to the A100 -- different card,")
    print("  host CPU and memory. The DELTA is what carries.")

c = next((k for k in rows if 'c_controller' in k), None)
if c:
    r = runs[c]
    bs = series(r, 'Batch Size')
    print(f"\n--- Q2: the controller, live (phase 2) ---")
    if bs:
        print(f"  batch: min {min(bs):.0f}  median {np.median(bs):.0f}  max {max(bs):.0f}")
        print(f"  distinct sizes visited: {sorted(set(int(x) for x in bs))}")
    util = series(r, 'gpu/util_recent')
    if util:
        print(f"  util: median {np.median(util):.1f}%  min {min(util):.1f}%")
    print("  RED FLAGS: batch pinned at the floor the whole run, a sawtooth across")
    print("  many sizes, or util flat while the batch climbs (the floor is chasing")
    print("  something it cannot move -- which the calibration predicts for uma).")
