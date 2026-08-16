"""
gauss_aug12 -- does the rows-LIVE arm sit on a higher, irreducible loss floor?

That is the SECOND half of D33's prediction, and the half a log Z table cannot show.
Both arms are expected to reach their own correct log Z; the claim is that the
rows-live arm pays for it, because a live-but-dead dim contributes additively to
Var(log w) and no policy can reduce that contribution -- the energy cannot see the
dim, so there is no gradient that improves it.

WHAT IS AND IS NOT IRREDUCIBLE, stated carefully. In the infinite-capacity,
infinite-T limit the floor is NOT a theorem: a live-but-dead dim's target marginal is
exp(-k*relu(|x|-1)^2), a perfectly valid (near-uniform, soft-edged) distribution that
a GFN could in principle match, driving TB to zero. The floor is a FINITE-T
REPRESENTATIONAL limit: a T-step gaussian chain started from a delta cannot reproduce
a near-flat box marginal exactly, and the residual mismatch lands in Var(log w). Two
consequences worth testing separately:
  - the floor should scale with n_dead                     (this script)
  - the floor should SHRINK with larger T                  (needs a T sweep; not run)
Reporting it as "irreducible" without that distinction would overclaim.

Reads:  fwd/bwd loss, eval_fwd/tb_err, eval_fwd/logw_std, and the LR controller trace
        (lr_ctrl/scale, peak_scale, cal_status) -- because a periodic ray calibration
        perturbs the LR every 500 steps and the recovery transient is easy to mistake
        for a convergence difference between arms.

    python configs/gauss_aug12/read_floors.py
"""
import glob
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ES = HERE.parent.parent
for p in (str(ES.parent), r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools'):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, str(HERE))

import spec  # noqa: E402

WANDB = ES / 'wandb'
KEYS = ('_step', 'fwd/loss', 'bwd/loss', 'eval_fwd/tb_err', 'eval_fwd/logw_std',
        'eval_fwd/emp_z', 'eval_fwd/jensen_z', 'log Z learned',
        'lr_ctrl/scale', 'lr_ctrl/peak_scale', 'lr_ctrl/cal_status',
        'lr_ctrl/divergences', 'lr_fused', 'protocol/stage_index')


def scan(path, keys=KEYS):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows, step, name = [], None, None
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            break
        kind = rec.WhichOneof('record_type')
        if kind == 'run' and name is None:
            name = rec.run.display_name or rec.run.run_id
        if kind != 'history':
            continue
        row = {}
        for item in rec.history.item:
            k = item.key or '.'.join(item.nested_key)
            if k in keys:
                try:
                    row[k] = json.loads(item.value_json)
                except Exception:
                    pass
        if '_step' in row:
            step = row['_step']
        if row:
            row['_step'] = step
            rows.append(row)
    return name, rows


ARMS = [f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}_{"on" if hold else "off"}'
        for sg in spec.SPACE_GROUPS for hold in (True, False)]


def newest_runs():
    out = {}
    for d in sorted(WANDB.glob('run-*')):
        wf = list(d.glob('run-*.wandb'))
        if not wf or wf[0].stat().st_size == 0:
            continue
        try:
            name, rows = scan(wf[0])
        except Exception:
            continue
        if not name or not name.startswith(f'{spec.TAG}_'):
            continue
        arm = name[len(spec.TAG) + 1:]
        if arm not in ARMS:
            continue
        stamp = d.name.split('-')[1]
        steps = max((r.get('_step') or 0) for r in rows) if rows else 0
        prev = out.get(arm)
        # newest by LAUNCH TIME in the dir name (mtime is unusable -- the sync service
        # touches and even grows old runs), but prefer a run with actual history
        if prev is None or (stamp > prev[0] and steps > 0):
            out[arm] = (stamp, rows, steps)
    return out


def tail_stats(rows, key, frac=0.25):
    vals = [(r['_step'], r[key]) for r in rows
            if key in r and r[key] is not None and isinstance(r[key], (int, float))
            and math.isfinite(r[key])]
    if not vals:
        return None
    vals.sort()
    n = max(1, int(len(vals) * frac))
    tail = [v for _, v in vals[-n:]]
    return {'last': vals[-1][1], 'median': statistics.median(tail),
            'min': min(v for _, v in vals), 'n': len(vals)}


def main():
    runs = newest_runs()
    if not runs:
        print('no gauss0812 runs with history found')
        return 0

    print("FLOOR COMPARISON -- final-quarter medians. 'off' = rows LIVE.\n")
    hdr = (f"{'arm':<13} {'steps':>6} {'fwd/loss':>10} {'bwd/loss':>10} "
           f"{'tb_err':>9} {'logw_std':>9} {'emp_z':>9} {'err':>8}")
    print(hdr)
    print('-' * len(hdr))
    table = {}
    for arm in ARMS:
        if arm not in runs:
            continue
        _, rows, steps = runs[arm]
        sg = int(arm.split('_sg')[1].split('_')[0])
        hold = arm.endswith('_on')
        want = spec.analytic_log_z(sg, hold)
        s = {k: tail_stats(rows, k) for k in
             ('fwd/loss', 'bwd/loss', 'eval_fwd/tb_err', 'eval_fwd/logw_std',
              'eval_fwd/emp_z')}
        f = lambda d: f"{d['median']:.4f}" if d else '--'
        empz = s['eval_fwd/emp_z']
        err = f"{empz['median'] - want:+.4f}" if empz else '--'
        print(f"{arm:<13} {steps:>6} {f(s['fwd/loss']):>10} {f(s['bwd/loss']):>10} "
              f"{f(s['eval_fwd/tb_err']):>9} {f(s['eval_fwd/logw_std']):>9} "
              f"{f(empz):>9} {err:>8}")
        table[arm] = s

    print("\nPAIRED off-minus-on. Positive = the rows-LIVE arm is worse.")
    print(f"  {'sg':>3} {'n_dead':>6} {'d fwd/loss':>11} {'d bwd/loss':>11} "
          f"{'d tb_err':>9} {'d logw_std':>11}")
    for sg in spec.SPACE_GROUPS:
        pre = f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}'
        on, off = table.get(pre + '_on'), table.get(pre + '_off')
        if not on or not off:
            continue
        n_dead = len(spec.dead_rows(sg))
        def d(k):
            a, b = off.get(k), on.get(k)
            return f"{b['median'] - a['median'] and (b['median'] - a['median']) * -1:+.4f}" \
                if (a and b) else '--'
        # off - on, explicitly
        def delta(k):
            a, b = off.get(k), on.get(k)
            return f"{a['median'] - b['median']:+.4f}" if (a and b) else '--'
        print(f"  {sg:>3} {n_dead:>6} {delta('fwd/loss'):>11} {delta('bwd/loss'):>11} "
              f"{delta('eval_fwd/tb_err'):>9} {delta('eval_fwd/logw_std'):>11}")

    print("\nLR CONTROLLER TRACE -- how much of the 'slow' is calibration transient.")
    print(f"  {'arm':<13} {'cals':>5} {'divs':>5} {'scale last':>11} "
          f"{'peak_scale':>11} {'lr_fused last':>14}")
    for arm in ARMS:
        if arm not in runs:
            continue
        _, rows, _ = runs[arm]
        cals = [r['lr_ctrl/cal_status'] for r in rows if 'lr_ctrl/cal_status' in r]
        divs = tail_stats(rows, 'lr_ctrl/divergences')
        sc = tail_stats(rows, 'lr_ctrl/scale')
        pk = tail_stats(rows, 'lr_ctrl/peak_scale')
        lr = tail_stats(rows, 'lr_fused')
        g = lambda d: f"{d['last']:.4g}" if d else '--'
        print(f"  {arm:<13} {len(cals):>5} {g(divs):>5} {g(sc):>11} {g(pk):>11} "
              f"{g(lr):>14}")

    print("\nnote: emp_z is a LOWER bound on log Z, so a negative err at a short budget "
          "is undershoot,\nnot bias. Overshoot is the falsifying direction.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
