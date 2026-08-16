"""
gauss_aug12 -- read every arm's log Z out of its local wandb datastore and table it
against the closed form from spec.py.

PRIMARY METRIC is `eval_fwd/emp_z`, the importance-sampling estimator
`logsumexp(log_r + log_pb - log_pf) - log N`. It is consistent regardless of policy
quality and approaches log Z FROM BELOW, so the test has a hard ceiling: rising to the
analytic value and stopping is a pass, exceeding it means the code or the prediction is
wrong. `log Z learned` is reported alongside but is only meaningful once converged --
certifying correctness from a trained value already nearly produced a false bias report
on this change.

Reports the LAST value and the best (highest) emp_z, because a run that converged and
then drifted should not be read as never having got there.

    python configs/gauss_aug12/read_results.py
"""
import json
import math
import os
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
TAG = spec.TAG
KEYS = ('eval_fwd/emp_z', 'eval_fwd/jensen_z', 'log Z learned',
        'eval_fwd/tb_err', 'train_step_time', '_step')


def scan(path, want_name=False):
    """
    History rows out of the binary datastore. See reference_local_wandb_reading.

    Also returns the run's display_name when asked: the run NAME is only in the
    datastore's `run` record, not in files/wandb-metadata.json (and config.yaml is not
    written for a live run), so matching arms by grepping the files directory silently
    finds nothing.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    rows, step, name = [], None, None
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break                      # live run: partial final record
        if data is None:
            break
        try:
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
        except Exception:
            break
        kind = rec.WhichOneof('record_type')
        if kind == 'run' and name is None:
            name = rec.run.display_name or rec.run.run_id
            if want_name:
                return name
        if kind != 'history':
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or '.'.join(item.nested_key)
            if key not in KEYS:
                continue
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                pass
        if '_step' in row:
            step = row['_step']
        if row:
            row['_step'] = step
            rows.append(row)
    return rows


def find_runs():
    """
    Map run_name -> newest run dir. Resolve by the LAUNCH TIME IN THE DIR NAME, never
    by mtime: the sync service touches and even grows old runs' files.
    """
    out = {}
    if not WANDB.exists():
        return out
    for d in sorted(WANDB.glob('run-*')):
        wf = list(d.glob('run-*.wandb'))
        if not wf or wf[0].stat().st_size == 0:
            continue                         # header race on a just-started run
        try:
            display = scan(wf[0], want_name=True)
        except Exception:
            continue
        if not display or not display.startswith(f'{TAG}_'):
            continue
        arm = display[len(TAG) + 1:]
        if arm not in ARMS:
            continue
        stamp = d.name.split('-')[1]         # YYYYMMDD_HHMMSS, launch time
        prev = out.get(arm)
        if prev is None or stamp > prev[0]:
            out[arm] = (stamp, d)
    return {k: v[1] for k, v in out.items()}


ARMS = [f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}_{"on" if hold else "off"}'
        for sg in spec.SPACE_GROUPS for hold in (True, False)]


def main():
    runs = find_runs()
    print(f"{'arm':<13} {'sg':>3} {'held':>5} {'analytic':>9} {'emp_z':>9} {'err':>8} "
          f"{'best':>9} {'jensen':>8} {'learned':>9} {'steps':>6}")
    print('-' * 92)
    verdicts = []
    for arm in ARMS:
        sg = int(arm.split('_sg')[1].split('_')[0])
        hold = arm.endswith('_on')
        want = spec.analytic_log_z(sg, hold)
        d = runs.get(arm)
        if d is None:
            print(f"{arm:<13} {sg:>3} {str(hold):>5} {want:>9.4f} {'--':>9} {'--':>8} "
                  f"{'--':>9} {'--':>8} {'--':>9} {'not run':>6}")
            continue
        wf = list(d.glob('run-*.wandb'))
        if not wf:
            print(f"{arm:<13} no .wandb datastore in {d.name}")
            continue
        rows = scan(wf[0])
        emp = [(r['_step'], r['eval_fwd/emp_z']) for r in rows
               if 'eval_fwd/emp_z' in r and r['eval_fwd/emp_z'] is not None]
        jen = [r['eval_fwd/jensen_z'] for r in rows if 'eval_fwd/jensen_z' in r]
        lrn = [r['log Z learned'] for r in rows if 'log Z learned' in r]
        steps = max((r['_step'] for r in rows if r.get('_step') is not None), default=0)
        if not emp:
            print(f"{arm:<13} {sg:>3} {str(hold):>5} {want:>9.4f} {'--':>9} {'--':>8} "
                  f"{'--':>9} {'--':>8} {'--':>9} {steps:>6}  (no eval yet)")
            continue
        last = emp[-1][1]
        best = max(v for _, v in emp)
        print(f"{arm:<13} {sg:>3} {str(hold):>5} {want:>9.4f} {last:>9.4f} "
              f"{last - want:>+8.4f} {best:>9.4f} "
              f"{(jen[-1] if jen else float('nan')):>8.3f} "
              f"{(lrn[-1] if lrn else float('nan')):>9.4f} {steps:>6}")
        verdicts.append((arm, want, last, best))

    print()
    # emp_z is a lower bound on log Z, so OVERSHOOT is the falsifying direction and
    # undershoot at a short budget is expected rather than a failure.
    over = [(a, b - w) for a, w, l, b in verdicts if b - w > 0.15]
    if over:
        print("OVERSHOOT -- emp_z cannot exceed log Z except by sampling noise:")
        for a, e in over:
            print(f"  {a}: best exceeds analytic by {e:+.4f}")
    else:
        print("no arm's emp_z exceeds its analytic log Z (the falsifying direction)")

    print("\ndelta check (rows-live minus rows-held, by space group):")
    print(f"  {'sg':>3} {'predicted':>10} {'measured':>10}")
    for sg in spec.SPACE_GROUPS:
        pre = f'{"abcde"[spec.SPACE_GROUPS.index(sg)]}_sg{sg}'
        on = next((b for a, w, l, b in verdicts if a == pre + '_on'), None)
        off = next((b for a, w, l, b in verdicts if a == pre + '_off'), None)
        pred = spec.analytic_log_z(sg, False) - spec.analytic_log_z(sg, True)
        got = f'{off - on:+10.4f}' if (on is not None and off is not None) else f'{"--":>10}'
        print(f"  {sg:>3} {pred:>+10.4f} {got}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
