"""
Gate 0 of the per-condition protocol: the offline sampler must reproduce the trainer's own
logged eval before any per-molecule number drawn through it is read.

It draws an eval-shaped batch from an archive -- rows uniform over the training conditions
file, then over the held-out file, at the run's eval_T and T -- through
eval/cond_panel/sampler.py, and sets the pooled readings beside the values the run logged
at the evals within --window steps of the archive's step:

    eval_fwd/jensen_z, eval_test/jensen_z           mean log w (nats)
    Reasonable Sample Fraction, eval_test/...       bound energy and 0.55 < c_p < 0.95

A reading passes when it lies inside the logged range widened by two of its own standard
errors. The logged range carries the eval's sampling noise and the policy's drift over the
window, which the offline draw cannot share. emp_z (log-mean-exp) is printed and not gated:
with a within-condition log w spread of tens of nats a pooled log-mean-exp is set by a
handful of rows, and two correct samplers disagree on it by several nats.

A failure stops the protocol: it means the offline path differs from the trainer's
(weights, P_B, condition, grid or reward), and every downstream number would inherit it.

    cd energy_sampling
    CUDA_VISIBLE_DEVICES=-1 python -m eval.cond_panel.harness_check \
        --checkpoint D:/.../<stem>_step28000.pt \
        --config configs/cond_tb_sep25/ctb25_extreme_l1_cont3.yaml --wandb-run fp4eydx6
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from eval.cond_panel.sampler import draw, load_run, pooled_levels, reasonable_mask

GATED = (
    # (label, logged key, offline stream, offline quantity)
    ('train jensen_z (nats)', 'eval_fwd/jensen_z', 'train', 'jensen_z'),
    ('held-out jensen_z (nats)', 'eval_test/jensen_z', 'test', 'jensen_z'),
    ('train reasonable fraction', 'Reasonable Sample Fraction', 'train', 'reasonable'),
    ('held-out reasonable fraction', 'eval_test/Reasonable Sample Fraction', 'test', 'reasonable'),
)
REPORTED = (('train emp_z (nats), not gated', 'eval_fwd/emp_z', 'train', 'emp_z'),
            ('held-out emp_z (nats), not gated', 'eval_test/emp_z', 'test', 'emp_z'))


def logged_window(run_id, keys, step, window):
    """Logged values of `keys` at evals within `window` steps of `step`, from the run's
    local wandb history (analysis/pull.py, cloud fallback)."""
    from analysis.pull import pull
    hist = pull(run_id, wanted=keys).history
    out = {}
    for key in keys:
        series = hist.get(key)
        if series is None:
            out[key] = None
            continue
        steps, vals = np.asarray(series, dtype=float)
        keep = (np.abs(steps - step) <= window) & np.isfinite(vals)
        out[key] = (steps[keep], vals[keep])
    return out


def stream_readings(run, batch, n, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    d = draw(run, run.random_rows(batch, n, g), batch_size=batch_size, seed=seed)
    lv = pooled_levels(d['log_w'])
    good = reasonable_mask(run, d['sample_batch']).float()
    lv['reasonable'] = float(good.mean())
    lv['reasonable_se'] = math.sqrt(lv['reasonable'] * (1 - lv['reasonable']) / good.numel())
    lv['emp_z_se'] = float('nan')
    lv['distinct_conditions'] = int(d['condition_id'].unique().numel())
    return lv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--wandb-run', required=True, help='run id whose logged eval is the reference')
    ap.add_argument('--n-train', type=int, default=None, help='defaults to the config eval_num_samples')
    ap.add_argument('--n-test', type=int, default=None, help='defaults to the config test_eval_num_samples')
    ap.add_argument('--window', type=int, default=750, help='steps either side of the archive step')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch-size', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='optional json of every reading')
    args = ap.parse_args()

    if args.device == 'cpu':
        assert not torch.cuda.is_available(), 'CPU run with a visible GPU: set CUDA_VISIBLE_DEVICES=-1'
    run = load_run(args.checkpoint, args.config, device=args.device)
    n_train = args.n_train or int(run.config['eval_num_samples'])
    n_test = args.n_test or int(run.config.get('test_eval_num_samples') or n_train)

    offline = {'train': stream_readings(run, run.conditions, n_train, args.batch_size, args.seed),
               'test': stream_readings(run, run.test_conditions, n_test, args.batch_size, args.seed + 1)}
    keys = [k for _, k, _, _ in GATED + REPORTED]
    logged = logged_window(args.wandb_run, keys, run.step, args.window)

    print(f'\nGate 0 at step {run.step}: offline draw ({n_train} train-condition rows, {n_test} '
          f'held-out rows, uniform over each file with replacement; '
          f'{offline["train"]["distinct_conditions"]} and {offline["test"]["distinct_conditions"]} '
          f'distinct conditions) against the evals logged by run {args.wandb_run} within '
          f'{args.window} steps of the archive. Pass: offline value inside the logged range '
          f'widened by 2 offline SE.\n')
    head = (f'{"reading":38s} {"offline":>9s} {"offline SE":>11s} {"logged min":>11s} '
            f'{"logged median":>14s} {"logged max":>11s} {"evals":>6s}  verdict')
    print(head)
    print('-' * len(head))
    verdicts = {}
    for label, key, stream, qty in GATED + REPORTED:
        val, se = offline[stream][qty], offline[stream].get(f'{qty}_se', float('nan'))
        win = logged.get(key)
        if win is None or win[1].size == 0:
            print(f'{label:38s} {val:9.3f} {se:11.3f} {"--":>11s} {"--":>14s} {"--":>11s} {0:6d}  NO LOGGED VALUE')
            verdicts[label] = 'no logged value'
            continue
        lo, med, hi = float(win[1].min()), float(np.median(win[1])), float(win[1].max())
        gated = (label, key, stream, qty) in GATED
        if gated:
            ok = (lo - 2 * se) <= val <= (hi + 2 * se)
            verdict = 'PASS' if ok else 'FAIL'
        else:
            verdict = 'reported'
        verdicts[label] = verdict
        print(f'{label:38s} {val:9.3f} {se:11.3f} {lo:11.3f} {med:14.3f} {hi:11.3f} {win[1].size:6d}  {verdict}')
    for stream in ('train', 'test'):
        o = offline[stream]
        print(f'\n{stream}: {o["n"]} finite log w ({o["nonfinite"]} non-finite), log w std '
              f'{o["logw_std"]:.2f} nats pooled over conditions, importance ESS fraction {o["ess_frac"]:.4f}')

    failed = [k for k, v in verdicts.items() if v == 'FAIL']
    print('\nGATE 0: ' + ('FAIL on ' + ', '.join(failed) if failed else 'PASS'))
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'step': run.step, 'offline': offline, 'verdicts': verdicts,
                       'logged': {k: (None if v is None else {'steps': v[0].tolist(), 'values': v[1].tolist()})
                                  for k, v in logged.items()}}, fh, indent=1)
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
