"""
Pair the in-process occupancy series against the out-of-process ones.

THREE SOURCES, AND THEY ARE NOT THREE MEASUREMENTS.

  ours      `gpu/util_recent` -- a trailing mean over `gpu_util_window_s` of
            samples taken by train.py's sampler thread.
  wandb     `system.gpu.0.gpu` from the run's system stream, ~15 s.
  cluster   a local `nvidia-smi --loop=10` trace (reference_smi.py) -- the exact
            instrument NYU HPC says the scheduler judges, and the same one the
            `joblogs/*_smi.csv` sidecar uses.

wandb and cluster are the same quantity read by two samplers; handoff section 2
found them in agreement on real arms. The cluster column is the one that matters,
because the cancellation bracket (<=0.40 cancelled, >=0.494 survived) is
calibrated against it -- an offset in ours misplaces that bracket, which is the
whole reason the levels have to match and not merely correlate.

THE STATISTIC. `gpu/util_recent` at time t is a mean over [t - window, t]. So
the comparable out-of-process number is the mean of that source over exactly the
same span -- like against like. Two other numbers are printed because they are
what an eye or a naive script would compute, and the difference between them and
the windowed one is not evidence of anything:

  lagged    ours(t) vs the NEAREST out-of-process sample. This is what a wandb
            chart shows. A trailing mean plotted against an instantaneous series
            is displaced by its own lag wherever occupancy is trending, which
            looks exactly like disagreement and is not.
  warm      the windowed comparison restricted to windows lying entirely after
            training began. The sampler thread starts before init (deliberately:
            the MLIP load and prior scan are card time the scheduler counts), so
            a run's first windows are part startup. Both sides see the same span
            so it cancels in principle; `warm` is here to show that it does.

WHAT COUNTS AS AGREEMENT. Not a threshold -- a SHAPE, judged against a measured
floor. b500 and b500r are the same config run twice, so their difference is this
table's resolution. The defect being tested for was a bias whose SIGN FOLLOWED
THE BATCH (-6 pts at 1000, +40 at 7410); a constant offset shared by every arm
is a calibration difference between two samplers, while an offset tracking
batch, T or width is the phase bias surviving. One arm answers neither.

Usage:  python compare_util.py --tag utilphase_sep09 [--ref cluster_ref.csv]
        python compare_util.py <run_id> [...] [--ref cluster_ref.csv]
"""

import sys

import numpy as np
import wandb

PROJECT = 'mkilgour/GFN Energy'


def load_reference(path, gpu_index=0):
    """(timestamps, utils) from reference_smi.py's CSV, for one GPU index."""
    rows = []
    for line in open(path):
        try:
            t, idx, util = line.strip().split(',')
        except ValueError:
            continue
        if int(idx) == gpu_index:
            rows.append((float(t), float(util)))
    a = np.array(sorted(rows))
    return (a[:, 0], a[:, 1]) if len(a) else (np.array([]), np.array([]))


def windowed(ours_t, ours_u, src_t, src_u, window, min_pts, warm_from=None):
    """Mean paired delta against `src` over matched trailing windows."""
    win, lag = [], []
    for t, u in zip(ours_t, ours_u):
        m = (src_t >= t - window) & (src_t <= t)
        if m.sum() < min_pts:
            continue
        if warm_from is not None and t - window < warm_from:
            continue
        win.append(u - float(src_u[m].mean()))
        lag.append(u - float(src_u[np.argmin(np.abs(src_t - t))]))
    if not win:
        return None
    return np.array(win), np.array(lag)


def report(run, ref):
    window = float(run.config.get('gpu_util_window_s', 900) or 900)
    hist = run.history(pandas=True)
    if 'gpu/util_recent' not in hist.columns:
        return None
    ours = hist[['_timestamp', 'gpu/util_recent']].dropna()
    ot = ours['_timestamp'].to_numpy(float)
    ou = ours['gpu/util_recent'].to_numpy(float)
    # the first LOGGED point is the first ten_step_reporting, i.e. training is
    # running by then -- good enough a boundary for 'warm'.
    warm_from = ot.min()

    sysh = run.history(stream='events', pandas=True)
    out = {}
    if 'system.gpu.0.gpu' in sysh.columns:
        s = sysh[['_timestamp', 'system.gpu.0.gpu']].dropna()
        # wandb's stream is ~15 s, so 3 points is ~45 s of a 120 s window; below
        # that the 'disagreement' is mostly which instants each sampler caught.
        out['wandb'] = windowed(ot, ou, s['_timestamp'].to_numpy(float),
                                s['system.gpu.0.gpu'].to_numpy(float), window, 3)
    if ref is not None and len(ref[0]):
        out['cluster'] = windowed(ot, ou, ref[0], ref[1], window, 3)
        out['cluster_warm'] = windowed(ot, ou, ref[0], ref[1], window, 3,
                                       warm_from=warm_from)
    return ou, out


def main(argv):
    ref = None
    if '--ref' in argv:
        i = argv.index('--ref')
        ref = load_reference(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
        print(f'reference trace: {len(ref[0])} samples on GPU 0\n')

    api = wandb.Api()
    if argv and argv[0] == '--tag':
        runs = list(api.runs(PROJECT, {'config.tag': argv[1]}))
    else:
        runs = [api.run(f'{PROJECT}/{rid}') for rid in argv]

    hdr = (f'{"run":26} {"B":>5} {"T":>4} {"W":>5} {"energy":>8} {"n":>4} '
           f'{"ours":>6} {"vs wandb":>9} {"vs clus":>8} {"warm":>7} {"lagged":>7}')
    print(hdr)
    print('-' * len(hdr))
    rows = {}
    for run in sorted(runs, key=lambda r: r.name):
        got = report(run, ref)
        if got is None:
            print(f'{run.name:26} -- no gpu/util_recent (too short, or no sensor)')
            continue
        ou, out = got
        cfg = run.config

        def fmt(key):
            v = out.get(key)
            return f'{v[0].mean():+7.1f}' if v is not None else f'{"--":>7}'

        clus = out.get('cluster')
        # wandb FLATTENS the config (flatten_wandb_params), so nested blocks
        # arrive as 'integrator_T', not config['integrator']['T'].
        print(f'{run.name:26} {cfg.get("batch_size", 0):5} '
              f'{cfg.get("integrator_T", 0):4} '
              f'{cfg.get("model_policy_hidden_dim", 0):5} '
              f'{str(cfg.get("energy_function", "?")):>8} '
              f'{len(ou):4} {ou.mean():6.1f} {fmt("wandb"):>9} {fmt("cluster"):>8} '
              f'{fmt("cluster_warm"):>7} '
              f'{(clus[1].mean() if clus is not None else float("nan")):+7.1f}')
        if clus is not None:
            rows[run.name] = clus[0].mean()

    if len(rows) > 1:
        v = np.array(list(rows.values()))
        print(f'\nacross {len(v)} arms vs the cluster instrument: mean {v.mean():+.1f}, '
              f'spread {v.min():+.1f} .. {v.max():+.1f} (range {np.ptp(v):.1f} pts)')
        base = [rows[k] for k in rows if k.endswith('b500') or k.endswith('b500r')]
        if len(base) == 2:
            print(f'measured floor (b500 vs b500r, same config): '
                  f'{abs(base[0] - base[1]):.1f} pts. Nothing smaller is an effect.')


if __name__ == '__main__':
    main(sys.argv[1:])
