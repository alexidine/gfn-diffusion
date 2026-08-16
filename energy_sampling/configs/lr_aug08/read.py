"""
Read the lr_aug08 battery. Maps offline wandb run dirs -> arm names by the
run_name stored in each run's config, so it does not depend on launch order.

  python configs/lr_aug08/read.py            # cross-arm table, then each arm
  python configs/lr_aug08/read.py a_climb    # one arm, with its LR sweep

EVERYTHING IS A BINNED MEDIAN, never a point sample. decisions.md D30 records
this being got wrong once and the error being the whole result: single samples of
fwd/tb_err against a within-window scatter of +-1 nat produced a reported
"upturn" and a "dead heat" that binned medians showed were both artefacts.
"""
import glob
import json
import os
import statistics
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WANDB = os.path.join(ROOT, 'wandb')

# what each arm has to show, so the read is against a stated expectation rather
# than against whatever the numbers happen to look like
ASKS = {
    'a_fixed':   'bwd/tb_err ~= 15.14 (local_aug08 a_frz) at the 3600-step bin, or the battery is void',
    'a_climb':   'does lr_fused reach ~1.25e-4, and does it PARK or sail past?',
    'b_climbB':  'same landing point as a_climb? the converged LR is what must be stable',
    'b_descend': 'does lr_fused fall from 4.0e-4 toward the band? LR falls while time rises, '
                 'which is the only arm that breaks a_climbs LR-vs-time confound',
    'c_low':     'fixed 1.56e-5 for a full run -- beats a_fixed (15.04) or the sweep was time',
    'c_verylow': 'fixed 5e-6 -- turning over here brackets the optimum near 1.5e-5',
}

# display order; anything found but unlisted is appended, so a new arm never
# silently vanishes from the report the way c_low did on its first read
ORDER = ['a_fixed', 'a_climb', 'b_climbB', 'b_descend', 'c_low', 'c_verylow']

KEYS = ['lr_fused', 'lr_ctrl/peak_scale', 'lr_ctrl/servo_hold',
        'lrprobe/alpha_median', 'lrprobe/fit_ok_rate', 'lrprobe/fit_beyond_rate',
        'lrprobe/second_diff_rel', 'lrprobe/step_norm',
        'bwd/tb_err', 'fwd/tb_err', 'replay/tb_err', 'fwd/tb_resid_clipped',
        # COVERAGE, and it is not optional here. A low-LR arm barely moves, so
        # every residual metric improves simply because the policy is not being
        # dragged anywhere -- "better because it did not train" is a degenerate
        # win and these are what separate it from a real one. EffDim is NOT a
        # live metric on this build (it survives only in comments), so:
        #   eval/wass_debiased  distance to the reference ensemble
        #   fwd/logw_std        spread of the forward log-weights
        #   fwd/ess_frac        effective sample fraction
        'eval/wass_debiased', 'fwd/logw_std', 'fwd/ess_frac']


def run_name_of(run_dir):
    """The arm name, taken from the run's DISPLAY NAME record.

    Not from a config record: an offline run's config never reaches the
    datastore as a config Record (verified 2026-08-08 -- zero config keys on a
    live run dir), and there is no files/config.yaml until a sync. train.py
    passes run_name as wandb's `name`, which does land in the run record."""
    files = glob.glob(os.path.join(run_dir, '*.wandb'))
    if not files:
        return None
    ds = DataStore()
    try:
        ds.open_for_scan(files[0])
    except Exception:
        # a run that started seconds ago has a zero-byte datastore and
        # open_for_scan asserts on the short header. Skipping is correct: a live
        # arm has nothing to read yet.
        return None
    for _ in range(200):                 # the run record is near the head
        try:
            data = ds.scan_data()
        except Exception:
            return None
        if data is None:
            return None
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof('record_type') == 'run':
            for it in rec.run.config.update:
                if it.key == 'run_name':
                    try:
                        v = json.loads(it.value_json)
                        return v.get('value') if isinstance(v, dict) else v
                    except Exception:
                        pass
            return rec.run.display_name or rec.run.run_id
    return None


def scan(run_dir):
    files = glob.glob(os.path.join(run_dir, '*.wandb'))
    if not files:
        return {}, {}
    ds = DataStore()
    try:
        ds.open_for_scan(files[0])
    except Exception:
        return {}, {}
    hist, cfg = {}, {}
    # Series are ALIGNED BY LOGGING EVENT, not by position in their own list.
    # `lrprobe/alpha_median` is absent until the window holds 3 readings and
    # `lr_ctrl/*` is emitted on a different path, so two series in the same run
    # have different lengths -- zipping them by index silently pairs a value
    # with the wrong step, which is exactly the kind of error a sweep would
    # report as a clean trend.
    row = -1
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        kind = rec.WhichOneof('record_type')
        if kind == 'config':
            for it in rec.config.update:
                try:
                    cfg[it.key] = json.loads(it.value_json)
                except Exception:
                    pass
        items = []
        if kind == 'history':
            items = rec.history.item
        elif kind == 'request':
            ph = getattr(rec.request, 'partial_history', None)
            if ph is not None and len(ph.item):
                items = ph.item
        if items:
            row += 1
        for it in items:
            key = it.key or '.'.join(it.nested_key)
            try:
                v = json.loads(it.value_json)
            except Exception:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            hist.setdefault(key, {})[row] = v
    return hist, cfg


def series(hist, key):
    """Values only, in row order -- for trend/bin views that need no alignment."""
    return [v for _, v in sorted(hist.get(key, {}).items())]


def bins(vals, n=6):
    if not vals:
        return []
    sz = max(1, len(vals) // n)
    return [statistics.median(vals[i:i + sz]) for i in range(0, len(vals), sz)][:n]


def fmt(x):
    if x != x:
        return '   nan'
    return f'{x:.4g}'


def sweep(name, hist, per_bin=4):
    """A servo arm climbing from a low seed is an LR SWEEP inside one run.

    This is worth more than the pass/fail the arm was designed for. The servo
    holds each `peak_scale` rung for `period` steps before multiplying, so the
    run visits a ladder of discrete LRs in order, and every eval carries
    `bwd/tb_err` and `lrprobe/alpha_median` at whichever rung was live. Grouping
    by rung therefore gives, from ONE arm and no extra compute:

      * performance(lr) -- where the optimum actually is on this route
      * alpha*(lr)      -- and therefore what alpha* READS at that optimum,
                           which is the calibration `target` needs

    Two caveats, both real. The rungs are visited in time order, so an LR effect
    is confounded with anything else drifting over the run (log-Z settling, the
    buffer filling, the balance controller moving) -- this identifies a
    candidate optimum, it does not measure one. And each rung gets only ~period
    steps, so the policy is never at equilibrium for the LR it is being scored
    at; the reading lags.
    """
    lr = hist.get('lr_fused', {})
    if len(lr) < 8:
        return
    rungs = {}
    for row, v in lr.items():
        rungs.setdefault(round(v, 12), []).append(row)
    if len(rungs) < 3:
        return
    print(f'\n  --- LR sweep within {name} '
          f'({len(rungs)} rungs; see sweep() for the two confounds) ---')
    print(f'    {"lr_fused":>10s} {"n":>3s} {"bwd/tb_err":>11s} {"fwd/tb_err":>11s} '
          f'{"alpha_med":>10s} {"fit_ok":>7s}')
    for v in sorted(rungs):
        idx = sorted(rungs[v])
        if len(idx) < per_bin:
            continue
        idx = idx[len(idx) // 2:]          # drop each rung's leading transient
        cells = []
        for k in ('bwd/tb_err', 'fwd/tb_err', 'lrprobe/alpha_median',
                  'lrprobe/fit_ok_rate'):
            s = hist.get(k, {})
            vals = [s[i] for i in idx if i in s]
            cells.append(statistics.median(vals) if vals else float('nan'))
        # flag a censored alpha*: a 'beyond' fit contributes span (a LOWER
        # bound), so a median taken where most fits are 'beyond' is biased down
        # and must not be read as a measurement of alpha* (step_probe.py
        # servo_reading). fit_ok_rate is the tell.
        mark = ' <-censored' if cells[3] == cells[3] and cells[3] < 0.5 else ''
        print(f'    {v:10.4g} {len(idx):3d} {cells[0]:11.4g} {cells[1]:11.4g} '
              f'{cells[2]:10.4g} {cells[3]:7.3g}{mark}')


def tail(hist, key, frac=0.15):
    """Median of the last `frac` of a series -- ONE statistic used everywhere.

    Not the final value and not a 6-bin split: point samples on these metrics
    have within-window scatter comparable to the effects being chased
    (decisions.md D30), and a coarse bin's edge lands wherever the run happened
    to end. 15% of a 5400-step arm is ~80 evals.
    """
    v = series(hist, key)
    if not v:
        return float('nan')
    return statistics.median(v[-max(1, int(len(v) * frac)):])


def cross_arm(found):
    """The head-to-head table. This is the thing to read first."""
    names = [n for n in ORDER if n in found]
    if len(names) < 2:
        return
    keys = ['lr_fused', 'bwd/tb_err', 'fwd/tb_err', 'fwd/logw_std',
            'lrprobe/alpha_median', 'lrprobe/fit_ok_rate']
    print('=' * 78)
    print('CROSS-ARM, median of the last 15% of evals')
    print('=' * 78)
    print(f'{"":22s}' + ''.join(f'{n:>13s}' for n in names))
    for k in keys:
        cells = ''.join(f'{tail(found[n][1], k):>13.4g}' for n in names)
        print(f'{k:22s}{cells}')
    # alpha* is a LOWER BOUND wherever most fits were 'beyond' -- flag it here
    # rather than let a censored number sit in a comparison table unmarked
    # Two levels, because censoring is graded rather than binary. Below 0.5 the
    # median IS a 'beyond' value, i.e. a bound. Between 0.5 and 0.8 the bottom
    # of the window is still pinned at `span` and drags the median down, just
    # less -- c_low read fit_ok 0.57 with 43% of its window at the floor, which
    # a single 0.5 flag would have passed as clean.
    def ok(n):
        return tail(found[n][1], 'lrprobe/fit_ok_rate')
    hard = [n for n in names if ok(n) < 0.5]
    soft = [n for n in names if 0.5 <= ok(n) < 0.8]
    if hard:
        print(f'\n  !! alpha_median IS A BOUND (fit_ok < 0.5): {", ".join(hard)}')
    if soft:
        print(f'\n  !  alpha_median partly censored, reads LOW (fit_ok < 0.8): '
              f'{", ".join(soft)}')
    print()


def main():
    want = sys.argv[1:] or None
    found = {}
    for d in sorted(glob.glob(os.path.join(WANDB, 'offline-run-*')),
                    key=os.path.getmtime):
        name = run_name_of(d)
        if not name or (want and name not in want):
            continue
        hist, _ = scan(d)
        found[name] = (d, hist)          # later runs of the same arm win

    cross_arm(found)

    for name in ORDER + [n for n in sorted(found) if n not in ORDER]:
        if name not in found:
            continue
        d, hist = found[name]
        rows = max((len(v) for v in hist.values()), default=0)
        print(f'\n=== {name}  ({os.path.basename(d)}, {rows} rows) ===')
        if name in ASKS:
            print(f'    asks: {ASKS[name]}')
        for k in KEYS:
            if k not in hist:
                continue
            v = series(hist, k)
            b = bins(v)
            print(f'  {k:26s} n={len(v):4d}  '
                  f'[{"  ".join(fmt(x) for x in b)}]   final={fmt(b[-1] if b else float("nan"))}')
        sweep(name, hist)


if __name__ == '__main__':
    main()
