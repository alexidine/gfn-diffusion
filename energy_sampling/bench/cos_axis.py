"""
WHAT DOES THE BENCH'S NOISE AXIS LOOK LIKE IN THE UNITS THE REAL SYSTEM REPORTS?

`calibrate_noise.py` measures `cos(g_t, g_{t-1})` on a real TB run. That number is
only useful if the bench can be read in the same units, and until now it could
not be: "0.9997 at noise 0.01, 0.29 at noise 2" appears in three files as PROSE
with no code behind it, so it could not be re-derived, extended to a new cell, or
checked. This file is that measurement.

THE STATISTIC IS COMPUTED HERE, NOT READ OFF A CONTROLLER, and that is
deliberate. `_hyper_tick` computes the same cosine but from inside a live servo,
where the rate is moving; the axis wants cos at a FIXED operating point.
Computing it the same way `calibrate_noise.py` does -- consecutive gradients of
the same branch, before any rescale -- is what makes the two tables comparable.

TWO NUMBERS, NOT ONE. Median cos alone does not locate the real system on this
axis, because the same median means different things at different widths:

    null |cos| for independent vectors = sqrt(2 / (pi * d))

At the bench's usual d=32 that null is 0.141, so a median of 0.29 is 2.1x chance
and a SINGLE reading is nearly worthless. At the real policy's d=6,163,969 the
null is 0.00032, so the same 0.29 is ~900x chance and one reading is essentially
noiseless. Matching only the median would put the real system in a cell where the
statistic is far noisier than it really is -- which flatters every blind arm and
penalises every measuring one. So the report prints median, IQR, null and
cos/null together, and a cell "matches" only on the pair.

COS DECAYS AS A RUN CONVERGES, so a full-run median pools regimes that a real
400-step window does not. Reported per QUARTILE of the run for that reason: the
first crucible-era attempt at this measurement read a converged model and got
cos ~ 0, which is the correct reading at a stationary point and not a noise
verdict.

    python -m bench.cos_axis            # the noise x dim grid
    python -m bench.cos_axis lr         # cos vs lr/oracle, one cell
"""
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.crucible import MLE, _mk
from bench.oracle import find_oracle

#: (label, game, kwargs, steps, lr_grid, extra_args) -- `_mk`'s cell shape, so
#: these are the crucible's own surfaces and not a second definition of them.
GRID = [
    (f'mle n{n:g} d{d}', 'mle', dict(MLE, noise=n, dim=d), 2000,
     (1e-6, 1e-1, 12), {})
    for d in (32, 2048)
    for n in (0.01, 0.1, 0.5, 2.0, 5.0)
]

#: THE REAL FUSED MEASUREMENT, and its provenance stated exactly, because the
#: first version of this constant cited a regime name rather than a file and the
#: name later resolved to a different model.
#:
#: FOUR windows, elj nehzor sg14 T10, equilibration, `fused`, 400 steps each:
#:      0.2871  phase1_exit, pre-fix        0.2889  r2 running.pt, pre-clip-fix
#:      0.2901  r2 running.pt @11001        0.3037  phase1_exit, POST-fix
#: Median of the four = 0.2895; full spread 0.0166, NOT the 0.003 claimed in
#: F-033 before the fourth existed. The p25/p75 below are from the 0.2901
#: window, which is the only one whose quartiles were recorded.
#:
#: `noise_calib_eq_descent.json` carries `stage: null` -- it predates stage
#: recording, so it is a PRE-fix artifact of the era when `eq_descent` still
#: pointed at a mutable `_running.pt`. The number is a real measurement of a
#: real fused window; only the regime LABEL was unstable. A post-fix eq_descent
#: window does not exist on disk yet.
#: COMPUTED FROM THE FILE, NOT TRANSCRIBED. The hardcoded quartiles were
#: p25=0.24 / p75=0.35 (width 0.11), inherited from an earlier session's
#: write-up. They match no window and no quartile of any artifact on disk: the
#: cited file gives p25=0.1998 / p75=0.3859, width **0.186 -- 1.69x wider**. The
#: median (0.2901) was correct, which is what let the error survive.
#:
#: Everything that leaned on 0.11 shifts: "the real signal is 4x tighter than
#: the bench" becomes ~2.4x, and the `bwd` comparison "13x wider" becomes 7.6x.
#:
#: AND THE IQR IS NOT THE CONTROLLER'S SNR. `med/null` is a LOCATION null --
#: E|cos| for independent isotropic vectors. It says the median is not zero; it
#: says nothing about one reading. The number a controller consuming one cos per
#: step actually sees is median/sd = 0.2901/0.1466 = **1.98**, not 903.
def _real_from_disk(path='noise_calib_eq_descent.json'):
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     path)
    if not os.path.exists(p):
        return None
    import json
    with open(p) as f:
        c = [x['cos'] for x in json.load(f)]
    if not c:
        return None
    a = np.asarray(c, dtype=float)
    return dict(label='REAL fused (from disk)', med=float(np.median(a)),
                p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
                sd=float(a.std()), n=len(a), dim=6_163_969)


REAL = _real_from_disk() or dict(
    label='REAL fused (file missing)', med=0.2901, p25=0.1998, p75=0.3859,
    sd=0.1466, n=399, dim=6_163_969)


def null_cos(d):
    """E|cos| between independent vectors in d dimensions."""
    return math.sqrt(2.0 / (math.pi * max(int(d), 2)))


#: The real measurement's window length. Every comparison against it has to use
#: the same window, for the reason in `measure`'s docstring.
REAL_WINDOW = 400


def measure(cell, lr=None, seed=0, seeds=(0, 1, 2), win=REAL_WINDOW):
    """
    Median cos at a fixed rate, over the whole run AND in windows of `win`.

    THE WINDOWED NUMBERS ARE THE ONES THAT COMPARE TO THE REAL SYSTEM, and the
    whole-run ones do not. `calibrate_noise` reads 400 consecutive steps of a
    long run -- a snapshot at one operating point. A bench run sweeps from
    cos ~ 1.0 (far from the optimum) to cos ~ -0.16 (in the noise ball) in 2000
    steps, so its whole-run IQR is dominated by that DRIFT, not by per-step
    scatter.

    Quantitatively: quoting the whole-run IQR made the bench look ~10x noisier
    per reading than the real system, which is the number that made a higher
    `hyper_beta` look safe on the real system. Windowing is the like-for-like
    comparison, and it is the one the beta question rests on.

    `lr=None` uses the oracle rate -- the operating point a working controller is
    supposed to find, and the one the crucible scores against.
    """
    surface = _mk(cell)
    if lr is None:
        lr = find_oracle(surface, seeds=seeds, verbose=False).lr
    import torch
    run = surface.make(float(lr), seed=seed, servo=False)
    prev, rows = None, []
    for _ in range(surface.steps):
        run.step()
        gs = [p.grad.detach().reshape(-1) for p in run.game.policy_params
              if p.grad is not None]
        if not gs:
            continue
        g = torch.cat(gs).float()
        n = float(g.norm())
        if prev is not None and n > 0:
            pn = float(prev.norm())
            if pn > 0:
                c = float(torch.dot(g, prev)) / (n * pn)
                if math.isfinite(c):
                    rows.append(c)
        prev = g.clone() if n > 0 else prev
    if len(rows) < 8:
        return None
    d = int(sum(p.numel() for p in run.game.policy_params))
    q = len(rows) // 4
    # per-window (median, IQR) -- the like-for-like statistic
    wins = []
    for i in range(0, len(rows) - win + 1, win):
        w = rows[i:i + win]
        wins.append(dict(at=i,
                         med=float(np.median(w)),
                         iqr=float(np.percentile(w, 75) - np.percentile(w, 25))))
    return dict(label=cell[0], lr=float(lr), dim=d, n=len(rows),
                med=st.median(rows),
                p25=float(np.percentile(rows, 25)),
                p75=float(np.percentile(rows, 75)),
                quartiles=[st.median(rows[i * q:(i + 1) * q]) for i in range(4)],
                windows=wins)


def _task(cell):
    try:
        return measure(cell)
    except Exception as e:                       # one cell must not hide the rest
        return dict(label=cell[0], error=f'{type(e).__name__}: {e}')


def _header():
    print(f'  {"cell":<16} {"lr":>9} {"dim":>8} {"median":>8} {"IQR":>15} '
          f'{"null":>8} {"x null":>7}   {"cos by quartile of run":<32}')


def _row(r):
    if r is None or 'error' in (r or {}):
        label = (r or {}).get('label', '?')
        print(f'  {label:<16} {(r or {}).get("error", "no samples")}')
        return
    nl = null_cos(r['dim'])
    qs = ' '.join(f'{x:>6.3f}' for x in r['quartiles'])
    iqr = f'{r["p25"]:.3f}-{r["p75"]:.3f}'
    print(f'  {r["label"]:<16} {r["lr"]:>9.3g} {r["dim"]:>8,} {r["med"]:>8.4f} '
          f'{iqr:>15} {nl:>8.4f} {r["med"] / nl:>7.1f}   {qs:<32}')


def grid(workers=6):
    print(f'{"=" * 118}\nCOS vs NOISE x DIM, at each cell\'s ORACLE rate\n{"=" * 118}')
    _header()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_task, GRID))
    for r in rows:
        _row(r)
    nl = null_cos(REAL['dim'])
    iqr = f'{REAL["p25"]:.3f}-{REAL["p75"]:.3f}'
    print(f'\n  {REAL["label"]:<16} {"--":>9} {REAL["dim"]:>8,} {REAL["med"]:>8.4f} '
          f'{iqr:>15} {nl:>8.4f} {REAL["med"] / nl:>7.1f}')
    sd = REAL.get('sd')
    if sd:
        print(f'\n  x null is a LOCATION null (E|cos| for independent vectors). It says '
              f'the median\n  is not zero -- it says NOTHING about one reading. Per-decision '
              f'SNR, which is\n  what a controller consuming one cos per step sees:  '
              f'median/sd = {REAL["med"] / sd:.2f}')
    return rows


def lr_sweep(mult=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0), noise=0.5, dim=32):
    """
    COS ALSO MOVES WITH THE RATE -- which is the entire hypergradient premise, and
    the reason one real cos value cannot by itself pin the real system's noise.

    Printed so the ambiguity is visible: a median of 0.29 is consistent with a
    quiet surface run hot and with a noisy surface run correctly, and only the
    IQR/null pair separates them.
    """
    cell = (f'mle n{noise:g} d{dim}', 'mle', dict(MLE, noise=noise, dim=dim),
            2000, (1e-6, 1e-1, 12), {})
    base = find_oracle(_mk(cell), seeds=(0, 1, 2), verbose=False).lr
    print(f'{"=" * 118}\nCOS vs RATE -- {cell[0]}, oracle lr {base:.3g}\n{"=" * 118}')
    _header()
    for m in mult:
        r = measure(cell, lr=base * m)
        if r:
            r['label'] = f'{m:g}x oracle'
        _row(r)


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    if 'lr' in sys.argv[1:]:
        lr_sweep()
    else:
        grid()
