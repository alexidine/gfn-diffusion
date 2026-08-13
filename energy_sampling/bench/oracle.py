"""
The oracle: the best FIXED learning rate a surface admits, found by brute force.

WHY THIS IS THE LOAD-BEARING PIECE. Every controller result so far has been
descriptive -- it ramped, it cut, it diverged. None of it says how much the
controller COST, because there was nothing to be worse than. An oracle converts
the whole battery into one number:

    regret = what the controller achieved / what the best fixed LR achieved

and it simultaneously defines "recovered to healthy", which otherwise has no
meaning: healthy at step t is `dist(t) <= tol * dist_oracle(t)`, measured
against the oracle's own trace rather than against a guess.

This is only possible because the surface is synthetic. On a crystal there is no
oracle and the controller can only be compared with itself.

WHAT "FIXED" MEANS HERE. The servo is put in its own shipping control arm --
`lr_servo_managed=()`, which makes LRController read and log while actuating
nothing -- and `lr_warmup_ratio=1`, which makes the envelope identically 1.0.
Both are real config settings, so the oracle arm is a configuration the trainer
would accept, not a bypass.

THE SWEEP IS CHECKED, NOT TRUSTED. `find_oracle` asserts the winner is INTERIOR
to the bracket. A minimum at an edge means the bracket was wrong and the number
is not an oracle -- it is the best of a badly chosen set, which is exactly the
kind of quietly-wrong baseline that would make every regret figure downstream
meaningless.
"""

import math

import numpy as np

from bench.harness import BenchRun
from bench.surfaces import GAMES

#: game train_key -> the args key holding that optimizer's base LR
#: (mirrors LRController._POLICY_BASE)
LR_KEY = {'fwd': 'lr_policy', 'bwd': 'lr_back', 'replay': 'lr_replay',
          'fused': 'lr_fused'}


class Surface:
    """
    One stage analogue: a game plus the settings the battery runs it under.

    Stage transitions are a HARD RESET by design -- `rearm_warmup` puts
    peak_scale back to 1.0 and forgets the ceiling at every transition -- so each
    stage can be studied independently from cold start, and a surface is exactly
    that. No chaining required, which is what makes the battery cheap.
    """

    def __init__(self, name, game, game_kwargs, steps=3000, lr_grid=None,
                 probe_batch=2048, extra_args=None):
        self.name = name
        self.game = game
        self.game_kwargs = dict(game_kwargs)
        self.steps = int(steps)
        self.lr_grid = lr_grid or (1e-6, 1e1, 15)     # (lo, hi, n) log-spaced
        self.probe_batch = int(probe_batch)
        self.extra_args = dict(extra_args or {})

    def lr_key(self, train_key):
        return LR_KEY[train_key]

    # -------------------------------------------------------------- run arms

    def _args(self, lr, train_key, servo, seed):
        args = {
            self.lr_key(train_key): lr,
            'min_lr': 1e-12,                 # the floor must not truncate the sweep
            **self.extra_args,
        }
        if servo:
            args.update({'adaptive_lr.warmup_steps': 50,
                         'ray_calibration.period': 50,
                         'ray_calibration.enabled': True})
        else:
            # the shipping control arm: reads and logs, actuates nothing, and the
            # envelope is identically 1.0 so the rate really is fixed
            args.update({'lr_servo_managed': (),
                         'lr_warmup_ratio': 1,
                         'ray_calibration.enabled': False})
        return args

    def make(self, lr, seed=0, servo=False, sensor=None,
             climber=None, braker=None, standard=None):
        """An UNSTARTED run at base LR `lr`. Scenarios drive it in segments so
        they can perturb state mid-run."""
        train_key = GAMES[self.game].train_key      # class attribute, no instance needed
        return BenchRun(
            game=self.game, need_batch_sizer=False, seed=seed,
            game_kwargs=dict(self.game_kwargs, lr=lr, seed=seed),
            probe_batch=self.probe_batch,
            args_overrides=self._args(lr, train_key, servo, seed),
            standard=standard,
            **({'sensor': 'none'} if not servo else
               {'sensor': sensor} if sensor else
               {'climber': climber, 'braker': braker}))

    def run(self, lr, seed=0, servo=False, steps=None, sensor=None,
            climber=None, braker=None, standard=None):
        """One run at base LR `lr`. `servo=False` is the fixed-rate oracle arm."""
        return self.make(lr, seed, servo, sensor, climber, braker,
                         standard).run(
            steps or self.steps, stop_on_divergence=False)


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------

def distance_trace(run):
    """Per-step ground-truth distance to the optimum, non-finite -> inf."""
    return np.array([h['dist'] if math.isfinite(h['dist']) else math.inf
                     for h in run.history], dtype=float)


def median_trace(runs):
    """Elementwise median across seeds, truncated to the shortest run."""
    n = min(len(r.history) for r in runs)
    stack = np.stack([distance_trace(r)[:n] for r in runs])
    return np.median(stack, axis=0)


def final_distance(runs, tail=50):
    """Median over seeds of the median distance over each run's last `tail` steps.

    A tail median rather than the last point: at a noise floor the final sample is
    a draw from the stationary distribution, and comparing single draws across
    arms would rank noise.
    """
    per_run = []
    for r in runs:
        t = distance_trace(r)[-tail:]
        finite = t[np.isfinite(t)]
        per_run.append(float(np.median(finite)) if len(finite) else math.inf)
    return float(np.median(per_run))


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

class OracleResult:
    def __init__(self, surface, lr, curve, trace, final, seeds):
        self.surface = surface
        self.lr = lr
        self.curve = curve          # {lr: final distance}
        self.trace = trace          # median distance per step at the oracle LR
        self.final = final          # final distance at the oracle LR
        self.seeds = seeds

    @property
    def cliff(self):
        """
        Lowest swept rate at which the surface actually blows up -- the top of the
        usable band.

        Needed because "hot but not catastrophic" is a band, not a multiple, and
        on an anisotropic surface it can be very narrow: on `mle` the oracle is
        4.33e-3 against a cliff at 7.3e-3, so the whole stable-but-hot region is
        1.7x wide and a scenario written as "2x the oracle" is testing
        catastrophe instead. Returns None if nothing in the sweep blew up.
        """
        above = [lr for lr in self.curve if lr > self.lr
                 and (not math.isfinite(self.curve[lr])
                      or self.curve[lr] > 1e3 * max(self.final, 1e-12))]
        return min(above) if above else None

    def hot_lr(self, frac):
        """
        A rate `frac` of the way from the oracle to the cliff, in log space.
        frac 0 = the oracle, 1 = the cliff. Falls back to a plain multiple when
        the sweep found no cliff.
        """
        cliff = self.cliff
        if cliff is None:
            return self.lr * (1.0 + 4.0 * frac)
        return self.lr * (cliff / self.lr) ** float(frac)

    def __repr__(self):
        return (f'<Oracle {self.surface} lr={self.lr:.4g} '
                f'final={self.final:.4g} over {len(self.curve)} rates>')

    def healthy_at(self, step, tol):
        """The distance a healthy run should be at or below at this step."""
        i = min(int(step), len(self.trace) - 1)
        return self.trace[i] * tol


def sweep(surface, lrs, seeds=(0, 1, 2)):
    """Fixed-LR runs across a grid. Returns {lr: final distance}."""
    curve = {}
    for lr in lrs:
        runs = [surface.run(lr, seed=s, servo=False) for s in seeds]
        curve[lr] = final_distance(runs)
    return curve


def find_oracle(surface, seeds=(0, 1, 2), refine=True, verbose=False):
    """
    Brute-force the best fixed LR, then CHECK it.

    Two checks, because a baseline nobody verified is worse than no baseline:

      1. the winner must be INTERIOR to the bracket -- a minimum at an edge means
         the bracket was wrong;
      2. the curve must actually have a minimum worth having -- the best rate must
         beat both edges by a real margin, or the surface is LR-insensitive over
         this range and "regret vs oracle" would be measuring nothing.
    """
    lo, hi, n = surface.lr_grid
    grid = list(np.logspace(math.log10(lo), math.log10(hi), n))
    curve = sweep(surface, grid, seeds)

    best = min(curve, key=lambda k: curve[k])
    idx = grid.index(best)
    if idx in (0, len(grid) - 1):
        raise ValueError(
            f'{surface.name}: the best fixed LR ({best:.3g}) is at the EDGE of the '
            f'bracket [{lo:.3g}, {hi:.3g}]. Widen Surface.lr_grid -- this is not an '
            f'oracle, it is the best of a badly chosen set.')

    edge = min(curve[grid[0]], curve[grid[-1]])
    if not (curve[best] < 0.5 * edge):
        raise ValueError(
            f'{surface.name}: best {curve[best]:.4g} vs bracket edges {edge:.4g} -- '
            f'the surface is barely LR-sensitive over this range, so regret against '
            f'it would measure nothing. Lengthen the run or widen the bracket.')

    if refine:
        # one bisection pass in log space either side of the winner
        span = math.log10(grid[1]) - math.log10(grid[0])
        extra = [10 ** (math.log10(best) + d * span / 2) for d in (-1, 1)]
        curve.update(sweep(surface, extra, seeds))
        best = min(curve, key=lambda k: curve[k])

    runs = [surface.run(best, seed=s, servo=False) for s in seeds]
    result = OracleResult(surface.name, best, curve, median_trace(runs),
                          final_distance(runs), seeds)
    if verbose:
        print(f'  oracle {surface.name}: lr={best:.4g}, final={result.final:.4g}')
        for lr in sorted(curve):
            mark = ' <-' if lr == best else ''
            print(f'    {lr:>12.4g}  {curve[lr]:.4g}{mark}')
    return result
