"""
The pooled optimum estimator -- what replaces `ray`'s per-reading servo.

Section 6B of `docs/design/lr_handoff_2026-08-21.md`. No torch: this is a
decision layer over numbers the sensor already produces, and it is tested as one.

WHAT IS POOLED, AND WHY THAT QUANTITY. A calibration returns alpha* = lr*/lr,
which depends on the rate the run happened to be at. The rate the run SHOULD be
at does not, so pool the implied optimum instead:

    y = log10(peak_scale * alpha*)

`peak_scale` is the controller's own multiplier, and lr = base * peak * envelope,
so peak * alpha* = lr* / (base * envelope) -- constant while base and envelope
are, whatever the controller did in between. Pooling alpha* directly would pool a
quantity the controller moves, so the estimator would be chasing its own tail.

WHY AN ESTIMATOR AND NOT A SERVO. v8 moved on EVERY reading:

    peak_scale <- peak_scale * (alpha_hat / alpha_target) ** eta

with eta 0.25 up and 0.5 down. Both of its measured defects follow from that
form and neither is a tuning problem:

  * the +-17% hold sawtooth was the loop chasing white noise. The within-stage
    optimum is stationary -- the variogram is flat from 500 to 40,000 steps and
    the mean is pinned to +-7% -- so successive readings are noise around a
    constant, and a per-reading servo tracks the noise.
  * the persistent ~15% hot offset was eta_up 0.25 against eta_down 0.5
    RECTIFYING that noise into drift. An asymmetric gain on a symmetric noise
    source is a ratchet.

An estimator over a window still tracks genuine drift -- it just requires
evidence to move. Symmetric in both directions, because the asymmetry was the
mechanism of the offset rather than a defence against anything.

CENSORED READINGS ARE BOUNDS, NOT POINTS. `above_range` says alpha* > lo and
`below_range` says alpha* < hi; the grid never resolved further. v8 used the
bound as a point estimate, which biases toward the grid centre in BOTH
directions -- and the below_range half of that bias is the dangerous one, since
it overstates alpha* and licenses a hotter rate than the evidence supports. Here
a bound enters as a one-sided hinge: it pulls only when the estimate is on the
wrong side of it, and says nothing when the estimate already satisfies it.

    f(m) = sum_i w_i (m - y_i)^2                     resolved readings
         + sum_j w_j max(0, lo_j - m)^2              lower bounds
         + sum_k w_k max(0, m - hi_k)^2              upper bounds

f is convex and piecewise quadratic, so its minimiser is found by bisection on
f'. When only bounds are in the pool the minimiser is an INTERVAL, and the
estimate is the point of that interval nearest the incumbent rate -- "no
evidence, no move", applied to the pooled objective rather than to one reading.
"""

from __future__ import annotations

import math
from collections import deque

#: Readings whose alpha* is a point estimate rather than a bound.
RESOLVED = 'bracketed'

#: Per-reading noise floor, in dex. Measured on this route at 0.20-0.25 dex
#: (handoff section 8e). A FLOOR rather than an estimate: the bracket is
#: discrete, so two consecutive readings landing on the same pair of doubling
#: rungs give an observed variance of exactly zero, and an SE of zero would let
#: any difference at all clear the move gate. Over-confidence is how the
#: predecessor's quorum became unreachable.
NOISE_FLOOR_DEX = 0.22


class OptimumPool:
    """Pooled log-optimum within one regime, with exponential forgetting.

    A regime is (stage, loss composition). `observe` refuses a reading whose
    regime differs from the pool's -- the caller resets on the transition, and
    the refusal is the tripwire for having forgotten to.
    """

    def __init__(self, half_life: float = 8.0, min_readings: int = 3,
                 move_t: float = 2.0, noise_floor_dex: float = NOISE_FLOOR_DEX,
                 capacity: int = 64):
        if half_life <= 0:
            raise ValueError(f'half_life must be positive, got {half_life}')
        self.half_life = float(half_life)
        self.min_readings = max(1, int(min_readings))
        self.move_t = float(move_t)
        self.noise_floor = float(noise_floor_dex)
        self.key = None
        self.rows = deque(maxlen=max(4, int(capacity)))
        self.n_admitted = 0
        self.n_rejected = 0
        self.reject_reason = ''
        self.n_resets = 0

    # ------------------------------------------------------------- the regime

    def reset(self, key) -> None:
        """Start a new pool. Called on a stage change or a composition move.

        Readings from the outgoing regime describe a different objective, and
        the whole point of pooling is that everything in the window is an
        estimate of ONE number.
        """
        if self.rows or self.key is not None:
            self.n_resets += 1
        self.key = key
        self.rows.clear()

    # -------------------------------------------------------------- admission

    def observe(self, reading, peak_scale: float, step: int, key) -> bool:
        """Fold one ray reading in. Returns True if it was admitted.

        `reading` is `RayCalibration.last`: a status plus alpha_star/lo/hi.
        `peak_scale` is the multiplier the MEASURED step was taken at, which is
        what makes the reading comparable with the others.
        """
        if key != self.key:
            return self._reject('regime_changed')
        if not (peak_scale > 0 and math.isfinite(peak_scale)):
            return self._reject('bad_peak_scale')
        status = reading.get('status')
        base = math.log10(peak_scale)
        row = {'step': int(step), 'y': None, 'lo': None, 'hi': None,
               'status': status}
        if status == RESOLVED:
            alpha = reading.get('alpha_star')
            if not (isinstance(alpha, float) and math.isfinite(alpha) and alpha > 0):
                return self._reject('bad_alpha_star')
            row['y'] = base + math.log10(alpha)
        elif status == 'above_range':
            lo = reading.get('lo')
            if not (lo and lo > 0):
                return self._reject('bad_bound')
            row['lo'] = base + math.log10(lo)
        elif status == 'below_range':
            hi = reading.get('hi')
            if not (hi and hi > 0):
                return self._reject('bad_bound')
            row['hi'] = base + math.log10(hi)
        else:
            # unresolved / inconsistent. NOT an error and NOT admitted: a
            # calibration that could not see the answer must not guess it, and
            # must not dilute the ones that could either.
            return self._reject(f'unusable_{status}')
        self.rows.append(row)
        self.n_admitted += 1
        return True

    def _reject(self, reason) -> bool:
        self.n_rejected += 1
        self.reject_reason = reason
        return False

    # -------------------------------------------------------------- the solve

    def _weights(self):
        """Exponential forgetting by reading, newest weight 1.

        By reading rather than by step: calibrations are periodic, so the two
        agree, and a count survives a period change while a step-based decay
        would silently reweight the whole window when `period` moved.
        """
        n = len(self.rows)
        lam = math.log(2.0) / self.half_life
        return [math.exp(-lam * (n - 1 - i)) for i in range(n)]

    def _minimiser_interval(self, w, incumbent=None):
        """[a, b], the set of minimisers of f. a == b whenever f is strictly
        convex there, which any single resolved reading is enough to make.

        NO EARLY-OUT ON THE EDGE SIGN. An earlier version returned the search
        edge as soon as f' was non-negative there, which is true at BOTH edges
        of a flat zero segment -- so a pool holding only upper bounds reported
        its minimiser two decades below the band those bounds actually allow,
        and a satisfied bound read as a demand to cut 25x. The interval is what
        the bounds-only case is FOR; collapsing it was the bug.

        The search range covers the incumbent as well as the readings, because a
        rate outside the readings' span is exactly the case where a bound has
        something to say.
        """
        # CLOSED FORM WHEN NO BOUND BINDS, and it is not just an optimisation.
        # Bisection lands within its tolerance of the true minimiser, so an
        # estimate that should sit EXACTLY on the setpoint can come back a
        # fraction below it -- and the ramp's accept test is a comparison
        # against exactly that setpoint, so a rung sitting precisely at target
        # would be rejected on arithmetic noise. Found that way.
        res = [(r['y'], wi) for r, wi in zip(self.rows, w) if r['y'] is not None]
        if res:
            rw = sum(wi for _, wi in res)
            mean = sum(y * wi for y, wi in res) / rw
            slack = all((r['lo'] is None or mean >= r['lo'])
                        and (r['hi'] is None or mean <= r['hi'])
                        for r in self.rows)
            if slack:
                return mean, mean

        vals = [r['y'] if r['y'] is not None else
                (r['lo'] if r['lo'] is not None else r['hi'])
                for r in self.rows]
        if incumbent is not None:
            vals = vals + [incumbent]
        lo_edge, hi_edge = min(vals) - 2.0, max(vals) + 2.0

        def deriv(m):
            g = 0.0
            for r, wi in zip(self.rows, w):
                if r['y'] is not None:
                    g += wi * (m - r['y'])
                elif r['lo'] is not None:
                    g -= wi * max(0.0, r['lo'] - m)
                else:
                    g += wi * max(0.0, m - r['hi'])
            return g

        # f' is non-decreasing. Its zero SET is [a, b]; the two bisections find
        # the two ends, which coincide unless the set is a flat segment.
        a = _bisect(deriv, lo_edge, hi_edge, upper=False)
        b = _bisect(deriv, lo_edge, hi_edge, upper=True)
        return (a, b) if a <= b else (b, a)

    def estimate(self, incumbent: float | None = None):
        """The pooled log10-optimum and its standard error, or None.

        `incumbent` is log10(peak_scale * alpha_target) -- the value that means
        "the rate is already where the estimator would put it". It only matters
        when the minimiser is an interval, i.e. when nothing in the pool
        resolved and the evidence is a set of bounds: there the estimate is the
        smallest move those bounds require, which may be no move at all.
        """
        if len(self.rows) < self.min_readings:
            return None
        w = self._weights()
        a, b = self._minimiser_interval(w, incumbent=incumbent)
        if incumbent is not None and a <= incumbent <= b:
            m = float(incumbent)
        else:
            m = a if incumbent is None or incumbent < a else b

        resolved = [(r['y'], wi) for r, wi in zip(self.rows, w) if r['y'] is not None]
        sw = sum(w)
        n_eff = (sw * sw) / sum(x * x for x in w) if sw > 0 else 0.0
        if len(resolved) >= 2:
            rw = sum(wi for _, wi in resolved)
            mean = sum(y * wi for y, wi in resolved) / rw
            var = sum(wi * (y - mean) ** 2 for y, wi in resolved) / rw
            # Bessel-like correction on the WEIGHTED count, so a window whose
            # weight is concentrated on one reading cannot report the spread of
            # a full one.
            r_eff = (rw * rw) / sum(wi * wi for _, wi in resolved)
            var *= r_eff / max(r_eff - 1.0, 1e-9)
            sd = math.sqrt(max(var, self.noise_floor ** 2))
        else:
            sd = self.noise_floor
        se = sd / math.sqrt(max(n_eff, 1.0))
        return {'log_opt': m, 'se': se, 'sd': sd, 'n': len(self.rows),
                'n_resolved': len(resolved), 'n_eff': n_eff,
                'interval': (a, b)}

    # --------------------------------------------------------------- the move

    def verdict(self, peak_scale: float, alpha_target: float):
        """What to do about the live rate: a dict, always -- never a bare number.

        `applied` is the multiplier on peak_scale. 1.0 is a hold, and a hold
        carries the same evidence fields a move does, because "the estimator saw
        nothing" and "the estimator was never asked" have to be told apart in the
        log.
        """
        out = {'action': 'hold', 'multiplier': 1.0, 'reason': '',
               'n': len(self.rows)}
        if not (peak_scale > 0 and alpha_target > 0):
            out['reason'] = 'bad_inputs'
            return out
        incumbent = math.log10(peak_scale) + math.log10(alpha_target)
        est = self.estimate(incumbent=incumbent)
        if est is None:
            out['reason'] = f'pool_short_{len(self.rows)}_of_{self.min_readings}'
            return out
        out.update({'log_opt': est['log_opt'], 'se': est['se'], 'sd': est['sd'],
                    'n_eff': est['n_eff'], 'n_resolved': est['n_resolved'],
                    'gap_dex': est['log_opt'] - incumbent})
        # The operating rate is the pooled optimum divided by alpha_target, so
        # in log10 the target peak is log_opt - log(alpha_target).
        gap = est['log_opt'] - incumbent
        bar = self.move_t * est['se']
        out['bar_dex'] = bar
        if abs(gap) <= bar:
            out['reason'] = 'within_pooled_se'
            return out
        out['action'] = 'move'
        out['reason'] = 'pooled_optimum_moved'
        out['multiplier'] = 10.0 ** gap
        return out

    # -------------------------------------------------------------- reporting

    def report(self, peak_scale=None, alpha_target=None) -> dict:
        """EVIDENCE, not a verdict. Without n_eff and the SE beside the gap, a
        hold is indistinguishable from an estimator with no power to do anything
        else -- which is the reading the predecessor's logs could not give."""
        out = {'lrpool/n': float(len(self.rows)),
               'lrpool/admitted': float(self.n_admitted),
               'lrpool/rejected': float(self.n_rejected),
               'lrpool/resets': float(self.n_resets)}
        if peak_scale is None or alpha_target is None:
            return out
        v = self.verdict(peak_scale, alpha_target)
        out['lrpool/action'] = float({'hold': 0, 'move': 1}.get(v['action'], -1))
        out['lrpool/multiplier'] = float(v['multiplier'])
        for k in ('log_opt', 'se', 'sd', 'n_eff', 'n_resolved', 'gap_dex', 'bar_dex'):
            if k in v:
                out[f'lrpool/{k}'] = float(v[k])
        return out


def _bisect(f, lo, hi, upper, tol=1e-10, iters=200):
    """An end of the zero set of a non-decreasing f.

    `upper=False` returns inf{m : f(m) >= 0}, `upper=True` returns
    sup{m : f(m) <= 0}. They coincide unless f has a FLAT zero segment, which is
    exactly the bounds-only case -- and returning that segment rather than an
    arbitrary interior point is what lets `estimate` answer "any rate in this
    band satisfies the evidence, so do not move".
    """
    for _ in range(iters):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if upper:
            lo, hi = (mid, hi) if v <= 0.0 else (lo, mid)
        else:
            lo, hi = (lo, mid) if v >= 0.0 else (mid, hi)
    return 0.5 * (lo + hi)
