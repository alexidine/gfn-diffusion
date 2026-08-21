"""
Replay Racing simulator -- gate 6.1 of `docs/design/lr_probe_protocol.md`.

WHY THIS EXISTS AND WHY IT LOOKS LIKE THIS. The two LR controllers this design
replaces both failed in their decision arithmetic, and both failures took tens
of thousands of GPU steps to become visible. The decision layer (`lr_race.py`)
is a pure function, so it can be hammered here instead -- on CPU, in seconds,
thousands of races at a time, against a truth that is known by construction.

IT SIMULATES THE MEASUREMENT, NOT THE SUMMARY. A simulator that hands the rule
tidy (effect, standard-error) pairs validates a model in which none of the
estimator's real hazards exist, and would certify a rule that cannot survive
contact with the trainer. So this generates per-unit, per-replicate HELD-OUT
LOSSES with the full variance structure -- a large shared component that pairing
must cancel, an arm x replicate training-path component, and an arm x unit
component that can be heavy-tailed and skewed -- hands them to the shipping
`decide()` unchanged, and lets the rule compute its own statistic.

THE FOUR COMPONENTS, and what each is there to break:

  shared[j][i]   the harvested batch's own difficulty and the condition's loss
                 level: hundreds of nats, common to every arm. It cancels in
                 the paired contrast, and a rule that lets it through would be
                 measuring batch difficulty instead of the learning rate.
  interact[a][j] the training path: an arm's luck on one replicate's sub-larder
                 and order. This is the component a same-larder confirmation
                 re-run does NOT resample, which is why replicates here use
                 disjoint sub-larders.
  eps[a][j][i]   per-unit noise, optionally Student-t (heavy) and/or skewed.
                 Skew is the one that matters: the sign test targets the MEDIAN,
                 so noise whose median and mean disagree moves the estimand.
  scale          a multiplier drifting across probes. Everything is
                 proportional to it, so a scale-invariant rule must be immune;
                 anything denominated in nats is not.

THE TRUTH. `quality(e) = -C_side * e^2` in `e = log10(lr / lr*)`, with separate
curvature above and below the optimum, because a loss curve that is steeper hot
than cold is what turns a fixed-size margin into a cold-biased hold band. Set
`curvature = 0` for the pure-noise cell, where every move is by definition a
false move and the drift bound is the whole test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from lr_race import (ENTRY_ARMS, FINE_ARMS, INCUMBENT, ArmScores, RaceConfig,
                     RaceRecord, decide, min_favoring, shifted_bracket)


@dataclass(frozen=True)
class SimSpec:
    """One cell: a truth, a noise structure, and a hazard."""
    #: Curvature of the loss in log-LR, above / below the optimum. The ratio is
    #: what matters; {1,3,10} are the required sweep in gate 6.1.
    curvature_hot: float = 1.0
    curvature_cold: float = 1.0
    #: Half-width of a flat PLATEAU around the optimum, in dex. Inside it every
    #: rate is genuinely equivalent and no rule can or should resolve anything;
    #: outside, curvature resumes. This is the honest version of the "flat
    #: surface" cell: a rule that random-walks on a globally flat surface is
    #: harmless by construction, but one that walks OUT of a plateau into the
    #: region where the rate does matter is not, and only this shape can tell
    #: the two apart.
    flat_halfwidth: float = 0.0
    #: Total progress an on-optimum window makes, in loss units.
    gain: float = 1.0
    #: sd of the arm x replicate (training path) component.
    sd_replicate: float = 0.25
    #: sd of the arm x unit component.
    sd_unit: float = 0.5
    #: sd of the shared component. Large on purpose: it is the thing pairing
    #: exists to remove, and a rule that leaks it will show up here.
    sd_shared: float = 50.0
    #: Student-t degrees of freedom for the unit component. None = Gaussian.
    unit_df: int | None = 5
    #: Skew of the unit component (0 = symmetric). Applied as a two-piece
    #: scaling, then re-centred on the MEAN -- which leaves the MEDIAN offset,
    #: exactly the case a symmetric-noise simulator cannot produce.
    unit_skew: float = 0.0
    #: How the skew varies with an arm's log-error. THIS IS THE ADVERSARIAL
    #: CELL FOR THE SIGN TEST. Measured here: with equal skew on both arms the
    #: paired difference keeps median 0 (skew cancels, P(favour) = 0.4996 over
    #: 200k draws), but with skew 0 against skew 1.5 the paired difference has
    #: mean -0.0009 and median +0.3522, i.e. P(favour) = 0.589. The sign test
    #: targets the MEDIAN, so an LR-dependent skew shifts its estimand while
    #: leaving the mean advantage at zero. `skew_slope` makes the skew a
    #: function of e so the drift bound is tested against it.
    skew_slope: float = 0.0
    #: Extra early progress for hot arms, vanishing by end of window: the
    #: sprint-then-fade shape the half-window check exists to reject.
    transient_hot: float = 0.0
    #: Log-error beyond which an arm can die, and the per-window hazard at the
    #: top of the tested range.
    danger_e: float = 0.6
    hazard_max: float = 0.0
    #: Multiplicative drift of the loss scale per probe (dex). Tests scale
    #: invariance of the decision rule.
    scale_drift: float = 0.0


DEFAULT_SPEC = SimSpec()


def _student_t(rng, df, size):
    return rng.standard_t(df, size=size) / math.sqrt(df / (df - 2.0)) if df and df > 2 else rng.standard_normal(size)


def _skew(x, skew):
    """Two-piece scaling, re-centred on the mean.

    Leaves E[x] = 0 while the MEDIAN moves -- so an estimator that targets the
    median (the sign test does) sees a shift that an estimator targeting the
    mean does not. That divergence is the point: it is the assumption failure
    a symmetric simulator is blind to.
    """
    if not skew:
        return x
    up, dn = 1.0 + skew, 1.0 / (1.0 + skew)
    y = np.where(x >= 0, x * up, x * dn)
    return y - y.mean()


class RaceSim:
    """Generates `RaceRecord`s from a known truth and runs the policy loop."""

    def __init__(self, spec: SimSpec = DEFAULT_SPEC, cfg: RaceConfig = RaceConfig(),
                 n_units: int = 10, seed: int = 0):
        self.spec = spec
        self.cfg = cfg
        self.n_units = int(n_units)
        self.rng = np.random.default_rng(seed)
        self.probe_index = 0

    # ------------------------------------------------------------ the truth

    def quality(self, e: float) -> float:
        """Progress multiplier at log-error `e`; 1.0 at the optimum."""
        s = self.spec
        dead = max(0.0, abs(e) - s.flat_halfwidth)
        if dead == 0.0:
            return 1.0
        c = s.curvature_hot if e > 0 else s.curvature_cold
        return 1.0 - c * dead * dead

    def hazard(self, e: float) -> float:
        s = self.spec
        if s.hazard_max <= 0 or e <= s.danger_e:
            return 0.0
        return min(1.0, s.hazard_max * (e - s.danger_e) / max(s.danger_e, 1e-9))

    # ------------------------------------------------------- one race record

    def race(self, e_incumbent: float, arms, kind: str = 'fine',
             replicates: int | None = None, expansions_used: int = 0,
             expect_bitwise: bool = False) -> RaceRecord:
        """Simulate every window of one race and return what the trainer would log.

        `e_incumbent` is the incumbent's log10(lr/lr*); an arm at multiplier `m`
        therefore sits at `e_incumbent + log10(m)`.
        """
        s = self.spec
        r = replicates or (self.cfg.screen_replicates if kind == 'screen'
                           else self.cfg.replicates)
        n_u = self.n_units
        scale = 10.0 ** (s.scale_drift * self.probe_index)

        # Shared across arms: batch difficulty. Drawn once per (replicate, unit)
        # because every arm trains on the same sub-larder and is scored on the
        # same held-out slice.
        shared = self.rng.normal(0.0, s.sd_shared, size=(r, n_u))

        out = []
        for m in arms:
            e = e_incumbent + math.log10(m)
            q = self.quality(e)
            interact = self.rng.normal(0.0, s.sd_replicate, size=(r, 1))
            skew = s.unit_skew + s.skew_slope * e
            eps_mid = _skew(_student_t(self.rng, s.unit_df, (r, n_u)), skew) * s.sd_unit
            eps_end = _skew(_student_t(self.rng, s.unit_df, (r, n_u)), skew) * s.sd_unit

            p_end = s.gain * q
            # Half the progress by mid-window, plus any transient a hot arm buys
            # and then gives back -- the shape the half-window check must reject.
            p_mid = 0.5 * s.gain * q + s.transient_hot * max(0.0, e)

            died = bool(self.rng.random() < self.hazard(e))
            mid = scale * (-p_mid + interact + eps_mid) + shared
            end = scale * (-p_end + interact + eps_end) + shared
            out.append(ArmScores(multiplier=float(m),
                                 mid=tuple(tuple(float(v) for v in row) for row in mid),
                                 end=tuple(tuple(float(v) for v in row) for row in end),
                                 died=died))

        return RaceRecord(arms=tuple(out), kind=kind, expansions_used=expansions_used,
                          expect_bitwise=expect_bitwise,
                          duplicate_spread=0.0 if expect_bitwise else None)

    # ---------------------------------------------------------- the policy

    def entry_event(self, e: float):
        """Screen -> (expansions) -> confirm. Returns (new_e, n_windows, trail)."""
        arms = ENTRY_ARMS
        used = 0
        windows = 0
        trail = []
        while True:
            rec = self.race(e, arms, kind='screen', expansions_used=used)
            windows += len(arms) * self.cfg.screen_replicates
            d = decide(rec, self.cfg)
            trail.append(d.action)
            if d.action in ('expand_up', 'expand_down'):
                used += 1
                shifted = shifted_bracket(arms, 'up' if d.action == 'expand_up' else 'down')
                # The incumbent stays in every screen: it is the reference every
                # contrast is formed against, and one extra window is cheap
                # against losing the ability to test at all.
                arms = tuple(sorted(set(shifted) | {INCUMBENT}))
                continue
            if d.action != 'candidate':
                return e, windows, trail
            break

        conf = self.race(e, (INCUMBENT, d.multiplier), kind='confirm')
        windows += 2 * self.cfg.replicates
        dc = decide(conf, self.cfg)
        trail.append(dc.action)
        if dc.moves:
            return e + math.log10(dc.multiplier), windows, trail
        return e, windows, trail

    def fine_probe(self, e: float):
        rec = self.race(e, FINE_ARMS, kind='fine')
        d = decide(rec, self.cfg)
        if d.moves:
            return e + math.log10(d.multiplier), len(FINE_ARMS) * self.cfg.replicates, d
        return e, len(FINE_ARMS) * self.cfg.replicates, d

    def run(self, n_probes: int, e0: float = 0.0, entry_first: bool = True):
        """The full policy: entry event, then fine probes, with the
        two-consecutive-same-direction-moves -> entry-bracket rule.

        Returns a dict of the trace and its summary statistics.
        """
        e = float(e0)
        trace = [e]
        moves = []
        windows = 0
        last_dir = 0          # sign of the previous move; 0 = the last probe held
        want_entry = entry_first
        for _ in range(n_probes):
            self.probe_index += 1
            before = e
            if want_entry:
                e, w, _ = self.entry_event(e)
                want_entry = False
                last_dir = 0
            else:
                e, w, _ = self.fine_probe(e)
            windows += w
            step = e - before
            moves.append(step)
            trace.append(e)
            direction = 0 if step == 0.0 else (1 if step > 0 else -1)
            # Two consecutive moves the SAME way mean the fine bracket is
            # chasing rather than centring -- the incumbent is far enough out
            # that a rung at a time will not catch it. Widen once. This replaces
            # the escalation ladder of the previous revision: no per-regime
            # state, no step-size schedule, and a hold resets it.
            if direction != 0 and direction == last_dir:
                want_entry = True
                last_dir = 0
            else:
                last_dir = direction
        arr = np.asarray(moves)
        n = len(arr)
        drift = float(arr.mean()) if n else 0.0
        se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else float('inf')
        return {
            'trace': np.asarray(trace),
            'moves': arr,
            'n_probes': n,
            'drift_per_probe': drift,
            'drift_se': se,
            'drift_ci': (drift - 1.96 * se, drift + 1.96 * se),
            'move_rate': float((arr != 0).mean()) if n else 0.0,
            'final_e': e,
            'mean_e': float(np.mean(trace)),
            'sd_e': float(np.std(trace)),
            'p_far': float(np.mean(np.abs(np.asarray(trace)) > 0.6)),
            'windows': windows,
        }


# --------------------------------------------------------------- the v1 rule

def decide_v1_highest_competitive(record: RaceRecord, cfg: RaceConfig = RaceConfig()):
    """The REJECTED rule, kept runnable so the gate can prove it fails.

    "Select the highest stable LR statistically competitive with the best
    candidate" -- i.e. promote whenever the top arm is not significantly WORSE
    than the best arm. This is the rule the v1 proposal specified and the reason
    this whole document exists; a test suite that cannot see its drift cannot
    certify the rule that replaced it (the house rule: re-introduce the bug and
    require a failure).
    """
    from lr_race import Decision, _evaluate, _unit_advantage  # control, not shipped logic

    inc = record.arm(INCUMBENT)
    live = [a for a in record.arms if not a.died]
    if inc is None or not live:
        return Decision('hold', reason='no_arms')

    # "Best" = highest mean advantage over the incumbent.
    adv = {a.multiplier: (sum(_unit_advantage(a, inc)) / max(len(_unit_advantage(a, inc)), 1))
           for a in live}
    best = max(live, key=lambda a: adv[a.multiplier])

    # COMPETITIVE = the best arm does not SIGNIFICANTLY beat it. This is the
    # v1 rule read literally, and the literal reading is the disease: on a flat
    # or noisy surface nothing is significantly worse than anything, so every
    # arm is competitive and the rule takes the top rung every single time.
    competitive = []
    for a in live:
        if a.multiplier == best.multiplier:
            competitive.append(a.multiplier)
            continue
        beaten = _evaluate(best, a, cfg.alpha, require_all_replicates=False)
        if not beaten['passed']:
            competitive.append(a.multiplier)
    if not competitive:
        return Decision('hold', reason='none_competitive')
    return Decision('move', multiplier=max(competitive), reason='v1_highest_competitive')


def run_v1(n_probes: int, spec: SimSpec = DEFAULT_SPEC, cfg: RaceConfig = RaceConfig(),
           n_units: int = 10, seed: int = 0, e0: float = 0.0):
    """Policy loop for the v1 rule, fine bracket only -- the drift control."""
    sim = RaceSim(spec, cfg, n_units, seed)
    e = float(e0)
    moves, trace = [], [e]
    for _ in range(n_probes):
        sim.probe_index += 1
        rec = sim.race(e, FINE_ARMS, kind='fine')
        d = decide_v1_highest_competitive(rec, cfg)
        step = math.log10(d.multiplier) if d.moves else 0.0
        e += step
        moves.append(step)
        trace.append(e)
    arr = np.asarray(moves)
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float('inf')
    # `mean_e` is the criterion that convicts this rule. Its DRIFT RATE, averaged
    # over a long run, is small once it has parked -- so a gate that tested drift
    # alone would certify it. The stationary offset is what sees the disease.
    return {'drift_per_probe': float(arr.mean()), 'drift_se': se, 'final_e': e,
            'move_rate': float((arr != 0).mean()),
            'mean_e': float(np.mean(trace)), 'sd_e': float(np.std(trace)),
            'trace': np.asarray(trace)}


# ------------------------------------------------------------------- report

def main():
    """`python -m bench.race_sim` -- the cells at a glance."""
    cells = [
        ('flat (no curvature, pure noise)', SimSpec(curvature_hot=0.0, curvature_cold=0.0)),
        ('plateau +-0.3 dex, curved outside', SimSpec(flat_halfwidth=0.3)),
        ('symmetric curvature', SimSpec()),
        ('asymmetric C+/C- = 3', SimSpec(curvature_hot=3.0)),
        ('asymmetric C+/C- = 10', SimSpec(curvature_hot=10.0)),
        ('skewed unit noise', SimSpec(unit_skew=0.8)),
        ('LR-DEPENDENT skew (adversarial)', SimSpec(skew_slope=1.5)),
        ('scale drift 0.05 dex/probe', SimSpec(scale_drift=0.05)),
        ('hazard 0.3 above 0.6 dex', SimSpec(hazard_max=0.3)),
        ('sprinter transient', SimSpec(transient_hot=0.3)),
    ]
    n = 2000
    print(f'{"cell":34s} {"drift/probe":>12s} {"95% CI":>22s} {"move%":>7s} {"mean e":>8s} {"sd e":>7s}')
    for name, spec in cells:
        r = RaceSim(spec, seed=11).run(n, entry_first=False)
        lo, hi = r['drift_ci']
        print(f'{name:34s} {r["drift_per_probe"]:12.5f} '
              f'[{lo:9.5f},{hi:9.5f}] {100 * r["move_rate"]:6.1f}% '
              f'{r["mean_e"]:8.3f} {r["sd_e"]:7.3f}')
    v1 = run_v1(n, seed=11)
    print(f'\n{"v1 rule (REJECTED, control)":34s} {v1["drift_per_probe"]:12.5f} '
          f'{"":22s} {100 * v1["move_rate"]:6.1f}% {v1["mean_e"]:8.3f} {v1["sd_e"]:7.3f}')
    print('  ^ note its DRIFT RATE is small once parked: the criterion that convicts it '
          'is the\n    stationary offset (mean e), not the drift. A gate testing drift '
          'alone would pass it.')

    print('\nescape from a mis-set seed (one entry event):')
    for name, e0 in [('8x cold', -math.log10(8)), ('8x hot', math.log10(8)),
                     ('800x cold', -math.log10(800))]:
        sim = RaceSim(SimSpec(), seed=7)
        e_after, windows, trail = sim.entry_event(e0)
        print(f'  {name:10s} e {e0:+.3f} -> {e_after:+.3f} '
              f'({windows:3d} windows, {"->".join(trail)})')

    print(snr_requirement())


def snr_requirement(cfg: RaceConfig = RaceConfig()) -> str:
    """THE DELIVERABLE FOR THE GPU PHASE.

    Everything about this rule's power collapses to one number the simulator
    cannot know: the replicate-level signal-to-noise ratio

        SNR = mean(replicate advantage) / sd(replicate advantage)

    for a one-rung contrast. It is set by the window length, the harvest, and
    the surface. So the simulator's job is not to guess it but to state what it
    must be -- and the W-sweep on real data is what has to meet it.
    """
    from math import comb
    from statistics import NormalDist
    nd = NormalDist()
    need = min_favoring(cfg.replicates, cfg.alpha / 2)
    lines = ['',
             'REQUIRED SNR (the number the W-sweep must deliver)',
             f'  test is {need}-of-{cfg.replicates} replicates favouring, '
             f'one-sided at alpha/2 = {cfg.alpha / 2:g}',
             '', f'  {"SNR":>6s} {"power":>8s}   interpretation']
    for snr, note in [(0.0, 'no signal: this is the false-move rate'),
                      (0.5, 'not usable'),
                      (0.8, 'marginal'),
                      (1.2, 'workable'),
                      (2.0, 'decisive')]:
        p = nd.cdf(snr)
        pw = sum(comb(cfg.replicates, k) * p ** k * (1 - p) ** (cfg.replicates - k)
                 for k in range(need, cfg.replicates + 1))
        lines.append(f'  {snr:6.1f} {100 * pw:7.1f}%   {note}')
    lines += ['',
              '  A contrast the probe must resolve therefore needs SNR >= ~1.2.',
              '  Signal grows with the window length while replicate noise does not,',
              '  so W is the lever -- which is exactly what the W-sweep sets.']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
