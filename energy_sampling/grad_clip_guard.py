"""
The gradient-norm clip as a GUARD: a per-branch, bounded-influence quantile bar.

WHAT THIS IS AN ALTERNATIVE TO. `gradient_norm_clip: auto` (utils.py
`resolve_derived_config`) resolves to `250 * grad_median(T)/6.6e3 * sqrt(W/512)`.
Because the reference T=25 IS the grad-median table's 6.6e3 entry, that
expression algebraically pins

    clip / median_pre_clip_norm = 250/6600 = 0.0379      (at W=512, EVERY T)

so the bar sits ~26x BELOW the median and binds on essentially every step. That
is not a guard, it is NORMALIZED GRADIENT DESCENT -- the optimizer sees the
gradient's direction and nothing about its magnitude. Measured, not inferred:
reading `.grad` after `train_step` returned gave EXACTLY 37.88 for 398 samples
across two step types and three runs (`bench/calibrate_noise.py`).

WHY THAT MATTERS UNDER ADAM, which is what every optimizer here is. Adam is
exactly invariant to a CONSTANT rescale of the gradient -- m -> cm, v -> c^2 v,
so m/sqrt(v) is unchanged -- and Kingma & Ba's trust-region property bounds the
per-coordinate step by the learning rate however large the gradient is. So the
classic "one exploding gradient, one catastrophic step" failure mode that norm
clipping was invented for (Pascanu, Mikolov & Bengio 2013, whose own threshold
heuristic is ~the AVERAGE norm -- i.e. fire on the tail, be inert on the body)
does not exist here. Two things follow, and both are the reason this file
exists:

  * An always-binding clip equalises every step's contribution to Adam's moment
    estimates regardless of that step's size.
  * What a guard actually buys is protection of the SECOND MOMENT. A spike
    enters v with weight (1-beta2)*||g||^2 and decays with half-life
    ln2/(1-beta2) ~ 700 steps at beta2=0.999, so the coordinates it touches
    learn substantially slower for hundreds of steps after it is gone. The
    aug02 arm died at a pre-clip norm 587x the clip
    (`docs/to_do_rebuild.md:231`) with its step bounded twice over -- which is
    what this picture predicts and what "clipping prevents blow-ups" does not.

WHY NOT AN EMA OF THE NORM -- the failure is already in this repo's record.
A floored median once fired on a clip-neutralised grad of 745, and then the
incident's own 1e4 norms lifted that median so it went blind exactly when the
real excursion began. ANY estimator whose
update is driven by the observed MAGNITUDE is contaminated by the excursion it
exists to catch. A mean is doubly wrong: gradient-norm distributions are
right-skewed, so an EMA mean is dominated by the tail it is meant to exclude.

THE UPDATE IS AN INDICATOR, NOT A MAGNITUDE:

    tau <- tau * exp(eta * (1[||g|| > tau] - (1 - p)))

the standard stochastic-approximation quantile estimator (the SGD step on the
pinball loss), written multiplicatively so it is scale-free. Its fixed point is
P(||g|| > tau) = 1 - p, i.e. tau is the p-th quantile. It reads only WHETHER the
norm exceeded the bar, never by how much, so a 587x spike moves tau by exactly
`eta` -- the same as a 1.001x exceedance. Contamination is bounded BY
CONSTRUCTION, rather than by the "freeze while hot" special case of
`docs/to_do_rebuild.md:1367` (SS A8). SS A8 is the right estimator for the wrong
consumer: it lost to alpha* as an LR sensor because it measures gradient
magnitude rather than step size, and gradient magnitude is precisely what a clip
bar is supposed to measure.

The rate limit also settles a requirement that otherwise conflicts. A guard must
follow a LEGITIMATE regime change upward but must NOT follow a spike. No
magnitude-driven estimator separates those; a rate-limited one does it for free,
because a sustained shift pushes every step while a spike pushes once. At
eta=0.01 a genuine 10x shift is absorbed in ln(10)/eta ~ 230 steps and a spike
costs 1%.

ADAPTATION IS ASYMMETRIC IN TIME, BY A FACTOR OF p/(1-p). An exceeding step
moves log tau by +eta*p, a quiet one by -eta*(1-p), so at p=0.99 the bar climbs
100x faster than it falls: ~ln(R)/(eta*p) steps to track a rise of R, but
~ln(R)/(eta*(1-p)) to track the matching fall (46k steps for 100x at
eta=0.01). That is the right sign for a guard -- being slow to tighten fails
toward "clips less", never toward silently becoming a preconditioner again --
but it does mean a stage whose gradients drop sharply would spend a long time
over-permissive, which is one of the reasons `refresh()` exists.

COLD START IS PARAMETRIC, STEADY STATE IS NOT. An empirical p=0.99 quantile needs
O(1/(1-p)) samples per branch before it means anything, which is a long
uncalibrated window. Warmup instead fits a LOGNORMAL to log||g|| -- two moments,
usable in tens of samples -- and seeds tau = exp(mu + z_p*sigma). The tracker
then corrects whatever the lognormal got wrong. Parametric where data is scarce
and variance dominates; non-parametric where it is plentiful and bias does.

ONE TRACKER PER BRANCH. In unfused phase 1, fwd/bwd/replay each reach their own
optimizer step (`train.py:2848-2852`) and therefore their own clip application,
and an MLE gradient and a TB gradient are different distributions. A single
shared bar is set by whichever branch dominates the step mixture, then
systematically clips the heavy branch and sits inert on the light one -- and the
existing telemetry cannot even see that, because `grad_norm_pre_clip` is one
scalar holding whichever branch stepped last.

SCOPE. This governs the policy clip in `train_step` only. The three
`z_calibration` sidecar steps clip `flow_model` alone, against a different and
much smaller parameter set, and keep the static `gradient_norm_clip`. Extending
the guard there is a separate question with its own distribution to characterise;
it is deliberately not smuggled in here.
"""
import math
import statistics
from typing import Dict, Optional

#: The step types that reach `step_loss`. Mirrors LRController.CHANNELS -- same
#: closed set, same source (train.py train_step's `step_type`).
CHANNELS = ('fwd', 'bwd', 'replay', 'fused')


class _Branch:
    """One channel's tracker. `tau is None` means never calibrated."""

    def __init__(self):
        self.tau: Optional[float] = None
        #: tau at the end of the most recent calibration. The saturation cap is
        #: relative to THIS, not to the run's first estimate, so a refresh gives
        #: the new stage a fresh allowance.
        self.baseline: Optional[float] = None
        #: warmup accumulator over log||g||
        self.warming = True
        self.warm_n = 0
        self.warm_sum = 0.0
        self.warm_sumsq = 0.0
        #: drained at every report
        self.n_obs = 0
        self.n_fired = 0
        self.n_saturated = 0
        #: lifetime, never drained -- distinguishes "this branch is quiet" from
        #: "this branch has never run", which look identical from a rate alone
        self.n_total = 0

    def to_dict(self):
        return {'tau': self.tau, 'baseline': self.baseline, 'warming': self.warming,
                'warm_n': self.warm_n, 'warm_sum': self.warm_sum,
                'warm_sumsq': self.warm_sumsq, 'n_total': self.n_total}

    def load(self, d):
        self.tau = d.get('tau')
        self.baseline = d.get('baseline')
        self.warming = bool(d.get('warming', True))
        self.warm_n = int(d.get('warm_n', 0))
        self.warm_sum = float(d.get('warm_sum', 0.0))
        self.warm_sumsq = float(d.get('warm_sumsq', 0.0))
        self.n_total = int(d.get('n_total', 0))


class GradClipGuard:
    """Per-branch adaptive clip bar. See the module docstring.

    Two calls per optimizer step, in this order:

        bar = guard.threshold(step_type)
        pre_clip = clip_grad_norm_(params, bar)
        guard.observe(step_type, pre_clip)

    `observe` recomputes the same bar internally rather than trusting a passed-in
    value; nothing mutates the tracker between the two calls, so the recomputed
    bar is provably the one that was applied, and the caller cannot desynchronise
    them by accident.
    """

    _STATE_VER = 1

    #: every key the `grad_clip_guard:` config block may carry
    _KEYS = ('enabled', 'p', 'eta', 'warmup_steps', 'warmup_clip', 'max_ratio',
             'refresh_on_stage')

    @classmethod
    def from_config(cls, static_clip, cfg):
        """Build from the `grad_clip_guard:` block. Absent block => disabled.

        UNKNOWN KEYS ARE A HARD ERROR. `percentile: 0.90` where `p: 0.90` was
        meant is silently ignored by getattr and the run proceeds at the default
        -- a config that reads as "bar at the 90th percentile" behaving as "bar
        at the 99th", with nothing in the log to say so. Same class of silent
        divergence between written and effective config that the retired-key
        preflight exists to stop, and cheap to close here because the block's
        schema is closed.

        Raised from Modeller.__init__, which runs immediately after
        get_train_args() and before the model, the datasets, or a single energy
        call -- i.e. this is a load-time failure in practice, not a first-use one.
        """
        if cfg is None:
            return cls(static_clip=static_clip, enabled=False)
        unknown = sorted(k for k in vars(cfg) if not k.startswith('_') and k not in cls._KEYS)
        if unknown:
            raise ValueError(
                f'grad_clip_guard: unknown config key(s) {unknown}. Expected a subset of '
                f'{list(cls._KEYS)}.')
        return cls(static_clip=static_clip,
                   **{k: getattr(cfg, k) for k in cls._KEYS if hasattr(cfg, k)})

    def __init__(self,
                 static_clip: float,
                 enabled: bool = False,
                 p: float = 0.99,
                 eta: float = 0.01,
                 warmup_steps: int = 100,
                 warmup_clip: str = 'static',
                 max_ratio: float = 100.0,
                 refresh_on_stage: bool = True):
        # Validation is at construction, i.e. at load: a clip bar that is wrong
        # is not visible from its outputs until it has already shaped a run.
        if not (0.0 < float(p) < 1.0):
            raise ValueError(f'grad_clip_guard.p must lie in (0, 1), got {p}')
        if float(eta) <= 0.0:
            raise ValueError(f'grad_clip_guard.eta must be > 0, got {eta}')
        if int(warmup_steps) < 30:
            raise ValueError(
                f'grad_clip_guard.warmup_steps = {warmup_steps} is below 30. The warmup '
                f'fits two moments of log||g||; under ~30 samples the sigma estimate is '
                f'noise and the seeded bar is arbitrary.')
        if warmup_clip not in ('static', 'off'):
            raise ValueError(
                f"grad_clip_guard.warmup_clip must be 'static' (keep the configured "
                f"gradient_norm_clip until the bar is calibrated) or 'off' (do not clip "
                f"at all while warming), got {warmup_clip!r}")
        if float(max_ratio) <= 1.0:
            raise ValueError(f'grad_clip_guard.max_ratio must be > 1, got {max_ratio}')
        if enabled and not (math.isfinite(float(static_clip)) and float(static_clip) > 0):
            raise ValueError(
                f'gradient_norm_clip must be a finite positive number for the guard to '
                f'have a warmup fallback, got {static_clip}')

        self.enabled = bool(enabled)
        self.static_clip = float(static_clip)
        self.p = float(p)
        self.eta = float(eta)
        self.warmup_steps = int(warmup_steps)
        self.warmup_clip = warmup_clip
        self.max_ratio = float(max_ratio)
        self.refresh_on_stage = bool(refresh_on_stage)
        #: z_p for the lognormal warmup seed. statistics, not scipy -- this is
        #: the only special function the module needs.
        self._z_p = statistics.NormalDist().inv_cdf(self.p)
        self._branches: Dict[str, _Branch] = {c: _Branch() for c in CHANNELS}
        self._refreshes = 0
        self._nonfinite = 0   # drained at report
        self._nonpositive = 0  # drained at report; a zero norm has no log

    # ------------------------------------------------------------------ config

    def announce(self):
        if not self.enabled:
            print(f'grad_clip_guard: OFF -- static gradient_norm_clip {self.static_clip:.4g}')
            return
        print(f'grad_clip_guard: ON (guard mode) | p {self.p} (target fire rate '
              f'{1.0 - self.p:.3g}) | eta {self.eta} | warmup {self.warmup_steps}/branch '
              f'({self.warmup_clip}) | cap {self.max_ratio:g}x baseline | '
              f'refresh_on_stage {self.refresh_on_stage} | fallback bar {self.static_clip:.4g}')

    # -------------------------------------------------------------------- bar

    def _branch(self, step_type: str) -> _Branch:
        st = self._branches.get(step_type)
        if st is None:
            # A closed set, from train_step. An unrecognised channel silently
            # falling back to the static bar is exactly the "inert flag" failure
            # mode: the run would look guarded and not be.
            raise KeyError(
                f'grad_clip_guard: unknown step_type {step_type!r}; expected one of {CHANNELS}')
        return st

    def is_calibrated(self, step_type: str) -> bool:
        """Is this branch clipping against its OWN fitted bar yet?

        False while the branch is warming, where `warmup_clip: static` means the
        bar is `gradient_norm_clip` -- a number fitted to nothing in particular.
        A high fire rate there says the static bar is wrong for this branch, NOT
        that the learning rate is wrong, and the two are easy to confuse: the
        guard re-warms at every stage transition when refresh_on_stage is set, so
        every transition briefly looks like saturation.

        MEASURED, run newlogic_qm9cond_newlogic 2026-08-17: var_conditioning
        opens at step 150 with fused grad norms 155.7 / 111 / 67 / 53 / 44
        against the static bar 37.88, so fused_fire_rate sat at 1.000 for the
        whole warmup and then collapsed to 0.000 the moment the fitted bar took
        over. LRController's clip gate read that as a hot rate and cut
        peak_scale, which also tripped the high-water envelope freeze -- pinning
        the rate 15.5x below where the ramp was headed, on evidence that was
        about the BAR and not about the rate.

        Callers gating a decision on the fire rate must consult this first.
        """
        if not self.enabled:
            return False
        return not self._branch(step_type).warming

    def threshold(self, step_type: str) -> float:
        """The norm to clip this branch's gradient at, right now."""
        if not self.enabled:
            return self.static_clip
        st = self._branch(step_type)
        if st.tau is not None:
            return st.tau
        # Never calibrated. 'static' keeps the run's existing behaviour through
        # the warmup window (conservative: the pre-clip norms we are measuring
        # are unaffected by it, since clip_grad_norm_ reports the norm BEFORE
        # rescaling); 'off' leaves the branch unguarded until the bar lands.
        return self.static_clip if self.warmup_clip == 'static' else float('inf')

    # ---------------------------------------------------------------- observe

    def observe(self, step_type: str, norm: float) -> None:
        """Feed this branch the pre-clip gradient norm the step just produced."""
        if not self.enabled:
            return
        st = self._branch(step_type)
        norm = float(norm)

        if not math.isfinite(norm):
            # NOT a quantile observation and not a training event either --
            # train.py skips the optimizer step on a non-finite gradient. Folding
            # it into the tracker would let a NaN streak ratchet the bar upward
            # while no learning happens at all.
            self._nonfinite += 1
            return

        st.n_total += 1
        st.n_obs += 1
        bar = self.threshold(step_type)
        fired = norm > bar
        if fired:
            st.n_fired += 1

        if norm <= 0.0:
            # An all-frozen or all-None-grad step. log(0) is undefined and the
            # observation carries no scale information; counted so a branch that
            # is quietly producing nothing is visible.
            self._nonpositive += 1
            return

        if st.warming:
            self._accumulate(st, norm)
            return

        # Steady state: the bounded-influence quantile step. `fired` is the whole
        # of the input -- the magnitude of the exceedance is deliberately unread.
        st.tau *= math.exp(self.eta * ((1.0 if fired else 0.0) - (1.0 - self.p)))
        self._apply_cap(st)

    def _accumulate(self, st: _Branch, norm: float) -> None:
        lg = math.log(norm)
        st.warm_n += 1
        st.warm_sum += lg
        st.warm_sumsq += lg * lg
        if st.warm_n < self.warmup_steps:
            return
        mu = st.warm_sum / st.warm_n
        var = max(0.0, st.warm_sumsq / st.warm_n - mu * mu) * st.warm_n / max(1, st.warm_n - 1)
        new_tau = math.exp(mu + self._z_p * math.sqrt(var))
        old = st.tau
        st.tau = new_tau
        st.baseline = new_tau
        st.warming = False
        st.warm_n = 0
        st.warm_sum = 0.0
        st.warm_sumsq = 0.0
        # The ratio IS the measurement of what a refresh did. A reset with
        # nothing measuring it is a second mechanism; this one reports its own
        # size every time it fires.
        if old is None:
            print(f'grad_clip_guard: calibrated -- bar {new_tau:.4g} '
                  f'(lognormal p{self.p:g} seed, geo-mean {math.exp(mu):.4g})')
        else:
            print(f'grad_clip_guard: recalibrated -- bar {old:.4g} -> {new_tau:.4g} '
                  f'({new_tau / old:.3g}x)')

    def _apply_cap(self, st: _Branch) -> None:
        """Bound the ratchet relative to the last calibration.

        Upward: a diverging run raises the norm on every step, so the tracker
        would follow it up at `eta` per step -- bounded per step but NOT
        cumulatively (1000 steps at eta=0.01 is e^10). Downward matters just as
        much and is easier to miss: a bar allowed to sink without limit ends up
        under the body of the distribution and SILENTLY REBUILDS THE
        PRECONDITIONER this module exists to replace.
        """
        if st.baseline is None:
            return
        lo, hi = st.baseline / self.max_ratio, st.baseline * self.max_ratio
        if st.tau > hi:
            st.tau = hi
            st.n_saturated += 1
        elif st.tau < lo:
            st.tau = lo
            st.n_saturated += 1

    # ---------------------------------------------------------------- refresh

    def refresh(self, reason: str = 'stage') -> bool:
        """Recalibrate every branch against the incoming regime.

        THE OLD BAR STAYS LIVE while the new one is measured. A hard reset would
        leave the run unguarded across exactly the turbulence the transition
        creates -- fresh Adam moments, new loss coeffs, a rebuilt optimizer --
        and the recalibration prints its own before/after ratio, so the reset is
        observable rather than assumed.
        """
        if not (self.enabled and self.refresh_on_stage):
            return False
        for st in self._branches.values():
            st.warming = True
            st.warm_n = 0
            st.warm_sum = 0.0
            st.warm_sumsq = 0.0
        self._refreshes += 1
        bars = ', '.join(f'{c}={self._branches[c].tau:.4g}'
                         for c in CHANNELS if self._branches[c].tau is not None)
        print(f'grad_clip_guard: refresh ({reason}) -- recalibrating over '
              f'{self.warmup_steps} steps/branch, holding [{bars or "uncalibrated"}]')
        return True

    # ----------------------------------------------------------------- report

    def report(self) -> Dict[str, float]:
        """Metrics for the 10-step report. Windowed counters are drained here.

        `fire_rate` is THE diagnostic: it must sit near 1-p. At 0 the guard is
        absent (the bar has drifted above everything, or the branch is inert); at
        1 it is a preconditioner again and the whole point has been lost. It is
        uninterpretable without `n`, so `n` ships beside it.
        """
        if not self.enabled:
            return {}
        out: Dict[str, float] = {
            'gradclip/refreshes': float(self._refreshes),
            'gradclip/nonfinite': float(self._nonfinite),
            'gradclip/nonpositive': float(self._nonpositive),
        }
        self._nonfinite = 0
        self._nonpositive = 0
        for c in CHANNELS:
            st = self._branches[c]
            if st.n_total == 0:
                continue  # never ran: omit, rather than publish a fictitious 0.0
            if st.tau is not None:
                out[f'gradclip/{c}_tau'] = float(st.tau)
            out[f'gradclip/{c}_n'] = float(st.n_obs)
            out[f'gradclip/{c}_fire_rate'] = (
                float(st.n_fired) / st.n_obs if st.n_obs else 0.0)
            out[f'gradclip/{c}_saturated'] = float(st.n_saturated)
            st.n_obs = 0
            st.n_fired = 0
            st.n_saturated = 0
        return out

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict:
        return {'ver': self._STATE_VER,
                'branches': {c: self._branches[c].to_dict() for c in CHANNELS},
                'refreshes': self._refreshes}

    def load_state_dict(self, state: Optional[dict]) -> bool:
        """Restore a checkpointed tracker. A missing or stale `ver` is DISCARDED
        and the tracker warms from scratch -- never reinterpreted.

        Worth persisting at all because a rewind is exactly when the guard
        matters most: a divergence-triggered reload that dropped the bar would
        re-enter the warmup window with the run already unstable.
        """
        if not isinstance(state, dict) or state.get('ver') != self._STATE_VER:
            return False
        branches = state.get('branches') or {}
        for c in CHANNELS:
            if isinstance(branches.get(c), dict):
                self._branches[c].load(branches[c])
        self._refreshes = int(state.get('refreshes', 0))
        return True
