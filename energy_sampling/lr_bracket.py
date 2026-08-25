"""
The brute-force learning-rate bracket -- the decision half. Pure: no torch, no
trainer. `lr_bracket_probe.BracketDriver` executes what this decides.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT. It runs a handful of fixed-LR
continuations from one mature checkpoint, asks of each only "did it survive", and
picks the highest configured rung a safety margin below the lowest rate that
blew up. There is no estimator here, no fitted curve, no confidence interval and
no ranking by final loss. Every number it acts on is a configured rung or a
boolean.

WHY IT LOOKS LIKE THIS. The mechanism it replaces measured a line-search optimum
(`alpha*`) and steered the rate to it. That statistic was uncorrelated with the
rate it steered -- the slope of log(alpha*) against log(lr) must be -1 and
measured 0.00 +- 0.2 across twelve runs and two stages -- so the controller's
machinery fired perfectly around a number that meant nothing, and a cruise that
never moved the LR was read as successful control. A bracket cannot fail that
way: its only sensor is "did this candidate detonate", which is not a statistic
and so cannot be uncorrelated with the rate.

THE THREE PLACES A BRUTE-FORCE BRACKET CAN STILL LIE, each of which has a
mechanism here rather than a comment:

  * THE ROOT IS NOT AT STEADY STATE. Optimizers are rebuilt at stage
    transitions, so Adam's step counter restarts and its update carries the bias
    correction sqrt(1 - beta2^t) / (1 - beta1^t): 0.153 at t=10, 0.627 at t=500,
    0.975 at t=3000. Bracket from a young root and every trial runs at a
    fraction of its nominal rate, so a too-hot rung survives because the rate
    under test is not the rate applied. `min_root_bias_correction` refuses to
    bracket in that state; the driver computes the factor from the optimizer's
    real step counter, not from the configured burn-in length.

  * THE HARD-FAILURE BAR CANNOT FIRE. A bar at 1e9 catches numerical death and
    nothing else; a rate one rung too hot on this route took the loss from about
    -25 to +318, which is finite and eight orders of magnitude below it. Every
    rung then survives, no boundary is found, and the bracket reports the same
    answer forever while appearing to work. The bars are therefore DERIVED from
    the root's own loss scale (see lr_bracket_probe.HardFailureBars) and this
    module refuses a configuration whose absolute bars are the 1e9 kind.

  * ONE UNLUCKY FAILURE TRUNCATES THE SEARCH. Hard failure is stochastic, so a
    single low rung that happens to blow up would bound the whole bracket far
    below the real boundary. `boundary_confirm_repeats` re-runs the lowest
    failing rung from the root under a DIFFERENT, derived seed -- a same-seed
    re-run is a deterministic replay that reproduces the failure by construction
    and confirms nothing.

WHY `trial_steps` IS SHORT AND HAS NO FLOOR. An earlier draft demanded
trial_steps > 1/(1 - beta2) = 1000. That is the warm-up time for Adam's moments
FROM SCRATCH, and every trial here restores a root whose moments are already
equilibrated: `v` estimates the gradient second moment, which changing the LR
does not invalidate. Divergence from a too-hot rate takes tens of steps. And
CRUISE IS THE LONG TRIAL -- a rate that survives the screen and detonates at
step 400 of cruise surfaces through the ordinary fatal-error path, at a cost of
one detonation rather than a silently wrong rate held for tens of thousands of
steps. The horizon is validated rather than assumed: `steps_to_failure` is
recorded for every failing candidate and a failure landing late in the horizon
sets `horizon_marginal`.

WHY `repeat_every` IS THE DOMINANT COST KNOB. A bracket costs about
len(candidate_scales) x trial_steps discarded steps. Six rungs at 150 steps is
900 discarded, negligible amortised over a stage. Six rungs at 1000 steps
repeated every 10k promoted steps is 60% of one stage's compute, which on the
MLIP routes can consume a 48-hour SLURM budget in calibration alone. 0 means
once per stage, and that is the default posture.
"""

from __future__ import annotations

import math

# ------------------------------------------------------------------- phases

BURN_IN = 'burn_in'      # exactly burn_in_steps at burn_in_scale
BRACKET = 'bracket'      # candidate trials are running
CRUISE = 'cruise'        # a rate has been promoted and is held fixed

#: Trial kinds. `screen` is the one pass over the configured grid; `confirm`
#: re-runs the lowest failing rung under a derived seed; `densify` trials the
#: single inserted rung between the boundary and the survivor below it.
SCREEN = 'screen'
CONFIRM = 'confirm'
DENSIFY = 'densify'

#: Selection outcomes. Each is a distinct epistemic state and they are NOT
#: collapsed: "a boundary was found and a rung below it selected" and "nothing
#: failed so the top rung was assumed unsafe" are different claims about the
#: run, and reporting them alike is how a mechanism that found nothing reads as
#: one that found something.
BRACKETED = 'bracketed'                  # a confirmed boundary, selection below it
UNBRACKETED_HIGH = 'unbracketed_high'    # nothing failed; no boundary was identified
ALL_FAILED = 'all_failed'                # every rung detonated; fall back to burn-in
NO_ELIGIBLE = 'no_eligible_candidate'    # a boundary, but nothing survives below the margin
#: A rung detonated but no failure REPRODUCED, so there is no confirmed boundary
#: and the search was not truncated -- yet the selection is still capped below the
#: lowest rung that blew up. Distinct from `bracketed` because no boundary was
#: confirmed, and distinct from `unbracketed_high` because something did fail.
CAPPED_UNCONFIRMED = 'capped_by_unconfirmed_failure'


class Trial:
    """One planned candidate run. `label` is the key its end state is stored
    under, and is unique across the bracket."""

    __slots__ = ('scale', 'kind', 'repeat', 'seed', 'label')

    def __init__(self, scale, kind, repeat=0, seed=None, label=''):
        self.scale = float(scale)
        self.kind = str(kind)
        self.repeat = int(repeat)
        self.seed = seed
        self.label = str(label)

    def __repr__(self):
        return (f'Trial({self.label}: scale {self.scale:.6g}, {self.kind}'
                + (f', seed {self.seed}' if self.seed is not None else '') + ')')


class Outcome:
    """What one trial did. `steps_to_failure` is None on a survivor.

    `decisive` (failures only): the failure was beyond reasonable doubt --
    non-finite, an absolute backstop, or an excursion far past the bar -- as
    classified by the driver at failure time. Decisive failures skip the
    confirmation re-run: empirically every decisive failure ever confirmed
    reproduced, while the only non-reproduction on record was a marginal
    1.1x-bar graze (wk8 c5). Evidence-scaled confirmation, owner 2026-08-25."""

    __slots__ = ('trial', 'ok', 'reason', 'steps_completed', 'steps_to_failure',
                 'decisive', 'loss_drift')

    def __init__(self, trial, ok, reason, steps_completed, steps_to_failure,
                 decisive=False, loss_drift=None):
        self.trial = trial
        self.ok = bool(ok)
        self.reason = reason
        self.steps_completed = int(steps_completed)
        self.steps_to_failure = steps_to_failure
        self.decisive = bool(decisive)
        #: {'loss_drift','se','t','n'} -- the trial's post-settle OLS loss
        #: drift (total fitted rise) and its standard error, or None when the
        #: post-settle window was too short to fit. The selection input.
        self.loss_drift = loss_drift

    @property
    def scale(self):
        return self.trial.scale


class LRBracket:
    """The bracket's state machine. The driver calls, in order:

        burn_in_complete(steps)      -> bool
        begin_bracket(step, bc)      -> arms the screen
        next_trial()                 -> Trial | None   (None = decided)
        record(trial, ...)           -> after each trial
        select()                     -> the verdict
        promote(scale, step)         -> enter cruise

    Every loop in here is bounded by construction: the screen is one pass over a
    fixed grid, confirmations are capped per rung and the rungs are finite, and
    densification happens at most once and never inside an interval it created.
    There is no convergence criterion anywhere, deliberately -- an unbounded
    settling gate is exactly what the retired controller used to hide a sensor
    that could not resolve.
    """

    #: A failure landing beyond this fraction of the horizon means the horizon is
    #: too short to be trusted -- the run reports `horizon_marginal` beside its
    #: selection. NOT auto-extended: the owner changes the config. Auto-extension
    #: would make the horizon a fitted quantity, which is the thing this
    #: mechanism exists not to have.
    LATE_FAILURE_FRAC = 0.6

    #: Above this, an absolute divergence bar is a numerical-overflow backstop
    #: rather than a bar that can fail a candidate. See the module docstring.
    INERT_BAR = 1.0e8

    def __init__(self,
                 mode: str = 'bracket',
                 burn_in_steps: int = 3000,
                 burn_in_scale: float = 0.05,
                 min_root_bias_correction: float = 0.9,
                 candidate_scales=(),
                 trial_steps: int = 150,
                 safety_rungs: int = 1,
                 repeat_every: int = 0,
                 boundary_confirm_repeats: int = 1,
                 boundary_densify: bool = False,
                 fixed_scale=None,
                 loss_abs=None,
                 grad_abs=None,
                 trial_settle_steps: int = 10,
                 logz_detour_nats=2.0):
        self.mode = str(mode)
        if self.mode not in ('bracket', 'fixed'):
            raise ValueError(f"lr_control.mode must be 'bracket' or 'fixed', got {mode!r}")

        self.burn_in_steps = int(burn_in_steps)
        if self.burn_in_steps < 0:
            raise ValueError(f'lr_control.burn_in_steps must be >= 0, got {burn_in_steps}')
        self.burn_in_scale = float(burn_in_scale)
        # NONZERO AND NON-INERT. Burn-in exists to move the samplers, buffers,
        # normalisation and interacting losses past their initialisation
        # transients; a burn-in at a rate that trains nothing does not do that,
        # and it leaves Adam's step counter advancing over gradients the model
        # never followed.
        if not (self.burn_in_scale > 0) or not math.isfinite(self.burn_in_scale):
            raise ValueError(
                f'lr_control.burn_in_scale must be a positive finite multiplier, got '
                f'{burn_in_scale!r}. Burn-in is deliberately conservative but it must '
                f'still train -- an inert burn-in reaches the root checkpoint with the '
                f'transients it exists to pass through still in progress.')
        self.min_root_bias_correction = float(min_root_bias_correction)
        if not 0.0 < self.min_root_bias_correction <= 1.0:
            raise ValueError(
                f'lr_control.min_root_bias_correction must lie in (0, 1], got '
                f'{min_root_bias_correction!r} -- it is a bias-correction FACTOR, and '
                f'1.0 is steady state.')

        self.trial_steps = int(trial_steps)
        if self.trial_steps < 1:
            raise ValueError(f'lr_control.trial_steps must be >= 1, got {trial_steps}')
        self.safety_rungs = int(safety_rungs)
        if self.safety_rungs < 0:
            raise ValueError(f'lr_control.safety_rungs must be >= 0, got {safety_rungs}')
        self.repeat_every = int(repeat_every or 0)
        if self.repeat_every < 0:
            raise ValueError(f'lr_control.repeat_every must be >= 0 (0 = once per stage), '
                             f'got {repeat_every}')
        self.boundary_confirm_repeats = int(boundary_confirm_repeats)
        if self.boundary_confirm_repeats < 0:
            raise ValueError(f'lr_control.boundary_confirm_repeats must be >= 0, got '
                             f'{boundary_confirm_repeats}')
        self.boundary_densify = bool(boundary_densify)

        # THE SWITCH-SPLASH WINDOW (owner decision 2026-08-24, from the toy
        # workout's D9): trials jump burn-in -> candidate instantly, and on a
        # quiet converged root that jump alone produced few-step excursions that
        # crossed k x span -- STOCHASTICALLY (the same rate failed a trial at
        # step 3 and then cruised 1500 steps clean in fixed mode). For the first
        # trial_settle_steps of every trial the BAR verdicts are not judged;
        # non-finite values and exceptions still fail, and a genuinely fatal
        # rate is convicted the step the window closes -- its wrecked state is
        # discarded on restore either way. The rate itself never changes, so
        # this is not self-rescue: it measures the road, not the jump.
        self.trial_settle_steps = int(trial_settle_steps or 0)
        if self.trial_settle_steps < 0:
            raise ValueError(f'lr_control.trial_settle_steps must be >= 0, got '
                             f'{trial_settle_steps}')
        if self.trial_settle_steps >= self.trial_steps:
            raise ValueError(
                f'lr_control.trial_settle_steps ({trial_settle_steps}) must be smaller '
                f'than trial_steps ({trial_steps}) -- a trial judged over zero steps '
                f'convicts nothing and every rung would survive.')

        # THE GUIDING-STAR GUARD (owner decision 2026-08-24): on these runs
        # log Z rises monotonically until saturation; a dip-and-recover is safe,
        # a detour is not. A candidate whose batch-mean log_Z_learned falls more
        # than this many nats below the ROOT's value fails its trial -- a hard
        # boolean, no fitted statistic, and it catches a minority-branch
        # destabilization (fwd trains Z) that the frac-weighted composite bar is
        # blind to under a dominant bwd frac. None/0 disables. Judged outside
        # the settle window, like the bars. Inert wherever Z does not train
        # (the MLE warm start), because log Z simply does not move there.
        self.logz_detour_nats = (None if logz_detour_nats in (None, 0, 0.0)
                                 else float(logz_detour_nats))
        if self.logz_detour_nats is not None and not (self.logz_detour_nats > 0):
            raise ValueError(f'lr_control.logz_detour_nats must be positive or null, '
                             f'got {logz_detour_nats}')

        self.fixed_scale = None if fixed_scale is None else float(fixed_scale)
        if self.mode == 'fixed':
            if self.fixed_scale is None or not (self.fixed_scale > 0):
                raise ValueError(
                    "lr_control.mode 'fixed' needs a positive lr_control.fixed_scale -- "
                    "the whole point of the mode is that the rate is stated in the "
                    "config rather than discovered.")

        self.candidate_scales = tuple(float(s) for s in (candidate_scales or ()))
        if self.mode == 'bracket':
            self._check_grid()
            self._check_bars(loss_abs, grad_abs)

        # ------------------------------------------------------------- state
        self.phase = BURN_IN
        self.resumed_mid_bracket = False
        self.root_step = None
        self.root_bias_correction = None
        self.promoted_scale = None
        self.promoted_at = None
        self.refusal = None            # why bracketing was refused, if it was
        self._results = []             # [Outcome], in the order they were run
        self._queue = []               # [Trial] still to run
        self._screened = False
        self._non_reproducing = []     # scales whose failure did not repeat
        self._boundary = None          # the boundary SCALE
        self._boundary_confirmed = False
        self._densified = False
        self._extra_scales = []        # rungs densification inserted
        self._brackets = 0             # how many bracket cycles this run has run
        self._discarded_steps = 0
        self._promoted_steps = 0
        self._verdict = None

    # ------------------------------------------------------------ validation

    def _check_grid(self):
        s = self.candidate_scales
        if len(s) < self.safety_rungs + 2:
            raise ValueError(
                f'lr_control.candidate_scales has {len(s)} rungs, which cannot support '
                f'safety_rungs {self.safety_rungs}: the grid needs at least one rung to '
                f'fail and {self.safety_rungs} below it to select from, i.e. '
                f'{self.safety_rungs + 2} rungs minimum. A grid this short can only ever '
                f'return its own bottom rung, which is a guess dressed as a measurement.')
        if any(not (v > 0) or not math.isfinite(v) for v in s):
            raise ValueError(f'lr_control.candidate_scales must all be positive and '
                             f'finite, got {list(s)}')
        if any(b <= a for a, b in zip(s, s[1:])):
            raise ValueError(
                f'lr_control.candidate_scales must be STRICTLY ASCENDING, got {list(s)}. '
                f'Selection is defined by position in this ordering -- "the lowest rung '
                f'that failed", "one rung below" -- so an unsorted grid makes the safety '
                f'margin mean nothing.')
        # WIDE ENOUGH THAT THE TOP IS EXPECTED TO FAIL. A grid whose whole span
        # sits inside the safe region finds no boundary every time and reports
        # `unbracketed_high` forever, which is the mechanism returning the same
        # answer while appearing to work.
        span = s[-1] / s[0]
        if span < 4.0:
            raise ValueError(
                f'lr_control.candidate_scales spans only {span:.3g}x ({s[0]:g} to '
                f'{s[-1]:g}). A bracket needs its top rungs to be EXPECTED to fail; a '
                f'grid narrower than 4x will report unbracketed_high every time and the '
                f'bracket will never have measured anything. Widen the grid.')
        if self.burn_in_scale > s[-1]:
            raise ValueError(
                f'lr_control.burn_in_scale {self.burn_in_scale:g} is above the top '
                f'candidate rung {s[-1]:g}. Burn-in is meant to be conservative relative '
                f'to the rates under test; a burn-in hotter than every candidate has '
                f'already run the run at a rate the bracket is about to condemn.')

    def _check_bars(self, loss_abs, grad_abs):
        """A bracket whose hard-failure bars cannot fire is not a bracket.

        The absolute bars are numerical backstops and stay that way. What this
        refuses is a configuration in which they are the ONLY bars -- because
        then no candidate can fail, every rung survives, `unbracketed_high` is
        reported every time, and the mechanism returns the same answer forever
        while every seam fires correctly. The load-bearing bar is derived from
        the root's own loss scale at bracket time (lr_bracket_probe), which is
        the only place a bar can be set that is guaranteed able to fail a
        catastrophic excursion on whatever route is running.
        """
        for name, bar in (('loss_abs', loss_abs), ('grad_abs', grad_abs)):
            if bar is None:
                continue
            if float(bar) >= self.INERT_BAR:
                raise ValueError(
                    f'lr_control.hard_failure.{name} = {float(bar):g} is at or above '
                    f'{self.INERT_BAR:g}, which catches numerical overflow and nothing '
                    f'else. Measured on this route, a rate one rung too hot took the '
                    f'loss from about -25 to +318 -- finite, and far under such a bar. '
                    f'Under it every candidate completes, no boundary is found, and the '
                    f'bracket reports the same answer forever while appearing to work. '
                    f'Set a bar that can fail a candidate.')

    # ------------------------------------------------------------- burn-in

    def burn_in_complete(self, steps_since_entry: int) -> bool:
        """STEP COUNT ONLY. Burn-in never waits on a learned metric to settle:
        an unbounded wait on a quantity the rate itself is moving is how the
        retired controller spent 800 steps of a run muted by its own gate."""
        return int(steps_since_entry) >= self.burn_in_steps

    def scale_now(self) -> float:
        """The multiplier the trainer should be running at RIGHT NOW, outside a
        trial. During a trial the driver sets the candidate scale directly."""
        if self.phase == CRUISE and self.promoted_scale is not None:
            return float(self.promoted_scale)
        return float(self.burn_in_scale)

    # ------------------------------------------------------------ the bracket

    def refuse(self, why: str, scale=None, step=None):
        """Give up on bracketing and hold a known-safe rate, loudly.

        THE FALLBACK IS NOT ALWAYS THE BURN-IN SCALE. On the FIRST cycle it is:
        nothing has been measured, so the conservative rate is all there is. On a
        REPEAT it is not -- the run has a promoted rate that a previous bracket
        measured and that has been training successfully ever since, and throwing
        it away to drop back to the burn-in scale would make a refusal COST the
        run its rate. Worse, clearing `promoted_at` disabled `repeat_due`
        permanently, so one refused repeat meant the run never re-bracketed
        again. Callers pass the rate to keep."""
        self.refusal = str(why)
        self.phase = CRUISE
        self.promoted_scale = float(self.burn_in_scale if scale is None else scale)
        # RE-ARM THE CLOCK rather than clearing it. A refusal is a reason to try
        # again later, not a reason to stop trying.
        self.promoted_at = None if step is None else int(step)
        self._verdict = None
        self._queue = []
        return self.promoted_scale

    def begin_bracket(self, step: int, bias_correction=None):
        """Arm the screen from a fresh root. Returns the list of screen trials."""
        self.phase = BRACKET
        self.root_step = int(step)
        self.root_bias_correction = (None if bias_correction is None
                                     else float(bias_correction))
        self._brackets += 1
        self._results = []
        self._screened = False
        self._non_reproducing = []
        self._boundary = None
        self._boundary_confirmed = False
        self._densified = False
        self._extra_scales = []
        self._verdict = None
        self._queue = [Trial(s, SCREEN, label=f'c{i}')
                       for i, s in enumerate(self.candidate_scales)]
        return list(self._queue)

    def next_trial(self):
        """The next trial to run, or None once the bracket has decided.

        BOUNDED BY CONSTRUCTION. The screen is one pass over a fixed grid; each
        rung is confirmed at most `boundary_confirm_repeats` times and there are
        finitely many rungs; densification runs at most once and never inside an
        interval it created. Nothing here retries on a criterion.
        """
        if self._queue:
            return self._queue.pop(0)
        if self.phase != BRACKET:
            return None
        self._screened = True
        nxt = self._plan_confirmation()
        if nxt is not None:
            return nxt
        return self._plan_densification()

    def _plan_confirmation(self):
        """Re-run the lowest unresolved failing rung under a DERIVED seed.

        THE SEED MUST DIFFER. Screen trials all restore an identical RNG state so
        they are comparable; a same-seed re-run is therefore a deterministic
        replay of the original trial. It reproduces the failure by construction,
        confirms nothing, and reads as a passing check. The seed is derived from
        the root so the confirmation is reproducible without being a replay, and
        it is the ONLY thing that differs -- root state, rate and horizon are
        identical.

        The recursion below is bounded by the rung count: each pass either
        returns a trial or writes one more rung off as non-reproducing, and
        there are finitely many rungs.
        """
        if self._boundary_confirmed:
            return None
        fails = self._unresolved_failures()
        if not fails:
            self._boundary_confirmed = True
            return None
        target = fails[0]
        # EVIDENCE-SCALED CONFIRMATION (owner, 2026-08-25): a DECISIVE screen
        # failure -- non-finite, absolute backstop, or an excursion far past the
        # bar -- is its own confirmation. The re-run exists to catch the
        # marginal coin flip near the bar (the stochastic switch splash, D9);
        # spending 150 steps re-proving a 25x-bar detonation protects nothing.
        # Every decisive failure ever confirmed reproduced; the single
        # non-reproduction on record was a 1.1x-bar graze.
        if any(getattr(o, 'decisive', False)
               for o in self._screen_outcomes()
               if o.scale == target and not o.ok):
            self._boundary = target
            self._boundary_confirmed = True
            return None
        if self.boundary_confirm_repeats == 0:
            # No confirmation budget: the first failing rung IS the boundary.
            # Stated rather than defaulted, because it is the configuration in
            # which one unlucky trial bounds the whole search.
            self._boundary = target
            self._boundary_confirmed = True
            return None
        done = [o for o in self._results
                if o.trial.kind == CONFIRM and o.scale == target]
        if any(o.ok for o in done):
            # A rung that survives its repeat is a NON-REPRODUCING failure, not
            # the boundary. The search continues upward from it -- and it is not
            # a survivor either, because it did detonate once.
            self._non_reproducing.append(target)
            return self._plan_confirmation()
        if len(done) >= self.boundary_confirm_repeats:
            self._boundary = target
            self._boundary_confirmed = True
            return None
        r = len(done) + 1
        return Trial(target, CONFIRM, repeat=r, seed=self.confirm_seed(r),
                     label=f'confirm_{self._scale_tag(target)}_r{r}')

    def _plan_densification(self):
        """One inserted rung between the confirmed boundary and the highest
        survivor below it. Refines the boundary by one step and nothing more:
        it runs once, it is never re-densified, and it has no exit criterion."""
        if not self.boundary_densify or self._densified:
            return None
        if self._boundary is None:
            return None
        below = [s for s in self._survivor_scales() if s < self._boundary]
        if not below:
            return None
        self._densified = True
        mid = math.sqrt(max(below) * self._boundary)
        if not (max(below) < mid < self._boundary):
            return None                     # degenerate interval; nothing to insert
        if any(mid == s for s in self.ordering()):
            # The inserted rung coincides with one already trialled, so its
            # outcome would be recorded against a scale that already has one and
            # could never change the verdict. Refuse rather than spend a trial.
            return None
        self._extra_scales.append(mid)
        return Trial(mid, DENSIFY, label=f'densify_{self._scale_tag(mid)}')

    def record(self, trial: Trial, ok: bool, reason=None,
               steps_completed: int = 0, steps_to_failure=None,
               decisive: bool = False, loss_drift=None):
        """Fold in one finished trial."""
        out = Outcome(trial, ok, reason, steps_completed, steps_to_failure,
                      decisive, loss_drift)
        self._results.append(out)
        self._discarded_steps += int(steps_completed)
        if trial.kind == DENSIFY and not ok:
            # the inserted rung detonated: it IS the boundary now, one step lower
            self._boundary = trial.scale
        return out

    # ------------------------------------------------------------- selection

    def _screen_outcomes(self):
        return [o for o in self._results if o.trial.kind in (SCREEN, DENSIFY)]

    def _unresolved_failures(self):
        """Failing screen rungs not yet written off as non-reproducing, ascending."""
        return sorted({o.scale for o in self._screen_outcomes()
                       if not o.ok and o.scale not in self._non_reproducing})

    def _survivor_scales(self):
        """Rungs that completed the full horizon AND never failed. A rung whose
        failure did not reproduce is deliberately NOT a survivor: it detonated
        once, and treating non-monotone outcomes conservatively is the rule."""
        failed = {o.scale for o in self._results if not o.ok}
        return sorted({o.scale for o in self._screen_outcomes()
                       if o.ok and o.scale not in failed})

    def _detonated_scales(self):
        """Every rung that blew up even ONCE, reproducing or not.

        THE SELECTION CEILING, and it is not the same thing as the boundary. A
        non-reproducing failure is correctly excluded from the boundary -- that
        is the whole point of confirming, and one unlucky trial must not truncate
        the search. But it must still CAP what may be selected, and it did not:
        with rung 0.2 detonating on its screen and surviving its confirmation and
        nothing else failing, `select` reported `unbracketed_high` -- "nothing
        failed" -- and chose 0.8, four times the rate that had blown up. The rung
        detonated. Whatever the confirmation says about reproducibility, running
        ABOVE it is not conservative."""
        return sorted({o.scale for o in self._results if not o.ok})

    def ordering(self):
        """Every rung this bracket trialled, ascending. Densification inserts
        into it, which is exactly how one inserted rung refines the margin."""
        return sorted(set(self.candidate_scales) | set(self._extra_scales))

    def select(self) -> dict:
        """The verdict. Uses ONLY the configured ordering and hard survival --
        never a final loss, never an interpolation between rungs."""
        order = self.ordering()
        survivors = self._survivor_scales()
        boundary = self._boundary
        marginal = any(
            o.steps_to_failure is not None
            and o.steps_to_failure > self.LATE_FAILURE_FRAC * self.trial_steps
            for o in self._results if not o.ok)

        verdict = {
            'status': None,
            'scale': None,
            'restore': None,             # trial label to restore, or 'root'
            'boundary_scale': boundary,
            'boundary_confirmed': bool(self._boundary_confirmed and boundary is not None),
            'non_reproducing': list(self._non_reproducing),
            'densified': bool(self._densified),
            'horizon_marginal': bool(marginal),
            'safety_rungs': self.safety_rungs,
            'survivors': survivors,
            'ordering': order,
            'margin_rungs': None,
            'selection_mode': None,
            'loss_drift': None,
            'selection_ceiling': None,
        }

        # THE SELECTION CEILING IS THE LOWEST RUNG THAT EVER DETONATED, which is
        # NOT the same as the boundary and is at or below it.
        #
        # The boundary answers "where does the search stop", and a failure that
        # does not reproduce must not set it -- that is what confirmation is for,
        # and one unlucky trial must not truncate the grid. The ceiling answers
        # "what may we run", and there a single detonation is enough: whatever
        # the confirmation says about reproducibility, selecting AT OR ABOVE a
        # rate that blew up is not conservative.
        #
        # Collapsing the two let the selection climb four rungs above a rung that
        # had detonated, and report `unbracketed_high` -- the claim that nothing
        # failed -- while doing it.
        detonated = self._detonated_scales()
        ceiling = min(detonated) if detonated else None
        verdict['selection_ceiling'] = ceiling

        if ceiling is not None:
            pos = order.index(ceiling)
            cut = pos - self.safety_rungs
            status = BRACKETED if boundary is not None else CAPPED_UNCONFIRMED
        else:
            # NOTHING FAILED. This is not a measured boundary and must not be
            # reported as one: the top of the grid is merely untested from above.
            pos = len(order)
            cut = len(order) - 1 - self.safety_rungs
            status = UNBRACKETED_HIGH

        if not survivors:
            verdict.update(status=ALL_FAILED, scale=self.burn_in_scale, restore='root')
            self._verdict = verdict
            return verdict

        eligible = [s for s in survivors if order.index(s) <= cut]
        if not eligible:
            verdict.update(status=NO_ELIGIBLE, scale=self.burn_in_scale, restore='root')
            self._verdict = verdict
            return verdict

        # SLOPE-FIRST SELECTION (owner 2026-08-25). "Hottest survivor" was
        # falsified repeatedly on var_conditioning: a rung can survive its
        # horizon while parking the loss ABOVE the root (the 0.566 and 1.13
        # promotions both poisoned the run), so stability is a CONSTRAINT
        # (eligibility, above) and PROGRESS is the objective. The rung with
        # the most-negative post-settle loss_drift wins; it must beat the
        # coldest eligible rung by 2 combined standard errors, else the
        # coldest wins -- on a plateau every drift is ~flat, no rung clears
        # the bar, and the selection correctly stays cold. Degenerate cases
        # fall back to the legacy hottest-survivor rule: drift is fitted only
        # when the post-settle window has >= 10 samples, so short-horizon
        # harnesses (and any misconfigured settle window) keep the old
        # behavior rather than selecting on absent data.
        by_scale = {}
        for o in self._screen_outcomes():
            if o.ok:                      # last write wins, same as _label_for
                by_scale[o.scale] = o
        drifts = {s: by_scale[s].loss_drift for s in eligible
                  if s in by_scale and by_scale[s].loss_drift is not None}
        coldest = min(eligible)
        selection_mode = 'survival_max'
        chosen = max(eligible)
        if coldest in drifts and len(drifts) >= 2:
            best = min(drifts, key=lambda s: drifts[s]['loss_drift'])
            gap = drifts[coldest]['loss_drift'] - drifts[best]['loss_drift']
            se = math.hypot(drifts[best].get('se') or 0.0,
                            drifts[coldest].get('se') or 0.0)
            if best != coldest and gap > 2.0 * se:
                chosen = best
            else:
                chosen = coldest
            selection_mode = 'loss_drift'
        verdict.update(status=status, scale=chosen, restore=self._label_for(chosen),
                       margin_rungs=pos - order.index(chosen),
                       selection_mode=selection_mode,
                       loss_drift=(drifts.get(chosen) or {}).get('loss_drift'))
        self._verdict = verdict
        return verdict

    def promote(self, scale: float, step: int):
        """Enter cruise. The repeat clock counts PROMOTED steps only -- the
        discarded trial compute is not training the run keeps, so charging it to
        the clock would re-bracket sooner the more it spent bracketing."""
        self.phase = CRUISE
        self.promoted_scale = float(scale)
        self.promoted_at = int(step)
        # a promotion supersedes any earlier refusal; leaving the latch set kept
        # lr_bracket/refused at 1.0 for the rest of the stage after a later
        # successful race (audit 2026-08-25)
        self.refusal = None
        self._queue = []
        return self.promoted_scale

    def note_promoted_steps(self, n: int):
        self._promoted_steps += int(n)

    def repeat_due(self, step: int) -> bool:
        """Is another bracket cycle due? `repeat_every` 0 means once per stage."""
        if self.mode != 'bracket' or self.repeat_every <= 0:
            return False
        if self.phase != CRUISE or self.promoted_at is None:
            return False
        return int(step) - int(self.promoted_at) >= self.repeat_every

    def next_repeat_step(self):
        if self.mode != 'bracket' or self.repeat_every <= 0 or self.promoted_at is None:
            return None
        return int(self.promoted_at) + self.repeat_every

    # -------------------------------------------------------------- plumbing

    def confirm_seed(self, repeat: int) -> int:
        """Deterministic in the root, distinct per repeat. Reproducible without
        being a replay; never derived from wall-clock or GPU nondeterminism."""
        return (int(self.root_step or 0) * 1_000_003 + 7919 * int(repeat)) % (2 ** 31 - 1)

    @staticmethod
    def _scale_tag(scale: float) -> str:
        return f'{float(scale):.6g}'.replace('.', 'p').replace('-', 'm').replace('+', '')

    def _label_for(self, scale):
        for o in self._results:
            if o.ok and o.scale == scale and o.trial.kind in (SCREEN, DENSIFY):
                return o.trial.label
        return None

    def results(self):
        return list(self._results)

    # ---------------------------------------------------------------- report

    _PHASE_CODE = {BURN_IN: 0, BRACKET: 1, CRUISE: 2}
    _STATUS_CODE = {BRACKETED: 0, UNBRACKETED_HIGH: 1, ALL_FAILED: 2,
                    NO_ELIGIBLE: 3, CAPPED_UNCONFIRMED: 4}

    def report(self) -> dict:
        """Enough to reconstruct the experiment, and nothing that would read as
        an explanation for the selection -- there is no alpha*, no cos, no fitted
        anything, because none of those entered the decision."""
        out = {
            'lr_bracket/phase': float(self._PHASE_CODE.get(self.phase, -1)),
            'lr_bracket/fixed_mode': 1.0 if self.mode == 'fixed' else 0.0,
            'lr_bracket/scale': float(self.scale_now()),
            'lr_bracket/brackets': float(self._brackets),
            'lr_bracket/discarded_steps': float(self._discarded_steps),
            'lr_bracket/promoted_steps': float(self._promoted_steps),
        }
        if self.root_step is not None:
            out['lr_bracket/root_step'] = float(self.root_step)
        if self.root_bias_correction is not None:
            out['lr_bracket/root_bias_correction'] = float(self.root_bias_correction)
        if self.promoted_scale is not None:
            out['lr_bracket/promoted_scale'] = float(self.promoted_scale)
        nxt = self.next_repeat_step()
        if nxt is not None:
            out['lr_bracket/next_repeat_step'] = float(nxt)
        v = self._verdict
        if v:
            out['lr_bracket/status'] = float(self._STATUS_CODE.get(v['status'], -1))
            out['lr_bracket/horizon_marginal'] = float(v['horizon_marginal'])
            out['lr_bracket/boundary_confirmed'] = float(v['boundary_confirmed'])
            out['lr_bracket/non_reproducing'] = float(len(v['non_reproducing']))
            out['lr_bracket/densified'] = float(v['densified'])
            if v.get('boundary_scale') is not None:
                out['lr_bracket/boundary_scale'] = float(v['boundary_scale'])
            if v.get('selection_ceiling') is not None:
                out['lr_bracket/selection_ceiling'] = float(v['selection_ceiling'])
            if v.get('margin_rungs') is not None:
                out['lr_bracket/margin_rungs'] = float(v['margin_rungs'])
        if self.refusal is not None:
            out['lr_bracket/refused'] = 1.0
        return out

    def summary(self) -> str:
        """The bracket, as a table a human can read in the run log. Every row is
        a rung and a boolean; there is nothing here to interpret."""
        lines = [f'lr_bracket: root step {self.root_step}, '
                 f'bias correction '
                 + ('n/a' if self.root_bias_correction is None
                    else f'{self.root_bias_correction:.3f}')
                 + f', horizon {self.trial_steps} steps']
        for o in self._results:
            when = ('' if o.steps_to_failure is None
                    else f' at step {o.steps_to_failure}')
            lines.append(
                f'  {o.trial.label:<24} scale {o.scale:<10.6g} {o.trial.kind:<8}'
                + (f' seed {o.trial.seed}' if o.trial.seed is not None else '')
                + ('  SURVIVED' if o.ok
                   else f'  FAILED ({o.reason}){when}'
                        + (' [DECISIVE]' if getattr(o, 'decisive', False) else '')))
        v = self._verdict
        if v:
            lines.append(
                f'  -> {v["status"]}: scale {v["scale"]:.6g}, boundary '
                + ('none' if v['boundary_scale'] is None
                   else f'{v["boundary_scale"]:.6g}')
                + ('' if v.get('selection_ceiling') is None
                   else f', ceiling {v["selection_ceiling"]:.6g}')
                + f', margin {v["margin_rungs"]} rung(s)'
                + (', HORIZON MARGINAL' if v['horizon_marginal'] else ''))
        return '\n'.join(lines)

    def race_rows(self):
        """The race as plain data, one row per trial -- same content as
        summary(), machine-shaped, for external reporting (the wandb race
        table). Until this existed the per-rung results lived ONLY in the run
        log, so comparing ladders across battery arms meant grepping N SLURM
        .out files while wandb carried just the verdict scalars."""
        return [{'label': o.trial.label,
                 'kind': o.trial.kind,
                 'scale': float(o.scale),
                 'seed': (None if o.trial.seed is None else int(o.trial.seed)),
                 'survived': bool(o.ok),
                 'steps_to_failure': (None if o.steps_to_failure is None
                                      else int(o.steps_to_failure)),
                 'reason': ('' if o.ok else str(o.reason or '')),
                 'decisive': bool(getattr(o, 'decisive', False))}
                for o in self._results]

    @property
    def cycle_index(self) -> int:
        """1-based index of the current/most recent bracket cycle (increments
        at begin_bracket). Keys the per-cycle race reports."""
        return int(self._brackets)

    # ------------------------------------------------------ checkpoint state

    _STATE_VER = 1

    def state_dict(self) -> dict:
        """What a resume needs. Trial RESULTS travel; trial STATES do not -- they
        live in host RAM and die with the process, so a resume that lands
        mid-bracket restarts the cycle from the resumed (already mature) state
        rather than pretending to hold checkpoints it no longer has."""
        return {
            'ver': self._STATE_VER,
            'phase': self.phase,
            'root_step': self.root_step,
            'promoted_scale': self.promoted_scale,
            'promoted_at': self.promoted_at,
            'brackets': self._brackets,
            'discarded_steps': self._discarded_steps,
            'promoted_steps': self._promoted_steps,
            'refusal': self.refusal,
        }

    def load_state_dict(self, state) -> bool:
        """DISCARD rather than reinterpret, as every other state holder here
        does. Returns whether anything was restored."""
        if not isinstance(state, dict) or state.get('ver') != self._STATE_VER:
            return False
        self.phase = state.get('phase', BURN_IN)
        self.root_step = state.get('root_step')
        self.promoted_scale = state.get('promoted_scale')
        self.promoted_at = state.get('promoted_at')
        self._brackets = int(state.get('brackets', 0))
        self._discarded_steps = int(state.get('discarded_steps', 0))
        self._promoted_steps = int(state.get('promoted_steps', 0))
        self.refusal = state.get('refusal')
        # A resume that lands mid-bracket has no trial states to restore, so the
        # cycle is re-armed rather than resumed half-decided. The driver says so.
        self.resumed_mid_bracket = self.phase == BRACKET
        if self.resumed_mid_bracket:
            self.phase = BURN_IN
        return True
