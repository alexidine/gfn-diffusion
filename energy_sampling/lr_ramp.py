"""
The checkpointed LR ramp -- section 7 of `docs/design/lr_handoff_2026-08-21.md`.

Selects a high but non-brittle initial rate WITHOUT waiting for numerical
divergence, by climbing geometric rungs on real on-policy training and stopping
at the first rung the evidence rejects. No torch here: this is the decision half,
and it is tested as one. The trainer supplies readings and executes the actions.

WHY IT IS NOT THE FROZEN TRIAL IT REPLACES. A rung is REAL TRAINING at that rate,
so the policy generates its own data and the rate governs how fast the target
distribution moves -- which in a GFlowNet is most of the dynamics. Trials on
recorded batches optimise a different function whose optimum sits elsewhere (that
argument is section 4 of the handoff, and it is why this module exists at all).
The price is that a rejected rung's steps are genuinely discarded on rollback,
which is what `projected_cost` exists to state up front rather than discover.

THREE RECONCILIATIONS the plan left open (section 8), resolved here and marked
`DECISION` so they are visible rather than inferred:

  DECISION 8a -- THE ACCEPT CRITERION IS IN alpha* AGAINST alpha_target.
    Section 7 asked that the no-improvement crossing sit 1.5-2x beyond the live
    step. Under the quadratic model that crossing is 2*alpha*, so the request is
    alpha* >= 0.75-1.0 -- cruise at about the ONE-STEP optimum, while the pooled
    controller that takes over targets alpha_target 4. As written the two phases
    pull against each other by ~4x, and the handover would read as controller
    instability rather than as a units mismatch. So a rung is accepted while its
    POOLED alpha* is at or above `alpha_target`, which is the same quantity, on
    the same scale, that cruise steers to. The margin is not lost: it is
    alpha_target, stated once instead of twice in different units.

  DECISION 8c -- THE RAMP RUNS AFTER THE STAGE-ENTRY TRANSIENT, NOT THROUGH IT.
    A ray reading taken while log Z is still making a large level shift describes
    the transition, not the stage. Classifying rungs against a moving target
    would put the whole weight of the decision on the residence rule, which is
    not what that rule is for. The caller gates `start()` on the same settling
    signal the pooled estimator uses.

  DECISION 8a(ii) -- CRUISE STARTS AT THE LATEST CLEAN RUNG, NOT ONE BELOW IT.
    Section 7's "one geometric rung below" was written when a clean rung ran
    ~4x hotter than cruise wanted, so a rung of margin sat on top of an already
    hot rate. Once the accept criterion IS alpha_target, subtracting another rung
    double-counts the same margin and hands cruise a rate its own estimator will
    immediately climb back out of. `cruise_backoff_rungs` restores the old
    behaviour at 1.

AND ONE THE PLAN RAISED THAT COSTS RESIDENCE (section 8d). "Do not reset
optimizer moments" means each rung is read partly through the previous rung's
Adam state: the second moment is an EMA with horizon ~1/(1-beta2), so a residence
shorter than that reads the rate it just left. `residence_floor` takes the max of
the configured dwell and that horizon, and says which one bound.
"""

from __future__ import annotations

import math

from energy_sampling.lr_pool import NOISE_FLOOR_DEX, OptimumPool

#: Rung classifications, section 7.
CLEAN = 'clean'
PENDING = 'pending'
BOUNDARY = 'boundary'
HARD_FAILURE = 'hard_failure'

#: Actions the trainer executes. Deliberately verbs the trainer owns, so this
#: module never names a checkpoint tag -- section 8b: `best` is a hardlink to
#: whatever `running` last wrote and the emergency rewind reads it, so a ramp
#: writing into the run's own namespace would edit its own rollback target.
DWELL = 'dwell'                 # stay, keep collecting
SAVE_CLEAN = 'save_clean'       # this rung passed: checkpoint it, then climb
CLIMB = 'climb'                 # raise to the next rung
DESCEND = 'descend'             # the FIRST rung was already rejected: go down
ROLLBACK = 'rollback'           # restore the latest clean rung and finish
FINISH = 'finish'               # ramp complete; `cruise_scale` is the handover


class RampLadder:
    """One checkpointed ramp, as a state machine over rungs.

    `peak_scale` is the controller's multiplier, which is what the ramp moves and
    what a rung IS. Everything here is in that unit, so the ladder never needs to
    know a base learning rate or how many parameter groups there are.
    """

    def __init__(self, alpha_target: float = 4.0, factor: float = 1.5,
                 min_residence_steps: int = 500, adam_beta2: float | None = 0.999,
                 min_readings: int | None = None, move_t: float = 2.0,
                 persistence: int = 2, min_adverse_families: int = 2,
                 max_scale: float = 64.0, min_scale: float = 1.0 / 64.0,
                 cruise_backoff_rungs: int = 0, max_residence_multiple: float = 3.0,
                 pool_kwargs=None):
        if factor <= 1.0:
            raise ValueError(f'factor must exceed 1, got {factor}')
        self.alpha_target = float(alpha_target)
        self.factor = float(factor)
        self.persistence = max(1, int(persistence))
        self.min_adverse_families = max(1, int(min_adverse_families))
        self.max_scale = float(max_scale)
        self.min_scale = float(min_scale)
        self.cruise_backoff_rungs = int(cruise_backoff_rungs)
        self._pool_kwargs = dict(pool_kwargs or {})
        self.move_t = float(move_t)
        # READINGS PER RUNG, DERIVED FROM THE NOISE AND THE RUNG SPACING.
        #
        # A rung verdict has to resolve at the granularity the ladder moves in:
        # one geometric rung, log10(factor). The pooled bar is move_t * se and
        # se is noise/sqrt(n), so
        #
        #     move_t * noise / sqrt(n) < log10(factor)
        #     =>  n > (move_t * noise / log10(factor))^2
        #
        # At the measured 0.22 dex, move_t 2 and factor 1.5 that is n > 6.25, so
        # SEVEN. The default was 3 -- inherited from the cruise controller, where
        # the window keeps growing -- which gives a bar of 0.25 dex against a
        # rung of 0.176: coarser than the thing it is deciding. That is the
        # "ramp is noisier than the controller it hands to" finding, in one line.
        noise = float(self._pool_kwargs.get('noise_floor_dex', NOISE_FLOOR_DEX))
        derived = math.ceil((self.move_t * noise / math.log10(self.factor)) ** 2)
        self.min_readings = (max(1, int(min_readings)) if min_readings is not None
                             else max(3, int(derived)))
        self.readings_bound_by = ('config' if min_readings is not None
                                  else 'rung_resolution')

        # SECTION 8d. Adam's second moment is an EMA with horizon ~1/(1-beta2);
        # a rung shorter than that is read partly through the previous rung's
        # moments, so the reading lags the rate. Which bound binds is recorded
        # rather than folded away, because a residence set by the moment horizon
        # and one set by the config have different reasons to change.
        moment_steps = (0 if not adam_beta2 else
                        int(math.ceil(1.0 / max(1e-9, 1.0 - float(adam_beta2)))))
        self.configured_residence = int(min_residence_steps)
        self.moment_residence = moment_steps
        self.residence = max(self.configured_residence, moment_steps)
        self.residence_bound_by = ('adam_moments'
                                   if moment_steps > self.configured_residence
                                   else 'config')
        self.max_residence = int(self.residence * max(1.0, float(max_residence_multiple)))

        self.state = 'idle'
        self.rung = 0
        self.scale = None            # peak_scale of the CURRENT rung
        self.entered_at = None       # step the current rung was entered at
        self.clean_scale = None      # peak_scale of the latest CLEAN rung
        self.clean_rung = None
        self.cruise_scale = None     # the handover, once FINISH is emitted
        self.outcome = None          # 'boundary' | 'hard_failure' | 'lower_bound_only'
        self.history = []            # one row per classified rung -- the evidence
        self._pool = None
        self._adverse_streak = 0
        self._below_streak = 0
        self._margin = None
        self._hard = None
        self._last = {}

    # ------------------------------------------------------------------ start

    def start(self, peak_scale: float, step: int) -> dict:
        """Begin at `peak_scale`. The caller gates this on the stage's settling
        signal -- DECISION 8c."""
        if not (peak_scale > 0 and math.isfinite(peak_scale)):
            raise ValueError(f'peak_scale must be positive, got {peak_scale}')
        self.state = 'running'
        self.rung = 0
        self._enter(float(peak_scale), int(step))
        return self._act(DWELL, 'ramp_started')

    def _enter(self, scale, step):
        self.scale = scale
        self.entered_at = int(step)
        self._adverse_streak = 0
        self._below_streak = 0
        self._margin = None          # (gap, bar) from the latest admitted reading
        self._hard = None
        # ONE POOL PER RUNG (section 7 step 7). The rate is part of the regime,
        # so readings from the rung below estimate a different number -- and the
        # pooled quantity `peak * alpha*` is invariant to the rate precisely so
        # that the SAME number is being estimated within a regime, which is what
        # makes a per-rung reset necessary rather than merely tidy.
        self._pool = OptimumPool(min_readings=self.min_readings,
                                 move_t=self.move_t, **self._pool_kwargs)
        self._pool.reset(('rung', self.rung))

    # -------------------------------------------------------------- observing

    def observe_ray(self, reading, peak_scale: float, step: int) -> bool:
        """Fold one ray calibration into the current rung's pool, and re-judge
        the rung's margin.

        THE MARGIN STREAK IS COUNTED HERE, NOT IN `tick`. Persistence has to be
        over READINGS -- `tick` runs every step, so a streak counted there would
        reach any threshold within a few steps of the pool first being able to
        speak, and "a single adverse reading must not reject a rung" (section 7)
        would be satisfied only in wording.
        """
        if self.state != 'running':
            return False
        ok = self._pool.observe(reading, peak_scale, step, ('rung', self.rung))
        if not ok:
            return False
        incumbent = math.log10(self.scale) + math.log10(self.alpha_target)
        est = self._pool.estimate(incumbent=incumbent)
        if est is None:
            return True
        gap = est['log_opt'] - incumbent
        bar = self.move_t * est['se']
        self._margin = (gap, bar)
        # DECISION 8a: the margin is "the pooled optimum is at or above the rate
        # cruise would steer to at this rung", i.e. alpha* >= alpha_target.
        #
        # THE SE BAR IS ONE-SIDED, and that asymmetry is load-bearing rather than
        # a taste. It buys PERSISTENCE against noise on the reject side only; it
        # does NOT widen the accept side. Measured while building this: with the
        # bar slack in both directions the ramp accepted every rung whose gap sat
        # inside it, so on a synthetic surface with a known optimum it climbed
        # two rungs PAST the setpoint and handed cruise a rate 1.6x hot -- eating
        # most of the margin alpha_target exists to provide. Climbing is the
        # speculative direction; the evidence has to say "still at or above", not
        # merely "not yet provably below".
        self._below_streak = self._below_streak + 1 if gap < -bar else 0
        return True

    def observe_coherence(self, families) -> None:
        """One on-policy coherence sample: {family_name: 'ok'|'adverse'|'unknown'}.

        Section 7 is explicit that distribution MOVEMENT is not failure --
        training is supposed to move the distribution -- so a rung is rejected on
        this evidence only when several families agree AND they keep agreeing.
        `unknown` is neither: a family a route does not publish must not read as
        healthy, and must not convict either.
        """
        if self.state != 'running':
            return
        adverse = sum(1 for v in (families or {}).values() if v == 'adverse')
        if adverse >= self.min_adverse_families:
            self._adverse_streak += 1
        else:
            self._adverse_streak = 0

    def observe_hard_failure(self, reason: str) -> None:
        """Nonfinite values, runaway parameters or optimizer state, or another
        explicit emergency. Rolls back IMMEDIATELY -- no persistence rule, no
        dwell, and no continuing past a boundary in search of catapult
        recovery."""
        if self.state == 'running':
            self._hard = str(reason)

    # ----------------------------------------------------------------- ticking

    def tick(self, step: int) -> dict:
        """What the trainer should do now. Called once per step; cheap."""
        if self.state != 'running':
            return self._act(DWELL, 'not_running')
        if self._hard is not None:
            return self._reject(HARD_FAILURE, self._hard, step)

        verdict, why = self._classify()
        if verdict == BOUNDARY:
            return self._reject(BOUNDARY, why, step)
        resident = int(step) - self.entered_at
        if resident < self.residence:
            return self._act(DWELL, f'residence_{resident}_of_{self.residence}')
        if verdict == PENDING:
            # Extend the dwell WITHOUT increasing the LR (section 7 step 8).
            # An unresolved rung is not a passed one, and climbing off it would
            # spend the next rung's evidence budget answering this rung's
            # question.
            #
            # BUT BOUNDED. The pooled SE has a floor (per-reading noise is
            # 0.20-0.25 dex and exponential forgetting caps the effective count),
            # so a rung whose gap sits inside that floor's band would extend its
            # dwell forever and the ramp would never terminate. Past
            # `max_residence` the rung is resolved on the POINT estimate alone,
            # which in this band always means below setpoint -- the safe
            # direction, and stated rather than reached by hanging.
            if resident >= self.max_residence:
                if self._margin is None:
                    # NO EVIDENCE AT ALL -- the pool never spoke, because every
                    # reading was unresolved or the sensor never fired. That is a
                    # SENSOR outcome, not a rung verdict, and convicting the rung
                    # for it would report a stability boundary the run never
                    # found. Stop and say so.
                    self.outcome = 'no_evidence'
                    self._record(PENDING, f'no_evidence_{why}', step)
                    return self._finish(self.clean_scale or self.scale,
                                        f'no_evidence_{why}', step)
                return self._reject(BOUNDARY, f'pending_timeout_{why}', step)
            return self._act(DWELL, f'pending_{why}')

        self._record(CLEAN, why, step)
        self.clean_scale, self.clean_rung = self.scale, self.rung
        nxt = self.scale * self.factor
        if nxt > self.max_scale:
            # Section 7: report `lower_bound_only`. Do NOT claim the stability
            # boundary was found -- nothing here has been rejected, so the only
            # thing established is that the ceiling is at least this high.
            self.outcome = 'lower_bound_only'
            return self._finish(self.clean_scale, 'max_scale_reached_cleanly', step)
        self.rung += 1
        self._enter(nxt, step)
        return self._act(CLIMB, 'rung_clean', scale=nxt, after=SAVE_CLEAN)

    def _classify(self):
        """(verdict, reason) for the CURRENT rung, ignoring residence.

        Reads streaks the observers maintain; forms no new evidence itself, so
        calling it every step is free and changes nothing.
        """
        if self._adverse_streak >= self.persistence:
            return BOUNDARY, f'coherence_adverse_x{self._adverse_streak}'
        if self._margin is None:
            return PENDING, f'pool_{len(self._pool.rows)}_of_{self.min_readings}'
        gap, bar = self._margin
        if self._below_streak >= self.persistence:
            return BOUNDARY, f'ray_margin_{gap:+.3f}dex_x{self._below_streak}'
        if gap >= 0.0:
            return CLEAN, f'ray_margin_{gap:+.3f}dex'
        # Below the setpoint but not yet CONFIDENTLY below: the band between
        # -bar and 0. Section 7's `pending` -- extend the dwell, do not climb.
        return PENDING, f'ray_margin_{gap:+.3f}dex_x{self._below_streak}'

    # ------------------------------------------------------------- finishing

    def _reject(self, verdict, why, step):
        """This rung is rejected. Whether the RAMP is over depends on whether a
        clean rung exists to fall back to.

        `outcome` is set only on the paths that FINISH. A ramp still descending
        has rejected a rung and has no outcome yet; stamping one mid-descent made
        a running ladder report `boundary` while it was still working, which is
        the sort of thing a reader takes at face value.
        """
        self._record(verdict, why, step)
        if self.clean_scale is not None:
            self.outcome = verdict
            cruise = self.clean_scale / (self.factor ** self.cruise_backoff_rungs)
            return self._finish(cruise, why, step, action=ROLLBACK)
        # The FIRST rung was already rejected: there is no clean checkpoint to
        # restore, so descend until one is established rather than finishing on a
        # rate nothing has passed (section 7).
        #
        # ONE RUNG IS THE FLOOR, NOT THE STEP. Section 7 says "descend
        # geometrically", and taken literally that wastes a full residence per
        # rung re-learning something the reading already said. Measured on
        # elj/mipcas phase 1: rungs 0 and 1 both returned `below_range` at the
        # BOTTOM of the alpha grid -- margin exactly -0.602 dex, i.e. log10(4),
        # the censored statement "alpha* < 1, so the optimum is at most a quarter
        # of this rate". Descending 1.5x against evidence of at least 4x needed
        # ~9 rungs and 9,000 steps to cover a cut the pooled estimator made in
        # one move.
        #
        # So the descent takes the LARGER of one geometric rung and what the
        # margin licenses. Safe by direction: down is the safe way to be wrong,
        # the bound is evidence the run already paid for, and a rung is still the
        # minimum so a marginal rejection behaves exactly as before.
        #
        # NOT named `step`. It was, for one revision, and it shadowed this
        # method's `step` PARAMETER -- so `_enter(nxt, step)` stamped the rung's
        # entry with the multiplier (0.25) instead of the step index. Every
        # subsequent rung then read `resident = step_ind - 0`, the residence gate
        # never bound, and the ramp finished on a spurious `no_evidence` timeout.
        # Invisible to the unit tests, which use short residences and
        # persistence 1 so the BOUNDARY short-circuit fires first; caught on
        # elj/mipcas, which is what section 7's "validate on a real run before
        # giving it authority" is for.
        shrink = 1.0 / self.factor
        if self._margin is not None:
            shrink = min(shrink, 10.0 ** self._margin[0])
        nxt = self.scale * shrink
        if nxt < self.min_scale:
            self.outcome = 'floor_reached'
            return self._finish(nxt, 'min_scale_reached_without_a_clean_rung', step)
        self.rung += 1
        self._enter(nxt, step)
        return self._act(DESCEND, f'initial_rung_rejected_{why}', scale=nxt)

    def _finish(self, cruise, why, step, action=FINISH):
        self.state = 'done'
        self.cruise_scale = max(self.min_scale, min(self.max_scale, float(cruise)))
        return self._act(action, why, scale=self.cruise_scale, after=FINISH)

    def _record(self, verdict, why, step):
        est = None
        if self._pool is not None and self._pool.rows:
            est = self._pool.estimate(
                incumbent=math.log10(self.scale) + math.log10(self.alpha_target))
        self.history.append({
            'rung': self.rung, 'scale': self.scale, 'verdict': verdict,
            'reason': why, 'entered_at': self.entered_at, 'left_at': int(step),
            'residence': int(step) - self.entered_at,
            'n_readings': 0 if self._pool is None else len(self._pool.rows),
            'log_opt': None if est is None else est['log_opt'],
            'se': None if est is None else est['se'],
        })

    def _act(self, action, reason, scale=None, after=None):
        self._last = {'action': action, 'reason': reason, 'rung': self.rung,
                      'scale': self.scale if scale is None else scale,
                      'then': after, 'state': self.state}
        return self._last

    # -------------------------------------------------------------- budgeting

    def projected_cost(self, from_scale: float, to_scale: float) -> dict:
        """Steps the ramp will spend getting from one rate to another.

        Section 8e: state the budget BEFORE building, because unlike the frozen
        trials this replaced, a rejected rung's steps are genuinely discarded on
        rollback. `discarded` is that rung -- one, since one rejection is enough
        to bracket the boundary and the design forbids continuing past it.
        """
        span = abs(math.log(max(to_scale, 1e-30) / max(from_scale, 1e-30)))
        rungs = max(1, int(math.ceil(span / math.log(self.factor))) + 1)
        return {'rungs': rungs, 'residence': self.residence,
                'steps': rungs * self.residence,
                'discarded': self.residence,
                'residence_bound_by': self.residence_bound_by}

    # -------------------------------------------------------------- reporting

    _ACTION = {DWELL: 0, SAVE_CLEAN: 1, CLIMB: 2, DESCEND: 3, ROLLBACK: 4,
               FINISH: 5}
    _VERDICT = {CLEAN: 0, PENDING: 1, BOUNDARY: 2, HARD_FAILURE: 3}

    def report(self) -> dict:
        """EVIDENCE per rung, not just the selected rate. Section 7 requires rung
        LR, residence, ray bounds, classification reasons and the selected cruise
        LR to be logged; a ramp that reported only its answer could not be
        audited afterwards, which is the whole complaint against its
        predecessor."""
        out = {'ramp/state': float({'idle': 0, 'running': 1, 'done': 2}[self.state]),
               'ramp/rung': float(self.rung),
               'ramp/rungs_clean': float(len(
                   [h for h in self.history if h['verdict'] == CLEAN])),
               'ramp/residence': float(self.residence)}
        if self.scale is not None:
            out['ramp/scale'] = float(self.scale)
        if self._last:
            out['ramp/action'] = float(self._ACTION.get(self._last['action'], -1))
        if self.history:
            last = self.history[-1]
            out['ramp/last_verdict'] = float(self._VERDICT.get(last['verdict'], -1))
            if last['log_opt'] is not None:
                out['ramp/last_log_opt'] = float(last['log_opt'])
                out['ramp/last_se'] = float(last['se'])
        if self.cruise_scale is not None:
            out['ramp/cruise_scale'] = float(self.cruise_scale)
        return out
