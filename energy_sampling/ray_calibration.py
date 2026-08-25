"""
Periodic ray calibration -- the LR sensor.

WHAT IT MEASURES. Along the real optimizer step d = theta_after - theta_before,
define theta(alpha) = theta_before + alpha*d. alpha is a dimensionless multiplier
on the step actually taken, and alpha* = argmin_alpha L(theta(alpha)) is the
factor the step was off by: alpha* = lr* / lr. Target 1 would sit exactly on the
one-step optimum; we deliberately target well above it (see ALPHA_TARGET).

THE PRIMITIVE, and the reason this module is small. Write the loss along the ray
as L(alpha) = L0 + a*alpha + b*alpha^2/2, so alpha* = -a/b. Then for alpha > 0

    L(alpha) - L(0) = alpha*(a + b*alpha/2) < 0   <=>   alpha < -2a/b = 2*alpha*

and therefore, substituting alpha -> 2*alpha,

    L(2*alpha) < L(0)   <=>   alpha < alpha*.                              (*)

So the sign of a SINGLE PAIRED DIFFERENCE between two measured losses answers
"is alpha* above this alpha". No parabola fit, no ratio, no derivative estimate,
no censoring: the quantity whose sign we need is measured directly. Evaluating
(*) across a doubling grid brackets alpha* and simultaneously answers the control
question "is alpha* above alpha_target".

That replaces the entire previous apparatus (per-probe parabola fits, a status
taxonomy, windowed medians, IQRs, censoring bookkeeping, quorum fractions). Those
existed because alpha* was estimated as a RATIO whose denominator -- the second
difference -- straddles zero: measured on this route its per-probe sd/mean is
~3.3, so ~40% of single fits come back concave and unusable. The ratio is never
formed here.

PAIRING IS THE WHOLE TRICK. L(2*alpha) and L(0) are evaluated on the SAME
sub-batch, so per-batch loss level -- which varies by hundreds of nats across
conditions -- cancels exactly in the difference. Sub-batches are the replicates,
and their spread gives the confidence interval directly.

COST. len(alphas) forward passes per sub-batch, n_sub sub-batches, once per
`period` train steps. Forward-only over STORED trajectories: no resampling and no
energy calls. Sub-batches are drawn and released one at a time -- the loop is
sub-batch outer, alpha inner -- so peak memory is one sub-batch regardless of
n_sub, and the pairing is exact.

SCOPE. d is taken over POLICY parameters only (decision D26 option b). The flow
(Z) head is LR-pinned separately and is held at its post-step value throughout, so
it contributes an identical constant to every evaluation and drops out of every
difference.

WHAT IS SCORED, AND WHY IT IS THE COMPOSITE. `loss_fn` may return one number or
a mapping of named components. The optimizer takes ONE step along the FUSED
gradient, so the loss alpha* has to be measured against is the frac-weighted sum
the step descends: a per-branch alpha* is the optimum for a direction nobody
took. `composite` is therefore the component the controller reads. The per-branch
components are bracketed too and reported as DIAGNOSTICS -- each branch is
already evaluated at each alpha to form the sum, so they are free, and branch
disagreement (fwd wanting 4x while replay wants 0.25x) is worth surfacing.
"""

import contextlib
import math
from collections.abc import Mapping

import numpy as np

import torch

#: The component `alpha_star`/`status` describe and the controller acts on.
#: Everything else `loss_fn` returns is a free diagnostic.
COMPOSITE = 'composite'


class RayCalibration:
    """
    Sensor only. `arm()` immediately before the optimizer step, `measure()`
    immediately after. Every parameter touched is restored bitwise.

    `alphas` MUST be closed under doubling over the tested range: to test (*) at
    alpha it needs the loss at 2*alpha. A grid point is TESTED when its double is
    also on the grid, so the default {0,1,2,4,8} tests alpha* against {1, 2, 4}
    and the top point buys only the contrast for 4. (The comment here used to
    say {0.5, 1, 2, 4}; 0.5 is not a grid point, so no paired difference for it
    is ever formed -- verified against `_bracket`, which iterates the grid
    itself.) The grid's LOWEST tested alpha is therefore its second entry: a rate
    hotter than that reads as `below_range`, a bound, and is never
    extrapolated.
    """

    def __init__(self,
                 params,
                 alphas=(0.0, 1.0, 2.0, 4.0, 8.0),
                 n_sub: int = 8,
                 period: int = 500,
                 t_crit: float = 2.0,
                 enabled: bool = False,
                 log_grid: bool = False,
                 dual_score: bool = False):
        self.enabled = bool(enabled)
        self.log_grid = bool(log_grid)
        # SCORE EVERY ALPHA A SECOND WAY, and log the disagreement.
        #
        # The ray rates a step by replaying the STORED trajectory at each alpha.
        # For a branch that trains on stored trajectories (`replay`) that is the
        # trained objective exactly. For one whose live draw re-samples a
        # backward path every step (`bwd` with dataset/prior sampling) it is
        # not, and the two can rank step sizes differently -- the stored path
        # was drawn from P_B at theta_before, so it goes off-distribution as
        # alpha moves theta.
        #
        # With this on, each sub-batch is scored at every alpha BOTH ways and
        # both are bracketed. `rayfresh/*` reports the second reading and
        # `rayfresh/gap_octaves` their distance. NOTHING ACTUATES ON IT: the
        # controller is still handed the replayed reading, so a run with this
        # on is a run with a diagnostic, not a run with a different controller.
        self.dual_score = bool(dual_score)
        self.last_fresh = {}
        self.alphas = tuple(float(a) for a in alphas)
        if 0.0 not in self.alphas:
            raise ValueError('alphas must include 0.0 -- it is the baseline of every contrast')
        self.n_sub = max(2, int(n_sub))
        self.period = max(1, int(period))
        if self.period % 10 != 0:
            # Metrics are drained on a 10-step clock; a period that is not a
            # multiple of it aliases, so some calibrations would never appear in
            # the log and `raycal/*` would silently describe a subset.
            raise ValueError(f'ray_calibration.period ({self.period}) must be a multiple of 10')
        # Calibrations that were due but produced nothing (no optimizer step to
        # measure, or no paired-evaluable batch available). Counted and logged:
        # a sensor that is silently never running and one that is satisfied look
        # identical from its outputs alone, which is the failure this module's
        # predecessor logged three separate times.
        self.n_skipped = 0        # genuine failures: nothing to draw
        self.n_deferred = 0       # due but no optimizer step yet -- retried next step
        # Due, but the controller had already decided the reading would be
        # discarded, so nothing was drawn. Counted and reported for the same
        # reason as the two above: a probe that is deliberately silent and one
        # that is broken must not look alike from the logs.
        self.n_refused = 0
        self.refuse_reason = ''
        self.skip_reason = ''
        self._last_done = None    # period index of the last COMPLETED calibration
        self._armed_at = None
        # Two-sided significance for a paired difference across n_sub replicates.
        # Fixed rather than configurable: it is a statistical convention, not a
        # tuning knob, and every value anyone would pick lies between 1.8 and 2.5.
        self.t_crit = float(t_crit)
        self._params = [p for p in params if p.requires_grad]
        self._before = None
        self.last = {}

    # ------------------------------------------------------------------ timing

    def due(self, step_ind: int) -> bool:
        """True from the moment a calibration falls due until one COMPLETES --
        deliberately NOT `step_ind % period == 0`.

        A calibration can only measure a step the optimizer actually took, and
        under gradient accumulation most steps are not that. An exact-modulo
        trigger therefore depends on a coincidence between two unrelated clocks:
        a boundary k*period is a stepping step only when the accumulation cycle
        DIVIDES period. When it does not, most boundaries are lost, and when the
        two are coprime almost none survive -- measured over 3000 steps at period
        500: cycle 4 (divides) gave 6 calibrations, cycle 8 gave 3, cycle 3 gave
        2, and cycle 7 gave ZERO. peak_scale then never moves and the run trains
        at its seed while the config says it is adaptive. Latching until a
        measurement succeeds removes the coincidence rather than making it less
        likely: all four cases give the full 6.

        Accumulation turns on whenever the batch cannot be run in one pass, so an
        OOM shrink is enough to land on an unlucky cycle.

        Cost of latching: one parameter-sized clone per step while pending, and
        pending normally lasts less than one accumulation cycle."""
        if not self.enabled or step_ind <= 0:
            return False
        idx = step_ind // self.period
        if self._last_done is None:
            self._last_done = idx      # first sight: wait for the NEXT boundary
            return False
        return idx > self._last_done

    @torch.no_grad()
    def arm(self, step_ind: int) -> bool:
        """Snapshot policy params before the optimizer step."""
        if not self.due(step_ind):
            return False
        self._before = [p.detach().clone() for p in self._params]
        self._armed_at = int(step_ind)
        return True

    def refuse(self, reason: str, step_ind: int) -> bool:
        """Consume a due calibration WITHOUT measuring it.

        Called instead of `arm` when the controller has already decided the
        reading would be discarded (`LRController.calibration_refusal`). Skips
        the parameter clone and, far more importantly, the `n_sub` replay draws:
        those consume RNG that nothing restores, so a calibration that changes
        no learning rate still moves every subsequent training step. That was
        F-039.

        THE PERIOD IS CONSUMED, exactly as a completed calibration consumes it.
        This is what keeps the applied path unchanged. `due` latches from the
        moment a calibration falls due until one completes, so a refusal that
        left `_last_done` alone would leave the latch pending through warmup and
        then fire on the FIRST step after it -- earlier than the boundary the
        old code fired on. Advancing it here means the same boundaries are
        consumed either way, so the first calibration that is actually applied
        lands on the same step as before this change.

        One edge remains, and it is stated rather than hidden: pre-change, a
        `deferred_no_step` (mid-accumulation, or a non-finite gradient) at a
        boundary left the latch pending and let a later step consume a LATER
        period index. A refusal cannot detect that case, because detecting it is
        what the clone is for. The two can therefore disagree by one period
        index only if a deferral straddles the end of warmup.
        """
        if not self.due(step_ind):
            return False
        self.n_refused += 1
        self.refuse_reason = str(reason)
        self._last_done = int(step_ind) // self.period
        self._before = None
        return True

    # ----------------------------------------------------------------- measure

    def defer(self, reason: str, step_ind: int) -> bool:
        """Let a due calibration stand WITHOUT consuming its period.

        The third outcome, distinct from `arm` and `refuse`: the reading is
        wanted and nothing is wrong, the INPUTS are not there yet -- a larder
        still filling after a stage transition, typically. `due` stays latched
        so the calibration fires as soon as they are, and no parameter clone is
        spent in the meantime.

        Counted as a deferral rather than a skip for the reason every other
        counter in this class exists: a sensor that is quietly waiting and one
        that is broken must not look alike from its outputs.
        """
        if not self.due(step_ind):
            return False
        self.n_deferred += 1
        self.skip_reason = str(reason)
        return True

    @torch.no_grad()
    def measure(self, draw_fn, loss_fn, loss_fn_alt=None) -> dict | None:
        """
        draw_fn() -> one fresh sub-batch. Called n_sub times.
        loss_fn(batch) -> the policy loss on `batch` at the CURRENT params,
        either as a float or as a mapping of named components (see the module
        docstring). A float is read as the composite alone.
        Neither may mutate training state: no tracker updates, no buffer writes,
        no log Z updates.

        loss_fn_alt, optional, DIAGNOSTIC: a second scorer over the same batch
        and the same alphas, bracketed identically and reported as
        `rayfresh/*`. Nothing actuates on it -- the return value is still the
        primary reading. Used to score each alpha the way training actually
        does (fresh backward path) alongside the way the ray does it
        (replayed), so the two can be compared on the same batches at the same
        parameters. It is called under `_rng_pinned`, so it neither breaks the
        pairing across alphas nor moves the run's random stream.

        Returns the reading, or None if there was no step to measure.
        """
        if self._before is None:
            return None
        after = deltas = None
        try:
            deltas = [p.detach() - b for p, b in zip(self._params, self._before)]
            sq = sum(float(d.pow(2).sum()) for d in deltas)
            if not math.isfinite(sq) or sq == 0.0:
                # Mid-accumulation or a non-finite gradient: there is no step to
                # rate. NOT a failure -- `due` stays latched and we retry on the
                # next step, so this costs a clone, not a calibration.
                self.n_deferred += 1
                self.skip_reason = 'deferred_no_step'
                return None

            # Hold theta_after explicitly and parameterise the ray from it:
            #   theta(alpha) = after + (alpha - 1) * delta
            # Restoring is then a copy of `after` rather than a recomputation of
            # before + delta, which is not bitwise identical in floating point.
            after = [p.detach().clone() for p in self._params]
            self._before = None

            def _set(alpha):
                for p, a, d in zip(self._params, after, deltas):
                    p.copy_(a) if alpha == 1.0 else p.copy_(a).add_(d, alpha=alpha - 1.0)

            # Sub-batch OUTER, alpha INNER: holds one sub-batch at a time and
            # keeps every contrast within a single batch.
            losses = []            # [k][i] -> {component: loss}, sub-batches x alphas
            fresh = [] if loss_fn_alt is not None else None
            if fresh is not None:
                # Cleared up front so a stale reading cannot be re-logged if this
                # calibration fails to produce one -- the exact way a dead sensor
                # comes to look like a live one.
                self.last_fresh = {}
            with contextlib.ExitStack() as stack:
                # Entered ONCE around the whole loop, so the run's random stream
                # is restored exactly once and the sub-batches are independent
                # draws from it in between.
                couple = (stack.enter_context(_rng_pinned())
                          if fresh is not None else None)
                for _k in range(self.n_sub):
                    batch = draw_fn()
                    if batch is None:
                        self._skip('no_batch')
                        break
                    row = []
                    for a in self.alphas:
                        _set(a)
                        row.append(_as_components(loss_fn(batch)))
                    losses.append(row)
                    if fresh is not None:
                        # SAME batch, SAME alphas, SAME parameters -- only the
                        # scoring rule differs, so any gap between the two
                        # readings is the scoring rule and nothing else.
                        reseed = couple()
                        frow = []
                        for a in self.alphas:
                            reseed()
                            _set(a)
                            frow.append(_as_components(loss_fn_alt(batch)))
                        fresh.append(frow)
                    del batch
            if len(losses) < 2:
                self._skip('too_few_subbatches')
                return None
            reading = self._summarise(losses, math.sqrt(sq))
            if fresh is not None and len(fresh) >= 2:
                self.last_fresh = self._reading(fresh, math.sqrt(sq)) or {}
            if reading is not None and self._armed_at is not None:
                # Satisfied THIS period. Keyed on the armed step, so a calibration
                # delayed across a boundary consumes that boundary too rather than
                # firing again immediately.
                self._last_done = self._armed_at // self.period
            return reading
        finally:
            if after is not None:
                for p, a in zip(self._params, after):
                    p.copy_(a)                     # exact restore to theta_after
            self._before = None

    def _skip(self, reason):
        self.n_skipped += 1
        self.skip_reason = reason

    # --------------------------------------------------------------- statistics

    def _bracket(self, table, K):
        """One component's [sub-batch][alpha] numbers -> (status, alpha*, tests).

        For each tested alpha with 2*alpha on the grid, the paired differences
        D_k = L_k(2*alpha) - L_k(0) are one sample per sub-batch. By (*),
        mean(D) < 0 <=> alpha* > alpha. Significance is an ordinary one-sample t
        on those differences -- they are i.i.d. across sub-batches by construction.
        """
        idx = {a: i for i, a in enumerate(self.alphas)}
        i0 = idx[0.0]
        tests = []
        for a in self.alphas:
            if a <= 0.0 or (2.0 * a) not in idx:
                continue
            j = idx[2.0 * a]
            D = [row[j] - row[i0] for row in table]
            if not all(math.isfinite(v) for v in D):
                continue
            m = sum(D) / K
            var = sum((v - m) ** 2 for v in D) / max(K - 1, 1)
            se = math.sqrt(var / K)
            t = m / se if se > 0 else (0.0 if m == 0 else math.copysign(math.inf, m))
            tests.append({'alpha': a, 'mean': m, 't': t,
                          'above': t < -self.t_crit,     # significant descent -> alpha* > a
                          'below': t > self.t_crit})     # significant rise    -> alpha* < a
        if not tests:
            return None

        # Bracket. `above` at alpha is evidence alpha* > alpha; `below` is
        # evidence alpha* < alpha. Take the strongest of each; if they contradict
        # (lo >= hi) the readings are inconsistent and the calibration is void.
        lo = max((x['alpha'] for x in tests if x['above']), default=None)
        hi = min((x['alpha'] for x in tests if x['below']), default=None)
        if lo is not None and hi is not None and lo >= hi:
            status, alpha_star = 'inconsistent', float('nan')
        elif lo is not None and hi is not None:
            status, alpha_star = 'bracketed', math.sqrt(lo * hi)
        elif lo is not None:
            status, alpha_star = 'above_range', lo      # a BOUND; never extrapolated
        elif hi is not None:
            status, alpha_star = 'below_range', hi
        else:
            status, alpha_star = 'unresolved', float('nan')
        return {'status': status, 'alpha_star': alpha_star, 'lo': lo, 'hi': hi,
                'tests': tests}

    def _summarise(self, losses, step_norm):
        """`_reading`, and it is THE one the controller reads (`self.last`)."""
        got = self._reading(losses, step_norm)
        if got is not None:
            self.last = got
        return got

    def _reading(self, losses, step_norm):
        """
        Turn the [sub-batch][alpha] table of COMPONENT dicts into a reading.

        PURE -- stores nothing. The dual-scoring diagnostic brackets its second
        table with exactly this code, so the two readings are comparable by
        construction rather than by a parallel implementation that could drift.

        `composite` carries the decision; every other component is bracketed the
        same way and reported as a diagnostic. A component missing from some
        sub-batch is dropped rather than imputed -- an absent branch is not a
        branch at loss zero.
        """
        K = len(losses)
        names = [k for k in losses[0][0]
                 if all(k in cell for row in losses for cell in row)]
        if COMPOSITE not in names:
            return None
        per = {}
        for name in names:
            table = [[cell[name] for cell in row] for row in losses]
            got = self._bracket(table, K)
            if got is not None:
                per[name] = got
        primary = per.get(COMPOSITE)
        if primary is None:
            return None

        agg = [sum(row[i][COMPOSITE] for row in losses) / K
               for i in range(len(self.alphas))]
        return {
            'alpha_star': primary['alpha_star'],
            'status': primary['status'],
            'lo': primary['lo'], 'hi': primary['hi'],
            'n_sub': K,
            'step_norm': step_norm,
            'tests': primary['tests'],
            'components': per,
            'aggregate': dict(zip(self.alphas, agg)),
            'losses': losses,
        }

    # -------------------------------------------------------------------- log

    # Explicit codes, never tuple.index(): a positional encoding silently
    # re-maps every historical run's logs the moment the tuple is reordered.
    _STATUS = {'unresolved': 0, 'bracketed': 1, 'above_range': 2,
               'below_range': 3, 'inconsistent': 4, 'warmup': 5}

    #: Reasons a calibration was refused before drawing. Explicit codes for the
    #: same reason `_STATUS` has them: a positional encoding re-maps every
    #: historical run's logs the moment the list is reordered.
    #:
    #: `branch_refused` is the structural one: an ACTIVE branch of this stage
    #: cannot be replay-scored (LarderScorer.refusal), so the composite the
    #: optimizer step descends cannot be formed. It refuses the period rather
    #: than deferring because nothing about it changes with time -- see
    #: `defer` for the case that does.
    _REFUSAL = {'': 0, 'warmup': 1, 'no_larder': 2, 'branch_refused': 3,
                'no_active_branch': 4}

    def report(self) -> dict:
        """Loggable view. Empty until the first calibration.

        `raycal/refused` is load-bearing, not decoration. With the pre-draw gate
        in place a probe inside warmup produces NO measurement at all, so every
        `raycal/*` measurement key is absent -- which is indistinguishable from
        a sensor that was never wired up unless something counts the refusals.
        That confusion is the exact failure this module's predecessor logged
        three separate times."""
        if not self.enabled:
            return {}
        if not self.last:
            return ({'raycal/skipped': float(self.n_skipped),
                     'raycal/deferred': float(self.n_deferred),
                     'raycal/refused': float(self.n_refused),
                     'raycal/refused_reason': float(
                         self._REFUSAL.get(self.refuse_reason, -1))}
                    if (self.n_skipped or self.n_deferred or self.n_refused)
                    else {})
        r = self.last
        out = {
            'raycal/skipped': float(self.n_skipped),
            'raycal/deferred': float(self.n_deferred),
            'raycal/refused': float(self.n_refused),
            'raycal/refused_reason': float(
                self._REFUSAL.get(self.refuse_reason, -1)),
            'raycal/alpha_star': r['alpha_star'],
            'raycal/status': float(self._STATUS.get(r['status'], -1)),
            'raycal/n_sub': float(r['n_sub']),
            'raycal/step_norm': r['step_norm'],
        }
        # THE PER-ALPHA GRID IS NOT LOGGED (owner decision, 2026-08-23). It used
        # to publish L_/dL_/t_ per tested alpha -- one key per rung per
        # statistic, so a doubling grid of 8 alphas emitted ~20 keys and made
        # this family the largest LR block in the run, burying `alpha_star` and
        # `status`, the only two anything actuates on.
        #
        # WHAT THAT COSTS, stated rather than discovered: the grid was the raw
        # evidence behind the bracket, so a reading that looks wrong can no
        # longer be re-derived from the run alone -- `status` and `refused_reason`
        # are now the whole audit trail. `r['tests']` and `r['aggregate']` are
        # still built and still returned by `measure`, so a caller that wants the
        # bracket has it; only the logging is gone.
        #
        # Per-branch brackets, FREE: each branch is already evaluated at each
        # alpha to form the composite. Diagnostic only -- nothing actuates on
        # them -- but branch disagreement is the reading that says the fused
        # step is a compromise rather than a consensus.
        # ...and the grid itself, OPT-IN (`ray_calibration.log_grid: true`). Off
        # by default for the parsimony reason above, but a reading that looks
        # wrong cannot be re-derived without it -- so it is one key away rather
        # than a code change, for exactly the diagnosis this was needed for.
        if self.log_grid:
            for x in r['tests']:
                tag = f"{x['alpha']:g}".replace('.', 'p')
                out[f'raygrid/dL_{tag}'] = x['mean']
                out[f'raygrid/t_{tag}'] = max(-99.0, min(99.0, x['t']))
            for a, v in r['aggregate'].items():
                out[f"raygrid/L_{f'{a:g}'.replace('.', 'p')}"] = v
        for name, c in r.get('components', {}).items():
            if name == COMPOSITE:
                continue
            out[f'raycal/branch/alpha_star_{name}'] = c['alpha_star']
            out[f'raycal/branch/status_{name}'] = float(self._STATUS.get(c['status'], -1))
        # THE DIAGNOSTIC PAIR (`ray_calibration.dual_score: true`). Same batches,
        # same alphas, same parameters, scored under a freshly sampled backward
        # path instead of the stored one. Read `gap_octaves` first: it is
        # log2(fresh alpha* / replayed alpha*), so 0 says the two objectives rank
        # step sizes identically and the replayed reading is sound on this stage,
        # while a large positive value says the replayed reading is calling for a
        # smaller step than the trained objective wants. NaN either side means
        # that reading did not resolve, which is itself informative -- a fresh
        # pass that never brackets is a different failure from one that brackets
        # somewhere else.
        f = self.last_fresh
        if f:
            out['rayfresh/alpha_star'] = f['alpha_star']
            out['rayfresh/status'] = float(self._STATUS.get(f['status'], -1))
            a_f, a_r = f['alpha_star'], r['alpha_star']
            if (math.isfinite(a_f) and math.isfinite(a_r)
                    and a_f > 0.0 and a_r > 0.0):
                out['rayfresh/gap_octaves'] = math.log2(a_f / a_r)
            for name, c in f.get('components', {}).items():
                if name == COMPOSITE:
                    continue
                out[f'rayfresh/branch/alpha_star_{name}'] = c['alpha_star']
                out[f'rayfresh/branch/status_{name}'] = float(
                    self._STATUS.get(c['status'], -1))
            if self.log_grid:
                for a, v in f['aggregate'].items():
                    out[f"rayfresh/L_{f'{a:g}'.replace('.', 'p')}"] = v
        return out


@contextlib.contextmanager
def _rng_pinned():
    """Common random numbers across the alpha grid, and no net RNG consumption.

    TWO jobs, both required for a fresh-sampling pass to mean anything.

    COUPLING, WITHIN A SUB-BATCH AND ONLY THERE. The bracket differences
    L(2a) - L(0) are taken within a sub-batch. If every evaluation drew its own
    backward path, that difference would be dominated by path noise rather than
    by the parameter change, and the t-test would report noise at high
    confidence -- precisely the failure that restricted this sensor to stored
    trajectories in the first place. `couple()` hands back a `reseed` to call
    before each alpha, making the sampled path a deterministic function of theta
    so the contrast isolates the step.

    Each sub-batch gets its OWN seed, and that two-level shape is the whole
    reason this is not a single rewind. n_sub replicates sharing one seed would
    share their path noise entirely, so the spread ACROSS sub-batches -- the
    denominator of the t-statistic -- would carry none of it: the variance would
    be understated and every bracket overconfident. A diagnostic built to expose
    a sampling defect must not quietly commit one.

    NEUTRALITY. The replayed pass consumes no RNG by construction, which is what
    makes probed and unprobed runs comparable (`Trainer._probe_dealer`). A fresh
    pass consumes plenty. Restoring on exit preserves that property, so turning
    the diagnostic on does not itself change the run it is measuring.
    """
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    npy = np.random.get_state()

    def couple():
        # Drawn FROM the outer stream, so it differs sub-batch to sub-batch --
        # and the outer stream is restored below, so drawing it costs nothing.
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())

        def reseed():
            torch.manual_seed(seed)          # CPU and every CUDA device
            np.random.seed(seed)

        return reseed

    try:
        yield couple
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(npy)


def _as_components(value):
    """A loss_fn result as a {component: float} dict.

    A bare float is the composite alone, which is what every caller predating
    the composite scoring returns.
    """
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    return {COMPOSITE: float(value)}
