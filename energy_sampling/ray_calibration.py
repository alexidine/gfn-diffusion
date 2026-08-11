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
"""

import math

import torch


class RayCalibration:
    """
    Sensor only. `arm()` immediately before the optimizer step, `measure()`
    immediately after. Every parameter touched is restored bitwise.

    `alphas` MUST be closed under doubling over the tested range: to test (*) at
    alpha it needs the loss at 2*alpha. The default {0,1,2,4,8} tests alpha* against
    {0.5, 1, 2, 4}.
    """

    def __init__(self,
                 params,
                 alphas=(0.0, 1.0, 2.0, 4.0, 8.0),
                 n_sub: int = 8,
                 period: int = 500,
                 t_crit: float = 2.0,
                 enabled: bool = False):
        self.enabled = bool(enabled)
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

    # ----------------------------------------------------------------- measure

    @torch.no_grad()
    def measure(self, draw_fn, loss_fn) -> dict | None:
        """
        draw_fn() -> one fresh sub-batch. Called n_sub times.
        loss_fn(batch) -> float, the policy loss on `batch` at the CURRENT params.
        Neither may mutate training state: no tracker updates, no buffer writes,
        no log Z updates.

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
            losses = []                            # [k][i] over sub-batches, alphas
            for _k in range(self.n_sub):
                batch = draw_fn()
                if batch is None:
                    self._skip('no_batch')
                    break
                row = []
                for a in self.alphas:
                    _set(a)
                    row.append(float(loss_fn(batch)))
                losses.append(row)
                del batch
            if len(losses) < 2:
                self._skip('too_few_subbatches')
                return None
            reading = self._summarise(losses, math.sqrt(sq))
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

    def _summarise(self, losses, step_norm):
        """
        Turn the [sub-batch][alpha] loss table into a bracket on alpha*.

        For each tested alpha with 2*alpha on the grid, the paired differences
        D_k = L_k(2*alpha) - L_k(0) are one sample per sub-batch. By (*),
        mean(D) < 0 <=> alpha* > alpha. Significance is an ordinary one-sample t
        on those differences -- they are i.i.d. across sub-batches by construction.
        """
        idx = {a: i for i, a in enumerate(self.alphas)}
        i0 = idx[0.0]
        K = len(losses)
        tests = []
        for a in self.alphas:
            if a <= 0.0 or (2.0 * a) not in idx:
                continue
            j = idx[2.0 * a]
            D = [row[j] - row[i0] for row in losses]
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

        agg = [sum(row[i] for row in losses) / K for i in range(len(self.alphas))]
        self.last = {
            'alpha_star': alpha_star,
            'status': status,
            'lo': lo, 'hi': hi,
            'n_sub': K,
            'step_norm': step_norm,
            'tests': tests,
            'aggregate': dict(zip(self.alphas, agg)),
            'losses': losses,
        }
        return self.last

    # -------------------------------------------------------------------- log

    # Explicit codes, never tuple.index(): a positional encoding silently
    # re-maps every historical run's logs the moment the tuple is reordered.
    _STATUS = {'unresolved': 0, 'bracketed': 1, 'above_range': 2,
               'below_range': 3, 'inconsistent': 4, 'warmup': 5}

    def report(self) -> dict:
        """Loggable view. Empty until the first calibration."""
        if not self.enabled:
            return {}
        if not self.last:
            return ({'raycal/skipped': float(self.n_skipped),
                     'raycal/deferred': float(self.n_deferred)}
                    if (self.n_skipped or self.n_deferred) else {})
        r = self.last
        out = {
            'raycal/skipped': float(self.n_skipped),
            'raycal/deferred': float(self.n_deferred),
            'raycal/alpha_star': r['alpha_star'],
            'raycal/status': float(self._STATUS.get(r['status'], -1)),
            'raycal/n_sub': float(r['n_sub']),
            'raycal/step_norm': r['step_norm'],
        }
        # Per-tested-alpha evidence: the mean paired difference and its t. These
        # ARE the measurement -- alpha_star is only their summary, so a reader who
        # distrusts the bracket can rebuild it from these.
        for x in r['tests']:
            tag = f"{x['alpha']:g}".replace('.', 'p')
            out[f'raycal/dL_{tag}'] = x['mean']
            out[f'raycal/t_{tag}'] = max(-99.0, min(99.0, x['t']))
        for a, v in r['aggregate'].items():
            out[f"raycal/L_{f'{a:g}'.replace('.', 'p')}"] = v
        return out
