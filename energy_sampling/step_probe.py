"""
Two-point step probe -- the LR sensor from docs/to_do_rebuild.md Part A.

WHAT IT MEASURES. Not gradient magnitude (which under Adam with tight clipping
is decoupled from the step actually taken -- see to_do_rebuild.md A2) but the
step itself: given the move the optimizer just made, was it the right SIZE?

Let  d = theta_after - theta_before  be the real optimizer step, momentum and
clipping included, and define the ray

    theta(alpha) = theta_before + alpha * d

so alpha is a dimensionless multiplier on the step actually taken: alpha=0 is
"didn't move", alpha=1 is "the step you took", alpha=2 is "twice that step,
same direction". Evaluate the loss at alpha in {0, 1/2, 1} on ONE frozen batch,
fit a parabola, and take alpha* = argmin:

    a      = 2 * (L0 + L1 - 2*Lh)                 (fitted curvature)
    alpha* = (3*L0 + L1 - 4*Lh) / (2 * a)

    alpha* ~ 1   step size correct
    alpha* < 1   overshot by 1/alpha*
    alpha* > 1   undershooting -- affirmative permission to grow

alpha* is dimensionless with a physical setpoint at 1.0, which is the whole
point: an absolute gradient bar has never transferred across T, energy
function, or clip setting, and a setpoint does not need to.

SCOPE (docs decision D26, option b). d is taken over POLICY parameters only.
The flow (Z) head is LR-pinned separately and the servo would not control it,
so including it would make alpha* rate a composite step. The head is held at
its post-step value for all three evaluations, so it contributes an identical
constant to L0/Lh/L1 and drops out of the fit entirely.

STAGE 1 (this module) is sensor + logging with NO actuation. It commits to no
design and cannot destabilise a run: every parameter it touches is restored
before returning, and it runs entirely under no_grad.

COST. Three forward passes over STORED trajectories per probe -- states and
energies are already in the batch, so no resampling and no energy calls. At the
default cadence this is a few percent of wall clock.
"""

import math
from collections import deque

import torch


# Relative floor on the second difference |L0 + L1 - 2*Lh|, as a fraction of the
# loss scale. Below this the parabola is flat to within float32 precision and
# alpha* is numerically meaningless -- see the `precision-limited` note in
# report(). float32 carries ~7 decimal digits, so a TB loss of ~1e3 resolves
# second differences no finer than ~1e-4 absolute; 1e-6 relative is a
# deliberately generous floor that only rejects the genuinely degenerate.
SECOND_DIFF_REL_FLOOR = 1e-6

# alpha* readings beyond this are recorded but not counted toward the windowed
# median -- a near-flat fit sends the argmin to +-inf, and one such reading
# would otherwise dominate any mean. The median is robust anyway; this is belt
# and braces for the dispersion statistic.
ALPHA_SANE_MAX = 64.0


def _fit_alpha_star(l0, lh, l1, span=1.0):
    """
    Fit L(alpha) = a*alpha^2 + b*alpha + c through (0,l0), (1/2,lh), (1,l1) and
    return (alpha_star, a, status).

    Derivation:
        c = l0
        l1 = a + b + l0                 =>  a +  b = l1 - l0
        lh = a/4 + b/2 + l0             =>  a + 2b = 4*lh - 4*l0
        subtract  =>  b = 4*lh - l1 - 3*l0
                      a = (l1 - l0) - b = 2*(l0 + l1 - 2*lh)

        alpha* = -b / (2a) = (3*l0 + l1 - 4*lh) / (4 * (l0 + l1 - 2*lh))

    so the whole fit turns on the second difference (l0 + l1 - 2*lh), which is
    a/2 and carries both the validity guards below.

    status is one of 'ok' | 'beyond' | 'nonfinite' | 'flat' | 'downward'.
    """
    if not all(math.isfinite(v) for v in (l0, lh, l1)):
        return float('nan'), float('nan'), 'nonfinite'

    second_diff = l0 + l1 - 2.0 * lh
    scale = max(abs(l0), abs(lh), abs(l1), 1e-30)

    # Flat / precision-limited: the three points are collinear to within what
    # float32 can resolve. Reported separately from 'downward' because the
    # remedies differ -- flat means the probe is under-resolved (raise the step
    # or the batch), downward means the local quadratic model is simply wrong.
    if abs(second_diff) < SECOND_DIFF_REL_FLOOR * scale:
        return float('nan'), 2.0 * second_diff, 'flat'

    # Downward-opening: the midpoint sits ABOVE the chord, so the parabola's
    # stationary point is a MAXIMUM. Condition is lh > (l0+l1)/2, i.e.
    # second_diff < 0 -- strictly weaker than "lh exceeds both endpoints",
    # which an earlier draft of the doc used and which would miss most
    # degenerate fits.
    #
    # BUT THE TWO SIGNS OF loss_delta_rel MEAN OPPOSITE THINGS, and collapsing
    # them was the defect that made the servo inert at low LR (measured
    # 2026-08-08: at lr ~1.4e-6 the probe returned 'downward' on 100% of fits,
    # so alpha* was always nan, bad_rate pinned at 1.0, and the loop could
    # never license the growth that was the entire point of seeding low).
    #
    #   l1 < l0  -- loss falls monotonically along the ray AND is accelerating
    #               downward, so the minimum lies BEYOND alpha = span. The
    #               quadratic model is not wrong, it is being fitted over a
    #               window too short to contain the basin. That is precisely
    #               "the step is too small", i.e. affirmative permission to
    #               grow, and it is the regime a deliberately-low LR seed puts
    #               the probe in. Report alpha* = span: a LOWER BOUND on the
    #               true argmin, which is the honest reading (the servo's own
    #               per-tick clip bounds how fast it may act on it).
    #   l1 >= l0  -- concave AND the step increased loss. No basin bracketed and
    #               no descent either; the local model really is wrong here.
    if second_diff < 0.0:
        if l1 < l0:
            return span, 2.0 * second_diff, 'beyond'
        return float('nan'), 2.0 * second_diff, 'downward'

    alpha = (3.0 * l0 + l1 - 4.0 * lh) / (4.0 * second_diff)
    return alpha, 2.0 * second_diff, 'ok'


class StepProbe:
    """
    Sensor only. Call arm() immediately before the optimizer step and measure()
    immediately after; measure() restores every parameter it touches.

    `params` must be the POLICY parameters (D26 option b) -- the same list the
    servo would eventually drive. Passing the flow head in here would silently
    change what alpha* means.
    """

    def __init__(self,
                 params,
                 cadence: int = 20,
                 window: int = 25,
                 enabled: bool = False,
                 span: float = 2.0):
        self.enabled = bool(enabled)
        self.cadence = max(1, int(cadence))
        # Evaluate at alpha in {0, span/2, span} rather than {0, 1/2, 1}.
        #
        # WHY (measured 2026-08-07, r2_wiring). The fit turns entirely on the
        # second difference, which scales as (arc length)^2 -- so a wider arc is
        # QUADRATICALLY better conditioned. It has to be, because the probe was
        # observed dying in phase 2: second_diff_rel fell to 8.7e-8, BELOW the
        # float32 resolution of ~1e-7, and fit_flat_rate climbed to 0.63 with
        # fit_ok_rate down to 0.25. The probe was rejecting three quarters of
        # its own readings.
        #
        # The driver is loss MAGNITUDE, not the step: a de-huberized (quadratic)
        # TB loss on ~25-nat residuals is ~625, and float32 resolves that to
        # ~6e-5, against a second difference of the same order. Huber's cap
        # actually kept the loss small enough to measure -- which is why the
        # de-huberized arm is where this showed up first, exactly opposite to
        # the Huber-suppression hypothesis I had going in.
        #
        # span=2 buys 4x. It still brackets the observed alpha* ~ 0.5 (which
        # sits at u = 0.25 of the arc), so the minimum stays interior.
        self.span = float(span)
        self._params = [p for p in params if p.requires_grad]
        self._before = None
        self._armed_at = None
        self.alpha_hist = deque(maxlen=int(window))
        # WINDOWED fit validity, for servo_reading(). The cumulative `counts`
        # below are a run-level summary; a controller needs to know whether the
        # fit is valid NOW, and a sensor that went bad 2000 steps ago is still
        # diluted in a run-long rate.
        self.status_hist = deque(maxlen=int(window))
        # status tallies are cumulative over the run, drained by report() into
        # rates -- a rising 'downward' or 'flat' rate voids the sensor
        # independently of what the alpha* values say (to_do_rebuild A3a.3).
        # 'nostep' and 'aborted' are not fit statuses: the probe returned or
        # raised before there was anything to fit.
        self.counts = {'ok': 0, 'beyond': 0, 'flat': 0, 'downward': 0,
                       'nonfinite': 0, 'nostep': 0, 'aborted': 0}
        self.last = {}

    def due(self, step_ind: int) -> bool:
        return self.enabled and (step_ind % self.cadence == 0)

    @torch.no_grad()
    def arm(self, step_ind: int) -> bool:
        """Snapshot policy params. Cheap clone, one parameter-sized buffer."""
        if not self.due(step_ind):
            return False
        self._before = [p.detach().clone() for p in self._params]
        self._armed_at = step_ind
        return True

    @torch.no_grad()
    def measure(self, loss_fn) -> dict | None:
        """
        loss_fn() -> float. MUST evaluate on one frozen batch (identical data at
        all three alpha) and MUST NOT mutate training state -- no tracker
        updates, no buffer writes, no log Z updates. It is called three times.

        Returns the reading dict, or None if the probe could not run.
        """
        if self._before is None:
            return None
        restore = None
        try:
            deltas = [p.detach() - b for p, b in zip(self._params, self._before)]

            # No step happened -- non-finite gradient, or mid-accumulation. Not
            # a sensor failure, so tallied separately from the fit statuses.
            sq = sum(float(d.pow(2).sum()) for d in deltas)
            if not math.isfinite(sq) or sq == 0.0:
                self.counts['nostep'] += 1
                return None

            def _set(alpha):
                for p, b, d in zip(self._params, self._before, deltas):
                    p.copy_(b).add_(d, alpha=alpha)

            # Hand _set to the finally block. From here on the parameters are
            # OFF theta_after between assignments, so any exception out of
            # loss_fn() must not leave them there: train.py catches
            # RuntimeError/ValueError around train_step, shrinks the batch and
            # CONTINUES, so an un-restored probe would train on from
            # theta_before or theta_before + span*delta (a 2x overshoot at the
            # default span) with nothing in the log to say so. CUDA OOM inside
            # the probe's extra forward is not hypothetical here -- grow-until-
            # OOM-then-cut is the accepted batch-sizing mode on this route.
            restore = _set

            # Three points at alpha in {0, span/2, span}. The fit is done in
            # normalised u = alpha/span (so the closed form is unchanged) and
            # alpha* = span * u* converts back to multiples-of-the-taken-step.
            s = self.span
            _set(0.0)
            l0 = float(loss_fn())
            _set(0.5 * s)
            lh = float(loss_fn())
            _set(s)
            l1 = float(loss_fn())
            _set(1.0)  # restore: bitwise theta_before + delta == theta_after

            # _fit_alpha_star works in normalised u = alpha/span, so a 'beyond'
            # verdict comes back as u = 1.0 and converts to alpha = span like
            # any other reading -- no special case at the call site.
            u, a, status = _fit_alpha_star(l0, lh, l1, span=1.0)
            alpha = u * s if math.isfinite(u) else u
            self.counts[status] += 1
            self.status_hist.append(status)
            if status in ('ok', 'beyond') and abs(alpha) <= ALPHA_SANE_MAX:
                self.alpha_hist.append(alpha)

            scale = max(abs(l0), abs(lh), abs(l1), 1e-30)
            self.last = {
                'alpha_star': alpha,
                'curvature': a,
                'status': status,
                'step_norm': math.sqrt(sq),
                'l0': l0, 'l_half': lh, 'l1': l1,
                # |second difference| / loss scale. If this sits near
                # SECOND_DIFF_REL_FLOOR the probe is PRECISION-LIMITED, not
                # measuring curvature -- the failure mode A3a did not
                # anticipate, and it voids alpha* as surely as a downward fit.
                'second_diff_rel': abs(l0 + l1 - 2.0 * lh) / scale,
                # Did the step REDUCE held-out loss at all? (l1-l0)/|l0| < 0 is a
                # descent step, > 0 means the step increased loss on data it was
                # not fitted to. Logged because a high downward-fit rate has two
                # very different explanations and this separates them: a step
                # that lands past the basin (l1 > l0, edge-of-stability /
                # catapult, normal per A7) versus a step direction along which
                # the surface is genuinely non-quadratic. dropout is 0 here, so
                # all three evaluations are deterministic and NEITHER can be
                # evaluation noise.
                'loss_delta_rel': (l1 - l0) / max(abs(l0), 1e-30),
                'span': s,
            }
            return self.last
        except BaseException:
            # The probe did not finish, so the parameters were somewhere on the
            # ray when it failed and the finally block below force-restores
            # them. Tallied because that restore is silent: without this an
            # arm that OOMs inside loss_fn() on every probe is indistinguishable
            # in the log from one where the probe never ran, and the fit rates
            # would keep reporting on a sensor that is no longer reading.
            # Re-raised immediately -- train.py's OOM handler still sees it.
            self.counts['aborted'] += 1
            raise
        finally:
            if restore is not None:
                # idempotent on the success path -- measure() already set 1.0
                restore(1.0)
            self._before = None
            self._armed_at = None

    def flush_window(self):
        """Drop every buffered reading, keeping the cumulative counts.

        Called when the LR regime changes under the probe -- exiting warmup, or a
        stage transition. The window is `window` probes deep at `cadence` steps
        apart (500 train steps at the defaults), so after such a change it keeps
        serving readings taken at a DIFFERENT learning rate for hundreds of
        steps, and alpha* is ~1/lr so those readings are biased by exactly the
        ratio of the two rates.

        Measured 2026-08-08 (`lr_aug08` b_descend): the servo correctly held
        through warmup, then took its first tick on a window still full of
        warmup-era readings (alpha_median 6.6, taken at an envelope-suppressed
        LR) and climbed 4.0e-4 -> 5.36e-4 -- a 34% overshoot in the wrong
        direction -- before the window refilled and it turned around. Holding a
        controller while its sensor keeps buffering is not holding it.
        """
        self.alpha_hist.clear()
        self.status_hist.clear()

    def _window_stats(self):
        """(median, n, iqr) over the alpha window, or None below 3 readings."""
        if len(self.alpha_hist) < 3:
            return None
        xs = sorted(self.alpha_hist)
        n = len(xs)
        med = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
        return med, n, xs[(3 * n) // 4] - xs[n // 4]

    def servo_reading(self):
        """
        The LR servo's input: (median alpha*, n, bad_rate), or None when cold.

        `bad_rate` is the WINDOWED fraction of probes whose fit was not usable
        -- flat (under-resolved), downward (the local quadratic model is wrong),
        or non-finite. A3a.3: a rising flat/downward rate voids the sensor
        independently of what the alpha* values say, so the controller needs it
        alongside the median rather than having to infer validity from the
        median's behaviour. 'nostep' and 'aborted' never reach status_hist --
        the probe returned or raised before there was a fit -- so they neither
        count as bad readings nor dilute the rate.

        'beyond' counts as USABLE, not bad. It is a one-sided reading (alpha*
        is only bounded below by span) but it is a correct one, and it is the
        reading that dominates at a low LR -- see _fit_alpha_star. Counting it
        as bad is what made the servo inert below ~1e-5.

        The MEDIAN is deliberate and so is the dispersion the servo pairs it
        with: measured within-run relative IQR is 0.5-1.0 (16 batt0807 runs),
        which is the wide branch of A4c -- servo-on-median, not a line search.
        At that spread the standard error of a 25-probe median is ~9% of the
        median, so a per-tick clip tighter than about +-15% spends most of the
        loop's authority on sampling noise.

        ⚠ THE MEDIAN IS CENSORED WHEN 'beyond' READINGS ARE COMMON, and it is
        censored DOWNWARD. A 'beyond' fit contributes exactly `span` -- a lower
        bound on alpha*, not an estimate of it -- so a window mixing bounds with
        real values reports something between the two. Measured 2026-08-08
        (lr_aug08 c_low, lr 1.56e-5): alpha_median read 3.5 where the 1/lr law
        predicts ~15, with fit_ok_rate only 0.43.

        For anyone MEASURING alpha*(lr) this is not harmless. **Read
        alpha_median together with fit_ok_rate, and treat a median taken below
        fit_ok_rate ~0.5 as a lower bound.**

        ⚠ AND `span` MUST EXCEED `target` WITH MARGIN, or the servo crawls out
        of a low LR. A censored window reports `span`, so the servo's multiplier
        there is exactly `span / target` -- not the clip. Measured 2026-08-08
        (lr_aug08 d_cal_below, span 2.0 against a calibrated target 1.87): the
        climb ran at **1.07x per tick instead of the 1.25 clip** for seven
        bins, then jumped to the full clip rate the moment fits turned `ok`.
        With target 1.0 the same window would have given 2.0/1.0 = 2.0, clipped
        to 1.25, and the crawl would not have happened.

        So the two knobs are coupled: **calibrating `target` upward shrinks the
        growth authority available in exactly the regime that needs it most.**
        Keep `span` at roughly 2x the intended target, or accept that a low seed
        takes thousands of steps to escape.
        """
        stats = self._window_stats()
        if stats is None or not self.status_hist:
            return None
        med, n, _ = stats
        bad = sum(1 for s in self.status_hist if s not in ('ok', 'beyond'))
        return med, n, bad / len(self.status_hist)

    def report(self) -> dict:
        """Loggable view. Empty until the first successful fit."""
        if not self.enabled:
            return {}
        total = sum(self.counts.values()) or 1
        out = {
            'lrprobe/fit_ok_rate': self.counts['ok'] / total,
            # a high beyond_rate is not a fault -- it is "the step is too small
            # to bracket the basin", i.e. the LR is below the probe's resolving
            # range. Read it together with fit_ok_rate: beyond -> ok as the
            # servo climbs is the signature of a working loop
            'lrprobe/fit_beyond_rate': self.counts['beyond'] / total,
            'lrprobe/fit_flat_rate': self.counts['flat'] / total,
            'lrprobe/fit_downward_rate': self.counts['downward'] / total,
            'lrprobe/nostep_rate': self.counts['nostep'] / total,
            'lrprobe/nonfinite_rate': self.counts['nonfinite'] / total,
            'lrprobe/aborted_rate': self.counts['aborted'] / total,
        }
        if self.last:
            out['lrprobe/alpha_star'] = self.last['alpha_star']
            out['lrprobe/curvature'] = self.last['curvature']
            out['lrprobe/step_norm'] = self.last['step_norm']
            out['lrprobe/second_diff_rel'] = self.last['second_diff_rel']
            out['lrprobe/loss_delta_rel'] = self.last['loss_delta_rel']
        stats = self._window_stats()
        if stats is not None:
            # The windowed MEDIAN is the servo's input; the IQR is the kill
            # switch. A3a.2: wide per-probe spread with a stable median is
            # acceptable -- kill only if the median itself wanders.
            med, n, iqr = stats
            out['lrprobe/alpha_median'] = med
            out['lrprobe/alpha_iqr'] = iqr
            out['lrprobe/alpha_n'] = n
        if self.status_hist:
            # MUST use the same usable-set as servo_reading(): this is the
            # logged mirror of the servo's own validity gate, and if the two
            # disagree the log misreports why the loop held.
            out['lrprobe/bad_rate_window'] = (
                sum(1 for s in self.status_hist if s not in ('ok', 'beyond'))
                / len(self.status_hist))
        return out
