import math

import numpy as np


class ModeBalanceController:
    """
    Phase-3 mode-frac balancer: nudges fwd_frac/bwd_frac/replay_frac toward
    whichever of {bwd, fwd, replay} the current under/over/Z-error coverage
    metrics call for, and anneals the controller's thresholds once all three
    stay within bounds for long enough. Holds a reference to its owning
    Modeller for the metrics it reads and the frac/streak state it mutates.
    """

    def __init__(self, modeller):
        self.modeller = modeller

    def step(self):
        under, over, zerr = self._get_controller_metrics()
        under, over, zerr = self._lookahead_controller_metrics(under, over, zerr)
        # what the controller actually thresholds on, exposed for logging --
        # the raw metrics alone can make state selection look impossible
        self.modeller.controller_projections = {'under': under, 'over': over, 'zerr': zerr}
        state = self._select_controller_state(under, over, zerr)
        self._nudge_mode_fracs(state)

    def _get_controller_metrics(self):
        m = self.modeller
        under = m.metric_tracker.get('bwd', 'under_coverage')
        over = m.metric_tracker.get('fwd', 'over_coverage')
        # richer, per-condition-aware replacement for the old single-batch-mean
        # jensen_z_err (quick_tb_stats): RMS over conditions with enough evidence
        # to trust (ConditionLogZTracker.rms_z_lag), so a badly-miscalibrated
        # minority of conditions can't hide behind an otherwise-fine average the
        # way a plain mean would let it.
        condition_log_z = getattr(m, 'condition_log_z', None)
        zerr = condition_log_z.rms_z_lag()

        if zerr is None:
            zerr = float("inf")  # bootstrap toward fwd
        if under is None:
            under = 1.0 # forward gets attention first
        if over is None:
            over = 0.0  # do not demand replay before fwd stats exist

        return under, over, zerr

    def _lookahead_controller_metrics(self, under, over, zerr):
        """
        Extrapolate each controller metric a few controller ticks into the
        future by EMA-smoothing its trend and projecting forward linearly, so
        _select_controller_state reacts to where a metric is heading rather
        than where it currently sits. Without this, mode-frac mass keeps
        moving at full speed right up to a threshold crossing and overshoots
        the optimal balance before the (lagging) raw metric ever reflects it.
        """
        m = self.modeller
        ctrl = m.args.controller
        lookahead = m.controller_lookahead
        return (
            self._log_ema_lookahead(lookahead['under'], under, ctrl),
            self._log_ema_lookahead(lookahead['over'], over, ctrl),
            self._log_ema_lookahead(lookahead['zerr'], zerr, ctrl),
        )

    @staticmethod
    def _log_ema_lookahead(state, value, ctrl):
        """
        under/over/zerr are already EMAs coming out of metric_tracker, so
        `value` is already a smoothed level - only its trend (the per-tick
        delta) needs its own EMA here. All three metrics are nonnegative nats
        living on a log scale (raw values run orders of magnitude above
        thresholds of order 1), so the trend is tracked in log space and the
        projection is multiplicative: a steep descent predicts "k-fold lower
        in `horizon` ticks", never "below zero". The linear projection this
        replaces could tunnel through the entire threshold band and flip the
        controller state while the raw metric was still far out of bounds.
        state: {'level', 'trend'} dict, mutated in place (persisted via
        MODELLER_STATE_DEFAULTS so it survives checkpoint reloads); 'level'
        holds the previous raw value, 'trend' is in log-nats per tick.
        Returns value * exp(horizon * trend), the predicted value
        `ctrl.lookahead_horizon` controller ticks ahead.
        """
        if not math.isfinite(value):
            return value
        if state['level'] is None:
            state['level'] = value
            return value

        eps = 1e-3  # keeps the log finite when a metric sits at its floor of 0
        log_delta = math.log(max(value, eps)) - math.log(max(state['level'], eps))
        trend_alpha = getattr(ctrl, 'lookahead_trend_alpha', 0.1)
        horizon = getattr(ctrl, 'lookahead_horizon', 5)
        state['trend'] = trend_alpha * log_delta + (1 - trend_alpha) * state['trend']
        state['level'] = value

        # cap the projection at e^±3 (~20x) of the current value: beyond that the
        # extrapolation is guessing, and the cap also defuses stale linear-unit
        # trends restored from pre-log-space checkpoints
        exponent = min(max(horizon * state['trend'], -3.0), 3.0)
        return value * math.exp(exponent)

    def _select_controller_state(self, under, over, zerr):
        """
        Priority order: Z convergence > undercoverage repair > replay repair >
        global tightening. Z is checked first, above even backward undercoverage
        repair -- under/over_coverage are themselves computed from TB residuals
        relative to log_Z (quick_tb_stats), so they're only meaningful once Z is
        reasonably calibrated; chasing coverage repairs against a miscalibrated
        Z means measuring with a broken ruler. This is the state/step-size
        selection logic most likely to change later, so it's kept isolated from
        the metric plumbing and frac math around it.
        """
        m = self.modeller
        ctrl = m.args.controller
        if zerr > ctrl.zerr_threshold:
            m.controller_anneal_streak = 0
            return "fwd"
        elif under > ctrl.under_threshold:
            m.controller_anneal_streak = 0
            return "bwd"
        elif over > ctrl.over_threshold:
            m.controller_anneal_streak = 0
            return "replay"
        else:
            # joint condition (all three metrics within threshold) satisfied this tick;
            # require it to hold for `anneal_patience` consecutive ticks before tightening
            # the margins, since a single tick is too susceptible to metric noise
            m.controller_anneal_streak += 1
            if m.controller_anneal_streak >= getattr(ctrl, 'anneal_patience', 1):
                self._anneal_controller_thresholds()
                m.controller_anneal_streak = 0
            return None

    def _anneal_controller_thresholds(self):
        ctrl = self.modeller.args.controller
        if ctrl.under_threshold > ctrl.min_threshold:
            ctrl.under_threshold *= ctrl.decay_rate
        if ctrl.over_threshold > ctrl.min_threshold:
            ctrl.over_threshold *= ctrl.decay_rate
        if ctrl.zerr_threshold > ctrl.zerr_min_threshold:
            ctrl.zerr_threshold *= ctrl.decay_rate

    def _nudge_mode_fracs(self, state):
        if state is not None:
            m = self.modeller
            ctrl = m.args.controller
            probs = np.array([m.fwd_frac, m.bwd_frac, m.replay_frac], dtype=float)
            probs /= probs.sum()

            idx = {"fwd": 0, "bwd": 1, "replay": 2}[state]

            m_floor = ctrl.min_mode_frac  # requires m_floor < 1/3
            free = 1.0 - 3.0 * m_floor  # total mass available above the floors

            # excess space: x_i = p_i - m_floor, with x_i >= 0 and sum(x) = free
            excess = np.clip(probs - m_floor, 0.0, None)
            s = excess.sum()
            excess = excess * (free / s) if s > 0.0 else np.full(3, free / 3.0)

            # EMA toward the one-hot on the boosted mode
            excess *= 1.0 - ctrl.beta
            excess[idx] += ctrl.beta * free

            m.fwd_frac, m.bwd_frac, m.replay_frac = m_floor + excess
