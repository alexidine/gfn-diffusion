# Under-coverage metric family

*Drift: **T** (theory). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The coverage statistics are one-sided moments of the trajectory-balance residual, computed per batch by `utils.py::quick_tb_stats` and published under the branch's namespace. This page defines each from the code, separates the three quantities they mix (the $\log Z$ level, the batch composition, the within-batch spread), and records how an empty window and the metric tracker's EMA treat each. The residual and its Huber loss are on [trajectory-balance](trajectory-balance.md); what a branch's residual can and cannot see is on [loss-composition](loss-composition.md); which controller reads which, and what it does with the reading, is on [balance-controllers](balance-controllers.md).

## The residual and its two halves

`quick_tb_stats` forms `resid = (log_pf + log_Z) - (log_pb + log_r)`, which is $r = \log Z - \log w$ for the per-trajectory log importance weight $\log w = \log R + \log P_B - \log P_F$. Both halves of $r$ are taken raw, not centred:

$$n^- = \min(r, 0), \qquad n^+ = \max(r, 0).$$

`over_coverage` is the unweighted RMS of $n^+$, $\sqrt{\overline{(n^+)^2}}$; `under_coverage_uniform` is the same on $n^-$. Both are in nats. A negative residual is a row whose $\log w$ exceeds $\log Z$: reward-times-backward mass the forward policy is not placing.

## The reward ramp: which rows enter the statistic

`under_coverage` differs from `under_coverage_uniform` only in the weights. `train.py::Modeller._ramp_params` reads `cfg:buffers.prior_buffer.ramp_floor` and `cfg:buffers.prior_buffer.ramp_width`, both in energy units of depth $d = E - E^\star_c$ above the sample's own condition's best known energy, and validates $0 < \texttt{ramp\_width} \le \texttt{ramp\_floor}$, raising otherwise; legacy `ramp_floor_range`/`ramp_knee_range` pairs are converted to `(floor, floor - knee)` and pass the same validation. `train.py::Modeller._reward_ramp_kwargs` translates that into reward space against the per-condition anchor maximum $M_c$ from `buffer.py::_per_condition_max`, giving `reward_floor` $= M_c - \texttt{ramp\_floor}$ per sample, with the buffer-wide maximum as fallback for a condition holding no anchors. With no anchor buffer, an empty one, no `condition_id`, or an unconfigured ramp it returns `reward_floor=None, ramp_width=None`, the uniform-RMS fallback.

Inside `quick_tb_stats` the weight is

$$w_i = \mathrm{clamp}\!\left(\frac{\log R_i - \texttt{reward\_floor}_i}{\texttt{ramp\_width}},\,0,\,1\right),$$

zero at or below the floor, saturating at one `ramp_width` above it. `under_coverage` is $\sqrt{\sum_i (w_i/\sum_j w_j)\,(n_i^-)^2}$: self-normalised, so the depth of the low-reward tail does not enter. `over_coverage` takes no weights; the code comment states the positive half stays uniform. `ramp_ess_frac` is the Kish effective sample size of $w$ as a batch fraction, $(\sum w)^2 / (n \sum w^2)$.

## The ladder: level, composition, spread

Three statistics share the negative-tail RMS and differ only in what each row is scored against.

`under_coverage` scores against $\log Z$. Since $r = \log Z - \log w$, a shift of $\log Z$ by $\delta$ shifts every residual by $\delta$: the statistic is Z-anchored and monotone non-increasing in $\log Z$, and `over_coverage` monotone non-decreasing. The `_forgetting_sensor` docstring states that a fill moving $\log Z$ by 4 nats moves it by about 4.

`relative_under` replaces $\log Z$ with the batch's own empirical normaliser: `neg_rel = (z_jensen_ref - log_w).clamp(max=0)`, where `z_jensen_ref` is each row's own condition's unweighted group mean of $\log w$ when `condition_id` is given and the single pooled batch mean otherwise. The ramp weights are applied to the squares as above. $\log Z$ does not appear, so the statistic is Z-invariant.

`relative_under` scores only ramp-qualifying rows but centres on all of them, so numerator and reference run over different populations. The code states the consequence algebraically: with a fraction $f$ of ramp-zero rows sitting $\Delta$ below the scored population, the centre lands at $\mu_{\text{scored}} - f\Delta$, and every scored row picks up a $+f\Delta$ offset before the one-sided clamp, so the statistic carries a floor of about $f\Delta$ even with a perfectly fit scored population. $f$ is set by buffer knobs: anchor top-up rate, churn, `cfg:buffers.prior_buffer.weighted_bwd_beta`, purge.

`relative_under_wcen` moves the centring onto the same weights, per group: `z_wcen_g = wdot_g / wsum_g`, falling back to the unweighted group mean where a group's weights sum to zero. $f$ then cancels between numerator and reference, leaving the spread among the rows the ramp scores. The code records that this is not a strict improvement, because $\Delta$ is itself a level-free defect, and that both are reported so their gap reads the composition. With no ramp configured the weights are uniform and `relative_under_wcen` is set equal to `relative_under` by construction.

The three carry, in order: level plus composition plus spread; composition plus spread; spread.

## The period-matched difference

`train.py::Modeller._forgetting_sensor` writes `<channel>_rise150` for each channel in `Modeller._FORGETTING_CHANNELS`, which is `('under_coverage', 'relative_under')`, producing `under_coverage_rise150` and `relative_under_rise150`. The value is the mean of the channel over the last `Modeller._UC_WINDOW_STEPS` = 150 steps minus the mean over the 150 before. Samples are kept as `(step, value)` pairs in `Modeller._uc_hist`, so the window is in training steps and not in calls; the sensor runs from `_update_rolling`, which `train.py::Modeller.record_fused_substep_losses` calls on every tenth trained substep of a type and on every untrained refresh-only substep. Non-finite values are skipped before the append. The key is written only when the history is full, defined as the oldest kept sample being within one observed stride of 300 steps old; until then it is absent rather than zero or nan. The sensor runs for `sub_type == 'bwd'` only.

## Empty windows, absent keys, and the EMA

The outcome when the ramp scores nothing differs by key.

- `under_coverage`, `relative_under` and `relative_under_wcen` are `nan` when `total`, the sum of the ramp weights, is zero: no row in the batch cleared its condition's floor.
- `ramp_ess_frac` is `0.0` in that same batch, its numerator being $(\sum w)^2$ and its denominator clamped away from zero.
- `ramp_ess_frac` is omitted entirely when the ramp is unconfigured, which the code comment distinguishes from a value of 1.0 meaning a ramp wide enough to score everything.
- `under_coverage_uniform` and `over_coverage` are always present and never nan from this path.

`utils.py::MetricTracker.update` computes `alpha = 1 - exp(-dt/period)` from the step gap and skips any non-finite value with `continue`, before the value write and before the `written_at` stamp. A nan batch therefore holds both the EMA and the freshness stamp `utils.py::MetricTracker.written_step` reports: a reader sees the previous value with no new-measurement mark.

The docstring of `quick_tb_stats` states the EMA criterion for its control family: the members are per-sample means and never ratios. `buffer.py` states the algebra a ratio fails, $\mathbb{E}[A/B] \ne \mathbb{E}[A]/\mathbb{E}[B]$. Against that criterion the members differ. Every member of the coverage ladder is a square root of a per-sample mean rather than a per-sample mean: the root is concave, so the EMA of the per-batch RMS sits at or below the RMS of the pooled squares, with equality only when the per-batch values are equal. `ramp_ess_frac` is a ratio whose denominator is small exactly when the weights are concentrated. `<channel>_rise150` is linear in its inputs, being a difference of two means, but it is already a window statistic over 300 steps and the tracker smooths it a second time. `slope_err` and `intercept_err`, from the same function, are a covariance over a variance and fail the criterion.

## What each is blind to

[loss-composition](loss-composition.md) derives the bound: with $\delta = \log Q - \log P$ for the policy's trajectory law $Q = P_F$ and the target's $P = R P_B / Z$, both proper, Markov's inequality on $\mathbb{E}_Q[e^{-\delta}] = 1$ and $\mathbb{E}_P[e^{+\delta}] = 1$ gives $Q(\delta < -m) \le e^{-m}$ and $P(\delta > +m) \le e^{-m}$. Here $\delta = r$ exactly. A forward-sampled batch is exponentially unlikely in $m$ to contain a residual at $-m$, so `fwd/under_coverage` is computed over a half of the field the draw itself suppresses at rate $e^{-m}$, and `fwd/over_coverage` is the half that draw can see. A backward batch draws terminals from the prior buffer and paths from $P_B$, the target-side measure, so the blindness reverses. The replay branch draws stored forward rollouts and inherits $Q$'s blindness at intake.

Within the visible half each member is additionally blind by construction. `relative_under` and `relative_under_wcen` contain no $\log Z$ and cannot report a level gap of any size. `under_coverage` contains the level and cannot separate a level gap from a spread. `relative_under_wcen` removes the composition term $f\Delta$, a real over-weighting of ramp-zero rows relative to anchor-sourced ones. `over_coverage` is unweighted and reports rows the ramp scores at zero on equal terms with the rest. `ramp_ess_frac` counts weights only and carries no residual information.

## Which controller reads which

The table lists each metric key the canonical config names, the config key it is declared under, and the controller that consumes it.

| key | read as | consumer in the canonical config |
|---|---|---|
| `bwd/under_coverage_rise150` | `cfg:stage.balance.metric` | `gated_ramp` guard sensor, terminal stage |
| `bwd/under_coverage` | `cfg:stage.balance.ratchet_metric` | `gated_ramp` level reference, terminal stage |
| `bwd/relative_under_wcen`, `fwd/over_coverage` | `cfg:stage.balance.metrics` | `ratio`, which the canonical terminal stage no longer declares |
| any of them | `cfg:stage.exit` and gate terms | resolved through `protocol.py::StageProtocol._resolve` |

Names resolve as `direction/metric` against the tracker EMA in `_resolve`; `gates/` and `eval/` prefixes read other stores. The consuming tick is `protocol.py::StageProtocol._gated_ramp_tick` or `::StageProtocol._ratio_tick`, both reached from `::StageProtocol._balance_tick`. What each does with the reading is on [balance-controllers](balance-controllers.md).

Two naming facts. The analysis package carries `bwd/under_coverage_wcen` in `analysis/keys.py` as a documented name that runs do not log, resolved there to `bwd/under_coverage`. And the docstring of `train.py::Modeller._bwd_retention_priority` refers to a `_bwd_under_center`, which no module defines.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.prior_buffer.ramp_floor`, `cfg:buffers.prior_buffer.ramp_width`, the legacy `ramp_floor_range`/`ramp_knee_range` pair, `cfg:buffers.prior_buffer.weighted_bwd_beta`, `cfg:conditional_worst_quantile`, `cfg:stage.balance.metric`, `cfg:stage.balance.ratchet_metric`, `cfg:stage.balance.metrics`.

Code: `utils.py::quick_tb_stats`, `::MetricTracker.update`, `::MetricTracker.get`, `::MetricTracker.written_step`, `::MetricTracker.rebase`; `train.py::Modeller._ramp_params`, `::Modeller._reward_ramp_kwargs`, `::Modeller._update_rolling`, `::Modeller.record_fused_substep_losses`, `::Modeller._forgetting_sensor`, `::Modeller._per_step_probe`, `::Modeller._eval_conditional_stats`, `::Modeller._bwd_retention_priority`; `buffer.py::_per_condition_max`; `protocol.py::StageProtocol._resolve`, `::StageProtocol._balance_tick`, `::StageProtocol._gated_ramp_tick`, `::StageProtocol._ratio_tick`; `analysis/keys.py`.

## Could be tooling

The nan, zero and omitted distinctions are checkable mechanically. Every key `quick_tb_stats` can emit is written into one `mets` dict plus two conditional blocks, so an AST walk over the function yields the key set and, per key, whether it is unconditional, guarded by the ramp configuration, or guarded by `total > 0`. Crossed against the metric names appearing in `cfg:stage.balance` and `cfg:stage.exit` across the committed configs, that names the controller inputs that can be absent and those that can be nan, without reading the tracker at runtime.

## Sources

The code above, read at the stamped commit, and the `buffers.prior_buffer` and terminal-stage `balance` blocks of the canonical config. Memory files located the code and were not used as evidence: project_under_coverage_actuator_ladder, project_under_coverage_gate_calibration, project_under_coverage_ramp_reparam, project_r2_ratio_metrics_dont_ema, project_relative_under_pooled_conditional_fix.
