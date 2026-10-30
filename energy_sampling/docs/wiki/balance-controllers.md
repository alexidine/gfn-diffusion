# Balance controllers

*Drift: **M** (mixed). Verified against commit `78de930`, 2026-10-04. Sources at the end.*

In a fused stage the three branch losses, forward, backward and replay, are combined into one loss with weights called *fracs*. The fracs are loss weights, not sample fractions: every active branch runs a full batch every step regardless of its weight ([loss-composition](loss-composition.md)). A stage's `balance` block names a controller that moves those weights during the stage, on a ten-step tick: the lexicographic kind nudges all three, and the other four kinds move two of them against each other. This page describes the controllers that exist, what each one does mechanically, and the fixed points and rails that follow from the arithmetic. It does not say which one to use.

The four split kinds (proportional, constraint, ratio, gated_ramp) share two conventions. A `pinned` map holds any mode outside the split at a fixed weight and is re-asserted on every tick, so the pin survives a resume. The two moving modes share a *pair mass*, the sum of their two fracs, and the controller moves the split of that mass; the pair mass itself is whatever the stage entered with. The lexicographic kind reads no `pinned` map; its per-mode floors are the stage's `min_fracs`, with `cfg:controller.min_mode_frac` for a mode the stage does not name. Under every kind, a branch whose frac falls below `deactivate_threshold` is skipped, not merely down-weighted, so a controller that drives a weight to zero switches that branch off. On a `cfg:controller.refresh_every` step a branch below the threshold whose metrics the stage reads is still evaluated, detached and at zero weight ([loss-composition](loss-composition.md)). On a stage with `cfg:stage.fwd_rollout_every` above zero the forward branch runs on the rollout cadence in place of the frac test and the refresh, and carries weight on such a step only when its frac is at or above the threshold (`train.py::Modeller._fwd_gates`, [rollout-cadence-and-dose](rollout-cadence-and-dose.md)).

## The kinds

### lexicographic

The rule-based controller, and the kind a `balance` block takes when it names none. An ordered list of rules, each naming a metric and a bar, with the bar in one of three forms: an absolute level (`above`), a floor (`below`), or a running best (`relative: best`). On each tick the rules are checked in order, the first one that fires supplies a boost vector, and the fracs are nudged toward that boost by an EMA with rate `controller.beta`. If no rule fires, the nudge is toward `default_boost`. An anneal event fires after `cfg:controller.anneal_patience` consecutive ticks on which no rule fired: it tightens the threshold of each `above` rule that carries `anneal`, and raises each `energy_config` coefficient named in the optional `anneal_coeffs` toward its `target` ([lambda-annealing](lambda-annealing.md)). `anneal_cooldown_steps` suppresses rule evaluation for a window after stage entry so no running best is captured from the entry transient; after each anneal event it holds the clean streak at zero for the same window while the rules are still evaluated. `cfg:stage.coeff_schedule` ramps `energy_config` coefficients on a step clock under any kind, and a coefficient named in both it and `anneal_coeffs` is refused at parse. This is the kind the canonical config's `var_conditioning` stage uses, with `default_boost` and every rule boost set to the same mix; in that configuration the nudge target equals the current mix, so the fracs do not move (`protocol.py::StageProtocol._balance_tick`).

### proportional

A static map from two metrics to a split. Each side's drive is its metric's excess over its target, clipped at zero; under `drive: absolute`, the default, the target split is the ratio of the two drives; the live split moves toward the target by an EMA with rate `alpha`:

$$
s_i = \max(m_i - t_i, 0), \qquad
\text{target} = \frac{s_a}{s_a + s_b}, \qquad
\text{share}_a \leftarrow (1-\alpha)\,\text{share}_a + \alpha\,\text{target}.
$$

With `drive: relative` the drive is $\max(m_i/t_i - 1, 0)$ instead, which makes the two sides dimensionless, and the target is the idle split tilted by the two drives:

$$
\text{target} = \frac{w_a(1+s_a)}{w_a(1+s_a) + w_b(1+s_b)}.
$$

The idle weights $w$ are the block's `default_boost` when it declares one and the stage's entry `fracs` otherwise. When both drives are zero the target is the idle split $w_a/(w_a+w_b)$ under either drive. The split is held where it is in three cases: either metric is absent; both idle weights are zero under `drive: relative`; both idle weights and both drives are zero under `drive: absolute`. The target is clamped to $[\text{floor}, 1-\text{floor}]$, and an optional `max_fracs` map of absolute per-mode ceilings clamps both the target and the share after the EMA. Because the map is static, its equilibrium is set by where the two targets sit relative to the metrics' floors: a target below a metric's floor gives that side permanent drive, and a target above the metric's operating range gives it none. An optional `anneal` block scales the whole target vector down after a streak of quiet ticks, one scale for both targets so their ratio is preserved (`protocol.py::StageProtocol._proportional_tick`).

### constraint

A one-sided integrator in the logit of one side's share. One mode is declared `constrain`; its metric has a hard bar. The other is best-effort. Both drives are relative shortfalls clipped at zero, and the integrator moves by their weighted difference:

$$
d_c = \max\!\left(\tfrac{m_c}{b_c} - 1, 0\right), \quad
d_r = \max\!\left(\tfrac{m_r}{b_r} - 1, 0\right), \quad
\theta \leftarrow \text{clip}\big(\theta + \text{clip}(g\,(d_r - p\,d_c), \pm\text{max\_step}),\ \theta_{lo}, \theta_{hi}\big),
$$

with $\text{share}_r = \sigma(\theta)$. `priority` $p$ is a gain multiple, so the constrained side wins contests without a discontinuity. The integrator holds when both drives are zero, so its equilibrium is a property of the bars rather than of their ratio. A best-effort bar set below its metric's floor never zeroes its drive, and $\theta$ walks to a bound; `protocol/cs_at_bound` reports $-1$ at the lower bound, $+1$ at the upper and $0$ between. `bounds` on at least one split mode are required at parse (`protocol.py::StageProtocol._constraint_tick`).

### ratio

An integrator in the logit of the numerator mode's share, on the log-ratio of the two metrics against a setpoint:

$$
e = \log\frac{m_n}{m_d} - \log(\text{setpoint}), \qquad
\theta \leftarrow \text{clip}\big(\theta + \text{clip}(g\,k\,e, \pm\text{max\_step}),\ \theta_{lo}, \theta_{hi}\big).
$$

Neither drive is clipped, so both sides always contribute and the error changes sign rather than vanishing. `numerator` is declared explicitly, because an inverted ratio is a sign flip on the whole loop. `converge_floor` fades the gain $k$ to zero as the larger of the two metrics approaches it. Two arithmetic properties matter. If the denominator sits at a floor $f$, the error reduces to $\log m_n - \log(f \cdot \text{setpoint})$, a fixed bar on the numerator. If the numerator is a positive half of the same residual field as the forward error, the two move together and $e$ carries the forward error's variation. Both are visible in the metric definitions, not in any run (`protocol.py::StageProtocol._ratio_tick`).

### gated_ramp

One sensor, two motions, in share space. A `ramp` mode's share drifts up by `up` per tick while a guard sensor sits at or below `bar`, and moves down by `down` per tick while the sensor exceeds it. `bounds` are hard rails, so a floor on the guard mode is a guarantee rather than an equilibrium. The canonical `equilibration` stage used it, with replay as the ramp mode and the backward branch as the guard, until 2026-10-30; that stage now declares no `balance` block and holds `fracs` at bwd 0.5, replay 0.5.

The ratchet is what gives the ramp a level reference. An optional second metric, `ratchet_metric`, has its stage minimum tracked as `gr_best`. The ramp is vetoed while the level sits above that best by more than `ratchet_tol`, and the veto is a Schmitt trigger: it trips at `best + ratchet_tol` and releases only once the level is back inside `best + ratchet_release_tol`. With both tolerances equal it is a plain threshold; at zero tolerance the ramp is released only on a tick whose level is at or below the stage minimum recorded so far. While the veto holds, the ramp mode's share moves down by `down` per tick, including on ticks where the guard sensor has no value yet. On a tick where `ratchet_metric` has no finite value the veto is not applied, and the trip state is kept for the next tick that has one.

`ratchet_cooldown_steps` freezes both fracs and records no best for a window after stage entry, anchored on the stage's first tick so it is correct on a resume. The reference is a running minimum that never relaxes upward: once tripped, the veto holds until the level is back within `ratchet_release_tol` of the lowest level recorded in the stage, and for the rest of the stage if it never is. The retired key `ratchet_z_bar` is refused at parse rather than ignored (`protocol.py::StageProtocol._gated_ramp_tick`).

## Shared quantities

| quantity | meaning | where it lives |
|---|---|---|
| pair mass | sum of the two moving fracs under the four split kinds; the controller splits it, never changes it | the split tick functions |
| `bounds` | absolute frac bounds per mode under constraint, ratio and gated_ramp, intersected into a share interval; a ceiling on one mode is a floor on the other | `_share_interval`, `_gated_ramp_tick`, `_parse_bounds` |
| `pinned` | fixed weight for the third mode under the four split kinds, re-asserted every tick | `_parse_pinned`, the split tick functions |
| `deactivate_threshold` | frac below which a branch is skipped, per stage or the global fallback | `fused_train_step` |
| tick | ten train steps | `StageProtocol.tick`, called from `train.py::Modeller.train` when the step index is a multiple of ten |

`controller.beta` is read only by the lexicographic nudge. Under `kind: proportional`, neither the parse of the block nor the tick reads `bounds` or the stage's `min_fracs`; `floor` clamps the target share on both sides and `max_fracs` caps each split mode's frac. Under `kind: constraint`, `ratio` and `gated_ramp`, `min_fracs` is folded into `bounds` at parse and an inconsistent pair fails to load, as does a lower bound below the stage's own `deactivate_threshold`. Under `lexicographic`, the fracs a stage declares are a seed for the nudge, not an allocation.

## Retired

The buffer freshness servo, `buffer_servo`, was retired on 2026-09-09 and its controller deleted. Its block is refused at parse with a message pointing at the residence and churn keys it used to scale ([replay-buffer](replay-buffer.md)). The proportional, constraint and ratio kinds are parsed and run but are not used by any stage in the canonical config.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:stage.balance.kind`, `cfg:stage.balance.pinned`, `cfg:stage.balance.bounds`, `cfg:stage.balance.metrics`, `cfg:stage.balance.targets`, `cfg:stage.balance.drive`, `cfg:stage.balance.alpha`, `cfg:stage.balance.floor`, `cfg:stage.balance.anneal`, `cfg:stage.balance.constrain`, `cfg:stage.balance.bars`, `cfg:stage.balance.gain`, `cfg:stage.balance.priority`, `cfg:stage.balance.max_step`, `cfg:stage.balance.numerator`, `cfg:stage.balance.setpoint`, `cfg:stage.balance.converge_floor`, `cfg:stage.balance.ramp`, `cfg:stage.balance.guard`, `cfg:stage.balance.metric`, `cfg:stage.balance.bar`, `cfg:stage.balance.up`, `cfg:stage.balance.down`, `cfg:stage.balance.ratchet_metric`, `cfg:stage.balance.ratchet_tol`, `cfg:stage.balance.ratchet_release_tol`, `cfg:stage.balance.ratchet_cooldown_steps`, `cfg:stage.balance.rules`, `cfg:stage.balance.default_boost`, `cfg:stage.balance.max_fracs`, `cfg:stage.balance.anneal_coeffs`, `cfg:stage.balance.anneal_cooldown_steps`, `cfg:stage.coeff_schedule`, `cfg:stage.min_fracs`, `cfg:stage.deactivate_threshold`, `cfg:stage.fwd_rollout_every`, `cfg:controller.beta`, `cfg:controller.min_mode_frac`, `cfg:controller.anneal_patience`, `cfg:controller.refresh_every`, `cfg:controller.deactivate_threshold`.

Code: `protocol.py::Stage._parse_balance`, `protocol.py::Stage._parse_pinned`, `protocol.py::Stage._parse_bounds`, `protocol.py::StageProtocol.tick`, `protocol.py::StageProtocol._share_interval`, `protocol.py::StageProtocol._balance_tick`, `protocol.py::StageProtocol._proportional_tick`, `protocol.py::StageProtocol._constraint_tick`, `protocol.py::StageProtocol._ratio_tick`, `protocol.py::StageProtocol._gated_ramp_tick`, `protocol.py::StageProtocol._nudge_mode_fracs`, `train.py::Modeller.fused_train_step`, `train.py::Modeller._fwd_gates`, `train.py::Modeller.train`.

## Sources

The code above, read at the stamped commit, and the canonical config's `var_conditioning` balance block (`equilibration` declares none). Memory files that describe these controllers (project_constraint_balance_controller, project_proportional_controller_and_buffer_fixes, project_gated_ramp_ratchet_pins_replay, project_ratio_controller_rails_on_floored_denominator, project_replay_freshness_servo) were used to locate the code, not as evidence; one of them describes the buffer servo as live, which it no longer is.
