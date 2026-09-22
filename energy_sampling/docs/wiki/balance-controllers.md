# Balance controllers

*Drift: **M** (mixed). Verified against commit `e17167e`, 2026-09-19. Sources at the end.*

In a fused stage the three branch losses, forward, backward and replay, are combined into one loss with weights called *fracs*. The fracs are loss weights, not sample fractions: every active branch runs a full batch every step regardless of its weight ([loss-composition](loss-composition.md)). A stage's `balance` block names a controller that moves two of those weights against each other during the stage, on a ten-step tick. This page describes the controllers that exist, what each one does mechanically, and the fixed points and rails that follow from the arithmetic. It does not say which one to use.

Every kind shares three conventions. A `pinned` map holds any mode at a fixed weight and is re-asserted on every tick, so the pin survives a resume. The two moving modes share a *pair mass*, the sum of their two fracs, and the controller moves the split of that mass; the pair mass itself is whatever the stage entered with. A branch whose frac falls below `deactivate_threshold` is skipped entirely, not merely down-weighted, so a controller that drives a weight to zero switches that branch off.

## The kinds

### lexicographic

The original rule-based controller. An ordered list of rules, each naming a metric and a bar, with the bar in one of three forms: an absolute level (`above`), a floor (`below`), or a running best (`relative: best`). On each tick the rules are checked in order, the first one that fires supplies a boost vector, and the fracs are nudged toward that boost by an EMA with rate `controller.beta`. If no rule fires, the nudge is toward `default_boost`. Optional `anneal_coeffs` tighten coefficients off a streak of clean ticks, and `anneal_cooldown_steps` suppresses rule evaluation for a window after stage entry so no running best is captured from the entry transient. This is the kind the current `var_conditioning` and terminal stages use, with every boost set to the same mix; in that configuration the nudge target equals the current mix, so the fracs do not move (`protocol.py::StageProtocol._balance_tick`).

### proportional

A static map from two metrics to a split. Each side's drive is its metric's excess over its target, clipped at zero; the target split is the ratio of the two drives; the live split moves toward it by an EMA with rate `alpha`:

$$
s_i = \max(m_i - t_i, 0), \qquad
\text{target} = \frac{s_a}{s_a + s_b}, \qquad
\text{share}_a \leftarrow (1-\alpha)\,\text{share}_a + \alpha\,\text{target}.
$$

With `drive: relative` the drive is $m_i/t_i - 1$ instead, which makes the two sides dimensionless. When both drives are zero the split is held where it is. Because the map is static, its equilibrium is set entirely by where the two targets sit relative to the metrics' floors: a target below a metric's floor gives that side permanent drive, and a target above the metric's operating range gives it none. An optional `anneal` block scales the whole target vector down after a streak of quiet ticks, one scale for both targets so their ratio is preserved (`protocol.py::StageProtocol._proportional_tick`).

### constraint

A one-sided integrator in the logit of one side's share. One mode is declared `constrain`; its metric has a hard bar. The other is best-effort. Both drives are relative shortfalls clipped at zero, and the integrator moves by their weighted difference:

$$
d_c = \max\!\left(\tfrac{m_c}{b_c} - 1, 0\right), \quad
d_r = \max\!\left(\tfrac{m_r}{b_r} - 1, 0\right), \quad
\theta \leftarrow \text{clip}\big(\theta + \text{clip}(g\,(d_r - p\,d_c), \pm\text{max\_step}),\ \theta_{lo}, \theta_{hi}\big),
$$

with $\text{share}_r = \sigma(\theta)$. `priority` $p$ is a gain multiple, so the constrained side wins contests without a discontinuity. The integrator holds when both drives are zero, so its equilibrium is a property of the bars rather than of their ratio. A best-effort bar set below its metric's floor never zeroes its drive, and $\theta$ walks to a bound; the fraction of ticks spent at a bound is reported as `cs_at_bound`. `bounds` are therefore required at parse (`protocol.py::StageProtocol._constraint_tick`).

### ratio

An integrator in the logit of the numerator mode's share, on the log-ratio of the two metrics against a setpoint:

$$
e = \log\frac{m_n}{m_d} - \log(\text{setpoint}), \qquad
\theta \leftarrow \text{clip}\big(\theta + \text{clip}(g\,k\,e, \pm\text{max\_step}),\ \theta_{lo}, \theta_{hi}\big).
$$

Neither drive is clipped, so both sides always contribute and the error changes sign rather than vanishing. `numerator` is declared explicitly, because an inverted ratio is a sign flip on the whole loop. `converge_floor` fades the gain $k$ to zero as the larger of the two metrics approaches it. Two arithmetic properties matter. If the denominator has a floor, the ratio falls as that metric improves toward the floor, so the setpoint moves further away as the model gets better. If the numerator is a positive half of the same residual field as the forward error, the two move together and $e$ carries the forward error's variation. Both are visible in the metric definitions, not in any run (`protocol.py::StageProtocol._ratio_tick`).

### gated_ramp

One sensor, two motions, in share space. A `ramp` mode's share drifts up by `up` per tick while a guard sensor sits at or below `bar`, and moves down by `down` per tick while the sensor exceeds it. `bounds` are hard rails, so a floor on the guard mode is a guarantee rather than an equilibrium. This is the kind the current `equilibration` stage uses, with replay as the ramp mode and the backward branch as the guard.

The ratchet is what gives the ramp a level reference. A second metric, `ratchet_metric`, has its stage minimum tracked as `gr_best`. The ramp is vetoed while the level sits above that best by more than `ratchet_tol`, and the veto is a Schmitt trigger: it trips at `best + ratchet_tol` and releases only once the level is back inside `best + ratchet_release_tol`. With both tolerances equal it is a plain threshold; at zero tolerance the ramp is released only on a tick that sets a new all-time low.

`ratchet_cooldown_steps` freezes both fracs and records no best for a window after stage entry, anchored on the stage's first tick so it is correct on a resume. The reference is a running minimum: a level recorded while the stage is still settling is one no later state returns to, and the veto then holds for the rest of the stage. The retired key `ratchet_z_bar` is refused at parse rather than ignored (`protocol.py::StageProtocol._gated_ramp_tick`).

## Quantities every kind shares

| quantity | meaning | where it lives |
|---|---|---|
| pair mass | sum of the two moving fracs; the controller splits it, never changes it | tick functions |
| `bounds` | absolute frac bounds per mode, intersected into a share interval; a ceiling on one mode is a floor on the other | `_share_interval`, `_parse_bounds` |
| `pinned` | fixed weight for the third mode, re-asserted every tick | every tick |
| `deactivate_threshold` | frac below which a branch is skipped, per stage or the global fallback | `fused_train_step` |
| tick | ten train steps | `Protocol.tick` |

Under `kind: proportional`, `controller.beta` and a stage's `min_fracs` are not read; the `floor` key is the analogous knob. Under `kind: ratio`, `min_fracs` is folded into `bounds` at parse and an inconsistent pair fails to load. Under `lexicographic`, the fracs a stage declares are a seed for the nudge, not an allocation.

## Retired

The buffer freshness servo, `buffer_servo`, was retired on 2026-09-09 and its controller deleted. Its block is refused at parse with a message pointing at the residence and churn keys it used to scale ([replay-buffer](replay-buffer.md)). The ratio kind is parsed and runs but is not used by any stage in the canonical config.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:stage.balance.kind`, `cfg:stage.balance.pinned`, `cfg:stage.balance.bounds`, `cfg:stage.balance.metrics`, `cfg:stage.balance.targets`, `cfg:stage.balance.drive`, `cfg:stage.balance.alpha`, `cfg:stage.balance.floor`, `cfg:stage.balance.anneal`, `cfg:stage.balance.constrain`, `cfg:stage.balance.bars`, `cfg:stage.balance.gain`, `cfg:stage.balance.priority`, `cfg:stage.balance.max_step`, `cfg:stage.balance.numerator`, `cfg:stage.balance.setpoint`, `cfg:stage.balance.converge_floor`, `cfg:stage.balance.ramp`, `cfg:stage.balance.guard`, `cfg:stage.balance.metric`, `cfg:stage.balance.bar`, `cfg:stage.balance.up`, `cfg:stage.balance.down`, `cfg:stage.balance.ratchet_metric`, `cfg:stage.balance.ratchet_tol`, `cfg:stage.balance.ratchet_release_tol`, `cfg:stage.balance.ratchet_cooldown_steps`, `cfg:stage.balance.rules`, `cfg:stage.balance.anneal_coeffs`, `cfg:stage.balance.anneal_cooldown_steps`, `cfg:stage.deactivate_threshold`, `cfg:controller.beta`, `cfg:controller.deactivate_threshold`.

Code: `protocol.py::Stage._parse_balance`, `protocol.py::StageProtocol._balance_tick`, `protocol.py::StageProtocol._proportional_tick`, `protocol.py::StageProtocol._constraint_tick`, `protocol.py::StageProtocol._ratio_tick`, `protocol.py::StageProtocol._gated_ramp_tick`, `protocol.py::StageProtocol._nudge_mode_fracs`, `train.py::Modeller.fused_train_step`.

## Sources

The code above, read at the stamped commit, and the canonical config's `equilibration` and `var_conditioning` balance blocks. Memory files that describe these controllers (project_constraint_balance_controller, project_proportional_controller_and_buffer_fixes, project_gated_ramp_ratchet_pins_replay, project_ratio_controller_rails_on_floored_denominator, project_replay_freshness_servo) were used to locate the code, not as evidence; one of them describes the buffer servo as live, which it no longer is.
