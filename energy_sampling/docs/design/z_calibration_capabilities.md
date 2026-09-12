# What the log Z machinery can actually do

Written 2026-09-10, before deciding how to key it. This is an inventory of the
**mechanisms that exist in the code today**, not a proposal. Argument, per
`PROTOCOL.md`.

The occasion is that `z_calibration` in `configs/mk_dev.yaml` has grown two
independent subsystems under one block name, plus a third that lives outside it
entirely, and their config surface no longer says which is which.

## 1. There are three actuators, not one

All three write the same object — `flow_model.scalar`, plus its EMA twin — and
all three are only meaningful on the **unconditional scalar head**. They differ
in what they cost and what they know.

| | `z_calibration_tick` (the SERVO) | `z_level_fill` (the FILL) | `bootstrap_z_by_rollout` |
|---|---|---|---|
| where | `train.py:5256` | `train.py:5712` | `train.py:5869` |
| when | every train step, frequency-modulated | every train step, gated | stage `on_enter` only |
| moves Z by | one Adam step at `lr_flow` | a closed-form target | a closed-form target |
| magnitude-aware | **no** (see §2) | yes | yes |
| marginal cost | one rollout + energy call **per step taken** (`rollout` mode) | **zero** (§3) | one large rollout, once |

## 2. Why the servo cannot close a large gap

The sidecar's step size is magnitude-blind **twice over**:

1. the Huber clips `dL/dZ` at `beta`, so a batch 800 nats out produces exactly
   the same gradient as one 10 nats out; and
2. Adam then normalises whatever survives, so the head moves ~`lr_flow` nats per
   step regardless.

Error therefore decays **linearly**, not exponentially. A large debt is not
recoverable inside a stage's budget at any frequency. This is the whole reason
the fill exists, and it is why "take more Z steps" is not an answer to "log Z is
40 nats out".

## 3. Why the fill is free

`get_tb_loss`'s residual is `log_pf + log_Z - log_pb - log_r`, i.e.
`log_Z - logw` for `logw = log_pb + log_r - log_pf`. So **any step that already
ran a forward TB branch has the fixed point in hand**: `_stash_z_fill_logw`
(`train.py:5689`) keeps that vector for one subtract on tensors the step already
produced. `winsorized_z_root(logw, beta)` then returns `(root, se, frac)` in
closed form.

No rollout. No energy call. Contrast the servo's `rollout` mode, which spends a
full rollout **and** an energy call per Z step to crawl toward the same number at
`lr_flow`.

**Consequence, and it is the one that matters:** on an unconditional run the
answer is available for free on every fwd step, and the servo is paying for
information it is being handed. `train.py:3412` already encodes this — the fill
runs *before* the tick, "so it must pre-empt the servo's rollouts rather than run
after they have spent an energy call each crawling at `lr_flow`".

## 4. The servo's own axes

`mode` — the step body:

- **`rollout`** — fresh fwd rollout under `freeze_policy` with the auxiliary Z
  terms zeroed, so sensor, actuator and fused loss share one fixed point. Costs
  a rollout + energy call *per step taken*.
- **`replay`** — same gradient over stored trajectories. No energy call, but Z is
  calibrated to the **buffer's** measure, which lags the policy by the replay
  buffer's mean residence. Raises unless intake and purge are both
  residual-independent. Marked "not recommended" in its own docstring.
- **`regression`** — least squares onto the tracker's `ema_logw` over cached
  condition embeddings. Nearly free, but a **mean-family** target, so its optimum
  differs from TB's winsorized one. Carries `min_visits`,
  `freshness_half_life_steps` and the rest of the regression sub-block.

`sensor` — the trigger reading, all compared against `threshold`:

- **`grad_rms`** — RMS per-condition clipped signed residual; the loss's own
  first-order condition. Zeroes at the rollout actuator's fixed point, so it
  **cannot latch**.
- **`rms`** — unclipped level error (the docstring says "dispersion"; see below). Floors at the winsorized-vs-mean skew
  gap, i.e. reads a standing offset at the fixed point: **a latch, not
  convergence**.
- **`worst`** — upper-tail quantile over conditions (`sensor_quantile`).
- **`pooled`** — `|EMA fwd/tb_resid_clipped|`. Blind to per-condition
  disagreement that cancels in the pool.

**All four are live on an unconditional run — `mk_dev`'s own comment on this is
wrong.** It says "rms/worst are per-condition dispersion and read identically
zero on a single-condition (unconditional) run". They are not dispersions: each
is an RMS or upper-tail quantile of the **signed level error itself**
(`rms_z_grad` = RMS of `z_grad_ema`, `buffer.py:2333`; `rms_z_bias` of
`z_bias_ema`, `:2349`; `worst_z_bias` of `|z_bias_ema|`, `:2418`), not of a
deviation from a cross-condition mean. With one condition each collapses to that
condition's own `|EMA residual|` — a perfectly live reading, and the value is
zero only when the tracker is cold (no condition has reached `min_visits`).

So the four sensors differ on the live route by **where the EMA lives and whether
it is clipped**, not by whether they read at all:

| sensor | reads | clipped |
|---|---|---|
| `grad_rms` | tracker `z_grad_ema` | yes, at `clip_beta` |
| `rms` | tracker `z_bias_ema` | no — unbounded, and one degenerate off-policy `log_pf` can dominate it |
| `worst` | tracker `\|z_bias_ema\|`, upper tail | no |
| `pooled` | `metric_tracker` `fwd/tb_resid_clipped` | yes, at `beta` |

`sensor_quantile` is still inert unless `sensor: worst`.

Rate: steps taken per train step = `min(gain * (sensor/threshold - 1),
max_steps_per_step)`, Bernoulli on the fraction, cut short once a rollout's own
fresh reading falls under `threshold * grace` (rollout mode only).

## 5. The fill's own axes

`fill_mode`:

- **`snap`** — gated overwrite. `fill_threshold` (how far off is far enough to
  stop trusting the servo; defaulted at `2 * beta`, where every row is saturated)
  **and** `fill_se` (whether *this* batch resolves the gap it claims; `se` is
  `+inf` when no row is unclipped, so a batch carrying only a sign can never
  license a fill) must **both** pass. `fill_cooldown_steps` then holds the fill
  off, because `fwd/tb_resid_clipped` — the `pooled` sensor — cannot be shifted
  from the fill site and keeps reporting the pre-fill level for its own time
  constant.
- **`absorb`** — 1-D Kalman filter on log Z. State `(Z, P)`; `P` grows by
  `fill_process_var` per step between applied measurements and shrinks when one
  lands. Each measurement `(root, se)` moves Z by `K = P_pred / (P_pred + se^2)`
  of its gap. **No thresholds**: the snap gates existed to keep noise out, and
  this weights noise down instead of discarding it. First measurement taken whole
  (diffuse prior). `fill_moment_reset` decides whether Adam's flow moments are
  dropped — a small absorbed step leaves them meaningful.

`fill_from_eval` (`off|report|fill`) — feed the **eval** rollout's `log w`
through the same actuator. That batch is 2500–10000 samples, so its `se` is the
lowest the run ever measures and the absorber's `K` is near 1.

**Under `absorb`, three of the snap keys are inert**: `fill_threshold` degenerates
to a bare enable switch (`> 0` arms it), `fill_se` enters the gain rather than a
gate (only a non-finite `se` blocks), and `fill_cooldown_steps` has nothing to
protect because the absorber reads the batch root directly.

## 6. What a fill invalidates

`log_Z` is the numeraire, so moving it **re-signs every residual at once**.

- The tracker's two level EMAs are shifted at the fill site — exactly for the
  unclipped one, to first order for the winsorized one.
- `metric_tracker`'s `fwd/tb_resid_clipped` **cannot** be shifted from there and
  keeps reporting the pre-fill level for its own time constant.
- Adam's moments for the flow parameter are dropped (snap always; absorb above
  `fill_moment_reset`), since they describe a pre-fill gradient.

## 7. The seam with rare rollouts

Under `fwd_rollout_every: N` the forward branch — the only branch that calls the
energy function, and the only source of a free `logw` — runs on 1 step in N.
That makes the two subsystems trade places:

- the **fill** gets its free measurement only on rollout steps, so log Z is
  pinned at the rollout cadence and drifts between (`z_pin_rollout_every` exists
  to add a gradient-free rollout whose only product is the fill stash, and whose
  rows do not enter the replay buffer);
- the **servo** in `rollout` mode makes its *own* energy call per Z step off a
  sensor that is frozen between rollouts — so it fires on every **skipped** step
  and restores exactly the cost the rare-rollout cadence was removing.

That is already refused at load: activating `fwd_rollout_every` requires
`flags.z_calibration: false` and `z_calibration.fill_threshold > 0`.

## 8. Summary of the redundancy

On an **unconditional** run with rare rollouts — the live route:

| surface | status |
|---|---|
| `mode: rollout` | pays a rollout + energy call to approach a number the fill has for free, and is refused outright alongside `fwd_rollout_every` |
| `mode: replay` | calibrates to the buffer's lagged measure; "not recommended" in its own docstring; requires residual-independent intake **and** purge |
| `mode: regression` | mean-family target, different optimum from TB's; needs the tracker populated |
| `sensor: rms / worst / grad_rms` | all LIVE on one condition (§4) — they differ from `pooled` only in which EMA they read and whether it is clipped |
| `sensor_quantile` | inert unless `sensor: worst` |
| `min_visits`, `freshness_half_life_steps`, + regression sub-block | inert unless `mode: regression` |
| `fill_se`, `fill_cooldown_steps` | inert under `fill_mode: absorb` |
| `fill_threshold` | degenerates to an enable switch under `absorb` |

What is load-bearing on the live route: `fill_mode`, `fill_process_var`,
`fill_moment_reset`, `fill_from_eval`, one enable switch, and `bootstrap_z` at
stage entry.

## 9. Open question for the keying

The servo and the fill are two mechanisms with different cost models, different
convergence behaviour, and — on the live route — almost no overlap in when they
are the right instrument. They currently share one block name, one enable path
(`flags.z_calibration` gates the servo; `fill_threshold > 0` gates the fill), and
a `mode` key that means the servo's mode while `fill_mode` means the fill's.

The naming decision is deferred to the owner. The inventory above is what has to
be keyed.
