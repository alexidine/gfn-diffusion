# Module: protocol (`protocol.py`)

> **Status: MODULE SNAPSHOT.** The verification dates below are historical.
> Use this document for explanation and navigation; verify material claims
> against current code, canonical config, and focused tests. See
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03;
balance-controller sections revised 2026-08-06 for `kind: ratio` (P7);
**line refs and P4/P8 revised 2026-08-08**.
Unconditional route (`configs/mk_dev.yaml`: two stages, `train_prior` →
`equilibration`; the second was called `naive` until 2026-08-08 and pre-rename
checkpoints will not resume, since `StageProtocol.stage` resolves by NAME).

---

## 1. What it is

One declarative engine that replaced three overlapping controllers
(`phases.PhaseController`, `ModeBalanceController`, `ForwardFirstController`).
A protocol is an ordered list of **stages**; each stage declares what train mode
runs, where backward draws come from, which loss coefficients override the base,
which behaviour flags are on, how the mode weights are controlled, and what
metric conditions end it.

The central design decision: **configuration owns behaviour, checkpoints own
only position.** A stage's live loss coefficients are a pure function of (base
config, current stage) — no schedules, no mutation across steps. This also fixed
a real legacy bug: the old controllers annealed `args.controller.*_threshold` in
place, which was never checkpointed, so anneal progress silently reset on every
resume.

Paper voice: *training proceeds through declaratively specified stages; within a
stage a controller continuously reallocates weight between the backward and
replay objectives according to two coverage statistics, and stage transitions
are triggered by metric conditions rather than step counts.*

## 2. Contract

**Owns** — `m.stage`, the per-mode `*_frac` values, `m.stage_ctrl` (all mutable
engine state: rule bests, streaks, live annealed thresholds, gate latches,
`request_eval`), and the transition barrier.

**Transition barrier** ([`advance`](protocol.py:1014)) — every transition, uniformly:
clear loss windows → reset batch-controller knee state →
`init_schedulers_optimizers` → `set_loss_coeffs` → `rearm_warmup` → stage
`on_enter` actions → `save('stage_start')`.
(`reset_spike_monitors` was the first item until 2026-08-08; it re-armed the
fire cooldowns so transition turbulence could not eat a fire. The v7 divergence
bar sits at ~1e9, which transition turbulence does not reach, so there is
nothing left for a transition to protect it from.)

That list is the module's best feature. Each item has a reason (stale loss
windows are a wrong ceiling for the incoming stream; Adam moments describe the
wrong surface; the outgoing stage's best-checkpoint minima shouldn't gate the new
stage's saves), and having them fire *automatically* means a new stage cannot
forget one. Route-specific physics goes in `on_enter`/`on_exit`; everything
generic is unconditional.

**Exit** — an AND-list of `{metric, above|below, patience}` terms. Tick-resolvable
terms are checked every 10 steps; when all reach patience the next eval is
*pulled forward* (`stage_ctrl['request_eval']`) and the transition executes
inside `evaluation()` with fresh eval metrics in hand. This is also what lets a
reloaded pre-transition snapshot replay its transition through the normal path.

## 3. The live route

| | `train_prior` | `equilibration` |
|---|---|---|
| `train_mode` | `bwd` | `fused` |
| Sampling | `dataset` | `prior` (churned buffer) |
| Losses | `mle` + `tbc`, `repeats: 2` | `tb` ×3 branches |
| Flags | `update_log_z`, `scramble_conditions`¹, `mle_gate` | `update_log_z`, `buffers_active` |
| Exit | `gates/mle_flat` ∧ `eval/wass_debiased < 0.015` ∧ `bwd/tbc < 2.0` | none — terminal |
| Balance | — | `kind: ratio` (was `proportional` until 2026-08-06 — P7) |

¹ inert on this route — see `module_training_modes.md` M3.

The `equilibration` balance splits the bwd/replay pair while `fwd` stays pinned —
a pin that is a **binary gate, not a dial** (`decisions.md` `P7`): the Z head's
sole gradient source is `fwd`, and Adam cancels a uniform scale, so above
`deactivate_threshold` the value does not matter.

```
e      = log(over_coverage / relative_under_wcen) - log(setpoint)
k      = clip((max(metrics) - converge_floor) / converge_floor, 0, 1)
theta <- clip(theta + clip(gain·k·e, ±max_step), theta_lo, theta_hi)
share_replay = sigmoid(theta)
```

with `setpoint: 5.0`, `gain: 0.02` (≈3.3k-step time constant at the measured
plant gain), `bounds: {replay: [0.05, 0.60]}`, `converge_floor: 1.0`.

`bounds.replay` was widened from 0.45 on 2026-08-08: both long `local_aug09`
arms **saturated** at 0.45 with `rt_rho` still above setpoint, so the controller
had lost authority — the bound, not the setpoint, was deciding the mix.

One number carries the judgement — `setpoint` is the exchange rate between the
two halves of the residual field — and `bounds` carry the safety, putting bwd in
[0.35, 0.90] of total, clear of the retention knee. `min_fracs.bwd: 0.25` is
folded into those bounds at parse (P3) and is the explicit collapse guard. Everything else is
mechanism. See P7 for the derivation and what it replaced.

## 4. Findings

**P1 — `drive: relative` fails silently when a target is mis-set, and nothing
reports it.** *(confirmed; **no longer live on this route** — superseded by P7's
`kind: ratio`, which has no clamp. Retained because `proportional` and
`constraint` both still carry the defect and both are still selectable.)*

[`_drive`](protocol.py:1308) returns `max(value/target − 1, 0)`, so a side whose
metric settles **below** its target contributes drive 0 forever. There is no
warning and no metric that says "this side is pinned."

It **was** live until 2026-08-06: `targets: {bwd: 3.0}` was calibrated against
the *old* `relative_under` metric and the config had since been switched to
`relative_under_wcen`, which reads lower by an unmeasured amount. See
`module_metrics.md` T2. The result was a **one-sided controller** — replay's
drive pulled share away from bwd, but bwd could never pull back, so the split
could only move in one direction from the idle mix. Every result produced under
that config should be read as a one-sided controller, not a two-metric balance.

The failure class has already bitten once: the `_drive` docstring records that
`report()` once hardcoded the absolute form while the split used the relative
one, "which is how a welded-off drive got read as 'one thing to watch' instead
of the cause." That fix made the logged drive match the computed one — but a
*permanently zero* drive still looks identical to a *momentarily satisfied* one.

**Fix**: emit ticks-since-last-nonzero-drive per side. Two lines, and it makes
this whole class self-announcing.

**Target placement is the design choice** *(revised 2026-08-05)*. Because
`s = max(v/target − 1, 0)`, a target **above** its metric's operating range
makes that side inert; **at the top** of the range makes it an excursion guard;
**inside** the range makes it a continuous allocator. Same code, three
controllers — and nothing reports which regime a side is in.

That, not any particular number, is what needs deciding. Specific settling bands
are intensely contextual (problem, T, W, stage) and are deliberately not recorded
here.

✅ **Resolved 2026-08-06 by `kind: ratio`** (P7, `decisions.md` D8). The **clamp**
is what makes the regime ambiguous, so the fix was to remove it rather than to
report around it: a signed log-ratio is never zero-by-satisfaction, so inert /
guard / allocator stops being a distinction the config can silently fall into.
The drive-liveness reporting fix is still wanted for the two kinds that keep the
clamp.

**P2 — a pinned drive also corrupts the anneal condition.** *(confirmed; **moot
on this route** — `kind: ratio` has no anneal and `decisions.md` D11 ruled against
adding one. Applies to any `proportional` stage that declares `anneal`.)*

The anneal streak advances when `s_a + s_b <= 0` — i.e. when *both* sides are
satisfied. With one side pinned at 0, the condition collapses to "is the other
side satisfied," so the anneal fires on half the evidence it was designed to
require. Since annealing *tightens* targets, it would eventually pull the mis-set
target back under the metric and un-pin the side — self-correcting, but only if
`anneal` is configured. **mk_dev's `equilibration` balance has no `anneal` block**, so
there is no self-correction on this route.

**P3 — three floor mechanisms, one live.** *(partly resolved 2026-08-06)*
Historically `min_fracs` was inert under `kind: proportional`, leaving
`balance.floor: 0.03` as the only protection, with
`controller.deactivate_threshold: 0.01` sitting *below* it so the deactivation
path could never trigger (`module_training_modes.md` M5).

Under `kind: ratio` the live bound is **`bounds`**, and it is now checked against
`deactivate_threshold` at parse time — a lower bound below it is rejected
outright, because a branch going dark while the controller still steers it is the
s706frkh failure. `balance.floor` does not exist for this kind.

**`min_fracs` is no longer inert — resolved 2026-08-08, and it was worse than
"inert".** It was read *only* by `_nudge_mode_fracs`, the lexicographic path, so
a stage declaring both `min_fracs` and an integrator balance had its floors
silently ignored by the integrator that was actually running. The floor was
declared, it appeared in the config, and it was not a floor. `_parse_bounds` now
**folds it into `bounds`**: a split mode with a `min_fracs` entry and no explicit
bound gets `[floor, pair_mass]`, and a declared bound *below* its own
`min_fracs` is a hard parse error, as is a `pinned` value below one.

Folding rather than checking-at-tick is deliberate. R3's complaint is that three
frac bounds live in three files; adding a fourth enforcement site would make that
worse. After the fold there is exactly **one** live bound per mode, and the two
ways of declaring it either agree or fail loudly at load. S3 (reject `min_fracs`
on split stages) is therefore **withdrawn** — the key now means something on
every kind, which is what it always looked like it meant.

**P4 — `buffer_servo` is not configured on `mk_dev`, but it is now exercised
elsewhere.** *(revised 2026-08-08)* The mechanism exists and its rationale is one
of the sharpest arguments in the codebase: branch weights cannot fix replay
**overfitting**, because down-weighting a memorised buffer trains less on it but
does not make it less memorised — so the two pathologies need two actuators. On
the default route replay overfitting is still unactuated.

What changed on 2026-08-07: the servo acquired a **derived** sensor
(`replay/ema_loss_mean ÷ replay/birth_loss_mean`, bar `1/e`), and it is now
declared by `local_aug07/r4_overfit_servo`, four `ctrl_aug03` arms, and **rb0808
arm 20**. It needed **no change to `protocol.py`** — the `numerator`/`denominator`
generality was already there. Two findings came out of running it, and both are
recorded in `module_modulators.md` rather than here because they are servo
properties, not protocol ones: D7 (the sensor was not in `metric_tracker`, so the
servo cold-started on every tick and emitted nothing) and D8 (validated as wired,
unproven as effective).

The protocol-side consequence of D7 is worth stating here though: **anything
`_resolve` reads must be in the metric tracker.** A servo whose sensor is absent
takes its cold-start early return, which is byte-identical in the logs to a servo
correctly holding. `Stage.metrics_needed` already force-refreshes the branches a
`buffer_servo` reads, so the *rollout* was happening; it was only the publication
that was missing.

**P8 — `kind: constraint` exists, is selectable, and was undocumented until
2026-08-08.** *(descriptive; superseded on this route by `kind: ratio` but not
retired)*

[`_constraint_tick`](protocol.py:1581). Same two-mode split and same integrator
coordinate as `ratio`, but the two sides are treated **asymmetrically** — one
metric is a bar that must hold, the other a best-effort objective:

```
d_c    = max(metric_c / bar_c − 1, 0)          # constrained side
d_r    = max(metric_r / bar_r − 1, 0)          # best-effort side
θ     ← clip(θ + clip(gain · (d_r − priority · d_c), ±max_step), θ_lo, θ_hi)
share_r = sigmoid(θ)
```

**What it encodes that `ratio` deliberately does not.** The two sides are not the
same kind of quantity. The constrained one (bwd absorbing the buffer's modes) has a
level statable *a priori* — `relative_under_wcen ≈ 2` means "no mode is badly
under-weighted" — while the best-effort one (`over_coverage` on fresh forward
samples) has no known reachable level. `priority` is a **gain multiple, not a
switch**, so the constraint wins contests without ever taking the whole batch: a
*soft* lexicographic order, with no discontinuity for the split to limit-cycle on
— which is what the rule-based controller did in replay_july26.

**An unreachable best-effort bar is a designed-for case, not a misconfiguration.**
Its drive never reaches zero, so θ walks until either the constrained side pushes
back or θ hits a bound. Both are the intended answer to *"take as good as we can
get provided the constraint holds"*, and both are legible: `cs_at_bound` reports
which. That is why `bounds` is **required** at parse time here as it is for
`ratio`.

**It still carries P1's clamp**, on both sides — so a side sitting under its bar
contributes nothing and is indistinguishable from a satisfied one. That is the
defect `ratio` was built to remove, and it is why `ratio` supersedes this on the
live route rather than the reverse. `constraint` is kept because `priority` is a
real asymmetry that `ratio` cannot express, and the asymmetry matches a measured
one: starving bwd collapses coverage fast but recovers fast, whereas starving
replay/fwd degrades the policy slowly and recovers slowly.

Shared machinery worth knowing about, since it is easy to assume each kind has its
own: `_parse_pinned` and `_parse_bounds` are **common to `proportional`,
`constraint` and `ratio`**, and `_share_interval` converts absolute frac bounds to
θ limits for both integrator laws specifically so the two cannot drift apart on the
direction of that conversion. Only one config in the tree selects it
(`ctrl_aug03/cs_servo`).

**P5 — two of six `ACTIONS` are undeclared on this route.**
`rebuild_prior_by_churn` and `reseed_prior_from_dataset` are implemented and
reachable but named by no stage, which is why `prior_buffer.init_fraction` is
marked inert (`module_buffers.md` B5).

**P6 — the transition barrier is a hard reset of every controller.** LR cuts are
forgiven, `stage_ctrl` is replaced, batch-knee state is cleared, Adam is rebuilt.
With two stages this fires once, so the cost is bounded — but it means **no
controller carries evidence across the phase boundary**, and the phase-2
controller begins from a cold start with metrics inherited from a stage that
trained a different objective. The first balance ticks in `equilibration` therefore act
on `train_prior`'s residual EMAs until they turn over (~100 steps at
`period=100`). Worth either warming the tracker before the first tick or
suppressing ticks until each read metric has refreshed.

**P7 — `kind: ratio`: the balance controller re-derived around the exchange
rate.** *(implemented 2026-08-06 — [`_ratio_tick`](protocol.py:1471); closes D8)*

The split allocates between two **disjoint** halves of one residual field: by the
blindness bound (`to_do_rebuild.md` §B1c) `fwd`/`replay` are structurally δ>0
instruments and `bwd`/prior own δ<0. Each mode therefore owns the error it is the
instrument for, and the only thing that has to be declared is their **exchange
rate**. That is one number, not two:

```
e     = log(v_num / v_den) − log(setpoint)
θ    ← clip(θ + clip(gain · k · e, ±max_step), θ_lo, θ_hi)
```

Four differences from `kind: proportional`, each aimed at a recorded failure:

| | `proportional` | `ratio` |
|---|---|---|
| Parameters | two targets — one redundant dimension | one setpoint = the exchange rate |
| Clamp | `max(v/t − 1, 0)`; a satisfied side is **inert** (P1) | none; signed, both sides always live |
| Law | static map, EMA'd — a wrong target is a permanently wrong split | integrator — equilibrium is a property of `setpoint` alone |
| Coordinate | linear | log, matching the multiplicative noise |

The log coordinate is **derived, not stylistic**: the observed oscillation has
constant *relative* amplitude (~2× peak-to-trough on `ty4xdlzo` while the level
falls 10×), so in logs the cycle is a symmetric perturbation the integrator
averages to zero and in linear units it biases the mean.

`converge_floor` fades the gain to zero as the larger metric approaches it. A
scale-free setpoint otherwise demands the same ratio whether the halves are 20
nats apart or 2, i.e. it keeps steering on noise after the trade has vanished.
The loop retires itself instead.

**What it does not do, deliberately.** It does not find the setpoint. That is an
optimisation against `J = bwd/jensen_z − fwd/jensen_z` = `KL(Q‖P) + KL(P̂‖Q)`
(Jeffreys; `log Z` cancels), and θ acts on J's **rate**, not its level — an
integrating plant, so slope estimation, so ~10k steps per usable measurement.
With J also carrying a limit cycle of run-varying period and amplitude, that is
an offline scoring statistic, not a control input. Setpoint comes from a
bracketing battery; the loop only holds it.

**Open risk: actuator authority.** Across the four ctrl_aug03 arms
`over_coverage ÷ relative_under_wcen` sat at 5.4–6.2 while θ moved 2× — a plant
gain of `d log ρ / d θ ≈ 0.15`. Confounded (buffer servo on in three arms) and a
2000-step screen, but if it holds, a setpoint far inside the band is unreachable
within safe bounds and the loop parks. `rt_at_bound` says so explicitly, and the
`(θ, ρ)` trace measures the gain on the way.

## 5. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| Config owns behaviour, checkpoints own position | **derived** — the alternative silently reset anneal progress on resume | stated in module docstring |
| Uniform automatic transition barrier | **derived** — every item has a stated failure it prevents | `advance` comments |
| Continuous split over lexicographic | **measured** — rule-based switching limit-cycled | replay_july26 |
| Relative-to-best rules over absolute bars | **measured** — the calibration floor legitimately rises as coverage grows, so absolute bars deadlock | run b9ze0p5c |
| Exit pulls the eval forward | **derived** — makes transitions reproducible on resume | `request_eval` |

**`kind: ratio` (live), per P7:**

| Choice | Warrant | Evidence |
|---|---|---|
| One setpoint, not two targets | **derived** — only the exchange rate moves the equilibrium; the second target is a redundant dimension free to be set wrong | P7 |
| Signed error, no clamp | **derived** — a one-sided drive's zero is indistinguishable from "welded off" | P1, and the same shape in `decisions.md` 2e |
| Log coordinate | **derived + measured** — the oscillation has constant *relative* amplitude (~2x while the level falls 10x), so logs make it symmetric and the integrator averages it out | `ty4xdlzo` `zmatch/delta_worst` |
| Integrator, not static map | **derived** — the equilibrium becomes a property of `setpoint` alone, independent of gain, idle mix and the metrics' absolute levels | P7 |
| `bounds` required, checked against `deactivate_threshold` | **measured** — a bound is what makes a wrong setpoint degrade to a fixed mix instead of a collapse | replay_july26 (bwd 0.001 → EffDim 1.3); s706frkh (dark branch) |
| `converge_floor` gating the gain | **derived** — a scale-free setpoint keeps steering after the trade it arbitrates has shrunk into noise | user, 2026-08-06 |
| `gain: 0.02` | **derived from a measured plant gain** — ~3.3k-step time constant at `d log ρ/dθ ≈ 0.15`, above the 1–2k absorption cycle | ctrl_aug03 (confounded, 2000 steps) |
| `setpoint: 5.0` | **measured, provisional** — one notch inside the observed 5.4–6.2 band, so arm 1 makes a real displacement without parking at a bound. The user's own prior is 3–5 | ctrl_aug03 |
| `bounds: {replay: [0.05, 0.45]}` | **measured on one side** — the implied bwd floor of 0.35 clears the retention knee (bwd share 0.3 → EffDim 5.33, 0.001 → 1.6); the replay ceiling is **arbitrary** | replay_july26 |

Structure traces to specific observed failures at nearly every decision point;
constants remain the weak part, but `ratio` has three of them instead of six, and
only one (`setpoint`) changes behaviour rather than rate or safety.

## 6. Failure signatures

| Symptom | First check | Cause |
|---|---|---|
| Split not moving | `rt_err`, then `rt_hold`, then `rt_gain_scale` | `rt_err`≈0 = at setpoint (correct). `rt_hold`==1 = a metric is absent/non-positive and the loop is holding. `rt_gain_scale`==0 = converged past `converge_floor`. The three are deliberately separate series: one zero meaning two things is the defect this controller exists to avoid |
| Split pinned at a bound | `protocol/rt_at_bound` ±1 | setpoint unreachable within `bounds` at the current plant gain — the bound *is* the mix; read `(rt_theta, rt_rho)` for the gain |
| Split drifting with no excursion | `rt_err` sign vs `rt_setpoint` | setpoint sits off the metrics' operating band |
| Controller oscillates | `gain` vs absorption cycle | time constant `1/(gain · d log ρ/dθ)` under ~2k steps |
| *(legacy kinds)* split parked at `default_boost`, or moving one way only | `prop_drive_*` for a permanent zero | P1 — mis-set target |
| *(legacy kinds)* anneal firing on a noisy stage | `prop_streak` | P2 — a pinned side halves the evidence |
| A mode goes dark unexpectedly | `min_fracs` vs `deactivate_threshold` | P3 — floors not stated together |
| Transition never fires | each exit term separately | AND-list — one unresolvable term blocks all |
| First ticks after a transition look wrong | `{mode}/*` EMA age | P6 — inherited stats |

## 7. Simplification candidates

**S1 — report drive liveness** (P1). ⏸ **Descoped 2026-08-06** — the live route
no longer has a clampable drive, so this is maintenance on `proportional` /
`constraint` rather than the module's highest-value change. Do it if either kind
is used again.

**S2 — state the three frac bounds in one place** (P3). ✅ **Done 2026-08-08.**
`ratio`/`constraint` enforce `bounds` ≥ `deactivate_threshold` *and* fold
`min_fracs` into `bounds`, so every relationship is now checked at parse rather
than documented and hoped for.

**~~S3 — reject `min_fracs` on split stages~~** — **withdrawn 2026-08-08.** The
key is live on every kind now (P3); there is nothing inert to reject.

**S4 — gate the first balance ticks on metric freshness** (P6). Unchanged, and
now worth more: an integrator that acts on the previous stage's EMAs writes that
error into its own state instead of washing it out.

**S5 — retire `kind: proportional` and `kind: constraint`?** Not yet. Both are
superseded on this route, but neither has been A/B'd against `ratio`, and
`constraint` encodes an asymmetry (`priority`) that `ratio` deliberately does
not (P8). Revisit after the first `ratio` arms — **rb0808 arm 23
`rep_fixed_fracs` is the relevant comparison and it is running now**: it is the
only arm in the battery with the balance controller *off*, fracs held at the
0.2/0.6/0.2 entry split, so it asks whether the `ratio` controller earns its keep
at length. Note what it is *not*: it is not a `ratio`-vs-`constraint` A/B, and
nothing in the tree currently schedules one.

## 8. Open questions

1. **Dissolved.** P1 wanted `bwd/relative_under_wcen`'s settled band in order to
   fix a target. `ratio` has no target, so the band is never needed. What *is*
   needed is the **plant gain** `d log ρ/dθ` — a different measurement, and one
   the controller's own `(rt_theta, rt_rho)` trace yields for free.
2. **Answered — no** (`decisions.md` D11).
3. Is replay overfitting actually occurring on this route? If yes, `buffer_servo`
   is unactuated (P4); if no, the servo is dead weight and should be documented
   as conditional-only.
4. **Answered — deliberate** (`module_metrics.md` §9.3): fast faithful sensing,
   per-consumer filtering. The `ratio` loop's own time constant is ~3.3k steps
   against the tracker's 100, for the same reason.
5. *(new)* **Does the setpoint move over training?** The design rests on holding
   it fixed. The test is two long arms at the edges of the usable range, scored
   on the log-slope of `J = bwd/jensen_z − fwd/jensen_z`; if their ordering flips
   mid-run the setpoint needs a slow trim, otherwise it does not.
6. *(new)* **Does ρ have enough authority?** The measured `d log ρ/dθ ≈ 0.15` is
   confounded and from a 2000-step screen. If it holds across the full range,
   setpoints well inside the band are unreachable and the loop parks — legibly,
   but uselessly.

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
