# LR control — current behaviour

Spec of what the code does now. No history, no rationale beyond what is needed to
read the numbers. Sources: `controller.py`, `lr_bracket.py`,
`lr_bracket_probe.py`, `configs/mk_dev.yaml`.

The mechanism is a **brute-force bracket**: burn in at a conservative fixed rate,
checkpoint the complete mature trainer state, run a configured grid of fixed-LR
continuations from that one checkpoint, and keep the highest rung a safety margin
below the lowest one that detonated. There is no estimator, no fitted curve, no
confidence interval and no continuous servo anywhere in the actuation path.

> **What this replaced.** Controller v8 steered `peak_scale` from a per-stage
> sensor (`ray` / `hyper` / `plateau`) under a warmup envelope, with a pooled
> estimator and a periodic re-probe. `ray` was killed 2026-08-23 by its own
> acceptance test: `alpha*` is defined as `s*/lr`, so the slope of `log(alpha*)`
> against `log(lr)` must be −1, and it measured 0.00 ± 0.2 across twelve runs,
> two stages and spans up to 2.7 decades. See `project_ray_sensor_postmortem` in
> memory. `ray` and `hyper` survive as opt-in diagnostics that reach no learning
> rate; `plateau` is deleted.

---

## 1. The controlled quantity

```
lr[key] = base_lr[key] × scale          clamped to [min_lr, max_lr]
```

- `base_lr[key]` — per-optimizer config value (`lr_policy`, `lr_back`,
  `lr_replay`, `lr_fused`). `protocol` action `set_lr_policy` may move it per
  stage.
- Only keys written `auto` are actuated. `lr_servo_managed` records which; an
  empty set means the controller reads and logs but moves nothing (a documented
  control arm — the bracket then refuses to run trials, because there is no rate
  under test).
- `lr_flow` (Z head) is pinned flat and never scaled, unless
  `lr_control.control_flow_lr`. The `max_lr` rail still applies to it, and is the
  only thing that can lower it.
- `scale` — **one scalar, piecewise constant**, for all managed policy groups.
  Every value it takes is a number written in the config: the burn-in scale, a
  candidate rung under trial, or the promoted rung.

**There is no envelope, no warmup ramp and no decay leg.** A continuous
multiplier underneath a candidate rung would mean the rate under test is not the
rate applied, which is how a too-hot rung survives its trial.

Only two things write `scale`: the bracket, and a stage transition. Both go
through `LRController.set_scale`.

---

## 2. The state machine

```
burn_in ──(exactly burn_in_steps)──> root checkpoint ──> trials ──> cruise
   ^                                                                  │
   └──────────── stage transition ────────────────────────────────────┘
                 (or, if repeat_every > 0, a repeat: new root, no new burn-in)
```

### 2.1 Burn-in

`burn_in_steps` steps at `burn_in_scale`, counted from stage entry. **Step count
only** — it never waits on a learned metric.

The length is set by Adam's bias correction, `sqrt(1−β₂ᵗ)/(1−β₁ᵗ)`, because
optimizers are rebuilt at every stage transition:

| t | 10 | 100 | 500 | 1000 | 3000 |
|---|---|---|---|---|---|
| factor | 0.153 | 0.309 | 0.627 | 0.795 | 0.975 |

Bracket from a young root and every trial descended from it runs at that fraction
of its nominal rate. Burn-in is paid once; trials are paid N times.

### 2.2 The root

A complete host-resident snapshot: model + EMA weights, **all** optimizer states
(including Adam's step counter), prior/replay/anchor buffers, `condition_log_z`,
the `MetricTracker` (whole object — `written_at` and `changed_keys` are not in
its `state_dict`), the `GradClipGuard` (whole object — its fire counters drain on
read), `stage_ctrl`, every counter, the logical step, and all four RNG streams.

Two refusals, both loud, both leaving the run at `burn_in_scale` for the stage:

- **bias correction below `min_root_bias_correction`**, computed from the
  optimizer's *actual* step counter. A stage that did not rebuild its optimizers
  enters with `t` already large and passes trivially, which is correct.
- **the loss window is too short to derive a hard-failure bar** from
  (`hard_failure.min_observations`). A bracket that cannot fail a candidate is
  not a bracket.

### 2.3 Candidate trials

For each rung in `candidate_scales`, ascending: restore the root exactly, set the
rate once, hold it for `trial_steps`, record survival. Serial, on one GPU.

- Every candidate that has not hard-failed runs the **full** horizon. There is no
  early winner selection and no ranking by loss.
- A candidate may **not** rescue itself: a hard failure ends that trial and is
  never answered by lowering its rate.
- Every surviving candidate's end state is kept, on a uniform path.
- `steps_to_failure` is recorded for each failure. A failure past 60 % of the
  horizon publishes `horizon_marginal` — reported, never auto-extended.

`trial_steps` is deliberately short (50–250) and **has no floor**. The
`> 1/(1−β₂)` argument is about warming Adam's moments from scratch; every trial
restores a root whose moments are equilibrated. Cruise is the long trial.

### 2.4 Hard failure

The only conditions that can fail a candidate:

| condition | bar |
|---|---|
| non-finite loss or gradient | — |
| loss excursion | `root_hi + loss_excursion_k × (root_hi − root_lo)`, per branch |
| gradient excursion | `grad_excursion_x × root_grad_hi` |
| absolute backstop | `hard_failure.loss_abs` / `grad_abs` |
| an exception that makes the continuation unusable | — |

The excursion bars are **derived at bracket time** from the last `root_window`
steps of burn-in, not configured. A **span**, not a ratio: on the MLE channel the
loss passes through zero and goes negative, so a ratio has no positive scale to
work against, while `hi + k·(hi−lo)` is well defined whatever the sign.

`config_invariants.lr_bracket_is_well_formed` rejects an absolute bar at or above
1e8 — that catches numerical overflow and nothing else, and under it every rung
survives forever.

An OOM inside a trial is **not** a candidate failure: it says nothing about the
rate. It propagates and aborts the bracket.

### 2.5 Selection

Uses only the configured ordering and hard survival.

- **boundary** = the lowest rung that failed, re-run `boundary_confirm_repeats`
  times from the root under a **derived seed** (`root_step × 1000003 + 7919 × r`).
  A same-seed re-run would be a deterministic replay and confirm nothing. A rung
  that survives its repeat is a *non-reproducing failure*: the search continues
  upward from it, and it is not counted as a survivor either.
- **selection** = the highest survivor at least `safety_rungs` below the boundary.
- Survivors *above* a failure are ignored — non-monotone outcomes are treated
  conservatively.
- `boundary_densify` (off by default) inserts one geometric rung between the
  boundary and the survivor below it and trials it once. Never re-densified.

| outcome | meaning | result |
|---|---|---|
| `bracketed` | a confirmed boundary, selection below it | restore that candidate's end state |
| `unbracketed_high` | nothing failed — **no boundary was identified** | `safety_rungs` below the top rung |
| `all_failed` | every rung detonated | restore the root, hold `burn_in_scale` |
| `no_eligible_candidate` | a boundary, nothing survives below the margin | restore the root, hold `burn_in_scale` |

Nothing is interpolated between rungs.

### 2.6 Cruise

The promoted rate is held fixed. After promotion the run resumes at
`root_step + trial_steps` — **one horizon, not the sum of all trial compute**;
the host loop consumes its step iterator to match. `repeat_every` counts promoted
steps only.

A hard failure in cruise goes through the existing rewind path
(`fire_loss_spike`, bounded by `max_reloads_per_1k_steps`). **The rate is not
cut**: it was chosen by a bracket, and a silent cut hides that selection error.

---

## 3. Fixed mode

`mode: fixed` performs the configured burn-in, switches to `fixed_scale`, and
runs no trials and no re-bracketing. For batteries of related runs where the rate
is already known.

---

## 4. Configuration

```yaml
lr_control:
  mode: bracket              # bracket | fixed
  seed_lr: 1.25e-4           # what an `auto` lr_* key resolves to
  control_flow_lr: false

  burn_in_steps: 3000        # LONG: must reach Adam steady state
  burn_in_scale: 0.05
  min_root_bias_correction: 0.9

  candidate_scales: [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
  trial_steps: 150           # SHORT, no floor
  safety_rungs: 1
  repeat_every: 0            # 0 = once per stage. THE DOMINANT COST KNOB
  boundary_confirm_repeats: 1
  boundary_densify: false
  fixed_scale: 0.2

  hard_failure:
    loss_excursion_k: 10.0
    grad_excursion_x: 100.0
    loss_abs: 1.0e+6         # backstop only; >= 1e8 refused at load
    grad_abs: 1.0e+6
    root_window: 200
    min_observations: 20

  ray_calibration: { ... }   # DIAGNOSTIC only; off unless a stage asks
```

**Cost.** One cycle discards about `len(candidate_scales) × trial_steps` steps —
900 at the shipped values. `repeat_every` matters more than `trial_steps`: six
rungs at 1000 steps on a 10k clock is 60 % of a stage, which on the MLIP routes
can consume a 48-hour SLURM budget in calibration alone.

**Load-time gates** (`config_invariants.py`): the grid must be strictly
ascending, span ≥ 4×, and hold at least `safety_rungs + 2` rungs; the absolute
bars must be able to fire; `auto` rates require an `lr_control` block;
`burn_in_steps` short of Adam steady state is reported as BASELINE.

---

## 5. Telemetry

`lr_ctrl/scale`, `lr_ctrl/divergences`, `lr_ctrl/lr_capped_groups` (only when
`max_lr` is set), `lr_ctrl/lr_floored_groups`.

`lr_bracket/phase`, `/scale`, `/brackets`, `/discarded_steps`, `/promoted_steps`,
`/root_step`, `/root_bias_correction`, `/promoted_scale`, `/next_repeat_step`,
`/status`, `/boundary_scale`, `/boundary_confirmed`, `/non_reproducing`,
`/densified`, `/margin_rungs`, `/horizon_marginal`, `/refused`, `/held_mb`,
`/bar_loss_<branch>`, `/bar_grad`.

Plus a human-readable table of the whole bracket in the run log
(`LRBracket.summary`): one row per trial, each a rung and a boolean.

**No alpha\* or cosine is published beside the selection.** Neither entered the
decision, and publishing them would read as an explanation for it. If a stage
opts into a diagnostic sensor, `lr_ctrl/calibrations` / `lr_ctrl/hypergrads` /
`lr_ctrl/hyper_cos` appear — and only then.

---

## 6. Tests

- `tests/lr/test_lr_bracket.py` — the decision layer: burn-in exactness, the grid
  and bar refusals, all four selection outcomes, the confirmation seed, the hard
  caps on confirm/densify, `horizon_marginal`, fixed mode, the repeat clock.
- `tests/lr/test_lr_bracket_driver.py` — the trainer side against a real (CPU)
  training loop: **two candidates at the same scale produce bitwise-identical
  losses**, no contamination A→B→A, the Adam counter round-trip, `strict=True`
  raising rather than printing, the fixed rate through a trial, no self-rescue,
  the promoted clock advancing by exactly one horizon. Every guarantee is
  mutation-tested — ten mutations, ten convictions.
- `tests/lr/test_lr_absolute_cap.py` — the `max_lr` rail and the `min_lr` floor.
