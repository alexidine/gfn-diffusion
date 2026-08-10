# Audit: everything since `postfix_lr8x` (ty4xdlzo)

Boundary: commit **`433cbd5`**, run started 2026-07-30, config `postfix_july30/8.yaml`,
T=60, ran to step 46240. Four commits since, all with the same commit message, so
motivation had to be read out of the diffs.

**Bottom line: I found no unmotivated or dishonest code change.** Every code change since
the boundary is backed by a named run and a stated failure mode, and several are genuine
bug fixes. The problems are not in the code — they are (a) two changes that silently make
the baseline non-comparable, and (b) one interaction between a *new* subsystem and an
*independent* config change that contaminates exactly the measurement I built the
`ctrl_aug03` battery on.

---

## 1. Code changes — all justified

| # | change | motivation | verdict |
|---|---|---|---|
| 1 | `checkpointing.reconcile_batch_size` | p307hzip resumed with `batch_size: 1000` + `max_batch_size: 1000` and trained at **2831**, silently — growth only ever moved size up | fixes a real bug |
| 2 | `controller._report_bars` | tw_july31 arm 14 detonated between the cut and reset bars and parked forever; prints bars in clip units, hard-fails on inverted ordering | diagnostic, honest about not being a calibration |
| 3 | `controller.on_explosion(force_ratio)` | `cut_ratio: 1.0` + reset fire + no rewind target = nothing at all stops a detonation | closes a real hole |
| 4 | recovery-inert warning | `recovery_target_frac <= cut_ratio` makes the whole recovery path dead code, silently | see §2 — this one has teeth |
| 5 | `_CUT_GRAD_OVER_CLIP` 30 → 100 | at 30 the "thrash" bar sat at 1.14× the tabulated grad median; 44gt5whr's only fire was 13% over and cost a permanent 2× LR cut for 17k steps | see §2 |
| 6 | `relative_under_wcen`, `ramp_ess_frac` | `relative_under` inherits a floor from batch composition (`f*Delta`), which is a buffer knob, not a policy one | additive, both metrics kept |
| 7 | `logw_std_within` folded into `quick_tb_stats` | consolidation | clean — verified **zero** orphaned callers of the removed `within_condition_logw_std` |
| 8 | `_frozen_training_state` | 8 dead tw_july31 arms held a bitwise-identical grad norm for 1052–1800 samples; 4 healthy arms never repeated once | empirically separated, well-calibrated |
| 9 | replay eviction: TTL → hazard + stalled + backstop | the hard TTL was doing 100% of eviction and culling the improving tail (`expired_delta` −12…−28 nats, `absorbed_frac` 0.000) | motivated |
| 10 | `z_calibration` (new subsystem) | the "more Z updates per policy update" actuator | see §3 |

---

## 2. Two changes that make the baseline non-comparable

Neither is a defect. Both mean *"do not compare against ty4xdlzo on this axis."*

**LR tripwires moved 3.3×.** The baseline ran `cut_grad_abs` 13631 = 30 × clip. Under
current code the same config resolves to 45439. Any comparison involving LR cuts crosses
a moved goalpost.

**Recovery was INERT in the baseline.** ty4xdlzo ran `recovery_target_frac` 0.5 with
`cut_ratio` 0.5 — exactly the condition change #4 now warns about. So in the baseline, an
LR cut was permanent for the stage. mk_dev now runs 0.85/0.5, which is live. The baseline
and current runs have *qualitatively different* LR dynamics after any fire.

---

## 3. THE FINDING: z_calibration multiplies replay intake, and the buffer shrank at the same time

`_z_rollout_step` calls `manage_replay_buffer` **once per calibration step**. Its own
docstring flags this and predicts the failure:

> "churn_rate is a per-CALL budget, so intake multiplies by the number of calibration
> steps in the tick … transiently compressing buffer turnover by the same factor. Bounded
> in practice … **but unbounded in principle. Watch replay_buffer_admitted** … **If the
> buffer starts reading as one-instant:** …"

That bound is not holding here. Measured across all five `ctrl_aug03` arms:

- at the `train_prior → naive` transition the z_cal sensor reads **6.67 nats** — 13× its
  0.5 threshold (this is the bwd→fwd Z handoff: phase 1 trains Z via `mle`/`tbc`, naive
  switches it to forward TB)
- z_cal therefore saturates at **8–30 steps per train step for ~300 steps**
- each of those steps admits up to `churn_rate` = 80 rows

So the replay buffer receives up to ~2400 admissions per train step into a **4000-row**
store — a complete flush every ~2 steps, for hundreds of steps, right at the point where
the naive stage's replay statistics begin.

The baseline had **none** of this: no z_calibration at all, `max_size` 10000, batch 1000.

**Consequence for `ctrl_aug03`:** the whole battery was scored on `replay/fwd scatter_err`,
and that ratio is inverted (0.90–0.92) at step 6690 — the first reading, ~10 steps in,
before any training pressure could accumulate. I read that as "memorization is immediate."
It is at least as consistent with "the metric is measuring an instrument." The z_cal
docstring even names the mechanism as its remedy #2: these rollouts fire *exactly when Z
is off*, so their residuals carry a level offset, and admission scored on `|resid|` reads
that offset as badness rather than measuring per-sample miscalibration.

Supporting: the baseline's Z sits at `fwd/tb_resid` **+0.079** (`tb_resid_clipped` 0.0067).
Ours sit at **+3.15…+3.70**. Different regime entirely.

---

## 4. Config drift, mk_dev vs the baseline

mk_dev is user-owned; this is a record, not a criticism.

| key | ty4xdlzo | mk_dev now | factor |
|---|---|---|---|
| `integrator.T` / `eval_T` | 60 | 10 | deliberate |
| `batch_size` | 1000 (grow → 50000) | 2831 (pinned) | ×2.8 |
| replay `max_size` | 10000 | **4000** | ÷2.5 |
| replay `churn_rate` | 50 | 80 | ×1.6 |
| replay residence | `max_residence_steps: 250` | `mean_residence_steps: 50` | redesign |
| **draws/row/step** | **0.10** | **0.71** | **×7** |
| naive `replay` frac | 0.048 | 0.20 | **×4** |
| naive `fwd` coeffs | `{tb: 1}` → trains policy | `{tb: 1, freeze_policy: 1}` → **Z only** | reversed |
| `admit_reward_min` | −600 | −50 | ×12 tighter |
| prior `reward_min` | −600 | 0 | |
| `condition_block_m` | 2 | 1 | |
| `reward_range` | 500 | 250 | ÷2 |
| balance metric (bwd) | `relative_under` | `relative_under_wcen` | |
| balance `alpha` ¹ | 0.002 | 0.005 | ×2.5 |
| `default_boost` ¹ | bwd .94 / replay .06 | .75 / .25 | |
| `max_fracs.replay` ¹ | 0.2 | 0.45 | |
| `hold_steps` | 4000 | 20000 | |
| `decay_halflife_steps` | 0 (none) | 25000 | |
| `recovery_target_frac` | 0.5 (inert) | 0.85 (live) | |

¹ These three rows are **frozen at the audit date.** `mk_dev` moved to
`kind: ratio` on 2026-08-06 (`module_protocol.md` P7), which has no `alpha`,
`default_boost` or `max_fracs` — the split's bounds are `bounds: {replay:
[0.05, 0.45]}` and its rate is `gain: 0.02`. The comparison above still records
what the two runs actually ran, which is what an audit is for; it no longer
describes the live config.

**The replay inversion is the product of three independent config moves**, none of which
is wrong alone: `max_size` ÷2.5 × `batch_size` ×2.8 × `replay_frac` ×4 ≈ **28× the
memorization pressure per step** relative to the baseline.

Note also that mk_dev's `naive` made fwd **Z-only** (`freeze_policy: 1`), which the
baseline did not. `ctrl_aug03` set it back to 0 — so on that axis my battery matches
ty4xdlzo, not mk_dev.

Two stale comments worth fixing: mk_dev's `cut_grad_abs` comment still says
"`auto -> 30 * gradient_norm_clip`" (now 100), and `lr_back`'s says the auto resolver is
the 1/T end.

---

## 5. train.py, all 21 hunks — complete

Every hunk accounted for. Nothing unmotivated, nothing that changes training behaviour
without a stated reason.

New subsystems (the bulk of +621): `z_calibration` + `_z_rollout_step`,
`_frozen_training_state` + `FrozenTrainingState`, `_per_step_probe`, the replay
eviction redesign, `max_reloads`.

Modifications to pre-existing code, all benign:

- import cleanup for the removed `within_condition_logw_std`
- `_update_rolling` / `_eval_conditional_stats`: duplicated within-condition computation
  deleted now that `quick_tb_stats` does it — behaviour-preserving
- `_grad_nonfinite_streak` added and reset on any finite gradient — feeds
  `_frozen_training_state` channel 1
- frozen check placed **before** the rewind tiers, with the reason stated (a rewind
  resets the signature and restarts the patience clock)
- `replay_cohort` keys re-split for the new eviction causes (`stalled`/`backstop`)
- new replay telemetry: `age_cv` (residence-distribution *width*, the quantity the
  redesign actually moves), `live_delta_mean`, `live_delta_stalled_frac`
- `fire_loss_spike` now counts **all** rewinds, not just terminal ones — motivated by an
  observed reset-tier rewind loop (aug02 `a2_T25_lr16_tight` / d7z705wc)
- array-valued metrics restricted to the `eval_period` grid — cosmetic wandb fix
  (off-grid evals made the histogram-over-time panels illegible); scalars still log

One judgement call worth knowing about rather than disputing: `_z_rollout_step` uses
`return_exp=True`, a D2H copy of the crystal batch per calibration rollout. The docstring
owns it ("the price of admission, literally"). It is a throughput cost, not a
correctness one.

---

## 5b. Two hypotheses tested and FALSIFIED (2026-08-03, after the audit)

**Sizing/reuse is NOT the cause.** `sz_hi_f05` matched the baseline's draws/row/step
(0.081 vs 0.10) — occupancy ~35000 at unchanged residence, no servo, no controller — and
the ratio still did not reach 1:

```
fx_static  (0.71 draws/row/step)  0.908 0.934 0.899 0.783 0.753 0.738 0.743
sz_hi_f05  (0.081)               0.806 0.933 0.951 0.901 0.900 0.878 0.875
fx_servo   (0.75, fast eviction) 0.912 0.945 0.915 0.867 0.963 0.951 0.916
```

The correctly-sized arm is still *worse* than the servo arm and nowhere near the
baseline's 1.93.

**"Ours are just less converged" is ALSO false.** The baseline's ratio was pulled as a
function of steps-since-naive-entry. It never inverts, at any point:

```
since_entry   fwd_sc  rep_sc  ratio
          0     16.2    18.6  1.149     <- already >1 at entry
        400     13.8    32.6  2.354     <- RISES as the buffer fills
       2000     10.9    17.2  1.567
      36000     4.47    8.24  1.845
```

At naive entry the baseline's `fwd/scatter_err` was **16.2**, the same order as our
21 — so this is not a convergence-level artifact. At comparable scatter the baseline sits
at 1.15 and climbs while ours sit at 0.91 and fall.

The admit-cap regime difference is real but insufficient: baseline cap/sigma ~1.2 early
vs ours ~0.70, a factor 1.7 (an earlier note said 9x — that compared against the
baseline's *converged* state and overstated it).

**What does line up is buffer-fill dynamics.** The baseline's `replay_buffer_length` was
still filling at 330 steps past entry (churn 50 into `max_size` 10000) and its ratio
*rises* to 2.4 as prioritized samples accumulate. Ours is full within ~2 steps —
`max_size` 4000 with z_cal multiplying intake 8-30x through the handoff — so it never
accumulates a prioritized population; it is a near-instantaneous snapshot. That is
precisely the "buffer reading as one-instant" condition §3's docstring warns about.

Direct test prepared: `configs/size_aug03/zc_off.yaml` — one variable off `fx_static`
(`z_calibration.enabled: false`), same buffer, same mix, same resume, 2000 steps.

## 5c. The size_aug03 2x2, complete (all cells, matched at step 8680)

`replay/fwd scatter` ratio:

| | fwd 0.05 | fwd 0.20 |
|---|---|---|
| **lo sizing** (0.71 draws/row/step) | 0.743 | 0.772 |
| **hi sizing** (0.081) | 0.875 | 0.892 |

Main effects **additive, no interaction** (sizing +0.126, fwd_frac +0.023, interaction
−0.012). The two factors are orthogonal, as predicted. **Neither, nor both, reaches 1** —
best is 0.892 against the baseline's 1.93.

Forward fit improves additively and `sz_hi_f20` is the best forward-side arm of anything
run today:

| | fx_static | sz_hi_f05 | sz_lo_f20 | **sz_hi_f20** |
|---|---|---|---|---|
| fwd `scatter_err` | 22.42 | 21.22 | 19.68 | **19.04** |
| fwd `tb_err` | 22.73 | 21.49 | 19.95 | **19.23** |
| fwd `over_coverage` | 22.05 | 20.80 | 19.28 | **18.56** |
| bwd `relative_under_wcen` | 4.073 | 3.816 | 4.115 | 3.847 |
| EffDim | 5.894 | 5.841 | 5.778 | 5.790 |

**RETRACTED: the fwd_frac -> Z effect.** On the lo-sizing pair `fwd/tb_resid_clipped`
went −0.303 -> −0.153 and I called it "4x the fwd weight halved the Z residual". The
hi-sizing pair does **not** replicate it (−0.364 -> −0.359). Across all five arms the
clipped residual reads −0.303 / −0.364 / −0.153 / −0.359 / −0.423 with no fwd_frac
pattern — `sz_lo_f20` is an outlier, not a trend, and at this magnitude one seed cannot
separate it from noise.

What IS consistent in both pairs is the RAW `fwd/tb_resid` falling with fwd_frac
(3.703 -> 3.244 at lo; 3.348 -> 2.663 at hi) — but that is the winsorization-offset
quantity (§ [[project-huber-winsorization-z-offset]]), not the Z error. So: fwd_frac
0.20 clearly improves forward fit; its effect on Z convergence is **not established**.

Also ruled out: `reward_rejected` is flat across all arms (1.39-1.92e4), so the tightened
`admit_reward_min` is not the differentiator either.

## 5d. z_calibration falsified too — and it is HURTING Z

`zc_off` (one variable off `fx_static`: `z_calibration.enabled: false`) reproduces the
inversion exactly: ratio **0.705** vs 0.743, same trajectory shape and decay. z_cal is
not the cause of the replay inversion.

But it settles the Z question, in the opposite direction to my earlier reading:

| | `fx_static` (z_cal ON) | `zc_off` (z_cal OFF) |
|---|---|---|
| `fwd/tb_resid_clipped` | −0.303 | **+0.010** |
| `gradnorm/flow_model` | 0.031 | 0.0007 |

**With z_calibration off, the fused fwd gradient alone puts the clipped residual at
+0.010 — dead on its Huber fixed point — using ~40x less Z gradient work.** z_cal's
early-out disarms at `threshold * grace` = 0.4, so it PARKS Z up to 0.4 nats off the
optimum the fused gradient would otherwise reach. The deadband is not a benign floor; on
this route, running z_calibration is worse than not running it.

(Caveat: `zc_off`'s overall fit is marginally worse — fwd `tb_err` 22.89 vs 22.73, fwd
scatter 22.49 vs 22.42 — but those gaps are inside the ~1-2% noise floor established by
the fx_servo/pp_servo near-replicate pair, while the Z difference is 30x that.)

## 5e. Everything hypothesised has now been falsified

| hypothesis | test | verdict |
|---|---|---|
| reuse / buffer sizing | `sz_hi_f05`, `sz_hi_f20` | falsified (helps +0.13, never reaches 1) |
| under-convergence | baseline history at matched scatter | falsified (baseline 1.15 at scatter 16.2) |
| fwd_frac / Z | size_aug03 2x2 | falsified (+0.02, no interaction) |
| tightened `admit_reward_min` | `reward_rejected` flat across arms | falsified |
| z_calibration intake | `zc_off` | falsified |

**The one factor never varied in isolation is `replay_frac`**, and the cross-battery
evidence points straight at it:

    replay 0.048  (ty4xdlzo baseline)      ratio 1.93
    replay 0.10   (cs_servo)               ratio 0.966   <- best of ctrl_aug03
    replay 0.20   (fx_servo / fx_static)   ratio 0.916 / 0.743

`rf_low` (one variable off `fx_static`: fracs 0.05/0.902/0.048, the baseline's replay
weight) is running to test it.

Remaining untested difference after that: **T = 10 vs 60**, which changes the residual's
tail structure and hence what prioritized admission has to select from.

## 5f. RESOLVED: the inversion is replay LOSS WEIGHT, and sizing only compensates for it

Final one-variable series off `ctrl_aug03/fx_static`, all at step 8680:

| arm | change | replay/fwd |
|---|---|---|
| fx_static | — (replay 0.20) | 0.743 (decaying) |
| zc_off | z_calibration off | 0.705 |
| sz_hi_f05 | occupancy 4000 -> 35000 | 0.875 |
| fx_servo | churn servo, boost 12 | 0.916 |
| **rf_low** | **replay_frac -> 0.048** | **0.948 (flat)** |
| **rf_low_hi** | **replay 0.048 + occupancy 35000** | **0.952** |

**Mechanism** (visible in the buffer): at replay_frac 0.20 the replay branch trains down
the residuals admission selected for, faster than churn replenishes them — the buffer
eats its own signal. Dropping the weight to 0.048 lifts `buf_mean_loss` 15.13 -> 23.55
and `replay/tb_resid` 7.23 -> 17.84.

**Sizing and loss weight INTERACT, they do not compose.** Sizing is worth +0.13 at
replay 0.20 and **+0.004** at replay 0.048. Occupancy only matters while the loss weight
is too high. The composite `replay_frac * batch/occupancy` I floated is wrong — discard it.

**Residual:** the ratio plateaus at ~0.95, short of 1.0 and of the baseline's 1.93. The
only structural difference left is **T = 10 vs 60**. Untested — needs a T=60 phase-1
prior, which does not exist locally.

**Side result worth keeping:** low replay_frac is a coverage-vs-calibration dial.
`rf_low_hi` has the best EffDim (5.974) and best `wass_debiased` (0.0079) of every arm
run today, and the worst fwd `tb_err` (23.69). It also corrects the ctrl_aug03 headline:
allocation DOES move bwd absorption (4.073 -> 3.702) once you span a wide enough range —
the ctrl_aug03 arms were capped at replay 0.10 by the constraint controller's own bounds.

## 6. What I'd do about it

1. **Re-test the `ctrl_aug03` conclusion with z_cal disabled**, since the ratio it was
   scored on is contaminated in exactly the window that mattered. Cheap: one arm.
2. **Adopt the z_cal docstring's own remedy #1** — pool the tick's rollouts and the train
   step's fwd batch into ONE `manage_replay_buffer` call at the normal `churn_rate`.
   That removes the intake multiplication entirely and makes the draw *more* selective.
3. The `size_aug03` battery already running is still the right test — `sz_hi` lands at
   **0.081** draws/row/step, which matches the baseline's 0.10 almost exactly.
