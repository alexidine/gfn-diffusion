# Module: LR controller (`controller.py`)

Pass 1 (audit + rationalize) 2026-08-03. **Rewritten 2026-08-08 for v7** — the
middle layer and the decay leg were deleted and the α\* servo was built, so most
of what pass 1 documented no longer exists. Unconditional route only. Empirical
claims tagged by warrant class.

Companion: [`step_probe.py`](../step_probe.py) is the sensor and has its own
module docstring; `to_do_rebuild.md` §A holds the design argument.

---

## 1. What it is

`LRController` owns every learning rate in the run. **Two regimes, nothing in
between** (`decisions.md` D4):

| Regime | Sensor | Response | Cost of a false positive |
|---|---|---|---|
| *slightly hot* | α\* < target | multiplicative cut of `peak` (one-sided since F8 — it can only fall) | ≈ 0 — nothing is discarded |
| *diverged* | non-finite, or past ~1e9 | reload the checkpoint **and** cut `peak` | progress since the last checkpoint |

On top of that sits a per-stage **warmup envelope** (exponential ramp → hold at
1.0, forever). The peak is a servo state, not a config constant, whenever the
config writes `auto`.

**As shipped the servo is a one-sided BRAKE, not a growth loop** — `clip: [0.8,
1.0]`, so `peak_scale` can only fall. The growth half was built, measured, and
withdrawn: see F6 (it costs ~2 nats) and F8 (it runs away). `seed_lr` is
therefore the operating LR rather than a starting point.

Paper voice: *the learning rate follows a short per-stage warmup and is then held
under a ceiling derived from a two-point step probe, which measures whether the
optimizer's own step was the right size; a coarse absolute divergence bar reloads
the last good checkpoint and cuts the rate.*

## 2. Contract

**Reads** — `modeller.step_ind`, `.phase`, `.optimizers`, `.step_probe`,
`.args.{lr_*, min_lr, lr_warmup_ratio, lr_servo_managed, adaptive_lr.*}`;
per-call `(step_type, current_loss, grad_norm)`.
**Writes** — `param_group['lr']` for every group in every optimizer;
`modeller.lr_ctrl` (versioned dict, `ver: 7`); an `lr_ctrl/*` metric block.

**Invariants**

- `envelope ≤ 1.0` always. The envelope is a warmup, never a climb.
- Every non-flow group's LR ≥ `min_lr`.
- Flow (Z-head) groups sit flat at `lr_flow` unless `control_flow_lr: true` —
  α\* is measured over **policy** parameters only (D26 b), so the servo has no
  sensor mandate over Z.
- `peak_scale` applies **only** to groups whose config key was written `auto`.
  A float is a fixed peak.
- The **ceiling** is instance state, never in `lr_ctrl`: the rewind that follows
  a divergence restores `lr_ctrl` from a healthy checkpoint and would otherwise
  erase the record that this LR just detonated (the djr13t0j sawtooth).
  `peak_scale` *is* checkpointed — a resume should keep the climb — and is
  clamped to the live ceiling on every read, which is what makes that safe.
- The envelope is a pure function of `step_ind - stage_start_step`, so the
  10-step call cadence cannot change what a config value means.

## 3. Mechanism

| Piece | Where | Behavior |
|---|---|---|
| Envelope | [`_envelope`](../controller.py:293) | `elapsed < warmup` → `(1/lr_warmup_ratio)^(1-frac)`; then flat 1.0 forever |
| Servo | [`_advance_servo`](../controller.py:315) | every `period` steps: `peak_scale *= clip(median α* / target, clip)`, damped by distance below the ceiling |
| Actuator | [`_apply_lrs`](../controller.py:207) | `lr = max(min_lr, base × envelope × (peak_scale if managed else 1))`; flow groups pinned |
| Divergence | [`check_spike`](../controller.py:153) | non-finite, or past `divergence_{loss,grad}_abs`. No cooldown, no latch |
| Cut | [`on_divergence`](../controller.py:177) | `peak_scale *= divergence_cut`, ceiling recorded here |
| Ceiling | [`_current_ceiling`](../controller.py:304) | relaxes upward at `ceiling_halflife_steps` |
| Stage reset | [`rearm_warmup`](../controller.py:263) | restarts the warmup clock, **forgets the ceiling, carries `peak_scale`** |

**The servo holds rather than guesses**, and says which: `lr_ctrl/servo_hold`
encodes `disabled / no_probe / warmup / cold / few_readings / fit_invalid`. That
is not decoration — this doc set now records three separate instances of *an
unreadable sensor and a satisfied controller producing identical silence*, and
the servo emits its actuator (`peak_scale`) alongside its sensor for the same
reason.

Two hold conditions carry real content:

- **`warmup`.** The envelope is below 1 during the ramp, so α\* rates a
  deliberately shrunken step and reads high. Acting on it would inflate
  `peak_scale` by exactly the warmup factor and then hand that back as a real LR
  the instant the ramp completed.
- **`fit_invalid`.** A windowed flat/downward/non-finite rate above
  `max_bad_rate` voids the sensor regardless of what the α\* values say (§A3a.3):
  *flat* means the probe is under-resolved, *downward* means the local quadratic
  model is wrong. `StepProbe.servo_reading()` returns that rate over the same
  window as the median, so the controller never has to infer validity from the
  median's behaviour.

## 4. Knobs and liveness

| Key | mk_dev | Live when | Warrant |
|---|---|---|---|
| `warmup_steps` | 1000 | every stage entry | **arbitrary** — the cold-Adam rationale is standard practice, the value is not calibrated |
| `servo.seed_lr` | **1.25e-4** | resolves every `auto` key | **measured** — the `fwd/tb_err` optimum at T=10 (F6). Under a one-sided clip this is the OPERATING LR, not a starting point: the servo never climbs back |
| `servo.target` | 1.0 | always | **measured, and it is now a CEILING not a setpoint** — α\* actually reads ~1.87 at `seed_lr`, so 1.0 sits ~5 SE below and binds only past lr ~2.3e-4. A target *at* the operating α\* would ratchet (F8) |
| `servo.period` | 200 | always | derived-ish — probe autocorrelation time is ~20–30 steps, so a 25-probe window at cadence 20 holds ~20 independent samples |
| **`servo.clip`** | **[0.8, 1.0]** | always | **measured — the structural fix.** `clip_hi: 1.0` makes the multiplier ≤ 1, so `peak_scale` can only fall and F8's positive feedback loop cannot form. [0.8, 1.25] restores the growth servo and needs D32's guard first |
| `servo.min_readings` | 10 | always | arbitrary |
| `servo.max_bad_rate` | 0.5 | always | arbitrary |
| `servo.bounds` | [0.1, 200] | only the LOWER bound can bind under a one-sided clip | off `seed_lr` 1.25e-4 the floor is 1.25e-5 — where a sustained brake would walk to. The ceiling is now inert |
| `servo.ceiling_halflife_steps` | 20000 | after a divergence | arbitrary |
| `divergence_*_abs` | 1e9 | always | **derived — deliberately uncalibrated.** D4: *"if we are only looking at hard blow-ups we can use almost any metric"* |
| `divergence_cut` | 0.5 | on divergence | inherited |
| `control_flow_lr` | false | always | **derived (unconditional)** — the flow head should be fast, and there is no downside on a `LearnableScalar` |
| `lr_warmup_ratio` | 10 | every stage entry | arbitrary now — the 1e-5-seed argument for it went with the seed. At `seed_lr` 1.25e-4 a ratio of 100 is again reachable above `min_lr` |

`_check_bars` **refuses** a divergence bar below 1e5. A graduated divergence bar
is the deleted cut tier coming back in through the config, and this is the one
place a config could reintroduce it silently.

## 5. Findings

**F0 — the step probe is not precision-limited, and its dispersion is wide.**
*(measured 2026-08-08, read-only across 16 `batt0807` runs: 1,855 α\* and 2,113
`second_diff_rel` readings)*

`second_diff_rel` sits at a median of **3.6e-2** against `step_probe.py`'s 1e-6
floor, with **0.28%** of probes below it. The parabola resolves real curvature,
not float32 rounding, so `D27`'s kill-gate **clears**.

Dispersion is the consequential half. Within-run relative IQR of α\* is
**0.5–1.0** (pooled 1.45), lag-1 autocorrelation ≈0.5 at cadence 20. Per §A4c
that is the **wide** branch: servo-on-median, not a line search. And **§A4's
`clip(median, 0.9, 1.1)` was sized at roughly one standard error of the median it
clips** — with IQR ≈0.6 over a 25-probe window, SE(median) ≈ 9%. The shipped
lower clip is 0.8 for that reason; the *upper* clip is 1.0, and that is F8's
doing rather than F0's.

Caveat: every source run is 3.5k–4.6k steps and `alpha_median` was still moving
at 75% of phase 2, so the dispersion figure is probably an over-estimate.

**F5 — the setpoint is the open question, and it is the one thing the servo
cannot derive.** *(measured 2026-08-08, `local_aug08` pair D — the same
measurement written up in `decisions.md` D30)*

The scaling law **passes**: raising `lr` 1.72× divided `alpha_median` by
**1.73** against a predicted 1.72. The sensor is measuring what it claims to.

But **following α\* made both arms worse.** The probe read ~1.7 at base LR —
"your step was 1.7× too small" — and taking exactly that step degraded
`bwd/tb_err` by 0.6–0.9 nats in both the frozen and unfrozen rows.

The reason is not that the sensor is wrong. **α\* is a local property of one ray
on one frozen batch, and raising `lr` changes the whole trajectory** — Adam's
second-moment state, the noise scale, the batch-to-batch step direction. The
local optimum along the taken step is not the best global LR. That is §A6's
objection, now measured rather than argued.

So `target` ships as a config key rather than a hardcoded 1.0. Three postures,
all reachable without a code change:

| Posture | Config | What it assumes |
|---|---|---|
| §A4 as written | `target: 1.0` | the local optimum is the global one |
| standing undershoot | `target: 1.5`–`2.0` | it is not, by a roughly constant factor |
| **one-sided brake** | `clip: [0.8, 1.0]` | α\* is trustworthy as a *ceiling* and not as a growth signal |

The third is what D3's ruling anticipated (*"the servo still gets built; what
changes is that it needs a separate hard ceiling"*) and what the pair-D
measurement most directly supports. **mk_dev ships `target: 1.0` so the ladder
gets measured rather than assumed.**

**F6 — the loop is correct, the setpoint is wrong, and the two are now separated
by a direct A/B.** *(measured 2026-08-08, `lr_aug08` pair A: two arms, 5400 steps,
one shared post-transient resume, identical in everything but what owns the LR)*

**Validity first.** `a_fixed` (LR pinned at 1.25e-4, servo off) finishes at
`bwd/tb_err` **15.04** against `local_aug08` `a_frz`'s **15.14** on the same
resume — a 0.10 gap against a documented seed floor of 0.04. **The v7 rewrite
changed the LR path and not training**, so everything below reads on something.

**The control loop is textbook.** `a_climb` seeded 12.5× low climbed 1e-5 →
3.2e-4 and parked:

| | |
|---|---|
| `alpha_median` at the landing point | **1.006** against `target: 1.0` |
| `peak_scale` | 32.2, against a `bounds` ceiling of 200 — a **fixed point, not a bound** |
| `lr_ctrl/servo_hold` | 0 for the entire run — never held, never blind |
| `fit_ok_rate` | 0.11 → **0.80**, with `fit_beyond_rate` 0.88 → 0.20 |

That last row is §A3b's predicted signature — `beyond` giving way to `ok` as the
climb brings the LR into the probe's resolving range — and it is the cleanest
confirmation available that the sensor fix was the right one.

**And `α* ∝ 1/lr` holds tightly *within* an arm.** Over `a_climb`'s eight
uncensored rungs (`fit_ok_rate` ≥ 0.5, spanning 4.2× of LR), `lr × α*` has median
**3.07e-4** with a spread of −5% / +18%, the outlier being the marginal-fit rung
at the low end. So the law `local_aug08` pair D established across two arms holds
inside one.

> **⚠ But `lr × α*` is NOT a route constant — it depends on the policy state
> too.** `a_fixed`, sitting at 1.25e-4 for its whole run, has its own product of
> **2.34e-4** — **24% below** `a_climb`'s, at overlapping learning rates. The
> difference is what the policy had been trained at: `a_climb` measured 1.25e-4
> while *passing through* it from below, `a_fixed` while *living* there.
>
> This is the sharpest limit on calibrating `target` from one run, and pair D
> tests it directly: with `target: 1.87` (taken from `a_fixed`), the arm lands at
> **1.25e-4 if `a_fixed`'s constant transfers** and at **1.64e-4 if `a_climb`'s
> does**. Those are distinguishable, so the arm answers whether a calibrated
> setpoint is a property of the *route* or only of the *run it was measured on*.

> **⚠ `alpha_median` is CENSORED where `fit_ok_rate` is low, and censored
> downward.** A `beyond` fit contributes exactly `span` — a *lower bound* on α\*,
> not an estimate — so a window mixing bounds with real values reports something
> between the two. Measured on `c_low` (lr 1.56e-5, `fit_ok_rate` 0.43):
> `alpha_median` reads **3.5** where the law predicts ~15. Harmless for the
> *servo* (`span` is above any sane target, so growth is still licensed, just at
> the clip rather than proportionally) and **not** harmless for anyone measuring
> α\*(lr). The low-LR end of the sweep above is affected — its `lr × α*` products
> are the smallest in the table, which is exactly the direction the bias
> predicts, so the law is probably tighter than 2.5–3.3e-4 suggests. `read.py`
> now marks censored rows; the rule is **read `alpha_median` with
> `fit_ok_rate`, and below ~0.5 treat it as a lower bound.**

**The verdict, and it is a direct comparison rather than an inference:**

| at 5400 matched steps | `a_fixed` (1.25e-4, hand-set) | `a_climb` (3.2e-4, α\*-chosen) | Δ |
|---|---|---|---|
| `bwd/tb_err` | **15.04** | 17.20 | **+2.16** |
| `fwd/tb_err` | **17.89** | 20.73 | **+2.84** |
| `fwd/logw_std` | 17.68 | 20.43 | +2.75 |
| `replay/tb_err` | 15.00 | 14.90 | −0.10 |

**Following α\* to `target: 1.0` cost 2.2 nats of backward calibration and 2.9 of
forward, against a hand-set LR, with everything else held.** This is a much
stronger form of pair D's 0.6–0.9 nat result: there the setpoint was approached,
here it was reached and held to 0.6%.

**So the sensor is fine and the setpoint is the entire content of the design.**
α\* transfers as a *shape* — it will tell you one LR is 3× hotter than another
without running both, and that is a real instrument. It does not transfer as a
*setpoint*, and §A4's 1.0 is off by a factor of ~2.6 in LR on this route.

**Where the optimum actually is — and why the sweep could not say.** `a_climb`'s
in-run sweep has `bwd/tb_err` rising monotonically across all 20 rungs, which
reads like "lower is better all the way down". That sweep is taken under a
*rising* LR, though, and `a_fixed` at a *constant* 1.25e-4 **improves** over the
same window. A curve measured while the LR is moving is not a steady-state curve.
An earlier draft of this entry read the sweep as locating an optimum at ~1.5e-5;
that was over-claimed, and `c_low` — 1.563e-5 held for a full run — settles it:

| at 5400 steps | `c_low` 1.56e-5 | `a_fixed` 1.25e-4 | `a_climb` 3.2e-4 |
|---|---|---|---|
| `bwd/tb_err` | **13.33** | 15.04 | 17.20 |
| `fwd/tb_err` | 18.73 | **17.89** | 20.73 |
| `fwd/logw_std` | 18.48 | **17.68** | 20.43 |

**The two branch metrics disagree, and that is the substance.** `bwd/tb_err` is
monotone in LR — but a policy that barely moves fits a *fixed* buffer better, so
lower is nearly guaranteed to win there and the metric cannot arbitrate a LR
choice on its own. `fwd/tb_err` is the on-policy branch and it is **U-shaped**,
with a minimum at 1.25e-4. So the reference LR is roughly right, the "lower is
better" reading was a bwd-only artefact, and the servo's 3.2e-4 is worse than
both by a clear margin.

> Two honesties. The U is **shallow** between 1.56e-5 and 1.25e-4 — 0.87 nats —
> and `c_low`'s `fwd/tb_err` was **still falling** at the end (19.65 → 18.78)
> while `a_fixed`'s had flattened, so a longer run could close or invert it.
> `c_verylow` (5e-6) was cut to make room for the calibrated-target pair once
> `b_descend` turned up something more important (F8).

**Consequence for what ships.** `target: 1.0` is refuted. Of F5's three postures
the one now best supported is the **one-sided brake**, `clip: [0.8, 1.0]`: the
configured float is the operating LR, and the servo only ever acts to stop it
exceeding the α\*=1 line — which this measurement places at ~3.2e-4, comfortably
above anything anyone would hand-set. That makes `target` mean *"the hottest LR
you will tolerate"*, which is a far more defensible thing to set than *"the
optimum"*, and it is what D3's ruling anticipated.

**F8 — 🔴 `target: 1.0` sits inside a POSITIVE FEEDBACK LOOP, and α\* cannot see
the failure it is driving.** *(measured 2026-08-08, `lr_aug08` b_descend — the
single most important result in this battery)*

`b_descend` descended to ~3.1e-4 as designed, held there for ~2000 steps, and
then ran away. Binned over the run:

| bin | `lr_fused` | `fwd/tb_err` | `bwd/tb_err` | `alpha_median` |
|---|---|---|---|---|
| 4 | 3.11e-4 | 21.03 | 17.47 | 1.006 |
| 7 | 3.70e-4 | 21.27 | 16.81 | 1.024 |
| 8 | 3.95e-4 | **24.56** | 16.19 | 1.024 |
| 9 | 3.66e-4 | **29.53** | 15.39 | 0.917 |
| 10 | 3.85e-4 | **32.69** | 15.33 | 1.135 |
| 11 | 4.49e-4 | **34.94** | 15.21 | 1.032 |

**The forward branch blew out 66% — `fwd/tb_err` 21 → 35, `fwd/logw_std` 20.8 →
33.6 — while `alpha_median` sat at 0.92–1.14 throughout.** The sensor never
registered it. Neither did `bwd/tb_err`, which *improved* monotonically across
the whole collapse (17.9 → 15.2): the backward branch fits a fixed buffer better
as the policy stops moving coherently, so it reads the failure as progress.

**And the loop is self-reinforcing.** As the policy degrades the local curvature
flattens, so α\* rises above target, so the servo raises `lr` — which is exactly
what the table shows, 3.1e-4 → 4.5e-4 *during* the blowout. Degradation feeds LR
feeds degradation.

**Nothing in the containment layer catches this.** The divergence bar is at 1e9
and `logw_std` reached 33.6; `_frozen_training_state` keys on bitwise-constant
gradients and these keep changing. The deleted `_terminal_policy_state` would not
have caught it either — its `logw_std` bound was 1000. This is a slow degradation,
not an explosion, and this module has never had an instrument for one.

**Two consequences, and they are the shipping decision:**

1. **The one-sided brake is not a preference, it is a structural fix.**
   `clip: [0.8, 1.0]` makes the multiplier ≤ 1 always, so `peak_scale` can only
   ever fall. Every LR rise in the table above becomes impossible by
   construction, and the positive feedback loop cannot form. That is a much
   stronger warrant than "pair D says growth is unreliable".
2. **The servo needs a guard that is not α\*.** α\* is a local property of one
   ray; a forward-branch collapse is a property of the sampling distribution, and
   the first is structurally blind to the second. A cheap candidate already
   logged every step: `fwd/logw_std` rising materially above its own recent
   level, which moved 20.8 → 33.6 here while every LR-side signal stayed clean.

**F7 — holding a controller while its sensor keeps buffering is not holding it.**
*(found 2026-08-08 on `lr_aug08` b_descend; fixed the same day)*

The servo correctly holds through warmup — but the **probe does not**. Its window
is `window × cadence` = 500 train steps deep at the defaults, so the first
post-ramp tick took its median from readings recorded at the envelope-suppressed
LR. Since `α* ∝ 1/lr`, those readings are biased high by exactly the warmup
factor, and the servo climbed on them.

Measured: `b_descend`, seeded at 4.0e-4 and required to *descend*, first read
`alpha_median` 6.6 (warmup-era) and climbed to **5.36e-4 — a 34% overshoot in the
wrong direction** — before the window refilled and it turned around and began
descending as designed.

Fixed with `StepProbe.flush_window()`, called at two places where the LR regime
changes under the probe: the first tick after the ramp completes (which then
holds `cold` until the window refills) and `rearm_warmup` (a stage transition
changes the loss surface *and* re-ramps). Covered by `test_lr_controller.py`.

> The `b_descend` and `c_low` arms were launched **before** this fix, so
> `b_descend`'s early excursion to 5.36e-4 is that artefact and not servo
> behaviour. It delays the descent; it does not invalidate it.

**The general shape is worth more than the instance.** Three of this module's
controllers now have a version of the same bug — a controller that reports
holding while some part of its loop keeps accumulating. The rule that falls out:
**when a controller holds, ask what its sensor is doing meanwhile.**

**F10 — ✅ with a CALIBRATED target the servo converges from BOTH sides onto the
hand-tuned optimum, and the approach direction costs more than the destination.**
*(measured 2026-08-08, `lr_aug08` pair CAL — named CAL, not D, because
`local_aug08` already owns "pair D")*

`target: 1.87`, read off `a_fixed`'s own second half, two-sided clip, seeded on
opposite sides:

| arm | seed | landed | α\* held | `bwd/tb_err` | `fwd/tb_err` |
|---|---|---|---|---|---|
| `d_cal_below` | 1.0e-5 (11× below) | **1.141e-4** | 1.926 | 15.71 (+0.67) | 20.09 (+2.20) |
| `d_cal_above` | 4.0e-4 (3.2× above) | **1.405e-4** | 1.903 | **14.78 (−0.26)** | **17.97 (+0.08)** |

**They agree to 1.23×, the hand-tuned 1.25e-4 sits between them, both hold α\*
within 3% of target, and neither ran away.** So the loop has a real, reproducible,
two-sided fixed point and a calibrated setpoint puts it on the optimum — the
servo doing the job it was built for.

**The direction of approach is the finding.** `d_cal_above` is statistically
indistinguishable from the hand-tuned arm. `d_cal_below` lands at the *same LR*
and is 2.2 nats worse on `fwd/tb_err`, because it spent most of the run climbing
through rates that were too low — **under-training is not recovered by arriving
eventually.** Descending onto the answer is nearly free; climbing onto it is not.

**Two mechanics this exposed:**

- **`span` and `target` are coupled.** A censored window reports `span`, so the
  servo's multiplier there is `span / target` — 2.0/1.87 = **1.07 per tick**, not
  the 1.25 clip. `d_cal_below` crawled for seven bins before its fits turned
  `ok`. Calibrating `target` upward shrinks growth authority in exactly the
  regime that needs it most; keep `span` ≈ 2× the intended target.
- **F7's fix is confirmed.** `d_cal_above` descended monotonically 4e-4 → 1.16e-4
  with **no** upward excursion, where the pre-fix `b_descend` climbed 34% the
  wrong way first.

**Consequence: three postures, all now measured** — see `decisions.md` D31. The
one that both finds the LR and forecloses F8's runaway is **seed above the guess,
calibrated target, one-sided clip**: it descends onto the optimum and can never
climb. Its cost is a hot transient (`d_cal_above` went `fwd/tb_err` 22.5 → 25.9
before recovering).

**F9 — the divergence response works end-to-end, and the 1e9 bar is slack.**
*(verified 2026-08-08, `lr_aug08` v0/v0b — the first time this path has ever
executed)*

`fire_loss_spike` was rewired in v7 (the `terminal` parameter dropped,
`on_explosion` → `on_divergence`, the no-rewind-target branch rewritten) and no
run had taken it. Forced with `lr_fused: 1e-1`, 800× the reference:

```
lr_ctrl DIVERGENCE: fused_loss = 9.913e+08 (bar 1e+06) -- reload + peak cut
Divergence response: rewind #1 + peak cut
lr_ctrl: peak_scale -> 0.5 (ceiling recorded)
   ... rewind #2 -> 0.25 ... rewind #3 -> 0.125 ...
UNRECOVERABLE at step 2690: 4 rewinds (cap 3) and the run keeps re-detonating
```

Every element fires: the bar, the paired reload+cut, the compounding across
repeats, the ceiling record, and `max_reloads` aborting cleanly via
`FrozenTrainingState` rather than hanging. **Including the documented failure
mode** — this arm's LRs are explicit floats, so they are *not* servo-managed, the
peak cut cannot lower them, and `max_reloads` is the only thing that stops the
run. That was written down in `verify.py` before the arm ran and is now observed.

**But the shipped bar is slack.** At 1e9 the same detonation reached
`grad_norm_pre_clip` **2.4e8** and `fwd/tb_err` **4.7e7** — dead by any reading —
with `lr_ctrl/divergences` still **0**. v0b had to lower the bar to 1e6 to trip
it. The tier D4 kept is live and correct; it simply sits eight orders above where
a run stops being recoverable. See `decisions.md` D32.

**F1–F4 — RETIRED with the middle layer.** They described the cut tier
(`recovery_target_frac ≤ cut_ratio` inertness), the decay leg's engagement, the
cut-factor floor's wrong base LR, and `_pre_trigger_cold`'s cross-call coupling.
None of that code exists. They are kept in git history and in `decisions.md`'s
closed section because they are the *evidence* the deletion rests on:

| Old finding | What it showed | Where it went |
|---|---|---|
| F1 | recovery was arithmetically inert in three shipped batteries | the recovery ramp is deleted |
| F2 | decay was live only in phase 2 and barely | the decay leg is deleted (D7/N2) |
| F3 (=`E4`) | the cut floor used `lr_policy` in a fused stage | the cut factor is deleted; `servo.bounds` is explicit and group-independent |
| F4 | `_pre_trigger_cold` was a fragile cross-call coupling | episode grouping is deleted |

## 6. Failure signatures

| Symptom | First metric | Cause |
|---|---|---|
| LR never moves off the seed | `lr_ctrl/servo_hold` | not "satisfied" — read the code: `no_probe` (no `step_probe` block), `fit_invalid`, `few_readings` |
| LR climbs to the bound and parks | `lr_ctrl/peak_scale` at `bounds[1]` | α\* is persistently > target; either the bound is too low or the setpoint is wrong (F5) |
| LR sawtooths | `lr_ctrl/peak_scale` + `lr_ctrl/divergences` | the ceiling's half-life is shorter than the time to re-detonate |
| Run alive but not improving | `lr_ctrl/peak_scale` ≤ 0.15 | LR-starved after repeated divergences |
| Repeated rewinds, run never dies | `lr_ctrl/divergences` climbing | `max_reloads` is the backstop; if the LRs are all explicit floats, the cut half of the response is a no-op by construction |
| `grad_norm` bitwise constant | `grad_norm_pre_clip` | frozen detonation — `_frozen_training_state`, not an LR problem |

**Check `lr_ctrl/peak_scale` first on any arm.** It is the cheapest single test
for "what LR did this actually run at", and under `auto` the configured number is
no longer the answer at all — the old [[live-vs-set-lr]] reading protocol now
applies to every servo-managed run rather than only to ones that took a cut.

## 7. Memory reconciliation

- `adaptive-lr-controller` — **wholesale stale**, and now doubly so: it describes
  the v5 probe/cruise controller, which v6 replaced and v7 replaced again.
- `lr-controller-semantics` — **retire.** Episode grouping, hot gating and the
  recovery ramp it documents are all deleted.
- `lr-tripwire-deadlock` — **retire as a live hazard, keep as history.** Both
  deadlock modes lived in the middle layer.
- `fused-mode-dead-lr-knobs` — **still correct and now sharper**: in a fused
  stage only `lr_fused` and `lr_flow` are read, so writing `auto` on
  `lr_policy`/`lr_back`/`lr_replay` there hands the servo keys nothing consumes.
- `live-vs-set-lr` — **keep, and promote.** See §6.

## 8. Simplification candidates

**S1 — `CHANNELS` and the per-`step_type` loss bar.** With one coarse bar the
per-branch loss check buys almost nothing over the grad-norm check; a diverged
policy shows on both. Collapsing to grad-norm alone would remove the last place
the controller needs to know branch names.

**S2 — `lr_servo_managed` is a resolver side-channel.** `resolve_derived_config`
records which keys were written `auto`, because after resolution `auto` and a
float are indistinguishable. It works, but the fact that *how a value was
written* is load-bearing deserves a first-class representation rather than a
tuple stapled to `args`.

## 9. Verdict: what this module is buying

Pass 1's verdict was that the design was **locally well-engineered and globally
unjustified** — every mechanism had a real incident behind it, and almost every
incident had been generated by the module itself. Six of the seven mechanisms it
listed were containment for the cut tier. That verdict is what got acted on.

Against the three stated goals, now:

- **Safe** — unchanged, and that is deliberate. Divergence bar + rewind +
  `max_reloads` + the frozen detector are exactly what they were; they answer to
  real external events and they work. The middle layer's removal took away no
  containment, because there is direct evidence it never provided any (1219ddv9
  degraded monotonically *through* a 100× cut; s706frkh's runaway ran at policy
  LR 1e-6).
- **Dynamic** — this is what changed. v6's only effect on the operating LR was
  downward and finding the envelope top was entirely hand tuning. The peak is now
  measured per run, which matters because there is no stable *here* to tune
  against: the problem shifts with the energy function, the space group, the
  conditions, `T` and `W`, so **every run is a transfer** (D3 revisited).
- **Not wasteful** — every documented waste event (replay_july26 at 1e-6,
  tw_july31 arms frozen at `cut_factor` 0.50, the aug02 rewind loop) was caused
  by machinery that is now gone.

**The honest residue** is F5. The servo's *sensor* is validated — the scaling law
holds to 1% — and its *setpoint* is not. A loop with a correct sensor and a wrong
setpoint tracks confidently to the wrong place, which is a failure mode the old
hand-tuned peak did not have. That is the thing to watch on the first long run,
and `target` / `clip` are one config edit away from either of the two safer
postures in F5's table.

---

*Warrant classes: **derived** (follows from the math) · **measured** (A/B'd,
run cited) · **inherited** (came from elsewhere, never re-examined here) ·
**arbitrary** (someone picked a number) · **contested** (conflicting evidence).*
