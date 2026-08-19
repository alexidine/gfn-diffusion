# Module: modulators — batch size + buffer servo

> **Status: MODULE SNAPSHOT.** The verification dates below are historical.
> Use this document for explanation and navigation; verify material claims
> against current code, canonical config, and focused tests. See
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

Pass 1 (audit + rationalize). Verified against the working tree, 2026-08-03;
**buffer-servo sections revised 2026-08-08** — the servo acquired a second,
*derived* sensor on 2026-08-07 (`to_do_rebuild.md` §B7d) and was configured for
the first time. **D3 revised 2026-08-10** — `train.py`'s `floor`/`stalled`
replay-eviction causes were retired outright (`module_buffers.md` B10),
which retires `toxic_min_draws` as a config key everywhere and makes the
servo's `toxic_min_draws` boost path permanently inert as a side effect (D3
below). Unconditional route. Covers
[`select_batch_size`](train.py:466), the OOM recovery path, and
[`_buffer_servo_tick`](protocol.py:1676). The LR controller — the third
modulator — has its own document.

---

## 1. What they are

Three controllers that change *how* training runs without changing *what* it
optimises. They share a shape: a sensor, a bounded actuator, and a reason the
actuator is the right one for that sensor.

*(The LR row changed wholesale on 2026-08-08 -- v7 replaced the absolute-bar
sensor and the cut-factor actuator with the alpha* servo. See
`module_lr_controller.md`, which is the document for it; the row here is a
cross-reference, not a summary.)*

| | Sensor | Actuator | Bound |
|---|---|---|---|
| **Batch size** | per-rung occupancy calibration (raw NVML samples), off by default | hold `batch_size`; grow only to the smallest rung clearing `batch_util_target` | `max_batch_size`, OOM ceiling, `max_step_seconds` |
| **Buffer servo** | a **configurable** `numerator ÷ denominator` — either the scatter ratio or, since 2026-08-07, `replay/ema_loss_mean ÷ replay/birth_loss_mean` | one multiplicative boost B on replay freshness | `max_boost`, deadband |
| **LR** | `alpha*` (two-point step probe), plus one coarse divergence bar | multiplicative `peak_scale` on servo-managed groups | `servo.bounds`, a relaxing ceiling, `min_lr` |

## 2. Batch size

**REPLACED at state 8 (2026-08-19)** — the throughput-knee walk this section used
to describe is gone; `train.select_batch_size` is the controller and
`docs/design/phase6_batch_sizer.md` is the argument. In brief:

**Objective, decided (user, 2026-08-16):** optimizer steps/sec at a threshold
effective batch `A = fused_grad_accum_min_samples`. For any `B <= A`, updates/sec
rises with `B`, so the throughput optimum is the **constant** `B = A` — the
configured `batch_size`. There is no knee to find and no walk; the old goal
statement (max samples/sec) let the effective batch float, which is the other
objective, with the opposite argmax on a saturating cost curve.

**Growth exists only to buy occupancy** — the number the scheduler cancels on
(cancelled at ≤40%, survived at ≥49.4%, out-of-process; `phase6_handoff.md` §2,
§4.3). With `batch_util_target` set (off by default), the sizer calibrates once
per stage: it climbs `batch_growth_factor` rungs, dwelling `batch_growth_interval`
steps at each to read a step-time median and a few **raw** occupancy samples, and
holds the smallest rung clearing the target. Three structural rules: occupancy
evidence may only *veto* rungs under that fixed selection rule (S1); a kept growth
must survive a full policy window of lived occupancy or it stands down (S2); and
UNKNOWN — no target, no sensor, no reading — never removes and never grows (S3).
When no rung clears, the verdict is **INFEASIBLE**, said loudly with the binding
bound named: batch is not the lever then; host stalls, work per kernel launch, or
the energy-pinned memory ceiling are (`phase6_handoff.md` §4.4).

The division of labour stands: gradient *quality* is owned by
`fused_grad_accum_min_samples` (accumulate when the batch is under target);
memory by the OOM ceiling (a domain bound that expires, with a restore-to-base
rule when it clears); runaway steps by `max_step_seconds`. None of them selects a
size. Rungs are few and geometric because `torch.compile` treats every distinct
size as a recompile plus its own CUDA graph — that cross-module constraint
survives from the old design.

## 3. Buffer servo

**The sensor is a config choice, and there are now two candidates.** The servo
itself only ever forms `numerator ÷ denominator` and tightens when the ratio is
**low**, so swapping sensors is a config edit, not a code change. That
generality — added before there was a second sensor to put in it — is what made
§B7d cost no `protocol.py` change at all.

**Sensor A (the original): `replay/scatter_err ÷ fwd/scatter_err`.** Replay draws
are a `|resid|`-prioritised resample of stored forward rollouts, so a replay batch
is *by construction* the hard tail of the forward distribution and its residual
spread should **exceed** fresh forward's — healthy is ~2×. The ratio crossing
below 1 says the policy fits reused stored trajectories better than the fresh
draws they were selected from. That is memorisation of the buffer's contents and
nothing else. **Its stated justification is stale and its thresholds are
uncalibrated — see D1.**

**Sensor B (2026-08-07, preferred): `replay/ema_loss_mean ÷
replay/birth_loss_mean`** — the memorisation sensor, `module_buffers.md` B8. Each
resident row's current residual against the one it was admitted with:

```yaml
buffer_servo:
  numerator:   replay/ema_loss_mean
  denominator: replay/birth_loss_mean
  bar:     0.368     # lambda*tau = 1 -- DERIVED, not calibrated
  release: 0.60      # lambda*tau ~ 0.5
  scale:   0.15
  gain:    0.05
  relax:   0.5
  max_step: 0.05
  max_boost: 8.0
```

**Why B is the better sensor, in one line:** its bar is *derived*. `ratio ≈
exp(−λτ)` puts the `λτ = 1` boundary at exactly `1/e = 0.368`, so nothing in the
threshold was measured and it transfers across problem, `T` and buffer size —
precisely what D1 says sensor A's `bar`/`release`/`scale` cannot do. It is also
internal to the buffer, so it does not depend on what the `fwd` branch is
currently being used for, which is what went stale in A.

B carries one coupling worth stating: **it is only unbiased under the uniform
hazard.** `birth_loss` exists only for residents, so it is the intake distribution
*of survivors*, and that equals the intake distribution only if survival is
independent of the residual (`module_buffers.md` §3). As of 2026-08-10 this
holds unconditionally — the residual-conditioned purge (`floor`/`stalled`)
that could invalidate it is retired outright, not merely disabled under
`prioritise` (`module_buffers.md` B10), so there is currently no code path
that would reintroduce the bias. Recorded here as a coupling to watch, should
a residual-conditioned purge ever come back.

**A note on what the actuator can and cannot reach.** §B7a proves intake *rate* is
the only lever on `λτ` — buffer size cancels out. So the boost below is not one
option among several; it is the only actuator that acts on this sensor at all.

**Actuator.** One multiplicative boost B: `churn_rate × B` and
`mean_residence_steps ÷ B`. Since occupancy = `churn_rate × mean_residence_steps`
(Little's law) and `draws_per_row = batch_size / churn_rate`, this leaves
**occupancy exactly invariant** and moves only reuse (1/B) and policy lag (1/B).
The code (`protocol.py` `_apply_buffer_boost`) also carries a `toxic_min_draws
÷ B` term, on the same reasoning — it was defined relative to expected draws
per row and would otherwise silently change meaning as B moves — but that term
is now dead: `toxic_min_draws` is a retired config key everywhere
(`module_buffers.md` B10), so the snapshot it would scale is always zero. The
boost the servo actually delivers today is two knobs, not three (D3).

One knob with one invariant is the best single piece of control design in the
codebase. `churn_rate`, `mean_residence_steps`, and `max_size` are three handles
on one steady state, and moving them independently is how a buffer ends up in a
corner nobody meant.

**Why a second controller and not a balance rule.** Loss weights cannot fix
overfitting: down-weighting replay trains less on a memorised buffer, but does
not make the buffer less memorised — and it gives up the residual-tail
correction replay exists to provide. Freshness acts on the cause. The two loops
are near-orthogonal by construction: this one holds occupancy fixed and changes
no frac; the balance controller changes fracs and touches no buffer knob.

**Deadband and release.** Tighten below `bar`, release above `release`, hold
between, with `relax < 1` making release the slower direction. The release term
exists specifically so the servo is not a ratchet whose fixed point is
`max_boost` — the same one-way-anneal failure recorded elsewhere in this
codebase. That is a lesson correctly applied.

## 4. Findings

**D6 — `samples/sec` and "backprop per second" diverge past gradient saturation.** *(open, worth knowing)*

The proxy assumes every sample is equally valuable. Once the gradient estimate
has saturated statistically, extra samples per step buy less *progress* while
still counting fully as *throughput* — so the knee detector finds where
**throughput** saturates, not where **statistical value** does. If statistical
saturation arrives first, the walk keeps growing the batch past the point of
useful return and the knee never objects, because samples/sec is still climbing.

The floor (`fused_grad_accum_min_samples`) guards the other end — too few
samples — but there is no corresponding ceiling on the statistical side. Cheap
check: is progress-per-sample flat across rungs, or does it fall as the batch
grows? If it falls before the throughput knee, the knee is the wrong stopping
rule.

Not an argument against the proxy — just the gap in it.

**D1 — the buffer servo's sensor rationale is stale for the current route.**
*(confirmed by user 2026-08-05: the docstring's claim rested on an assumption that
held only for a specific set of experiments. **Superseded rather than fixed,
2026-08-07** — the answer was a new sensor with a derived bar, §3 sensor B, not a
recalibration of this one. The finding stands for any config still pointing the
servo at the scatter ratio, and the docstring is still wrong.)*

The docstring asserts "Both branches now carry policy gradient (fwd runs
freeze_policy 0 in this route), so the two sides differ only in their sampler,
which is what makes the ratio a clean generalization gap." The `naive` stage now
sets `fwd: { freeze_policy: 1.0 }` — fwd trains **Z only**.

The metric is arguably *still* valid, and possibly stricter: with the policy no
longer trained on fwd, forward draws are genuinely held out from policy
training, which is a cleaner train/heldout split than the one the docstring
describes. But two things follow: the stated justification is wrong, and the
calibrated healthy value (~2×) plus the `bar` / `release` thresholds were
measured under the other branch-role assignment and cannot be assumed to
transfer. Re-read before enabling.

**D2 — ✅ the servo is now configured on `mk_dev`.** *(revised twice on
2026-08-08 — first to record that it had arms, then to record that it shipped)*

`mk_dev`'s `equilibration` stage carries a `buffer_servo` block on sensor B
(`replay/ema_loss_mean ÷ replay/birth_loss_mean`, bar 0.368), alongside
`prioritise.enabled: true`. **Those two go together and the coupling is not
optional:** sensor B is only unbiased under a residual-independent hazard, and
`prioritise` is exactly what switches the residual-dependent purge causes off
(`module_buffers.md` §3). A config with the servo and without `prioritise` is
reading a survivorship-biased sensor.

First 150 steps under that configuration: `lambda_tau` fell 0.55 → 0.02 (i.e.
`resid_vs_intake` → ~0.98, a near-pure delay line) and `bs_log_boost` stayed 0.
That is the servo correctly **idle** — uniform intake means residents enter
representative and are not being corrected at their own trajectories, which is
the condition it exists to detect the absence of.

rb0808 arms 19/20 were the planned cluster read of the same question; that
battery is **cancelled**, so the natural-onset question moves to whatever
replaces it.

The pathology question in the original entry is **answered, for an induced case**:
a deliberate intake starve (`churn_rate: 3`, `mean_residence_steps: 400`) drove
`replay/lambda_tau` to a peak of **1.43** with `resid_vs_intake` down to 0.24,
well past the derived 0.368 bar. So the sensor detects induced memorisation. What
is *not* answered is whether it arises **naturally** on this route — that is what
arm 19 is for.

**D7 — the servo read `None` on every tick and was silent in exactly the way a
satisfied servo is. ✅ FIXED 2026-08-07.** *(the sharpest instance of a pattern
this codebase now has three of)*

`StageProtocol._resolve` reads every servo sensor through
`metric_tracker.get(direction, metric)`. `absorption_stats()` was being published
**only into the wandb metrics dict at report time**, so the tracker never held
`replay/ema_loss_mean`. The servo resolved `None`, took its cold-start early
return on every tick, and **never emitted `protocol/bs_log_boost` or `bs_ratio`
at all** — indistinguishable, in the logs, from a servo correctly deciding to
hold. The induced overfit above ran to λτ 1.43 with the actuator inert.

Fixed by publishing the sensor into the tracker from `replay_train_step`
([train.py:3116](train.py:3116)). Deadband verified against the observed
trajectory afterwards: at ratio 0.24 the drive is **+0.85**, stepping `log_boost`
+0.043/tick toward more churn.

> **The general lesson, third instance.** A control loop whose *sensor* is
> unreadable is silent in exactly the same way as a control loop that is
> *satisfied*. This is `module_protocol.md` S1's drive-liveness argument again, and
> the servo would have announced itself immediately if it reported ticks-since-last
> -nonzero-drive. **Two rules fall out, and both are cheap:** a metric a controller
> reads must go through `metric_tracker.update`, not just the report dict
> (`module_metrics.md` T7); and every servo should emit its **actuator**, not only
> its sensor. `protocol/bs_log_boost` is now emitted for exactly that reason.

**D8 — the servo is validated as WIRED and unproven as EFFECTIVE.**
*(open; this is the honest status)*

Post-fix, `protocol/bs_ratio` appears with 200 readings spanning 0.27–0.56, dipping
below the bar as intended. But `λτ` ended at 0.719 against the no-servo arm's
0.652 — **no evidence of successful control**, because the loop was far too weak
for the perturbation: `gain 0.05 × ~8 ticks` over 2000 steps is ~1.4× churn
recovery against a **27× intake starve** (`churn_rate` 80 → 3). Reaching
`max_boost: 8` at 0.043/tick needs ~48 ticks ≈ 12k steps.

**A null result there means nothing, and the cluster arm was sized to fix that**:
rb0808 arm 20 gets 35k steps / eval 500 = 70 ticks × `max_step` 0.03 = `log_boost`
2.1 = **8.2× churn**, inside `max_boost: 12`. The loop *can* recover within budget,
so a null result from arm 20 **is** informative. Read arm 20 against arm 19, and
check `bs_log_boost` actually moved before concluding anything about λτ.

**D3 — the servo writes its output back over its own input.** *(user 2026-08-05:
"perhaps too complicated"; user 2026-08-08: "haven't a clue what D3 is even
talking about" — **restated below in plain terms, because that reaction is the
finding**)*

**Plainly: the servo has one number to remember (how much extra churn to ask
for), and it is stored in three places, one of which is the config it is
supposed to be modifying.**

The three knobs the servo moves are `churn_rate`, `mean_residence_steps` and
`toxic_min_draws`. **Revised 2026-08-10: the third is now dead.**
`buffers.replay_buffer.toxic_min_draws` (and `toxic_delta_threshold`) are
retired config keys — setting either raises `ValueError` at load
(`module_buffers.md` B10, `train.py:4490`), because the `stalled` eviction
cause they fed is deleted outright, not merely gated. `_apply_buffer_boost`
below still contains the `toxic_min_draws` branch, but it can never fire: the
snapshot `getattr(rb, 'toxic_min_draws', 0) or 0` is always `0` now, since no
valid config can set the attribute, so `base['toxic_min_draws'] > 0` is
always false. Harmless — it just means the servo has moved two live knobs,
not three, since this change, and the code below (and the read-site table
past it) describes one more store than currently does anything.
`_apply_buffer_boost` **overwrites them on `args`**:

```python
rb.churn_rate = max(1, int(round(base['churn_rate'] * boost)))     # protocol.py:1757
```

Since `args` is where the *configured* value lived, after the first boosted tick
the config no longer holds what the yaml said. So the servo needs a private copy
of the original — `_rb_base` — which it snapshots on its **first call**
([line 787](protocol.py:787), populated at [line 1752](protocol.py:1752)). Three
stores for one number: `bs_log_boost` in `stage_ctrl` (checkpointed), `_rb_base` on
the instance (not checkpointed), the live product on `args`.

It is correct today for a reason that has nothing to do with the servo: `args` is
re-parsed from yaml at every process start, so the first snapshot always catches a
pristine value. **Nothing enforces that.** If a `StageProtocol` were ever
constructed against an `args` that had already been boosted — a second protocol in
one process, a resume path that restored `args`, a test — `_rb_base` would snapshot
a *boosted* value as the base and the boost would compound on every launch,
silently and multiplicatively.

⚠ **The fix recorded here was wrong, and that is why it never got done.**
*(found 2026-08-08 while verifying the entry)* It said: *"read the base off config
each tick instead of snapshotting it."* **That cannot work** — the config *is* the
thing being overwritten, so after tick 1 there is no pristine value left to read.
There is no separate immutable copy anywhere.

**The fix that does work: stop mutating `args` at all.** `bs_log_boost` is already
in `stage_ctrl` and already checkpointed, so the boost is already durable — it just
needs to be applied at **read** time instead of write time. The read sites were
three, all inside `manage_replay_buffer`; as of 2026-08-10 (`toxic_min_draws`
retired) they are down to two, and the line numbers below have moved with the
`floor`/`stalled` deletion (`module_buffers.md` B10):

| Knob | Read at | Becomes |
|---|---|---|
| ~~`toxic_min_draws`~~ | — retired key, no longer read anywhere | — moot |
| `mean_residence_steps` | [train.py:4518](train.py:4518) | `base / B` |
| `churn_rate` | [train.py:4575](train.py:4575) | `base × B` |

One accessor, two live call sites, and **both** `_rb_base` and `_apply_buffer_boost`
delete outright. That removes two of the three stores rather than one, makes the
compounding failure unrepresentable rather than merely unlikely, and drops the
`_apply_buffer_boost(1.0)` reset-on-stage-exit special case (below) because there
is no longer any state to reset. The `toxic_min_draws` branch inside
`_apply_buffer_boost` can simply be deleted along with the rest rather than
ported — see the note above D3's code block.

⏳ **Outstanding as of 2026-08-08**, and deliberately not done while the rb0808
battery is in flight — arm 20 is the servo arm, and this changes the manage path it
runs through. Also still true: `_buffer_servo_tick` resets a previous stage's boost
via `_apply_buffer_boost(1.0)` **only if `_rb_base` is non-`None`**
([line 1730](protocol.py:1730)), so the reset is conditional on whether the cache
was ever populated — exactly the coupling this entry was opened about.

**D4 — batch OOM and the throughput knee are two controllers on one variable.** *(confirmed, benign)*

An OOM cuts by `oom_batch_shrink_factor` and sets `batch_size_cooldown_until`;
the knee pins. They couple through `batch_size_ever_oomed`, which switches growth
to the slow interval — which in turn lengthens the dwell each rung gets before
its throughput is measured. So an early OOM makes every subsequent knee
measurement more reliable, and a run that never OOMs measures on shorter dwells.
Not a defect, but it means knee decisions are not directly comparable between an
OOMed and a non-OOMed arm.

**D5 — the knee needs 10 step-time samples and the window clears at every
transition.** So the first `batch_growth_interval` after a stage change cannot
make a knee decision, and the batch grows blind for that window. Bounded and
probably fine; worth knowing when reading a step-time trace across a transition.

## 5. Warrants

| Choice | Warrant | Evidence |
|---|---|---|
| Marginal throughput, not utilisation | **derived** — utilisation pegs at 100% below the knee and cannot discriminate on either side; the choice is principled, not an instrumentation workaround (`torch.cuda.utilization()` is available) | docstring |
| Knee pin per stage, cleared at transition | **derived** — a baseline from the outgoing stage's step cost poisons the incoming comparison | `advance` |
| Periodic knee recheck | **derived** — the knee moves as fused composition drifts | docstring |
| Coarse growth (few distinct sizes) | **derived** — compile recompiles per size | docstring |
| `batch_growth_min_gain: 0.15` | **arbitrary** | — |
| Grow-until-OOM as the operating mode | **accepted** — documented and deliberate | — |
| Boost holds occupancy invariant | **derived** — Little's law | docstring |
| `toxic_min_draws` rides 1/B | **derived, now moot** — preserved its definition while the key was live; the key is retired 2026-08-10 (`module_buffers.md` B10) and the servo's corresponding boost term is permanently inert | docstring; D3 |
| Release term (non-ratchet) | **measured** — one-way anneals fixed-point at the bound | prior controller-ratchet failure |
| Servo `bar` / `release` / `scale` **on sensor A** | **measured but stale** | D1 |
| `bar: 0.368` **on sensor B** | **derived** — `ratio ≈ exp(−λτ)` puts `λτ = 1` there, so it transfers across problem, T and buffer size | `module_buffers.md` B8 |
| Intake rate as the actuator (not buffer size) | **derived** — §B7a: buffer size cancels out of `λτ` | `to_do_rebuild.md` §B7a |
| `scale` saturating the drive (constant-rate ramp) | **measured** — the raw deficit is bounded by `bar` and sat at ~0.03 live, needing ~23k steps to traverse the boost range, i.e. inert exactly where the servo lives | docstring, `_parse_buffer_servo` |
| Servo emits its **actuator** (`bs_log_boost`), not just its sensor | **measured** — without it, a servo with no authority looked identical to one correctly holding | D7 |
| Gradient stability from accumulation, not batch inflation | **derived** — separates two jobs of one knob | [train.py:311](train.py:311) |

## 6. Failure signatures

| Symptom | First check | Cause |
|---|---|---|
| Step time grows, samples/s flat | `batch_size` vs throughput | past the knee — knee opt off or `min_gain` too low |
| Batch oscillating rung-to-rung | `batch_size_pinned_at` | recheck period too short relative to composition drift |
| Batch grows blind after a transition | `_recent_step_times` length | D5 |
| Replay fits better than fwd | `replay/scatter_err ÷ fwd/scatter_err` < 1 | buffer memorisation — the servo's whole reason to exist |
| Buffer being fitted at its own trajectories | `replay/lambda_tau` > 1, `resid_vs_intake` < 0.368 | the same pathology on the derived sensor (B) |
| Servo pinned at `max_boost` | `bs_log_boost` flat at ceiling | release term not firing — check `relax` and `release` |
| **`bs_log_boost` / `bs_ratio` absent from the run entirely** | is the sensor in `metric_tracker`, or only in the report dict? | D7 — the servo is resolving `None` and cold-starting on every tick. **This looks exactly like a servo correctly holding**, so absence of the series is the only tell |
| Servo reading, but λτ unmoved vs the control | `bs_log_boost` range × `gain` × ticks vs the size of the perturbation | D8 — no authority. Compute the reachable boost before believing a null |
| Buffer in a corner after tuning | occupancy vs `churn_rate × τ` | three knobs moved independently instead of via B |

## 7. Simplification candidates

**S1 — ✅ UNBLOCKED and largely resolved 2026-08-07.** The servo was "elegant in
design but has to be reconsidered along with the whole replay construction"
(user), so this was held pending the redesign. The redesign landed, and the
outcome was better than a re-derivation: the actuator survived **unchanged** (it
was already the right one — §B7a), and only the *sensor* was replaced, by config.
D1's stale rationale is routed around rather than repaired. What remains open is
D8 (authority) and D3 (`_rb_base`), neither of which is about what replay *is*.

**S2 — ~~derive `_rb_base` from config each tick~~; apply the boost at READ time
and delete the cache** (D3). *(restated 2026-08-08 — the original form was
unworkable: the config is the thing being overwritten, so there is no pristine
value left to re-read. See D3 for the read sites — two of them, live, as of
the 2026-08-10 `toxic_min_draws` retirement.)* Still chosen over
checkpointing `_rb_base`: it removes two stores instead of one, and the compounding
failure becomes unrepresentable rather than unlikely. **Sequenced after rb0808
lands** — it changes the manage path arm 20 is running.

**S3 — ✅ approved: log the knee decision as a metric**, not just a print.
Comparing arms on whether and where they pinned currently requires reading
stdout.

## 8. Open questions

1. **Partly answered, 2026-08-07.** Does replay memorisation occur on this route?
   *Induced*, yes and clearly — λτ peaked at 1.43 under a deliberate 27× intake
   starve, so the sensor discriminates (D2, `module_buffers.md` B8). *Naturally*,
   still unknown: rb0808 arm 19 `rep_starve` vs arm 20 `rep_starve_servo` is the
   test. Note the historical scan already suggests the answer is "usually no" —
   λτ < 0.5 on most of 33 archived arms, including every `ctrl_aug03` servo arm at
   λτ ≈ 0.02.
2. **Confirmed arbitrary, no evidence** (user). `batch_growth_min_gain: 0.15` —
   a jump buying 14% is reverted, 16% is kept, off a median of 20 step times.
   Worth knowing the estimator's spread before trusting the threshold; more
   worth knowing whether the *rule* (marginal throughput gain) is right at all,
   given D6.
3. **Answered.** OOM is not a cost — "I don't actually care if we OOM, just that
   the GPU is being used." So the grow/OOM cycle and the knee detector are
   independent and both fine; the safety margin can be thin.

   Worth noting the two goals are not the same, and the stated one is stricter:
   nvidia-smi utilisation saturates *below* the knee, so "the GPU is being used"
   is satisfied well before "samples/sec is maximised." The knee is the binding
   target; OOM tolerance just means it can be approached without care.
4. *(new)* D6 — does progress-per-sample stay flat across rungs, or fall before
   the throughput knee?

---

*Warrant classes: **derived** · **measured** · **inherited** · **arbitrary** · **contested**.*
