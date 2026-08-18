# Synthesis: design philosophy, working parts, key interactions

> **Status: SNAPSHOT, NOT CURRENT STATE.** This document explains the system as
> understood at the dated working states below. Verify every material behavioral
> claim against current code, `configs/mk_dev.yaml`, and focused tests. Current
> authority and routing are defined by root `AGENTS.md`, `docs/README.md`, and
> [`EPISTEMIC_PROTOCOL.md`](EPISTEMIC_PROTOCOL.md).

Top-level snapshot over the module documents. Unconditional route, working tree
@ 2026-08-03; **§1 annotated 2026-08-08 — the thesis is under active challenge.**
The module docs are the detail for the same historical snapshot.

---

## 1. The thesis, stated precisely

> 🔴 **Read §1a before treating this section as settled.** The measurement on
> 2026-08-07 found that `freeze_policy: 1.0` on `fwd` — the knob that *implements*
> the thesis below — is the single variable separating improving arms from
> degrading ones. The section is left intact because it is still the design's
> stated intent and the argument for it is sound; §1a records what contradicts it.

The working description of the design goal has been:

> converge GFlowNet models on-policy, with large pre-built buffers providing
> diversity

The code does something more specific, and the difference matters enough that
the paper should lead with the precise version:

> **The policy is trained entirely off-policy. The on-policy branch trains only
> the partition function — because Z is the ruler every off-policy residual is
> measured against.**

In the terminal (`naive`) stage:

| Parameter group | Trained by | Sampler | Prevents |
|---|---|---|---|
| **Z** (flow head) | `fwd` | on-policy rollouts | Z going stale, which corrupts every off-policy residual by a common offset |
| **policy** | `bwd` | churned prior buffer | mode collapse / loss of diversity |
| **policy** | `replay` | \|resid\|-prioritised resample of `fwd` rollouts | over-weighting of the policy's own errors |

Each parameter group has exactly one trainer, and each trainer exactly one
sampler. That is the whole architecture.

**Why the policy comes off the on-policy branch.** On-policy TB is
*mode-seeking*: it only ever sees the policy's own support, so its gradient can
sharpen what the policy already does but cannot discover what it has missed.
Taking the policy term off `fwd` removes that pressure entirely. Prioritised
replay then puts on-policy information back in — reweighted toward the residual
tails — so the system gets on-policy *data* without on-policy *mode-seeking*.

**Why Z stays on-policy.** Z is the normaliser of the on-policy distribution.
Every off-policy TB residual is `(log_pf − log_pb) − (log_r − log_Z)`; if Z lags,
every off-policy branch's gradient carries the same offset, and the policy's only
way to comply with a uniformly-negative residual is to move mass off-support —
inflating everywhere at once. That mechanism is documented in `z_level_loss`, and
it is why `relative_under` re-centres on the batch's own empirical normaliser
instead of on Z.

So the buffers are not a *supplement* to on-policy training. They **are** the
policy's training signal, and the on-policy branch's job is to keep the ruler
current.

## 1a. 🔴 What contradicts §1 — `freeze_policy` may be the cost

*(added 2026-08-08. Full data in `to_do_rebuild.md` §0c; the decision this feeds is
`decisions.md` D30, which rb0808 arms 4/5/6/14 are running now.)*

`freeze_policy: 1.0` on `fwd` is the implementation of "the policy is trained
entirely off-policy." An arm isolating exactly that variable — replay left **on**
and prioritised, same batch, same `beta`, same κ — flips the sign on all three
headline metrics. Common window, steps 2650–3450, mipcas ELJ, T=10:

| arm | fwd | replay | `fwd/tb_err` | `fwd/r2` | `over_coverage` |
|---|---|---|---|---|---|
| `r5` | policy grads | **off** | 21.54 → **20.97** ✅ | −3.26 → −2.54 | 20.85 → 20.27 |
| `r9` | **policy grads** | **on** | 21.63 → **21.14** ✅ | −3.30 → −2.47 | 20.94 → 20.50 |
| `r6_b10` | **frozen** | on | 21.94 → **23.54** ❌ | −3.46 → −5.15 | 21.27 → 22.88 |

**Two conclusions, and the second was not what the experiment was built for.**

1. **`freeze_policy` is what costs convergence here**, not replay. Unfreeze `fwd`
   and the sign flips regardless of the replay configuration.
2. **Replay is neutral, not harmful.** With `fwd` unfrozen, replay-on and
   replay-off are tied. So the "is a corrector necessary at all" arm does *not* say
   delete replay — it says replay neither helps nor hurts once the forward branch
   trains the policy.

**Independent corroboration, found separately and not connected until afterwards:**
`nys7cfrt` — the strongest baseline on the cluster and one of the best runs on
record — **has no `freeze_policy`**. The best historical run was not following the
thesis either.

**Honest limits, because this is one measurement.** 800 steps, one seed, one
problem. `tb_err` is a forward-branch metric and unfreezing `fwd` trains the policy
*on* that branch, so part of the gain could be the metric moving toward its own
trainer. `fwd/r2` and `over_coverage` move the same way, which argues against pure
metric-gaming, but the clean test is a held-out coverage measure over a longer
horizon — which is what the cluster arms are for.

**What survives either way.** The *mechanism* argument in §1 is untouched:
on-policy TB really is mode-seeking, and the blindness bound really is exponential.
What is in question is whether the correct response is to remove the policy
gradient from `fwd` **entirely** (a binary) or to weight it (a dose). rb0808 arm 14
`d30_dose` tests exactly that: Adam is invariant to a uniform loss rescale, so
total weight cancels and what changes is *adding a gradient direction at weight
0.2*. Dose-dependent ⇒ the confound is real; all-or-nothing ⇒ architectural.

## 2. The four jobs

Every module in the system does one of four things. This is the test to apply to
anything new:

- **(a) Maintain the ruler** — keep Z current on-policy.
- **(b) Supply diversity** — get states the policy would not reach on its own.
- **(c) Correct over-weighting** — fix what the policy gets wrong about states it
  does reach.
- **(d) Modulate rate** — set how fast (a)–(c) interact, without changing what
  they optimise.

| Module | Job | Verdict |
|---|---|---|
| TB objective (×3 branches) | a, b, c | **the thesis itself** |
| Prior buffer | b | literally "large pre-built buffer providing diversity" |
| Replay buffer | c | best-warranted subsystem in the codebase |
| Anchor buffer | b, indirectly | **weakest link — see §5** |
| Metric family | d (sensing) | level/spread decomposition maps error onto actuator |
| Protocol | sequences a–c; allocates between b and c | good structure; constants reduced to one judgement (`setpoint`) by `kind: ratio`, 2026-08-06 |
| Batch controller | d | clean |
| Buffer servo | d (freshness) | right idea, not configured, stale rationale |
| LR envelope | d | correct shape for the measured surface |
| **LR middle layer** | **none** | contains only problems it created |
| 9 of 12 loss terms | none, on this route | conditional-only |
| MMD family, `purge()`, 2 helpers | none | dead code |

## 3. What the design gets right

These are the arguments worth putting in a paper, roughly in order of strength.

**The trainer/sampler bijection.** One trainer per parameter group, one sampler
per trainer, enforced by a `freeze_policy`/`freeze_z` contract that detaches
**once at the source** rather than per-term. That single structural choice is
what makes branch roles composable — the freeze holds regardless of which
downstream terms are active.

**Level/spread as an actuator map.** `E[r²] = mean(r)² + Var(r)`. The level
component is what Z training can fix; the excess is what only policy training
can. Every control metric is an EMA-safe per-sample mean, never a ratio, because
the ratio family it replaced could not be EMA'd and was unreachable at low
sample counts.

**`E_μ[log P_F] ≤ −H(μ)`.** The collective level gap is not something backward
training can act on — the cloud cannot translate under normalisation. This is
why `relative_under` re-centres, and it is a real theorem doing real work.

**Freshness as a distinct actuator.** Loss weights cannot fix overfitting:
down-weighting a memorised buffer trains less on it but does not make it less
memorised. So replay freshness gets its own loop, with **one knob and one
invariant** — the boost holds occupancy exactly fixed (Little's law) and moves
only reuse and lag. `churn_rate`, `mean_residence_steps`, and `max_size` are
three handles on one steady state; moving them independently is how a buffer ends
up in a corner nobody meant.

**Memoryless residence over a hard TTL.** Exponential residence (CV ≈ 1) is a
wide lag distribution and therefore a strong lowpass on the policy → buffer →
gradient path; a hard age cap gives a uniform age profile with a sharp edge,
concentrating phase lag at one frequency. On a system with a documented phase-2
limit cycle, that is not abstract.

**Config owns behaviour; checkpoints own only position.** Live loss coefficients
are a pure function of (base config, current stage). The alternative — annealing
thresholds in place — silently reset progress on every resume.

**The uniform transition barrier.** Optimiser rebuild, monitor cooldown, LR
re-warm, batch-knee reset, coefficient rebuild, `stage_start` snapshot: every
item has a specific failure it prevents, and firing them automatically means a
new stage cannot forget one.

## 4. Key interactions

Couplings that are invisible from any single module.

**`fwd_frac` is set by the replay buffer, not by Z.** `fwd` is pinned at 0.2 not
because Z needs that much signal — Adam cancels a single source's scale — but
because replay admission draws its candidates from the `fwd` batch, and a thin
candidate pool degrades replay from prioritised to FIFO. **The Z branch's size is
a replay-buffer parameter.**

**Z lag propagates into every coverage metric.** A lagging Z makes the
Z-anchored `under_coverage` read "everything is under-covered," starving the
controller's other modes. Hence the re-centred variant. Z health is therefore a
*controller* input, not just a diagnostic.

**Huber `beta` silently rescales the anchor gate.** `tb_resid_clipped` is dL/dZ
only up to the constant `beta`, and it is one of the two metrics gating anchor
admission. `beta` is a per-mode, per-stage-overridable coefficient. Nothing
enforces the global invariant it depends on.

**Batch growth is shaped by `torch.compile`.** Every distinct batch size is a
recompile plus its own CUDA graph, so growth is deliberately coarse — the run
visits only ~log_f(max/base) sizes. A finer growth schedule would be better
control and worse throughput.

**Batch growth would multiply buffer churn without `churn_batch_ref`.** Churn
pacing is deliberately decoupled from the live batch size so a growth event does
not multiply turnover.

**A stage transition resets every controller at once.** LR cuts forgiven,
`stage_ctrl` replaced, batch-knee state cleared, Adam rebuilt, EMAs inherited.
With two stages this fires once — but the phase-2 controller's first ticks act on
phase-1's residual EMAs until they turn over.

## 4a. Why the replay buffer is necessary

> **Sharpened 2026-08-05** by [`to_do_rebuild.md`](to_do_rebuild.md) Part B,
> which derives the argument below rather than asserting it. δ is the pointwise
> log-ratio `log Q − log P` between the sampler and the target, so
> `E_Q[e^{−δ}]` and `E_P[e^{+δ}]` are both fixed by normalisation — which makes
> "the two blind spots are complementary" a theorem with a rate
> (`Q(δ < −m) ≤ e^{−m}`), and gives the buffer's value as a `1/q` visit-rate
> multiplier rather than a variance argument. Nothing below is overturned.

Replay was originally conceived narrowly — to tame the tails of the forward
residuals. Its current structural role is larger, and it needs a derivation that
matches. Split policy training into two jobs:

- **Discovery** — raise mass where reward is high and the policy has none.
- **Correction** — lower mass where the policy over-weights relative to reward.

**Off-policy TB from a buffer does discovery and is blind off its own support.**
Training on buffer samples pushes `log_pf(τ)` toward `log_pb + log_r − log_Z` for
each buffer trajectory. A policy satisfying all of them is correct *on*
`supp(μ_buffer)` — and the residual mass `1 − Σ_{x ∈ supp(μ)} P_F(x)` is
distributed arbitrarily over the complement, where the bwd loss cannot see it.
The policy can satisfy every buffer constraint perfectly while placing mass on
states the buffer never covered.

**On-policy TB does correction well and discovery badly.** Mode-seeking is
exactly that regions with `P_F ≈ 0` receive ≈ 0 gradient weight, so it cannot
raise mass it does not already have. The same property makes it excellent at the
other job: mass that should not be there has high `P_F` by construction, and so
receives full weight.

The two blind spots are complementary, which is why both samplers are needed.
That justifies *a* corrector. The buffer is justified separately: the correction
signal lives in rare high-`|resid|` samples, so a plain on-policy batch spends
most of its gradient on the uninformative bulk and its tail is too thin to
matter. Accumulating those samples across steps is what makes the corrector's
gradient actually about the tail; storing trajectories makes the replay exact
and free.

> **Replay is on-policy training, restricted to the half where on-policy
> training works, importance-reweighted toward the residual tails, with a buffer
> to make that reweighting statistically feasible.**

Corroboration that the system already encodes this: the balance controller drives
replay's share off `fwd/over_coverage` — the positive-residual tail, i.e.
precisely the correction job.

**Two arms test the two claims.** (i) Delete replay, keep bwd only: does
`over_coverage` grow without bound? (ii) Replace replay with `|resid|`-weighted
reweighting *within the live fwd batch* — no buffer, no storage, no TTL: does it
match? If (ii) matches, the whole replay subsystem collapses to a sampling
weight and the churn/hazard/cap machinery goes with it.

## 5. Where it does not hold together

**The anchor buffer is conditional machinery on an unconditional route.**
*(resolved 2026-08-03)* Its purpose is anti-forgetting for discovered modes. On
unconditional problems essentially every basin is pre-discovered and already in
the prior dataset — nothing to forget — so it is near-redundant here. On
conditional problems modes are discovered per condition and can be lost, which
is where it earns its cost.

So the thesis is safe: on this route diversity *is* pre-built, and the archive
should be marked conditional-only rather than tuned. This also explains why its
five gating constants are untested — the regime that would exercise them has not
run.

**The LR middle layer serves none of the four jobs.** Cut tier, latch, recovery,
hot clock, AIMD. Every mechanism in it exists to contain a problem created by the
mechanism before it; the only parts answering to external reality are the reset
tier and its rewind. It is complex where complexity does not pay and absent where
it would — the peak LR, which has a **measured 2.4× convergence effect and a
cliff above it**, is entirely hand-tuned.

**Three control loops can weld themselves off silently, and one currently has.**
LR recovery (when `recovery_target_frac ≤ cut_ratio` — fixed now, but it voided
three shipped batteries first), the bwd balance drive (**live**), and the buffer
servo (untested). All three share one shape: a one-sided drive whose zero is
indistinguishable from "satisfied." None reports how long it has been at zero.

**The balance controller's time constant is not comfortably separated from its
plant.** See below.

## 6. Time constants

The design principle is separation: each loop must be slower than the plant it
acts on.

| Loop / plant | Constant |
|---|---|
| Replay residence τ | 50 steps (mean, exponential) |
| `MetricTracker` EMA | ~100 train steps |
| **Absorption cycle** (bwd dragging in buffer states degrades fwd calibration) | **1000–2000 steps** |
| **Balance controller** (`kind: ratio`, `gain: 0.02`, one tick / 10 steps) | **~3300 steps** |
| LR warmup | 1000 steps |
| LR hold → decay onset | 21000 steps |
| LR decay half-life | 25000 steps |

Sensor (100) sits comfortably above the buffer (50), and the controller above
the sensor. **This was the binding complaint until 2026-08-06**: at `alpha:
0.005` the proportional controller's 2000 steps sat *at the top* of its own
plant's range (1000–2000), so at the fast end of the absorption cycle the margin
was 2× and at the slow end there was none.

`kind: ratio` widens it, and — worth noting — the constant is now **derived
rather than picked**. The loop is an integrator against a measured plant, so its
time constant is `1/(gain · d log ρ/dθ)`: at `gain: 0.02` and the measured
`d log ρ/dθ ≈ 0.15` that is ~3300 steps, comfortably past the slow end. The
caveat rides along: that plant gain is from a confounded 2000-step screen, so the
time constant is only as good as it is, and it must be re-read off the
controller's own `(rt_theta, rt_rho)` trace on the first real arm.

There is also a **second oscillation** the table does not cover, and it is not a
loop we built: the objective itself limit-cycles with a period of order 1000
steps and constant *relative* amplitude (~2× peak-to-trough on `ty4xdlzo`, while
the level falls 10× over the run). Its period and amplitude do not appear to
transfer between runs. Two consequences worth carrying: the `MetricTracker` EMA
passes ~79% of an 800-step cycle, so every consumer must do its own filtering
(the controller's integrator does, at ~6%); and any *score* read off a single
point is ambiguous by the cycle amplitude, so arms should be compared on
log-medians over ≥3 cycles or on fitted slopes.

## 7. If someone reads only one thing

The system is **one objective (TB) evaluated on three sampling distributions**,
with the flow head and the policy each receiving gradient from exactly one of
them. Diversity enters through a large, slowly-churned buffer; correction enters
through a small, fast-turnover prioritised replay of the policy's own rollouts;
and the on-policy branch exists to keep the partition function current so the
other two mean anything.

Everything else is rate control, and rate control is where most of the
complexity, most of the unwarranted constants, and all of the documented
failures live.

---

## Document map

| Doc | Covers |
|---|---|
| [`module_losses.md`](module_losses.md) | `gflownet_losses.py` — objectives, freeze contract, what is live |
| [`module_buffers.md`](module_buffers.md) | `CrystalBuffer` + prior / replay / anchor policies |
| [`module_training_modes.md`](module_training_modes.md) | fwd / bwd / replay / fused dispatch, fracs-as-weights |
| [`module_metrics.md`](module_metrics.md) | `quick_tb_stats`, `MetricTracker`, load-bearing set |
| [`reading_runs.md`](reading_runs.md) | **the interpretive layer.** Read order, metric tiers, the `R*` standing principles, the routine confound list. What a shape *means*, as against what a metric *is* (`module_metrics.md`) |
| [`module_protocol.md`](module_protocol.md) | stages, exits, transition barrier, balance controller |
| [`module_modulators.md`](module_modulators.md) | batch size, OOM, buffer servo |
| [`module_lr_controller.md`](module_lr_controller.md) | LR warmup envelope, the alpha* servo, the divergence bar. **Rewritten 2026-08-08 for v7** -- the cut/latch/recovery middle layer and the decay leg are deleted, so anything citing them is pre-v7 |
| [`module_bench.md`](module_bench.md) | `bench/` — CPU sandbox running the **real** controllers against synthetic surfaces with known ground truth. Where control-logic questions get answered without cluster time |
| [`decisions.md`](decisions.md) | **the entry point.** Part 1 is the docket (items needing the user's own call); Part 2 the register by closing condition; Part 3 closed |
| [`to_do_rebuild.md`](to_do_rebuild.md) | **forward-looking: the rebuild plan.** Part A LR controller, Part B batch construction, §0 the read-only gate battery, §C order of work — and the measured results from the 2026-08-07 local shakedown |
| [`audit_since_ty4xdlzo.md`](audit_since_ty4xdlzo.md) | code + config diff of the current HEAD against the trusted `ty4xdlzo` reference |
| ~~`register.md`~~, ~~`questions.md`~~ | **deleted 2026-08-06** — consolidated into `decisions.md` after they drifted into contradicting each other. Do not recreate; new open items go straight there |

**Convention worth knowing before cross-referencing.** `F-*` is the **global**
evidence series in [`findings.md`](findings.md) and is the ID to prefer. Under
[`PROTOCOL.md`](PROTOCOL.md), measurements migrate out of the module docs into
that series as each module is reworked — `module_buffers.md` has been done, so
its `B1`, `B6`, `B9` and `B10` no longer exist (B9's κ=0 identity is now `F-004`).

The remaining finding IDs are **module-local**: `B0…B8` in `module_buffers.md`,
`L1…L11` in `module_losses.md`,
`P1…P8` in `module_protocol.md`, `T0…T7` in `module_metrics.md`, `M1…M8` in
`module_training_modes.md`, `D1…D8` in `module_modulators.md`, `S*` per document.
`R1…R19` in `reading_runs.md` are **principles, not findings** — they carry no
grade and no scope line, because they are method rather than evidence.
Two of these collide with other series and the collisions are live: `to_do_rebuild.md`
uses `§B1…§B9` for the *design argument* (always written with the `§`), and
`decisions.md` uses `D2`, `D8`, `D26`… for *decisions* (a different `D` series from
`module_modulators.md`'s). Always qualify with the filename.
