# Phase 6 — the batch sizer, replaced

Argument, in the `docs/PROTOCOL.md` sense: it records *why* the replacement is shaped
this way, and is revised when the reasoning changes.

**STATUS: BUILT, 2026-08-19, state 8.** The control law is
`train.Modeller.select_batch_size` (+ `_conclude_batch_calibration`); the walk's four
config keys are retired (`utils._RETIRED_KEYS`); `batch_util_target` (unset = hold
the base, S3) and `batch_growth_cap` (rung step: geometric until the increment
reaches the cap, linear after -- bounds the selection's absolute overshoot past
the occupancy crossing to one accum quantum; 0 = uncapped) are the new keys, and the detection cases live in
`bench/test_batch_traps.py` / `bench/test_oom_ceiling_expiry.py`. The state-8 record
in `config_state.py` is the transition. Sections below are the argument that produced
it; §0.15 has been REVISED against `phase6_handoff.md`, which post-dates the original
decision and overturns its sensor choice.

Phase 6 says **replace, do not patch further**, with two objectives in strict priority:
satisfy the cluster utilization requirement, then maximize throughput subject to it.

**This document does not pick a single knee location or setpoint — and no longer needs
to.** The objective decision (§0.0) makes the throughput answer a constant, and the
handoff's measurements instantiate the occupancy side: an empirical threshold bracket
(§0.15) and a measured `t(B)` cost model (handoff §4.1). Every parameter below still
has a unit, a meaning, an instantiation route, and a stated behaviour when unset.

---

## 0. Three things that had to be established before a control law could be written

### 0.0 The objective, DECIDED — and priority 2's answer is a constant

**Decision (user, 2026-08-16): maximize optimizer steps/sec at a *threshold effective
batch size*.** Effective batch — samples per optimizer update — is held at
`A = fused_grad_accum_min_samples`; subject to that, maximize update rate.

This is the right cut because it removes the confound that made §0.1's two candidate
objectives incomparable: it **holds gradient quality fixed**, so the remaining question
is purely a rate.

Its consequence is larger than it looks. For any micro-batch `B <= A`:

```
updates/sec = 1 / (ceil(A/B) · t(B))  ~=  samples_per_sec / A
```

and `samples_per_sec = B/t(B)` is increasing in `B`. **So the optimum is `B = A`,
exactly** — and `configs/mk_dev.yaml` already ships it (`batch_size: 1000 ==
fused_grad_accum_min_samples: 1000`). Accumulation runs until `fused_accum_count >= A`,
so a `B` that does not divide `A` overshoots the effective batch; at `B = A` the
overshoot is zero, which reinforces the same point.

Three things follow, and they simplify the whole design:

1. **Priority 2 is a constant, not a search.** There is no knee to find. Its answer is
   `A`, and it takes no measurement to compute.
2. **Every growth above `A` belongs to priority 1.** Above `A` the effective batch is no
   longer held at threshold — growth buys gradient quality the desideratum holds fixed
   and pays update rate for it. So growth must be justified by *occupancy*, must be
   minimal, and must stop the moment the constraint is met.
3. **This inverts the shipping controller**, which climbs until `samples_per_sec`
   saturates — well above `A`, for throughput reasons this objective does not count.
   That is not a defect in the code; it is the code correctly optimizing the other
   objective.

**It also mostly dissolves trap (b):** with no throughput-driven walk there is no
descent, and `_batch_floor` stops being load-bearing. The trap's detection cases stay,
because the walk could be reintroduced and because the sandbox must catch it if it is.

**What this re-prioritizes in the measurement:** the ladder matters **below** `A`, not
above it. If `A` already sits past the hardware's saturation point, `samples_per_sec` is
flat below it and any `B` in the flat region is equally good — pick the smallest, for
memory headroom. See `phase6_measurement_delta.md` Δ6, which this promotes from last to
central.

### 0.1 Why the objective had to be decided: the two candidates have OPPOSITE argmaxes

`train.py` maximizes `samples_per_sec`, justified by *"updates/sec = samples_per_sec /
accum_target, so step time does not enter"*. The premise is `train.py:2467`:

```python
accumulating = accum_target > self.batch_size      # STRICTLY below
```

So:

- `B < accum_target` (fused only): `updates/sec = sps(B)/accum_target` — proportional to sps.
- `B >= accum_target`, or any non-fused stage: `updates/sec = 1/t(B) = sps(B)/B`.

Through a saturating cost curve `t(B) = t_fixed + B/s`:

| objective | value | shape in `B` | argmax |
|---|---|---|---|
| `samples_per_sec` | `B/t(B)` | rises, saturates | **largest** rung |
| `updates_per_sec` (`B ≥ accum`) | `1/t(B)` | falls | **smallest** rung |

`configs/mk_dev.yaml` ships `batch_size: 1000` **equal to** `fused_grad_accum_min_samples:
1000`, and `_batch_floor()` is `args.batch_size` — so **every reachable batch sits at or
above the target, entirely in the regime where the justifying identity fails.**

Three consequences the design is built around:

1. **FLAT / KNEED / NON-MONOTONE is a property of the *objective*, not of `t(B)`.** A
   curve that is KNEED in samples/sec is monotone-decreasing in updates/sec.
   Classification must happen downstream of the objective, in the controller — never
   baked into the measurement. The measurement request is right to ask for `t(B)` and
   `S(B)` and not for a classification.
2. **Trap (b)'s descent is *wrong* under one objective and *correct* under the other.**
   `train.py:417-419` says exactly this — "flat throughput genuinely does argue for the
   smallest batch" — and then implements only the samples/sec half.
3. ~~**The objective is therefore a PARAMETER, not a decision this design gets to
   make.**~~ **SETTLED in §0.0** — steps/sec at a threshold effective batch. The
   ambiguity was never resolvable by measurement, because the two candidates differ in
   what they hold *fixed*, not in what they predict: `samples_per_sec` lets the effective
   batch float, `updates_per_sec` at threshold pins it. That is a modelling choice about
   gradient quality, and it was the user's to make. It stays a logged parameter in the
   code so a board can still sweep it, but it now has a decided default and a reason.

### 0.15 The proxy — REVISED 2026-08-19 against the handoff, which overturns the sensor

**Original decision (user, 2026-08-16): adopt `gpu/util_policy` — the 2 h rolling
average — as the proxy, with an observed cancellation floor of ~60 %.** The ~60 %
number was derived from the **wandb system stream** (out-of-process), not from the
sensor itself; that matters below, because the number survives the sensor's death.

**The sensor half is FALSIFIED** (`phase6_handoff.md` §2, measured on
`a100_stab_aug16`): against the out-of-process stream over the same trailing window,
`gpu/util_policy` errs by −6 to **+40 points with a sign that flips with batch**, so
no fixed margin corrects it. Arms self-reporting 87–89 % were cancelled. Cause: the
in-process sampler sits inside the training portion of the loop, so eval, figure
logging and archiving contribute no samples. The metric survives as a logged series;
nothing may be *decided* on it.

**The replacement proxy is the out-of-process reading** — the wandb system stream
(~14 s cadence) and the `nvidia-smi` sidecar CSV (10 s, `joblogs/*_smi.csv`), which
read the same NVML counter. This upgrades the adoption from case 3 toward case 2: the
out-of-process numbers **agree with the only cluster-visible outcome available** —
arms at 38–40 % were cancelled at `02:00:2x`, a run at 49.4 % survived to its cap
(handoff §4.3). Still not full case 2 — cluster support, asked directly (2026-08-19),
supplied a sidecar recipe (confirming the 10 s `nvidia-smi` loop as the sanctioned
instrument) but **not the scheduler's statistic, window or threshold**, so the rule
itself remains inferred. Their sample output also shows 4-GPU nodes at wildly
different utilizations: any sidecar or system-stream reading **must filter to our GPU
index**, or it imports a neighbour's workload.

**The number that travels: threshold ≤ 49.4 % (a run survived there), cancellations
observed at ≤ 40 %, target 60 % as margin over the whole bracket** — margin still
required, both because the rule is inferred and because the in-run calibration
samples share the eval-blindness direction of error.

**The UMA-infeasibility consequence is RETRACTED.** The 52/44/49/42 ladder that made
the UMA feasible set look empty was read off the falsified sensor on the dev box, and
the handoff (§4.4) shows the real mechanism was batch pinning — the energy call's
memory ceiling capping the rollout batch — with `prod0810_nehzor_uma` sustaining
77.3 % for 47.9 h at rollout batch 1853. `INFEASIBLE` remains a required *output* of
the controller (built: the argmax-occupancy hold plus a diagnosis naming the binding
bound), but it is a per-run verdict, not a property of the route.

### 0.2 The utilization proxy cannot be a closed-loop input at controller timescales

Derived from the shipped sensor and the shipped growth cadence — how many growth
decisions fire inside **one** `gpu/util_policy` window (`F-036`):

| route / step time | growth interval | `util_recent` | policy ÷ interval |
|---|---:|---:|---:|
| ELJ dev box, 1.4 s *(measured)* | 70 s | 14 smp | **103×** |
| ELJ A100 batch 1000, ~0.5 s *(est)* | 25 s | 15 smp | **288×** |
| ELJ A100 batch 7410, ~3.0 s *(est)* | 150 s | 15 smp | **48×** |
| MLIP prod0810, 181 s *(measured)* | 9050 s | **4 smp — ABSENT** | **0.8×** |
| MLIP prod0810, 262 s *(measured)* | 13100 s | **3 smp — ABSENT** | **0.5×** |

The ratio spans ~600× and crosses 1. Worse, `_gpu_util_mean` is an unweighted trailing
mean over a window that is **never cleared at a batch change**, so after moving to `B₂`
the reading is `(n₂·U(B₂) + n₁·U(B₁))/(n₁+n₂)`: **reading `U(B₂)` to within a fraction
ε of the step requires dwelling `(1−ε)·W` at `B₂`.** At ε = 0.1 and `W` = 7200 s that is
1.8 h *per rung* — a 5-rung survey is ≥9 h of dwell before priority 1 is evaluable, per
stage. That is not a calibration; that is the run.

**Therefore priority 1 is FEED-FORWARD.** A predicate over a `U(B)` relation,
evaluated at candidate `B`. The in-run windowed reading is demoted to a slow,
stage-granularity **audit** that can *invalidate* the relation but never steers a
decision. This follows from window arithmetic alone, so it does not wait on Phase 4.

**Clarified 2026-08-19 — feed-forward does not mean no probing.** The 1.8 h-per-rung
dwell above is a property of the never-cleared trailing MEAN, not of measurement
itself: RAW samples taken during a rung's own dwell (60 s cadence in-process, 10 s on
the sidecar) make a per-rung occupancy reading a **minutes-scale** measurement. So
the built calibration walks the ladder once per stage over real train steps, reads
raw samples per rung, computes the selection once, and holds it; during the run the
only occupancy consumers are the S2 audit (one full policy window at the selected
rung, compared against the base rung's calibration) and drift watching via the
logged series. Probe at entry, hold, audit — never a closed loop on the windowed
mean.

### 0.3 The previous sandbox could not have detected trap (a)

`bench/old/clock.py`'s occupancy is `busy(B)/t(B)` with `busy = t_fixed·(1−host_frac) +
B/sps_max` — **monotone rising and saturating for every parameter setting.** It can
express the belief that batch is the occupancy lever and cannot express the measurement
that refuted it. Any occupancy rule tested there is right by construction.

That is why the sandbox is replaced too, and why its only measured cell is a
**table**, not a fit — see §3.

---

## 1. The structural rules

Everything else is bookkeeping. In plain words first, because each rule is just a
shipped failure mode inverted:

- **S1** — an occupancy reading may only *veto* batch sizes ("these rungs don't clear
  the constraint"), never *choose* one; the choice itself is a fixed rule (smallest
  surviving rung). Exists because the old occupancy rule actively drove batch upward,
  outranking a throughput gate that had refused every growth.
- **S2** — if candidates were excluded because "bigger batch → higher occupancy" and
  occupancy then didn't rise, the exclusion is withdrawn. The constraint keeps
  earning its removals.
- **S3** — no reading means no constraint: not "fine", not "assume the worst". With
  nothing measured the controller holds B = A, which is the shipping default.

Since §0.0 made priority 2 a constant, these three rules plus "hold at A, grow only
for occupancy" essentially *are* the whole controller.

**S1 — AUTHORITY DIRECTION (kills trap (a)).** Priority 1 is **set-valued and
subtractive**: it may remove rungs from a candidate set and may do nothing else.
Priority 2 is **the only selector**. There is no code path in which an occupancy
reading determines a batch size.

Strict priority stops being an `if`-ladder ordering — which is what it was, and what
let a false premise win — and becomes a *type* relationship:

```
feasible : set[int] -> set[int]        # priority 1. Subtractive. Cannot select.
select   : set[int] -> int             # priority 2. The only selector.
```

The deleted `gpu_util_floor` was an actuator holding a veto over another veto. Under S1
the same false premise produces a `feasible` that does not depend on `B` — and a
predicate that does not depend on `B` cannot order the batch anywhere.

**S2 — FALSIFICATION DIRECTION (kills trap (a) *and* the ratchet, one mechanism).** A
constraint that has removed candidates and has **not** produced the effect its own model
predicts, within its own declared window, **stands down and re-admits them.**

This already exists in one place — the runaway guard's unresponsiveness check
(`train.py:522-535`, *"the overrun is NOT batch-driven … standing down for stage"*) —
and it is precisely the mechanism that would have caught trap (a) **in flight**: the
occupancy rule removed small batches and utilization then went 52 → 42%, i.e. moved the
wrong way. Generalizing it to every constraint is the single highest-value idea here.

**S3 — EVIDENCE DIRECTION.** `UNKNOWN` never removes. `train.py:392-394` already states
this for the sensor (*"None means 'no reading', never 'fine'"*); the design extends it
to the verdict. A proxy that cannot answer leaves the candidate in. **This is today's
default path, not an edge case** — with no proxy, `feasible` is the identity and the
controller is a pure throughput optimizer over a bounded ladder.

---

## 2. The control law

### 2.1 Domain — reachability only, and it is finite before the first move

```python
def domain(m) -> list[int]:
    lo = P.B_LO or m._batch_floor()
    hi = min(m.args.max_batch_size,
             m.batch_size_oom_ceiling or INF,     # a BOUND, not a pin
             m._wallclock_feasible_hi or INF)     # runaway guard -> a BOUND
    if hi < lo:
        return []                                  # INFEASIBLE_DOMAIN: log, abstain
    f = P.LADDER_RATIO or float(m.args.batch_growth_factor)
    n = floor(log(hi / lo) / log(f))
    return sorted({int(round(lo * f**k)) for k in range(n + 1)} | {lo, hi})
```

`|L|` is finite and known before any move. **That number is the termination bound.**
Nothing in it is tuned to stop a walk, which is what makes termination independent of
the floor's *value* — set `P.B_LO = 1` and the argument is unchanged.

### 2.2 Shape classification, and the instrument it selects

The controller measures a **ladder**, then classifies the objective's shape over it,
then picks its instrument from the classification. It does not assume a shape.

| shape | meaning | instrument |
|---|---|---|
| `FLAT` | no rung resolvably better | **hold**; a walk is the wrong tool. Absorbing. |
| `KNEED` | interior optimum, **bracketed both sides** | select the argmax |
| `CEILING_BOUND` | best rung is the top, nothing above measured | **bound, not a knee** |
| `FLOOR_BOUND` | best rung is the bottom, nothing below measured | **bound, not a knee** |
| `NON_MONOTONE` | ≥2 sign changes | full survey only; local probes forbidden, sticky |
| `UNRESOLVED` | differences inside dispersion | abstain, hold incumbent |

`FLOOR_BOUND` is the mirror the candidate designs kept omitting, and it is not
hypothetical: under `updates_per_sec` the shape **is** monotone-decreasing, so a
classifier with only `CEILING_BOUND` returns "bracketed knee" at the lowest measured
rung with nothing below it — the exact defect the bracketing rule exists to prevent,
reflected.

**A knee claim requires bracketing rungs on both sides.** If the objective is still
improving at the top rung the finding is *"no knee below the ceiling"*, never *"the knee
is at the top rung"*. This is the clause that forbids inferring a knee from a monotone
climb that merely ran out of memory — the shape of the never-measured 10k+ knee.

### 2.3 Why `NON_MONOTONE` forbids local probes

Measured on the shipping gate: same cost model, same config, different starting batch —
`1000 → 1650` (40% of best), `2722 → 4491` (67%), `7410 → 7410` (84%). A two-point local
gate cannot distinguish a one-off step in the cost curve from saturation, and the
recheck only ever retests *downward*, so it cannot escape a bad pin. A ladder spanning
the reachable range is immune by construction; a local gate is not, at any starting
point. So the classifier's verdict **gates the instrument**, and the gate is sticky.

---

## 3. Parameters — the table, and what happens when each is unset

Nothing here has a value chosen from data that does not exist. The "unset" column is the
one that matters, because unset is the current state.

| parameter | unit | meaning | instantiated from | if unset |
|---|---|---|---|---|
| `util_predict(B, ctx)` | → `(pct, confidence)` \| `UNKNOWN` | the feed-forward `U(B)` relation | measurement-request arms A + B | returns `UNKNOWN` ⇒ `feasible` is the identity ⇒ pure throughput optimizer. **Correct today.** |
| `util_requirement` | pct + statistic + window | the cluster's rule | §0 email, or arms C/D/E | `UNKNOWN`; no rung removed |
| `OBJECTIVE` | enum | `samples_per_sec` \| `updates_per_sec` | gradient-quality-vs-`B`, **not funded** | `samples_per_sec` (today's behaviour), and the *choice is logged* so a board can sweep it |
| `B_LO` | samples | ladder bottom | reachability | `_batch_floor()` — today's value, but now a **domain bound**, not a stop |
| `LADDER_RATIO` | — | rung spacing | pre-flight | `batch_growth_factor` (1.65) |
| `DWELL_STEPS` | steps | per-rung measurement length | **Δ2** (not currently measured) | provisional dwells; rungs marked `provisional` and never used to *reject* |
| `RUNG_TTL_STEPS` | steps | staleness before re-measure | **Δ4** drift classification | `batch_knee_recheck_steps`; if drift measures STATIONARY, disable |
| `RESOLVE_EPS` | relative | when two rungs differ | within-job dispersion, **Δ2** | abstain ⇒ `UNRESOLVED` ⇒ hold |
| `AUDIT_WINDOW` | seconds | S2's falsification window | proxy window | `gpu_util_policy_window_s`; S2 dormant, logged as dormant |

**The unset column is a design commitment.** With every parameter unset the controller
is a bounded, terminating throughput optimizer that never removes a rung and never
claims a knee it did not bracket. That is strictly better than today and it ships before
the cluster data.

---

## 4. Where each safety mechanism lives

Dropping one of these is a regression, so each has a named home rather than being
absorbed into the control law.

| mechanism | today | in the replacement |
|---|---|---|
| OOM ceiling | pin + expiry | **domain bound** `hi`; expiry unchanged. A ceiling can no longer *pin* — it shrinks the ladder |
| `max_step_seconds` runaway guard | proportional cut + stand-down | **domain bound** `_wallclock_feasible_hi`; the stand-down latch is S2's first instance |
| grad-accum floor | cut stops at `accum_target` | unchanged, in `domain`'s `lo` |
| cooldown after a cut | `batch_size_cooldown_until` | unchanged, checked before any move |
| stage transition clearing | `protocol.py:1296-1302` | unchanged; the **ledger is per-stage** and cleared with it |
| never grow blind | rung with no baseline is measured first | subsumed: selection reads the ledger, and an unmeasured rung is not a candidate |
| remember what OOM'd | ceiling per stage, checkpointed | unchanged |

The pattern: **every mechanism that used to *pin* becomes a *bound*.** A pin is a
decision that competes with the selector; a bound shrinks the selector's domain. That is
S1 applied to the safety layer.

---

## 5. Trap prevention *and* detection

Prevention by construction is a claim. Detection under injection is evidence. The
sandbox is `bench/gpu.py`, `bench/batch_runner.py`, `bench/batch_metrics.py`,
`bench/batch_arms.py`; the cases are `bench/test_batch_traps.py`.

### Trap (a)

**Prevented** by S1: an occupancy predicate cannot select. If `U(B)` is measured flat or
declining, `feasible` does not depend on `B` and orders nothing. **Falsified in flight**
by S2 even if the model is wrong.

**Detected** by `OccupancyFloor`, which restores the deleted rule *in the textual
position it occupied* — top of the ladder, own early return.

**The verdict needed a second clause, and the false-positive check is what found it.**
Dominance alone — worse on the objective *and* not better on the constraint — convicts
the **shipping** controller too, because one exploratory probe up a declining curve costs
~1%:

| arm | sps | occ % | final `B` | distinct | |
|---|---:|---:|---:|---:|---|
| `null` | 57.70 | 52.0 | 100 | 1 | |
| `ship` | 57.14 | 51.6 | 100 | 2 | explored once, **returned** |
| `ship+occfloor` | 24.41 | 42.0 | 741 | 5 | **retained** the growth |

A detector that reddens for the current code as well as the injected code distinguishes
nothing. The separator must not be a magnitude threshold — *"worse by more than X%"* is
exactly the selected bar this project has retracted results over. The structural one:
the injected arm **kept** its growth (`net_rungs = 641`); the shipping controller ended
where it started (`net_rungs = 0`). A sign test on the trajectory's endpoints, with zero
as the natural boundary.

So the verdict is `dominates(arm, null) and descent(arm)['net_rungs'] > 0`. Negative
controls on `RISING` and `FLAT` report `UNRESOLVED`, not `PASS`; the shipping controller
is not convicted.

**A gap stated rather than closed:** an arm that wastes a great deal and still returns to
its start escapes this verdict. That is the cost-of-exploration question — a different
one — and it is **reported** as a number (`exploration_cost`, measured 0.0097 for the
shipping controller here) rather than folded into a pass/fail.

**Why this is not a chosen bar:** the two axes are never made commensurable. There is no
exchange rate between percent-occupancy and samples/sec anywhere in the verdict, which
is what lets it convict without knowing the cluster's threshold. `assert util_drop < X`
would need `X`, and `X` is Phase 4's undelivered deliverable.

### Trap (b)

**Prevented** by construction: there is no walk. Selection is an argmax over a finite
ladder, and `FLAT` is absorbing.

**Detected** by two assertions, because — measured — **one is not enough**:

- **B1, structural.** The device is stationary, so a converged controller returns the
  same answer at any horizon: `n_distinct` must not depend on it. Measured: floor intact
  → 2 / 2 / 2 at 20k / 40k / 60k; floor removed → **13 → 19**. *Contains no constant at
  all.* This is the property that made the tracking benchmark the one result that held.
- **B2, objective.** Where the device charges for switching (`recompile_s=30`), the
  objective convicts on its own, against `min` over the Fixed arms — the **worst**
  constant batch, regenerated in-cell.

**A correction to my own earlier claim, and to the trap's headline.** I argued the
objective would convict trap (b) via samples-delivered. Measured: at zero switching cost
the descent costs **exactly nothing** (regret ±0.0000) — the objective is *blind*, which
is why B1 exists and why `test_trap_b_objective_is_BLIND_at_zero_switching_cost` asserts
the blindness so nobody replaces B1 with a throughput check.

Two further measured corrections:

1. **The descent does not run to 1.** It runs to the closed-form knee and stops:
   `t_fixed=0.001 → 50`, `0.01 → 366`, `0.1 → no descent at all`. Reaching batch 1
   requires `t_fixed` **exactly** zero — a measure-zero, physically impossible point,
   and the only point at which the headline "descends forever" is reachable. The real
   defect is *"descends to a level the configuration never chose"*, which is what
   actually cost prod0810 (a whole stage at 0.825× its configured batch).
2. **The floor stops the descent, not the churn.** With the floor intact under FLAT the
   batch oscillates `1000 ↔ 1650` permanently — 57 transitions in 60k steps, `n_distinct
   = 2`. B1 passes it. `n_transitions` is therefore reported beside `n_distinct` rather
   than folded in, and under a dynamo cache-limit model that churn is not free.

### Grid edges, declared

Per the addendum: an unresolved reading at a boundary is not a verdict.

- `selection_edge(trace) ∈ {'low', 'high', None}` — both ends. An arm resting on
  `max_batch_size`, the OOM ceiling, or the floor has produced a **bound**.
- `excluded_fraction(trace)` — rows where the device extrapolated, dropped from every
  score and **reported beside it**.
- `cell_can_rank(scores, seed_spread)` — a cell whose between-arm range does not exceed
  its within-arm seed spread is declared a **NULL CELL**, never averaged.

**The two places a constant is unavoidable, named as the addendum requires:**

1. `cell_can_rank`'s ratio of **1.0** (signal ≤ noise). A domain boundary, not a tuned
   value — below it the cell is not measuring the arms. `bench/audit.py` uses 10 for the
   same job, and 10 *is* a choice.
2. B2's reference is `min` over the Fixed arms, not the best. The best-arm denominator is
   the one `bench/README.md` records as removed, and one of its three recorded failure
   modes was a grid-edge winner.

---

## 6. What this design consumes from measurement

Shape, not values — the values are the measurement's job. Written as a delta in
[`phase6_measurement_delta.md`](phase6_measurement_delta.md).

| consumed | shape of the requirement |
|---|---|
| `t(B)`, `S(B)` | ≥5 rungs geometric over the **reachable** range, bracketing an optimum on **both** sides, per route |
| **within-process** `t(B)` | the same ladder measured back-to-back in one process — a different quantity from the cross-job ladder, and the one a startup calibration actually consumes |
| `DWELL_STEPS` | the sampling distribution of a k-step median at fixed `B`; nothing currently measures whether 20 is enough |
| `U(B)` | at the ladder rungs, both routes, classified RISING / FLAT / DECLINING against a measured floor |
| drift | `S(B)` at fixed `B` across ≥4 disjoint blocks within one stage → DRIFTING / STATIONARY |
| `B_max` | bisected, **repeated across launches**; if it moves by more than one rung the ladder top is a random variable |

---

## 7. The design's weakest points, stated

1. ~~**The objective is unresolved and the two candidates disagree in argmax.**~~
   **SETTLED** — §0.0 decided it (steps/sec at threshold effective batch) before the
   build, and the built controller has no throughput search at all.
2. **`DWELL_STEPS` unset means every rung is provisional**, and a provisional rung may
   not reject. With no measurement the controller therefore surveys and holds, and never
   rejects a rung — safe, and slower than it needs to be.
3. **S2's audit window is the proxy's window.** On the ELJ route that is ~100 growth
   intervals, so falsification is stage-granular at best. S2 is a slow safety net, not a
   loop.
4. **The sandbox's cells are planted except one.** `umaperf0812` is the only cell tied to
   a measurement, it is one arm on one route with no repeat and no floor, and its top
   rung's occupancy was never recorded. Nothing here may be quoted as evidence about
   hardware.
