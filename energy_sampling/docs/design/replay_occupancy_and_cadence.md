# Replay buffer × forward rollouts, in the simplest terms

Written 2026-09-07 for review. Companion to `rarer_rollouts.md` and
`z_fill_process_variance.md`.

---

## The whole thing in four lines

Under store-all (`churn_rate = batch_size`) with a full replay batch drawn every step:

```
admissions per step  =  B / N
draws per step       =  B                       (fracs are loss WEIGHTS; every branch runs a full batch)
occupancy            O =  B · τ / N
reuse per row        =  τ · B / O  =  N         (Little's law -- τ and B cancel)
```

`B` = batch_size, `N` = `fwd_rollout_every`, `τ` = `mean_residence_steps`.

Verified on `rr07_rr_n7_v1` (B=400, N=7, τ=35): predicted O = 2000, observed 2024–2381;
predicted reuse = 7, observed `replay_buffer_expired_draws` = **7.53**.

## The two failure modes, and the two variables they key off

| failure | what drives it | the row property | the metric |
|---|---|---|---|
| **staleness** — the row no longer represents the policy | how much the policy moved since the row was born | **age** | `replay/policy_drift_ess_frac` (also `_std`, `_nats`) |
| **overfit** — the policy fits these exact stored rows | how many times this row was trained on | **draws** | `replay/val_gap_nats` (with `_se`) |

Staleness happens whether or not a row is ever drawn — the policy moves regardless.
Overfit happens only through drawing. Different mechanisms, different variables.

### The loss weights are in both, with opposite signs

Age and draws are the *row* properties. What converts them into nats is how hard, and in
which direction, the policy is being pushed while the row is resident:

```
overfit_i    ∝  (times row i was drawn) × replay_frac   =  N · replay_frac
staleness_i  ∝  (steps row i was alive)  × bwd_frac     =  τ · bwd_frac
```

Training on row *i* pulls the policy **toward** it — that is absorption, and it is the
overfit term. Training on anything else moves the policy in directions uncorrelated with
row *i* — that is the staleness term. Since `bwd_frac + replay_frac = 1` (fwd pinned at
0), the *total* policy motion per step is roughly fixed and the split sets its
**direction**: replay-heavy moves toward the buffer, bwd-heavy moves away from it.

**Consequence: `gated_ramp` is already a staleness/overfit controller.** It sets the same
two failure modes that τ and N do. Four knobs, two failures, and one pair is already in
closed loop. Any second loop on τ/N against the same two metrics must be separated in
timescale from the frac loop or the two will fight — see §"Closing the loop".

## The knobs, and which failure each one owns

Under a memoryless hazard the age distribution of resident rows is exponential with mean
τ **whatever the arrival pattern**, so:

| knob | sets | costs |
|---|---|---|
| **τ** (`mean_residence_steps`) | mean age = τ → **staleness** | occupancy ∝ τ |
| **N** (`fwd_rollout_every`) | reuse = N → **overfit** | energy calls per step = 1/N |
| — | occupancy `O = B·τ/N` | is a *consequence*, not a lever |

**τ and N are independent levers on the two failures.** τ does not touch reuse (it
cancels in Little's law). N does not touch mean age (the hazard is memoryless).

What *is* welded is **overfit and cost**: both are N. Buying energy savings buys reuse,
with no way to give the reuse back except by drawing less than a full batch per step —
which cuts across `fracs` being loss weights.

> Correcting an earlier claim of mine: I said reuse was "welded to the cost knob" and
> implied staleness came with it. Wrong. Staleness has its own knob, τ. What is welded
> is overfit↔cost, not staleness↔overfit.

## The third constraint: the draw needs something to draw from

`O ≥ B`, or each step's batch of B is drawn with heavy repetition from a smaller pool.
Since `O = B·τ/N`, that is simply

```
τ ≥ N
```

The generator currently ships **τ = 5N**, i.e. O = 5B = 2000 rows. That is the only role
`τ = 5N` plays — it is a diversity floor, not a staleness choice, and it is why staleness
has *looked* coupled to N: tying τ to N makes mean age scale with N by construction.

`max_size` (12000) does not bind at all right now — the buffer runs at ~2.1k, 18% of cap.

## What to do with it

Untie τ from N and let each knob answer its own metric:

1. **Set τ from the staleness target.** Read `policy_drift_ess_frac`; raise τ if drift is
   cheap, lower it if drift is expensive. This is exactly the "tune the hazard timescale
   on mean drift" proposal, and it is the right shape: a *population* actuator driven by a
   *population* statistic. Per-row eviction on per-row drift is the biased version —
   it selects on the quantity the sensor then measures, so the survivors read healthy
   because the actuator removed the evidence.
2. **Set N from cost, bounded above by overfit.** Currently overfit is not binding
   (`val_gap_nats` 1.0 ± 1.9 at N=7), so N is free to rise until it is.
3. **Check `O = B·τ/N ≥ B`** — and preferably `≫ B`. This is the constraint that a loose
   max-period cadence violates first.

## What the numbers say today

At N=7, from `rr07_rr_n7_v1`:

- **staleness is the binding constraint**: `policy_drift_ess_frac` = **0.15**,
  `policy_drift_std` = **12.9 nats**. Rows are ~13 nats off-policy when reused.
- **overfit is not binding**: `val_gap_nats` = **1.0 ± 1.9 nats**, `resid_vs_intake`
  = 0.94 against a 1/e bar of 0.368. Both say no measurable memorisation.

So the immediate move is *down* on τ (less staleness, at the price of occupancy) and *up*
on N (cheaper, until the overfit metric can see something) — with `τ ≥ N` keeping the draw
diverse, which at large N is the binding one.

**Caveat on (2):** `val_gap_nats` cannot resolve below ~4 nats at `val_cap` 256. If N is
to be set by balancing against overfit, `val_cap` has to rise first, or the controller is
steering on noise.

---

## Closing the loop: what can and cannot be derived today

The goal is for τ and N to follow the two metrics rather than be tuned by hand. Two of
the four things that needs are available; two are not.

### Derivable now — the overfit bar's FLOOR

`val_gap_nats` is a difference of medians with a MAD-based standard error, so its bar is
bounded below by its own resolution:

```
se(val_gap_nats) ≈ 1.858 · MAD / sqrt(val_n)      ≈ 2 nats at val_cap 256
```

A 3σ bar is therefore **≈ 6 nats**, and anything under ~4 fires on noise. That is a floor
on the *bar*, not a statement about where overfit actually starts. To make the metric
usable as a controller input, `val_cap` must rise: se ∝ 1/√n, so **256 → 2048 gives
se ≈ 0.7 nats** and a 3σ bar near 2. That is the cheapest single change that makes an
overfit-driven N loop possible at all.

### Derivable now — that staleness is NOT a correctness problem

Worth stating because it changes what the bar means. Stored `(x, τ)` re-scored under the
current policy gives an **unbiased** TB residual: TB is off-policy-correct, and
`log_pb`/`log_r` are exact for the stored trajectory. Drift does not bias the replay
gradient. What it costs is **relevance** — a stale row trains the policy in a region it
no longer visits, and in the limit the replay branch degenerates into a second bwd branch
against a stale distribution. So the drift bar is a *usefulness* threshold, not a
validity one, and it cannot be derived from estimator theory.

### NOT derivable today — either optimum

Every measurement in hand is at **one operating point**: N=7, τ=35, B=400, on a warm
start we now know was bad (45% nonthermal). One point cannot locate a threshold in a
two-parameter space. Numbers we have there:

| metric | at N=7, τ=35 | reading |
|---|---|---|
| `policy_drift_ess_frac` | 0.15 | 85% of the replay gradient is spent on trajectories the policy has left |
| `policy_drift_std` | 12.9 nats | ~29% of the median residual the loss is fixing (≈45 nats) |
| `val_gap_nats` | 1.0 ± 1.9 | consistent with zero; below its own resolution |

### The measurement that would derive them

τ/N ∈ {1, 5, 20} sets occupancy (400 / 2000 / 8000 rows, all under the 12k cap); N ∈
{7, 20, 50} sets reuse. A 3×3 grid separates the two axes cleanly, and from the good warm
start a 600-step arm costs ~3.5 min — **~32 minutes for the grid**. Read
`policy_drift_ess_frac` against τ and `val_gap_nats` against N; the bars are where each
curve leaves its noise floor. Without that, any threshold written now is a guess wearing
a derivation.

### If both loops are closed, separate their timescales

`gated_ramp` already moves `bwd_frac`/`replay_frac` against these same two failure modes,
at 10-step ticks. A τ/N loop reading the same metrics is a second controller on the same
outputs. Give the frac loop the fast timescale and the τ/N loop a much slower one (a
few hundred steps at least), or specify explicitly which loop owns which metric — two
controllers steering one plant off one sensor pair is how a limit cycle gets built.
