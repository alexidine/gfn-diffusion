# The case for `fill_process_var` — why the absorber, and what q should be

Written 2026-09-07 for review. Subject: `z_calibration.fill_process_var` (q), the one
number that turns `z_level_fill`'s absorber from a lag machine into a filter. Companion
to `rarer_rollouts.md` item E.

**Claim in one line:** q is a property of the *system* (how fast log Z's fixed point
moves, in nats²/step); a fixed EMA rate α is a property of the *observation schedule*.
Under a design that deliberately sweeps the schedule, only the former can be set once.

---

## 1. The mechanism

Per fill, with `Δ` = steps since the last applied fill and `se` from `winsorized_z_root`:

```
P_pred  = P + q·Δ                        # how far Z could have moved while we weren't looking
K       = P_pred / (P_pred + se²)        # this measurement's share
log Z  += K · (root − log Z)
P      ← P_pred · se² / (P_pred + se²)
```

Two inputs, both of which vary in this design: **Δ** (the rollout cadence, 1–300 under
the loose-period proposal) and **se** (400-row training fills, 500-row eval fills,
4000-row bootstraps — a ~3× spread today and growing).

## 2. Why not a fixed-α EMA

A fixed-α EMA on the root has time constant **Δ/α steps**. It is not wrong; it is
*schedule-bound*. Two consequences:

**Cadence.** Change N and the smoothing changes proportionally, so α must be retuned per
arm. Worse, any sweep over N confounds cadence with smoothing — the thing the sweep is
supposed to measure. The absorber gets this right with one constant:

| N (steps between fills) | 1 | 7 | 20 | 50 | 100 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| K at q=0.01, se=1.5 | 0.064 | 0.162 | 0.257 | 0.373 | 0.481 | 0.667 |
| K at q=0.25, se=1.5 | 0.28 | 0.57 | 0.73 | 0.85 | 0.91 | 0.97 |

Longer blind interval → more trust in the new measurement, automatically.

**Precision.** At N=7, se=0.8 gives K=0.281 against se=1.5's K=0.162. The 4000-sample
bootstrap earns 1.75× the weight of a 400-row training fill. An EMA weights them
identically, which is the specific thing that made the step-0 fill — the *worst*
measurement of the whole run, se 2.60 at 18% unclipped — set the level outright.

## 3. q measured, not guessed

If `root_t = Z_t + ε_t` with `ε ~ (0, se²)` and Z a random walk of per-step variance q,
then consecutive fills satisfy

```
var(root_t − root_{t−1}) = 2·se² + q·N        ⇒    q = (var(Δroot) − 2·mean(se²)) / N
```

The observation noise is subtracted off, so what remains is the process rate alone.
Measured on the three N=7 acceptance arms (`rr07_rr_n7_v1`, `_uc0`, `_rat`):

| run | window | var(Δroot) | 2·se² | q | √q (nats/step) |
|---|---|---:|---:|---:|---:|
| v1 | 0–200 | 9.09 | 7.17 | 0.274 | 0.52 |
| v1 | 400–2000 | 8.03 | 4.53 | 0.499 | 0.71 |
| uc0 | 0–200 | 10.02 | 7.12 | 0.414 | 0.64 |
| uc0 | 400–2000 | 11.36 | 10.75 | 0.088 | 0.30 |
| rat | 0–200 | 9.90 | 7.26 | 0.376 | 0.61 |
| rat | 400–2000 | 13.12 | 11.38 | 0.248 | 0.50 |

**Shipped: q = 0.010 (0.1 nats/step). Measured: 0.09–0.50 (0.3–0.7 nats/step).**

Two readings fall out:

- **q is short by roughly 9–50×.** That is the whole explanation for K sitting at 0.162:
  the steady state of the recursion at q=0.01, N=7, se²≈2.3 is P=0.368, K=0.16, against
  observed medians P=0.352, K=0.162. The filter did exactly what it was told.
- **There is no transient/cruise split.** Cruise (0.088–0.499) overlaps transient
  (0.274–0.414). An earlier draft of this argument proposed a decaying q; the data does
  not support it. One constant, **q ≈ 0.25**.

## 4. What this does not establish

- **The random-walk model is assumed, not tested.** If the root's motion is systematic
  drift (the policy improving) rather than a random walk, q is not a noise parameter —
  it is a responsiveness knob wearing a statistical costume, and the estimator above is
  just "how much does the root move, minus how much of that is measurement error." The
  number is still the right one to use; the *justification* would be weaker.
- **The spread is 5.7×** across six estimates. This licenses "q ≈ 0.25, not 0.01"; it
  does not license three significant figures.
- **All six estimates come from one system (ELJ mipcas), one cadence (N=7), and a warm
  start we now know was bad** (`dev_elj_p2_cruise`, 45% nonthermal). The measurement
  should be repeated from `dev_race_L2_transition` (5.8% nonthermal) and at the cadence
  actually chosen, before q is written into the cluster generators.
- **It assumes consecutive fills' errors are independent** and that `se` is itself
  well estimated. Both are plausible, neither is checked here.

## 5. What would falsify it

- A **negative** q estimate on a healthy run: `var(Δroot) < 2·se²` means the root moves
  less than its own measurement noise, i.e. the random-walk term is unnecessary and
  a fixed K is the honest model. None of the six were negative.
- **q varying strongly with N** across a cadence sweep. q is defined as per-step, so a
  correct model gives the same q at N=1 and N=50. If it does not, the root's motion is
  driven by something other than elapsed steps and the whole parameterisation is wrong.
- **K near 1 at every fill** after the change, with log Z tracking root noise. That
  would say q is now too large and the filter has degenerated to a snap.

## 6. If the argument is rejected

The fallback is not a plain EMA — it is the table in §2, computed by hand: set `K`
directly, per cadence. That is the same filter with the derivation done offline, and it
is honest as long as the per-cadence values are recomputed whenever N changes. What is
*not* defensible is a single fixed α across a cadence sweep, because that silently
changes the smoothing timescale by the same factor as the cadence.

## 7. Proposed change

- `fill_process_var: 0.01 → 0.25`, on the local arms first.
- Re-estimate from `rr07_rr_n7_race` (the good warm start) before the cluster generators
  take it — §4 says the current six are all from a bad phase-1 exit.
- Log the estimator alongside the fill so q is checkable from any run rather than by a
  scratch script: `var(Δroot)` and `mean(se²)` over a trailing window are two more series
  on the `z_fill/` block.
