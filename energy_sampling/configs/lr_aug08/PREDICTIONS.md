# lr_aug08 — predictions, written BEFORE the arms finished

> ## OUTCOME (pair A, 5400 steps) — P1 ✅, P3 ✅, P2 ⚠ NOT ESTABLISHED
>
> | | predicted | measured |
> |---|---|---|
> | **P1** servo lands at α\*=target | ~2.9e-4 | **3.2e-4**, `alpha_median` **1.006**, `peak_scale` 32.2 vs a 200 bound — a fixed point, not a bound |
> | **P3** α\* at the reference is ≫1 | law-based | ✅ `lr × α*` = 2.5–3.3e-4 across 26×; α\*=1 sits at 3.2e-4 |
> | **P2** the fixed point is too hot | ~8× above optimum | ✅ **as a head-to-head**: a_climb is +2.19 nats `bwd/tb_err` and +2.89 `fwd/tb_err` vs a_fixed. ❌ **as a location**: see below |
>
> **Validity gate PASSED.** `a_fixed` finished at `bwd/tb_err` 15.04 vs
> local_aug08 `a_frz`'s 15.14 (seed floor 0.04). The rewrite changed the LR path
> and not training.
>
> **P2's falsifier fired on P2 itself, and it was my own arm that did it.** The
> in-run sweep rises monotonically with LR, which I read as locating an optimum
> near 1.5e-5. But `a_fixed` at a *constant* 1.25e-4 **improves** over the same
> window (`fwd/tb_err` 22.4 → 17.9), which the sweep would have predicted gets
> worse. A curve measured under a *moving* LR is not a steady-state curve. So the
> direction is established and the location is not — `c_low` (1.56e-5 held for a
> full run) and `b_descend` are the arms that settle it.
>
> **What survives without qualification:** the loop converges reliably, holds its
> setpoint to 0.6%, obeys `α* ∝ 1/lr` — and following that setpoint is worse than
> a hand-set LR by ~2 nats. The sensor works; the setpoint is wrong.


Recorded 2026-08-08 at ~1350/5400 steps of pair A, from two numbers already in
hand. Written down first because the α\* law below makes every remaining arm's
landing point *calculable*, and a prediction that is only stated after the fact
is not a test of anything.

## The law these rest on

`α* ∝ 1/lr`, measured **twice, independently**:

| source | LR ratio | α\* ratio | error |
|---|---|---|---|
| local_aug08 pair D (cross-arm, 800 steps) | 1.72 | 1.73 | 0.6% |
| **lr_aug08 a_fixed vs a_climb** (this battery, matched steps) | 3.28 | 3.22 | **1.7%** |

`a_fixed` reads `alpha_median` **2.33** at lr 1.25e-4; `a_climb` reads **7.51** at
3.815e-5. So on this route:

```
alpha*(lr)  =  2.33 * (1.25e-4 / lr)        =>   lr(alpha*) = 2.91e-4 / alpha*
```

## Predictions

**P1 — the servo's fixed point is where α\* = target, so with `target: 1.0` every
servo arm lands at lr ≈ 2.9e-4**, regardless of which side it starts on.

| arm | seed | predicted landing |
|---|---|---|
| `a_climb` | 1.0e-5 (below) | **~2.9e-4**, reached from below |
| `b_climbB` | 1.0e-5 (below) | **~2.9e-4**, same as a_climb |
| `b_descend` | 4.0e-4 (above) | **~2.9e-4**, reached from above |

`a_climb` and `b_descend` converging on the same value **from opposite
directions** is the fixed-point test — pair B supplies it, so the pair C built
for that purpose is freed up for the calibration question instead.

**P2 — that fixed point is far too hot.** The in-run LR sweep has `bwd/tb_err`
rising monotonically with lr (13.50 at 1.56e-5 → 14.14 at 3.8e-5), and
extrapolating to 1.25e-4 gives ~15.9 — which is *exactly* what `a_fixed`
independently reads (15.89). So the sweep's rung effect is tracking a real LR
effect and not just elapsed time. If the trend continues, the optimum is at or
below **~1.5e-5**, roughly **8× below** the "known-good" reference and **19×**
below the servo's target-1.0 fixed point.

**P3 — therefore α\* at the empirical optimum is ~18, not 1.** From the law:
`α*(1.56e-5) ≈ 18.3`. §A4 assumed 1.0. If P1 and P2 both hold, `target` is wrong
by more than an order of magnitude and **the sensor is fine while the setpoint is
the entire content of the design.**

## What would falsify each

- **P1 fails** if the arms land in different places, or drift without settling →
  the loop integrates noise; ship `clip: [lo, 1.0]` as a one-sided brake.
- **P2 fails** if `b_descend`'s `bwd/tb_err` *rises* as its LR falls. That is the
  arm that breaks the time confound: in `a_climb`, LR and elapsed time increase
  together, so the two cannot be separated within it. In `b_descend` they move in
  opposite directions.
- **P3 fails** if the sweep's minimum is not below the reference — then α\* ≈ 2.3
  at the optimum and `target` is a modest correction rather than a rethink.

## The one that would be worst

All three holding **and** `a_fixed` failing to reproduce local_aug08's 15.14.
Then the v7 rewrite changed training rather than the LR path and none of this
reads on anything. Check `a_fixed` first.
