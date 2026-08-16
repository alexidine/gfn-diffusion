# lr_aug08 — results

Seven arms, 2026-08-08, all on one shared post-transient resume (`batt0807_p1`
running.pt @ step 2650), T=10, batch 1000, `checkpoint_read_only`. Buffers held
at the **pre-B7b** configuration throughout so the LR axis is not confounded with
the prioritised-replay package.

Predictions were written down first: [`PREDICTIONS.md`](PREDICTIONS.md).
Analysis lives in `docs/module_lr_controller.md` F6–F9 and `docs/decisions.md`
D31/D32.

---

## The arms

| arm | LR regime | steps | outcome |
|---|---|---|---|
| `a_fixed` | pinned 1.25e-4, servo off | 5400 | **validity gate — PASSED** |
| `a_climb` | servo, target 1.0, seed 1e-5 | 5400 | converged to 3.2e-4 (α\*=1.001); **−2.2 nats vs a_fixed** |
| `b_descend` | servo, target 1.0, seed 4e-4 | 5400 | 🔴 **RUNAWAY** |
| `c_low` | pinned 1.563e-5, servo off | 5400 | best bwd, worse fwd |
| `v0`/`v0b` | forced detonation @ 1e-1 | 400/200 | divergence path **PASSED** |
| `v1_zcalrep` | `z_calibration.mode: replay` | 400 | runs, **harmful** |
| `d_cal_below` (**pair CAL**) | servo, target **1.87**, seed 1e-5 | 5400 | → 1.141e-4; right LR, **+2.2 fwd** (path cost) |
| `d_cal_above` (**pair CAL**) | servo, target **1.87**, seed 4e-4 | 5400 | → 1.405e-4; **matches a_fixed** ✅ |

## Head-to-head at 5400 matched steps

Every cell is the **median of the last 15% of evals** — one statistic throughout,
never a point sample (`decisions.md` D30 records that error costing a whole
reading once).

| | `c_low` 1.56e-5 | `a_fixed` 1.25e-4 | `a_climb` 3.2e-4 | `b_descend` 4.4e-4 |
|---|---|---|---|---|
| `bwd/tb_err` | **13.33** | 15.04 | 17.20 | 15.27 |
| `fwd/tb_err` | 18.73 | **17.89** | 20.73 | **34.91** 🔴 |
| `fwd/logw_std` | 18.48 | **17.68** | 20.43 | **34.66** 🔴 |
| `alpha_median` | 13.81 ⚠ | 2.01 | 1.001 | 1.033 |

⚠ `c_low`'s α\* is **censored** — `fit_ok_rate` was only 0.57 there, so 13.81 is a
lower bound (see `step_probe.py::servo_reading`).

`a_fixed` reproduced `local_aug08` `a_frz` (15.04 vs 15.14, seed floor 0.04), so
the v7 rewrite changed the LR path and **not** training. Everything else reads on
something.

## What the battery established

**1. The control loop is correct.** `a_climb` climbed 26×, parked, and held
`alpha_median` at **1.006** against a target of 1.0 — `peak_scale` 32.2 against a
bounds ceiling of 200, so a fixed point and not a bound, with `servo_hold` 0 for
the entire run.

**2. `α* ∝ 1/lr`, and now within a single run.** Over the eight uncensored rungs
of `a_climb`'s sweep, `lr × α*` has median 3.07e-4, spread −5%/+18%.

**3. The setpoint is wrong.** Following α\*=1.0 costs 2.16 nats of `bwd/tb_err`
and 2.84 of `fwd/tb_err` against a hand-set LR. **α\* transfers as a *shape*, not
as a *setpoint*.**

**4. 🔴 And α\*=1.0 is a positive feedback loop.** `b_descend` blew `fwd/tb_err`
21 → 35 with `alpha_median` at 0.92–1.14 throughout, the LR creeping 3.1e-4 →
4.5e-4 *because* of the degradation. `bwd/tb_err` **improved** across the whole
collapse. No guard fired.

**→ shipped:** the one-sided brake, `clip: [0.8, 1.0]`. The multiplier is then
≤ 1 always, so `peak_scale` can only fall and the loop cannot form.

**5. ✅ And with a CALIBRATED target it works, from both sides.** `target: 1.87`
(read off `a_fixed`'s own second half) seeded 11× below and 3.2× above converged
to **1.141e-4** and **1.405e-4** — agreeing to 1.23×, with the hand-tuned 1.25e-4
*between* them, α\* held within 3% of target in both, and no runaway.

**6. The approach DIRECTION costs more than the destination.** `d_cal_above` is
indistinguishable from the hand-tuned arm (bwd −0.26, fwd +0.08). `d_cal_below`
lands at the **same LR** and is **2.2 nats worse on fwd**, having spent the run
climbing through rates that were too low. Descending onto the answer is nearly
free; climbing onto it is not — under-training is never recovered by arriving
eventually. This was `paird.py`'s third pre-registered falsifier, and it fired.

## Three defects the battery found

| | found by | fixed |
|---|---|---|
| probe returns 100% `downward` below ~1e-5, so the servo is **inert** | the 150-step smoke arm | `beyond` / `downward` split (§A3b) |
| servo holds through warmup but the probe keeps **buffering**, so the first tick acts on warmup-era readings — 34% overshoot, wrong direction | `b_descend` | `flush_window()` (F7) |
| `alpha_median` is **censored downward** where `fit_ok_rate` is low | `c_low` | documented; `read.py` marks affected rows |

The first is the one to remember: **a 150-step smoke arm caught a defect that
would have made a 6-GPU-hour battery measure nothing** — every arm would have sat
at its seed and reported a clean, stable, entirely fictional result.

## Two caveats not to lose

- **The `fwd` U is shallow** — 0.84 nats between 1.56e-5 and 1.25e-4 — and
  `c_low`'s `fwd/tb_err` was still falling at the end while `a_fixed`'s had
  flattened. A longer run could close or invert it.
- **`lr × α*` is not a route constant.** `a_fixed`'s is 2.34e-4 against
  `a_climb`'s 3.07e-4 — 24% apart at overlapping LRs — because one measured while
  *living* at that LR and the other while *passing through*. Pair CAL is what says
  whether a calibrated setpoint is a property of the route or only of the run it
  was measured on. (**`pair D` in these docs always means `local_aug08`'s** freeze x LR 2x2 -- hence the different name here.)

## Reading the arms

```bash
python configs/lr_aug08/read.py
```
