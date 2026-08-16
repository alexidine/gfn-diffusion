# ctrl_aug03 — controller battery for the `naive` stage

Launched 2026-08-03 02:09 local, sequential, detached. Deadline 07:09.
Status: `run_logs/STATUS.txt` (one START/END line per arm, with exit code and minutes).

## What changed in the code

Two additions to `protocol.py`, both opt-in per stage, both inert unless a config
declares them. Nothing else in the training stack was touched.

**1. `balance.kind: constraint`** — a one-sided integral (dual-ascent) controller.

    d_c = max(metric_c / bar_c - 1, 0)              # constrained side
    d_r = max(metric_r / bar_r - 1, 0)              # best-effort side
    theta += clip(gain * (d_r - priority * d_c), ±max_step)   # then clip to bounds
    share_r = sigmoid(theta)

`kind: proportional` is a static map, so its equilibrium is set by where the two
targets sit *relative to each other* — which is why a target that is a guess produces
a confidently wrong fixed point and nothing corrects it. (mk_dev's own `targets.bwd:
3.0` was annotated UNCALIBRATED for a metric it had already been swapped away from.)
An integrator moves whenever a drive is nonzero and holds when both are zero, so the
equilibrium is a property of the **bars**, and only bars that are actually *reachable*
have to be right. That is the direct answer to "the proportional target is hard to pick
ab initio": only the bwd bar — the one you can state a priori — has to be.

`priority` is a gain multiple, not a switch, so the constraint wins contests without
ever taking the whole batch: a soft lexicographic order with no discontinuity for the
split to limit-cycle on. `bounds` are load-bearing, not cosmetic — if the constrained
metric's floor sits above its bar, the drive never zeroes and theta parks at a bound,
so **the bound is the mix the run actually gets** and it must be safe to run forever.
A parse-time guard rejects a lower bound below the stage's `deactivate_threshold`.

**2. Stage-level `buffer_servo`** — the replay-overfitting actuator you asked for.

Sensor `replay/scatter_err ÷ fwd/scatter_err`; actuator a single boost `B` applied as
`churn_rate × B`, `mean_residence_steps ÷ B`, `toxic_min_draws ÷ B`. By Little's law
that holds **occupancy exactly invariant** and moves only reuse (`batch/churn`) and
policy lag. One knob, one invariant — churn, residence and max_size are three handles
on one steady state, and moving them independently is how a buffer ends up somewhere
nobody meant.

It is a *second* controller rather than a balance rule because loss weights cannot fix
overfitting: down-weighting a memorized buffer trains less on it, but does not make it
less memorized, and it also gives up the residual-tail correction replay exists for.
The two loops are near-orthogonal by construction — this one changes no frac, the
balance controller touches no buffer knob.

Deadband 1.0–1.5 with a slower release (`relax` 0.25) keeps it from being a ratchet
whose fixed point is `max_boost`.

## Arms

| # | run_name | controller | buffer servo |
|---|---|---|---|
| 0 | `fx_static` | fixed mix 0.05 / 0.75 / 0.20 | off |
| 1 | `cs_servo`  | constraint integrator | on |
| 2 | `fx_servo`  | fixed mix | on |
| 3 | `pp_servo`  | proportional (incumbent) | on |

Run order is by information value, so a short night still buys the important
comparisons. `fx_static` vs `fx_servo` isolates the buffer servo on a static plant;
the other three isolate the control law at a matched servo.

Held fixed everywhere: pure TB + `lr_flow` 0.1; **fwd pinned at 0.05 training both Z
and the policy** (`freeze_policy: 0`, changed from mk_dev's Z-only fwd); z_calibration
on; resume from `dev_mk_dev_…573c92_phase1_exit.pt` (step 6680) so phase 1 never
re-runs and contributes no variance; 2400 steps/arm, eval 500 / figs 2000.

Arms 1 and 3 get the **same bars** (bwd `relative_under_wcen` 2.0, replay
`fwd/over_coverage` 10.0) so the A/B is of the control law, not of two calibrations.

## Entry conditions, measured (bound to this resume point, not general)

From a 140-step smoke run of the `cs_servo` config:

- `replay/scatter_err ÷ fwd/scatter_err` = **0.964** — the overfitting inversion is
  real and present at the stock buffer (churn 80 / residence 50 / draws_per_row 35).
  The servo has something to act on.
- `bwd/relative_under_wcen` ≈ **3.4** — the 2.0 bar is breached from step 0.
- `fwd/over_coverage` ≈ **23** — against a bar of 10.
- `train_step_time` 1.38 s at batch 2831; eval ≈ 50 s.

## RESULT — `fx_static` baseline (2400 steps, done 02:57, 1.20 s/it, clean exit)

Both pathologies the new controllers target are real **and progressive** under a fixed
mix with a static buffer:

| quantity | entry | 2400 steps | |
|---|---|---|---|
| `replay/fwd` scatter ratio | 0.964 | **0.763** | inversion deepens — buffer overfitting is not a transient |
| `bwd/relative_under_wcen` | ~3.4 | **3.99** | drifts further from the 2.0 bar, not toward it |
| `fwd/over_coverage` | ~23 | 20.4 | roughly flat |

Other endpoint values: `fwd/scatter_err` 20.8, `replay/scatter_err` 15.9,
`bwd/scatter_err` 5.28, `fwd/tb_err` 21.1, `fwd/r2` −3.10 (r2 is a damage detector
only here, and fwd now carries policy gradient at only 0.05 weight — don't over-read
it). Fracs held exactly 0.0500 / 0.7500 / 0.2000 and `bs_log_boost` stayed 0.0,
confirming nothing moved on this arm.

This makes the battery well-posed: there is a worsening target for each controller.

**Consequence for `cs_servo`:** if a 0.75 bwd share lets `relative_under_wcen` drift
*up* to 4.0, the constraint controller's move to a 0.85 share will not halve it, so the
2.0 bar is probably unreachable by allocation alone and the arm should park at the
replay floor (`cs_at_bound` = −1). That is designed behaviour, and it is itself the
finding: it bounds how much of the absorption deficit is an allocation problem at all.
It also means `cs_servo` and `pp_servo` bracket the dose-response (replay 0.10 vs 0.25)
rather than both converging, which is what makes the direction question answerable.

## RESULT — `cs_servo` vs `fx_static`, matched at step 8680 (2000 past resume)

The constraint controller did exactly what was predicted: parked at the replay floor
(`cs_at_bound` = −1, fracs 0.05 / 0.85 / 0.10) because the 2.0 bar is not reachable by
allocation. The buffer servo climbed to **boost 7.1** (churn 80 → 570, residence 50 →
7.0, draws/row 35.4 → 5.0).

| metric | `fx_static` | `cs_servo` | |
|---|---|---|---|
| **replay/fwd scatter** | 0.743 | **0.966** | the overfit sensor |
| **bwd `relative_under_wcen`** | 4.073 | **3.622** | the constraint target |
| bwd `tb_err` | 15.27 | **14.00** | |
| `wass_debiased` | 0.00829 | 0.00770 | not interpretive on crystals |
| fwd `scatter_err` | 22.42 | 22.83 | wash |
| fwd `over_coverage` | 22.05 | 22.44 | wash |
| fwd `tb_err` | 22.73 | 23.12 | wash |
| **EffDim** | 5.894 | 5.905 | no collapse either side |

Ratio trajectory — the two arms start together and separate at ~step 7350:

```
fx_static  0.908  0.934  0.899  0.783  0.753  0.738  0.743   <- decays
cs_servo   0.913  0.945  0.916  0.964  0.963  0.970  0.966   <- held flat
```

So the new scheme is better on both quantities it targets, equal on everything else,
and costs nothing measurable. **But `cs_servo` changes two things at once** — the
controller cut replay 0.20 → 0.10 *and* the servo made the buffer fresher, and either
could explain the held ratio. `fx_servo` (fixed mix + servo) is the arm that separates
them, and it is the one to read first:

- `fx_servo` holds ~0.96 → **the buffer servo did it**; the frac cut is not needed.
- `fx_servo` decays like `fx_static` → **the frac cut did it**, and the servo at boost
  7 is not sufficient on its own.

## RESULT — attribution via `fx_servo`: the BUFFER SERVO is the load-bearing half

`fx_servo` differs from `fx_static` in exactly one thing (the servo) and from `cs_servo`
in exactly one thing (the frac controller), so the three arms attribute cleanly.

| metric | `fx_static` | `fx_servo` | `cs_servo` |
|---|---|---|---|
| | fixed, no servo | **fixed + servo** | constraint + servo |
| replay/fwd scatter | 0.743 | **0.916** | 0.966 |
| bwd `relative_under_wcen` | 4.073 | **3.616** | 3.622 |
| fwd `scatter_err` | 22.42 | **21.50** | 22.83 |
| fwd `over_coverage` | 22.05 | **21.11** | 22.44 |
| fwd `tb_err` | 22.73 | **21.76** | 23.12 |
| bwd `tb_err` | 15.27 | **13.47** | 14.00 |
| EffDim | 5.894 | 5.876 | 5.905 |
| `wass_debiased` | 0.00829 | 0.01201 | 0.00770 |
| replay frac | 0.20 | 0.20 | 0.10 |
| servo boost | — | **12 (SATURATED)** | 7.1 |

Two findings, one of them unexpected:

**1. The buffer servo does most of the work.** It alone takes the scatter ratio
0.743 → 0.916 (~78% of the total recovery) *and* improves every forward metric.

**2. The bwd absorption deficit was not an allocation problem.** The constraint
controller exists to fix `bwd/relative_under_wcen`, and it moved 10% of the batch
weight to bwd to do it — landing at 3.622. The buffer servo got **3.616 without
touching a single frac**. Allocation bought nothing beyond what freshness bought.
The plausible mechanism is indirect: bwd draws from the *prior* buffer, not the replay
buffer, so a fresher replay set helps only by keeping the policy better calibrated,
which then lowers the residual spread on prior-buffer states.

On the arms as configured, **`fx_servo` — the simplest arm carrying the servo — is the
best overall**: it beats `cs_servo` on every forward metric and ties it on bwd
absorption, losing only on the scatter ratio itself.

### Caveats, and the follow-up this points to

- **`fx_servo`'s servo SATURATED at `max_boost` 12** and its `bs_ratio` was still 0.92,
  under the 1.0 bar — the controller wanted to go further and could not. The optimum is
  not bracketed. The obvious next run raises `max_boost` (and/or cuts the base
  `mean_residence_steps`) and asks whether the ratio clears 1.0.
- `wass_debiased` is the one metric that disagrees (`fx_servo` 0.0120 vs 0.0083). Per
  the standing note it is not interpretive on crystals, and EffDim is flat across all
  three arms (5.88–5.91), so nothing collapsed — but it is unexplained, not dismissed.
- One seed, 2000 steps, one resume point. This is a screen.

## FINAL — all four arms, matched at step 8680

| metric | `fx_static` | `cs_servo` | `fx_servo` | `pp_servo` |
|---|---|---|---|---|
| | fixed, no servo | constraint + servo | **fixed + servo** | proportional + servo |
| replay/fwd scatter | 0.743 | 0.966 | 0.916 | 0.929 |
| **bwd `relative_under_wcen`** | 4.073 | **3.622** | **3.616** | **3.624** |
| fwd `scatter_err` | 22.42 | 22.83 | 21.50 | 21.12 |
| fwd `over_coverage` | 22.05 | 22.44 | 21.11 | 20.70 |
| fwd `tb_err` | 22.73 | 23.12 | 21.76 | 21.35 |
| bwd `tb_err` | 15.27 | 14.00 | 13.47 | 13.55 |
| EffDim | 5.894 | 5.905 | 5.876 | 5.833 |
| replay frac (end) | 0.200 | 0.100 | 0.200 | 0.217 |
| servo boost (end) | — | 7.1 | 12 (sat) | 12 (sat) |

### 1. Allocation does not move bwd absorption. Freshness does.

The three servo arms land at **3.622 / 3.616 / 3.624** — a 0.2% spread — across replay
fracs of 0.10, 0.20 and 0.217. A 2.2× range of allocation produces no effect on the
metric the frac controller exists to steer, while the servo alone moves it 4.073 →
3.62. Whatever `bwd/relative_under_wcen` is limited by on this route, it is not the
bwd/replay split.

### 2. `pp_servo` is a near-replicate of `fx_servo`, which gives a noise floor.

The proportional controller crept only 0.200 → 0.217 (alpha 0.005 = a 2000-step time
constant, so it was still converging toward its ~0.25 fixed point). That makes
`pp_servo` effectively a second sample of "fixed mix + servo", and the two agree to
~1–2% on the forward metrics and 0.013 on the ratio. `cs_servo`'s forward numbers
(22.83 / 22.44 / 23.12) sit **outside** that band, so driving replay down to 0.10 is a
real cost, not noise.

### 3. Recommendation

**Ship the buffer servo; leave the fracs at the fixed mix.** It is the simplest
configuration, it wins or ties on every metric, and the frac controller — in either
form — buys nothing on this route. `kind: constraint` behaved exactly as designed and
is worth keeping for cases where a constraint really is allocation-limited; this one
is not.

### 4. Follow-up `fx_servo_hi` (`max_boost` 30): NEGATIVE — boost ~12 is the ceiling

One variable off `fx_servo`. It climbed to boost **19.2** (churn 1538) and:

| | `fx_servo` (boost 12) | `fx_servo_hi` (boost 19.2) |
|---|---|---|
| replay/fwd scatter | 0.916 | **0.842** (worse) |
| `bs_ratio` | 0.922 | **0.849** (worse) |
| bwd `relative_under_wcen` | 3.616 | 3.664 |
| fwd `scatter_err` | 21.50 | 21.51 |
| fwd `over_coverage` | 21.11 | 21.13 |
| fwd `tb_err` | 21.76 | 21.77 |
| bwd `tb_err` | 13.47 | 13.48 |
| EffDim | 5.876 | 5.902 |

**The forward metrics are identical to 4 significant figures.** Churning 60% harder
changed nothing about training quality, and moved the servo's own sensor the *wrong*
way. So `max_boost` 12 was not a limitation — it is at or past the useful ceiling, and
this arm validates the original setting rather than extending it. **Keep `max_boost`
at 12.**

### The open question this raises about the sensor

`replay/fwd scatter` is **not monotone in the actuator**: more churn made it worse
while leaving every training metric untouched. The servo's control law assumes the
opposite, so past boost ~12 it is pushing on a knob that has stopped responding — which
is precisely why `max_boost` is load-bearing as a safety bound.

This does not undermine the `fx_static` → `fx_servo` A/B (one variable, better on every
metric). But it does weaken the *interpretation* that the ratio is a clean overfitting
sensor being driven back to health: the same knob pushed further degrades the ratio
with no cost to training. Whether the ratio is the right sensor, or merely correlated
with the right one over a limited range, is now an open question — and the honest answer
is that this battery cannot distinguish them. A sensor sweep (ratio vs boost, held at
several fixed boosts with the servo off) would settle it and is the natural next run.

`wass_debiased` across all five arms (0.0083 / 0.0077 / 0.0120 / 0.0112 / 0.0089) shows
no pattern in boost or in controller, while EffDim stays within 1.2% (5.83–5.91). That
is consistent with it being noise here rather than signal, as the standing note says.

## The prediction to check first

At those entry drives the two control laws **disagree in direction**:

- constraint: `d_r − priority·d_c` = 1.32 − 3(0.70) = −0.79 → replay driven **down**
  to its 0.10 floor.
- proportional: tilt gives replay 0.266 of the pair → **up** into its 0.25 ceiling.

So `cs_servo` and `pp_servo` should separate immediately and maximally. Whichever
improves coverage answers the question. Read `protocol/cs_at_bound` first: 0 means the
controller is still steering, ±1 means it parked and the bound — not the bar — is what
set the mix.

## Log channels added

`protocol/cs_theta`, `cs_share`, `cs_at_bound`, `cs_drive_{bwd,replay}`, `cs_bar_*`;
`protocol/bs_boost`, `bs_ratio`, `bs_churn_rate`, `bs_residence`.

## Step budget — compare at 2000

`fx_static` runs **2400** steps past the resume; the other three run **2000**.
The battery launched with 2400 everywhere, but a clean rate delta measured 15 min in
gave 2.03 s/it (GPU only ~44% utilised — partly IO/CPU-bound on buffer churn), which
put the servo arms within ~2% of the runner's 90-min per-arm timeout. A mid-run kill
would have produced arms at silently mismatched step counts, so the three arms that had
not yet started were retuned down to 2000 (the runner reads each arm's YAML at that
arm's start time, so no restart was needed and `fx_static` kept its 2400).

**Compare every arm at step 8680 (2000 past resume).** `fx_static`'s extra 400 steps
are a free tail on the reference arm, not part of the comparison. `experiment_log.yaml`
carries `compare_at_steps: 2000` on every row.

## Caveats

- Each arm's **stdout is block-buffered to file**, so `protocol:` lines don't appear in
  `run_logs/*.log` until the process exits. Progress is in `*.err.log` (tqdm); state is
  in wandb and in the checkpoints. Do not read stage transitions from merged-log
  ordering — it reorders.
- If the battery runs long, `pp_servo` is the arm that gets skipped or truncated. A
  truncated arm is not comparable at a different step count; `STATUS.txt` flags it.
