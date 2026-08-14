# bench — LR controller sandbox

CPU, seconds per run. Drives the shipping `LRController` and `RayCalibration`
against synthetic surfaces where the right answer is known.

**Rebuilt 2026-08-13.** The previous generation is in [`old/`](old/), intact and
still runnable. It was retired after an adversarial review found it had
accumulated defects faster than results — the summary of what went wrong is at
the bottom of this file, because most of it is reusable as a list of things not
to do again.

```bash
python -m bench.board            # the leaderboard: MLE game, Adam, 8 seeds
python -m bench.board 20 sgd     # 20 seeds, SGD control
python -m pytest bench/ -q
```

## The four pieces

| file | what it is |
|---|---|
| `surfaces.py` | the games. `MLEGame` is the simple case: `½θᵀHθ + cΣθ⁴ + gᵀθ`, log-spaced spectrum, exact optimum at the origin, exact `alpha_star_true`. |
| `arms.py` | the controllers, plus the fixed-rate ladder and the no-sensor control. |
| `runner.py` | one (game, arm, seed) → a trace. Nothing else. |
| `metrics.py` | five numbers, all pure functions of a trace. |
| `board.py` | the leaderboard. |

## Design rules, each of which is a scar

**No oracle.** Fixed rates are arms. The old stack selected a "best fixed rate"
and divided by it; that selection went wrong three separate ways (a rate 187x off
on one surface family, a grid-edge winner that bypassed its own guard, a
denominator pinned at the metric's floor) and each time it silently rescaled
results rather than failing. A leaderboard needs no reference: "at worst ~2x the
best fixed rate" is read by comparing rows.

**No threshold, no budget, no censoring.** The old topline was
`steps_to_target(run)/steps_to_target(oracle)` against a `2x budget`. Measured on
its last full run, 92% of all "over budget" events were non-arrivals and 3 of 900
runs were finite-and-over — the metric reported its own censoring. `final_loss`
is uncensored and continuous.

**Metrics are pure functions of the trace.** A metric can be corrected without
re-running anything. Four re-runs were spent on metric definitions in the old
stack because they were entangled with execution.

**Adam by default.** Hypergradient's rule is derived from the SGD update
(`dθ/dlr = −g`); production runs Adam everywhere (`train.py:1647`); every cell of
the old bench ran SGD. Under Adam the step direction is `m̂/(√v̂+ε)`, so the
derivation does not carry over. SGD stays as a control.

**Paired seeds.** One seed is one noise stream, shared across arms, so a row
difference is the arm and not the draw.

**Every arm must be distinguishable from `null`.** Enforced in `test_arms.py`. An
arm that silently no-ops does not error here — it posts a plausible row. `ray+ray`
did exactly that during the rebuild (armed after the optimizer step, so all 1900
readings returned `None`) and was bit-identical to the do-nothing arm.

## The metrics

| metric | definition | why this shape |
|---|---|---|
| `final` | median loss over the last 100 steps, relative to the best arm | median not mean (heavy tails); window not last-value (one minibatch draw) |
| `lead` | share of steps with the lowest smoothed loss, paired by seed | the only arm-vs-arm metric, and the reason no oracle is needed. Ties split evenly |
| `lr sd`, `max jump` | sd and worst single-step move of **log** lr | controllers act multiplicatively, so log is the natural space; `max jump` is what "wild swings" means operationally, and is far more legible than a 4th moment |
| `backslide` | share of the run where the loss is higher than `horizon` steps ago by more than its own noise explains | `mean(diff>0)` on a smoothed series returns ~0.5 for ANY noisy series — measured 0.501 on a flat loss — so it would have ranked arms by batch noise |
| `div / abort / nonfin` | counts, summed over seeds | the goal is "never 50x", a tail statement. A mean hides an arm that is excellent on 5 seeds and detonates on the 6th |

## What the arms are

`hyper` is **not** Baydin et al.'s published rule, and the old docs called it
that. Published is additive and unnormalised (`lr += β⟨g_t,g_{t-1}⟩`); this is
multiplicative on the cosine. They share a fixed point and nothing else — β goes
from units of `lr/gradient²` to dimensionless, which is exactly why one β serves
every surface here, and the additive form's self-annealing near a stationary
point is gone. On measured real gradients (‖g‖≈583, cos≈0.29) the published rule
at the paper's β=1e-7 would multiply the LR by ~80 in one step. The cosine form
is the defensible choice; the citation was not.

`ray+ray` goes through the real `LRController.on_calibration`, so it carries the
production `ratio**eta` damping, abstention policy and recorded ceiling. `hyper`
has no production counterpart — there is no hypergradient in `controller.py` —
so it writes `peak_scale` through the same actuator and bounds, but its update
law is bench code. The old harness claimed all arms shared an update law; they
do not, and pretending otherwise made the comparison look tighter than it was.

## Result (2026-08-13, MLE game, Adam, 8 seeds, 12000 steps)

```
arm                 nats behind    lead  lr/best   backslide  div
ray+ray                    0.00  74.7%     0.48      16.3%     0
fixed@0.001                0.92   1.0%     1.00       0.0%     0
fixed@0.003                1.95   1.7%     3.00       3.7%     0
fixed@0.01                 3.20   5.2%    10.00      13.4%     0
fixed@0.03                 4.21  10.0%    30.00      16.0%     0
hyper b=0.02               4.87   1.2%    89.09      12.7%     0
fixed@0.1                  5.35   6.1%   100.00      15.8%     0
ramp+plateau               5.98   0.0%   250.00      11.9%     0
null (no sensor)          17.33   0.0%     0.12       0.0%     0
```

**`ray+ray` beats the entire fixed ladder** — first time any controller here has
won rather than tied. It climbs to ~0.09 by step 2200, then decays to ~1e-3 by
5000 and tracks the optimum from there: an approximate decay schedule, which is
the right shape for this surface. `max jump 0.693` = ln 2, a deliberate 2x cut
from a resolved reading, with **zero divergences** — the probe braking, not a
tripwire.

**`hyper` ends 89x too hot and loses to four fixed rates.** It finds a rate that
is right early and never comes back down as the optimum decays. Bounded and
proportional means it cannot make a large correction, and near the noise floor
its cooling signal is weak.

Only the `lr/best` column shows this — `nats behind` alone cannot, because the
surface is flat above the optimum under Adam.

### Why the horizon changed the answer

At **2000** steps this board said the opposite: hyper 1.05x and a tie for first,
`ray+ray` 712285x behind. That run was measuring a race, not control:

- `warmup_steps = 1000` is half of it, and fixed arms get no warmup at all;
- an Adam run converges to its noise floor here in ~600 steps, so everyone who
  arrived tied and everyone who did not lost by a meaningless multiple;
- the ceiling for a controller was a TIE with the best fixed rate.

At **12000** steps the best fixed rate walks `0.03 → 0.01 → 0.003 → 0.001` — it
**decays ~30x** — so no fixed rate is right throughout and tracking can win. The
decay is the noise ball, NOT the quartic: identical at `quartic=0` and `0.01`,
one notch hotter early at `0.1`. `surfaces.py`'s claim that the quartic makes
`alpha*` RISE along the path is not what dominates.

### Score on the noise-free loss

`MLEGame`'s training loss is `½θᵀHθ + cΣθ⁴ + noise·θ`. As θ→0 the first terms
vanish and the loss becomes **the noise draw alone, sign and all** — so scoring
on it near the optimum ranks arms on a coin flip. `expected_loss()` zeroes the
noise term; the controller still acts on the noisy loss, as in production.

Checked: the ranking is unchanged under both (ray+ray first either way), so the
win is not an artifact of the loss definition. What moved is `backslide` — 0.7%
→ 16.3% for ray+ray — because the noisy loss was masking genuine uphill drift.

### The SGD control, and why the optimizer was worth rebuilding for

Same game, same arms, same seeds, `optimizer='sgd'`, 12000 steps:

```
arm                 nats behind    lead  lr/best   lr sd  max jump  backslide   div  nonfin
hyper b=0.02               0.00  36.9%     0.16   1.165     0.043      13.4%     0       0
hyper b=0.02 step          0.00  40.8%     0.16   1.165     0.043      13.4%     0       0
fixed@0.001                1.88   0.0%     1.00   0.000     0.000       2.5%     0       0
ray+ray p=100              2.04   0.0%     0.24   1.067     0.693       0.6%     7       0
fixed@0.003                3.04  22.3%     3.00   0.000     0.000      11.1%     0       0
ramp+plateau               3.26   0.0%     3.68   1.318     0.693       9.8%     8       0
null (no sensor)          11.89   0.0%     0.12   0.374     0.023       0.0%     0       0
fixed@0.0001              12.45   0.0%     0.10   0.000     0.000       0.0%     0       0
fixed@1e-05               16.57   0.0%     0.01   0.000     0.000       0.0%     0       0
fixed@0.01                never   0.0%    10.00   0.000     0.000      42.9%    32       0
fixed@0.03                never   0.0%    30.00   0.000     0.000      68.8%    32       0
fixed@0.1                 never   0.0%   100.00   0.000     0.000      68.8%    32       0
```

**The optimizer changes every answer.** The best fixed rate moves 10x (0.03 →
0.001) and the three rates that are *stable and near-optimal on Adam* all die on
SGD. But note what it does NOT do: **hyper is FIRST here**, tied at 0.00 across
both operands, against third on Adam.

THE TABLE THAT USED TO SIT HERE WAS A DIFFERENT EXPERIMENT. It reported hyper at
5.94x, `ramp+plateau` at 396.58 and ~1590 divergences per fixed arm, and
concluded that hyper "goes from comfortably inside the 2x goal to well outside
it". Reproduced exactly: those are the numbers for a **2000-step run with
`rewind=False`** — i.e. the horizon this file's own next section declares invalid
("that run was measuring a race, not control"), and a runner that no longer
exists. At the current default the conclusion inverts. Every number on this page
is specific to its optimizer AND its horizon AND its rewind setting, and this is
what happens when only the first is stated.

Note the catastrophe columns doing their job: `fixed@0.01` on SGD has a *better*
`backslide` than several healthy arms, because a run that spends most of its
length dead has nothing to slide back from. Counts, not averages.

### Known gaps

- ~~No parameter rewind.~~ **STALE — the rewind exists.** `Run` defaults to
  `rewind=True` and `_fire_loss_spike` restores parameters, a deep-copied
  optimizer state and the game's `extra_state`, on production's rate-based reload
  budget. This entry is what made the SGD table above survive: it read as a known
  limitation rather than as a table from a runner that had been replaced.
- **The MLE cell is effectively noiseless, and that is load-bearing.** Measured,
  consecutive gradients on it agree by 1.0000 in every quartile, and still 0.9985
  at 100x the surface's noise — the noise enters as an additive term while the
  signal grows with ‖θ‖. Real system for comparison: 0.29 on `fused`. So both
  gradient sensors are exercised outside the regime they would run in, `HyperSNR`
  is unusable here (removed from this board), and ray's probe resolves only 8.7%
  of firings against 63–82% on `trackboard` and `eqboard`. Use `bench.cos_axis`
  to place any new cell before trusting it.
- **`ray+ray`'s win on this board is not a resolved measurement.** Restricted to
  readings it actually resolved it is bit-identical to `null`: it starts cold,
  every reading returns "the step is far too small", that is not a resolved
  status, so it never acts and never reaches a rate where it could resolve. All
  of its movement comes from the grid-edge verdicts, so `alpha_target` is
  essentially not exercised here. A probe scoring random numbers does much worse,
  so the sensor does carry information — just not through its brackets.
- Only one game (`MLEGame`), one cell, no perturbation scenarios.
  `trackboard`/`eqboard`/`eqsuite` cover the others; `TrackingGame` is the only
  surface that passes its own fitness checks in every cell.

## What killed the previous generation

Kept because every item is a live hazard, not history:

- A reference rate used as four different things at once (denominator, scenario
  start, band centre, hot-start base). Changing it for one purpose broke the other three.
- Guards that ran before the value they guarded was re-selected.
- A scenario set where **44 of 65** cell×scenario columns had zero spread across
  every arm including the control — the battery was ~10 binary trials wearing
  1300 runs.
- Nine of thirteen "independent" cells sharing one reference rate, denominator
  and target.
- A metric whose band (2.0x) was exactly the reciprocal of the controller's
  divergence cut (0.5), so one cut landed bit-exactly on the boundary.
- Constants transcribed between documents rather than computed from data. One
  IQR was wrong by 1.69x and propagated into three conclusions.
- Docstrings that described the opposite of the code (`_time_oracle`: "NOT WIRED
  IN. Nothing calls this", while being called).
