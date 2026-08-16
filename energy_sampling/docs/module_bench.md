# Module: `bench/` — the controller sandbox

A CPU-only harness that runs the **real** control code against synthetic loss
surfaces and a synthetic GPU, on a laptop, in seconds. It exists so that
questions about batch sizing, LR scheduling and stage transitions stop costing
cluster time.

## What it is for

Every controller in this codebase is problem-blind. `LRController` reads
`args`, `optimizers`, `step_ind`, `phase`, `lr_ctrl`, `ray_cal` — and nothing
else. `RayCalibration` takes a parameter list and two callables.
`increment_batch_size` is a control law over a timing series. None of them can
see the energy function.

They are nevertheless only exercisable today by launching a crystal run, which
is why four expensive failures were all found on the cluster: the ray-probe
clock aliasing, `batch_growth_min_gain` scoring a steps/hour loss as a win,
post-OOM blind growth, and v7's unreachable quorum. All four are pure control
logic and all four are reproducible here in milliseconds.

The second reason is **ground truth**. On a crystal there is no oracle, so
"the controller is broken" and "the problem is hard" are not distinguishable.
Every surface here knows its own optimum, its own curvature, and — for
`equilibration` — the exact LR at which it goes unstable.

## What it cannot do

It transfers **mechanism and correctness**, never parameter values. No LR,
`alpha_target`, batch size or tolerance measured here should be copied into a
production config. Curvature, anisotropy and loop gain are properties of the
real problem; the bench can tell you what a controller *does* with a given
surface, not which surface a crystal presents.

Grade accordingly: a derivation checked against a measurement is `MECHANISM`; a
number off one synthetic surface is `OBSERVED` and carries the surface in its
scope line.

## Layout

| File | Role |
|---|---|
| `clock.py` | `SyntheticGPU` — step time with a **planted** knee and OOM ceiling, plus the answer the controller should find |
| `surfaces.py` | the three games (below) |
| `fake_modeller.py` | duck-typed `Modeller` stand-in; `make_args` carries mk_dev's defaults |
| `real_modeller.py` | builds the **actual** `train.Modeller` on CPU, for checking the fake against |
| `harness.py` | `BenchRun` — the driver loop, transcribed from `train.py:1668-1710` |
| `oracle.py` | brute-forces the best **fixed** LR a surface admits, and checks it |
| `scenarios.py` | the battery: regret, recovery, detectability; `python -m bench.scenarios` |
| `experiments.py` | the answer-producing runs; `python -m bench.experiments [name]` |
| `test_fidelity.py` | asserts the fake still stands in for the real one |
| `test_*.py` | the regression suite |

## The discipline that keeps it honest

**The bench fakes the modeller, never the controller.** `LRController`,
`RayCalibration`, `Modeller.increment_batch_size` and
`Modeller.handle_train_epoch_error` are imported and run unmodified — the last
two are bound onto the fake class as plain functions, so the batch-sizer tests
execute the shipping code with a fake `self`.

This matters because the alternative rots silently. `energies/twenty_five_gmm.py`
still defines `energy(self, x)` against a `BaseSet` that has required
`energy(x, mol_batch, log_temperature, return_exp)` for months; the toy cannot
run and nothing noticed, because nothing imported it. A reimplemented control law
would drift the same way and report green while doing so.

Candidate control *policies* are A/B'd through `BenchRun(reading_filter=...)`,
which sits between the sensor and the controller. The baseline arm is therefore
always the real code.

## The fidelity harness

A stand-in is only worth anything while it still stands in, and that claim dies
silently. So `test_fidelity.py` builds the **real** `train.Modeller` from the
real `configs/mk_dev.yaml` — possible on CPU since the guard at `train.py:129` —
and checks the fake against it:

- every attribute a controller reads exists on both (`COUPLING_SURFACE`);
- every config key a controller reads exists on both (`ARGS_SURFACE`);
- the bench's transcribed mk_dev values still **match the shipping config** —
  if mk_dev is retuned this fires, and the message says which key drifted;
- given the same args and step, the real and fake modellers come out at the
  **same learning rate**, across warmup and past it, on separate optimizer
  objects.

`optimizers` and `fused_accum_count` are `DEFERRED_SURFACE`: read by
controllers, but built by `init_gfn`, which needs the model and the datasets. A
bare `Modeller` does not have them, and a test pins that split so the fake stops
inventing them if it ever changes.

This harness immediately earned its keep — it caught that the bench was
importing `controller` while `train.py` imports `energy_sampling.controller`,
i.e. **two distinct module objects for the same file**. Behaviourally identical
here (the module holds no state) but `isinstance` failed, and anything
module-level would have diverged. The bench now imports by the same path
`train.py` does.

## The battery

`scenarios.py` is where the bench stops describing and starts scoring. Every
scenario is measured against the **oracle** — the best fixed LR the surface
admits, found by brute force — on three axes:

| | |
|---|---|
| **regret** | final distance ÷ oracle's final distance. 1.0 = the controller is free. |
| **recovered_at** | first step after which the run stays within `tol` of the oracle's own *trace*. `None` = never. |
| **detectability** | would anything in the logs have told you? Scored from `raycal/status`, `lr_ctrl/peak_scale`, `lr_ctrl/divergences`. |

The third exists because both absorbing failures found so far are **silent**: a
run stranded at its seed LR trains orders of magnitude worse and looks like slow
progress. A battery that only scored recovery would produce controllers that
handle the failures we can already see.

Scenarios: `oracle_fixed` (the metric's own noise floor), `cold_start`,
`blowup_100x` (injected on `peak_scale` mid-run), `stuck_cold_100x`,
`hot_half_to_cliff`, `hot_90pct_to_cliff`. Stage transitions are **not** a
scenario — `rearm_warmup` resets `peak_scale` and forgets the ceiling at every
transition, so the post-transition state *is* cold start and each surface can be
studied independently.

The hot arms are placed as a fraction of the log distance from oracle to
**cliff**, not as a fixed multiple, because the stable-but-hot band can be
narrow: on `mle` the oracle is 4.33e-3 and the cliff 7.3e-3, so "2× the oracle"
is already past it and would test catastrophe instead of the question asked.

**The divergence response is a rewind, not just a cut.** `train.py`'s
`fire_loss_spike` reloads the running checkpoint and *then* cuts the peak
(`load_model_only`, train.py:2023), aborting past `max_reloads_per_1k_steps`.
`BenchRun` therefore checkpoints on the same 50-step clock, restores parameters,
optimizer state and non-parameter game state (`extra_state`) on divergence, and
aborts on budget. This is not optional detail: an earlier version modelled the
cut alone, which left parameters non-finite after any blow-up and produced
regret figures of 10⁸ with 1985 divergences. Modelling half a mechanism gave a
confident, reversed conclusion — see F-018's method correction.

## The sensor race

`sensor_race()` runs five arms through the **same actuator** — same warmup,
bounds, ceiling, tripwire and rewind — changing only where the verdict comes
from:

| arm | verdict source | can it climb? | can it brake? |
|---|---|---|---|
| `ray` | the alpha probe | yes | yes, late |
| `plateau` | `train.py`'s ReduceLROnPlateau, transcribed | **no** | yes |
| `ramp` | none — a constant raise every period | yes | tripwire only |
| `ramp_plateau` | ramp up, plateau down | yes | yes |
| `none` | nothing | no | no |

`ramp` is the null hypothesis and not a strawman: 72–82% of real probe readings
come back at a grid edge, and at a grid edge the servo applies exactly this
constant. If `ray` cannot beat `ramp`, the probe is not paying for itself.

`ramp` is implemented by handing the real actuator a synthetic `above_range`
reading rather than writing `peak_scale` directly, so it takes the same warmup
hold, bounds and asymmetric update as every other arm.

`plateau` is transcribed from `train.py:4030-4068`, including the two details
that matter: it reads the **smoothed** value (a best-ever over raw samples
ratchets down to the luckiest noise draw and then cuts a still-improving run),
and its improvement bar is **absolute** (these channels are unbounded below).

**The oracle is checked, not trusted.** `find_oracle` refuses a bracket whose
minimum sits at an edge (the bracket was wrong) and refuses a surface where the
best rate does not beat the edges by 2× (regret against it would measure
nothing). A baseline nobody verified is worse than no baseline, because every
number downstream inherits it.

That guard has already earned its keep: `var_cond` at the default batch fails it,
because batch 1000 ≥ `n_cond` 256 means every step sees every condition, the
per-condition levels are never stale, and the surface's whole mechanism is
switched off. `SURFACES['var_cond']` therefore pins `batch_size: 32` — a
load-bearing setting, not a tuning choice.

## The three games

A quadratic bowl would make the bench worthless: `alpha*` would be constant,
batch size would be decoupled from optimization, and there would be no
multi-step instability for a one-step sensor to miss. Each game reproduces the
character of one stage.

**`mle`** — single player, regression-like.
`L = ½θᵀHθ + c·Σθ⁴ + gᵀθ`, `H` log-spaced over `cond`. The quartic makes local
curvature `H + 12c·diag(θ²)`, so it falls along the path and `alpha*` rises —
which is the argument for periodic recalibration, made mechanical.
`alpha_star_true()` is exact and verified against a brute-force line search.

**`var_cond`** — two players, cooperative, sparse levels.
One shared potential, but the per-condition level `ζ_c` only updates for
conditions the batch drew. Batch size therefore buys **condition coverage**,
not just noise reduction: staleness scales as `n_cond/batch`. The `t_c` are
drawn with wide spread because per-condition level dispersion is what breaks
pooled metrics in the real run.

**`equilibration`** — three players, no joint potential.
Policy chases a level, level responds **anti-phase**, buffer supplies the only
restoring force. The sign is the design: a symmetric `+/+` coupling makes the
fixed point a saddle and nothing is stable at any LR. Anti-phase makes it a
spiral with a finite, computable LR boundary.

Because the iteration is linear in `(θ, ζ, μ)`, `stability_lr()` is exact, and
the gap between it and `one_step_lr()` is the whole `alpha_target` question in
closed form.

## Gradient noise and cost

Per-batch noise enters as a linear term `gᵀθ` with `g ~ N(0, σ²/B)` — exact
minibatch scaling, differentiable, so autograd and `RayCalibration` see a
genuine stochastic gradient. Optimizers are real `torch.optim`; plain SGD by
default because it is the only choice for which `alpha*` has a closed form.

`SyntheticGPU` models `t(B) = t_fixed + B/sps_max`, which gives the growth
gate's acceptance bound in closed form:

```
B_max = sps_max · t_fixed · tol / (f − 1 − tol)
```

Nothing sleeps; step time is a number appended to the same deques the real loop
appends to.

**Discreteness is opt-in.** Real step time is not smooth in batch, and three
separate effects are modelled, all off by default so the closed form stays exact
when you want it:

| knob | models | effect |
|---|---|---|
| `tile` | wave quantisation — work dispatches in full waves, so a batch costs `ceil(B/tile)·tile` | throughput is **sawtooth**, not monotone |
| `recompile_s` | `torch.compile`'s per-shape recompile + CUDA graph (~30–60 s here) | charged once per size ever seen — i.e. on the **first measurement of a new rung** |
| `regimes` | cuBLAS/cuDNN kernel switches past certain sizes | a step in `sps_max` |

With any of these on, `knee_bound` **refuses** rather than returning a number
that quietly means nothing, and `expected_pin` walks the ladder against the
actual cost model. The closed form is the smooth special case of the walk, not
the other way round.

## Running it

```bash
python -m pytest bench/ -q
```

The LR-controller, ray-probe and game suites need no `train.py` import and run
in about a second. The batch-sizer suite pays ~11 s once for `train.py`
(wandb, mxtaltools, PyG), then runs in virtual time.

Local invocation needs the project venv and `PYTHONPATH`; see
`reference_local_run_recipe`.

## Known faithfulness gaps

- **No policy network, no trajectories, no buffer.** Anything whose behaviour
  depends on rollout structure — `traj_checkpoint`, SDE discretisation, the
  replay buffer's admit/purge rules — is out of scope by construction. The
  surfaces stand in for the *shape* of each stage's optimization problem, not
  for the sampler that generates it.
- **The games are low-dimensional and their curvature is diagonal.** Real
  anisotropy has structure the `mle` spectrum does not.
- **A smooth cost model is still the default.** Discreteness has to be asked for
  (`tile`, `recompile_s`, `regimes`); a test written without them is testing the
  easy case. Failing on the smooth model is conclusive; passing it is necessary,
  not sufficient.

Closed since the first version: `Modeller.__init__` used to call
`torch.cuda.set_per_process_memory_fraction` and `torch.cuda.init()`
unconditionally, so the real object could not be built without a GPU. It is now
guarded (`train.py:129`) and `real_modeller.py` builds it on CPU.
