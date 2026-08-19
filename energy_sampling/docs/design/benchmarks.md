# Canonical benchmarks

§11 of [`infrastructure_stabilization.md`](infrastructure_stabilization.md), whose
Phase 4 is gated on it: *specification before measurement, so the sweep produces a
rerunnable suite rather than a one-off.*

Argument, in the `docs/PROTOCOL.md` sense. It records why the benchmarks are cut
this way and is revised when the reasoning changes. The specifications themselves
are data in [`benchmarks/registry.yaml`](../../benchmarks/registry.yaml), loaded
and validated by `benchmarks/registry.py`; this file does not restate a field that
lives there.

**Nothing here has been run.** Every number below is either cited from an existing
measurement with its conditions, or explicitly named as unmeasured. §10 is the list
of what could not be specified without measurement.

---

## 1. What a benchmark is for, and what it is not

The goal is that **"rerun the standard A100 throughput suite" resolves to exactly
one thing**. Today it resolves to a session reconstructing a workload from a nearby
config, which is the same failure the analysis package was built to end.

A benchmark here is a *specification*: a workload, a fixed work quantity, a metric
set, a floor, and a correctness reference. It is not a framework. There is no
runner, no result store, and no scheduler in `benchmarks/`, deliberately — every
previous generation of measurement machinery in this project grew a harness and
then spent its budget on the harness. Executing a benchmark is `train.py` with the
resolved overrides applied.

Three rules are inherited from [`bench/metrics.py`](../../bench/metrics.py) and are
enforced by the validator rather than asked for in prose.

**No metric may depend on a reference rate.** The old controller topline was
`steps_to_target(run)/steps_to_target(oracle)` and it failed three separate ways —
a ceiling below the threshold, a denominator pinned at the metric's own floor, and
a reference rate 187x off on one surface family. The identical hazard exists here
and is more tempting, because "1.8x faster" is what a throughput result feels like
it should say. A reference rate becomes load-bearing in several places at once and
then cannot be changed for one of them. So the recorded quantity is always
absolute — `train_step_time`, `samples_per_sec`, `energy/ms_per_sample` — and a
comparison is made **by reading two rows**, each with its own floor. A ratio may be
printed for the reader; it may never be the thing stored or gated on.

**Catastrophes are counted, never averaged.** OOM events, batch-ceiling expiries,
non-finite gradient steps, divergence rewinds, and failures to complete the work
quantity are raw counts. An arm that is excellent on four repeats and detonates on
the fifth has a fine mean and is unusable.

**An incomplete run is not a fast run.** A benchmark that OOMed at step 12 executed
different work; its timing is not a smaller sample of the same thing.
`score_repeats` excludes it and reports the exclusion. This is the same trap
`_final_loss_or_death` exists for: a rewind restores a healthy state, so a run that
dies has an excellent-looking tail right up to the moment it dies.

---

## 2. Fixed work quantity — and the five ways it silently is not fixed

A throughput number needs a denominator that does not move. Every item below moves
it, all of them are live in `configs/mk_dev.yaml`, and none of them announces
itself. They are neutralised in `defaults.overrides` and the reasons are here.

**The batch is a moving denominator.** `grow_batch_size` (with the state-8 sizer's
occupancy ladder behind it) and the OOM shrink path change `batch_size` mid-run;
observed range 982–3000 on the dev box and 1650–7410 on the cluster. A benchmark
pins it. `grow_batch_size: false` and `max_batch_size == batch_size` are
**independent hard stops** and setting one alone pins nothing — the validator
requires both, because a battery has already been run in which all six arms sat at
1000 for the opposite reason nobody intended.

**`max_step_seconds` cuts the batch mid-benchmark.** Canonically 60 s. It is a
runaway guard, not a tuning knob, but on a slow MLIP step at production `T` it
fires, halves the batch, and the run continues under a different work quantity
while still reporting one number. Set to 0.

**`epochs` is an absolute step index, not a count.** `train.py` runs
`trange(init_step, args.epochs + 1)` with `init_step` restored from the checkpoint.
A warm-started benchmark carrying `epochs: 400` against a step-6680 resume executes
**zero steps** and produces a clean, empty, entirely plausible result — the
swallowed-diagnostic shape. `registry.epochs_for(bid, resume_step)` is the only
supported way to compute it and the validator rejects an `epochs_formula` that does
not mention `resume_step`.

**`z_calibration` corrupts the throughput denominator, not merely the timing.** Its
rollouts run *inside* the step-timing window, while `_throughput['samples']` is
charged only `attempted_batch` (`train.py:2168-2170`). So `samples_per_sec` moves at
constant batch as the sensor converges — this is the mechanism that walked
`prod0810` to batch 12226. Disabled in defaults; its cost is a separate question,
not a background rate.

**Periodic work aliases with the window.** `ray_calibration.period` is 500 steps and
each firing costs `n_sub: 8` paired sub-batches across 8 alphas; the fused
force-refresh (`controller.refresh_every: 10`) runs a full rollout of every
non-dormant branch. A 400-step window contains zero or one calibrations depending
where it starts, which shows up as run-to-run spread that is really aliasing. The
LR sensor is disabled in defaults; the refresh period is handled by requiring fused
measurement windows to be whole multiples of 10.

**A stage transition is not steady state.** `z_calibration` costs ~12x at
transitions, optimizers are rebuilt, the LR envelope restarts, and
`rebuild_prior_by_churn` fires on entry to `equilibration`. **A benchmark measures
inside one stage and never across a transition**, which means warm-starting from a
pinned, immutable checkpoint. `_running.pt` is not immutable — every run of that
config overwrites it, and a diagnostic that pointed at one silently measured two
different models under one name on the same day.

Given all of that, the fixed work quantity is:

> **`measure_steps` optimizer steps at a pinned batch and a pinned `T`, inside one
> protocol stage, after `warmup_steps` discarded steps.**

Warmup is discarded because the first steps carry allocator growth and, on a
compiling host, a 30–60 s inductor recompile per distinct batch size.

---

## 3. The noise floor

> A benchmark needs a stated noise floor or it cannot support a comparison.

**No floor in this repo has been measured for the training path.** The
`noise floor` occurrences in `bench/` refer to the *loss* floor of a synthetic
surface, which is a different quantity. Every `noise_floor.measured` in the registry
is `null`, `floor_for()` raises rather than returning a default, and
`test_no_floor_is_measured_yet` is expected to fail once the first floor lands. An
unmeasured floor is not a small floor.

**How it is measured.** `repeats` independent **process launches** of the identical
config, the run statistic being the median of the primary metric over the
measurement window, and the floor being the **relative span** `(max - min) / |median|`
across those repeats.

- **Separate launches, not re-timing inside one process.** Step-to-step scatter
  within a run is a different and much smaller quantity than launch-to-launch
  spread, which absorbs allocator state, clock and thermal state, host contention,
  and — on the cluster — a different node. A floor built from within-run scatter is
  too tight, and a too-tight floor makes noise into findings.
- **Span, not sd.** At 3–5 repeats an sd estimates nothing. A span is what was
  actually observed, it is conservative, and adding repeats can only widen it
  honestly rather than shrinking it by assumption.
- **A comparison must exceed the floor.** `exceeds_floor(a, b, floor)` tests
  `|a-b| > floor * midpoint`. It is symmetric: neither argument is a denominator.

**Two different floors, and conflating them is the trap.** *Fixed seed* isolates
machine and kernel nondeterminism and is the floor for "did this code change alter
the cost". *Varied seed* additionally absorbs the draw and is the floor for "is this
operating point better". A change tested against the fixed-seed floor when the
outcome depends on the draw will read as significant when it is not. The registry
records `seed_policy` per benchmark; every entry currently declares `fixed`, because
these are cost benchmarks.

**On MLIP, `torch.equal` is the wrong bar.** UMA is not bit-reproducible on GPU:
measured 3.6–4.4e-3 eV run-to-run on the same construction path, against 2.8–4.0e-3
eV between two *different* construction paths — so an exact assertion measures the
GPU's reduction order rather than the code. With tf32 off the same-path spread falls
to 9.2e-5 eV, a 51x tightening, so tf32 is a deliberate speed trade and not a bug.
At the lattice-energy scale that ~4e-3 eV is about **0.1 kJ/mol** against energies
of order 100. *(RTX 5080, `esen_s.pt`, 12 CSD crystals.)*

The honest test is a **control comparison**: measure the same-path spread on the
same batch inside the same test, and require the cross-path spread not to exceed it
by more than a small factor. A real construction bug is orders of magnitude out, not
a factor of four. `test_uma_gpu_real_batches.py` is built this way, and
`verify_fairchem_batch_equivalence` / `verify_mace_atomicdata_equivalence` are the
preprocessing-side harnesses — those *are* exact, because they compare two
constructions of the same tensors rather than two forward passes.

The pattern to copy for a new tolerance is `test_batch_invariance.py`: an absolute
bar set **100x above the measured two-call noise floor and 5x below the faintest
observed signal**, on an adversarial rather than a random draw. Its first version
used a scale-relative tolerance that was larger than the effect on 397 of 400
crystals, and it passed while blind.

---

## 4. The profiles, and why none stands in for another

`fwd`, `bwd`, `replay` are **samplers**, not objectives; `fused` is not a fourth
sampler but a step that runs the others and combines their losses.

| profile | sampler | energy calls per step | notes |
|---|---|---|---|
| `bwd` / `dataset` | atomic dataset (`train_prior` MLE) | **none** | the energy keys are ABSENT, not zero |
| `bwd` / `prior` | churned prior buffer | **unmeasured** — read `energy/calls` | admission and churn may score |
| `replay` | replay buffer, stored trajectories | none for the draw | replays exactly; cheapest branch |
| `fwd` | on-policy rollout | one, full batch | the expensive branch |
| `fused` | all three, every step | ≈ sum of the above | **independent of the fracs** |

**In `fused` mode the fracs are loss weights, not throughput shares.** Every active
branch runs every step; step cost is roughly the sum and does not move when a frac
moves. So a fused profile cannot be predicted by weighting the single-branch
timings by the fracs, and "raise `bwd_frac` to spend more time on bwd" is wrong.
This is the single reason `fused` must be benchmarked in its own right.

**Isolating `fwd` or `replay` needs care, because neither is a legal stage.**
`protocol.TRAIN_MODES` is `('bwd', 'fused')`. A solo-fwd or solo-replay profile is a
fused stage with the other branches driven below `deactivate_threshold`, and
`controller.refresh_every` pushed beyond the window so the force-refresh rollout —
which runs a *full* rollout of a dormant branch purely to keep its stats fresh —
does not contaminate the measurement.

**A solo-replay benchmark can silently become a bwd benchmark.** When the replay
buffer is unavailable, `train.py:2157` folds `replay_frac` into `bwd` without
comment. So the replay benchmark's liveness check is not decoration: it must show
that `Replay Frac` was not folded and that `replay/*` metrics actually moved. This
is R2 — confirm the thing ever fired — and an inert mechanism is the most common
explanation for a null result here.

**Conditional is not unconditional plus a constant.** It changes the model (a
conditioning embedding and, on the molecule route, a frozen Mo3ENet conditioner),
the batch composition (condition sampling, condition-blocked draws), and the eval
path (`eval_test` on held-out conditions). Both a toy-conditional and a
molecule-conditional benchmark exist so the conditioning overhead can be separated
from the energy cost.

**Toy is not a cheap MLIP.** On the toy and ELJ routes the policy rollout dominates
and **batch is the occupancy lever**. On the MLIP route the energy call dominates —
UMA measured ~5.5 ms/sample against ELJ's ~0.3 at eval, an 18x gap — and batch is
*not* the occupancy lever. The retired `gpu_util_floor` grew batch 100 → 741 on a
UMA arm for occupancy that never improved and cost 58% of throughput; the comment
recording that measurement is about the MLIP route and does not transfer to an MLP
policy over an analytic energy. Both statements are true and neither generalises,
which is exactly why they are separate benchmarks.

---

## 5. Hardware classes

| | local | A100 |
|---|---|---|
| GPU | RTX 5080 Laptop, 16303 MiB | A100, cluster |
| OS | Windows 11 | Linux |
| `compile_policy: auto` | **resolves off** (no inductor on native Windows) | resolves on |
| scheduler | none | cancels on low utilisation, 7200 s window |
| co-tenancy | one run at a time; `cuda_memory_fraction: 0.9` is 14673 of 16303 MiB | — |

**Absolute numbers never transfer between classes.** Only same-class comparisons are
valid, and the registry's `comparison.rule` says so per benchmark.

**LOCAL-adequate:** the analytic toy (`latent_gaussian`, dataset ships in-repo at
`mxtaltools/mini_datasets/mini_new_csd.pt`), the conditional toy, and every ELJ
profile including the branch ladder, the conditional molecule route, and eval cost.
These are the development set and they answer step-cost questions completely.

**A100-REQUIRED:** anything whose answer is a property of the cluster.

- `gpu/util_policy` and every occupancy-versus-batch question. The local box has no
  scheduler, so it cannot answer what gets a job cancelled.
- Absolute throughput at production batch and production `T`. 16 GB caps the batch
  well below the cluster's.
- `compile_policy` at all — it is structurally off locally, so its benefit and its
  per-batch-size recompile stall are unmeasurable here.

**LOCAL-partial (the MLIP routes):** a local UMA or MACE run gives valid step-cost
*deltas* at a pinned small batch — this is what `configs/umaperf0812/` did, measuring
step time 17.68 → 9.75 s and `energy/ms_per_sample` 8.93 → 4.51 on a controlled A/B —
and invalid absolutes. The stated caveats there are the right ones: local RTX 5080
not A100, T=10 not 60, so the local energy share (0.82) is an upper bound on the
cluster's.

**One cross-class check that must be run on both.**
`batched_pbc_neighbour_list`'s fast path returns `None` and silently falls back to
the O(Σn²·K) all-pairs kernel when `torch_cluster` is not importable — roughly 92x
more work, presenting as a slow GPU rather than a missing dependency. Every MLIP
benchmark's liveness list requires the fast path to be shown taken, on whichever
class it ran.

---

## 6. Metrics

Every literal is already logged; none of this needs new instrumentation. The
registry's `metrics` block is the catalogue and the validator refuses a benchmark
that names anything outside it.

**Cost:** `train_step_time`, `samples_per_sec`, `step_time_max10`, `Batch Size`.
**Energy:** `energy/ms_per_sample`, `energy/calls`, `energy/seconds`,
`energy/seconds_in_step`, `energy/frac_of_step`, `energy/frac_outside_step`.
**Occupancy:** `gpu/util_recent` (900 s), `gpu/util_policy` (7200 s), `vram/*`.
**Eval:** `eval_step_time`, `eval_sampling_time`, `eval_figs_time`,
`inter_eval_time`, `initialization_time`.
**Catastrophe counts:** `batch/oom_events`, `batch/ceiling_expiries`,
`batch/oom_ceiling`, `gradnorm/nonfinite_steps`.

**MACE additionally carries a four-phase split** —
`energy/mace_{build,collate,xfer,forward}_s` plus `energy/mace_host_frac` and
`energy/mace_forward_frac` — which is precisely the preprocessing / neighbour-list /
forward / host↔device breakdown Phase 5.0 asks for. **UMA has no equivalent**, so the
UMA benchmark declares `energy/mace_host_frac` unusable and cannot report the split
without new instrumentation. That asymmetry is a gap in the instrument, not in the
benchmark; see §10.

`energy/frac_of_step` is the load-bearing one on the MLIP route: paired with
utilisation it separates "the MLIP call is expensive" from "the MLIP call is idle
waiting on the host", which no other metric here can do — this cluster logs no CPU
columns.

**Four metrics are not readable everywhere, and the registry records why per
benchmark.** A metric has three states — live, absent, and not-meaningful-on-this-
route — and collapsing the third into "absent" or rendering it as zero is worse than
crashing.

- `energy/*` is **absent** on `bwd`/`dataset`, because `drain_energy_timing` returns
  `{}` when nothing was timed. That absence is the assertion, not a gap.
- `gpu/util_policy` is meaningless on any run shorter than 7200 s; the window never
  fills and a partial window still prints a number. Sampling is every 60 s, so a
  900 s window holds ~15 samples. The validator refuses `gpu/util_policy` as a
  primary metric unless the benchmark declares `min_wallclock_s >= 7200`.
- `energy/ms_per_sample` pools **every** energy call in the window — training, eval
  sampling, anchor screening, prior churn. It is a clean per-sample cost only when
  eval is off, which is why the eval-cost benchmark declares it unusable and the
  throughput benchmarks disable eval.
- `samples_per_sec` counts training samples only, so it is not a throughput measure
  for the eval path.

---

## 7. Correctness reference

A cost benchmark still has to show it computed the right thing, or it is timing a
faster wrong answer.

| reference | where it applies | bar |
|---|---|---|
| `closed_form` | `latent_gaussian` | **exact** — `log Z` analytically known |
| `tier_c_repeat_spread` | ELJ fused | the same config run twice is the null distribution |
| `control_comparison` | UMA, MACE | same-path spread measured in the same test; cross-path must not exceed it by more than a small factor |
| `paired_control` | branch ladder, conditional | the neighbouring profile at identical batch and `T` |

The analytic toy is the sharpest instrument available and it costs seconds:
`log Z = (n_live/2) log(2πT) + n_live log w`, plus `n_dead · log(2 + √(π/k))` when
dead rows are left live — the box wall is soft, so the fictitious volume per row is
3.77 at `k=1`, not 2. It is the only benchmark permitted to declare
`exactness: exact`, and the validator enforces that: on any GPU or MLIP path the
same run disagrees with itself, so an exact bar there measures reduction order.

For config consolidation specifically, tiers A and B — parsed config and
deterministic pre-runtime state — are exact and cheap and are `config_snapshot.py`'s
job, not a benchmark's. Only tier C needs a run, and it should be run on the toy
first, where it collapses to an exact test.

---

## 8. Comparison criteria

A result is admissible when all of the following hold. They are checkable, and most
of them are checked.

1. **Same hardware class**, and the class is recorded.
2. **The work quantity was actually executed** — the run completed
   `warmup_steps + measure_steps` at the declared batch, with `Batch Size` flat.
3. **Liveness passed** — every branch the benchmark claims to exercise left moving
   metrics; every branch it claims to skip left frozen ones.
4. **The floor is measured**, from at least the declared number of repeat launches,
   covering every primary metric.
5. **The difference exceeds the floor** on the metric being claimed, tested
   symmetrically with no denominator.
6. **Catastrophe counts are reported alongside**, never folded in. A faster arm with
   OOM events is not a faster arm.
7. **Incomplete repeats are excluded and the exclusion is stated.**
8. **The claim carries its scope line** — T, problem, stage, steps, seeds — per
   `PROTOCOL.md`. T dominates outcomes here and a cost claim without it is
   unreadable.

Grades follow `PROTOCOL.md`: a single benchmark run is `OBSERVED` and does not
generalise. `REPLICATED` needs the effect to exceed the *measured* floor. A
throughput claim promoted to a flat statement needs a mechanism, and "it was faster
on one run" is not one.

---

## 9. Running one

There is no runner by design. A benchmark resolves to a config and a step budget:

```bash
python -m benchmarks.registry
```

lists the benchmarks, the suites, and — currently — every benchmark whose floor is
unmeasured. `registry.resolved_overrides(bid)` gives the deep-merged override dict
and `registry.epochs_for(bid, resume_step)` gives the `epochs` value. Materialising
those into a config file under `configs/` is `configs/generate.py`'s job (Phase 2),
which is why the registry stores overrides as data rather than shipping YAML files
of its own.

`configs/umaperf0812/` is the worked precedent for the surrounding discipline:
`checkpoint_read_only: true` so an unattended benchmark cannot clobber a checkpoint,
arms run sequentially because two CUDA consumers on 16 GB is a known collision,
`gpu_guard` refuses to start on a busy card, and `timeout` caps each arm.
`bench/calibrate_noise.py` is the precedent for driving the real loop rather than
reimplementing it — including its scar, that suppression by an invented attribute
(`save_checkpoints`, which does not exist anywhere in the codebase) wrote real
checkpoints for as long as the docstring promised otherwise.

---

## 10. What could not be specified without measurement

Everything below is a hole in this specification that only a run can close. It is
listed rather than filled with a plausible number.

**Every noise floor.** All twelve are `null`. Until each is measured, no benchmark
in this registry supports a comparison, and `floor_for` raises rather than letting
one happen quietly. This is the single largest gap and it gates Phase 4's
acceptance.

**Whether the declared `measure_steps` are enough.** 400 steps (200 on MLIP) is
chosen from the precedent of `umaperf0812` and `calibrate_noise`, not from a
measurement. The right window is the one where the median stops moving as the window
grows, and that is a property of the floor — so the window length cannot be fixed
before the floor is measured. Expect to revise both together.

**Energy calls per step on `bwd`/`prior` and on `fused`.** Only one is documented:
`bwd`/`dataset` never calls the energy. Whether the prior draw, admission and churn
trigger energy calls, and how many a fused step makes in total, is read directly off
`energy/calls` — the instrument exists, the reading does not.

**Every A100 absolute.** Step time, samples/sec, the batch rungs actually
reachable at 80 GB, whether `compile_policy: auto` engages, and what its recompile
stall costs per rung. The rung ladder in `a100-batch-scaling-elj` is copied from
observed cluster batch sizes (1650–7410) and is a starting grid, not a measured
range.

**Whether the neighbour-list fast path is taken on the A100.** It fails silently and
has never been checked there. Also unmeasured: the module docstring's claim that
matscipy/ASE is 64.9% of the AtomicData build at 128 graphs — that is the number the
whole neighbour-list approach rests on and Phase 5.1 requires it re-measured.

**MACE, entirely.** There is no local MACE cost measurement on record. The UMA
numbers do not transfer: `energy/mace_host_frac` exists precisely to establish
whether MACE splits host/device the way UMA does, and that is an open question.

**UMA's phase split — CORRECTED 2026-08-18: it EXISTS.** This section previously
said there was no UMA counterpart to `drain_mace_phase_timing`, and that was
wrong. `uma_utils.drain_uma_phase_timing` reports guard / build / forward,
`uma_host_frac` and `uma_forward_frac`, and it is live on every UMA run —
measured `host_frac` **1.7 %** against MACE's **68 %**, i.e. the UMA host side is
already substantially solved by the AtomicData vectorisation
(`uma_flag_vectorised: 1`).

It is also more careful than the MACE one where it matters. `graph` is **nested
inside `forward`**: mxtaltools hands fairchem an empty `edge_index` and
`crystal_inference_settings` sets `external_graph_gen=False`, so the model runs
`otf_graph` and builds the neighbour list ITSELF inside the forward. A plain
build/forward split therefore charges graph construction to the forward and
concludes the forward dominates — true, uninformative, and exactly the reading
that would retire the neighbour-list question for the wrong reason.

**What is real is that the graph keys are OFF BY DEFAULT.** They appear only
under `MXT_UMA_GRAPH_TIMER=1`, because the timer adds a CUDA synchronise per
graph build and a measurement cost has no business on every production step. So
UMA's forward reads 98.3–98.6 % of the call and **nothing decomposes it** until
that flag is set — which is a flag on a profiling arm, not missing
instrumentation. The same pattern governs the MACE construction paths
(`MXT_BATCHED_MACE_NEIGHBOURS`, `MXT_GPU_MACE_BATCH`), both off by default.

**The utilisation proxy, and its case.** Phase 4 wants `gpu/util_policy` adopted
under case 1 (documented statistic), 2 (proxy shown to agree with cluster-visible
evidence) or 3 (most conservative reading plus a stated margin). Nothing here
establishes which. The subsidiary question — how `gpu/util_policy`,
`gpu/util_recent` and `nvidia-smi` sampling relate — is measurable on any A100 job
and has not been measured.

**Eval's share of wall clock.** One measurement exists: 1.8% on `prod0810_mipcas_elj`
at `eval_period 500`, `eval_num_samples 10000`, T=60. That is one run on one route
and is `OBSERVED`, so `elj-eval-cost` exists to establish it rather than assume it.

**The conditioning overhead.** No measurement separates the conditioner's forward
cost from the batch-composition cost from the held-out eval cost. The paired
conditional/unconditional benchmarks are designed to, and have not run.

**Whether repeat spread on the cluster is dominated by the node.** A floor measured
across launches that happened to land on one node understates a floor across launches
that did not. If the cluster does not pin nodes, the floor has to be measured across
several, and that changes the required repeat count.

---

## 11. Acceptance

Phase 4 closes when, for each benchmark in `a100-throughput` and `local-dev`:

- [ ] the floor is measured from repeat launches and recorded in `noise_floor.measured`
- [ ] a baseline is recorded as a graded finding in `findings.md` with its scope line
- [ ] the liveness assertions were checked and passed
- [ ] catastrophe counts are reported alongside every baseline

and separately:

- [ ] a named utilisation proxy with its case (1/2/3) **written down**, because a
      proxy adopted under case 3 and later remembered as case 1 is how a margin
      quietly becomes a law.
