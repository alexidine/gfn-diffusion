# Local synthesis shakeout — what was verified, what failed, what was fixed

*2026-08-19, dev box (RTX 5080 laptop, torch 2.8.0+cu128, `torch.compile` OFF on
Windows). Arms spawned by `make.py` in this directory; the cluster's go/no-go
list is `RUNSHEET.md` beside it.*

**Grades follow `docs/EPISTEMIC_PROTOCOL.md`. Every run below is n=1 and
therefore `OBSERVED`: it says what happened once on one box.** No performance
verdict appears here — the rollout is dispatch-bound locally and every floor in
`benchmarks/registry.yaml` is an A100 number. Where a question is a cluster
question it is written into `RUNSHEET.md`, not answered here.

---

## 1. The unit tier

**Full suite, minus four bench statistical files** (`test_surface_fitness`,
`test_tracking`, `test_equilibration`, `test_calibrate_noise` — seeded
controller batteries, not correctness gates):
**1174 passed, 3 min 09 s.** The `-m fast` lane is 937 tests in 64 s. Re-run
green after the state-9 unit change (§6.8), which needed three historical corpus
configs migrated — `configs/bsz/b{1000,500}.yaml` and
`configs/shakeout_aug16/qm9_cond.yaml` were stamped state 8 and stopped loading.
`test_the_loadable_corpus_is_measured_not_assumed` is what caught that, which is
exactly the job its docstring claims; `config_state migrate --write` fixed all
three as a pure stamp bump, since none carries the reinterpreted key.

The suite does not run without `PYTHONPATH` carrying both repositories
(`mxtaltools;gfn_diffusion`) — without it nine bench files fail to COLLECT on
`ModuleNotFoundError: energy_sampling`, which stops pytest before any test runs.
That is the documented launch recipe, recorded here because the failure looks
like a broken suite rather than a wrong invocation.

**The slowest survivors, measured** (so the exclusion list stops being folklore):
`test_latent_gaussian::test_analytic_log_z` 40.0 s,
`test_dead_latent_rows_deep::test_log_z_unbiased` 25.7 s,
`test_latent_gaussian::test_bounding_coeff_dial` 11.6 s, then a shelf at 5–7 s.
Nothing else is close, so the four excluded files plus these three are where the
suite's wall clock lives.

## 2. The two MLIP equivalence gates, with their numbers read

Both are in the sibling repository (`mxtaltools/tests/`), outside this project's
pytest scope, so a run of the suite above does NOT cover them.

**MACE batched neighbour lists — `tests/test_pbc_neighbours.py`: 64 passed.**

**THE GPU GATES ON BOTH ROUTES SKIP SILENTLY UNLESS A MODEL IS NAMED, and the
tests that skip are exactly the ones carrying the correctness claims.** This is
the failure class F-047/handoff §3.1 keep hitting from the other direction — not
a switch that reports success without running, but a *gate* that reports success
without running:

| invoked as | reports | what actually ran |
|---|---|---|
| `pytest tests/test_uma_external_graph.py` | 5 passed, 4 skipped | the convention tests; **no energy, no gradient, no wrap-invariance** |
| ...with `UMA_CHECKPOINT` set | **9 passed** | all of it |
| `pytest tests/test_mace_gpu_real_batches.py` | 11 skipped | **nothing** |
| ...with `MACE_CHECKPOINT` set | **11 passed** | energy + gradient on real batches |

Both models are on this box (`D:\crystal_datasets\esen_s.pt`,
`acr_112025_mh1_stagetwo.model`); nothing but the environment variable stood
between "11 skipped" and "11 passed". A run of the project suite in §1 does not
cover any of this — these live in the sibling repository.

*(A third skip cause is separate: the MACE file first refused with "GPU
pre-flight refused: no GPU" while 12 GB were free and another arm held the card
— the known device-blind guard. `GFN_GPU_GUARD=0` clears it.)*

**The same guard was also seen doing its job properly, which is worth recording
beside the false refusal.** When a launcher bug started two runs on top of a
live one, the pre-flight refused both with the arithmetic shown — projected need
14672 MiB against 9574 MiB free, the co-tenant named by pid and config path, and
`GPU preflight: DO NOT LAUNCH`. So the guard is not simply unreliable: it
reports a real block accurately and errs only in the "is there a GPU at all"
direction when another process holds memory.

With `UMA_CHECKPOINT` supplied, all nine UMA tests execute and reproduce F-047
on this box:

| | measured here | F-047 |
|---|---:|---:|
| edges fairchem misses, n=4 | 102 / 26688 (0.38%) | 102 / 26688 |
| edges fairchem misses, n=9 | 318 / 51592 (0.62%) | 318 / 51592 |
| wrap-invariance, **internal** path | **0.2437 eV** | 0.243 |
| wrap-invariance, **external** path | 0.00415 eV | 0.0066 |
| tf32 nondeterminism control | 0.00360 eV | 0.0053 |

The discriminator is intact: wrapping atoms by lattice vectors is a symmetry of
the crystal, the external path moves within a factor of the nondeterminism
control, and the internal path moves 68× the control on a 1718 eV scale.

## 3. The fast paths EXECUTE — read off a real run, not inferred

`synth_aug19_a2_mace_fast` (MACE, acridine sg14/zp1) and
`synth_aug19_a3_uma_ext` (UMA, mipcas sg2/zp1):

| metric | read | meaning |
|---|---:|---|
| `energy/mace_flag_batched_nl` | **1.0** | batched NL executed on every call |
| `energy/mace_flag_gpu_batch` | **1.0** | device-built dict executed |
| `energy/nl_fastpath_frac` | **1.0** | `torch_cluster` radius search taken |
| `energy/nl_radius_calls` / `nl_allpairs_calls` | **10 / 0** | zero fallbacks |
| `energy/mace_host_frac` | **0.203** | against 0.676 pre-fix (handoff §3.1) |
| `energy/uma_flag_external_graph` | **1.0** | F-047's default executed |
| `energy/uma_ext_graph_s` vs `uma_forward_s` | 0.0218 / 0.1429 | the build is ~15% of the forward it feeds |
| `energy/uma_host_frac` | 0.044 | UMA's host side stays solved |

**One reading is worth keeping because it looks like a defect and is not.** The
MACE flags read **0.909 (10 of 11 calls) on the first report** and 1.0 after.
The eleventh call is the gas-phase leg, which runs `pbc=False`, and the
`gpu_batch` branch requires `pbc` by construction. So a fraction below 1.0 does
not by itself mean a missed fast path — it means the window contained a
non-periodic call. On the cluster, read the flags on a window of pure training
steps, or expect this exact shortfall.

## 4. What those two arms did NOT measure, and how that was caught

Both arms reported **`energy/frac_of_step` = 0.0** with 5–10 MLIP calls for the
whole run. The cause is structural, not a dead sensor: spawned fresh on the full
`unconditional_tb` protocol, a short arm spends its entire budget in
`train_prior` — a bwd/dataset MLE stage that makes **no energy call inside a
training step**. Every MLIP call they counted came from the init prior
re-analysis and from eval.

So the executed-path claims in §3 stand (the calls happened, and the flags report
the branch taken), while nothing in those arms measured the MLIP where the
handoff asks about it: inside the fused training step. Re-spawned on a **single
terminal `equilibration` stage** — which is also the F-046-safe shape for a fresh
MLIP arm, since the replay buffer then fills at the run's own T.

*Run lineage, since the wandb runs outlive the configs: the MLIP arms went
`a2_mace_fast` / `a3_uma_ext` (train_prior, the arms §3's flags come from) →
`_eq` (terminal equilibration; MACE OOM'd at batch 2, UMA not launched) →
`_eq2` (batch 1/2 with `traj_checkpoint`). Only the `_eq2` pair is still on
disk — `make.py` generates exactly the six arms in `INDEX.tsv`, and superseded
YAMLs were deleted rather than left to disagree with their generator.*

**The UMA re-run then produced the numbers the first pair could not**
(`a3_uma_ext_eq2`, batch 2, `traj_checkpoint: true`, 12 reports, 91–146 UMA calls
each — real training traffic rather than init and eval):

| metric | read | was, on the train_prior arm |
|---|---:|---:|
| `energy/uma_flag_external_graph` | **1.0** on all 12 reports | 1.0, from 5 calls |
| `energy/frac_of_step` | **0.194** (range 0.140–0.220) | **0.0** |
| `energy/uma_ext_graph_s` / `uma_forward_s` | 0.560 / 3.266 = **17%** | 0.022 / 0.143 |
| `energy/uma_host_frac` | 0.067 | 0.044 |

So on this box the external graph build costs about a sixth of the forward it
feeds. **That is not comparable to the A100's measured 26.8% internal-graph share
(handoff §3.2) and is not a speed claim** — different builder, different
hardware, no compile. It is here because it is the ratio the cluster arm should
be read against, and it is now known to be logged and non-trivial.

**MACE, by contrast, hit a hard local ceiling.** In the fused stage it OOM'd at
**batch 2** (4.58 GiB short of what was free), then at **batch 1 with
`traj_checkpoint: true`** as well (4.79 GiB short), each time ending in
`RuntimeError: Cascading OOM Failure` — the shrink path reaches batch 1 and gives
up. The bwd/MLE arm could never have shown this: it makes no in-step energy call.
**MACE fused training does not fit on a 16 GB laptop card at any batch size**, so
its in-step cost is a cluster measurement and nothing local will substitute. This
says nothing about the A100's ceiling.

## 5. The batch sizer's input on this box

`pynvml` is **not installed in the project venv**, so `torch.cuda.utilization()`
raises and the sensor falls through to `gpu_guard`'s `nvidia-smi` path. It works:
`a1` logged `gpu/util_recent` = `gpu/util_policy` = **33.5%** throughout. Both
windows read identically because the run (~6 min) is shorter than either.

**That sizes the calibration, and locally it is the sample count that binds, not
the dwell.** A rung needs a 50-step dwell AND ≥3 raw occupancy samples at the
60 s sample period, so ~180 s per rung — about 257 steps at `a1`'s measured
0.70 s/step at batch 1000. The first `a4` draft budgeted 400 steps, which buys
barely one rung; it was re-sized to 900 before launch on that measurement.

**The walk ran end to end and reached a conclusion** (`a4`, warm-started ELJ,
target 60%, `max_batch_size` 8000):

| step | batch | phase | per-rung raw reading | trailing window |
|---:|---:|---|---:|---:|
| 440 | 1000 | calibrating | **23.7%** (the base rung) | — (not yet 5 samples) |
| 550 | 1600 | calibrating | | 16.6% |
| 740 | 2560 | calibrating | | 23.4% |
| 880 | 3560 | calibrating | **71.0%** | 27.0% |
| 1030 | **3560** | **hold / `target_met`** | | 37.2% |

Every rung step is the shipped capped-geometric one (600, then 960, then the
1000 cap), the walk only ever ascends, and it stopped at the **smallest** rung
that cleared the target rather than continuing to `max_batch_size` — which is
the S1 selection rule doing exactly its job. No OOM ceiling was ever set.

**Two things in that table deserve care, and neither is a defect.**

*First, occupancy rose steeply with batch here — 23.7% → 71.0% over a 3.56×
batch.* F-045 measured the opposite on the A100 ELJ route (38 → 48% over 7.4×)
and concluded batch was not an occupancy actuator there. These are different
machines, different batch ranges and a different bottleneck mix, so this does not
overturn F-045 — but it does mean **the local box cannot be used to predict which
way that curve goes on the cluster**, in either direction.

*Second, the per-rung reading and the trailing window disagree by ~34 points at
the same batch* (71.0% against 27.0%). That gap is mostly the window doing what
the design says it does wrong: a 900 s trailing mean at step 880 still contains
the 1000/1600/2560 rungs, so it is an average over the climb, not a reading of
the rung. This is precisely why the calibration uses raw per-rung samples and why
`phase6_batch_sizer.md` §0.2 refuses the windowed mean as a control input. **But
it also means the local run cannot confirm the 71% is right** — both numbers come
from the same in-process sampler that F-045 refused, and the out-of-process
stream is the only arbiter. `RUNSHEET.md` §4 keeps that as the cluster check.

**The S2 audit is unproven, not passed.** It arms one full policy window (7200 s)
after a held growth; `a4` held at ~step 1030 and ended at 1330, minutes later. The
audit never came due, so nothing was falsified and nothing was confirmed.

**TWO RUNS OF THE SAME ROUTE DISAGREED ABOUT THE SAME RUNG, and this is the
result most worth carrying to the cluster.** The integration run
(`synth_real_aug19_uncond`, same box, same archive, same route, same effective
60% target) climbed *through* 3560 — the rung `a4` had just held at 71% — and
kept going to 4560:

| rung | `a4` per-rung reading | integration run |
|---:|---:|---|
| 1000 | 23.7% | (base) |
| 2560 | — | did not clear |
| **3560** | **71.0% → HELD, `target_met`** | **did not clear → climbed on** |
| 4560 | — | still calibrating at the budget's end |

So the decision variable moved by more than 11 points on the same rung, and the
two runs selected different batches. **A named confound first, because it may be
the whole effect:** the integration run overlapped with a full pytest suite on
the same machine, and the occupancy sensor cannot tell our training process's
work from anyone else's. That is not a defect in the sizer — it is the sensor
measuring the card, correctly.

But it does establish the shape of the risk, and it is a shape the design
already worries about: **the selection is made from ≥3 samples
(`_BS_MIN_UTIL_SAMPLES`) of a quantity that is noisy and not exclusively ours.**
Three samples is a small basis for a decision that then holds for a whole stage.
On a shared cluster node the analogous contaminant is a co-tenant, which is
exactly what the sidecar recipe warns about (`phase6_batch_sizer.md` §0.15: a
reading "must filter to our GPU index, or it imports a neighbour's workload").
S2 is the mechanism meant to catch a selection made on a bad reading, and S2 is
the thing no local run is long enough to exercise. `RUNSHEET.md` §4 asks for
both: the reproducibility of the selection, and one run long enough to audit it.

**And the ladder is longer than the design's prose suggests — this is the
operational consequence worth carrying to the cluster.** Rung count from the
shipped arithmetic (base 1000, factor 1.6, cap 1000):

| `max_batch_size` | rungs | floor from the sample period alone |
|---:|---:|---|
| 8000 | 9 | ≥ 27 min per stage |
| **20000** (now canonical) | **21** | **≥ 63 min per stage** |

The sample period binds while a step is faster than 3.6 s; above that the 50-step
dwell binds instead, and at prod0810's measured 181 s/step one rung is ~2.5 h, so
21 rungs is ~53 hours — longer than any job. **On MLIP routes the ladder cannot
finish**, and such a run ends in `phase: calibrating` with the batch parked on
whatever rung it reached rather than on a selection. `RUNSHEET.md` §4a carries
the per-route consequence.

## 5b. The stage transition, and what it wrote

`a5` (latent_gaussian from step 0, the only arm permitted to write checkpoints)
crossed `train_prior → equilibration` cleanly and exercised the whole handoff:

- the exit fired on published gates — `gates/mle_flat`, `bwd/tbc` and
  `eval/wass_debiased` all carry `protocol/exit_age_*` and
  `protocol/exit_streak_*` series, so the gates were *published and consumed*
  rather than merely declared;
- optimizers rebuilt and the LR re-warmed over 1000 train steps;
- `grad_clip_guard` refreshed across the boundary while **holding its outgoing
  bar** (`[bwd=176.3]`) rather than going unclipped during recalibration;
- the stage engaged at its configured allocation (fwd 0.05 / bwd 0.45 /
  replay 0.50, fused, `bwd_sampling prior`);
- **both `on_exit` actions wrote what they promise**: `_phase1_exit.pt` with its
  `_phase1_exit_buffers.pt` sidecar, and `_prior.pt`.

`a1` crossed the same boundary from a warm start, so the transition is covered
from both entry directions. No OOM occurred at either.

## 5c. The conditional route

`a6` (QM9-conditional ELJ, fresh, 300 steps) loaded and ran clean on the route
the problem block selects, with the F-042 Z trio travelling with
`protocol: conditional_vargrad` rather than being hand-set. What it verifies is
**wiring, not quality** — 300 steps from scratch on a conditional manifold is
nothing, and the numbers say so (`eval_test/cond_tb_err` 819,
`eval_test/ess_frac` 0.002).

What is worth recording is that the held-out machinery is alive: **33 populated
`eval_test/*` keys**, 27 `eval_fwd/*`, 35 conditional/r2-family keys, and
`bwd/condition_log_z_visited_frac` = 1.0 (every condition visited). Held-out
eval is the thing `reading_runs.md` R17 says to read FIRST on this route, and it
is absent-by-default whenever `test_molecules_path` is null — so confirming it
populates is the check that makes a later conditional battery readable.

## 5d. The two integration runs (`configs/synth_real_aug19/`)

Two ~20 minute runs overriding **nothing** but run identity, resume point and
budget — the unconditional arm reports **2 deviations from canonical**, which is
the point of it. Both warm-started from phase-1 exits so the budget is spent in
the terminal stage rather than re-running a solved MLE.

**Unconditional** (resumed at step 430, 890 steps in `equilibration`): losses
descending on all three branches (`bwd/tb_err_worst` 12.07 after −21.3,
`fwd` 21.09, `replay` 38.7), `bwd/log_Z_learned` 23.34 and rising, both coverage
terms falling, occupancy 44% and climbing as the ladder grew the batch. Step time
carries a clean ~500-step oscillation, which is the figure cadence, not noise.

**Conditional** (resumed at step 18450, 600 steps): the route the problem block
selects, with the F-042 Z trio travelling with it. Held-out eval populated (33
`eval_test/*` keys, every condition visited). Against the *fresh* 300-step `a6`
arm the warm start is visibly a different regime, which is the sanity check that
the resume did what it claims:

| | fresh (`a6`, 300 steps) | warm (18450 + 600) |
|---|---:|---:|
| `eval_test/cond_tb_err` | 819 | **94.0** |
| `eval_test/Reasonable Sample Fraction` | 0.044 | **0.238** |
| `eval_test/Cond Reasonable Failing Frac` | 0.99 | **0.87** |

No quality claim is made from either: 600 local steps on a conditional manifold
resolves nothing (`project_conditional_improvement_stops_not_wobbles`). What is
established is that the route loads, resumes full state, trains, and reports its
held-out metrics.

## 6. Defects found, and what was done about each

**Fixed (in scope):**

1. **`analysis/cli.py` did not parse at all** — a literal newline inside a string
   literal at line 170, so `python -m analysis` died with `SyntaxError` on
   import. The whole Tier 0–3 analysis package was unusable. Fixed to `'\n'`.
2. **The canonical config's occupancy ladder was armed but inert**, which failed
   18 tests across five files and is what started this session's unit tier.
   `batch_util_target: 0.6` sat with `grow_batch_size: false` and
   `max_batch_size: 1000`; `config_invariants.util_target_actuable` reported two
   ERRORs — the ladder never runs, and it has one rung. (It did NOT report the
   third problem, the value's units; see 9 below.) **Owner-authorised** and
   corrected to `batch_util_target: 60`, `grow_batch_size: true`,
   `max_batch_size: 20000` — the last of these putting F-045's missing ELJ
   15–25k rung inside the domain for the first time.
3. **Eight test fixtures assumed the disarmed defaults** and broke when the
   canonical config armed the ladder. Each was pinned to the regime it is
   *about* rather than relaxed: `bench/batch_arms.Null/Fixed/Ship` now set
   `batch_util_target: 0` explicitly (their docstrings define them as the
   no-target controller), the prod0810 fixture in `test_oom_ceiling_expiry`
   likewise, and two `test_batch_traps` cells pin `batch_growth_factor: 1.65`
   because their power checks are properties of that rung geometry and the
   shipping factor moved to 1.6.
4. **`bench/old/harness.py` called a method that no longer exists** —
   `increment_batch_size`, retired with the state-8 sizer. It had been masked by
   `grow_batch_size` defaulting false; arming the canonical config made every
   LR-controller cell reach it. Now calls `select_batch_size` behind the same
   `need_batch_sizer` guard train.py uses.
5. **`test_ray_probe_gate.py` tested a contract the controller had reversed.**
   Warmup is no longer a calibration refusal (`calibration_refusal` returns None
   unconditionally; the ramp is watched and only ACTUATION is withheld, reported
   as `warmup_ramp`). The file asserted the pre-reversal behaviour throughout.
   Rewritten to pin what is true now, including a tripwire asserting the
   predicate currently refuses nothing, and keeping the `refuse` machinery's
   contract so a future refusable case slots back in without moving the applied
   path.
6. **`test_replay_gating.py`'s `SimpleNamespace` stub** lacked `lr_controller`
   and `_ray_askers`, which the real `_ray_probe_armed`/`_check_ray_wiring` now
   read.

**Flagged, NOT fixed — this one is a design question:**

7. **With `grow_batch_size: false`, an OOM cut is permanent for the run.**
   `train.py` calls `select_batch_size` only under that flag, and the
   restore-the-base rule *and* the OOM-ceiling expiry both live inside it.
   Measured on the bench fake driving the real methods:

   | `grow_batch_size` | base | after OOM cut | after 4000 more steps | ceiling |
   |---|---:|---:|---:|---:|
   | `true` | 1000 | 625 | **1000** (restored) | expired |
   | `false` | 1000 | 625 | **625** | **1000, latched forever** |

   That is the prod0810 failure the base-restore was written against — "a stage
   judged on occupancy running under-sized for its life". It is not reachable
   from the canonical config any more (which now ships `grow: true`), but it is
   reachable from any config that turns growth off, which was the canonical
   default until today and is what several corpus configs carry. Whether
   `grow_batch_size` should gate *recovery* as well as *growth* is a design call,
   so it is reported rather than changed.

**Fixed after owner adjudication — the unit was the problem, not the range:**

8. **`batch_util_target` is now a FRACTION of the card, and the invariant
   enforces it (state 9).** The rule's old range clause was `0 < target <= 100`,
   so the `batch_util_target: 0.6` this session started from **passed the
   percentage check** — a legal 0.6% target. Only the other two clauses
   (`grow_batch_size` false, one-rung ladder) caught the config; had those been
   right, it would have loaded clean and the ladder would have held the *first*
   rung it measured, since any occupancy clears 0.6%: the constraint reporting
   itself as served while serving nothing, which is the exact defect the rule
   exists to prevent. The owner's intent was a fraction, so that is what the key
   now is: canonical ships `0.6`, the range gate is `(0, 1]`, and the single
   percent conversion lives at the one read site in `train.select_batch_size`.
   A leftover `60` now fails loudly instead of asking for 6000% occupancy.
   Recorded as `config_state` state 9 with an empty mechanical transition and
   the reason stated: the two readings **overlap** at 0.6, which is valid under
   both and means opposite things, so no migration may rescale it and a human
   has to decide.

   **Verified at runtime, not just at load, and the test is the interesting
   part.** Under the OLD reading `0.6` means 0.6%, which the base rung's
   occupancy clears immediately — so the ladder would hold batch 1000 and report
   `target_met` on its first measurement. Under the NEW reading it means 60%,
   which the base rung does not clear, so the ladder must CLIMB. The integration
   run reads 28.6% at the base and is climbing (1600 and rising), which only the
   new reading produces. A load-time range check could not have distinguished
   these; the behaviour does.

**Flagged, not fixed — analysis-side blindness:**

9. **`python -m analysis` reports the stage as UNKNOWN on a healthy run.** On
   `a1` it printed `stages [] current=UNKNOWN route=unknown` and skipped **9
   stage-scoped mechanisms**, saying "phase is not in the summary". `phase` IS in
   the summary and in 31 history rows (values 1 then 2). So the route the whole
   check layer keys off is not being resolved on a run that logs everything it
   needs, and every topline printed is the fallback set.
