# Cluster runsheet — what the A100 step must measure

**Status: DRAFT, pending the local shakeout's own results (this file is written
beside `make.py`, which spawns the local arms).** Grades follow
`docs/EPISTEMIC_PROTOCOL.md`: a single arm is `OBSERVED` and does not generalise.

This is the go/no-go list. It exists because the local box can prove correctness
and liveness and **cannot** prove anything the scheduler judges: `torch.compile`
is off on Windows, the policy rollout is dispatch-bound locally, and every floor
in `benchmarks/registry.yaml` is an A100 number. Each arm below therefore carries
the **prediction it tests**, so the arm can come back negative rather than merely
finishing.

---

## 0. What local already established, and what it did NOT

Detail and evidence in `FINDINGS.md` beside this file. Everything below is n=1
and `OBSERVED`.

| Question | Local verdict | Still owed by the cluster |
|---|---|---|
| MACE batched-NL edge sets | **PASS** — `test_pbc_neighbours` 64/64; `test_mace_gpu_real_batches` 11/11 once `MACE_CHECKPOINT` is set | nothing — correctness travels |
| UMA external graph (F-047) | **PASS** — 9/9 once `UMA_CHECKPOINT` is set; internal path moves 0.2437 eV under wrapping vs a 0.0036 eV control | nothing — correctness travels |
| fast paths EXECUTE | **PASS** — all four flags 1.0, `nl_allpairs_calls` 0 | the same flags **on the A100**, where `torch_cluster` has never been verified |
| MACE cost inside a fused step | **NOT MEASURABLE HERE** — OOMs at batch 1 even with `traj_checkpoint`, ~4.8 GiB short on a 16 GB card | §1 and §2 below are the only place this can be read |
| batch sizer control law | **RUNS AND CONCLUDES** — climbed 1000→1600→2560→3560 on the shipped rung arithmetic, held the smallest rung clearing the target, recorded `target_met` | the law under an occupancy signal the SCHEDULER agrees with; the S2 audit, which needs a run longer than one policy window |
| stage transitions | **CLEAN** from both a warm start and step 0: optimizers rebuilt, LR re-warmed, clip guard held its outgoing bar, both `on_exit` snapshots written | transition behaviour at production batch and on MLIP routes, where OOM lives |
| conditional route wiring | **LIVE** — 33 populated `eval_test/*` keys, every condition visited | anything about conditional quality; 300 local steps says nothing |
| occupancy levels, throughput, compile, OOM ceilings | **out of scope** | all of §1–§4 below |

**One local result changes how §1 should be READ.** The MACE executed-path flags
read **0.909, not 1.0, on a report window containing the gas-phase leg**: that
call runs `pbc=False` and the `gpu_batch` branch requires `pbc`. A fraction below
1.0 is therefore not by itself a missed fast path. Read the flags on a window of
pure training steps, or expect exactly this shortfall and check the call count.

---

## 1. MLIP fast paths — the executed-path confirmation

**Why this arm exists at all.** Three switches in `a100_stab_aug16` reported
success without reaching the code (handoff §3.1). The metrics now report the
EXECUTED fraction, and this arm is what proves that on the hardware where the
`torch_cluster` import has never been checked.

| read | prediction | what a miss means |
|---|---|---|
| `energy/mace_flag_batched_nl` | **1.0** | the batched path did not run; every MACE timing below is void |
| `energy/mace_flag_gpu_batch` | **1.0** | the device-built dict did not run (it requires the batched NL — handoff §3.1's VOID arm) |
| `energy/nl_fastpath_frac` | **1.0** | `torch_cluster` did not import on the cluster and the O(Σn²·K) all-pairs kernel is running SILENTLY — presents as "the GPU is slow" |
| `energy/uma_flag_external_graph` | **1.0** | F-047's default did not take; the run is scoring on the edge-dropping internal graph |

**A no-go, not a note.** If any of the four reads below 1.0, the arm's cost
numbers describe a configuration nobody chose, and the remaining MLIP arms should
not launch until it is explained.

## 2. The unrun A/B the handoff names

`BATCHED_NL=1` versus `BATCHED_NL=1 + GPU_BATCH=1`. The handoff retracts v2's
"not worth pursuing" — that arm ran the flag alone with the batched NL off, so
the branch was never taken and its column is a control with a different seed.

**Prediction:** host work is still 22.3% of the MACE call after the NL fix
(handoff §3.1), and this path removes the remaining per-graph AtomicData
construction and collation, so `energy/mace_host_frac` should fall materially
below 0.223 and `energy/mace_ms_per_sample` with it. A null result here is
informative and should be reported as one — but only if §1 reads 1.0, or the arm
is inert again.

## 3. The UMA speed question, which local could not answer

F-047 claims **correctness**, explicitly not speed: locally the two builders are
parity (0.6–1.3×, noisy) and end-to-end is 0.95–0.99×. The A100 measured 26.8% of
the UMA forward as graph construction (handoff §3.2), and whether that converts
is exactly what this reads.

**Read:** `energy/uma_ext_graph_s` against `energy/uma_forward_s`.
**Prediction:** the external build costs materially less than the 26.8% share it
displaces, so `uma_forward_s` falls. If it does not, the graph share does not
convert and F-047 stands as a correctness fix alone.

**The local reading, as the thing to compare against — NOT as a prediction.**
On the dev box in a fused stage (`a3_uma_ext_eq2`, batch 2, 12 reports, 91–146
calls each) the external build is **17% of the forward it feeds**
(`uma_ext_graph_s` 0.560 vs `uma_forward_s` 3.266), the whole UMA call is
**19.4% of the training step**, and `uma_host_frac` is 0.067. Different builder,
different hardware, no compile — so the only use of these numbers is to say the
series are live and non-trivial, and to give the cluster arm something to be
surprised by.

## 4. The batch sizer, armed

The canonical config now ships the ladder **armed** (`batch_util_target: 60`,
`grow_batch_size: true`, `max_batch_size: 20000` — owner edit 2026-08-19). The
stabilization ledger's Phase 6 box stays open until a cluster run demonstrates
the outcome, and this is that run.

**Reads:** the per-rung table in `batch_sizer` (batch, `med_s`, `util`,
`n_util`), `batch/med_step_s`, `updates_per_sec`, `batch/accum_target`,
`vram/peak_train_mb`, and the sizer's phase/reason codes.

**Predictions, in the order they can fail:**

1. **The walk reaches a conclusion.** Either `target_met` at some rung, or
   `infeasible` naming the binding bound. A run that ends `calibrating` means a
   rung starved — check `n_util` against the 60 s sample period and the dwell.
2. **The occupancy the sizer reads is the in-process sensor, which is the one
   F-045 refused.** Its per-rung readings are RAW samples taken during the rung's
   own dwell, not the trailing windowed mean — that is the design's answer to
   §0.2 — but the samples still come from `torch.cuda.utilization()`, so the
   **verdict must be checked against the out-of-process stream** (wandb
   `system.gpu.0.gpu`, or the sidecar CSV filtered to our GPU index). If the
   sizer concludes `target_met` at a rung the out-of-process stream reads below
   40%, the calibration inherits the eval-blindness and the conclusion is wrong
   in the dangerous direction.
3. **The batch never falls below the configured base** except by an OOM cut or
   the `max_step_seconds` guard. A base below `batch_size` with neither of those
   in the log is the prod0810 failure rebuilt.
4. **The S2 audit fires** on any growth that is held, one policy window later,
   and either confirms or stands down. A held growth with `audit_at` never
   reached means the run was shorter than one policy window and the audit is
   simply unproven — say so rather than reading it as a pass. **At least one arm
   must therefore run longer than `gpu_util_policy_window_s` (7200 s) past its
   selection**, or S2 ships untested into production.
5. **THE SELECTION MUST BE REPRODUCIBLE, and locally it was not.** Two runs of
   the same route on the same box, same target, disagreed about the same rung by
   >11 occupancy points and selected different batches (`FINDINGS.md` §5a) —
   with a named confound (a concurrent test suite) that may account for all of
   it. Run **the same arm twice** and compare the selected rung. If the two
   disagree, the selection is being made on 3 samples of a noisy shared counter,
   and the fix is more samples per rung (`_BS_MIN_UTIL_SAMPLES`) or a longer
   dwell — not a different target.
   **Corollary: filter every occupancy reading to our own GPU index.** A
   co-tenant on a multi-GPU node is the cluster's version of the confound above,
   and the sidecar's sample output shows 4-GPU nodes at wildly different
   utilizations.

**The ladder top matters.** F-045's retraction says the missing rung is **ELJ at
15–25k**, and `qm9anchor_aug14` sat at 57–68% at batch 20000 for 34–48 h on a
*different* route. `max_batch_size: 20000` puts that rung inside the domain for
the first time. Whether ELJ unconditional reaches threshold there is the single
measurement that decides whether Phase 6 can use batch at all.

### 4a. BUDGET THE CALIBRATION — it is longer than it looks, and on MLIP it does not finish

A rung is held until it has BOTH a `batch_growth_interval` dwell in steps AND
`_BS_MIN_UTIL_SAMPLES` (3) raw occupancy samples at `gpu_util_sample_period_s`
(60 s). **The slower constraint wins, and which one that is flips with the
route.** The shipped capped-geometric ladder from base 1000 is:

| `max_batch_size` | rungs | floor from the sample period alone |
|---:|---:|---|
| 8000 | 9 | ≥ 27 min per stage |
| **20000** (canonical) | **21** | **≥ 63 min per stage** |

The sample period binds whenever a step is faster than 3.6 s (3 × 60 / 50). Above
that the **dwell** binds instead, and it is brutal: at prod0810's measured
181 s/step, one rung is 50 × 181 s ≈ 2.5 h, so a 21-rung ladder is ~53 hours —
**longer than any job**. The MLIP arms will therefore end mid-walk, in
`phase: calibrating`, with the batch parked at whatever rung they reached and no
conclusion recorded.

**So decide per arm, before launching:**

- **ELJ:** the ladder is affordable. Budget ≥ 1 h of calibration per stage on top
  of whatever the arm is for, and remember `protocol.advance` clears the
  conclusion at every stage transition — so a two-stage run pays it twice.
- **MLIP:** either cap `max_batch_size` to a handful of rungs, or leave
  `batch_util_target: 0` and let the arm measure something else. An MLIP arm with
  the full ladder armed is not a batch-sizer experiment; it is a run that spends
  its life calibrating.
- Either way, **read `batch/sizer_phase`**: ending at `calibrating` (1) rather
  than `hold` (2) means no conclusion was reached, and the final batch is a rung
  the walk happened to stop on rather than a selection.

## 4b. What the local arms deliberately removed, so the cluster must not assume it is covered

Two scope cuts were taken locally on purpose. Both remove exactly the phenomena
the handoff blames for the last battery's MLIP collapse, so their absence here is
**not** evidence that the phenomena are gone:

- **Truncated priors.** The local MLIP arms point at 256/512-row prior fixtures,
  because `init_prior_dataset` re-scores the WHOLE prior through the energy
  function before training starts — 205k rows (MACE) / 176k (UMA) at full size.
  That init pass is precisely the transient handoff §4.4 blames for the cluster
  arms' early OOM (`uma_b250` peaked at 27.7 GB against a 56.8 GB cap, with both
  OOMs landing early). **The cluster arms must run the real priors**, and
  `vram/peak_train_mb` across the init pass is a reading in its own right.
- **`internal_oom_recovery` stays `false`,** as every physical config sets it.
  The energy call therefore receives the entire rollout batch, and its memory
  ceiling IS the rollout ceiling. That pinning is the thing §3.4 exists to
  unpick; nothing local touched it.

## 5. Standing confounds to declare before comparing anything

From `docs/reading_runs.md` §4, the ones this battery can actually hit:

- **UMA energies moved.** F-047 shifts them by ~0.24 eV systematically, so runs
  before and after this default flip are not comparable on the MLIP route —
  beyond the tf32 floor they already were not.
- **T cannot cross a full-state resume** (F-046): the replay buffer stores
  T+1-length trajectories. An arm changing T takes weights only and runs one
  terminal stage from step 0.
- **`epochs` is an absolute index.** A warm start with a small `epochs` runs zero
  steps and reports as a clean run.
- **Arms differing by omission are duplicates.** The canonical config's armed
  ladder is the live instance: an arm that means "no ladder" must now say
  `batch_util_target: 0` explicitly, because it no longer gets that by default.

## 5a. The local integration runs, and what they are evidence OF

`configs/synth_real_aug19/` is a second, smaller battery: two ~20 minute runs,
one unconditional and one conditional, warm-started from phase-1 exits and
overriding **nothing** except run identity, resume point and budget. The
unconditional arm reports **2 deviations from canonical** — that number is the
point. Everything current is live in them, including the armed ladder.

They are evidence that the integrated stack trains cleanly on both routes at
canonical settings. They are **not** evidence about occupancy, throughput or
memory ceilings, for all the reasons in this document's header.

## 5b. Pre-flight — cheap checks that must pass before any arm is queued

Each of these cost minutes locally and each caught something real. None of them
needs cluster time, and every one of them fails in a way that looks like success.

1. **Run the two MLIP gates WITH their models named.**
   `UMA_CHECKPOINT=<esen_s.pt>` and `MACE_CHECKPOINT=<...stagetwo.model>`.
   Without them the suites report "5 passed, 4 skipped" and "11 skipped" — the
   skipped ones being every energy and gradient assertion. On the cluster the
   paths differ, so this is a per-environment check, not a settled one.
2. **`python -m config_snapshot <arm>.yaml --check` on every generated arm**, and
   read the deviation summary the generator prints. An arm that does not load
   reports zero deviations, which reads as clean.
3. **Confirm `torch_cluster` imports in the cluster's environment.** Everything in
   §1 turns on it, its absence is silent, and it has never been checked there.
4. **Confirm the executed step count from the run itself**, not from `epochs`.
   `epochs` is an absolute index against a warm start's step, so a mis-set budget
   produces a clean run that trained nothing.
5. **Decide `grow_batch_size` deliberately on every arm.** With it false the
   sizer never runs at all — including its OOM-ceiling expiry and its
   restore-to-base rule, so an OOM cut is permanent for that run (`FINDINGS.md`
   §6.7). An arm that wants no ladder should set `batch_util_target: 0` with
   growth left ON, not turn growth off.

## 6. Open, and not for this battery

- The scheduler's own statistic, window and threshold — one email to an admin,
  not another battery (handoff §2, F-045).
- §3.4's OOM-recovery / sync pair: the mechanism that would decouple the energy
  batch from the rollout batch. The handoff quantifies the prize (15–70× of batch
  on MLIP) and says the first step is **reproducing** the failure, not designing
  around it.
- The `*_dcgm.txt` sidecars (SM_ACTIVE) remain the one genuinely independent
  instrument and remain unread (F-048).
