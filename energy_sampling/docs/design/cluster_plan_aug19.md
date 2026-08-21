# Cluster experiment plan — questions, and the arm that answers each

**Status: PROPOSAL, 2026-08-19.** Written after the local shakeout
(`configs/synth_aug19/`, `synth_real_aug19/`, `synth_prof_aug19/`). Nothing here
is scheduled; the point is to agree what is being asked before anything is
queued. Grades follow `docs/EPISTEMIC_PROTOCOL.md`.

**The organising rule: every arm carries a prediction it can FAIL.** An arm that
cannot come back negative is a demonstration, not an experiment, and this
battery has room for neither. Where an arm's answer is already bracketed
locally, the local number is given as the thing to be surprised by.

---

## 0. Read this first — what local already settled, so nobody re-runs it

| Settled locally | Do NOT spend cluster time on |
|---|---|
| MACE batched neighbour lists and the UMA external graph are numerically correct (64/64, 11/11, 9/9; F-047's 0.2437 eV wrap discriminator reproduced exactly) | re-deriving correctness |
| The fast paths execute and report it (`*_flag_*` = 1.0, `nl_allpairs_calls` = 0) | *whether* they work — only whether they work THERE |
| The wall-clock energy split is honest (`energy/gpu_over_wall` = 1.002, n=50) | validating `energy/frac_of_step` again |
| The sizer walks, selects the smallest clearing rung, and reaches a conclusion | the control law's basic operation |
| Fused Adam removes ~94% of Adam's host syncs. **The local ~10% step-time / ~10-point occupancy gain is RETRACTED** — it was order-confounded and did not replicate on the cluster (C1: 0.538 vs 0.533 s, occupancy 36.2 vs 40.9, fused marginally worse) | micro-benchmarking Adam in an isolated loop — that instrument synchronises anyway and reports the effect as noise |

---

## 1. GO/NO-GO gate — runs before anything else is queued

| # | Question | Arm | Prediction | If it fails |
|---|---|---|---|---|
| **G1** | Does `torch_cluster` import on the cluster? Everything in §2 rests on it and its absence is SILENT. | any MACE arm, 50 steps | `energy/nl_fastpath_frac` = **1.0**, `nl_allpairs_calls` = **0** | STOP. Every MLIP cost number would describe the all-pairs fallback, which is orders of magnitude slower and presents as "the GPU is slow". Fix the environment, restart. |
| **G2** | Do the fast paths execute under the cluster's environment and compile settings? | same arm | `mace_flag_batched_nl` = `mace_flag_gpu_batch` = `uma_flag_external_graph` = **1.0** | STOP for that route. 0.909 is expected and fine ONLY if the window contains the gas-phase (`pbc=False`) leg — check the call count before accepting it. |
| **G3** | Does anything compile-only break? Inductor is off on the dev box, so nothing local can see this. | one ELJ + one MLIP arm, 200 steps, `compile_policy: auto` | no compile-fallback warnings; step time settles after the first recompiles | Not a stop, but every timing in the battery must then be labelled eager. |

## 2. The MLIP questions the handoff left explicitly unrun

| # | Question | Arm | Prediction | Notes |
|---|---|---|---|---|
| **M1** | **The A/B the handoff retracts its own verdict on:** batched NL alone vs batched NL + device-built dict. The earlier arm ran the second flag with the first OFF, so the branch was never entered and its column is a control with a different seed. | 2 arms, MACE acridine sg14/zp1, batch 100, T=10, identical seed and archive | host work is still 22.3% of the call after the NL fix, and this path removes the remaining per-graph AtomicData build — so `energy/mace_host_frac` falls **materially below 0.223**, and `ms_per_sample` with it | A null result is useful and should be reported as one — but only if G2 reads 1.0, or the arm is inert again. |
| **M2** | Does the UMA external graph BUY anything, or is it correctness only? | 2 arms, UMA mipcas, external graph on/off | the A100 measured 26.8% of the forward as graph construction; if the external builder is cheaper, `uma_forward_s` falls by a visible fraction of that | **Local says parity** (the build is 17% of the forward it feeds; end-to-end 0.95–0.99x). F-047 claims correctness and explicitly not speed. Flat is a real answer and leaves the claim unchanged. |
| **M3** | What does an MLIP route cost per step at production T with the fast paths on, and does it hold occupancy? | 1 UMA arm, T=60, **full prior**, hours | `prod0810_nehzor_uma` held 77.3% for 47.9 h at rollout batch 1853, so the route is feasible; the collapse to 25–125 was our configuration | Run the REAL prior. The local arms used 256/512-row fixtures, which removes precisely the init transient handoff §4.4 blames for that collapse. |

## 2b. PRODUCTION SHAKEOUT — the whole stack, hours, nothing held back

**This is the arm the rest of the plan exists to earn, and it was missing from
the first draft.** Everything above is a controlled comparison with one thing
varied. None of it answers *does the production configuration survive contact
with reality for hours*, which is a different question and the one that decides
whether this stabilization work is finished.

**Configuration: production, not diagnostic.** Full prior, production T, the
MLIP fast paths on (they are in-code defaults), the batch sizer ARMED,
`cuda_memory_fraction: 0.9` (0.7 was the previous battery's generator error and
handed back 22% of the card), checkpoints writing, cluster eval/figure cadence,
fused Adam on. Several hours. Two arms, UMA and MACE, because their memory and
host profiles differ and neither predicts the other.

| # | Question | Prediction | What a failure looks like |
|---|---|---|---|
| **P1** | Does a UMA production run hold occupancy above the cancellation bracket for hours? | `prod0810_nehzor_uma` held **77.3% for 47.9 h** at rollout batch 1853, and the fast paths since then only remove host work — which RAISES occupancy (handoff §3.3). So: at or above that, and comfortably past the ≥49.4% survival mark. | Below ~40% is a run that gets cancelled. Below prod0810 means the fast paths cost occupancy, which would contradict §3.3 and is the most interesting possible result here. |
| **P2** | Does it survive without an OOM cascade? | no `Cascading OOM Failure`; `batch/oom_events` small and non-escalating; `vram/peak_train_mb` well under the cap | The init prior re-analysis is the known transient (handoff §4.4: `uma_b250` OOM'd early at 27.7 GB peak against a 56.8 GB cap). A cascade here re-opens the energy-batch decoupling question with a real cost attached. |
| **P3** | **Does the batch sizer behave on an MLIP route, or does it need to be off there?** | on MLIP step times a rung costs a DWELL, not samples — at 181 s/step, `batch_growth_interval: 50` is 2.5 h per rung and 21 rungs cannot finish. So either the ladder is configured short (see below) and concludes, or it must be turned off for MLIP and that is a documented limitation. | Ending in `phase: calibrating` after hours, with the batch parked on an unvalidated rung, is the failure — and it is the DEFAULT outcome if the ladder is left at ELJ settings. |
| **P4** | Does the run remain resumable? | a mid-run archive reloads and continues without a step-count or buffer-shape complaint | T is fixed within a run, so F-046 does not bite here; what is untested is a full-state resume of an MLIP run with the sizer's conclusion in the checkpoint. |
| **P5** | Do the losses actually move over hours? | log Z rising smoothly, `tb_err_worst` trending down, no deep log Z dive at the stage transition | This is the only arm long enough to say anything about training at all. Read it per `docs/reading_runs.md`, not as a throughput number. |

**P3 needs a decision before launch, and it is the one thing in this section
that is not just "run it".** The ladder's rung cost on an MLIP route is set by
`batch_growth_interval` in STEPS, and MLIP steps are ~100x longer than ELJ ones.
Three options, and the arm should pick one deliberately rather than inherit ELJ's:

1. **Short ladder** — set `max_batch_size` near the route's known memory ceiling
   (a few rungs, not 21). Cheapest, and honest: on MLIP the batch is bounded by
   the energy call's memory anyway, so a 20000 ceiling is fiction there.
2. **Short dwell** — drop `batch_growth_interval` to ~5–10 steps for MLIP. At
   181 s/step the 3-sample occupancy requirement is met by a SINGLE step, so the
   dwell is pure overhead beyond what a step-time median needs.
3. **Ladder off** (`batch_util_target: 0`) — hold the configured batch, and
   record that the sizer is an ELJ-route feature until the MLIP rung cost is
   addressed.

Recommendation: **(1) plus (2)** on the first production arm — a short ladder
with a short dwell, so the sizer is genuinely exercised on the route rather than
excluded from it. If that still cannot conclude, (3) is the finding, and it
belongs in the stabilization ledger rather than in a config comment.

**B4 rides here.** The S2 stand-down audit needs a run held more than one policy
window (7200 s) past its selection, and this is the only arm in the plan that is
long enough. No separate arm required — just confirm `audit_at` was reached.

## 3. The batch sizer — the box the stabilization ledger keeps open

| # | Question | Arm | Prediction | Notes |
|---|---|---|---|---|
| **B1** | Does the ladder reach a conclusion within a job on ELJ? | ELJ, ladder armed (`batch_util_target: 0.6`), `max_batch_size: 20000` | ends `batch/sizer_phase` = hold, with `target_met` or `infeasible` | **Budget it:** 21 rungs, and each needs both a dwell and 3 occupancy samples at 60 s — **at least 63 min of calibration per stage**, paid again after every stage transition. Ending at `calibrating` means no conclusion and the batch is a rung it stopped on. |
| **B2** | **Is the selection REPRODUCIBLE?** | B1's arm run **twice**, identical config | both runs select the same rung | Locally two runs of the same route disagreed by >11 occupancy points and picked different batches — with a concurrent test suite as a named confound. The selection rests on **3 samples** of a counter that is not exclusively ours. If they disagree, the fix is more samples per rung, not a different target. |
| **B3** | Does the in-run calibration AGREE with the number the scheduler judges? | B1's arm | the per-rung reading sits within a few points of `system.gpu.0.gpu` over the same window | **The one that matters.** Calibration samples the same in-process counter F-045 refused. A `target_met` at a rung the out-of-process stream reads under 40% is wrong in the dangerous direction: a job that reports itself healthy and is cancelled anyway. |
| **B4** | Does the S2 stand-down audit work? It has never executed. | **rides on the §2b production arm** — the only one held more than 7200 s past its selection | the audit fires and either confirms the growth or stands it down | No local run outlives one policy window, so S2 ships untested. It needs no arm of its own: just confirm `audit_at` was reached and read the verdict. |
| **B5** | Does batch buy occupancy on ELJ at 15–25k — the rung F-045 names as missing? | ELJ, FIXED batches 7410 / 12000 / 20000, ladder off | F-045 measured 38→48% over a 7.4x batch and called batch a weak actuator; `qm9anchor_aug14` sat at 57–68% at batch 20000 on a *different* route | **This decides whether Phase 6 can use batch at all.** Locally occupancy rose steeply with batch (23.7→71.0%), contradicting F-045 — a different machine, so it predicts nothing, but the sign is genuinely open. |

## 4. Occupancy and the scheduler

| # | Question | Arm | Prediction | Notes |
|---|---|---|---|---|
| **O1** | What statistic, window and threshold does the scheduler actually use? | **not an experiment — one email to an admin** | — | Every occupancy number in this project is inferred from cancellations. Support supplied a sidecar recipe but not the rule. Cheapest open item in the plan, and it can retire an inference the whole of §3 leans on. |
| **O2** | Do the two out-of-process instruments still agree at production batch? | any long arm: sidecar CSV vs wandb system stream | median absolute difference ~1.3 points (F-048, 65 job pairs) | Filter the sidecar to OUR GPU index. Sample output shows 4-GPU nodes at wildly different utilizations, and a co-tenant is the cluster's version of B2's confound. |
| **O3** | Is the DCGM sidecar telling a different story? | any long arm, read the `*_dcgm.txt` files | SM_ACTIVE below NVML utilization, possibly far below | NVML "utilization" is the fraction of time ANY kernel was resident, not how full the device was. Both current instruments read NVML, so they *cannot* disagree about this. **The only genuinely independent instrument, and it has never been read.** |

## 5. Cost of the changes shipped this week

| # | Question | Arm | Prediction | Notes |
|---|---|---|---|---|
| **C1** | Does fused Adam's local gain survive compile? | ELJ, `MXT_FUSED_ADAM` 1 vs 0, otherwise identical, **order reversed between replicates** | local end-to-end A/B: `batch/med_step_s` **-10.4%**, `gpu/util_recent` **+10.2%**, samples/sec +5.2% | Compile may absorb the gain (it fuses launches anyway) or amplify it (a shorter step makes each stall a larger fraction). An isolated `optimizer.step()` benchmark said 1.6% = noise and was the WRONG instrument: it synchronises anyway, and the cost of a sync is the next step's work not being queued. Do not repeat that measurement. |
| **C2** | Where do the OTHER ~98% of device-to-host syncs come from? | ELJ, `profiling.trace.with_stack: true`, few steps | — | ~11 600 syncs/step measured; only ~220 are visible from Python, of which Adam was 70%. The rest originate below Python and are **unattributed**. `with_stack` exists for exactly this, and it is now cheap to run for many steps since the 748 MB chrome trace became opt-in (`write_trace`). |
| **C3** | Is the step still host-bound once compile is on? | ELJ, op table, compile on | local (eager) measured Self CPU 6.0 s against Self CUDA 1.5 s over 8 steps | That 4:1 is profiler-inflated and eager-only. Compile is the intervention most likely to change it, and it does not exist locally. |

## 6. Sequencing, and why

**Only ONE step is a real barrier.** The dependency below is about not wasting
GPU time on arms whose interpretation depends on an earlier answer — it is not a
wall-clock chain. With ~16 A100s available concurrently, everything after the
gate can be queued at once; the ordering matters for *what you read first*, not
for what you launch first.

1. **G1–G3 first, on short arms — the one true barrier.** Three switches in the
   last battery reported success without reaching the code. The gate costs
   minutes and voids every MLIP number downstream if it fails, so nothing else
   should burn hours until it passes.
2. **Then launch everything else together.** Specifically:
   - **§2b production arms (UMA, MACE)** — the long poles, hours each, so they
     start first and run while the short arms come back. B4 rides on them.
   - **B5** — read this one EARLY among the results: if batch does not buy
     occupancy at 20k on this route, the ladder's ceiling is wrong and B1's
     63-minute calibration is being spent on a lever that does not move.
   - **M1/M2** — cheap, and the handoff's explicitly unrun items.
   - **B1–B3** — the sizer on ELJ.
3. **O1 needs no GPU at all** — it is an email to an admin, and it can retire an
   inferred rule that §3 and §2b both lean on. Send it before anything is queued.

The reading order, once results land: G1–G3 → B5 → §2b → M1/M2 → B1–B3.

## 6b. A warm-start hazard that will bite the cluster arms

**Warm-starting from a `_phase1_exit.pt` archive does NOT skip phase 1.** Found
the hard way locally: a conditional arm resumed from its own phase-1 exit and
then spent its entire budget re-running the MLE stage it had already finished.
Two mechanisms combine, and each is invisible on its own:

1. `skip_if: prior_loaded` — the mechanism that exists for exactly this — fires
   **only on a fresh run**. `protocol.py:1335` returns immediately when
   `step_ind != 0`, because a resumed run is deliberately left wherever its
   checkpoint says. A resume therefore re-enters the stage the archive exited.
2. A stage's `exit:` is an **AND-list**. On `conditional_vargrad`'s `train_prior`
   that is `gates/mle_flat` **and** `eval/wass_debiased < 0.015` **and**
   `bwd/tbc < 2.0`, so loosening the MLE gate alone cannot release it — and two
   of those three are quality bars unrelated to whether MLE is done (`wass` is
   not interpretive on crystal targets at all).

**Consequence for this plan:** any arm here that warm-starts from a phase-1 exit
and expects to run in its terminal stage must either replace the `exit:` block
(as the local shakeout did — one satisfiable term, `mle_gate` set to call any
descent flat) or start fresh with a `prior_model_name` so the skip is reachable.
Otherwise the arm silently measures the wrong stage, and the only symptom is a
`phase` of 1 in an otherwise clean run.

**Check it costs nothing:** confirm `protocol: stage '<a>' -> '<b>'` appears in
the log, and that `phase` in the summary is the stage you meant to measure.

## 7. What this plan deliberately does NOT ask

- Anything settled locally (§0).
- Anything about conditional-route *quality*. Nothing under ~5 units is resolvable at one seed, so a conditional quality question needs its own battery with its own seeds — not a rider here.
- The energy-batch decoupling (`internal_oom_recovery`). The handoff prices it at 15–70x of batch on MLIP routes and says the first step is **reproducing** the failure that got it disabled. That is a debugging task, not an arm.

---

# Round 2 — the shift cap, and the arms it unblocks

Added after the `mem` wave came back. Everything above stands; this is what
`mem` changed.

## 8. What `mem` established, and the mechanism behind it

`mem` refuted the fragmentation diagnosis outright. MACE at batch 100 cascades
100 → 23 with three OOM events inside the first ten steps, and **every allocator
knob was null** — `garbage_collection_threshold`, `max_split_size_mb`, and
`traj_checkpoint` all moved it nowhere. Only raising `cuda_memory_fraction`
0.9 → 0.97 moved the settling batch at all (23 → 38), which says the cost is a
transient inside the energy call rather than anything the allocator can pack
around.

Local profiling then found a mechanism. `lattice_shift_range` clamps only from
*below*, so as a cell flattens its interplanar spacing → 0 and the requested
shift range is unbounded; and `batched_pbc_neighbour_list` takes `.max(dim=0)`
across the batch, ghost-expanding **every** graph on the worst cell's grid.
Measured at 128 acridine graphs / 2944 atoms / 6 Å:

| squash | grid | K | peak MiB | with cap |
|---|---|---|---|---|
| physical | [3,3,2] | 245 | 254 | 254 |
| ×0.01 | [3,3,52] | 5 145 | 1 725 | 361 |
| ×0.003 | [3,3,171] | 16 807 | 3 825 | 361 |
| ×0.001 | [3,3,510] | 50 029 | **OOM** | 361 |

The edge count rose only 2.5× while memory rose 15× — so it is ghost expansion
that 127 sane cells never needed. A fresh policy emits exactly those cells, which
is why the failure is an early transient and why it clears if the run survives.

**F-049 stands but is narrower than it reads.** It says the cost is the
energy-call transient, and that remains what was measured. It is *not* a claim
that the MACE route cannot scale — this hardware has run energy batch 73 against
policy batch 3000 at T=100, so the ceiling is an engineering number. §10 is the
arm that measures it.

## 9. The cap, and why this shape

`MAX_SHIFT_RANGE = 8`, applied inside `lattice_shift_range`, counted every time
it binds. Physical acridine needs `n_i` ≤ 3 at a 6 Å cutoff, so 8 leaves 2.7×
headroom and bounds K at 17³ = 4913.

**Shifts, not `max_num_neighbors`.** Truncating the neighbour list of a
*geometrically sound* cell is the silent-wrong-energy failure this module exists
to prevent — the one F-047 removed from the UMA path. Capping shifts drops
distant images of a cell that is already meaningless.

**Overridable by `MXT_MAX_SHIFT_RANGE`**, so the comparison is a single-key
toggle rather than a comparison between two builds. A cross-version A/B cannot
isolate the cap from anything else that moved.

**Where the cap binds, it changes the energy.** That is only harmless because a
cell that degenerate is rejected by the bounding term anyway — an argument that
is not self-verifying. `capped_frac` must be read alongside sample acceptance: if
capping is common *and* those samples are being accepted, the cap is silently
corrupting energies, and per-graph grids stop being an improvement and become
the fix.

## 10. The `nlcap` wave

| arm | toggle | question | prediction |
|---|---|---|---|
| `nl0_shiftcap_off` | `MXT_MAX_SHIFT_RANGE=100000` | does the uncapped path reproduce `mem`? | 3 OOM events, batch collapses 100 → 23 |
| `nl1_shiftcap_on` | `MXT_MAX_SHIFT_RANGE=8` | does the cap stop it? | `capped_frac` > 0 early, `oom_events` 0, batch **holds at 100** |
| `nl2_shiftcap_ladder` | cap on, ladder armed from 25 | *how far* does MACE go now? | settles 100–400 |

`nl0`/`nl1` are byte-identical configs — verified by diff, not assumed. The
`.env` is the only axis. `nl2` starts at 25, *below* the size that cascaded, so a
failure to climb is informative rather than a repeat of the crash.

**Read `energy/nl_shift_capped_frac` first, and treat 0 as a real negative.** If
the cap never binds on the cluster then cluster degeneracy never reached it, and
this is not the cluster's mechanism. The local evidence is an inference from
scaling — N·K, with the MACE supercell batch ~10–30× our local atom count, which
brackets the observed 72 GB — not a measurement of the cluster failure.

## 11. Sequencing

`nlcap` is a **gate**. Submit it alone, because every held MACE arm's batch
depends on its answer.

```bash
sbatch configs/cluster_aug19/submit_nlcap.sbatch
```

Then, only if `nl1` holds its batch, the arms held on the OOM. They need no edit
— the cap is the in-code default, so they pick it up as they stand:

```bash
sbatch configs/cluster_aug19/submit_mlip.sbatch
```

`p2_mace_prod` goes last, with `batch_size` refit to whatever `nl2` settles at
rather than the 100 it currently carries:

```bash
sbatch configs/cluster_aug19/submit_prod.sbatch
```

## 12. Still open, and NOT in this wave

- **Per-graph shift grids** — the real fix, so a degenerate cell costs only
  itself. A ragged-K change to the ghost expansion; a bigger job than the cap,
  and the cap does not remove the waste, only bounds the damage.
- **The four crashed `sizer` arms** — cause unknown, `sacct` never retrieved.
  `b5_fixed7410` and `b5_fixed12000` reported occupancy from partial runs, so
  those numbers are provisional and should not be quoted as settled.
- **`c2_syncs`** — its stack-attributed op table is on cluster disk and wants
  *reading*, not re-running. It is what would attribute the ~11 600 device→host
  syncs per step, of which only ~220 are visible from Python.
