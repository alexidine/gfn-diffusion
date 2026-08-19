# Phase 6 handoff — what the A100 battery measured

**Status: ACCEPTED 2026-08-19 (v4).** From `a100_stab_aug16`, 2026-08-16 → 08-18 on
the cluster, plus historical `prod0810` runs where they bear on the same
question. Grades follow `PROTOCOL.md`: a single arm is `OBSERVED` and does not
generalise.

Two consumers: the **MLIP optimizer** (§3) and the **batch sizer** (§4). §2 is
mandatory for both — it says which occupancy number is real, and every
occupancy figure in this document depends on it.

---

## 1. Bottom line

1. **Make the batched MACE neighbour list the default IN CODE** (after the
   equivalence gate) — not an env var someone has to remember. It cuts
   neighbour-list time **25×**, the MACE energy call **2.6×**, the step **27%**,
   *and raises utilization from 54% to 82%*. A behaviour this large should not
   depend on an unset variable defaulting to the slow path.
2. **UMA has the same bottleneck, inside fairchem's forward** — 27% of it is
   graph construction. Reachable only by building the graph externally and
   passing it in.
3. **Faster is not the enemy of occupancy.** Removing *host* work raises
   utilization; only removing *GPU* work lowers it. §3.3 corrects a claim in v1
   that had this backwards.
4. **Rollout cost is determined to ±3%:** `t = T × (9.25 ms + 5.907 µs × B)` for
   the 512-wide policy on ELJ. The overhead is **per rollout step**.
5. **`gpu/util_policy` is unusable; the out-of-process sampler is what to use.**
   See §2 — this is not "occupancy can't be measured", it is "one specific
   metric lies".
6. **MLIP at production T is feasible** — `prod0810_nehzor_uma` held 77.3% for
   47.9 h at T=60, rollout batch 1853. Our arms collapsing to 25–125 is our
   configuration, not the route (§4.4).
7. **Whether to decouple the rollout and energy batch sizes is a DESIGN CHOICE
   to explore, not a settled win.** They are currently pinned together
   (`internal_oom_recovery: false`, deliberately — §4.4), and that pinning costs
   15–70× of batch on MLIP. But separation only pays inside a modest ratio: the
   exploration is where the policy and the energy each stop gaining, and if those
   sit close together, decoupling buys little. See §4.4 for the bound.

---

## 2. How occupancy was measured, and what to use

**Three readings exist. Only two are usable, and they are not the same thing.**

| source | cadence | in/out of process | verdict |
|---|---|---|---|
| `gpu/util_policy`, `gpu/util_recent` | 60 s, in the training loop | **in** | **UNUSABLE** |
| wandb `system.gpu.0.gpu` | ~14 s | **out** | **usable — this is the source for every occupancy number below** |
| `nvidia-smi` sidecar CSV, in `joblogs/*_smi.csv` | 10 s | **out** | usable; higher cadence, plus throttle reasons and per-process attribution |

**Why the in-process one fails.** Against the wandb system stream over the *same*
trailing 7200 s window, same statistic:

| batch | `gpu/util_policy` | out-of-process | error |
|---:|---:|---:|---:|
| 1000 | 31–32% | 37–39% | −6 |
| 4491 | 68–71% | 44–49% | +23 |
| 7410 | 87–89% | 47–49% | **+40** |

The sign flips with batch, so no fixed margin corrects it. Arms self-reporting
87–89% were cancelled for low occupancy. Cause: the sampler sits inside the
training portion of the loop body, so eval, figure logging and archiving
contribute no samples, and the longer and more internally varied the step, the
less representative that instant is.

**So occupancy *is* measurable — every figure in §3 and §4 comes from the
out-of-process stream, including the threshold bracket in §4.3.** What is
refused is `gpu/util_policy` specifically. Phase 4's proxy question resolves to:
adopt the out-of-process sampler, on the grounds that it agrees with the only
cluster-visible outcome available (cancellations).

**CROSS-CHECKED 2026-08-19 — the two out-of-process sources AGREE; the wandb
stream is confirmed.** The sidecar CSVs were pulled from cluster scratch and
compared against each matching run's `system.gpu.0.gpu` (matched by time-span
overlap, since arms were requeued under the same display name; CSV timestamps
are cluster-local UTC−4). 65 CSV↔run pairs: **median |Δ| on mean utilization is
1.3 points, and every pair with ≥2 h of overlap agrees within 4.4 points (most
within 2), including on the trailing-7200 s window.** Deltas above 5 occur only
on overlaps under ~15 min, where cadence and window-edge alignment dominate.
The check also reproduces this document's own numbers independently: the
cancelled `u_scale1000` arms read 37–39% on *both* instruments, `w3_elj_b1000`
49.3–49.7%, `w3_mace_batchednl` 81.4–81.6%. So there is no second problem: the
in-process `gpu/util_policy` is the lone outlier, as §2 claims.

**Still undocumented:** the scheduler's own statistic, window and threshold. Every
observed kill lands at `Elapsed 02:00:2x` with `sacct Reason: None`, so the
working model is a mean over ~7200 s. One email to an admin beats another
battery.

---

## 3. For the MLIP optimizer

### 3.1 The MACE neighbour list — the largest win available

**What is being compared:** three single launches, identical except for one
environment variable. All warm-started from the same `f4_acridine_mace` step-1000
archive, acridine sg14/zp1, **batch 100, T=10, policy width 512**, 400 steps,
`traj_checkpoint: true`. The `energy/mace_*_s` values are **seconds accumulated
between 10-step reports**, so they are comparable to each other but are not
per-call times; `ms_per_sample` and `train_step_time` are the normalised
quantities. Control is the `mace_r0` floor arm at the same batch, T and archive.

| | control | **batched NL** | gpu batch *(VOID — see below)* |
|---|---:|---:|---:|
| `mace_flag_batched_nl` | 0 | **1** | 0 |
| neighbour-list seconds / report | 2.427 | **0.098** | 1.955 |
| build seconds / report | 2.639 | **0.182** | 2.085 |
| forward seconds / report | 1.311 | 1.161 | 1.158 |
| `mace_nl_frac_of_build` | 0.920 | **0.541** | 0.938 |
| `mace_host_frac` | 0.676 | **0.223** | 0.653 |
| `energy/ms_per_sample` | 4.465 | **1.707** | 3.621 |
| `energy/frac_of_step` | 0.453 | **0.220** | 0.366 |
| `train_step_time` (s) | 1.005 | **0.734** | 0.964 |
| **occupancy (out-of-process)** | **54.2%** | **81.6%** | **56.4%** |

Neighbour time **−25×**, energy call **−2.6×**, step **−27%**, occupancy
**+27 points**. The measured `mace-fused-uncond` floor on `energy/ms_per_sample`
is 6.9%, so the effect clears its own noise floor by more than an order of
magnitude — but it is one launch per arm, so `OBSERVED`.

**Gate before shipping.** This measured cost, not correctness. The shift-grid
range fails silently: too small a range drops long edges and the energy merely
moves. `test_pbc_neighbours.py` and `verify_mace_atomicdata_equivalence` make the
exact edge-set comparison and must pass on the batched path first.

**Scope:** batch 100, T=10. The module records a batch-size crossover (~24
crystals) and an earlier 10.8× reading that did not reproduce. Re-measure at the
batch the route runs.

**THE `MXT_GPU_MACE_BATCH` ARM IS VOID — it measured nothing, and v2's "not worth
pursuing" is retracted.** That path has a precondition:

```python
AL_mace_utils.py:169
gpu_batch = USE_GPU_MACE_BATCH and USE_BATCHED_MACE_NEIGHBOURS and pbc
```

It is reachable only with the batched neighbour list also on — its own docstring
says so, because a per-graph host neighbour list would reintroduce the loop it
removes. **I ran the flag alone**, with `MXT_BATCHED_MACE_NEIGHBOURS` off, so the
branch was never taken. Its column in the table above is a control with a
slightly different seed, nothing more.

**The correct A/B is `BATCHED_NL=1` versus `BATCHED_NL=1 + GPU_BATCH=1`**, and it
is unrun. On the prior evidence there is every reason to expect a real effect:
host work is still 22.3% of the call even after the NL fix, and this path removes
the remaining per-graph AtomicData construction and collation.

**A logging gap this exposed:** `energy/mace_flag_gpu_batch` reports the
*environment variable*, not whether the branch executed. Had it reported the
effective path, this arm would have announced itself as inert instead of looking
like a null result. Same for `mace_flag_batched_nl`. Worth fixing before the
re-run, since it is the third time in this battery that a switch failed to reach
the code and reported success.

### 3.2 UMA — 27% of the forward is graph construction

`MXT_UMA_GRAPH_TIMER=1`, batch 250, T=10, 400 steps. `OBSERVED`.

| | |
|---|---:|
| `uma_forward_frac` | 0.979 |
| `uma_graph_frac_of_forward` | **0.268** |
| `uma_graph_s` / `uma_forward_s` per report | 1.143 / 4.262 |
| `uma_host_frac` | **0.021** |

UMA's host side is already solved — 2.1% against MACE's pre-fix 67.6%, so the
AtomicData vectorisation did its job. What is left sits **inside** the forward.

**The direction this implies, and yes it is the obvious one:** mxtaltools
currently hands fairchem an empty `edge_index` and
`crystal_inference_settings` sets `external_graph_gen=False`, so the model runs
`otf_graph` and builds the neighbour list itself. **If we can build a correct
neighbour list faster than fairchem's internal one, we can pass it in** — set
`external_graph_gen=True` and supply a real `edge_index`. We now have a candidate
builder: the batched path that just delivered 25× on MACE. That makes both routes
one problem instead of two.

Same gate applies, and harder here: the edge set must match fairchem's exactly,
or the energy changes silently.

### 3.3 What optimisation does to occupancy — v1 had this backwards

v1 claimed "optimising the energy call lowers utilization, so a speedup can get a
job cancelled." **That is wrong as a general statement, and the batched-NL arm in
§3.1 disproves it: it made the step 27% faster and raised occupancy by 27
points.** The correct distinction:

| what you remove | effect on occupancy | evidence |
|---|---|---|
| **host / CPU-gated work** | **rises** — the gaps that were idle GPU time disappear | batched NL: host_frac 0.676 → 0.223, occupancy 54% → 82% |
| **GPU work, gaps untouched** | falls — the numerator of `busy/(busy+idle)` shrinks | our tuned T=10 arms: 6.7× faster per sample than `prod0810`, and less occupied |
| **VRAM per sample** | rises — headroom converts to batch, and batch raises occupancy | §4.2 |

Low occupancy here is overwhelmingly **host gating and inefficient GPU work**, so
end-to-end efficient GPU execution raises it. Executing batches faster is
intrinsically good and needs no defence.

The one case that still deserves a check: a change that shrinks GPU kernel time
without touching host gaps. That is the narrow claim v1 over-generalised.

---

### 3.4 Load-bearing workarounds to flush out

Two mechanisms are currently shaped by problems nobody has root-caused. Both are
explicitly in scope for this agent, and both are cheap to investigate relative to
what they gate.

**(a) The in-energy OOM-recovery loop — disabled because it was unreliable.**
`batched_analyze_crystal_batch` carries an adaptive shrink loop
(`molecular_crystal.py:797-807`): catch a CUDA OOM, cut the internal batch to
0.65×, `gc.collect()` / `empty_cache()` / `synchronize()`, sleep 0.1 s, retry —
with a hard `assert False, "Cascading OOM failure"` at batch 1. It was turned off
across all physical configs because it *sometimes broke*.

This is the mechanism that would decouple the two batch sizes (§4.4), so making
it trustworthy has a large, quantified payoff. The failure mode itself is
undiagnosed — "sometimes it would break" is the whole record — so step one is
reproducing it, not redesigning around it. A retry loop that can end in
`assert False` mid-training is also the wrong shape for a hot path regardless of
frequency.

**(b) `gc.collect()` / `empty_cache()` / `synchronize()` — and a correction
worth having before you start.** These were once believed wasteful but kept
because the memory stack destabilised without them. **The code has since moved,
and they are no longer in the per-step hot path.** `molecular_crystal.py:812-830`
records the current state: the `not use_recovery` branch returns early, and every
physical config sets `internal_oom_recovery: false`, so training-time energy calls
never reach that block. It was *restored* on 2026-08-14 for a different reason —
the init pass over the whole prior dataset was leaving supercell-shaped blocks in
the caching allocator, and `cuda_memory_fraction` is a hard cap, so unusable
cached blocks still count against it.

**The two interact, and that is the point.** Re-enabling chunking per (a) puts
those syncs back on the hot path — a `synchronize()` per sub-batch serialises host
and device, which is precisely the host gating that depresses occupancy (§3.3).
So (a) and (b) cannot be worked separately: a decoupled energy batch is only
worth having if its recovery path is both reliable *and* sync-free.

Worth verifying rather than assuming: that the early return really does keep the
hot path clean under every route and stage, and whether the original
"unstable without them" observation still reproduces on current code.

---

## 4. For the batch sizer

### 4.1 A predictive cost model

**Scope: ELJ (`mipcas` sg2/zp1), policy width 512 throughout (`s_emb_dim`,
`policy_hidden_dim`, `flow_hidden_dim`, `cond_hidden_dim` all 512; 4 layers;
`dplr_rank` 6), `bwd`/`dataset` branch with zero energy calls.** Six cells,
n=1 each.

```
t_rollout = T × (0.00925 + 5.907e-6 × B)   seconds
```

| cell | measured | model | err |
|---|---:|---:|---:|
| T=10, B=1000 | 0.1516 | 0.152 | +0.3% |
| T=60, B=1000 | 0.9378 | 0.910 | −3.0% |
| T=100, B=1000 | 1.542 | 1.516 | −1.7% |
| T=10, B=7410 | 0.5303 | 0.530 | −0.1% |
| T=60, B=7410 | 3.208 | 3.182 | −0.8% |

**The 9.25 ms fixed term is per rollout step, not per training step** — 61% of
the per-step cost at B=1000, 17% at B=7410. That is why batch buys occupancy, and
why it buys so little at small batch. The model is fitted, not derived, and holds
only for this width and energy function; the sizer should check its own residuals
against it rather than trusting it.

### 4.2 Both T and batch raise occupancy

Out-of-process sampler, same six cells:

| | B=1000 | B=7410 |
|---|---:|---:|
| T=10 | 45.5% | 78.7% |
| T=60 | 61.4% | **92.0%** |
| T=100 | 66.0% | (OOM, batch fell to 3705) |

### 4.3 The threshold, bracketed

All occupancy figures out-of-process (§2).

| arms | occupancy | outcome |
|---|---:|---|
| `u_scale1000_*`, T=10 | 38–40% | **cancelled at 02:00:2x** |
| `w3_elj_b1000`, T=60 | 49.4% | survived to the 3 h SLURM cap |
| `w3_elj_b2722` / `b7410`, T=60 | 70.2 / 76.9% | survived to cap |

**Cancelled at ≤40%, survives at ≥49.4%.**

### 4.4 Why our MLIP arms collapsed — and the two-batch-size finding

Our T=60 MLIP arms OOM-collapsed: `uma_b1000` → 125, `uma_b250` → 62,
`mace_b100` → 25, at 36–57% occupancy. **v1 read this as MLIP being infeasible at
production T. That was wrong**, and the historical record says so plainly:

| run | energy | T | hours | rollout batch | occupancy |
|---|---|---:|---:|---:|---:|
| `prod0810_nehzor_uma` | uma | 60 | **47.9** | 1853 | **77.3%** |
| `prod0810_mipcas_uma` | uma | 60 | 7.4 | 4491 | 70.5% |
| `prod0810_mipcas_uma` | uma | 60 | 5.6 | 103 | 66.7% |

**UMA sustains 70–77% occupancy at T=60 for days.** Two configuration
differences, both confirmed from the runs' own logged configs, explain the gap —
and the first is the important one.

**(1) THE ENERGY BATCH AND THE ROLLOUT BATCH ARE NOT THE SAME NUMBER.**
`energy_config.internal_oom_recovery` is **absent** from prod0810's logged config,
so the code default applied (`molecular_crystal.py:75`, `= True`): the energy
function **chunks internally**, on a sticky size fixed by the init-time prior
re-analysis — recorded for prod0810 uma as **2150 against a 2722 rollout**. Every
current config sets it explicitly `false`, so the energy call receives the
*entire* rollout batch in one go and its memory ceiling becomes the rollout
ceiling.

**That `false` is a deliberate decision, not a configuration slip** (MK,
2026-07): internal chunking was retired in favour of grad accumulation because
**the repeated OOM-recovery loop inside the energy call was not trustworthy — it
sometimes broke.** Grad accumulation gets the same effective update from a
smaller batch without a recovery loop in the hot path. So the trade was made
knowingly; what this battery adds is its *price*, which had not been measured:
the rollout batch is now pinned to the energy call's ceiling, and on the MLIP
routes that costs 15–70× of batch and the occupancy that comes with it.

So "batch 1853" and "batch 62" are not comparable quantities. The historical runs
were sustaining a large *rollout* batch — which is what drives policy work and
therefore occupancy — while the energy call stayed inside memory.

**This is a design input for the sizer, not a footnote.** There are two batch
sizes:

| | drives | bounded by |
|---|---|---|
| **rollout batch** | policy work per step → occupancy, throughput | activation memory (mitigated by `traj_checkpoint`) |
| **energy batch** (chunk) | energy-call memory | the MLIP's own footprint |

Decoupled, the sizer can raise the rollout batch for occupancy without the energy
call following it up. Pinned, the *smaller* ceiling governs both.

**But separation has a natural bound, and the exploration is where each side
stops gaining.** Taken to the limit it is absurd: if the policy needed batch 20k
to saturate while the energy ran at 25, a step would issue 800 energy sub-calls,
and the per-call overhead would swamp anything the decoupling bought. Gradients
also saturate well below where occupancy does, so a large rollout batch is not
free of diminishing returns either. The useful question is not "should they be
decoupled" but **how far apart the two saturation points actually are** — if they
are close, pinning costs little and the reliability argument for staying pinned
wins outright.

**So the ask is not "turn the flag back on".** The mechanism that decouples them
is currently disabled for a good reason (above). What is worth engineering is a
*trustworthy* way to run the energy at its own batch size — §3.4. Two further
constraints on any such design:

- The canonical config's own comment records that internal chunking **"swallowed
  OOMs the batch controller needed to see"**. It buys headroom by blinding the
  controller to the energy call's real ceiling. **Neither the internal chunk size
  nor internal OOM events are logged today**, and a sizer that relies on chunking
  needs both.
- The sticky chunk size is set by the init-time prior re-analysis, so it is
  chosen once, on a different distribution from the one training visits.

**(2) `cuda_memory_fraction` 0.7 on our arms against 0.9 in production** — we
handed back 22% of the card before starting. Straightforwardly my error in the
battery generator, and unrelated to (1).

A third observation makes (1) the likelier culprit than a steady-state limit:
peak train VRAM on `uma_b250` was **27.7 GB against a 56.8 GB cap**, so the
steady state was nowhere near the ceiling. Both OOMs landed early, consistent
with a transient — the `rebuild_prior_by_churn` entry pass scoring the whole
prior through an unchunked energy call.

**Grad accumulation also decouples batch from update quality.**
`fused_grad_accum_min_samples: 1000` means a batch of 25 accumulates 40
micro-steps into a 1000-sample update. A small batch is a *throughput and
occupancy* problem — more host round-trips per update — not an
optimisation-quality problem. The memory ceiling is a constraint to work under,
never a verdict on the route.

### 4.5 What to read

| metric | why |
|---|---|
| `batch/med_step_s` | the 20-step median `increment_batch_size` **actually acts on**. Every other cost metric is a 10-step report mean, so a curve built from those is not one the controller can reproduce. |
| `updates_per_sec`, `batch/accum_target` | optimizer-step throughput. Differs from `samples_per_sec` below `fused_grad_accum_min_samples`, which is where every MLIP arm runs, and the two objectives have different argmaxes there. |
| `vram/peak_train_mb` | peak reset at each eval, so the memory constraint is a train-phase quantity. `vram/peak_reserved_mb` stays a run-lifetime max and is often the *eval* peak. |

All three are new in this battery and verified emitting on a real run.

---

## 5. Open

- **`w3_elj_b500`** ended at 136 min, no OOM, 41.4% occupancy — neither the 3 h
  cap nor a clean 2 h kill. It sits inside the threshold bracket, so explaining
  it could tighten §4.3 from both sides.
- **No floors on any wave-3 arm** — every cell is n=1. The three measured floors
  (`elj`/`uma`/`mace-fused-uncond`, 5 launches each, T=10) are in
  `benchmarks/registry.yaml`.
- **`elj_b20000` OOM'd to 5000** and `roll_T100_b7410` to 3705, so the top of both
  ladders is a ceiling, not a data point.
- **The scheduler's statistic, window and threshold** (§2).

Retractions and mechanisms: `docs/findings.md` F-041 → F-046. Arm-by-arm design
and cost: `configs/a100_stab_aug16/RUNSHEET.md`. Measured floors:
`benchmarks/registry.yaml`.
