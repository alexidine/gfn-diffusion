# Metrics added in the 2026-08-19 stabilization work

Scope note: this is a **pointer list**, not a definitions file — `module_metrics.md`
owns definitions and `reading_runs.md` owns the interpretive method. It exists
because a run now emits several dozen new series and a reader needs to know which
few decide anything.

★ = worth a dashboard slot.

## The batch sizer's account of itself (new with the state-8 replacement)

| metric | what it answers |
|---|---|
| ★ `batch/sizer_phase` | 0 none / 1 calibrating / 2 hold. **Ending at `calibrating` means no conclusion was reached** — the final batch is a rung the walk stopped on, not a selection. |
| ★ `batch/sizer_reason` | 1 no_target · 2 target_met · 3 infeasible · 4 sensor_off · 5 no_headroom · 6 stood_down · 7 wallclock_cut. Encoded as a series because prints do not survive a hard-killed run. |
| ★ `batch/med_step_s` | the 20-step median the controller **actually acts on**. Every other cost metric is a 10-step report mean, so a curve built from those is not one the controller could reproduce. |
| ★ `batch/accum_target` | with `updates_per_sec`, the optimizer-step throughput pair. Diverges from `samples_per_sec` below the accumulation floor — which is where every MLIP arm runs. |
| `batch/sps_rung` | samples/sec at the live rung. |
| `batch/oom_ceiling`, `batch/oom_min`, `batch/oom_events`, `batch/ceiling_expiries` | the OOM ceiling's life: where it sits, the smallest size ever seen to fail, how often allocation failed, how often the ceiling expired and re-probed. |
| `batch/sizer_rungs` | rungs measured so far. |

## Memory

| metric | what it answers |
|---|---|
| ★ `vram/peak_train_mb` | peak **reset at each eval**, so it is a train-phase quantity. |
| `vram/peak_reserved_mb` | run-lifetime max — usually the **eval** peak. Not the same question; do not compare them casually. |
| `vram/{live,reserved,cached}_mb`, `vram/peak_train_phase_mb` | allocator state. |

## MLIP: which code path actually executed

The point of this group is that three switches in the last battery reported
success without reaching the code. These report the **executed fraction**, counted
at the branch — not the environment variable.

| metric | what it answers |
|---|---|
| ★ `energy/mace_flag_batched_nl`, `energy/mace_flag_gpu_batch` | 1.0 = every call took the path. NB a window containing the **gas-phase leg** reads below 1.0 legitimately: that call is `pbc=False` and the gpu-batch branch requires `pbc`. |
| ★ `energy/uma_flag_external_graph` | 1.0 = F-047's external `edge_index` was handed in, rather than fairchem building its own (edge-dropping) graph. |
| ★ `energy/nl_fastpath_frac` (+ `nl_radius_calls`, `nl_allpairs_calls`) | whether `torch_cluster`'s radius search ran, or it fell back **silently** to the O(Σn²·K) all-pairs kernel. The fallback presents as "the GPU is slow", never as a missing dependency. |
| `energy/mace_host_frac`, `mace_forward_frac`, `mace_nl_frac_of_build` | where a MACE call's seconds go. |
| `energy/uma_host_frac`, `uma_graph_frac_of_forward`, `uma_graph_s`, `uma_forward_ex_graph_s` | the same split for UMA, with graph construction broken out of the forward. |
| `energy/mace_calls`, `uma_calls`, `mace_flag_hoisted`, `uma_flag_vectorised` | call counts and the remaining path flags. |

## Energy cost attribution

| metric | what it answers |
|---|---|
| ★ `energy/frac_of_step` | where the step's seconds go. Paired with occupancy it separates *the MLIP call is expensive* from *the MLIP call is idle waiting on the host*. **Reads 0 on bwd/dataset MLE stages, correctly** — those make no in-step energy call. |
| `energy/seconds_in_step` vs `energy/seconds` | in-step share vs **every** call including eval, init and prior churn. Only the first may be divided by the step window; the raw total gave 1.48 on the first real run. |
| `energy/ms_per_sample` | directly comparable across energy functions. |
| `energy/frac_outside_step` | how much MLIP the run does outside training. |

## Profiling (both layers ship OFF)

| metric | what it answers |
|---|---|
| ★ `energy/gpu_over_wall` | **the validation number.** CUDA-event device time over the wall clock measured around the same call. `energy/frac_of_step` is a wall-clock subdivision taken without synchronising, and wall clock around async CUDA work measures when the *launches* returned — this says whether that subdivision is honest. |
| `energy/seconds_gpu` | the device-side companion to `energy/seconds`. Absent means *not measured*, never zero. |
| `perf/*_ms` | per-region device time. **Only one region is instrumented today** (`energy`), so this layer validates the energy split and cannot decompose the step. |

The second profiling layer (`profiling.trace`) writes a chrome trace and an op
table to `profiling_results/` and never to wandb; it is the layer that can
actually decompose a step, and it is bounded to a few steps because left on,
`torch.profiler` dominates what it measures.

## Ray probe visibility

`raycal/refused` + `raycal/refused_reason` — load-bearing rather than decoration:
a refused probe produces **no** `raycal/*` measurement keys at all, which is
otherwise indistinguishable from a sensor that was never wired up.

## Not new, and still not admissible for decisions

`gpu/util_recent` and `gpu/util_policy` are the **in-process** reading. F-045
refused them: the error runs −6 to +40 points with a sign that flips with batch,
because the sampler sits inside the training portion of the loop and eval,
figure logging and archiving contribute no samples. The scheduler judges the
**out-of-process** number (wandb `system.gpu.0.gpu`, or the `nvidia-smi` sidecar
filtered to our GPU index). They survive as a logged series; nothing may be
decided on them.
