# GPU occupancy

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

*Occupancy*, or *utilisation*, is the fraction of wall clock during which the GPU was executing at least one kernel. The trainer measures it as a percent, averages it over two trailing windows, and publishes those as `gpu/util_recent` and `gpu/util_policy`. This page is what that number is, how the trainer obtains it and where it is read. The batch sizer is [batch-size](batch-size.md); the preflight guard and memory caps that share `gpu_guard.py` are [compute-guards](compute-guards.md); what a scheduler does with an occupancy figure is [cluster-operations](cluster-operations.md).

## What the number is

The underlying counter is NVML's `utilization.gpu`: the percent of time over the driver's own sample period during which one or more kernels was resident. It indicates *any kernel running*, not how much of the device that kernel uses, so one small kernel held for the whole period reads 100.

Write $g$ for the seconds per step in which some kernel is resident and $h$ for the seconds in which none is, so

$$
\text{util} = \frac{g}{g + h}.
$$

$g$ is work the device is asked to do; $h$ is time the host spends between launches, in Python dispatch, in data movement, or blocked outside the process. The trajectory loop issues a fixed number of kernel launches per rollout step, independent of the batch, so widening the batch enlarges each kernel without adding launches: $g$ grows while $h$ stays, and the ratio rises.

## The sensor: source, cadence, retention

`train.py::Modeller._read_gpu_util` returns one raw percent from whichever of two sources answers. It tries `torch.cuda.utilization()` first, which requires the pynvml bindings and reads the current CUDA device, already selected by `CUDA_VISIBLE_DEVICES`. If that raises, it falls back to `gpu_guard.py::gpu_memory`, which shells out to `nvidia-smi --query-gpu=...,utilization.gpu` and returns the utilization field of one row of a table listing every card on the node; the row comes from `gpu_guard.py::_visible_index`, which parses the first entry of `CUDA_VISIBLE_DEVICES` and returns zero for an unset or empty variable, a negative index, or a UUID spelling; `gpu_memory` falls back to the first row when that index is past the end of the table. The source is re-selected per sample, so one run's series can mix the two. If neither answers, `train.py::Modeller._sample_gpu_util` sets `_gpu_util_off`, prints that there will be no `gpu/util_*` metrics for this run, and stops; `_gpu_util_off` is not cleared anywhere.

`train.py::Modeller._start_gpu_util_thread` runs the sampler on a daemon thread, started from `train()` only. The thread ticks at a quarter of `cfg:gpu_util_sample_period_s`, and the gate inside `_sample_gpu_util` on that same period fixes the spacing; that gate enforces a minimum spacing and never a maximum, so a slow `nvidia-smi` call stretches the interval a sample covers. `train.py::Modeller._announce_gpu_util_source` prints once, before the thread starts, naming which sensor answered, which device or `nvidia-smi` row it read, the period and both window lengths.

Readings are appended as `(timestamp, percent)` pairs to a deque whose `maxlen` is `train.py::Modeller._gpu_util_capacity`: the wider window length over the sampling period, with 25% headroom, floored at 512, so capacity follows the period. Every reader goes through `train.py::Modeller._gpu_util_samples`, which copies the deque under a lock, a deque at `maxlen` popping on append.

## The two windows are one deque

`train.py::Modeller._gpu_util_mean(window_s)` selects the samples inside the trailing window and returns their unweighted arithmetic mean. Timestamps select and never weight, so a sample separated from its neighbour by the whole window counts as much as one separated by a period. Two admission bounds apply together: fewer than five samples returns `None`, and so does a set spanning less than `train.py::_UTIL_MIN_SPAN_S`, 60 seconds. `None` means no reading, and the metric is then absent from the report.

`gpu/util_recent` is `_gpu_util_mean(cfg:gpu_util_window_s)` and `gpu/util_policy` is `_gpu_util_mean(cfg:gpu_util_policy_window_s)`, both emitted from the ten-step report. They are two window lengths over one deque filled by one function: where the deque holds fewer samples than either window would span, both select the same samples and report the same number. No sample count is published beside either metric. Samples already collected keep being averaged after the sensor goes inert, so both report a stale mean until those samples age out, then vanish.

## What a reading includes and leaves out

The counter is device-wide, not per-process: neither source excludes another job's kernels, so a co-tenant's work on the card is inside the reading. Host idle imposed from outside the process is inside it the other way, since seconds in which this run waits on the node rather than launching kernels enter $h$ and lower the ratio with no change in $g$. The reading is also blind to what is not sampled: the sampler runs on wall clock and knows nothing of the step body, so a stretch of low occupancy falling between two sample times does not enter the mean.

## Where the reading is consumed

There are two consumers: the metric stream above, and `train.py::Modeller.select_batch_size`, which the training loop calls only when `cfg:grow_batch_size` is true, and which holds the base batch without reading occupancy when `cfg:batch_util_target` is at or below zero, when the sensor is off, or when there is no headroom above the base. That key is a fraction of the card, converted to percent once at the single read site. The sizer reads raw per-rung samples out of `_gpu_util_samples` during a calibration dwell, requiring at least `train.py::_BS_MIN_UTIL_SAMPLES` of them spanning `_UTIL_MIN_SPAN_S`, and reads `_gpu_util_mean(cfg:gpu_util_policy_window_s)` once for its stand-down audit; it never reads the published windows as a control input. The selection rule, the ladder and the audit are on [batch-size](batch-size.md). `config_invariants.py::util_target_actuable` errors when a target is set outside $(0, 1]$, with `cfg:grow_batch_size` false, or with `cfg:max_batch_size` at or below `cfg:batch_size`.

`utils.py::_RETIRED_KEYS` records `gpu_util_floor`, deleted 2026-08-13, which grew the batch whenever the windowed mean fell below a threshold; a config carrying it fails at load with that record as the message.

## Launch count and `compile_policy`

`cfg:compile_policy` decides how many kernel launches a rollout step issues, the $h$ side of the ratio. `train.py::Modeller.maybe_compile_policy` reads it as `false` (the default, eager), `true`, `'auto'` or `'step'`; `'auto'` and `'step'` enable only on Linux with CUDA, inductor not supporting CUDA on native Windows, so one config resolves to eager on a Windows dev box and compiled on a Linux node. Under `true` or `'auto'` the trunk submodules `t_model`, `s_model`, `forward_policy`, `backward_policy` and `flow_model` are compiled in place on the train and EMA models; under `'step'` the trunk list is empty and `models/gfn.py::GFN.compile_step_kernels` compiles the per-timestep bodies whole. The conditioner is never compiled. Compilation is lazy at first forward and `torch._dynamo.config.suppress_errors` is set, so a backend failure degrades to eager with a console warning; `cache_size_limit` is raised to 24, every distinct batch size being a distinct shape, and `torch._functorch.config.donated_buffer` is set false, a compiled backward with donated buffers raising on any backward taken with `retain_graph=True`.

## Utilisation against throughput

Utilisation is a ratio and throughput is a rate, and they are not monotonically related. Removing device work lowers $g$; if that lowers $g$ by a larger factor than it lowers $g + h$, the step is faster and the ratio falls. Removing host work lowers $h$ alone: faster step, higher ratio. Widening the batch raises $g$ against a fixed launch count, so the ratio rises while samples per second may rise or fall. The trainer publishes the rate separately as `samples_per_sec` and `updates_per_sec`, the latter `samples_per_sec` over `batch/accum_target`; none of the three is a function of the others.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:gpu_util_sample_period_s`, `cfg:gpu_util_window_s`, `cfg:gpu_util_policy_window_s`, `cfg:batch_util_target`, `cfg:compile_policy`, `cfg:grow_batch_size`, `cfg:batch_size`, `cfg:max_batch_size`, `cfg:batch_growth_interval`, `cfg:fused_grad_accum_min_samples`.

Code: `train.py::Modeller._read_gpu_util`, `._sample_gpu_util`, `._gpu_util_capacity`, `._gpu_util_samples`, `._start_gpu_util_thread`, `._announce_gpu_util_source`, `._gpu_util_mean`, `.select_batch_size`, `.maybe_compile_policy`; `train.py::_UTIL_MIN_SPAN_S`, `_BS_MIN_UTIL_SAMPLES`; `gpu_guard.py::gpu_memory`, `._visible_index`; `models/gfn.py::GFN.compile_step_kernels`; `config_invariants.py::util_target_actuable`; `utils.py::_RETIRED_KEYS`.

## Could be tooling

Two of the facts above are checkable rather than described. The first is the sample count behind a published mean: `_gpu_util_mean` already holds the selected list and could emit its length and span beside the value, which makes an inert or stale sensor visible at the row rather than by its later disappearance, and makes it readable whether the two windows hold the same samples. The second belongs at config-generation time: from `cfg:gpu_util_sample_period_s`, the two window lengths and an expected step time, print the maximum samples either window can hold, the step time above which each falls under the five-sample bound, and whether `cfg:batch_growth_interval` steps spans `_UTIL_MIN_SPAN_S` of wall clock.

## Sources

The code above, read at the stamped commit, and the comments on these keys in the canonical config. Repo: docs/design/phase6_measurement_request.md section 2, docs/design/phase6_handoff.md sections 2 and 3.3. Memory files located the code and were not used as evidence: project_util_policy_overstates_and_batch_is_not_the_lever, project_uma_optimisation_lowers_utilization, project_gpu_util_sensor_needs_smi_fallback, project_phase6_measurement_request, project_low_util_kill_is_node_contention_gpu_busy_constant, project_compile_policy_cluster_verification, project_policy_rollout_is_dispatch_bound.
