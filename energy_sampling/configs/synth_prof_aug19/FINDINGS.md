# Profiling run — what the instrumentation says, read for the first time

*2026-08-19, dev box (RTX 5080 laptop, `torch.compile` OFF on Windows), ELJ
mipcas sg2/zp1 warm-started at step 430, **fixed batch 1000** (ladder
deliberately off), 500 steps. `OBSERVED`: one run, one box.*

Both profiling layers ship `enabled: false` and had never been read on a real
run. This turned both on. **No timing number here transfers to the A100** — the
rollout is dispatch-bound locally and inductor is off. What transfers is the
*shape* of where the time goes, and one specific mechanism.

---

## 1. The validation the module asked for FIRST — and the prediction failed

`profiling.py`'s docstring says the first thing worth doing is checking
`energy/seconds_in_step` against CUDA events, because a wall-clock subdivision of
an async step can be confidently wrong while the step total it sums to is exactly
right.

**Predicted before the run: `energy/gpu_over_wall` < 1** — ELJ is a cheap energy
on a dispatch-bound box, so device time should be a fraction of the wall window.

**Measured: 1.002** (n=50 reports, min 0.987, max 1.019).

So the prediction was wrong and the news is good: over the energy region the
device timeline and the host timeline agree to within ±1.5%. **`energy/frac_of_step`
and `energy/seconds_in_step` are honest** — no async spillover is being
attributed to a later region. That is the metric every MLIP optimisation decision
leans on, and it now has a measurement behind it rather than an assumption.

**Stated precisely, because the number is easy to over-read.** CUDA events
measure the interval between the two events *executing on the stream*, which
includes any stream idle inside the region. `gpu_over_wall ≈ 1` therefore means
*the region's device work completes within the region's wall window* — no
mis-attribution. It does **not** mean the GPU was busy for 100% of it.

## 2. The step is host-bound, roughly 4:1

From the bounded `torch.profiler` window (8 steps from step 630):

```
Self CPU time total:   6.013 s      (751 ms / step)
Self CUDA time total:  1.496 s      (187 ms / step)
```

**Caveat that limits how hard this can be pushed:** `torch.profiler` inflates the
CPU side, so 4:1 is an upper bound on host-boundedness rather than an estimate of
it. The direction is not in doubt and it agrees with the standing dispatch-bound
result (~937 `nn.Module` calls per training step; widths 64/256/512 costing the
same). The magnitude is not trustworthy from this instrument.

Where the CUDA time actually goes is unremarkable and healthy — it is matmuls:
`aten::mm` 556 ms (37.2% of self CUDA, 18 470 calls at 30 µs), `aten::addmm`
236 ms, plus the magma/cutlass sgemm kernels behind them. There is no surprising
kernel eating the GPU.

## 3. THE LEAD: ~11 600 device→host syncs per training step, and most are Adam's

The single largest CPU line in the op table:

| op | calls (8 steps) | per step | CPU total |
|---|---:|---:|---:|
| `aten::item` | 92 618 | **~11 577** | 1.112 s (**18.5%**) |
| `aten::_local_scalar_dense` | 92 618 | ~11 577 | 1.039 s (17.3%) |
| `Memcpy DtoH (Device → Pinned)` | 91 564 | ~11 445 | 48.9 ms |

Every one of those is a **device→host synchronisation**: the host blocks until
the device reaches that point. This is precisely the "host gating" that handoff
§3.3 identifies as the thing whose removal *raises* occupancy.

**Attribution, measured rather than grepped.** The profiler was run with
`with_stack: false`, so it counts these without saying where they come from. Two
probes settled it:

1. **The ELJ energy call makes ZERO of them** — patching `Tensor.item` around one
   real `analyze(['reduction_en','elj'])` on 200 graphs tallied 0. So they are in
   the training loop, not the energy backend.
2. Patching `Tensor.item` around a real short training run and tallying callers:

```
      1157  adam.py:755 in <listcomp>      <-- 70% of Python-visible calls
      1157  adam.py:758 in <listcomp>          (two lines, same cause)
       175  molecular_crystal.py:1043 in condition_samples
       157  molecular_crystal.py:898  in init_blank_crystal_batch
        81  train.py:3123 in _update_rolling
```

`adam.py:753-758` is PyTorch's own multi-tensor Adam:

```python
bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
```

`_get_value(step)` is `step.item()`. So the `foreach` Adam path performs **two
device→host syncs per parameter tensor per optimizer step** — ~154 syncs per
step here — and it is framework code, not ours.

**The standard remedy is a constructor argument.** `torch.optim.Adam(...,
fused=True)` (CUDA) keeps the step counts on device and removes the syncs
entirely; `capturable=True` does the same for a different reason. All four
optimizers (`train.py:2031, 2043, 2048` and `1449`) are constructed with neither.

**NOT CHANGED, and deliberately.** Fused Adam is not bitwise identical to the
foreach path, so flipping it changes the numerics of every run — that is an
owner decision under *schema may change freely, behavior may not*, not a defect
fix. It is written here with its evidence so the decision can be made cheaply.
The honest way to take it would be the tier-C harness: same config, fused on and
off, compare against the measured same-config spread.

**The Python-visible tally does not account for all 11 577.** The probe caught
~220/step at the Python level against the profiler's ~11 577 at the ATen level,
so most `aten::item` calls originate below Python (C++/autograd internals, or
ops that lower to `_local_scalar_dense`). Attributing the remainder needs a run
with `with_stack: true`, which is exactly what that knob's config comment says
it is for. That is the obvious next probe and it was not run.

## 3a. Fused Adam, adopted and MEASURED — and the isolated benchmark misled

> **RETRACTED 2026-08-20 by the cluster A/B.** The end-to-end numbers below
> (-10.4% step time, +10.2% occupancy) did NOT replicate. `cluster_aug19`'s C1
> pair, same config, 3000 steps each, measured `batch/med_step_s` 0.538 with
> fused against 0.533 without, occupancy 36.2 against 40.9 and samples/sec 1609
> against 1618 -- fused marginally WORSE on all three. The order confound named
> below as the reason not to bank it (fused ran first, foreach second,
> back-to-back) is the likely whole effect, and the order-reversed replicate was
> never run.
>
> **What survives:** the sync count (2314 -> 136, -94%) and the instrument
> lesson -- an isolated `optimizer.step()` loop cannot measure a sync-removal
> change, because it synchronises anyway. Do not quote the throughput numbers.


`fused=True` on all four optimizers (`train.py`, CUDA-guarded, `MXT_FUSED_ADAM=0`
forces the old path so the change stays A/B-able). Adopted with bit-identity
knowingly given up (user, 2026-08-19).

**It took, verified by measurement rather than by the flag:** Adam's `.item()`
calls fell **2314 -> 136** (-94%) over a real short run, and the two arms print
`Adam fused=True` / `fused=False`.

**Two measurements of the same change disagreed, and the disagreement is the
lesson.**

| | isolated `optimizer.step()` loop | end-to-end A/B, 301 steps each |
|---|---:|---:|
| verdict | 0.960 vs 0.945 ms, **1.6% = noise** | **faster and better occupied** |

End-to-end, fused vs foreach:

| metric | fused | foreach | delta |
|---|---:|---:|---:|
| `batch/med_step_s` | 0.577 | 0.644 | **-10.4%** |
| `gpu/util_recent` | 29.2 | 26.5 | **+10.2%** |
| `samples_per_sec`, `updates_per_sec` | 1584 | 1506 | +5.2% |
| `train_step_time` median | 0.674 | 0.689 | -2.3% |

**Why the microbenchmark was the wrong instrument, stated so it is not repeated:**
it wraps `optimizer.step()` in `torch.cuda.synchronize()` to time it. A host sync
costs nothing in a loop that synchronises anyway -- the cost of a sync is *the
next step's work not being queued while the host waits*. So the benchmark
measured the one condition under which the defect is free. A change that removes
host stalls can only be measured with the pipeline intact.

**NOT yet banked, and the reason is order.** Both arms ran back to back, fused
first, so any drift lands entirely on the second arm. `train_step_time`'s spread
(stdev ~0.15) also swamps its own -2.3%; the -10.4% is more trustworthy because
`batch/med_step_s` is already a 20-step median, and `gpu/util_recent` rests on
~6 samples per arm. **An order-reversed replicate is queued** (`*_r2`, foreach
first): if the effect survives the swap it is not drift.

## 4. A practical limit on the trace layer

The chrome trace for **8 steps** is **748 MB** (`record_shapes` and `with_stack`
both false). The canonical config's comment warns the trace "grows too large to
open"; this is that warning with a number on it. The companion op table is 8.8 KB
and is where the content above came from — on this codebase the **table is the
usable artifact and the JSON is mostly unopenable**. Budget accordingly, and
prefer shrinking `active_steps` over trimming anything else.

## 5. What this run did NOT establish

- Nothing about the A100: no compile, different dispatch cost, different ratio.
- Nothing about MLIP routes — this is ELJ. The MACE/UMA energy split is
  `energy/*` territory and MACE does not fit on this card in a fused stage at all.
- No claim that removing the Adam syncs would raise occupancy by any particular
  amount. The mechanism is established; the size of the prize is not.
