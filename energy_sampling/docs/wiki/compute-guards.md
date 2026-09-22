# Compute guards

*Drift: **C** (code-bound). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A *compute guard* is a mechanism that keeps a run inside the machine it is on: it refuses a launch, caps an allocation, cuts a size after a failure, or trades memory for time. This page takes them in launch order. What a batch size costs and how the sizer chooses one is [batch-size](batch-size.md); the occupancy sensors are [gpu-occupancy](gpu-occupancy.md); job submission is [cluster-operations](cluster-operations.md).

## The pre-flight guard

`train.py`'s `__main__` calls `gpu_guard.py::require_free_gpu` after parsing args and before `Modeller` is constructed, so nothing in the process has touched CUDA when the check runs; `GPUBusy` becomes `SystemExit` there. The default is one run per card: `gpu_guard.py::check` blocks when more than `cotenants - 1` other training runs are present, `cotenants` being 1 unless `GFN_COTENANTS` or an explicit argument raises it, and a co-tenancy above 1 is additionally checked against arithmetic, `cotenants * need` versus the card's total.

What it sees is other *training* processes: `gpu_guard.py::training_processes` counts a python process whose argv holds a token whose basename is in `TRAIN_ENTRYPOINTS` (`train.py`, `train_conformer.py`), excluding this process's own lineage and collapsing a venv launcher stub with its child into one run. What it does not see is per-process GPU memory, which `nvidia-smi` reports as `[N/A]` under Windows WDDM; other tenants therefore enter only through aggregate free memory, read for this process's visible device (`_visible_index`).

`gpu_guard.py::project_need_mb` has two bases in order: a measured peak for this config's exact `config_signature` from `.vram_registry.json`, written by `train.py::Modeller.record_vram_peak` in the eval block, once the step index passes `max(50, cfg:eval_period)`, or `declared_ceiling_mb`, `cfg:cuda_memory_fraction` times the card's total. There is no parametric model of activation memory, and a measured peak is never scaled across batch sizes; the signature carries the `cfg:traj_checkpoint` regime, `cfg:eval_num_samples` and `cfg:buffer_device`. The room check itself runs only when other tenants are present, when `cotenants > 1`, or when the projection is a measurement, and compares a measured peak raw.

The check is skipped entirely when `no_gpu_visible` is true (`CUDA_VISIBLE_DEVICES` empty or all `-1`), when `GFN_GPU_GUARD` is falsy, or under a batch scheduler (`SLURM_JOB_ID` and four siblings). A truthy `GFN_ALLOW_GPU_SHARING` does not skip it: the block is computed and printed, then downgraded to a warning.

## The per-process cap

`train.py::Modeller.__init__` calls `torch.cuda.set_per_process_memory_fraction(cfg:cuda_memory_fraction, device=0)` and then `torch.cuda.init()`, so the cap is in place before the context exists; with no CUDA available both calls are skipped and the run says so. At import, `train.py` sets `PYTORCH_CUDA_ALLOC_CONF` to `expandable_segments:True` by `setdefault`, a floor the environment can override.

The cap bounds *this process* and counts reserved memory, including allocator cache the run cannot reuse (`Modeller.vram_ledger`, `vram_metrics`). It bounds nothing outside the process: a second training run, the desktop compositor, or any other tenant of the card. `gpu_guard.py` records three BSODs on 2026-08-11/12 and states that the driver there does not raise on oversubscription but takes the machine down. The refusal sits before CUDA initialisation rather than in a handler.

## OOM inside the step loop

The step body wraps `train_step` in `except (RuntimeError, ValueError)` and routes to `train.py::Modeller.handle_train_epoch_error`; `FrozenTrainingState` subclasses `RuntimeError` and is re-raised above that clause. The handler prints the error, re-raises anything `utils.py::is_cuda_oom` rejects, increments `batch_oom_events`, and returns without acting at step 0. Otherwise it zeroes every optimizer's gradients, clears `fused_accum_count`, and collects.

The rest branches on membership of the step type in `protocol.py::TRAIN_MODES`, which is `('bwd', 'fused')`: a `fwd` or `replay` step type takes the other branch. On the train branch, the failing size is folded by minimum into `batch_size_oom_min` and `batch_size_oom_ceiling` with the ceiling's expiry clock restamped, `batch_size` is multiplied by `cfg:oom_batch_shrink_factor`, a cooldown of `cfg:oom_cooldown_steps` is set, and the sizer conclusion and step-time windows are dropped; a cut reaching 1 raises `RuntimeError("Cascading OOM Failure")`. Otherwise, which covers the eval call sites `eval_fwd`, `eval_bwd` and `anchor_refresh`, neither the ceiling nor the train batch moves: `eval_batch_cap` alone is cut, and an OOM at a draw of 1 raises. `Modeller.eval_draw_size` returns `min(batch_size, eval_batch_cap)`, so the eval draw starts at the train batch and only falls; `reset_eval_draw_size` clears the cap, and although its docstring says a stage transition rebuilds the size through it, the method has no call site in the package.

Stage transitions are outside this handler. `protocol.py::StageProtocol.advance` runs a stage's `on_enter` actions from inside `evaluation()`, which the step loop calls after the try block closes, and neither `advance` nor `_run_action` carries OOM handling, so an allocation failure in a transition action reaches no batch cut.

## The energy function's own recovery

`energies/molecular_crystal.py::MolecularCrystal.batched_analyze_crystal_batch` reads `cfg:energy_config.internal_oom_recovery`, overridable per call. False returns early to a single `analyze_crystal_batch` on the whole batch, so an OOM propagates to the caller and reaches the handler above. True runs a chunking loop over a sticky `self.batch_size`, created lazily at 1000 on the `uma` and `mace` routes and 10000 otherwise, grown by 1% (at least 1) per chunk while it is below `n_samples` and below 100000 and has not OOMed in this call, and cut to 0.65 of itself on an OOM, with an assertion at a chunk size of 1. Each cut prints `OOM in energy evaluation: dropping chunk size to ...` and collects. The canonical config sets the key false, and the whole-dataset init scans pass `internal_oom_recovery=True` explicitly.

## Wall clock, recompute and compilation

`cfg:max_step_seconds` is a ceiling on the median of the last 20 recorded step times, which exists only once at least 10 are recorded; it is checked inside `Modeller.select_batch_size` before the growth walk, and so only while `cfg:grow_batch_size` is on. The cut is proportional to the overshoot, is floored at the accumulation target on a fused stage, and stands down for the stage once a cut is measured not to have moved the median ([batch-size](batch-size.md)).

`cfg:traj_checkpoint` gradient-checkpoints each trajectory step: `models/gfn.py::GFN._run_step` wraps a pure step function in `torch.utils.checkpoint` with `use_reentrant=False` when `GFN._use_traj_checkpoint` returns true, which requires the flag, `torch.is_grad_enabled()`, and membership of the branch (`fwd`, `bwd`, `replay`) in `cfg:traj_checkpoint_modes` when that list is set. Trajectory activation memory becomes order 1 in $T$ instead of order $T$, and the forward of each checkpointed step is recomputed during the backward pass. The step functions are pure in their inputs, with all randomness pre-drawn at loop level and passed in, so the backward recompute replays them exactly; `tests/crystal/test_periodic_scoring.py::test_traj_checkpoint_and_grads` asserts the rollout states and both log-probability sequences are bitwise equal to the flag off under the same seed, and that the gradients reaching both policies are finite. What it does not remove: peak memory is the worst step's, so a large backward-branch allocation can leave no contiguous block for the next rollout. Both flags are runtime attributes set in `Modeller.init_gfn`, outside `gfn_config`, so they enter neither checkpoint nor problem identity; the stage action `set_traj_checkpoint` moves the flag at a transition, clearing `traj_checkpoint_modes` when it turns checkpointing on.

- **[calibration, bench, 2026-07-24]** 33.6x less trajectory activation memory at $T = 100$, at 1.7x step time.
- **[calibration, tck_sep10, 2026-09-13]** `[fwd]` 20% faster per step than checkpointing every branch, batch 1600 on mipu, values identical.

`cfg:compile_policy` takes false, true, `auto` or `step`. Under `auto` and `step`, `Modeller.maybe_compile_policy` enables only when `platform.system() == 'Linux'` and CUDA is available, so on native Windows both resolve to eager and every compile-dependent path is unreachable there. When enabled it sets `torch._dynamo.config.suppress_errors = True`, raises `cache_size_limit` to 24, sets `torch._functorch.config.donated_buffer = False` (donated buffers hard-raise on the `retain_graph` backward the fused gradient-geometry diagnostic takes), and compiles the trunk submodules in place, or the per-timestep bodies through `GFN.compile_step_kernels` in `step` mode. Any exception degrades to eager with a printed warning, and each distinct batch size is a recompile.

`cfg:anomaly_detection` is applied once, before the step loop, by `Modeller.set_detect_anomaly`: `torch.autograd.set_detect_anomaly(True)` plus a hook on every `requires_grad` parameter of `gfn_model` raising `RuntimeError(f"NaN/Inf gradient in {name}")` on a non-finite gradient. The config notes a dramatic slowdown when true.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:cuda_memory_fraction`, `cfg:anomaly_detection`, `cfg:compile_policy`, `cfg:traj_checkpoint`, `cfg:traj_checkpoint_modes`, `cfg:max_step_seconds`, `cfg:oom_batch_shrink_factor`, `cfg:oom_cooldown_steps`, `cfg:batch_oom_ceiling_retest_steps`, `cfg:grow_batch_size`, `cfg:eval_period`, `cfg:eval_num_samples`, `cfg:buffer_device`, `cfg:energy_config.internal_oom_recovery`, `cfg:stage.on_enter` (`set_traj_checkpoint`). Environment: `GFN_GPU_GUARD`, `GFN_ALLOW_GPU_SHARING`, `GFN_COTENANTS`, `CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF`.

Code: `gpu_guard.py::require_free_gpu`, `.check`, `.training_processes`, `._visible_index`, `.no_gpu_visible`, `.config_signature`, `.declared_ceiling_mb`, `.project_need_mb`, `.record_peak`, `._skip_reason`; `train.py::Modeller.__init__`, `.handle_train_epoch_error`, `.eval_draw_size`, `.reset_eval_draw_size`, `.select_batch_size`, `.maybe_compile_policy`, `.set_detect_anomaly`, `.init_gfn`, `.vram_ledger`, `.record_vram_peak`; `train.py::FrozenTrainingState`; `utils.py::is_cuda_oom`; `models/gfn.py::GFN._use_traj_checkpoint`, `._run_step`, `.compile_step_kernels`; `energies/molecular_crystal.py::MolecularCrystal.batched_analyze_crystal_batch`; `protocol.py::StageProtocol.advance`, `._run_action`, `TRAIN_MODES`.

## Could be tooling

Two checks here are mechanical. A launch-time report: given a config path, print the `config_signature`, whether the registry holds a measurement for it, which basis `project_need_mb` would use, and whether the room check would run. And a call-graph check for the unprotected region: walk from `StageProtocol.advance` through `_run_action` and flag every reachable function that calls the energy function or draws a batch, since those are the allocations no handler covers.

## Sources

The code above, read at the stamped commit, and the compute and energy blocks of the canonical config with their comments. Eight memory files (project_gpu_preflight_guard, project_transition_oom_is_fatal, project_internal_oom_recovery_semantics, project_traj_checkpoint_refactor and others) located the code and were not used as evidence.
