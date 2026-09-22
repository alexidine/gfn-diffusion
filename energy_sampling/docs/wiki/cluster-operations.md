# Cluster operations

*Drift: **C** (code-bound). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A *battery* is a directory under `configs/` holding one YAML file per *arm*, a tab-separated index, and one SLURM submission script. This page is about what that directory contains once written, how it reaches the cluster, and what the cluster does with it. How the YAMLs are derived from the canonical config is [battery-generation](battery-generation.md); what a checkpoint holds is [checkpoints-and-resume](checkpoints-and-resume.md).

## The submission artefacts

A generator writes three things beside its arm YAMLs: `joblogs/` with a `.gitkeep` inside it, an index, and `submit_<battery>.sbatch`. The `.gitkeep` text says it ships the directory to the cluster because SLURM cannot create the directory named in `--output`.

The script is formatted from one template, `configs/final_sep19/make.py::SBATCH`, among whose substituted fields are the battery name, the tag, the wall time and the last array index. Its header at `a637e70`, each directive beside the value the template wrote into one battery's script:

| directive | value in `configs/prod_sep20/submit_prod_sep20.sbatch` |
|---|---|
| `--time` | `2-00:00:00`, from `configs/final_sep19/make.py::WALL` |
| `--gres` | `gpu:a100:1` |
| `--mem`, `--cpus-per-task`, `--tasks-per-node` | `48G`, `8`, `1` |
| `--array` | `0-7`, written as `0-{last}` with `last` the arm count minus one |
| `--account`, `--job-name` | `torch_pr_226_chemistry`, the battery tag (`p20`) |
| `--output` | an absolute cluster path into the battery's `joblogs/`, named `%x_%A_%a.out` |
| `--mail-user`, `--mail-type` | the owner's address, `END,FAIL` |

The array index is a *row* selector, not an arm name: `ROW=$((SLURM_ARRAY_TASK_ID + 2))` indexes `INDEX_a.tsv` past its header, and `awk` pulls the arm name, the warm-start source arm, the prior filename and the prior's byte size out of that row. A comment in the generated file states that `--array` is rewritten by `make.py`.

Paths are absolute and cluster-side throughout: `configs/mle_w3_sep16/make.py::CLUSTER_CKPTS` and `::CLUSTER_DATA` are the two roots, and `::_scan_local_paths` raises at generation time on any config string beginning `c:`/`d:` or containing a backslash.

## What the script does before `train.py` runs

Each array task, in order:

1. **Sentinel.** If `{CKPTS}/{ARM}.dead` exists, the task prints that the arm aborted UNRECOVERABLE on an earlier leg and exits 0.
2. **Cross-repo guards.** It greps `MXtalTools/mxtaltools/common/sym_utils.py` for `MONO_CLASS`, exiting `FATAL` if absent, and branches on which of three states that file is in to decide whether to export `MXT_NIGGLI_TRICLINIC=1` ([cell-fundamental-domains](cell-fundamental-domains.md)).
3. **Data guard.** `stat -c %s` on the prior file must equal the byte count in the index row.
4. **Resume or seed.** `{CKPTS}/*{ARM}_*_running.pt` if it exists; otherwise the source arm's `*_best.pt`, where anything other than exactly one match is `FATAL`.
5. **Prior model.** The arm's own `*_prior.pt` if one exists, else the source arm's, else the literal `null`.
6. **Substitution.** `sed` replaces `WARM_CHECKPOINT_PLACEHOLDER` and `PRIOR_MODEL_PLACEHOLDER`, writing a *resolved* YAML into `joblogs/`; a surviving placeholder is `FATAL`.

The launch is `srun singularity exec --nv` with a read-only overlay, two `--bind` mounts, `--pwd` at `energy_sampling`, `PYTHONPATH` prefixed with both repository roots, an inline `python -c` assertion that `mxtaltools.common.sym_utils.NIGGLI_TRICLINIC` is true, and `python -u train.py --config ${RESOLVED}`, piped through `tee` into `joblogs/<arm>_<jobid>.trainlog`. After the pipe, `grep -q UNRECOVERABLE` on that log creates the `.dead` sentinel.

## Two repositories, both at the intended commit

`PYTHONPATH` is set to `${PROJECT_ROOT}/MXtalTools:${PROJECT_ROOT}/gfn-diffusion`, so the arm imports from two working copies pulled independently. The generator reads its base config from `git show HEAD:energy_sampling/configs/mk_dev.yaml` (`configs/final_sep19/make.py::committed_mk_dev`) and warns with the output of `configs/mle_w3_sep16/make.py::dirty_files`, which lists `configs/mk_dev.yaml` and every modified or untracked `.py` outside `configs/`, `tests/` and `SCRATCH/`. The `MONO_CLASS` grep and the inline `NIGGLI_TRICLINIC` assertion both read the *second* repository's working copy, and both exit the task before the step loop begins.

## wandb logging

`train.py::Modeller.train` opens `wandb.init(project="GFN Energy", config=..., name=self.run_name, tags=[self.args.tag])` with no `mode` argument, so the mode comes from the environment and nothing in the submission script sets it. `WANDB_MODE` is set in exactly two places in the repository, both outside training and both to `disabled`: `bench/calibrate_noise.py` and `tierc_smoke.py`.

`train.py::Modeller._log_metrics` catches every exception from `wandb.log`, prints on the first failure only, publishes the cumulative count as `run/log_failures`, and raises after `train.py::Modeller._MAX_LOG_FAILURES` (200) consecutive failures. The comment beside the batch-controller series in `train.py::Modeller.ten_step_reporting` states that wandb uploads no console log at all for a run left in state `crashed`, while history series survive a hard kill.

`cfg:eval_period` gates the whole evaluation block and `cfg:figs_period` gates `do_figs` inside it; `configs/prod_sep20/p20_mip_n5.yaml` sets 500 and 1000 against the canonical config's 250 and 500, whose comment states that `figs_period` must be a multiple of `eval_period` or it will not fire. `config_invariants.py::figs_period_fires` raises that as an `ERROR` violation, and `utils.py::_report_config_invariants` prints violations at load without raising; its docstring records 47 of 2,244 configs in the tree with a non-multiple `figs_period`.

## What a live log looks like

The step loop is `trange(init_step, self.args.epochs + 1)`, and tqdm rewrites one line with a carriage return, so a `cat` of the `.out` file shows the bar parked at the last completed iteration whether or not the process is still stepping. Two code paths advance that bar while the run makes no progress: a non-finite-gradient streak, which returns before the optimizer steps and aborts only at `cfg:nonfinite_abort_streak` consecutive steps, and the rewind loop, which reloads state and continues. Both end in `train.py::FrozenTrainingState` with a message beginning `UNRECOVERABLE`, the string the submission script greps for.

Artefacts with independent clocks sit beside the log. `joblogs/<arm>_<jobid>.info` is written once at start from `nvidia-smi -L`, `scontrol show job` and the node list. `joblogs/<arm>_<jobid>_smi.csv` is an `nvidia-smi --query-gpu=... -l 10` sidecar started under `stdbuf -oL` before `srun` and killed in an `EXIT`/`TERM` trap, so its mtime advances every ten seconds for as long as the *job* exists, independent of the Python process. The same trap writes `joblogs/<arm>_<jobid>_sacct.txt`. `train.py::Modeller.vram_ledger` prints at named points through initialization.

## The occupancy reading and the cancellation

Two occupancy readings exist, sampled on different clocks. In-process, a daemon thread started by `train.py::Modeller._start_gpu_util_thread` samples every `cfg:gpu_util_sample_period_s` seconds, and `train.py::Modeller._gpu_util_mean` publishes trailing means as `gpu/util_recent` over `cfg:gpu_util_window_s` and `gpu/util_policy` over `cfg:gpu_util_policy_window_s`; a window with fewer than five samples, or spanning less than `train.py::_UTIL_MIN_SPAN_S` (60 s), returns `None`, which is absence of a reading. `train.py::Modeller._announce_gpu_util_source` prints once which sensor answered and which card it read: `torch.cuda.utilization()` reads the already-remapped current device, while the `nvidia-smi` fallback selects a row of a node-wide table and does the `CUDA_VISIBLE_DEVICES` remap itself (`gpu_guard.py::_visible_index`).

Out of process, wandb's `system.gpu.0.gpu` stream and the `_smi.csv` sidecar sample on a wall clock that knows nothing about the training loop. The comment at the publication site records that the scheduler judges the out-of-process number, not `gpu/util_policy`, and `cfg:gpu_util_policy_window_s` defaults to 7200 s to match the window the cancellation is averaged over. The cancellation line on a trailing device-utilisation mean of about two hours is **[calibration, prod_sep02, 2026-09-07]** 54 to 55 percent. Utilisation multiplied by step time is a per-run constant across that run's fast and slow regimes to within **[calibration, prod_sep02, 2026-09-07]** 5 to 9 percent coefficient of variation. What the percent itself measures is [gpu-occupancy](gpu-occupancy.md); reading a kill off a log is [failure-signatures](failure-signatures.md).

## Multi-leg jobs

`--time` bounds a single leg: an arm needing more than one leg is continued by resubmitting the same file, whose resume branch then finds `*{ARM}_*_running.pt` and whose prior-model lookup prefers the arm's own snapshot over its source arm's. `cfg:epochs` bounds the absolute step counter, so a leg whose restored step already exceeds it runs zero iterations and exits through the finish path. Buffers ride in a sidecar written at eval cadence, so a leg restores buffers up to `cfg:eval_period` steps staler than its weights ([checkpoints-and-resume](checkpoints-and-resume.md)).

Resubmission is per array, not per arm: the same `--array=0-{last}` range re-enters every row, and the only per-arm suppression is the `.dead` sentinel.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:run_name`, `cfg:tag`, `cfg:epochs`, `cfg:eval_period`, `cfg:figs_period`, `cfg:checkpoints_dir`, `cfg:checkpoint_name`, `cfg:prior_model_name`, `cfg:nonfinite_abort_streak`, `cfg:gpu_util_window_s`, `cfg:gpu_util_sample_period_s`, `cfg:gpu_util_policy_window_s`.

Code: `train.py::Modeller.train`, `._log_metrics`, `._MAX_LOG_FAILURES`, `._start_gpu_util_thread`, `._sample_gpu_util`, `._gpu_util_mean`, `._gpu_util_capacity`, `._announce_gpu_util_source`, `.vram_ledger`, `.ten_step_reporting`; `train.py::_UTIL_MIN_SPAN_S`, `train.py::FrozenTrainingState`; `gpu_guard.py::_visible_index`; `utils.py::_report_config_invariants`; `config_invariants.py::check`, `::figs_period_fires`; `configs/final_sep19/make.py::SBATCH`, `::WALL`, `::committed_mk_dev`; `configs/mle_w3_sep16/make.py::dirty_files`, `::_scan_local_paths`, `::CLUSTER_CKPTS`, `::CLUSTER_DATA`; `configs/prod_sep20/submit_prod_sep20.sbatch`, `configs/prod_sep20/INDEX_a.tsv`, `configs/prod_sep12/submit_prod_sep12.sbatch`.

## Could be tooling

Several of the checks above are string comparisons a script could make before anything is submitted. A preflight could resolve each index row the way the script does and report what it would find: which checkpoint the glob selects and whether it is ambiguous, whether the prior file's size matches, which prior model the arm would be handed, whether a placeholder survives. A two-repository commit stamp could go in the resolved YAML, so the pair of HEADs behind a wandb run is readable from the run.

## Sources

The code above, read at the stamped commit; `configs/prod_sep20/` and `configs/prod_sep12/` as written in the tree at that commit; the canonical config's evaluation and occupancy blocks; `docs/reading_runs.md` section 8; `configs/prod_sep02/analysis/low_util_cancellation/REPORT.md` for the two tagged calibrations. Memory files located the code and were not used as evidence: project_cluster_push_and_submit_workflow, feedback_live_log_looks_like_a_dead_job, project_low_util_kill_is_node_contention_gpu_busy_constant, project_prod_sep02_battery, project_prod_sep12_phase2_from_mle09_best, feedback_cluster_eval_fig_periods, project_epochs_is_absolute_and_resumes_can_start_past_it.
