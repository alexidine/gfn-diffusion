# The exact-density flow as a train.py policy

*Plan, 2026-10-27. Status: agreed by the owner (direction and shape). Built: Shape items 1-2 and the checkpoint and
config work item (`models/nf_policy.py`, `models/policy_interface.py`, the `model:` section); the rest is open. Not policy: the owner decisions it
records are listed under "Decisions"; everything else is a plan, to be checked against the code as it lands.*

## Why

`inn_modeller.py::InnModeller` trains the conditional exact-density spline flow by subclassing the trainer
(`train.py::Modeller`) and overriding `train`, `evaluate`, `step_loss` and the reporting. Every override replaced
trainer logic with a private copy, and every mechanism not copied silently became an INN default. Found so far: the
level the stored rows are scored against, the stored-row source and churn, the buffer metrics, a fixed gradient clip
of 10 that binds on nearly every step where `train.py` uses `grad_clip_guard`, no checkpoint/resume of buffers or
optimizer, no loss-spike rewind, ~100 flags outside config validation, and about ten metrics logged under
`train.py`'s names with a different meaning.

## Decisions (owner, 2026-10-26/27)

- One central training script with modular components: the model is swapped in; no parallel trainers.
- The flow enters `train.py` as a policy (a model), not as a trainer subclass. `train.py`'s loop, stage protocol,
  losses, buffers, clipping, LR control, evaluation and checkpointing run unchanged; only what is intrinsically
  multi-step is adapted.
- One prior construction for both phases (anchors + their noised cloud, noising at eval), as the SDE trainer.
- The family is named NF (normalizing flows): coupling-layer INNs and autoregressive flows are both NFs. The policy
  is `models/nf_policy.py::NFPolicy`.
- One config file, the model selected by kind: mk_dev's `model: {kind, shared, sde, nf}` (`kind` sde | nf; `shared`
  the geometry both read; `sde` the SDE architecture; `nf` the NF settings). The loader resolves it
  (`config_invariants.canonical_model_section`), so SDE configs, checkpoints and problem hashes are unchanged. INN
  flags enter as keys only where no trainer key already covers them.

## Shape

1. **A flow policy class in `models/`**, beside `models/gfn.py::GFN`, wrapping the existing flow
   (`inn_modeller.py::LatentFlow` / `ConditionalLatentFlow`, the atom encoder, the level head). It implements the
   policy interface with one step: `get_traj_fwd` (states `[N, 2, D]` = `[s0, x]`, `log_pfs [N, 1]` = log q(x | c)
   with gradients, `log_pbs` = 0, `log_flow[:, 0]` = the level head), `get_traj_bwd` (the deterministic path x -> s0:
   log P_B = 0, log P_F = log q(x | c)), `get_traj_replay` (re-score a stored `[s0, x]` under the current q).
2. **A model-agnostic trainer.** `train.py`, `protocol.py` and `checkpointing.py` read about 30 `gfn_model.*`
   attributes (`flow_model`, `backward_policy`, `pb_frozen`, `ang_mask`, `dead_rows`, `full_flow`, `force_on`,
   `conditions_type`, the `_live_*` stash, ...). Each read is either moved behind a method both policies implement or
   given a flow answer; no scattered `if flow:` branches. One config key selects the policy class where the model is
   built (`train.py::Modeller._build_gfn_config`, `checkpointing.py::Checkpointer._build_gfn` / `_gfn_config_from`).
3. **Trainer additions useful to every policy:** the `gauss` anchor tile in `Modeller._noise_and_condition`; the
   density readouts (per-molecule NLL on held-out molecules, the importance-weighted term, round-trip and
   reward-currency start-up checks, cross-condition metrics, excess against the held-out floor), written against the
   policy's density estimate (exact for the NF, `log_pf_estimate` for the SDE, owner 2026-10-27). Held-out data is
   train.py's `test_molecules_path` (whole molecules); the INN's held-out split of rows within a molecule is not
   ported. The INN's `updates_per_batch` and stored-row bank are covered by replay construction and maintenance.
   Each addition only once the first ported runs show a need.
4. **Retire `InnModeller`** once the ported flow reproduces it on one system (see Verification).

## What carries over unchanged (from the 2026-10-27 audit)

- Every loss the conditional route uses works with length-1 trajectories (log P_B = 0, log q exact): terminal MLE,
  TB with Huber, the forward Z sidecar, pooled conditional VarGrad. Not applicable: TBC, DB/SubTB, replay-tail
  re-propagation (`stored_force_k`, `resample_last_k`); none is used on the conditional route.
- The prior dataset, anchor buffer, prior buffer, condition tracker and Emin(c) hold terminal states only. Replay
  stores trajectories, which become `[s0, x]` pairs re-scored exactly.
- The eval and logging layer; checkpoint/resume with buffer sidecars; loss-spike rewind; `grad_clip_guard`; the LR
  controller; stage actions; anchor screening/growth (the confirm estimate becomes exact: `confirm_k` 1); config
  validation.

## Work items

| item | size | notes |
|---|---|---|
| Policy interface + flow policy class | medium | the ~30 attribute reads; condition id must reach the policy for atom tokens (the trajectory calls receive only `condition` and `mol_batch` today); a unit test that every attribute train.py reads exists on both policies (a missing one is a silently inert flag) |
| Trajectory-count defaults | small | `gradient_norm_clip: auto` comes from a (W, T) rule fitted for T 10-100: set explicitly for the flow; `freeze_pb` / `set_pb_freeze` a no-op; check the 8 `integrator.T` sites; eval_T = integrator.T = 1 |
| Checkpoint and config | medium | flow class and its settings in the saved model config; the INN flags become YAML keys so `config_invariants` and `config_snapshot --check` cover them |
| Trainer additions (item 3) | medium | each with its own test |
| Recalibration | medium-large | LR seed (the conditional fused rate is ~2.5e-5, 4x below the INN's), clip-guard warm-up, `pooled_beta`, `progress_gate` bars, hot-LR bars: all fitted on SDE runs; settled by short real runs (AGENTS.md: loop components are verified by a run) |

## The objective question

The canonical conditional protocol (`mk_dev.yaml` `protocols.conditional_vargrad`) is not the INN's recipe. Its
stage 1 scrambles conditions (`scramble_conditions`: the product is the unconditional mixture); its stage 2 freezes
the forward policy and trains it only through pooled VarGrad over replay and prior rows, with one forward rollout in
20 steps feeding replay and the Z sidecar. The INN trains on its own fresh draws every 4 updates against
per-condition tracker levels. Plan: the first ported runs use the existing protocol unchanged (the test of the
flow inside the central trainer). If it underperforms, the INN-style term enters as a stage option of `train.py`, not
as a second trainer. Owner to confirm at that point.

## Verification

1. Unit: the policy interface test; the length-1 trajectory shapes and log P_B = 0; `gflownet_losses.py` on a
   length-1 batch against hand-computed values; a checkpoint round trip of the flow policy.
2. Parity, before any new experiment: on one system (MIPCAS per the tuning rule, then the conditional qm9rc 1%
   subset), the ported flow's stage-1 likelihood and stage-2 metrics against `InnModeller` at matched settings.
3. Short real runs at production-shaped parameters for each recalibrated loop component.

## Pre-existing defects found by the audit (fix regardless)

- `train.py::Modeller._fwd_gates`: a rollout every step until `replay_warmup_rows` (20,000) rows are admitted, with
  no step bound. Needs a termination bound (AGENTS.md: every wait needs one).
- INN today: the excess-energy degrees-of-freedom bar uses `data_ndim` (dead rows counted); `under_coverage` is
  always the uniform form; `gpu_guard.TRAIN_ENTRYPOINTS` omits `inn_modeller.py`; the module docstring and a
  `train()` comment contradict the code.

## Not in scope

The conformer route; the SDE policy's own behaviour (its attribute reads move behind the interface without changing
what it does: the existing crystal tests must pass unchanged).
