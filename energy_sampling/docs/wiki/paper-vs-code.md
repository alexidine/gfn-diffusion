# Paper vs code

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The manuscript describes a training method; the code in this repository runs one. This page is a ledger of where the two descriptions differ. Every manuscript-side statement here is the reading recorded in `docs/design/training_workflow_vs_manuscript.md`, a note written 2026-09-07 against `Crystal_Generator (7).pdf`; the manuscript itself was not read for this page, and the page numbers, section labels and entry classes below are that note's. Every code-side statement is a read of the tree and of `configs/mk_dev.yaml` at the stamped commit.

The note's classes are **NEW** (in code, absent from the paper), **CHANGED** (the paper describes an older version), **REMOVED** (the paper describes something the code does not do) and **MISDESCRIBED** (the code is unchanged and the paper is wrong about it), with a warning mark on entries it calls regressions in the method rather than in the text. The table below sets, for each of six subjects, the note's reading of the manuscript beside the code at the stamped commit and the note's class, and a section follows on each. The mechanisms are described elsewhere: the branch assembly and coefficient inventory are on [loss-composition](loss-composition.md), the store the backward branch draws from on [prior-buffer](prior-buffer.md), the frozen archive behind it on [anchor-buffer](anchor-buffer.md), the third branch on [replay-buffer](replay-buffer.md), the level estimators and the fill on [z-calibration](z-calibration.md).

| subject | note reports the manuscript says | code does | note's class |
|---|---|---|---|
| prior dataset and noised buffer | noised prior states, 200k pre-generated before training, older rows exiting preferentially | `cfg:buffers.prior_buffer.source` `anchors`, rebuilt at stage entry to `init_fraction` 0.25 of `max_size` 250000 through an energy gate, no age term in any exit channel | CHANGED (A.8, B.5), NEW (the gate) |
| anchor draw and importance weighting | anchors selected by importance weight, described in three sentences of S2 | `cfg:buffers.anchor_buffer.refresh_every_n_evals` 0 and `replay_beta` 1.0, so the draw is uniform over a frozen archive | REMOVED, marked a method regression (B.3) |
| online noise magnitude | the jitter magnitude is calibrated per dataset | `cfg:buffers.anchor_buffer.noise_log_range` `[-2.5, -1.5]` is a fixed constant; the dataset's calibrated `log_noise_range` is read only by the offline build | marked a method regression (B.4) |
| Z warm start | phase 1 warm-starts log Z with Z-only TB gradients, and exits on an MLE tolerance | no TB term in phase 1, `cfg:bwd_loss_coeffs.freeze_z` 1 throughout; log Z set at phase-2 entry, exit on the W1 progress gate | CHANGED (B.6, B.7) |
| loss composition | phase 3 mixes forward and backward TB, log Z from the forward loss, a fwd:bwd step ratio driven by an eq. 13 fit | one fused step of loss-weighted branches, forward at `freeze_policy` and frac 0, a replay branch the paper does not mention, `balance.kind: gated_ramp` | CHANGED + NEW (A.6, B.1), REMOVED then replaced (A.7, B.8) |
| stages | three phases, smoothly transitioning into backward TB | two stages, `train_prior` then `equilibration`, one hard boundary | REMOVED (A.5) |

## Prior dataset and noised buffer

The note reports S2 describing the backward branch's terminals as noised prior states drawn from a set of about 200k rows generated before training, with older rows exiting preferentially. The code draws them from `prior_buffer`, seeded at init from the prebuilt prior dataset (`cfg:buffers.prior_buffer.seed_source` `prior_dataset`), rebuilt at equilibration entry by `train.py::Modeller.rebuild_prior_by_churn` to `init_fraction` 0.25 of `max_size` 250000 through the online admission gate, and topped up from noised anchors by `train.py::Modeller.top_up_prior_from_anchors`.

The note records an interim mechanism between those two states: the churn budget drawn from a frozen copy of the phase-1 policy taken by the `snapshot_prior` stage action, with anchors as backfill. The canonical config sets `cfg:buffers.prior_buffer.source` to `anchors`, so no prior model is in the top-up loop, while `snapshot_prior` remains in `train_prior`'s `on_exit` list and in `protocol.py::ACTIONS`.

The note reports no admission gate in the manuscript. The code admits a noised row only where its energy sits below that condition's best known energy plus `cfg:buffers.prior_buffer.ramp_floor` (100), with a record-breaker bypass. Rows leave through `train.py::Modeller.manage_prior_buffer` by that same gate, capped per call at `expire_max_frac` 0.1, or on overflow by low residual among often-drawn rows; no exit channel reads age.

## Anchor draw and importance weighting

The note reports S2 describing anchors selected by importance weight. The archive is frozen (`cfg:buffers.anchor_buffer.frozen` true, guarding `buffer.py::AnchorBuffer.admit` and `::AnchorBuffer.thin`), its surprise refresh is off (`refresh_every_n_evals` 0, so `train.py::Modeller.refresh_anchor_buffer_surprise` never sweeps and anchor `ema_loss` stays all-NaN), `thin_every_n_evals` is 0, and `replay_beta` 1.0 makes the random floor the whole draw, so `top_up_prior_from_anchors` draws uniformly. The note marks this entry a method regression rather than a text delta.

## Online noise magnitude

The note reports §4.3 step 2 and S2 stating that the jitter applied to a stored row is calibrated. `train.py::Modeller._noise_and_condition` applies `cfg:buffers.anchor_buffer.noise_log_range`, a fixed `[-2.5, -1.5]`, at both of its anchor call sites. The dataset's own calibrated `log_noise_range` is computed at build time and stored in the prior dataset by `data_processing/collate_prior.py`, which is also the only module that reads it back; no module a training step executes reads the stored value. The note marks this entry a method regression.

A second jitter geometry sits beside the constant, in neither document. `cfg:buffers.anchor_buffer.tile` `shaped` routes the draw to `train.py::Modeller._shaped_anchor_tile`, a per-anchor Gaussian read from the sidecar at `shape_path` and centred on that anchor's stored relaxed minimum, under which `noise_log_range` is unread; `tile` is `iso` in the canonical config.

## Z warm start

The note reports the manuscript warm-starting log Z in phase 1 with Z-only TB gradients and exiting on an MLE tolerance. Phase 1 runs `cfg:bwd_loss_coeffs.mle` 1.0 with `tbc` 0.0 and no TB term, and `freeze_z` is 1 on the backward and replay blocks of the base config, so log w is only accumulated, detached, into `buffer.py::ConditionLogZTracker`. The stage exits on `gates/progress_done`, the per-column Wasserstein progress gate computed by `train.py::Modeller.progress_metrics`.

The level is set at phase-2 entry by the `bootstrap_z:rollout:4000` action on `equilibration`'s `on_enter`, running `train.py::Modeller.bootstrap_z_by_rollout`. Thereafter it moves only through `train.py::Modeller.z_level_fill`; the `z_calibration` servo is off by the stage flag of the same name, and `protocol.py::Stage` refuses a stage declaring that flag together with a nonzero `fwd_rollout_every`.

## Loss composition

The note reports page 14 describing a third phase mixing forward and backward TB with log Z taken from the on-policy forward TB loss. In the canonical config `equilibration` carries `fwd: { tb: 1.0, freeze_policy: 1.0 }` at `fracs.fwd` 0.0, below the stage's `deactivate_threshold` of 0.01, so the forward branch trains nothing; its rollouts still enter the replay buffer and supply the level measurement. The policy is trained by the backward branch and by a replay branch the note reports as absent from the manuscript: stored forward trajectories re-scored under the current densities, drawn by `train.py::Modeller.draw_replay_sample` through `buffer.py::CrystalBuffer._sample_indices` with `cfg:buffers.replay_buffer.prioritise.enabled` true, `kappa` 1.0, `floor_frac` 0.25 and `symmetric` true, so the draw probability is a floored power of the absolute residual with self-normalised weights undoing it.

The note also reports page 14 and S4 describing a fwd:bwd step ratio driven by a slope and intercept fit. No step ratio exists: `cfg:stage.fracs` are loss weights on a single fused step fired by `train.py::Modeller.fused_train_step`, and the slope and intercept survive as eval-time fits in `eval/evaluations.py`. The live controller is `balance.kind: gated_ramp` on `bwd/under_coverage_rise150`, with a ratchet on `bwd/under_coverage` the note does not describe; the controller itself is on [balance-controllers](balance-controllers.md).

## Stages

The note reports three phases and a smooth transition into backward TB. The `unconditional_tb` protocol declares two stages, `train_prior` and `equilibration`, the second with no `exit` block, so it runs to `epochs`; backward TB on the noised buffer runs concurrently with replay and with the rollouts from the first fused step. The boundary is one transition: a fresh optimizer, the learning-rate burn-in leg, and the prior buffer rebuilt through the admission gate.

## Where the code has moved since the note

At the stamped commit all six subjects still differ, and the code side of four of them differs from what the note records.

- The canonical config sets `cfg:z_calibration.fill_mode` to `absorb`, a one-dimensional Kalman filter in which every measurement moves log Z by its precision share, with `fill_process_var` 0.01 nats squared per unmeasured step; where the key is absent `train.py::Modeller.z_level_fill` falls back to `snap`, the mode the note describes, whose `fill_threshold` and `fill_se` gates are under `absorb` an arming switch and a key the config marks snap-only. `fill_from_eval` is `fill`, so the eval rollout also supplies a measurement.
- The entry bootstrap applies its rollout's log w through `z_level_fill`, whose root is `gflownet_losses.py::winsorized_z_root`, not the mean log w the note records.
- The `gated_ramp` numbers are `up` 0.004, `down` 0.006, `bar` 0.0 and `bounds: { bwd: [0.25, 0.9], replay: [0.1, 0.75] }`, against the note's 0.0017, 0.043, 1 nat and a 0.5 floor on MLIP systems; `ratchet_metric`, `ratchet_tol`, `ratchet_cooldown_steps` and `ratchet_release_tol` are not in the note.
- The rollout cadence is `fwd_rollout_every` 20 beside a `fwd_rollout_triggers` block and `z_pin_rollout_every`, where the note records the cadence alone.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.prior_buffer.{source, seed_source, init_fraction, max_size, ramp_floor, expire_max_frac}`; `cfg:buffers.anchor_buffer.{frozen, refresh_every_n_evals, thin_every_n_evals, replay_beta, noise_log_range, tile, shape_path}`; `cfg:buffers.replay_buffer.prioritise.{enabled, kappa, floor_frac, symmetric}`; `cfg:z_calibration.{fill_mode, fill_threshold, fill_se, fill_process_var, fill_from_eval}`; `cfg:stage.{fracs, deactivate_threshold, fwd_rollout_every, z_pin_rollout_every, fwd_rollout_triggers, balance}`; `cfg:fwd_loss_coeffs.freeze_policy`; `cfg:bwd_loss_coeffs.{mle, tbc, freeze_z}`; `cfg:replay_loss_coeffs.freeze_z`. Stage actions `snapshot_prior`, `bootstrap_z`, `rebuild_prior_by_churn`.

Code: `train.py::Modeller.fused_train_step`, `.draw_replay_sample`, `.z_level_fill`, `.bootstrap_z_by_rollout`, `._noise_and_condition`, `._shaped_anchor_tile`, `.rebuild_prior_by_churn`, `.top_up_prior_from_anchors`, `.refresh_anchor_buffer_surprise`, `.manage_prior_buffer`, `.progress_metrics`; `buffer.py::AnchorBuffer.admit`, `::AnchorBuffer.thin`, `::CrystalBuffer._sample_indices`, `::ConditionLogZTracker`; `gflownet_losses.py::winsorized_z_root`; `protocol.py::Stage`, `protocol.py::ACTIONS`; `data_processing/collate_prior.py`; `eval/evaluations.py`.

## Could be tooling

Every code-side half of this ledger is a read on a named config key or symbol, so the re-check is scriptable: a walk resolving each cited `cfg:` path against the canonical config and each `file.py::symbol` against the tree would report, per entry, whether the value the ledger states is the value the config holds and whether the symbol still exists. The manuscript half is not reachable that way and remains a hand read against the note and the PDF's page and section numbers.

## Sources

Repo: `docs/design/training_workflow_vs_manuscript.md` in full, Part 1 for the workflow and Part 2 entries A.5 to A.8 and B.1 to B.8 for the deltas in these six subjects; `configs/mk_dev.yaml` and the code above, read at the stamped commit. Memory files project_manuscript_delta_audit_sep07 and project_paper_claims_noised_buffer_not_prior_model located the note and were not used as evidence.
