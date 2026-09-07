# Hand-off: rarer rollouts v1 — what is built, how to test it, what is open

Written 2026-09-07 evening for the session that takes over local testing. The
design is `docs/design/rarer_rollouts.md` (long; the invariants and the last two
sections are the parts that matter). This file is the short version.

## What is on `conditional` now (branch `rarer-rollouts-v0`, merged)

Every step is a fused bwd + replay policy step; the forward branch runs on 1 in N
steps and is the only energy call. Six additions on top of that (all shipped ON in
the generators except D, which has its own arm):

| item | what it does | keys (default = today's behaviour) | wandb |
|---|---|---|---|
| A | each replay row stores the log p_F its generating policy gave it; on every replay draw the current policy's score is compared | `fwd_rollout_drift_max` (stage, 0 = off trigger) | `replay/policy_drift_std` (headline), `_nats`, `_ess_frac`, `_covered_frac` |
| B | 10% of admitted rows are held out of training and re-scored every 10 steps: the buffer's generalisation gap | `buffers.replay_buffer.val_frac` (0 → off), `val_cap` 256, `val_min` 64 | `replay/val_loss`, `replay/val_gap`, `replay/val_n`, `replay_buffer_val_rows` |
| C | the eval rollout's log w is fed to the Z fill (2500–10000 samples: the most precise Z measurement the run makes) | `z_calibration.fill_from_eval` off/report/**fill** | `z_fill/eval_gap`, `eval_se`, `eval_n`, `eval_refused_<reason>` |
| D | on rollout steps the forward TB loss trains the policy with the batch's own root standing in for log Z (level-blind) | stage `loss_coeffs.fwd.tb_z_source: batch_root`, `fracs.fwd` > 0, `freeze_policy: 0` | `fwd/tb_err`, `fwd/scatter_err` (unchanged meaning); `fwd/tb` changes meaning |
| E | the fill is a 1-D Kalman filter: Z moves by K = P/(P + se²) of each measurement's gap; first measurement taken whole | `z_calibration.fill_mode` snap/**absorb**, `fill_process_var` 0.01, `fill_moment_reset` 0.5 | `z_fill/K`, `z_fill/P`, `z_fill/dz`, `z_fill/fired` |
| F | the forgetting sensor reads `bwd/relative_under` (centred on the batch's own mean log w) instead of the Z-anchored under-coverage, which was measuring the fill | none (metric renamed) | `bwd/relative_under_rise150`, `protocol/gr_sensor`, `gr_share`, `gr_fired` |

Also: `_fwd_gates` (rollout vs train gate, one helper), `_z_fill_head_is_fillable`
(the stash arming predicate), `BufferColumnError` (a replay sidecar without the two
new columns is refused at restore, not filled).

## Why F changed shape

Offline read of all four rr07 arms: every window where `bwd/under_coverage` rose by
more than 1 nat was a window where log Z *fell* by 1–6 nat (the fill), and the guard
cut replay to its rail on that — in both runs where the controller was live, replay
was starved for ~1000 steps and the run finished worse than the inert ones.
`relative_under`'s rise never exceeded 1 nat on the same runs. A warm-up would have
hidden this, not fixed it.

## Local acceptance — one run, then one comparison

`configs/local_rr_sep07/rr_n7.yaml` (N = 7, batch 400, 2000 steps, ELJ mipcas, from
`lp02`'s phase-1 exit). Compare on wandb (project GFN Energy, tag `rr07`) against
`rr07_rr_n7` — the pre-v1 run of the same arm.

Pass, per item:
- A: `replay/policy_drift_covered_frac` ≥ 0.95 by step 400; `policy_drift_std` finite,
  > 0 and *growing between rollouts, resetting at each*; `rollout/drift_trigger_fires` absent.
- B: `replay_buffer_val_rows` ≈ 200 by step 500; `replay/val_n` ≥ 64 on every write after
  step 400; `replay/val_gap` small and stable (a rising gap is memorisation — the sensor
  `resid_vs_intake` should agree).
- C: `z_fill/eval_refused_*` absent after the stage enters equilibration
  (`eval_refused_head` on the step-50 eval is correct: the head is not fillable in
  train_prior); `z_fill/eval_se` ≈ `z_fill/se` × sqrt(400/500) locally (the 1/√B model).
- E: `z_fill/K` in (0, 1], near 1 on the first fill and on eval fills, smaller on noisy
  early training batches; `z_fill/P` shrinking then settling; `z_fill/dz` never a
  multi-nat jump after the first fill. No `blocked_by_se` storm.
- F: `protocol/gr_sensor` present from step 300, staying below 1 through the early
  transient; `gr_share` ramping up from 0.5 (not cut to 0.10 by step 760, which is
  what the old sensor did); final energy within noise of 4.95 (the inert rr_n7), not
  the 10.5 of the cut arm.
- D (separate): `configs/local_rr_sep07/rr_n7_fwd.yaml`. `fwd/tb_err` at step 2000 no
  worse than rr_n7 + 10% (a fall is the win); `z_fill/gap` still present at the 1-in-7
  cadence (the widened stash gate); rollout-step wall time < 1.25× non-rollout.

Run command (local, from `energy_sampling/`, env `GFN_GPU_GUARD=0`, venv python):
```bash
python train.py --config configs/local_rr_sep07/rr_n7.yaml
```

## Cluster

`configs/rr_sep07/` (4 smoke arms N = 10, 8 production arms N ∈ {5, 20}) regenerated
on the current tree and load with zero invariant errors; `INDEX_*.tsv` and the two
sbatch files are beside them. Single-leg jobs only until the `lj_coeff` resume fix
(Monday). The cadence sweep said: N = 20 ≈ N = 7 in quality per step, N = 50 not
viable, ELJ speed saturates ~2.7× by N = 20; {5, 20} brackets the useful range.

## Open

- The drift trigger (`fwd_rollout_drift_max`) ships off: no run has reported a drift
  number yet, so no bar can be chosen honestly. Read `replay/policy_drift_std` on the
  acceptance run first.
- `fill_from_eval: fill` moves Z at eval cadence by up to the eval batch's full gap
  (K ≈ 1). Under the absorber that is the intended behaviour; if the eval Z and the
  training Z disagree systematically (`z_fill/eval_gap` vs `z_fill/gap` of opposite
  sign, repeatedly), the eval batch is not a sample of the training policy at the
  training target and `report` is the safe setting.
- `lj_coeff` resume crash (production resumes): open, Monday.
- Two manuscript-vs-code method regressions unrelated to this work
  (`docs/design/training_workflow_vs_manuscript.md`, Part 2, B.3–B.4).
