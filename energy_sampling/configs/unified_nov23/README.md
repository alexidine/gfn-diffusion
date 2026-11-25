# unified_nov23

One arm, `unified`, on the crystal store route (`buffers.store`). `make.py` builds it from
`pool8sg4_next_nov22/make.py::v4_kick` on the committed `mk_dev.yaml`; every key not listed here is v4_kick's.

Stages of the arm. B = batch rows per step, R = forward rows per condition.

| stage | steps | branch | rows from | loss | Huber knee (nats) |
|---|---|---|---|---|---|
| `train_prior` (MLE) | 150,000 | bwd | fresh physical kick of a uniform anchor, unscored | `mle` | none |
| `tb` (fused) | 100,000 | bwd, share 0.5 | uniform condition, uniform stored child | TB, Z(c) detached | 80 |
| | | replay, share 0.5 | uniform condition, stored on-policy row, `replay_weight: duty` | TB, Z(c) detached | 10 |
| | | fwd sidecar | rollout every 20 steps, R = 2 | `emp_z` on Z(c) only | 10 |

Store: `ram_gb: 224` (one constant, `RAM_GB`, also written to `#SBATCH --mem`), `ram_frac: 0.5`, `mle_draw: anchor`. The row
limit is `config_invariants.store_row_limit(224, 12, 0.5)` = 595,342,001 rows.

Scoring: `fwd_rollout_every = max(1, round(2 B R / SCORES_PER_STEP))` = 20 at B 1000, R 2, `SCORES_PER_STEP` 200. Each rollout
row owes one physical kick of an anchor, scored and stored as a child.

## Review before launch

- **Stage lengths.** 150,000 MLE + 100,000 TB steps, `epochs` 250,100. `pool8sg4_next_nov22/README.md` estimates 25-26 h for
  v4_kick's 150,000 MLE steps. The TB step rate on the store route is unmeasured. The job limit is 48 h.
- **Resubmission.** `continue_from_checkpoint` is false, so a resubmitted job starts a fresh run under the same `run_name`.
  Set it to true in `unified.yaml` before resubmitting a job that stopped at the limit.
- **LR hold.** MLE: `lr_control.schedule`, 0.05 x `seed_lr` (1.25e-4) for 3,000 steps, geometric to 0.2 x by step 4,000. TB:
  held at 0.2 x `seed_lr` = 2.5e-5, flow LR set to 1.0e-4 on entry. Both values are carried from earlier arms.
- **Replay weight.** This arm uses `duty` (`make.py`); the code default is `uniform`, and `visits` is the third rule of `crystal_store.REPLAY_WEIGHTS`. `duty` is refused with `rotate_frames` or `mirror_frames`.

## Regenerate

From `energy_sampling/`:

```
python configs/unified_nov23/make.py
python -m config_snapshot configs/unified_nov23/unified.yaml --check
python -m config_snapshot configs/unified_nov23/unified.yaml --card card.html
```

The first writes `unified.yaml` and `submit_unified_nov23.sbatch`. Arm settings live in the yaml; the sbatch reads them from it.

Local smoke config on the pool4 test prior (`smoke: true`, buffers in host RAM, nothing written under `D:`):

```
python configs/unified_nov23/make.py --local OUT_DIR --ckdir CHECKPOINT_DIR --batch 1000 --mle-steps 40 --tb-steps 120
```

Optional: `--rollout-every N` (default from the scoring budget), `--ram-gb G` (default 4; `CrystalStore` allocates at the row
limit when built), `--eval-samples E` (default 10 x batch), `--eval-period P` (default a quarter of the steps).

## Launch

Needs a checkout where the `--check` above passes. From `energy_sampling/` on the cluster:

```
sbatch --array=0 configs/unified_nov23/submit_unified_nov23.sbatch
```
