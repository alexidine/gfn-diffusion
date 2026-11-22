# pool8sg4_next_nov22

Three MLE arms that follow up `pool8sg4_nov20/mle4`, one Slurm array index each. `make.py`'s docstring is the arm
definition; it builds each arm by importing its reference's `make.py`.

Arm definitions. The reference is the run the arm differs from; every other key is the reference's.

| index | arm | reference | what differs | step cap |
|---|---|---|---|---|
| 0 | `v4_kick` | `pool8sg4_nov20/mle4`, fresh policy | `atom_tokens: split_pooled`; `live_kernel: chart`, `live_log_range: [-2.0, -0.375]`; batch fixed at 1000 | 150,000 |
| 1 | `p1_b250` | `pool8_smooth_nov21/x_fixed`, fresh policy | batch 250, `fused_grad_accum_min_samples: 250` | 150,000 |
| 2 | `x4_aug` | `mle4` weights at step 125,000, `pool8_auglong_nov18` recipe | rotation frames, held-out plateau exit, LR anneal; batch fixed at 1000; archives every 5,000 | 115,000 + 10,000 |

## Learning rate

The rate is a multiple of `seed_lr` = 1.25e-4.

- `v4_kick`, `p1_b250`: 0.05 for 3,000 steps, geometric to 0.2 by step 4,000, then 0.2 to the end (2.5e-5).
- `x4_aug`: 0.2 from its first step until the plateau exit or the 115,000-step cap, then geometric to 0.02 over 10,000
  steps, then snapshot and stop.

## Expected wall time

Estimates against the 48 h limit, scaled from wandb measurements of `mle4` (oc1rx96t, lxshwwd0), `x_fixed` (zwz98a68)
and `auglong` (qtffycou). None of the three arms has run on the cluster.

| arm | train step (s) | eval cost (s per 500 steps) | startup (min) | full cap (h) |
|---|---|---|---|---|
| `v4_kick` | 0.47 (mle4) | 54 | 35 | 25-26 |
| `p1_b250` | 0.09-0.15 | 23 | 8 | 6-9 |
| `x4_aug` | 0.60-0.67 | ~60 | 35 | 26-28 |

## GPU utilisation

The cluster cancels jobs on a low 2-hour mean. Survivors have sat at 43-45% (`mle4`, first 2 h); cancelled runs at 14%
and 24%. A September analysis put the line near 54%, which `mle4`'s survival contradicts, so the current line is not
known.

`p1_b250` is the arm at risk. From `x_fixed` (86% at 0.51 s/step, batch 1400), GPU work is about 0.32 ms per row, so
batch 250 is about 0.08 s of GPU per step beside 0.01-0.07 s of host time: 53-86% while training. To raise it without
changing rows per step, run a second seed on the same card through `run_group.sh`'s MPS route (not built).

## Launch

From `energy_sampling/` on the cluster, after pulling:

```
sbatch --array=0-2 configs/pool8sg4_next_nov22/submit_pool8sg4_next_nov22.sbatch
```

The job stops before training if an arm's prior, conditions or test file is missing, or if `x4_aug`'s source
checkpoint is missing from `checkpoints/`.

## Known limits

- `python -m config_snapshot configs/pool8sg4_next_nov22/p1_b250.yaml --check` reports one BASELINE departure, effective
  batch 250 < 1000. That is the arm's definition. The trainer reports BASELINE items and does not stop on them.
- `p1_b250` is not the same policy class as `mle4`'s P-1 slice: it has one group's chart (centroids on circles) and no
  pooled symmetry-operator input, so it controls for rows per step and not for the four-group layout.
- `x4_aug` takes its step labels from the source: `global_step` = step + 125,000.
- Local smokes ran at batch 64 on a K=0 prior holding P-1 rows only. Real rows of groups 14, 19 and 61, the 30.6M-row
  GPU memory load with the chart kick's centre table or with rotation frames, the real step-125,000 file, and all step
  rates and utilisation on the cluster are unproven until the first hour of each job.
