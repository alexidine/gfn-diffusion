# pool8_kick_nov20

One P-1 pool8 K=4 prior that differs from the current one,
`pool8_nov09/priors/pool8_p1_train_k4_metric_switched.pt`, only in the children's kick-radius distribution. It is
for a kick-scale MLE test (owner 2026-11-20; the hub builds the training arms).

- **Current prior:** metric kicks at 10^U(-2.0, -0.375) Å. Because the draw is log-uniform, half the children sit
  below about 0.065 Å.
- **This build:** 10^U(-0.675, -0.375) Å, which is 0.21–0.42 Å. The top radius stays at the calibrated -0.375 (not
  re-calibrated); the range is narrowed to a factor of 2 below it.

## What is the same

- **Anchors:** the same band sets. `train_list.txt` is copied from the current prior's build, and setup checks the
  band folder still lists them.
- **Held-out split:** the same descent groups. `setup` writes them from the current prior
  (`prior_kick_compare.py units`), and the build takes them through `build_noised_prior.py --heldout_units` instead
  of drawing.
- **Build settings:** K 4, `--child_kernel metric`, FD chart, eLJ scale 0.36358, window_kT 6, split v1.
- **Energy and forces:** scored under the switched eLJ (mk_dev's `lj_envelope` 1.0), with `--forces`. The current
  prior had these by restamp; this build scores them directly.

## What differs besides the radii

- **The kick draws themselves:** the shards are seeded 1000 + i. The current prior came from one GPU build at seed 0,
  so the directions are not paired with the current prior's; only their distribution is the same.
- **The lowest-anchor rule:** the builder's "a condition's lowest anchor stays in training" is not re-applied. The
  split is copied whole, and the shard logs count any condition whose lowest anchor under the switched energy is
  held out.

## Steps

`submit.sh` chains three jobs with afterok:

| step | job | resources | what it does |
|---|---|---|---|
| 1 | setup.sbatch | 1 core, 32 GB, 1 h | checks the band records and the build list; reads log_hi from the current prior (refuses unless -0.375) and writes log_lo = log_hi - 0.3; writes `heldout_units.pt` |
| 2 | shard.sbatch | array 0-99, 4 cores, 16 GB, 6 h | shard i = band sets 29i+1 .. 29i+29; the recipe of `configs/pool8sg_prior_nov17/shard_template.sbatch` (the sg14/19/61 builds with forces) plus `--log_lo` and `--heldout_units` |
| 3 | merge.sbatch | 8 cores, 128 GB, 8 h | merges the shards to the prior, then `prior_kick_compare.py compare CURRENT NEW --window_kT 10 6` to `compare.txt`; **fails when the anchor sets differ** |

The checks in `compare.txt`:

1. Anchor sets per split, keyed on (identifier, space group, Z', basin_group, basin_id), plus the largest
   crystal-parameter difference over the matched anchors.
2. Each child's seed-energy excess over its parent anchor in kT (median, P90, the share within 6 kT), for training
   and held-out children, plus the stored kick radii.
3. The rows `noised_prior.select_rows` keeps at 10 and 6 kT: training children, buffer rows, held-out children.

The build makes no conditions or test file. The anchors are the current prior's, so the run's existing files apply.

Output: `/scratch/mk8347/data/crystal_datasets/conditional/pool8_kick_nov20/priors/pool8_p1_train_k4_metric_loglo-0.675.pt`

Launch, on the cluster after pulling:

    bash /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool8_kick_nov20/submit.sh

**Checked locally (2026-11-20):** synthetic unit tests only, `tests/crystal/test_prior_kick_compare.py`.
- The units → `--heldout_units` round trip holds out the same descent groups.
- The compare passes on identical anchors and fails when one changes split.
- Child excess is read against the parent, and the window counts match `select_rows`.

No build ran locally (local work is stopped). The shard array is the first end-to-end run of `--heldout_units`.

## Training arms (hub, `make.py`)

Two unaugmented MLE continuations of pool8_big_nov15/big from its step-50,000 weights, 15k steps each, identical but
for the prior: `mle_ctrl` (the current switched prior) and `mle_kick` (this build). `make.py` asserts the two configs
differ only in `prior_path` and `run_name`. Submit after merge.sbatch has written the prior and `compare.txt` passes:

    sbatch /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/pool8_kick_nov20/submit_pool8_kick_nov20.sbatch

Read: the overfit gaps within each arm, eval_fwd/jensen_z, Excess Energy Nats Per DoF P50, Nonthermal Fraction
(equipartition), and -log q on the rows both arms share (eval_bwd/mle_anchors, eval_test/mle). Each arm's own training
and held-out -log q are on different rows and do not compare across arms.
