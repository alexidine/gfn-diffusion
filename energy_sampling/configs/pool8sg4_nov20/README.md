# pool8sg4_nov20

The first conditional run over four space groups at once: P-1 (2), P2_1/c (14), P2_12_12_1 (19) and Pbca (61), on the
pool8 molecules, switched eLJ. Owner 2026-11-20: start training and watch the usual metrics plus Z(sg) and per-group
statistics. `make.py`'s docstring has the arms and every change from the single-group recipe.

## Order on the cluster

1. **Check, then merge** (CPU, `prep_pool8sg4_nov20.sbatch`, 256 GB): `check_priors.py` compares the four training
   priors' manifests memory-mapped (seconds) and stops on anything `merge_noised_priors.py` would refuse; then the
   merge writes `pool8sg4_nov20/priors/pool8_4sg_train_k4_metric.pt`, P-1 first. The check can also be run alone on
   a login node, before queueing the merge.
2. **Smoke** (`--array=0`, ~1 h): mle4's config at 300 steps, evals every 100. Read before mle4: the load's peak RAM
   (`_sacct.txt` MaxRSS, `_ram.txt`), peak VRAM (`_smi.csv`), the landed batch, steps per second, and that every
   `sg2/`, `sg14/`, `sg19/`, `sg61/` key and the `zsg/` panels appear.
3. **mle4** (`--array=1`): fresh policy, unaugmented MLE, capped at 60k steps (about 17 h at ~1 step/s).
4. **tb4**: `python configs/pool8sg4_nov20/make.py --tb <mle4 archive file name>` writes `tb4.yaml`; then
   `--array=2`. bigbuf's TB stage, 10k steps.

## Known limits

- **Not comparable to the single-group P-1 runs.** In the four-group slot layout only phi stays a circle (a slot wraps
  only when it does in every group), so P-1's centroid rows become intervals, and the row standardisation pools groups.
  The sg2/ numbers are not the bigbuf/auglong P-1 numbers.
- **Z(sg) is a trend, not a preference.** Each group's log Z lives on its own FD chart (different free cell
  coordinates, pose normalisation), so compare a group's curve over time, not levels across groups; the zsg/ p(g)
  readouts are a sampling allocation, not a thermodynamic group preference, until a cross-group convention is declared
  (owner decision).
- **eval_test/mle is P-1 only.** The conditions and test files are P-1's (rollouts draw the group uniformly, so the
  forward readouts cover all four groups); a four-group test file needs P-1's held-out anchors restamped and merged
  with the groups' held-out anchors.
- **VRAM.** With `buffer_device: cuda` and children-only training rows the GPU holds two copies of the prior rows; at
  four groups that may not fit. Config-only fallbacks if the smoke shows it: `buffer_device: cpu` (slower draws) or
  training on anchors and children (changes the MLE data; owner call).
- **Wall time.** P-1's held-out optimum came at ~42-51k steps; four groups may need several times that, past one
  2-day job; mle4 is capped at 60k as a first look.
