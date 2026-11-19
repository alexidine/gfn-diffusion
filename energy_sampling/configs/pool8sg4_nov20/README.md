# pool8sg4_nov20

The first conditional run over four space groups at once: P-1 (2), P2_1/c (14), P2_12_12_1 (19) and Pbca (61), on the
pool8 molecules, switched eLJ. Owner 2026-11-20: start training and watch the usual metrics plus Z(sg) and per-group
statistics. `make.py`'s docstring has the arms and every change from the single-group recipe.

## Order on the cluster

0. **Test file** (CPU, ~20-30 min, independent of step 1): `sbatch configs/pool8sg_prior_nov17/test4_uniform.sbatch`.
1. **Check, then merge** (CPU, `prep_pool8sg4_nov20.sbatch`, 256 GB): `check_priors.py` compares the four training
   priors' manifests memory-mapped (seconds) and stops on anything `merge_noised_priors.py` would refuse; then the
   merge writes `pool8sg4_nov20/priors/pool8_4sg_train_k4_metric.pt`, P-1 first. The check can also be run alone on
   a login node, before queueing the merge.
2. **Smokes** (`--array=0` smoke, `--array=1` smoke_cpu; ~1 h each, can run together): mle4's config at 300 steps,
   evals every 100, with the buffers on the GPU (smoke) and in host RAM (smoke_cpu). Read before mle4: which one
   loads, peak RAM (`_sacct.txt` MaxRSS, `_ram.txt`), peak VRAM (`_smi.csv`), the landed batch, steps per second, and
   that every `sg2/`, `sg14/`, `sg19/`, `sg61/` key and the `zsg/` panels appear.
3. **mle4** (`--array=2`): fresh policy, unaugmented MLE, capped at 60k steps (about 17 h at ~1 step/s). If only
   smoke_cpu loaded, set `buffer_device: cpu` in make.py's mle4 and regenerate first.
4. **tb4**: `python configs/pool8sg4_nov20/make.py --tb <mle4 archive file name>` writes `tb4.yaml`; then
   `--array=3`. bigbuf's TB stage, 10k steps.

## Known limits

- **Not comparable to the single-group P-1 runs.** In the four-group slot layout only phi stays a circle (a slot wraps
  only when it does in every group), so P-1's centroid rows become intervals, and the row standardisation pools groups.
  The sg2/ numbers are not the bigbuf/auglong P-1 numbers.
- **Z(sg) is a trend, not a preference.** Each group's log Z lives on its own FD chart (different free cell
  coordinates, pose normalisation), so compare a group's curve over time, not levels across groups; the zsg/ p(g)
  readouts are a sampling allocation, not a thermodynamic group preference, until a cross-group convention is declared
  (owner decision).
- **Test file.** Uniform over the four groups: the same 125 held-out molecules in each group, at most 8 crystals per
  (molecule, group), built by `configs/pool8sg_prior_nov17/test4_uniform.sbatch`. The conditions file is P-1's:
  rollouts draw each molecule's group uniformly.
- **VRAM.** Measured locally (norbornane, 2026-11-20): GPU rows cost ~2.7 KB each (2,400 MiB for 888k rows), about 2.5x
  the on-disk size, so a pool8 row is ~1.1 KB and one GPU copy of the four groups' rows is ~35-40 GB. The trainer keeps
  two: prior_dataset (the MLE rows) and the static prior buffer (seeded whole from every scored training row). The
  buffer's seed source is now held in host RAM (`Modeller.init_prior_buffer_source`), but the two copies still do not
  fit beside the model on one A100; smoke vs smoke_cpu decides the placement. The lasting fix is compact rows (an FD
  point, a condition index, energies and force per row, the molecule looked up from the conditions table: ~100-200 B a
  row) or one shared copy; neither is built.
- **Wall time.** P-1's held-out optimum came at ~42-51k steps; four groups may need several times that, past one
  2-day job; mle4 is capped at 60k as a first look.
