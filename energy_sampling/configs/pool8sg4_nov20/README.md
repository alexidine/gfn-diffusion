# pool8sg4_nov20

The first conditional run over four space groups at once: P-1 (2), P2_1/c (14), P2_12_12_1 (19) and Pbca (61), on the
pool8 molecules, switched eLJ. Owner 2026-11-20: start training and watch the usual metrics plus Z(sg) and per-group
statistics. `make.py`'s docstring has the arms and every change from the single-group recipe.

## Order on the cluster

0. **Test file** (CPU, ~20-30 min, independent of step 1): `sbatch configs/pool8sg_prior_nov17/test4_uniform.sbatch`.
1. **Check, then merge** (CPU, `prep_pool8sg4_nov20.sbatch`, 256 GB; done 2026-11-20: 40.8M rows, 45.8 GB): `check_priors.py` compares the four training
   priors' manifests memory-mapped (seconds) and stops on anything `merge_noised_priors.py` would refuse; then the
   merge writes `pool8sg4_nov20/priors/pool8_4sg_train_k4_metric.pt`, P-1 first. The check can also be run alone on
   a login node, before queueing the merge.
2. **Smokes** (`--array=0` smoke, `--array=1` smoke_cpu; ~1 h each, can run together): mle4's config at 300 steps,
   evals every 100, with the buffers on the GPU (smoke) and in host RAM (smoke_cpu). Read before mle4: which one
   loads, peak RAM (`_sacct.txt` MaxRSS, `_ram.txt`), peak VRAM (`_smi.csv`), the landed batch, steps per second, and
   that every `sg2/`, `sg14/`, `sg19/`, `sg61/` key and the `zsg/` panels appear.
3. **mle4** (`--array=2`): fresh policy, unaugmented MLE, capped at 60k steps, rows on the GPU, anchor seed
   `seed_floors_only` (the 30.6M seed rows only set each condition's energy floor, in 1M-row chunks; no anchor buffer,
   none admitted while `frozen`). History, 2026-11-20:
   - smoke (GPU, wandb poq5fwwy) and mle4 qicjg8lf ran out of GPU memory in the anchor seed (one clone of all 30.6M
     rows), before step 0;
   - smoke_cpu q0c73xp4 and mle4 owr4qv4f (`buffer_device: cpu`) ran, at 2.0-2.4 s/step and 7-50% GPU utilisation;
     the cluster cancelled owr4qv4f (SIGTERM, 2 h 13 m, step 2,026) with a 2-hour mean GPU utilisation of 23%.
   - mle4 4yg2mp5i (GPU rows, seed_floors_only) stopped at step 3,380 after ~2 h 10 m with a 2-hour GPU mean of 17%:
     each step's row draw permuted all 30.6M rows on the host (~1.2 s of a 1.5 s step). `buffer.py::uniform_rows`
     draws in ~0.2 ms.
   Startup is ~33 min at ~1% GPU utilisation (40.8M rows loaded and prepared); each `vram [...]` ledger line now ends
   with the seconds since start, to place that time. The single-group run (vtba4cf5) averaged 60% over its first 2 h.
   **mle4_150k** (`--array=4`, owner 2026-11-21): carries mle4 on to 150,000 steps by a full load of its running
   checkpoint, so the optimizer, LR state, step count and pass counter continue. Submit with a dependency on the mle4
   job. It reuses run name `mle4`, so it rewrites that run's running, best and final files; the step archives stay. In
   wandb it is a new run of the same name whose curves start at the resumed step.
4. **tb4**: not ready. `make.py --tb` raises until the four-group anchor-buffer placement is settled: the TB stage's
   prior buffer tops up from noised anchor-buffer rows, and mle4 builds none.

## Known limits

- **Not comparable to the single-group P-1 runs.** In the four-group slot layout only phi stays a circle (a slot wraps
  only when it does in every group), so P-1's centroid rows become intervals, and the row standardisation pools groups.
  The sg2/ numbers are not the bigbuf/auglong P-1 numbers. Their -log q is a density on the same chart box
  (`FDSlotLayout.pack` copies the chart rows and log q carries the standardisation term), so the levels share a measure,
  but the two models treat those rows differently.
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
