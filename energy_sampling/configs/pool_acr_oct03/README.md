# pool_acr_oct03: the acridine leg of the pooled prior build

Acridine, P2_1/c, Z'=1, MACE `acr_newmodel.model`, the universal conformer. It follows the `pool_oct02` protocol (GFN 5fc53fe2): the prior is built from the search's own end states plus a training-temperature walk from each. It is a separate battery because three things differ from MIPCAS and NEHZOR: the energy is MACE, the pool is one campaign with no older files, and the molecule is planar.

Stages (`launch.sh`):

1. **premerge** (CPU). One coordinator curate pass over every end state of the `acr_zp1_sep30` campaign within 10 kT. Its shards are linked, not copied. Clustering uses the campaign's atomwise identity cut, 0.134. Config: `pool_coord.yaml`.
2. **export** (GPU). `data_processing/pool_anchors_planar.py` writes the basins within 5 kT, in the chart the trainer reads, to `anchors.pt`. Handedness -1 rows are re-described at +1 exactly, because the molecule is planar: no row is embedded approximately and none is dropped.
3. **walk** (GPU, 4 shards). `data_processing/capped_mc.py` runs from `anchors.pt` with `--energy_function mace` and the arguments in `FLOOD_ARGS`. That file is one line, identical to the `pool_oct02` protocol. Starts are thinned at 0.134.
4. **assemble** (GPU, `launch.sh assemble`, submitted separately once the walk is done). `data_processing/pool_assemble_mace.py`:
   - leaves out states outside the trainer's latent box (64 anchors);
   - leaves out anchors and walk states the trainer's density penalty touches, before the de-dupe. Under `acr_newmodel` the walk's 15 kT ceiling is above the whole binding energy (13.1 kT), so 55% of the walk states are expanded cells below a packing coefficient of 0.55;
   - de-duplicates in latent space at 0.01, widening the walk states' radius until the file fits 400,000 rows;
   - writes the 8 normaliser images of every kept state, and re-scores a sample with MACE;
   - does not apply the molecule's own C2 relabelling (owner 2026-10-05: the model conditions on one labelled conformer).

   Output: `acridine_mace_pooled_oct03_pc055_prior.pt`, copied into `conditional/priors/` for the training arms.

**Do not train on `acridine_mace_pooled_oct03_prior.pt`.** That is the first assembly (job 19219651): no density filter, walk radius 0.284, 90% of its walk states below a packing coefficient of 0.55.

Launch, from anywhere on the cluster after pulling both repositories:

    bash configs/pool_acr_oct03/launch.sh all

Output: `/scratch/mk8347/data/crystal_datasets/pooled_oct02/acridine_mace/` holds `pool/registry.pt`, `anchors.pt` and `flood/shard_<k>/`.

Cluster runs: premerge 19108908 (24,022 basins within 10 kT), export 19108909 (10,986 anchors within 5 kT, none left out), walk 19108910 (9,027 walkers x 300 steps, 1,131,800 accepted states), first assembly 19219651.

The file the MLE arm was first pointed at (314,402,033 bytes) was assembled on the dev box from the cluster's anchors and walk shards with an earlier version of the assembly module, and uploaded. It still carries cached lookup fields the trainer may refuse. `launch.sh assemble` replaces it with the committed job's output.
