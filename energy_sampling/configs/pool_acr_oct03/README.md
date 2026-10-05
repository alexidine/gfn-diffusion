# pool_acr_oct03: the acridine leg of the pooled prior build

Acridine, P2_1/c, Z'=1, MACE `acr_newmodel.model`, the universal conformer. It follows the `pool_oct02` protocol (GFN 5fc53fe2): the prior is built from the search's own end states plus a training-temperature walk from each. It is a separate battery because three things differ from MIPCAS and NEHZOR: the energy is MACE, the pool is one campaign with no older files, and the molecule is planar.

Stages (`launch.sh`):

1. **premerge** (CPU). One coordinator curate pass over every end state of the `acr_zp1_sep30` campaign within 10 kT. Its shards are linked, not copied. Clustering uses the campaign's atomwise identity cut, 0.134. Config: `pool_coord.yaml`.
2. **export** (GPU). `data_processing/pool_anchors_planar.py` writes the basins within 5 kT, in the chart the trainer reads, to `anchors.pt`. Handedness -1 rows are re-described at +1 exactly, because the molecule is planar: no row is embedded approximately and none is dropped.
3. **walk** (GPU, 4 shards). `data_processing/capped_mc.py` runs from `anchors.pt` with `--energy_function mace` and the arguments in `FLOOD_ARGS`. That file is one line, identical to the `pool_oct02` protocol. Starts are thinned at 0.134.
4. **assemble** (GPU, `launch.sh assemble`, submitted separately once the walk is done). `data_processing/pool_assemble_mace.py` leaves out anchors outside the trainer's latent box, de-duplicates in latent space at 0.01 (walk states widened until the file fits 400,000 rows), writes the 8 normaliser images of every kept row, and re-scores a sample with MACE. The molecule's own C2 relabelling is not applied (owner 2026-10-05: the model conditions on one labelled conformer). Output: `acridine_mace_pooled_oct03_prior.pt`.

Launch, from anywhere on the cluster after pulling both repositories:

    bash configs/pool_acr_oct03/launch.sh all

Output: `/scratch/mk8347/data/crystal_datasets/pooled_oct02/acridine_mace/` holds `pool/registry.pt`, `anchors.pt` and `flood/shard_<k>/`.

Verified locally on 2026-10-03:
- The export on 200 campaign basins: 0 rows left out, largest energy change 0.006 kT.
- The walk for 12 steps on 41 starts: re-scored energies match the stored ones to 0.004 kJ/mol.

The job scripts themselves have not run on the cluster.
