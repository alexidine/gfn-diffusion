# inn_acr_oct23: the autoregressive flow on acridine, old prior, old MACE

Acridine, P2_1/c, Z'=1, trained with the exact-density autoregressive flow (`inn_modeller.py`) as `inn_mps_oct21` trained
MIPCAS eLJ, on:

- the energy of the September production arms (`prod_sep20/p20_acr_n5`, `prod_sep23_ft/p23_acr_ft`): MACE
  `acr_112025_mh1_stagetwo.model` (57,094,417 bytes, the same file on the dev box and in `/scratch/mk8347/data/`), not
  `acr_newmodel.model`, which binds acridine about half as strongly;
- the prior of those arms, `acridine_sg14_zp1_mace_prior_dataset_w3.pt` (14,671 anchors + 147,712 noised rows, all at
  handedness +1, stored `mace` under the old checkpoint, the old conformer `acridine_conformer.pt`, N at index 13), put
  through the pooled-prior assembly of `pool_acr_oct03` (`data_processing/pool_assemble_mace.py`), with two pre-passes
  that file needs and a campaign's registry does not.

## The prior: `acridine_oldmace_oct23_prior.pt`

271,120 rows = 33,890 states x their 8 normaliser images, 428,115,487 bytes, energy -75.92 .. -50.98 kJ/mol (10 kT).
Built on the dev box, 2026-10-23, from the scripts in `prior_build/`:

1. **Radius**, `thermal_oldmodel.py`: the 1 kT latent kick under the old checkpoint, measured as `pool_acr_oct03` measured
   it under the new one (MXtalTools `thermal_newmodel.py` with model, molecule and pool swapped). The 64 lowest anchors,
   relaxed under the old checkpoint, return their stored energies: they are relaxed minima, and they hold only 5 basins
   more than 0.15 apart in atomwise RDF. 32 hop-style kicks per size per basin; median over the 5 basins [min, max]:

   | median energy rise | latent kick size | atomwise RDF distance |
   |---|---|---|
   | 1 kT | 0.0242 [0.0210, 0.0256] | 0.0948 [0.0885, 0.0980] |
   | 2 kT | 0.0348 [0.0331, 0.0369] | 0.1327 [0.1273, 0.1351] |
   | 3 kT | 0.0418 [0.0384, 0.0478] | 0.1574 [0.1507, 0.1670] |

   Radius 0.0242 (new checkpoint: 0.0312). It is also the stage-1 noise, `--mle_noise_log_range` -1.6162 = log10 0.0242,
   as MIPCAS eLJ's 0.0244 was in `inn_mps_oct21`.
2. **Reduced cells**, `reduce_cells.py`: the old file predates MXtalTools' monoclinic class walls, and 3,851 of its anchors
   (none of its noised rows) carry a reduction penalty (max 0.496 raw), which the trainer scores at `reduction_coeff`
   1000 and `pool_anchors`' export would leave out. Each is re-described in its reduced cell (`crystal_fd.ops.reduce_cell`'s
   basis change and origin shift, the trainer's chart by `pool_anchors.Chart`; 1,546 through the inversion partner, which
   the planar molecule matches at +1 to 1.4e-6 A). All re-read the trainer's way at their stored energy (median |d|
   0.0015, max 0.085 kJ/mol), penalty 0, inside the latent box. `crystal_fd` was untracked in the dev-box MXtalTools
   checkout when this ran; the check above is what the file rests on.
3. **Image-aware thinning**, `image_thin.py`: the old anchors hold one crystal under several of its normaliser images
   (74% have another anchor within 0.0242 of one of their images), which the assembly's thinning on stored rows does
   not see, and its fold would then write such a crystal once per description. Candidates in the assembly's order, after
   its 10 kT ceiling; one is dropped when any of its 8 images lies within 0.0242 of an image of a kept one: 5,291 of
   14,671 anchors and 28,673 of 144,914 noised rows kept. The noised rows stand in for the walk states (the old file has
   no capped-MC walk; none is run).
4. **Assembly**, `pool_assemble_mace.py` unchanged, `--radius 0.0242 --ceiling_kT 10`: nothing outside the box or under
   the density penalty; its thinning finds 78 close pairs among the stored rows and keeps 5,290 anchors and 28,600 walk
   states; 2,000 rows re-read at their stored energy (median |d| 0.0015, max 0.109 kJ/mol), reduction penalty max 8e-7.
   States by energy above the lowest (anchors, walk): 0-1 kT 103, 17; 1-2 kT 246, 232; 2-5 kT 3,260, 7,338;
   5-10 kT 1,681, 21,013.

```
python configs/inn_acr_oct23/prior_build/reduce_cells.py D:/crystal_datasets/conditional/priors/acridine_sg14_zp1_mace_prior_dataset_w3.pt <b>/reduced_w3.pt --mlip_path D:/crystal_datasets/acr_112025_mh1_stagetwo.model
python configs/inn_acr_oct23/prior_build/image_thin.py <b>/reduced_w3.pt <b>/thin --radius 0.0242
python -m data_processing.pool_assemble_mace <b>/thin/anchors.pt <b>/thin/walk <b>/acridine_oldmace_oct23_prior.pt --sg 14 --mol D:/crystal_datasets/acridine/acridine_conformer.pt --mlip_path D:/crystal_datasets/acr_112025_mh1_stagetwo.model --radius 0.0242 --ceiling_kT 10 --check 2000 --chunk 8 --device cuda --vram 0.5
```

Run them on the GPU, one at a time: on the CPU, `pool_assemble_mace`'s MACE check committed 92 GB on the dev box.

## The runs

`submit_inn_acr_oct23.sbatch`, flags in `common.args` and `arms.tsv`: one stage-1 run, then the two stage-2 arms (`base`,
2000 fresh draws per update; `b1000`, 1000) at once on the same A100 under MPS, 2-day job, 40-hour stage-2 clock.

Local smoke on the dev box (RTX 5080 laptop GPU, cap 0.4 = 6.4 GB, an 8,000-row slice of the prior = 1,000 states with
their images), 2026-10-23:

- stage 1, 400 steps at batch 8192: 3.5 min with three 2,000-draw evals; held-out NLL 6.93 -> -6.99, reasonable fraction
  of draws 0.001 -> 0.13; the stored rows' reward rebuilt from the latent agrees with the stored one (median 6e-4 nats).
- the untrained flow's step-0 eval ran the card out of memory on ONE crystal before `energy_config.mlip_max_packing_coeff`
  existed: uniformly over the sg 14 latent box 73% of cells have a packing coefficient above 1 (median 3.3, largest
  ~600, shortest cell length 0.55 A). With the guard at 2.0 every eval completes.
- stage 2, 20 updates of 2,000 draws from that 400-step flow: 15.7 min; MACE is 99% of an update at 22-27 ms per crystal
  (adaptive chunks of ~7 crystals at this cap; ~0.9 GB per crystal); the guard took 0.4-2% of draws; host RAM peak 11.6 GB.
  An A100 share runs larger chunks, so the cluster rate is not measured here: the clock schedule stretches each arm's
  rate decay over whatever speed it gets.

Submit, on the cluster, after `git pull` in gfn-diffusion (the prior must be in `conditional/priors/` at its byte count):

```
sbatch /scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling/configs/inn_acr_oct23/submit_inn_acr_oct23.sbatch
```
