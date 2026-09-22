# Structure comparison metrics

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

Two objects decide that two crystal structures are the same packing: a distance between radial distribution functions, computed for every pair in a batch, and `ccdc.crystal.PackingSimilarity`, an out-of-process overlay of two packing shells molecule by molecule. This page defines both from the code, the RDF channel schemes and their invariances, the cut constants the analysis scripts carry, and the sliced-Wasserstein metrics logged during training, which measure a different thing. What the basin counts and landscapes are used for is [landscape-structure](landscape-structure.md); how the structures compared were generated is [crystal-search-method](crystal-search-method.md).

## The RDF a structure carries

`crystal_rdf.py::crystal_rdf` histograms the lengths of a crystal batch's *intermolecular* edges only (`edge_index_inter`), over `rrange`, default $(0,10)$ Angstrom in 100 bins. The channelled modes use `::smooth_1d_histogram`, which replaces each distance by a Gaussian of one bin's width rather than a hard assignment. Counts are divided by the shell volume $\tfrac{4}{3}\pi\big((r+\Delta r)^3-r^3\big)$, so the output is a radial density and not the dimensionless $g(r)$, and the last Angstrom is damped by a raised-cosine envelope, `::smooth_cutoff`. `crystal_analysis.py::MolCrystalAnalysis.compute_rdf` then divides by `z_prime`, so an RDF from a $Z'=2$ cell is on the same scale as one from $Z'=1$.

## Three channel schemes

`mode` selects the channel partition, and the modes are mutually exclusive, `mode` being a single string.

- **`elementwise`** (`::get_elementwise_dists`). An edge is assigned by the unordered pair of endpoint atomic numbers over a fixed element list (B, C, N, O, F, P, S, Cl, Br unless `atomic_numbers_override` is given), giving $\tfrac12 E(E+1)=45$ channels, invariant to any permutation of atoms of one element. It indexes its lookup table with the raw `atom_types` values, `lut[Z1]`, without the `.long()` cast `::get_atomwise_dists` applies.
- **`atomwise`** (`::get_atomwise_dists` on `atom_inds`). Every atom carries its index within the molecule, `torch.arange(mol_size)` tiled over the cluster, so an edge is assigned by the unordered pair of atom indices: $\tfrac12 A(A+1)$ channels for an $A$-atom molecule. `mol_size` comes from `crystal_batch.num_atoms[0]`, and the code's comment records that it assumes one repeated molecule.
- **`envwise`** (`::get_1WL_env_labels`, then `::get_atomwise_dists` on the labels). Bonds are the intramolecular edges shorter than 1.8 Angstrom; colour refinement runs on that graph from the initial invariant (atomic number, degree) for at most 32 rounds, and the labels are broadcast from the asymmetric unit onto every image, per $Z'$-molecule, giving $\tfrac12 A'(A'+1)$ channels. The refinement is 1-WL, while the branch comment in `crystal_rdf` says 2WL.

That last invariance separates the two molecule-level modes on a symmetric molecule. Where the molecular graph has a nontrivial automorphism, acridine's $C_2$ axis, two packings differing by applying it to one molecule are the same packing. `envwise` gives the exchanged atoms one label, so both get an identical RDF; `atomwise` keeps the indices distinct, the exchanged pair populates different channels, and the two land far apart. The `calibrate_basin_metric.py` module comment records that $P(\text{COMPACK match})$ against `atomwise` distance is non-monotonic, and records it as a property of the molecule and not of $Z'$.

Channel counts can coincide across modes. `generate_figs.py::require_rdf_mode` asserts that two results files carry the same `rdf_mode` and that neither is missing it; its docstring states that for a 13-atom molecule `atomwise` and `envwise` both give 91 channels, so a mismatch between those two modes raises nothing. `new_analysis.py::run_analysis` reads `cfg:rdf_mode` with no default and asserts it is `atomwise` or `envwise`.

## The distance between two RDFs

`crystal_rdf.py::compute_rdf_distance` normalises each channel of each RDF to unit sum, takes the 1-D earth mover's distance per channel as $\sum_i|\mathrm{cumsum}(p_1)_i-\mathrm{cumsum}(p_2)_i|$ (`::earth_movers_distance_torch`) times the bin width, and averages over *active* channels, those with nonzero mass in either argument, optionally reweighted by `channel_weights`. Each channel being normalised first, the distance sees the shape of each pair correlation and not its amount, and it carries the bin width as a unit: `rdf_cutoff_sweep.py` records that a distance computed under the pipeline's fixed bin convention at a shorter physical cutoff is $10/\text{cutoff}$ times the physical value. Pairwise matrices come from `::compute_rdf_distmat`; `::compute_rdf_distmat_parallel` returns $\log_{10}(1+d)$ instead, so the two are not on one scale.

## COMPACK

`crystal_analysis.py::MolCrystalAnalysis.batch_compack` converts molecules to unit cells, writes `compack_*.cif` for the requested indices, and fans `::single_compack_run` over a process pool. Each worker reads reference and test CIFs with `ccdc.io.CrystalReader`, builds a `PackingSimilarity` engine with `settings.packing_shell_size = 20`, and returns `(result.rmsd, result.nmatched_molecules)`: the RMSD of the overlaid shells and how many of 20 molecules matched; distance and angle tolerances and `allow_molecular_differences` stay at engine defaults. Because the call compares many tests against one reference, `compare.py::compack_confirm` invokes it once per query, with that query as the reference.

Inside the worker, `single_compack_run` catches `AttributeError`, prints `Analysis failed`, and returns `0, 0`. That is by value indistinguishable from a hit with zero molecules matched; `compack_confirm` says so in its docstring, and `calibrate_basin_metric.py` drops rows with `0` matched and `0.0` RMSD before binning.

## Two uses of one distance

`collate_prior.py` thins a search output to anchors with `clustering.py::greedy_bottom_up_anchors2` at `d_cut = 10 ** log_noise_range[1]`. That number comes from `utils.py::new_calibrate_prior_noise`, which noises latents, regresses $\log_{10}|\Delta\text{reward}|$ on $\log_{10}$ of the latent displacement (`geometry_utils.py::simple_latent_distance`, Euclidean with wrapped angular rows), and inverts the fit at `low_cut` and `high_cut` in units of $kT$. `low_cut` is 0.05 and `high_cut` is 10.0 in the signature. `log_noise_range[1]` is a *latent-space* displacement, the one at which the energy moves by `high_cut` $kT$, and the thinner applies it in that space, through `torch.cdist`, which is unwrapped Euclidean, while the calibration measured its displacements with the wrapped `simple_latent_distance`. `compare.py` loads the same scalar from the prior file and uses it as a threshold on `compute_rdf_distance`, printing it as `rdf match cutoff` at load and as the `basin cutoff` in the match counts. The docstring's name for the second use, `rdf_match_cut`, is not a symbol; the code holds it in `match_cuts`.

Other cuts are module constants: `catchment.py::RETURN_CUT` is 0.10, and `rarefaction.py::CUT` is 0.10, passed to `fcluster` on average linkage. The `RETURN_CUT` comment in `catchment.py` records that the optimiser relaxes an exact structure to 0.046 to 0.050 from its own reference, and the constant sits above that band. `new_analysis.py` carries none: it probes up to 50 samples against the whole set and takes `d.quantile(0.15)` as `d_cut`, with `d_kernel = d_cut / 3`, so the radius moves with the run. `::reference_proximity` reports each experimental form's nearest-sample distance as a ratio to that `d_cut`, beside `polymorph_basin_index`, an argmin with no threshold.

## Calibrations

- **[calibration, `calibrate_basin_metric.py` on acridine sg14-Z'2, 2026-08-25]** `envwise` distance below 0.085 is the same packing; 0.085 to 0.147 is 29% ambiguous; above 0.147, none matched. Recorded in the `landscape_report.py` module docstring, which also states that the cut is not to be carried to another combination unchecked.

`rdf_cutoff_sweep.py` records that this calibration was measured at a 10 Angstrom RDF cutoff and does not transfer to another cutoff.

## RDFs across code versions

`compare.py::ensure_rdf` keys its cache on file, sample count and `RDF_KWARGS`; `new_analysis.py` treats a cached result whose `rdf_mode`, `std_orientation`, `lj_coeff`, `train_T` or `legacy_triclinic_walls` differs from the current configuration, or is absent, as stale. Neither is keyed on the version of `crystal_rdf.py`. The only shape condition on a distance is inside `compute_rdf_distance`, an assert that the two arguments have the same number of channels; two tensors that agree there return a number whatever produced them. `levels.py` states the same about `RDF_KWARGS`: an RDF computed under different settings still subtracts cleanly and still returns a number.

## Sliced Wasserstein

`eval/evaluations.py::sliced_wasserstein` does not compare structures. It takes two clouds of *latents* in `std_params` space, zeroes the projection rows whose variance over the combined cloud is below $10^{-12}$, renormalises the directions, and averages the order-$p$ gap between matched quantiles of the two projected clouds; the sizes need not match. `train.py` calls it with `n_proj=500` and a seeded generator, once against a fresh prior draw and once between two independent prior draws of the same size, logging `wass`, `wass_null` and `wass_debiased = raw - null`; the null estimates the positive finite-sample floor at which two equal-size empirical clouds of one distribution sit. The same construction runs against the anchor buffer, split into two disjoint draws with the sampler subsampled to match, as `wass_anchor`, `wass_anchor_null` and `wass_anchor_debiased`. `eval/wass_debiased` is listed in `analysis/keys.py::TOPLINE_MLE` and appears as a stage exit term, `below: 0.015`, in the shipped stage protocols, for example `configs/a100_stab_aug16/base_uncond.yaml`. Without the variance mask every reading scales by $\lVert\theta_{\text{live}}\rVert$, which does not cancel in the difference.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:rdf_mode`, `cfg:std_orientation`, `cfg:max_n_clusters`, `cfg:num_samples`, `cfg:analysis_batch_size`, `cfg:reload_results`. All six appear in `eval/paper1_results/new_analysis.yaml`, and `cfg:rdf_mode` is per run there with no global default.

## Could be tooling

A mode-and-version stamp could travel with the RDF tensor rather than the results file: an equality check on mode, cutoff, bin count and the `crystal_rdf.py` version would raise where today a number is returned. The COMPACK failure could carry a distinct sentinel, so that dropping engine failures is not left to each caller's filter, and the anchor-thinning radius and the RDF identity cut could be two named fields in the prior file, one in latent units and one in EMD units.

## Sources

Code, mxtaltools: `mxtaltools/analysis/crystal_rdf.py` (`::crystal_rdf`, `::get_1WL_env_labels`, `::get_atomwise_dists`, `::get_elementwise_dists`, `::smooth_1d_histogram`, `::smooth_cutoff`, `::compute_rdf_distance`, `::earth_movers_distance_torch`, `::compute_rdf_distmat`, `::compute_rdf_distmat_parallel`); `mxtaltools/dataset_utils/data_class_methods/crystal_analysis.py` (`::MolCrystalAnalysis.compute_rdf`, `::MolCrystalAnalysis.batch_compack`, `::single_compack_run`); `mxtaltools/common/clustering.py::greedy_bottom_up_anchors2`; `mxtaltools/common/geometry_utils.py::simple_latent_distance`.

Code, gfn_diffusion: `eval/evaluations.py::sliced_wasserstein`; `utils.py::new_calibrate_prior_noise`; `data_processing/collate_prior.py`; `analysis/keys.py::TOPLINE_MLE`; `train.py`; `eval/nikos_comparison/` (`compare.py::compack_confirm`, `::ensure_rdf`, `levels.py::RDF_KWARGS`, `catchment.py::RETURN_CUT`, `rarefaction.py::CUT`, `calibrate_basin_metric.py`, `landscape_report.py`); `eval/paper1_results/` (`new_analysis.py::reference_proximity`, `::run_analysis`, `generate_figs.py::require_rdf_mode`, `rdf_cutoff_sweep.py`, `new_analysis.yaml`).

The code above read at the stamped commit (`a637e70`; the mxtaltools working tree beside it), and its module docstrings. Memory files reference_rdf_basin_metric_calibration, project_stored_rdfs_not_comparable_across_code_versions, feedback_wass_not_interpretive_on_crystals and feedback_nearest_cluster_is_not_membership located the code and were not used as evidence.
