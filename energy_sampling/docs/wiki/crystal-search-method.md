# Crystal search method

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The brute-force crystal structure search is the batched local optimiser in `mxtaltools/crystal_search/`. It takes one conformer, a list of space groups and a list of $Z'$ values, instantiates many crystals per combination, initialises each at random, and relaxes all of them at once by gradient descent on an energy. Its output is a list of `MolCrystalData` objects on disk: the anchors a prior dataset is thinned from, and the landscape a comparison against known polymorphs is drawn on. This page covers the mechanics and what a single-start local descent can return. The landscape itself is [landscape-structure](landscape-structure.md); comparing two structures is [structure-comparison-metrics](structure-comparison-metrics.md); the walls the reduction penalty evaluates are derived on [cell-fundamental-domains](cell-fundamental-domains.md).

## Initialisation

`crystal_search/utils.py::init_samples_to_optim` builds the work list: for each molecule, each space group in `cfg:sgs_to_search` and each $Z'$ in `cfg:zp_to_search`, `cfg:num_samples` `MolCrystalData` objects with placeholder cell parameters, then shuffled. `cfg:init_sample_method` `'data'` or `'in_config'` instead optimises crystals supplied ready-made.

`crystal_search/utils.py::get_initial_state` fills the cell parameters per batch, seeded at `cfg:opt_seed + 10000 * batch_idx`. On `'random'` with `cfg:init_reduced` it calls `crystal_ops.py::MolCrystalOps.sample_random_reduced_crystal_parameters`, otherwise `::sample_random_crystal_parameters`; `'reasonable'` calls `::sample_reasonable_random_parameters`. The reduced sampler draws cell angles first and lengths second (lengths are drawn at a target packing coefficient and so depend on the angles), projects with `geometry_utils.py::enforce_crystal_system`, evaluates `crystal_analysis.py::MolCrystalAnalysis.compute_cell_reduction_penalty` at its default `margin=0.0`, and keeps only rows at exactly zero, resampling the rest until every row is valid. Centroids, orientations and, at $Z'>1$, handedness are drawn once afterwards, outside the rejection loop. `cfg:init_target_cp` sets the packing coefficient target, as a float or one of two distributions.

## The descent loop

`crystal_ops.py::MolCrystalOps.optimize_crystal_parameters` clones the batch, orients the molecules, and hands the full cell parameters to `crystal_opt_utils.py::gradient_descent_optimization`. The free variables sit in `crystal_opt_utils.py::CrystalParams`, one `nn.Parameter` per row, with any `fixed_dims` columns held as a buffer. Each iteration writes the current parameters onto a fresh clone of the initial batch, calls `::clean_cell_parameters` at `mode='hard'` (lengths clipped at 3 Angstrom, angles bounded around $\pi/2$, centroids into the aunit box), then scores through `crystal_analysis.py::MolCrystalAnalysis.analyze`.

`crystal_opt_utils.py::compute_loss` selects the per-row objective by `cfg:opt.optim_target`: an energy column (`lj`, `qlj`, `elj`, `silu`, `ellipsoid`, `uma`, `mace`), the reduction penalty (`reduce`), a score-model output, an RDF distance (`rdf_dist`), a latent distance (`latent_dist`), or a centroid separation (`inter_overlaps`). `::compute_auxiliary_loss` adds, where configured, a squared packing-coefficient term, a box restriction $\sum 80000/\ell^{12}$ on asymmetric-unit lengths, a compression term gated below packing coefficient 0.65, an umbrella repulsion against a stored latent record, and under `cfg:opt.enforce_reduced` a $10^4 \times \mathrm{relu}(\text{reduction\_en})$ penalty. The row mean is backpropagated, gradients clipped to `cfg:opt.grad_norm_clip`.

`cfg:opt` is a list, and `crystal_search/run_search.py::crystal_search` runs its entries in order, collating one stage's output as the next stage's input batch.

## Convergence, the step cap, and batch composition

The loop runs while `s_ind < max_num_steps - 1` and `not converged.all()`. From step 50 onward (`min_num_steps`, a constant inside `gradient_descent_optimization`, not a config key), `crystal_opt_utils.py::check_convergence` takes an EMA of the parameter trajectory (`::ema_trajectory`), measures the mean absolute step-to-step change over the last 50 smoothed steps per row, and marks a row converged below `cfg:opt.convergence_eps`. Above 95% of rows marked, the flag is filled `True` for every row and the loop ends for all of them.

Stopping is therefore a property of the batch, not of a row: a row that meets the criterion keeps stepping until the batch stops, and a row that has not met it stops when the quorum does. The learning rate is likewise one schedule for the whole batch: a single `MultiplicativeLR` over the one optimizer, whose factor is 1 unless `cfg:opt.anneal_lr`, in which case one annealing factor shrinks every row's step size together. Which structures share a batch is set by `cfg:batch_size`, by the shuffle in `init_samples_to_optim`, and at runtime: an OOM saves the best parameters so far to `opt_intermediates.pt` and re-raises, after which `crystal_search` cuts the batch size to 0.9 and restarts the set from them (`::recover_opt_state`); `cfg:grow_batch_size` multiplies the size by 1.2 between sets. A row returns not its final state but its best: `argmin` over the recorded per-step loss picks a step index out of `params_record`, and the batch is re-analysed there.

## One start is a local minimum

Gradient descent from one initialisation returns a local minimum of the objective, in the basin containing that start. Any quantity defined as a minimum over coordinates that are not being scanned, $U_{\text{eff}}(\phi) = \min_{r} U(\phi, r)$, is a global optimisation over $r$ at every $\phi$; one descent from one start returns a local minimum instead, an upper bound on $U_{\text{eff}}(\phi)$ and never a lower one. The error is one-sided, grows with the number of pinned coordinates, and is not reported: the loop terminates on parameter motion, not on any property of the minimum reached. The search has this shape, its scanned coordinate discrete (space group, $Z'$) and its unscanned ones the twelve or more cell parameters.

## Seeding versus sampling for a named basin

The random initialiser covers the latent cube, not a nominated point in it. The parameter vector has twelve components at $Z'=1$ and more above it; the probability that a draw lands within a fixed latent radius of one pre-specified structure falls off with that dimension, and the expected hits are the sample count times that probability times the probability of descending back to the structure from there. Seeding sets the first factor to one: a draw placed at a chosen latent displacement from the target and relaxed leaves only the return probability. Basin width in the return sense and basin measure under the initialiser are on [landscape-structure](landscape-structure.md).

## The reduced-cell search mode

Two independent switches act on the cell domain. `cfg:init_reduced` makes the initialiser hard: rows are resampled until the reduction penalty is exactly zero at `margin=0.0`, so every start lies in the closed fundamental domain. `cfg:opt.enforce_reduced` acts during and after descent: `reduction_en` joins the analysis computes, the penalty is evaluated at `margin=0.01` rather than 0 (a shrunken zero set), the $10^4$ term enters the loss, and the returned list is filtered to rows whose penalty at `margin=0.0` is below $10^{-3}$, so a stage can return fewer rows than it took. Which walls those are is per crystal system and per space group: triclinic gets the Niggli-condition walls by default, with `MXT_LEGACY_TRICLINIC_WALLS=1` restoring the pre-Niggli set, monoclinic the class walls ([cell-fundamental-domains](cell-fundamental-domains.md)). Inside the loop the walls are soft, a term in the loss; the hard cell constraints there are `clean_cell_parameters` and `enforce_crystal_system`.

## Output, chunks and stamps

`crystal_search` writes `<cfg:out_dir>/<cfg:run_name>.pt`, a list of `MolCrystalData` objects, after every set, and resumes from it unless `cfg:force_restart_run`; `cfg:save_trajs` writes the per-stage record, parameter trajectory included, beside it. A wave of searches is a set of run names sharing a stem and ending in an integer, which `data_processing/utils.py::load_search_chunks` reassembles by globbing `<stem>_<digits>.pt`, preferring a chunk corrected under `f047_rescored`. With `require_uma_state` it raises on any chunk whose `uma_energy_state` stamp is missing or below `::UMA_ENERGY_STATE`, and `data_processing/collate_prior.py` passes that flag on the `uma` route, so an unstamped chunk stops a prior build rather than entering it.

A second stamp lives on the training side. `energies/molecular_crystal.py::MolecularCrystal.stamp_lj_coeff` writes the run's `lj_coeff` onto every freshly built batch as a per-graph attribute inside `::analyze_crystal_batch`, mxtaltools `crystal_analysis.py::MolCrystalAnalysis.compute_eLJ_energy` scales its output by it, and `::assert_lj_coeff_stamped` refuses a batch with no stamp or with values disagreeing with the run's, checking every row. The searcher does not stamp its output.

The `uma` and `mace` targets load a predictor in `crystal_search/utils.py::parse_opt_config` and pass it into `analyze`; the routes are [mlip-energy-routes](mlip-energy-routes.md).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

Search config (mxtaltools `configs/crystal_searches/`, not the training config): `cfg:init_sample_method`, `cfg:init_reduced`, `cfg:init_target_cp`, `cfg:opt_seed`, `cfg:num_samples`, `cfg:sgs_to_search`, `cfg:zp_to_search`, `cfg:batch_size`, `cfg:grow_batch_size`, `cfg:out_dir`, `cfg:run_name`, `cfg:save_trajs`, `cfg:force_restart_run`; per stage, `cfg:opt.optim_target`, `cfg:opt.enforce_reduced`, `cfg:opt.compression_factor`, `cfg:opt.cutoff`, `cfg:opt.init_lr`, `cfg:opt.anneal_lr`, `cfg:opt.convergence_eps`, `cfg:opt.optimizer_func`, `cfg:opt.grad_norm_clip`, `cfg:opt.max_num_steps`.

Environment variables: `MXT_LEGACY_TRICLINIC_WALLS`.

Code, mxtaltools (paths under that repo's `mxtaltools/`): `crystal_search/run_search.py::crystal_search`; `crystal_search/utils.py::init_samples_to_optim`, `::get_initial_state`, `::parse_opt_config`, `::recover_opt_state`; `crystal_search/crystal_opt_utils.py::gradient_descent_optimization`, `::CrystalParams`, `::compute_loss`, `::compute_auxiliary_loss`, `::check_convergence`, `::ema_trajectory`; `dataset_utils/data_class_methods/crystal_ops.py::MolCrystalOps.optimize_crystal_parameters`, `::sample_random_reduced_crystal_parameters`, `::sample_random_crystal_parameters`, `::sample_reasonable_random_parameters`, `::clean_cell_parameters`; `crystal_analysis.py::MolCrystalAnalysis.analyze`, `::compute_cell_reduction_penalty`, `::compute_eLJ_energy`; `common/geometry_utils.py::enforce_crystal_system`.

Code, gfn_diffusion: `data_processing/collate_prior.py`; `data_processing/utils.py::load_search_chunks`, `::UMA_ENERGY_STATE`; `energies/molecular_crystal.py::MolecularCrystal.stamp_lj_coeff`, `::assert_lj_coeff_stamped`, `::analyze_crystal_batch`.

## Could be tooling

A batch-composition record is mechanical. `gradient_descent_optimization` already holds `params_record`, the per-step loss and the stopping step, so writing per row the step it first met `convergence_eps`, the step the batch stopped, and which of the three conditions stopped it would make truncation a column in the output. The reduction filter's per-stage drop count is known at the same point and is not recorded either.

The second is a stamp audit: `load_search_chunks` already refuses a stale `uma_energy_state`, and the same pass could report per chunk the search config hash, each stage's optimisation target and `enforce_reduced`, so a prior built from chunks produced under different settings is visible before thinning.

## Sources

The code above, read at the stamped commit (mxtaltools at `d23a71e7`), and the search configs under mxtaltools `configs/crystal_searches/`. Memory files feedback_batch_composition_changes_relaxation_outcome, feedback_constrained_relaxation_returns_false_minima, project_mipcas_niggli_reduced_search_right_angle_pinning, project_mipcas_nig_sep17_cluster_battery and project_search_miss_is_the_initialiser_not_the_basin located code and were not evidence.
