# Conformer force field and prior

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A conformer problem has no crystal and no periodic neighbour list: the reward is built from one molecule's intramolecular energy, and the states the trainer needs before a policy exists come from a fitted prior over conformations. This page is about those two objects and the statistics that say where a set of drawn conformations sits relative to the target. Left out and named: the spanning-tree chart and the internal coordinates ([conformer-chart-and-internal-coordinates](conformer-chart-and-internal-coordinates.md)); the levels, stages and losses of a run ([conformer-training](conformer-training.md)); the frozen graph encoder ([molecule-encoder](molecule-encoder.md)).

Throughout, $U$ is the force field's energy in kcal/mol, $T$ the sampling temperature in the same units, and $d$ the width of the latent state $x \in [-1,1]^d$, which `ConformerTorsions` carries as `data_ndim`.

## The force field: MMFF94 in torch

`cfg:energy_config.force_field` selects which of two builders `conformer_torsions.py::ConformerTorsions._make_ff` calls. At `'mmff'` it is `mxtaltools/conformers/energy.py::ff_from_mmff`, a complete copy of MMFF94: bond stretch, angle bend, stretch-bend, out-of-plane, proper torsion, van der Waals and electrostatics, each in MMFF's own functional form and evaluated by `energy.py::intramolecular_energy` from Cartesian positions. Bonds are harmonic times MMFF's quartic factor (`ForceField.bond_cs`); angles are harmonic times the cubic factor (`angle_cb`), with rows flagged in `angle_linear` switching to the $2k(1+\cos\theta)$ form MMFF uses at a linear centre; torsions are MMFF's three-term Fourier expansion written as three rows of $V[1 + \cos(n\phi - \gamma)]$; out-of-plane runs over the Wilson angles at a three-coordinate centre, with a force constant MMFF allows to be negative and the code does not clamp; van der Waals is `energy.py::buffered_147`, exact as published at `vdw_softcore_frac` zero and continued linearly below `softcore_frac * rstar` otherwise; electrostatics reuse `pair_index` and inherit its separation rule, with a 0.75 scale on 1-4 pairs (`MMFF_ELE_14_SCALE`) and the buffering constant `ele_delta`.

Parameters come from RDKit's MMFF94 typing through `AllChem.MMFFGetMoleculeProperties` and its per-term accessors, read off connectivity and bond orders rather than geometry, so the field is a function of the molecular graph alone. The van der Waals lookup takes the third and fourth returns of `GetMMFFVdWParams`, the donor-acceptor rescaled $R^*$ and $\epsilon$.

Verification is per term rather than on the total. `tests/conformer/test_mmff_matches_rdkit.py::test_every_term_matches_rdkit` switches every RDKit term but one off and compares at a *perturbed* geometry, so a term near zero cannot pass by being small: tolerance $3\times10^{-4}$ kcal/mol for angle, stretch-bend and out-of-plane, which the file attributes to the decimal precision RDKit's accessors report, and $10^{-9}$ for the other four. `::test_total_matches_rdkit` checks the sum at $10^{-3}$, and `::test_vdw_donor_acceptor_rescaling_is_load_bearing` asserts the un-rescaled parameters fail.

The other branch of `_make_ff` is `energy.py::ff_from_reference`, which measures $r_0$ and $\theta_0$ off the embedded conformer and carries no torsion term: bonds, angles, a ring-closure restraint and soft-core 12-6 pairs (`energy.py::soft_core_lj`), with its parameters a function of the embedding. `energy.py::ff_from_graph` adds proper torsions to that set from hand-written tables keyed on `(element, degree)` covering four bond types, six angle types and two torsion centres, with `energy.py::_lookup` raising on any key outside them; no code path in this trainer calls it, and its own docstring still says it carries no torsion term. `ForceField` separates a field that does not carry a term from one carrying it as zero: every MMFF-only field is `None` on both non-MMFF builders, and `intramolecular_energy` branches on `is None` for each.

## Construction cost, tiling, and the batch cache

A batch here is one molecule replicated, and the field is rebuilt for it rather than evaluated per copy. `builder.py::collate` dispatches on object identity of the specs: `[spec] * n` takes `builder.py::_collate_replicated`, which tiles molecule zero's index arrays with an atom offset in a fixed number of array operations, and distinct-but-equal specs fall through to the loop. `ff_from_mmff` does the same for parameters: `energy.py::_repeated_block` returns the first block and a repeat count when the reduced pair rows are one block repeated, so the RDKit lookups run once and are tiled, and the repetition is checked element by element, never inferred from `n_mols`. `mxtaltools/conformers/tests/test_conformers.py::test_mmff_pair_parameters_are_the_single_molecule_answer_tiled` asserts the $n$-copy field equals the $n=1$ field repeated.

`ConformerTorsions._batch` caches the collated tree and the force field on the device, keyed on batch size, and `::potential_energy` and `::jacobian_energy` call it before use. The bound is in samples, not entries: the entry just requested is kept and moved to most-recently-used, then least-recently-used entries are dropped while more than one entry remains and the summed batch sizes exceed `_batch_cache_slack` (1.5) times the requested size.

## The prior over conformations

`mxtaltools/conformers/prior.py::InternalPrior` is a table of per-type marginals fitted from observed conformers. Types are `(element, degree)` tuples; `::bond_key` takes the sorted pair, `::angle_key` keeps the apex type and only the elements of the outer atoms, `::torsion_key` takes the central bond alone. Each marginal is a `::Histogram1D` over the spans `R_RANGE`, `THETA_RANGE` and `PHI_RANGE`, the last flagged periodic, mixed toward uniform by `fatten` (0.15), so `sample` and `log_prob` describe the same density and it has full support.

`ConformerTorsions.sample_prior_states` assembles a draw in the chart's own degrees of freedom, keyed on the placement-slot numbering of the spanning tree and the references `r0`, `th0`, `ph0` measured on it. Ring systems go first: a connected component of non-bridge bonds is one block, drawn jointly from a fitted pucker subspace or a discrete bank when `ring_blocks` resolves one, held planar where aromatic, and otherwise rattled about the reference at `ring_jitter_scale` times thermal width; the bank-to-block correspondence is positional over the block's non-$r$ rows, and a mismatched kind sequence raises rather than permuting. Sibling torsions about a shared parent are grouped by `::torsion_groups`, every member taking the leader's angular displacement, with improper rows excluded. Remaining acyclic rows draw from their type's marginal; $r$ and $\theta$ may instead draw at `::thermal_rtheta_sigma`, $\sqrt{kT/2k}$ from the force field's own constants. `joint_rings=False` gives every ring row an independent marginal and is documented as a negative control, with `stats['closure_err']` measured on both arms.

## Clash and descent repair

A product of marginals over torsions lets a long chain intersect itself, and the excess lands in the pair term rather than in any bonded term. `conformer_modeller.py::ConformerModeller._draw_prior_states` optionally repairs it with `cfg:energy_config.prior_relax_steps` steps of `energies/prior_baselines.py::descend`, chunked; `descend` holds an autograd graph over the batch. The code default is 0. `ConformerModeller.init_anchor_buffer_seed` runs the same descent at `cfg:buffers.anchor_buffer.seed_relax_steps` on the anchor seed.

`descend` optimises the **state** $x$, not the degree-of-freedom rows: $x$ is the level's own coordinate, so a step cannot leave the level's manifold and there is nothing to project back, including where one state column drives several dihedral rows. The default optimizer is `'rprop'` (`torch.optim.Rprop`, `lr` 0.02; `'adam'` is available at 0.05), and the function's first docstring line still names Adam. It returns the best point seen rather than the last, and clamps $x$ into $[-1,1]$ after each step. `prior_baselines.py::tier_minimum` is the same descent from many starts, 150 steps by default, returning best and worst over starts: a multi-start local search, hence an upper bound on the true minimum, so every excess built on it is a lower bound.

## Thermal target, coverage, ESS

Excess is $(U - U_{\min})/T$, and

$$T_{\text{eff}}/T \;=\; 1 + \frac{2\,\mathrm{median}(\text{excess})}{d}.$$

Equipartition puts the median excess of a harmonic system at $d/2$. Three sites compute a quantity under this name and they do not share a zero.

- `energies/conformer_eval_metrics.py::thermal_stats` takes $U_{\min}$ from `tier_minimum` and applies the formula as written, so a correctly thermal sample reads 2.0. It publishes `frac_within_equipartition`, the fraction with excess at or under $d/2$, beside it.
- `prior_baselines.py::cell` feeds the same formula from `::excess_kt`, which is $(U - \text{zero})/T - d/2$; the subtracted $d/2$ shifts the reading down by one, so a thermal sample reads 1.0 there.
- The two print sites in `conformer_modeller.py` (`_draw_prior_states` reporting and `init_anchor_buffer_seed`) use the median minus the batch's own minimum, in kcal/mol, and their message states 2.0 as thermal.

`energies/prior_smoke.py` additionally bands a per-term $T_{\text{eff}}$, $2\langle E_{\text{bond}}\rangle/(nkT)$ over free non-ring bonds and the same over angles, where equipartition fixes the value at 1; the bands are 0.70 to 1.40 and 0.70 to 1.50.

The whole-energy statistic is degenerate on a raw prior draw with free $r$ and $\theta$: the draw width and the score use the same force constants, which cancel. `prior_baselines.py::teff_is_degenerate` marks exactly those cells and excludes optimised draws, uniform draws and levels where $r$ and $\theta$ are frozen.

*ESS.* `prior_diagnostics.py::ess_fraction` is the Kish fraction $\left(\sum w\right)^2 / (n \sum w^2)$ on self-normalised log weights, reported with bootstrap intervals and with $D = \log(1/\text{ESS fraction})$ in nats.

*Coverage.* `prior_diagnostics.py::basin_reference` enumerates the target's rotamer basins as the product of per-group leader centres, realises each with the remaining rows at the reference, scores it, and marks it accessible within `accessible_kt` (10.0) of the best. `::coverage_report` assigns draws by nearest centre and reports which accessible basins received none, with a binomial upper bound $3/n$ on an unobserved basin's probability. The two run in opposite directions: ESS is a functional of the draws obtained, so a basin never proposed contributes no large weight and no warning; coverage enumerates the target's basins first and asks what the sampler assigns them. `basin_reference` is sampler-independent, and `::rotamer_basin_labels` with `conformer_eval_metrics.py::basin_coverage` applies the same partition to a policy's samples.

## Wells and basins

A *dihedral well* is an interval of one projected torsion coordinate between barriers of that projection. A *basin* of the full surface is the set of points from which descent on $U$ over all $d$ coordinates reaches one minimum. They are different partitions: a barrier present in the projected coordinate need not be present in the full surface, so a monotone descent can begin in one dihedral well and end in another without crossing anything. `rotamer_basin_labels` labels the first; `descend` endpoints define the second. How deterministic the descent map is, is the conditional entropy $H(b_K \mid \phi_0, \text{bond})$ of the descent label over $K$ independent draws of the coordinates left free by a fixed torsion vector, in bits against the uniform value $\log_2 k$ for a $k$-rotamer bond. No such statistic is computed in this repository; the labels and the descent it would be built from are.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:energy_config.{smiles, level, force_field, internal_prior_path, prior_relax_steps, prior_sample_size, energy_clip, temperature, log_temperature}`; `cfg:buffers.anchor_buffer.seed_relax_steps`.

Code, gfn_diffusion: `energies/conformer_torsions.py::ConformerTorsions`; `conformer_modeller.py::ConformerModeller`; `energies/prior_baselines.py`; `energies/prior_diagnostics.py`; `energies/conformer_eval_metrics.py`; `energies/prior_smoke.py`; `build_conformer_prior_dataset.py`; `tests/conformer/test_mmff_matches_rdkit.py`.

Code, mxtaltools: `mxtaltools/conformers/energy.py`; `mxtaltools/conformers/builder.py`; `mxtaltools/conformers/prior.py`; `mxtaltools/conformers/tests/test_conformers.py`. Individual symbols are cited inline above.

## Could be tooling

The descent-reproducibility entropy above has no implementation; `rotamer_basin_labels` and `descend` are the pieces. `energy.py::worst_clash` and `energy.py::closure_error` are diagnostics on the force field reached only from `mxtaltools/conformers/demo.py`; the closure figure the evaluation path reports comes from a different symbol, `energies/ring_metrics.py::closure_error`. `vdw_softcore_frac` is a constructor argument of `ff_from_mmff` with no config key reaching it.

## Sources

Repo: docs/design/conformer_parameterisation.md, and the code above read at the stamped commit. Agent memory files on the conformer force field, prior draw cost, descent and thermal statistics located the code and were not used as evidence.
