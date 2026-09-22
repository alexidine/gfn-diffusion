# Landscape structure

*Drift: **T** (theory). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

A molecular-crystal energy landscape is treated here as a countable set of *basins* sampled by *captures*, in the shape of a species-abundance problem: each optimised structure is one capture, each basin one species. This page defines those objects, the coverage and richness estimators over them, the two scalars a basin carries, what an embedding of the distance matrix measures, why a pre-specified basin is hard to hit, and the motif descriptors. Left out: how structures are generated, [crystal-search-method](crystal-search-method.md); the distance function and its calibration against COMPACK, [structure-comparison-metrics](structure-comparison-metrics.md).

## A basin is a relaxation endpoint plus an identity cut

The definition is operational, in two stages.

**Relaxation.** `crystal_opt_utils.py::gradient_descent_optimization` holds the cell and pose coordinates in a `::CrystalParams` module and steps them downhill on `::compute_loss` plus `::compute_auxiliary_loss`. `::check_convergence` calls a row converged once the mean absolute step of its EMA-smoothed parameter trajectory (`::ema_trajectory`) over the last fifty recorded steps falls below `convergence_eps`, and marks the whole batch converged once more than 95% of rows pass. An endpoint is a numerically stationary point of the optimiser's own objective under its own stopping rule, not an exact minimum: two starts in one true basin land near each other, not at one point.

**Identity.** Endpoints are compared by a structure distance (`crystal_rdf.py::compute_rdf_distance` on RDFs computed under one fixed setting, `levels.py::RDF_KWARGS`), assembled into a square matrix, agglomerated with `scipy` `linkage`, and cut with `fcluster(criterion='distance')`. A basin is a cluster of that cut. Average linkage merges on the mean distance between groups and bounds neither a group's diameter nor its separation from the next, so a group can chain until its mean internal distance exceeds its distance to a neighbour. Complete linkage merges on the maximum pairwise distance, so every group has diameter at most the cut, at the price of splitting groups whose members are jointly far apart.

A basin count is therefore a statement about the triple (metric, cut, linkage), and labels are portable across none of the three. Both linkages are in use: the coverage and motif scripts (`landscape_report.py`, `rarefaction.py`, `rarefy_sweep.py`, `richness_sweep.py`, `packing_motifs.py`) take average, the atlas scripts (`atlas/tight_regions.py`, `atlas/coarse_landscape.py`) take complete. `calibrate_basin_metric.py::main` measures the probability that a pair at a given distance is a confirmed match, per RDF mode, against `compare.py::compack_confirm`.

## Coverage, richness, and the discovery curve

Let $n$ be the captures, $S$ the distinct basins observed, $n_i$ the occupancy of basin $i$, and $f_k$ the number of basins captured exactly $k$ times.

**Good-Turing mass coverage.** Under multinomial sampling of independent captures the expected total probability of the species absent from a sample of size $n$ is $\mathbb{E}[f_1]/n$, so $C = 1 - f_1/n$ (`richness_sweep.py::report`) estimates the fraction of *probability mass* seen. It is not a fraction of species, and it is a property of the sample: $f_1/n$ falls as $n$ grows, so one landscape reports a larger $C$ from a larger draw.

**Chao1 richness.** From the first two frequency counts,

$$\hat S = S + \frac{f_1^2}{2f_2}, \qquad \hat S = S + \frac{f_1(f_1-1)}{2}\ \text{ when } f_2 = 0,$$

in `landscape_report.py::chao` and inline in `richness_sweep.py::report` and `rarefaction.py::chao_project`. It infers unseen species from the singleton-to-doubleton ratio: unbiased when the rare species share one detection probability, a lower bound on richness otherwise.

Both assume captures are independent draws from a fixed distribution over basins. A pool thinned by near-duplicate removal (`clustering.py::greedy_bottom_up_anchors`, which walks structures in ascending energy and keeps one only if its latent distance to every kept structure exceeds a cut) violates that by construction, reading as one capture per basin and complete coverage. A pool built from each arm's lowest-energy structures violates it in the occupancy direction instead.

**Rarefaction.** The expected distinct basins in a subsample of $m$ of the $n$ captures, without replacement, is closed form,

$$\mathbb{E}[S(m)] = S - \sum_i \binom{n-n_i}{m} \Big/ \binom{n}{m},$$

evaluated in logs through `lgamma` (`rarefaction.py::logC`, `::rarefy`). Its difference quotient is the marginal discovery rate, and whether that rate falls with $m$ *at fixed cut* separates a bending curve from a linear one. Extrapolation past $n$ is parametric: with $f_0 = \hat S - S$,

$$S(n+m) = S + f_0\Big[1 - \big(1 - \tfrac{f_1}{n f_0 + f_1}\big)^{m}\Big]$$

(`rarefaction.py::chao_project`). A hold-out fit on a random fraction of the captures, compared against the observed $S(n)$, bounds the reach at which the projection has been tested; past that reach the abundance structure of unseen rare basins is unconstrained by the sample.

## Occupancy and depth

A basin carries two scalars. **Occupancy** is its capture count: a property of the map from initialiser to endpoints, the measure the starting distribution places on the basin's catchment times the probability that relaxation from there returns to it, and so a function of the initialiser, the optimiser and the step budget. **Depth** is the minimum energy over its members: a property of the energy surface at the endpoint, independent of how the basin was reached. Nothing in the construction ties them. Occupancy is estimable only from a random sample of captures.

## What a two-dimensional embedding measures

Classical multidimensional scaling turns the distance matrix into coordinates: with $J = I - \tfrac1n \mathbf{1}\mathbf{1}^{\mathsf T}$, double-centre the squared distances, $B = -\tfrac12 J D^{\circ 2} J$, and set $X_k = V_{:,1:k}\,\mathrm{diag}(\sqrt{\lambda_{1:k}})$ from the leading eigenpairs. `atlas/mds_map.py::main` and `atlas/projection_quality.py::main` score it by Kruskal stress-1, $\sqrt{\sum(d-\hat d)^2/\sum d^2}$, and by the correlation between true and embedded distances; a high correlation with poor stress means the order of similarities survives while the scale is compressed.

Two tests in `atlas/tight_regions.py::main` ask what a tight region is. **Density:** each point's mean distance to its $K$ nearest neighbours in the layout against the same quantity in the true matrix, scored by Pearson $r$. High $r$ means local density on screen tracks local density in the metric; low $r$ means projection collapse. **Partition recovery:** cut the layout into exactly as many groups as the metric clustering produced (`fcluster(linkage(pos, method='complete'), t=S, criterion='maxclust')`) and compare labellings by adjusted Rand (`::ari`). Matching the group count makes this a comparison of partitions rather than of resolutions.

The density follows from the first stage of the basin definition. The points are relaxation endpoints, so a distance of essentially zero repeated many times is one structure reconverged to from many independent starts, and the count of points in a knot is that basin's occupancy. On unrelaxed samples there is no reconvergence, so every point is its own singleton.

## Hitting a named basin in $d$ dimensions

The optimisation coordinates are the latent vector whose layout `crystal_ops.py::MolCrystalOps.latent_to_cell_params` records as three cell lengths, three cell angles, $3Z'$ centroid components and $3Z'$ orientation components, so $d = 6 + 6Z'$. Write $R$ for the characteristic radius of the initialiser's distribution there and $\rho$ for a named basin's catchment radius, the displacement beyond which relaxation no longer returns to it, measured by displace-and-relax around a known structure with return judged at a distance cut (`catchment.py::main`, `::RETURN_CUT`).

One proposal lands inside the catchment with probability scaling as the volume ratio $(\rho/R)^d$, so expected hits over $n$ proposals go as $n(\rho/R)^d$ times the return probability inside. The typical spacing $s$ from a proposal to its nearest other proposal solves $n(s/R)^d \approx 1$, giving $s \approx R\,n^{-1/d}$. Hitting the named basin needs $\rho \gtrsim s$, that is $n \gtrsim (R/\rho)^d$: the budget is exponential in $d$ while the spacing it buys shrinks only as $n^{-1/d}$. Landing in *some* basin needs only the union of catchments to cover the sampled region, which many modest basins achieve while no single one is large. `init_bias_check.py` places the catchment radius and the initialiser's distance distribution in one space, separating a small catchment from a mis-centred initialiser; its null for the second is proposal-to-nearest-other-proposal, every output descending from a proposal.

## Motif classes

For a rigid planar aromatic a basin representative can be described by how molecular planes relate. `packing_motifs.py::motif_of` builds the periodic cluster around a reference molecule (`crystal_building.py::MolCrystalBuilding.mol2cluster`), fixes a frame on the reference from the singular vectors of its centred coordinates (`::ref_frame`: long in-plane axis, short in-plane axis, normal; `::planarity`, built on `::plane_normal`, is the out-of-plane residual guarding a non-planar input), and per neighbour inside the centroid cutoff emits $\theta$, the angle between plane normals folded to $[0,90]^\circ$; $h$, the centroid offset along the reference normal, the stack height; $s$, the in-plane offset, the slip; plus the centroid distance and the signed frame components. Two neighbours at $\theta \approx 0$ are separated by $h$: $h$ at a stacking separation is a $\pi$-stack, $h \approx 0$ is two molecules side by side in one sheet.

`packing_motifs.py::classify` labels the whole coordination shell on two axes rather than by majority vote over neighbours. Axis one: does any neighbour satisfy $\theta < 25^\circ$ and $3.1 \le h \le 3.9$ Å, a $\pi$-stack. Axis two: the median $\theta$ of the remaining neighbours ($\theta \ge 25^\circ$) against $45^\circ$. The classes are gamma (stack plus edge-to-face), beta (stack plus shallow tilt, stacked layers), herringbone (no stack, edge-to-face) and sheet (no stack, coplanar). The module docstring carries a second, per-neighbour taxonomy with its own bands, which `classify` does not use. The boundaries are conventional; the label names a region of the angles and distances.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

None; nothing here runs from the training config. The identity cut is a script flag (`--cut` on `landscape_report.py` and `packing_motifs.py`) or a module constant (`CUT` in `rarefaction.py` and `atlas/tight_regions.py`, `CUTS` in `rarefy_sweep.py`, `RETURN_CUT` in `catchment.py`). Energy strata are `--low-energy` and `LOW`; RDF modes and distance bins are `MODES`/`BINS` in `calibrate_basin_metric.py`. The distance setting is `levels.py::RDF_KWARGS`.

## Sources

The code cited above, read at the stamped commit: scripts under `energy_sampling/eval/nikos_comparison/`, and in mxtaltools `crystal_search/crystal_opt_utils.py`, `analysis/crystal_rdf.py`, `common/clustering.py`, `dataset_utils/data_class_methods/crystal_building.py` and `crystal_ops.py`. `docs/design/` carries no derivation of crystal landscape structure, so none is stated here. Memory files located the code and were not used as evidence.
