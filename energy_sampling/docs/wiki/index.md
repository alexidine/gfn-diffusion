# GFN knowledge base

Forty-nine pages, each drafted from the code and the derivation notes, verified against the code at the stamped commit, and reviewed against the writing protocol. Owner-choice sections are placeholders until the owner writes them. Not yet written: acridine-reference-system, experiment-design; dropped as retired: lr-selection-protocol, lr-sensors, controller-bench.

## Theory of the objective

- [trajectory-balance](trajectory-balance.md) — theory page: the TB residual and Huber loss, what zero and variance imply, the A/B/C level decomposition, the ELBO floor, the gradient routes.
- [vargrad](vargrad.md) — theory page: the VarGrad objective from the log-weight up, the estimator, the Huber influence, condition grouping, the zero-sum coefficients.
- [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md) — theory page: the two streams and their levels, log Z as the coupling in TB, the pooled term and level tether in VarGrad, buffer absorption, freezing P_B.
- [flat-directions-and-limit-cycles](flat-directions-and-limit-cycles.md) — theory page: the two flat directions of the TB parameterisation, what the residual sees along each, the variance schedule in v-time, and the telemetry where a flat direction meets a lagged force.
- [gradient-routes-and-terminal-force](gradient-routes-and-terminal-force.md) — theory page: the explicit and pathwise gradient routes on a Gaussian step, truncation of the Jacobian product, the sampler/non-sampler asymmetry, the terminal reward force and stored-force replay, the terminal variance floor.
- [loss-composition](loss-composition.md) — mixed page: the three branches, the fused weighted sum, the full coefficient inventory, the blindness bound, held-out quantities, dead keys.
- [mle-phase](mle-phase.md) — mixed page: the `train_prior` MLE warm start, its objective and two gradient channels, the level exit gate, the exit snapshots and what a resume re-enters.
- [lambda-annealing](lambda-annealing.md) — theory page: the flow/physical energy mixture, the exponential family the mixing scalar indexes, the anneal event that moves it, and what a rung injects into the residuals.
- [z-calibration](z-calibration.md) — mixed page: the closed-form log Z estimators, the level fill and its two modes, the Z-only servo, the stage-entry bootstrap, and the conditional Z(c) path.

## Conditioning

- [conditional-route](conditional-route.md) — mixed page: the condition channel into policy and flow head, the scramble seam, condition identity and which conditions each draw visits, the per-condition normaliser and its three Z settings, the held-out stream and the per-condition metric families.
- [prior-density-models](prior-density-models.md) — mixed page: the kNN and flow density models fitted to prior draws, their coordinates and geometry checks, the kNN dimensional bias, the ground truth available, the slope calibration gate, and how a fitted density enters the reward.
- [molecule-conditions-and-anchors](molecule-conditions-and-anchors.md) — mixed page: the config keys naming the condition, prior and held-out files, what each file must hold, the identifier registry and condition ids, the QM9 and anchor build scripts with their frame and split rules, and the prior path's place in problem identity.
- [molecule-encoder](molecule-encoder.md) — mixed page: the 2D-graph encoder behind the conformer condition, its inputs and structural encodings, the attention and broadcast global steps, the probe battery's targets, scoring and skeleton-grouped split, and the frozen cache with its atom-order and checkpoint stamps.

## Training dynamics and control

- [learning-rate](learning-rate.md) — mixed page: the five rate keys and the optimizers they reach, the run-level scale and its burn-in and promotion legs, the hard-failure bars and the fire/rewind path, the per-stage drawdown sensor, and what the width and length scaling argument derives.
- [gradient-clipping](gradient-clipping.md) — mixed page: the per-branch adaptive clip bar and its one policy clip site, the static derived clip and where it survives, the loss-value and reward-gradient clips upstream, and what a binding norm clip does to the update under descent and under Adam.
- [stage-protocol-engine](stage-protocol-engine.md) — mixed page: the stage spec a config may declare, the four entry points, the ACTIONS vocabulary, what a transition resets and what it leaves alone.
- [balance-controllers](balance-controllers.md) — mixed page: the loss-weight controllers that exist, what each does mechanically, their fixed points and rails.
- [under-coverage-metric-family](under-coverage-metric-family.md) — theory page: the coverage statistics as one-sided moments of the TB residual, the reward ramp that weights them, the level/composition/spread ladder, and what each is blind to.

## Buffers and rollouts

- [prior-buffer](prior-buffer.md) — mixed page: the store the backward branch draws terminals from, its two intake sources, the per-condition admission gate, the three eviction channels, the stage actions and the churn telemetry.
- [prior-buffer-row-geometry](prior-buffer-row-geometry.md) — theory page: where a stored row sits relative to the target's thermal shell, the isotropic and shaped jitter tiles, what a likelihood term on stored rows fits, and a floor-versus-shell contrast that needs no normaliser.
- [anchor-buffer](anchor-buffer.md) — mixed page: the permanent archive of low-energy states, its anchor-energy currency, the six membership sites and the two frozen-guarded primitives, the restore policy hook, and the jitter seam out to the prior buffer.
- [replay-buffer](replay-buffer.md) — mixed page: the store of forward trajectories, uniform admission, the hazard and backstop eviction arms, the prioritised draw and its importance weights, the held-out split and the intake baseline.
- [rollout-cadence-and-dose](rollout-cadence-and-dose.md) — mixed page: what a forward rollout buys, Little's-law identities, the dose variable, the Z-pinning invariants.

## Compute and energy

- [batch-size](batch-size.md) — mixed page: everywhere the batch is consumed, its effect on step time and gradient noise, accumulation semantics, the sizer.
- [gpu-occupancy](gpu-occupancy.md) — mixed page: what the occupancy percent measures, the two-source sensor and its sampling thread, the two trailing windows over one deque, and where the reading is consumed.
- [compute-guards](compute-guards.md) — code-bound page: the mechanisms that keep a run inside its machine, from the pre-flight GPU refusal and the per-process memory cap through the OOM handler's train and eval branches and the energy function's own chunk recovery to the wall-clock ceiling, trajectory checkpointing and compilation.
- [mlip-energy-routes](mlip-energy-routes.md) — mixed page: the uma and mace energy routes, their dispatch and batch builders, the gas-phase reference and lattice energy definition, grad state, crash substitution and the ground-truth gates.
- [reward-construction](reward-construction.md) — mixed page: how a scored structure becomes log R, the energy terms and their temperature factors, the lj_coeff currency, the two clips, and the physical leg beside the composite.
- [crystal-force-fields](crystal-force-fields.md) — theory page: the pair-sum and MLIP energy backends, the ELJ closed form and its calibration coefficient, the cluster and edge lists every pair sum reads, the typed exp-6 field the design note describes against the skeleton in the code, and what lattice energy means per backend.

## Crystal physics

- [cell-fundamental-domains](cell-fundamental-domains.md) — theory page: the unit-cell fundamental domain per setting class, the walls the code evaluates, and where derivation and code differ.
- [asymmetric-unit-reduction](asymmetric-unit-reduction.md) — theory page: the pose redundancy of the asymmetric unit under the Euclidean normaliser, its discrete cosets and free translation axes, the fold the code implements, the standard-frame convention, and where derivation and code differ.
- [latent-dimension-structure](latent-dimension-structure.md) — code-bound page: the crystal latent row by row, the clamps and physical maps, the rows the build discards and how they are held out of the SDE, and the wrapped rows and their scoring.
- [landscape-structure](landscape-structure.md) — theory page: basins as relaxation endpoints under an identity cut, the coverage, richness and rarefaction estimators over captures, occupancy versus depth, what a two-dimensional embedding measures, the dimensional cost of hitting a named basin, and the packing-motif descriptors.
- [crystal-search-method](crystal-search-method.md) — mixed page: the batched local optimiser behind the crystal search, its initialiser and reduced-cell modes, the staged descent loop and its batch-wide stopping and learning rate, what one start returns, and the chunk and coefficient stamps on its output.
- [structure-comparison-metrics](structure-comparison-metrics.md) — mixed page: the RDF channel schemes and their invariances, the earth-mover distance between two RDFs, the COMPACK overlay and its failure return, the identity and thinning cuts the analysis scripts carry, and the sliced-Wasserstein latent metrics logged in training.

## Conformers

- [conformer-chart-and-internal-coordinates](conformer-chart-and-internal-coordinates.md) — mixed page: the internal-coordinate tree behind a conformer state, its canonical ordering up to Aut(G), the four freedom tiers and what freezing does to the target, the state-to-DoF map and its inverse, the redundant and singular DoF families, and the two column tables that address the chart from the graph side.
- [conformer-force-field-and-prior](conformer-force-field-and-prior.md) — mixed page: the MMFF94 torch force field behind the conformer reward and its per-term verification, the tiling and batch cache that make a draw cheap, the fitted internal prior and its ring and torsion-group paths, descent repair in state space, and the thermal, ESS and basin-coverage statistics.
- [conformer-training](conformer-training.md) — code-bound page: the two conformer entry points and their config schemas, what the conformer trainer inherits and overrides from the crystal trainer, the stage and loss settings it ships, the condition data layer and buffer hooks, the batch cache, and the per-row chart dispatch.
- [conformer-conditioning-and-carrier](conformer-conditioning-and-carrier.md) — code-bound page: the fixed-width carrier layout that holds molecules of different coordinate counts in one state, the pad handling through policy, log-probs and energy, the per-row chart dispatch, the condition channel and identifier namespace, the baked encoder embeddings, and what is built but not reached.

## Infrastructure

- [config-validation](config-validation.md) — code-bound page: the load path from YAML to the trainer's config object, the retired-key and state-version gates, the invariant rules and their three severities by entry point, the derived-value resolution, and what absence means.
- [battery-generation](battery-generation.md) — code-bound page: the program that writes a battery's arms, the library and hand-forked generator paths, which base each reads, the assertions and load check, and what makes an arm's identity.
- [checkpoints-and-resume](checkpoints-and-resume.md) — code-bound page: what a checkpoint holds, the files a run writes, the three start paths, what a resume does not do.
- [cluster-operations](cluster-operations.md) — code-bound page: what a battery ships to SLURM, the submission script's guards and placeholder resolution, the two-repository launch, the wandb channel, the live-log artefacts and the occupancy readings, and multi-leg resubmission.
- [failure-signatures](failure-signatures.md) — code-bound page: the map from mechanism to trace, the stdout messages and exceptions each failure emits, the counters that survive a hard kill, and the stalls that emit nothing.
- [run-reading](run-reading.md) — mixed page: the metric namespaces a run publishes and their writers, the tracker EMA behind three of them, the held-out streams, the relations the code states between keys, the analysis package and the figure budget.
- [dev-traps](dev-traps.md) — code-bound page: the mechanisms that produce a silent wrong result, from in-place PyG device moves and shape-classified batch fields to dual-imported modules, collection-time globals, blind tolerances and checks that pass without running.
- [codebase-map](codebase-map.md) — code-bound page: where the training code lives, what each module and subpackage holds, the entry points, the MXtalTools boundary, the artefact conventions, and the config and test trees.
- [paper-vs-code](paper-vs-code.md) — mixed page: the ledger of six subjects where the manuscript's description, as recorded in the design note's delta audit, and the code at the stamped commit differ, plus where the code has moved since that note.
- [external-dependency-validation](external-dependency-validation.md) — method page: the three levels of a validation gate against a reimplemented third-party computation, why internal parity is silent on a shared defect, the required elements and what invalidates a gate, and how the MLIP gates are wired as tests.

How a page is produced: [writing-protocol](writing-protocol.md).

## Conventions on every page

- **A page describes mechanics and theory.** What the code does, verifiable against it; what the theory says, derivable. Where the two disagree the page says so and stops. No history, no morals, no verdicts on what was tried.
- **Drift class** in the first line: T (theory, does not move with code), C (code-bound), M (mixed).
- **Verified against** a commit hash. Pages are overwritten in place.
- **Calibrations** are the only numbers that carry a run: constants used operationally, tagged `[calibration, run, date]`, with no interpretation attached.
- **Owner choices** are written by the owner, in three buckets: bullets bitten, design choices, priorities. A page written by an agent leaves the section as a placeholder.
- **Code references** in a fixed notation, checked by a script: `cfg:block.key` for a config key, `file.py::Class.method` for a symbol.
