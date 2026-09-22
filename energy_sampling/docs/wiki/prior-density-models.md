# Prior density models

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The prior policy is a sampler: draws are cheap and there is no tractable $\log p$. A density fitted to its draws supplies one, and can then serve as an energy in its own right or as one leg of a path toward the physical energy. This page covers the candidate models the repo carries, the coordinate each is fitted in against the one its consumer scores, what ground truth exists, the gate a candidate is run through, and how a fitted density enters the reward. The schedule that walks $\lambda$ along that path is [lambda-annealing](lambda-annealing.md); the training route is [conditional-route](conditional-route.md); the conditions themselves are [molecule-conditions-and-anchors](molecule-conditions-and-anchors.md).

## What is fitted, and to what

`build_prior_flow.py` reconstructs a policy from a checkpoint's `gfn_config` and `model_eval`, draws terminals from it (`::draw_terminals`, conditions sampled with replacement from the conditions file's `embedding`), and fits a flow to those draws. `build_prior_knn_reference.py` freezes a draw from `--source dataset` (a `prior_path` file) or `--source latents`; its docstring records that the prior model's own marginal needs a live modeller and is not one of its sources, so a dataset-built reference is the prior's support rather than the prior model's density.

Neither artifact is conditional: the flow builder pools draws over the sampled conditions, and no condition enters `PriorFlow.energy` or `PriorKNN.energy`. What an artifact represents is the condition-marginal of the policy's terminal distribution.

Both are tied to a geometry. The wrap mask and dead rows follow from (space group, `max_z_prime`, `periodic_centroids`), are stored in the file, and are re-checked against the live policy in `train.py::Modeller.init_gfn` through each model's `verify_against_policy`. The flow also stores `traj_T` and checks it against `cfg:integrator.T`, the terminal distribution depending on trajectory length while `T` is absent from `problem_def`. `PriorKNN` stores a sha256 over its coordinates (`::reference_digest`) and `::PriorKNN.load` refuses a file that no longer hashes to it.

## The kNN estimate, and its bias in $d$ dimensions

`energies/prior_knn.py::PriorKNN.energy` returns

$$E(x) \;=\; d_{\text{live}} \log r_k(x),$$

with $r_k$ the distance to the $k$-th nearest reference point under a minimum-image metric of period `LATENT_PERIOD` (2.0) on the wrapped dims and a Euclidean metric elsewhere. Dead rows leave the distance, so $d_{\text{live}}$ counts live dims, and $r_k^2$ is floored at `min_radius`$^2$. The Loftsgaarden-Quesenberry estimator is $\log \hat p(x) = \log k - \log N - \log V_d - d\log r_k(x)$, so $E$ is $-\log\hat p$ with every $x$-independent term dropped, and an additive constant is what a within-group contrast removes.

The surviving term carries a bias that is not constant in $x$. The $k$-nearest-neighbour ball encloses mass $k/N$, so $r_k \propto (k/(N p(x)))^{1/d}$ and the leading smoothing error scales as $r_k^2 \propto N^{-2/d}$, which is $N^{-1/6}$ at $d = 12$. The estimator therefore reads the average density over a ball that is not small compared with the distribution's own scale, and $\exp(-E) \approx p^{s}$ with $s \ne 1$. The module docstring records the measured consequence, regressing $E$ on a closed-form $-\log p$ at $d_{\text{live}}=12$, $k=32$: slope 0.72 for a concentrated prior and 1.37 for a broad one, the sign of the distortion flipping with the prior's width, against slope near 1.0 at $d=2$ and $d=3$. The distortion being multiplicative in $\log p$, it does not cancel in a contrast. Both ends are pinned by `tests/crystal/test_prior_knn.py::test_calibration_holds_at_low_dimension` and `::test_calibration_fails_at_the_shipped_dimension`, the second asserting `slope < 0.9` together with `corr > 0.95`.

The tails follow from the same form: off the reference cloud $E$ rises only as $d_{\text{live}}\log r$, so this term does not bound the box; `MolecularCrystal.generator_energy`'s `bounding_energy`, computed from `raw_latents`, does.

No mixture density fitted to prior draws exists in the repo. The two mixtures here are toy *targets* with closed-form `log_prob`, `energies/nine_gmm.py::NineGaussianMixture` and `energies/twenty_five_gmm.py::TwentyFiveGaussianMixture`.

## The flow

`energies/prior_flow.py` carries `NAME = "maf_circular"`: a masked autoregressive flow whose per-dim transform is a monotone rational-quadratic spline (`::_rq_spline`). Each `::_Block` runs a `::_MADE` conditioner over all dims in one ordering, applies a per-dim shift, then the spline. A wrapped dim contributes a (sin, cos) feature pair sharing one autoregressive degree, is folded by `::_wrap` after the shift so that shift is a unit-Jacobian rotation, and ties its end derivatives ($d_K = d_0$), giving a diffeomorphism of the circle on $[-1,1]$; a linear dim pins its end derivatives to 1 and is the identity outside $[-\text{bound},\text{bound}]$. Block orderings cycle `b % 3` between forward, reversed and a seeded permutation, so at the shipped `n_blocks=4` the fourth block repeats the first ordering.

`::_Flow.log_prob` works in internal coordinates: wrapped dims folded to $[-1,1]$ against a uniform base contributing $-\log 2$ each, linear dims standardised by the fixed `mu`/`sd` of `::_Flow.set_scaler` against a standard normal, with that standardisation's constant log-Jacobian $-\sum\log\mathrm{sd}$ carried explicitly. The normaliser is exact by construction.

The fit coordinate is not the scoring coordinate. `build_prior_flow.py` fits `states[:, -1]`, the raw terminal state. The consumer scores `latents`, from `crystal_batch.latent_params(gauge_fix_free_axes=self.is_crystal)`, whose last operation is `.clip(min=-1, max=1)`. Inside the box the two agree; a coordinate outside it is mapped onto a face, so as scored the distribution has atoms on the box faces, is many-to-one there, and has zero gradient outside. The policy is held in the box only by the soft quadratic `bounding_energy`. `build_prior_flow.py` refuses to fit above a 1% rate of coordinates at exactly $\pm 1$, but evaluates that test on the raw draws, which carry no clip atom.

Two further fit mechanics. `PriorFlow.fit` sizes the run by timing 25 probe steps against a `t0` taken before the host-to-device transfer and the model and optimizer construction, so the measured per-step cost includes one-off setup and the `time_budget` term can select far fewer steps than `cfg['steps']`. And `fit` sets neither `dead_rows` nor `traj_T` (only `save` takes them, only `load` restores them), so `verify_against_policy` on a freshly fitted object compares against the constructor's `()` and `None`. Fit selects the state with the best held-out log-likelihood over `val_frac` of the draws.

## What ground truth is available

**IWAE.** For a terminal $x$ and $K$ backward rollouts, `gflownet_losses.py::log_pf_estimate` returns $\log\hat p_f(x) = \mathrm{logsumexp}_k(\log P_F^{(k)} - \log P_B^{(k)}) - \log K$, rollouts tiled terminal-major. The weights are unbiased for $p_f(x)$, so the estimate is consistent in $K$ and biased low by a Jensen gap which at $K=1$ is $\mathrm{KL}(q_{\text{bwd}}(\tau|x)\|p(\tau|x))$: a state-dependent function that falls with $K$. In-repo callers are `::terminal_mle` under `estimator="exact"` and the anchor-buffer confirmation pass in `train.py`; no script here builds a calibration reference from it.

**The analytic toy.** `latent_gaussian` is the one target with a real crystal parameterisation (`is_crystal` true: dead rows, periodic angle dims and the box are live) and an exact partition function. `configs/gauss_aug12/spec.py::analytic_log_z` gives, with $P = \tfrac12\log(2\pi T)+\log w$ and $k$ the bounding coefficient,

$$\log Z \;=\; n_{\text{live}}P \;+\; [\text{rows live}]\Big(n_{\text{angle}}\log\big(2+\sqrt{\pi/k}\big) + n_{\text{free}}P\Big),$$

where $n_{\text{angle}}$ and $n_{\text{free}}$ partition the dead rows. For any latent-scored energy `generator_energy` sets `reduction_energy` and `jacobian_energy` structurally to zero. It is the only `benchmarks/registry.yaml` entry declaring `exactness: exact`.

## The calibration gate

`energies/density_calibration.py::calibrate` regresses a candidate's $E_{\text{model}}(x)$ on $-\log p_{\text{true}}(x)$ and returns a `Calibration` carrying `slope`, `intercept`, `corr`, `residual_sd`, `signal_sd`, `n`. `Calibration.passes` is $|\text{slope}-1| \le$ `tolerance`, default `DEFAULT_SLOPE_TOLERANCE` 0.05. Slope $s$ means $\exp(-E) \propto p^{s}$, so at $s \ne 1$ a policy sitting exactly at the fitted density still sees a systematic gradient, while the intercept is an additive constant a within-group contrast removes. Correlation does not separate the two; the module docstring records a slope-0.72 model correlating 0.99.

Three synthetic targets supply samples with exact log-density in the metric the candidate is scored in: `::wrapped_gaussian_draw`, `::wrapped_mixture_draw`, and `::warped_mixture_draw`, which pushes a wrapped mixture through $y = z + a\sin(\pi z)$, a bijection of the circle for $a\pi<1$ fixing $z=\pm1$, so its log-density follows by change of variables. The second target's own comment records that a wrapped mixture places a mixture candidate inside its hypothesis class.

## How a density enters the energy

**As the energy.** `energy_function: latent_knn` scores `prior_knn.energy(latents)` directly, and the constructor raises without `prior_knn_path`. It sits on the same diagonal as `latent_gaussian` (`is_crystal` true, `latent_energy` true), so jacobian and reduction terms are structurally zero, and it is the one energy `MolecularCrystal` does not append to `self.computes`, there being no same-named batch attribute for `analyze()` to produce. `config_invariants.py::_ANGULAR_ENERGY_FUNCTIONS` lists it.

**As a leg of a mixture.** With `cfg:energy_config.prior_flow_path` set, `generator_energy` computes `flow_energy = self.prior_flow.energy(latents) * temperature`, the multiplication cancelling the later division by $T$ so the flow leg stays a log-density rather than a tempered Boltzmann energy. The total is

$$E_{\text{total}} \;=\; (1-\lambda)E_{\text{flow}} + \lambda E_{\text{phys}} + \text{bounding\_coeff}\times E_{\text{bound}},$$

with $E_{\text{phys}}$ = crystal energy + `reduction_coeff` $\times$ reduction + jacobian. Bounding is added once outside the mix, being computed from `raw_latents` that a stored row's re-score supplies itself. Both endpoints are taken exactly rather than arithmetically, so a non-finite physical leg cannot poison a $\lambda = 0$ total. `generator_energy` raises when a flow and a live `reward_range`/`energy_clip` are both present; `physical_energy_clip` rescales the physical leg only. The legs are captured in `MolecularCrystal.REWARD_LEGS` order `('flow', 'phys', 'bound')`, and the mix being linear in them a stored pair re-mixes to any $\lambda$ by a weighted sum. The constructor raises on a `lambda_mix` other than 1.0 without a `prior_flow_path`. `lambda_mix` is mutated during a run by `train.py::Modeller.set_energy_coeffs` from `protocol.py::StageProtocol.energy_coeffs`, whose geometric ramp is [lambda-annealing](lambda-annealing.md).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:energy_function` (values `latent_knn`, `latent_gaussian`); `cfg:energy_config.{prior_knn_path, prior_knn_k, prior_knn_min_radius, prior_flow_path, lambda_mix, physical_energy_clip, bounding_coeff, reduction_coeff, temperature}`; `cfg:integrator.T`, checked against the flow's stored `traj_T`. `mk_dev.yaml` carries `prior_flow_path: null` and `lambda_mix: 1.0`; the three `prior_knn_*` keys are `MolecularCrystal` constructor arguments and are absent from it. The anneal entry is a stage's `balance.anneal_coeffs.lambda_mix: {target, rate}`.

Code: `energies/prior_knn.py::PriorKNN.energy`, `::PriorKNN.load`, `::PriorKNN.verify_against_policy`, `::reference_digest`; `energies/prior_flow.py::PriorFlow.fit`, `::PriorFlow.energy`, `::PriorFlow.save`, `::PriorFlow.verify_against_policy`, `::_Flow.log_prob`, `::_Flow.set_scaler`, `::_rq_spline`, `::_wrap`, `::_MADE`, `::_Block`; `energies/density_calibration.py::calibrate`, `::Calibration`, `::wrapped_gaussian_draw`, `::wrapped_mixture_draw`, `::warped_mixture_draw`; `energies/molecular_crystal.py::MolecularCrystal.generator_energy`, `::MolecularCrystal.REWARD_LEGS`; `gflownet_losses.py::log_pf_estimate`, `::terminal_mle`; `train.py::Modeller.init_gfn`, `::Modeller.set_energy_coeffs`; `protocol.py::StageProtocol.energy_coeffs`; `build_prior_flow.py::draw_terminals`; `build_prior_knn_reference.py::build_wrap_mask`; `configs/gauss_aug12/spec.py::analytic_log_z`; `config_invariants.py::_ANGULAR_ENERGY_FUNCTIONS`; mxtaltools `crystal_ops.py::MolCrystalOps.latent_params`.

## Could be tooling

- No in-repo path runs a fitted `PriorFlow` through `density_calibration.calibrate`, and nothing tests that `_Flow.log_prob` normalises, though the gate's docstring asks every candidate to be run through it.
- `build_prior_flow.py`'s boundary-atom guard evaluates the un-clipped coordinate, so it cannot observe the atoms the scoring coordinate creates.
- `PriorFlow.save` stores `cfg['steps']`, the cap, but neither `fit_seconds` nor the step count the probe selected.
- `verify_against_policy` checks wrap mask, dead rows and `T` but not the condition set, although `provenance` already carries `problem_hash`.

## Sources

Repo, read at the stamped commit: the code listed above, plus `configs/mk_dev.yaml`, `benchmarks/registry.yaml`, `tests/crystal/test_prior_knn.py` and `tests/crystal/test_density_calibration.py`. Memory files project_latent_knn_prior_energy, project_prior_density_model_selection, project_prior_flow_build_defects and project_latent_gaussian_is_the_only_conditional_ready_toy located the code and were not used as evidence.
