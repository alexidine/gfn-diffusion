# Prior buffer row geometry

*Drift: **T** (theory). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

The backward and replay branches do not roll out a policy; they take terminal states that already exist and build trajectories to them. Where those states sit in configuration space, relative to where the target keeps its mass, is a property of the buffer. This page is about that geometry: the target's mass around an energy minimum, where the two jitter tiles put a stored row relative to it, what a likelihood term on such rows is maximised by, and how the two regions can be compared without a normaliser. Left out and named: admission, eviction, draw and churn of the prior buffer ([prior-buffer](prior-buffer.md)) and of the permanent archive ([anchor-buffer](anchor-buffer.md)); the residual and its level decomposition ([trajectory-balance](trajectory-balance.md), [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md)).

Throughout, the target is $\pi(x) \propto R(x) = \exp(-E(x)/T)$ on the latent state $x$: `train.py::Modeller.top_up_prior_from_anchors` forms `temperature = 10 ** log_T_tensor` and `energy = -reward * temperature`, so one unit of $E/T$ is one nat of $\log R$ and is written $kT$ below. The crystal latent is the cell and pose vector `latent_params()` returns, which `train.py::Modeller._batch_latents` reads gauge-fixed on the crystal route; both jitter primitives clip it into an open box of half-width one, and $d$ is its width.

## Floor and shell

Take a basin around a minimum $x_0$ with $E(x) = E_0 + \tfrac12 (x-x_0)^\top H (x-x_0)$, $H$ positive definite. Whiten with $y = H^{1/2}(x-x_0)/\sqrt{kT}$. Under $\pi$ the whitened coordinate is standard normal, so the excess energy $\varepsilon = E - E_0$ satisfies

$$\varepsilon/kT \;=\; \tfrac12 \|y\|^2 \;\sim\; \tfrac12 \chi^2_d, \qquad p(\varepsilon) \;\propto\; \varepsilon^{d/2-1} e^{-\varepsilon/kT},$$

a Gamma density with shape $d/2$ and scale $kT$: mean $(d/2)\,kT$, mode $(d/2 - 1)\,kT$, standard deviation $\sqrt{d/2}\,kT$. In the radial variable $r = \|y\|$ the same statement reads $p(r) \propto r^{d-1} e^{-r^2/2}$, with mode $\sqrt{d-1}$ and width of order $1/\sqrt{2}$, independent of $d$. Two consequences, for any such basin. The density is largest at the floor $r=0$ and the mass is not: the Jacobian $r^{d-1}$ suppresses the floor, and the fraction of basin mass with $\varepsilon$ below one $kT$ falls as $d$ grows. And the region carrying that mass is a shell at an energy of order $kT$ per degree of freedom above the floor, whose radius in *physical* latent units is direction-dependent: along eigenvector $v_i$ of $H/kT$ the half-width is $\lambda_i^{-1/2}$, so a spread of eigenvalues is a spread of thermal widths and no single isotropic radius is thermal in more than one direction. A direction with $\lambda_i \le 0$ has no thermal width; the box walls bound it.

## A point row, an isotropic kick, a shaped kick

A stored anchor is a *point*: one configuration, carrying no width. The two tiles that turn it into a training row are selected by `cfg:buffers.anchor_buffer.tile` and both run at the seam `train.py::Modeller._noise_and_condition`.

**`iso`.** `log_noise_latent_parameters` in `mxtaltools/dataset_utils/data_class_methods/crystal_ops.py` draws a direction $u$ uniform on the sphere (a standard normal divided by its norm) and a radius $\rho = 10^{U}$ with $U$ uniform on `cfg:buffers.anchor_buffer.noise_log_range`, and displaces the *stored* anchor by $\rho u$, then clips into the open box. Before the clip the displacement length is exactly $\rho$, not a Gaussian: each draw lands on a sphere, and the log-uniform radius makes the mixture over spheres flat in $\log \rho$. The energy this reaches follows from the displacement: in the quadratic model it is

$$\varepsilon \;=\; \tfrac{\rho^2}{2}\, u^\top H u, \qquad \mathbb{E}_u[u^\top H u] = \mathrm{tr}(H)/d,$$

so the row's excess energy is $\rho^2$ times the local curvature averaged over directions, with a spread across draws from the eigenvalue spread of $H$ and a further spread across anchors from how stiff each basin is. A radius fixed in latent units is a different number of $kT$ in every basin and along every direction, and is thermal only where $\rho \approx \lambda_i^{-1/2}$.

**`shaped`.** `train.py::Modeller._shaped_anchor_tile` replaces that with a per-anchor Gaussian read from a sidecar: $x \sim \mathcal{N}(x_{\min}, V)$, $V = \sum_i w_i^2 v_i v_i^\top$, with $x_{\min}$ that anchor's stored relaxed minimum rather than the stored anchor, $v_i, \lambda_i$ the stored eigenpairs, and $w_i = \sqrt{\texttt{tile\_temperature}/\lambda_i}$. Writing $\delta = \sum_i w_i z_i v_i$ with $z$ standard normal, the excess energy in the quadratic model is

$$\varepsilon/kT \;=\; \tfrac12 \sum_i \lambda_i w_i^2 z_i^2 \;=\; \tfrac{\tau}{2}\sum_i z_i^2 \;\sim\; \tfrac{\tau}{2}\chi^2_d, \qquad \tau = \texttt{tile\_temperature},$$

the same Gamma law as the thermal shell with $kT$ replaced by $\tau\,kT$: an energy of $\tau/2$ per mode, whatever the mode's stiffness. That identification holds when the stored $\lambda_i$ are eigenvalues of the Hessian of $E/kT$; the sidecar has no builder here and `_load_anchor_tile` checks only shape, version and provenance, not units. Two further departures are in the code. A direction with $\lambda_i \le 0$ takes `tile_width_cap` outright, and every $w_i$ is capped there, so the relation holds only over directions whose thermal width is under the cap; outside them the energy is whatever the surface does at the cap radius. And the draw is clipped into the open box, which truncates the shell wherever an anchor sits near a wall. Both tiles end at the same three primitives, the clip, `latent_to_cell_params` and `clean_cell_parameters(mode='hard')`.

## What a likelihood term on stored rows fits

With `cfg:bwd_loss_coeffs.mle` live, `gflownet_losses.py::get_gfn_backward_loss` calls `::terminal_mle`. At `repeats = 1` the `bound` estimator returns $-(\log P_F(\tau) - \log P_B(\tau\mid x))$ per row, whose expectation under $P_B(\cdot\mid x)$ is the negative of an evidence lower bound on $\log P_F(x)$; at `repeats > 1` the `exact` estimator returns the DReG form of the $K$-sample IWAE bound on the same quantity, and `::log_pf_estimate` is the plain reduction $\log\hat p_F(x) = \mathrm{logsumexp}_k(\log P_F - \log P_B) - \log K$. The reward appears in none of them. The objective is $\max_\theta \mathbb{E}_{x\sim\mu}[\log P_F(x)]$, minimised in KL by $P_F = \mu$, where $\mu$ is the empirical distribution of the rows the buffer holds and nothing else. Point rows make $\mu$ a mixture of atoms; the terminal marginal of an SDE sampler with a Gaussian last step cannot be atomic, so the attained optimum is the narrowest bump that last step can represent about each row.

Under trajectory balance the same geometry enters through the level rather than the shape. With $A = \mathrm{KL}(P_F\|\pi P_B)$, $B = \mathrm{KL}(\mu P_B\|P_F)$ and $C = \mathrm{KL}(\mu\|\pi)$, the backward level is $J_B = \log Z^\star - C + B$ and the branch offset is $\Delta = A + B - C$ ([trajectory-balance](trajectory-balance.md), [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md)). $C$ is a functional of the row geometry alone: it moves when the rows move from the floor to a shell, or from one shell to another, with the policy untouched.

## A floor-and-shell contrast without $Z$

Two sets built from the same anchors, a floor set $\mathcal{F}$ (the rows, or their relaxed minima) and a shell set $\mathcal{S}$ (those anchors displaced by either tile), support a statistic in which the normaliser cancels. Define

$$D_{\text{model}} = \overline{\log \hat p_F}\big|_{\mathcal{F}} - \overline{\log \hat p_F}\big|_{\mathcal{S}}, \qquad D_{\text{target}} = \overline{\log R}\big|_{\mathcal{F}} - \overline{\log R}\big|_{\mathcal{S}},$$

with $\log\hat p_F$ from `::log_pf_estimate` over $K$ backward rollouts per state (`models/gfn.py::GFN.get_traj_bwd`). $\log \pi = \log R - \log Z^\star$, so $Z^\star$ cancels from $D_{\text{target}}$; $p_F$ is a normalised density, so no learned scalar enters $D_{\text{model}}$. The difference $D_{\text{model}} - D_{\text{target}}$ is the model's log density ratio to the target at the floor minus the same ratio on the shell, in nats, positive when the model is more peaked about the floor than the target is. It is an estimator: $\log\hat p_F$ is a lower bound biased low with bias falling roughly as $1/K$ (the same reduction `cfg:buffers.anchor_buffer.confirm_k` feeds), that bias cancels in the difference only to the extent that the backward posterior is equally tight on both sets, and $\log\hat p_F$ depends on $P_B$, so a comparison across checkpoints requires the same backward policy. No such contrast is computed in this repository; the quantities it is built from are.

## The two keys, mechanically

`cfg:buffers.anchor_buffer.noise_log_range` is the `[log10_min, log10_max]` pair passed to `log_noise_latent_parameters`, and `cfg:buffers.anchor_buffer.tile` selects which draw runs; the seam `_noise_and_condition` jitters in place *before* conditioning, so the noised state is what is conditioned and scored. Two call sites pass the anchor-buffer indices: `train.py::Modeller.top_up_prior_from_anchors` and `::Modeller.seed_prior_from_condition_minima`. Under `iso` those indices are unread and `noise_log_range` sets the radius; under `shaped`, `noise_log_range` is unread, the indices are the key into the sidecar, and a call supplying none raises rather than falling back. The sidecar named by `cfg:buffers.anchor_buffer.shape_path` is read once per anchor set, at the end of `train.py::Modeller.apply_anchor_buffer_policy`, by `::Modeller._load_anchor_tile`: an unset `shape_path` raises first, then four checks, each fatal and with no isotropic fallback, on existence, `format_version == 1`, row count against the anchor buffer, and the sha1 of `anchor_buffer.x` over its contiguous float32 bytes against the recorded `anchor_x_sha1`, plus a `dim` check against the latent width. `tile_temperature` and `tile_width_cap` enter the cached `_anchor_tile` dict there. `conformer_modeller.py::ConformerModeller._noise_and_condition` raises under `shaped`, the tile being defined over cell parameters. The load prints `anchor tile: shaped, n=..., sha ok, ...`; each shaped draw prints its post-clip displacement quantiles, which `train.py::Modeller.log_buffer_stats` emits as `anchor_tile/disp_p50` and `anchor_tile/disp_p90` and drains, so a window with no shaped draw emits nothing.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:buffers.anchor_buffer.{tile, shape_path, tile_temperature, tile_width_cap, noise_log_range, confirm_k}`; `cfg:bwd_loss_coeffs.{mle, repeats}` and the same on `cfg:replay_loss_coeffs`.

Code: `train.py::Modeller._noise_and_condition`, `::Modeller._shaped_anchor_tile`, `::Modeller._load_anchor_tile`, `::Modeller.apply_anchor_buffer_policy`, `::Modeller.top_up_prior_from_anchors`, `::Modeller.seed_prior_from_condition_minima`, `::Modeller.log_buffer_stats`; `gflownet_losses.py::terminal_mle`, `::log_pf_estimate`, `::get_gfn_backward_loss`; `models/gfn.py::GFN.get_traj_bwd`; `buffer.py::AnchorBuffer`; `conformer_modeller.py::ConformerModeller._noise_and_condition`; `mxtaltools/dataset_utils/data_class_methods/crystal_ops.py::MolCrystalOps.log_noise_latent_parameters`.

## Could be tooling

The floor-and-shell contrast is assembled from `log_pf_estimate`, a stored anchor set and either tile, and is not a wired diagnostic; the held-out flag `is_val` on `buffer.py::CrystalBuffer`, which the anchor archive inherits, would fix the row set across checkpoints. The tile sidecar has no builder in this repository: `_load_anchor_tile` defines the format it reads (`format_version`, `anchor_x_sha1`, `n`, `dim`, `x_min`, `evals`, `evecs`) and nothing writes one.

## Sources

Repo: the code above at the stamped commit, `configs/mk_dev.yaml` (the `buffers.anchor_buffer` block and its mechanics comments), docs/design/prior_buffer_sizing.md. Memory files project_floor_mass_overcoverage_sep18, project_thermal_tile_prototype, project_pbfrozen_family_probe, feedback_prior_buffer_anchors_only_default located the code and were not used as evidence.
