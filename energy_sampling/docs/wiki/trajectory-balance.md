# Trajectory balance

*Drift: **T** (theory). Verified against commit `e17167e`, 2026-09-20. Sources at the end.*

Trajectory balance (TB) is the objective this codebase trains a diffusion-style sampler with. It asks that a single learned scalar $\log Z$ reconcile, on every trajectory, the forward policy's density, the backward policy's density and the reward. This page defines the residual and the loss shape the code builds, states what its zero and its variance imply, reproduces the level decomposition of `docs/tb_level_decomposition.tex`, and describes the gradient the loss delivers. Left out and named: the estimators and servo machinery that place $\log Z$ between fills are [z-calibration](z-calibration.md); the directions the objective does not restore are [flat-directions-and-limit-cycles](flat-directions-and-limit-cycles.md); how the two streams are combined is [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md); the normaliser-free sibling objective is [vargrad](vargrad.md).

## The residual and the loss

A trajectory $\tau = (s_0 \to \cdots \to s_T)$ runs from a fixed source to a terminal $x = x(\tau) := s_T$, scored by a strictly positive reward $R(x)$ with target $\pi(x) = R(x)/Z^\star$, $Z^\star = \sum_x R(x)$. The forward policy $P_F(\tau)$ is normalised over complete trajectories; the backward policy $P_B(\tau \mid x)$ over the trajectories terminating at each $x$. The TB condition is

$$Z\,P_F(\tau) \;=\; R(x(\tau))\,P_B(\tau \mid x(\tau)) \qquad \text{for every } \tau .$$

Summing over the trajectories ending at a fixed $x$ gives $Z\,P_F^{\top}(x) = R(x)$ for the terminal marginal $P_F^{\top}$, and summing that over $x$ forces $Z = Z^\star$ and $P_F^{\top} = \pi$.

In the code both densities are products of per-step Gaussian transition densities. `models/gfn.py::GFN._fwd_step` scores an increment under $\mathcal{N}(\Delta t\,\mu_\theta,\ \Delta t\,C)$ via `::GFN.fwd_gauss_logprob`, with $C = \mathrm{diag}(d)$ or $\mathrm{diag}(d) + VV^\top$ under DPLR, and the backward kernel through `::GFN.gauss_logprob`; the final backward step is deterministic to the source and contributes $\log P_B = 0$. `::GFN.get_traj_fwd` and `::GFN.get_traj_bwd` return the per-step stacks, which the loss reduces by `log_pfs.sum(-1)` and `log_pbs.sum(-1)`. Dead latent rows are excluded from both densities before summing (`::GFN._live_only`); a term differing between $P_F$ and $P_B$ enters the residual as a constant and is absorbed by $\log Z$.

The learned normaliser is read as `log_flow[:, 0]`, written by `::GFN._step_flow`. On the unconditional route, and on a conditional route with `scalar_flow`, `::GFN.init_flow_model` makes it a `models/architectures.py::LearnableScalar` whose single parameter is `flow_model.scalar`; otherwise it is a `scalarMLP` reading a detached condition embedding.

Writing $\log w(\tau) := \log R(x) + \log P_B(\tau \mid x) - \log P_F(\tau)$, the per-trajectory residual is

$$r(\tau) \;=\; \log Z + \log P_F(\tau) - \log P_B(\tau \mid x) - \log R(x) \;=\; \log Z - \log w(\tau),$$

which is `gflownet_losses.py::get_tb_loss`'s `tb = log_pf + log_Z_per_traj - log_pb - log_r`. The loss is not the square: it is `beta * F.smooth_l1_loss(tb, zeros, reduction='none', beta=beta)`, and the outer $\beta$ cancels `smooth_l1`'s inner $1/\beta$, giving

$$\rho(r) = \begin{cases}\tfrac12 r^2, & |r| \le \beta\\ \beta|r| - \tfrac12\beta^2, & |r| > \beta\end{cases}\qquad \psi(r) \equiv \rho'(r) = \mathrm{clip}(r, \pm\beta).$$

Under `tb_z_source: 'persistent'` (`train.py::Modeller.tb_z_source`), `get_tb_loss` substitutes a detached per-condition estimate from `buffer.py::ConditionLogZTracker.lookup` wherever `target_mask` is True, so those rows carry no gradient to the flow model; under `'batch_root'`, `gflownet_losses.py::batch_root_z` substitutes this batch's own detached Huber fixed point. The per-row term enters its branch through `::combine_branch_terms`, which sums the branch's terms rather than averaging them (changed 2026-08-26), after an optional elementwise `soft_clip`.

## What zero implies, and what the mean and the variance measure

$r(\tau) = 0$ for every $\tau$ is exactly the TB condition, hence "$P_F = \pi P_B$ as path measures and $Z = Z^\star$", where $(qP_B)(\tau) := q(x(\tau))P_B(\tau\mid x(\tau))$ for a distribution $q$ over terminals.

The scalar profiles out of the quadratic core: at any fixed policy, $\min_{\log Z}\mathbb{E}[(\log Z - \log w)^2] = \mathrm{Var}(\log w)$, attained at $\log Z = \mathbb{E}[\log w]$. At the minimising scalar the objective's remaining value is the spread of $\log w$; a constant shift of $\log w$ is matched by the scalar. $\mathrm{Var}(\log w) = 0$ almost surely forces $\log(P_F/\pi P_B)$ constant, and two normalised path measures with constant ratio are equal.

The note's importance-sampling identity is $\mathbb{E}_{P_F}[w] = Z^\star$, since $\sum_\tau P_F(\tau)\,R(x)P_B(\tau|x)/P_F(\tau) = \sum_x R(x)$; it holds for the forward sampling distribution specifically. By Jensen, $\mathbb{E}_{P_F}[\log w] \le \log Z^\star$, a lower bound whose gap is named in the next section. `utils.py::quick_tb_stats` reports both: `tb_err`, the pooled RMS residual, floored at $\mathrm{std}(\log w)$; and `tb_resid_clipped`, the signed batch mean of $\mathrm{clip}(r, \pm\beta)$, which is $\mathrm{d}L/\mathrm{d}\log Z$ up to the $\beta$ scale.

## The level decomposition

The assumptions are those of the tex note: finite sums over trajectories, $R > 0$, $P_F$ and $P_B$ normalised as above, and all three divergences finite. The note records that everything holds verbatim per condition $c$ with $R(\cdot|c), \pi(\cdot|c), Z^\star(c), Z(c)$, since nothing in the algebra couples conditions.

Let $\mu$ be the distribution over terminal states held in the buffer, and define the levels

$$J_F := \mathbb{E}_{P_F}[\log w], \qquad J_B := \mathbb{E}_{\mu P_B}[\log w],$$

and the three divergences

$$A := D_{\mathrm{KL}}(P_F \| \pi P_B), \qquad B := D_{\mathrm{KL}}(\mu P_B \| P_F), \qquad C := D_{\mathrm{KL}}(\mu \| \pi).$$

$A$ has the model in the first argument (mode-seeking), $B$ in the second (mass-covering), $C$ is a property of the buffer alone. An auxiliary lemma: $D_{\mathrm{KL}}(qP_B \| pP_B) = D_{\mathrm{KL}}(q\|p)$ for any $q, p$ over terminals, because the shared $P_B$ factors cancel inside the logarithm.

**Forward level.** Substituting $R = Z^\star \pi$,
$$J_F = \mathbb{E}_{P_F}\!\left[\log\frac{Z^\star \pi(x) P_B(\tau|x)}{P_F(\tau)}\right] = \log Z^\star - D_{\mathrm{KL}}(P_F \| \pi P_B) = \log Z^\star - A,$$
so $J_F \le \log Z^\star$ always: the Jensen statement above with the gap identified as $A$.

**Backward level.** With $Q := \mu P_B$, the same substitution and then inserting $Q$ inside the logarithm give
$$J_B = \log Z^\star + \mathbb{E}_Q\!\left[\log\tfrac{\pi P_B}{P_F}\right] = \log Z^\star - D_{\mathrm{KL}}(Q \| \pi P_B) + D_{\mathrm{KL}}(Q\|P_F) = \log Z^\star - C + B,$$
the middle term by the shared-kernel lemma. Unlike $J_F$, this is not a lower bound on $\log Z^\star$: it exceeds it whenever $B > C$.

**Master theorem.** Subtracting, with $\log Z^\star$ cancelling,

$$\Delta := J_B - J_F = A + B - C .$$

Two checks. If the policy is exact but the buffer arbitrary, $A = 0$ and $B = C$, so $\Delta = 0$. If the buffer is thermal but the policy is not, $\Delta = D_{\mathrm{KL}}(\pi P_B\|P_F) + D_{\mathrm{KL}}(P_F\|\pi P_B)$, the symmetrised (Jeffreys) divergence. $\Delta$ is a function of $(P_F, P_B, \mu, \pi)$ and not of $Z$; it decreases in $C$. The branch means are $\mathbb{E}_{P_F}[r] = \log Z - J_F$ and $\mathbb{E}_{\mu P_B}[r] = (\log Z - J_F) - \Delta$, so no single $\log Z$ centres both unless $\Delta = 0$. The note also records $D_{\mathrm{KL}}(P_F^\top\|\pi) \le A$, by the KL chain rule factoring $A$ into that divergence plus $\mathbb{E}_{x\sim P_F^\top} D_{\mathrm{KL}}(P_F(\cdot|x)\|P_B(\cdot|x)) \ge 0$; neither $B$ nor $\Delta$ admits such a bound.

**ELBO floor.** With $\mathrm{ELBO}(\mu) := \mathbb{E}_\mu[\log R] + \mathbb{H}(\mu) = \log Z^\star - C$, the backward-level lemma gives $J_B - \mathrm{ELBO}(\mu) = B \ge 0$, with equality iff $P_F = \mu P_B$.

**Corollary (the reachable backward levels).** Hold $\log Z$ detached at $z_0$. The mean backward residual $z_0 - J_B$ can be driven to zero by training $P_F$ only if $z_0 \ge \mathrm{ELBO}(\mu)$: as $P_F$ varies, $J_B$ ranges over $[\mathrm{ELBO}(\mu), \infty)$, unbounded above because $B \to \infty$ as $P_F$ withdraws mass from the buffer's trajectories, bounded below by the floor. Below the floor no forward policy makes the backward residual mean-zero; the note states that a squared-error optimiser does not stop there but distorts $P_F$ in the demanded direction. In `mk_dev.yaml` the `bwd` and `replay` branches ship `freeze_z: 1.0`, which detaches `log_Z_learned` at its source in `gflownet_losses.py::get_gfn_backward_loss`.

## The gradient

Write $\psi_i = \mathrm{clip}(r_i, \pm\beta)$ for row $i$ of a batch of $B$ rows.

**With respect to $\log Z$.** $\partial L/\partial \log Z = \tfrac1B\sum_i \psi_i$: the residual inside the knee, saturating at $\pm\beta$ outside it. This is nondecreasing in $\log Z$ and runs from $-\beta$ to $+\beta$, so its root is bracketed by $[\min_i \log w_i - \beta,\ \max_i \log w_i + \beta]$; `gflownet_losses.py::winsorized_z_root` bisects for it over a fixed iteration count and returns the root, a standard error $\mathrm{rms}(\psi)/(\sqrt{B}\cdot \text{frac\_unclipped})$, and `frac_unclipped`, which is $\partial^2 L/\partial (\log Z)^2$, bounded above by 1. The root is Huber's location estimator Winsorised at $\beta$, not the mean; $\beta \to \infty$ recovers the mean of $\log w$.

**With respect to policy parameters.** $\partial r/\partial \theta$ has two routes. The explicit route differentiates the densities at fixed samples: $\psi_i(\nabla_\theta \log P_F(\tau_i) - \nabla_\theta \log P_B(\tau_i|x_i))$, a score-function term on whichever policy is not the sampler and a direct term on the other. The pathwise route exists only when the sampled states carry gradient: `::GFN.fwd_propagate` and `::GFN.bwd_propagate` detach the step mean and variance under `detach_traj`, which the branch loss builders key on `loss_coeffs.traj_grads == 0`. With the path live, $x_{t+1}$ depends on $\theta$ through the reparameterised increment, so $\partial L/\partial x$ flows back as a product of per-step Jacobians; `cfg:fwd_loss_coeffs.path_grad_last_k` keeps that chain alive for the last $k$ steps only, and `cfg:fwd_loss_coeffs.path_grad_scale` at 0 detaches the noise-scale channel. `cfg:fwd_loss_coeffs.reward_grads` decides whether $\partial \log R/\partial x_T$ joins that flow: at 0 the reward is scored under `no_grad`. The per-step norms are recorded as `pathgrad/state_grad_step{i}`.

The branches ship different routes in `mk_dev.yaml`: `fwd` and `replay` carry `traj_grads: 0.0`, `bwd` carries `traj_grads: 1.0`. On the backward branch the trajectory is drawn from $P_B$ itself, so the explicit $\nabla_{\theta_B}\log P_B$ term is a score under the sampling distribution, whose expectation vanishes only where the multiplying weight is independent of $\tau$; $\psi_i$ is not.

## Where the note and the code differ

The note writes the TB loss as the square of the residual and reads off $\log Z = J_F$ or $\log Z = J_B$ as the branch-centring choices. The code's loss is the Huber form above, so the zero-gradient level is the $\beta$-Winsorised root of $\log w$ rather than its mean. The note's table gives $\operatorname{logmeanexp}$ of $w$ on the forward stream as the estimator of $\log Z^\star$ and $\operatorname{logmeanexp}(\log w) - \operatorname{mean}(\log w)$ as the estimator of $A$; the level the loss itself trains toward is `winsorized_z_root`, a third quantity. The note is a discrete-sum derivation; the code's state space is continuous, so the sums are integrals and $P_B$'s normalisation is over paths.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:fwd_loss_coeffs.{tb, beta, tb_z_source, traj_grads, path_grad_last_k, path_grad_scale, reward_grads, freeze_policy, freeze_z, db, subtb}`, and the same keys on `cfg:bwd_loss_coeffs` and `cfg:replay_loss_coeffs`.

Code: `gflownet_losses.py::get_tb_loss`, `::batch_root_z`, `::winsorized_z_root`, `::combine_branch_terms`, `::get_gfn_forward_loss`, `::get_gfn_backward_loss`, `::update_and_lookup_condition_log_z`, `::get_db_loss`, `::get_subtb_loss`; `models/gfn.py::GFN.get_traj_fwd`, `::GFN.get_traj_bwd`, `::GFN.get_traj_replay`, `::GFN._fwd_step`, `::GFN._bwd_step`, `::GFN.fwd_propagate`, `::GFN.bwd_propagate`, `::GFN.fwd_get_logvars`, `::GFN.fwd_gauss_logprob`, `::GFN.gauss_logprob`, `::GFN._live_only`, `::GFN._step_flow`, `::GFN.init_flow_model`; `models/architectures.py::LearnableScalar`; `buffer.py::ConditionLogZTracker.lookup`, `::ConditionLogZTracker.rms_z_grad`; `utils.py::quick_tb_stats`; `train.py::Modeller.tb_z_source`.

## Sources

Repo: `docs/tb_level_decomposition.tex`, `docs/design/vargrad_convergence_theory.md`, and the code above at the stamped commit. Memory files project_logz_fixed_point_identity, project_bwd_stationarity_low_T, project_z_convergence_gradient_signal and project_traj_grads_asymmetry located the code and were not used as evidence.
