# VarGrad

*Drift: **T** (theory). Verified against commit `e17167e`, 2026-09-19. Sources at the end.*

VarGrad is the objective this codebase uses in place of trajectory balance whenever a learned normaliser is unwanted or unavailable. It trains a sampler by penalising the *spread* of a trajectory's log-weight across a group of trajectories sharing a target, rather than each log-weight's deviation from a learned scalar. The group's own mean takes the scalar's place, so no $\log Z$ enters the policy loss. Two structural facts govern how it behaves here: the group size is emergent rather than configured, and the per-row gradient coefficients sum to zero, so the level of the log-weights is a direction the objective does not see. Left out: `pooled_vg`, the cross-branch term that spans forward and buffer rows in one group so the forward/backward level offset stops being a flat direction, and the `level_gap` tether on the same coordinate, are [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md); the lambda annealing path is [lambda-annealing](lambda-annealing.md).

## The log-weight, and why the objective needs no learned $\log Z$

A trajectory $\tau$ ends at a terminal state $x$ with reward $R(x)$; write the forward and backward trajectory densities $P_F(\tau)$ and $P_B(\tau|x)$. The log-weight is

$$u(\tau) \;=\; \log R(x) \;+\; \log P_B(\tau|x) \;-\; \log P_F(\tau).$$

Define the target trajectory distribution $P^*(\tau) = R(x)P_B(\tau|x)/Z^*$ with $Z^* = \sum_x R(x)$. Then identically

$$u(\tau) \;=\; \log Z^* \;-\; \log \frac{P_F(\tau)}{P^*(\tau)},$$

so $\mathbb{E}_{P_F}[e^{u}] = Z^*$ for any full-support $P_F$, and $\mathbb{E}_{P_F}[u] = \log Z^* - \mathrm{KL}(P_F \| P^*)$: the mean of $u$ is the Jensen lower bound on $\log Z^*$, tight exactly when $P_F = P^*$. In the code $u$ is `log_ratio = log_r + log_pb - log_pf`, built in `gflownet_losses.py::condition_grouped_empirical_z` and its repeats-grouped siblings `::vg_lb` and `::vg_lme`.

Trajectory balance minimises $\mathbb{E}\big[(u - \log Z)^2\big]$ jointly over the policy and a learned scalar (`gflownet_losses.py::get_tb_loss`, residual `log_pf + log_Z - log_pb - log_r`). The scalar enters quadratically and profiles out: at any fixed policy,

$$\min_{\log Z}\ \mathbb{E}\big[(u-\log Z)^2\big] \;=\; \mathrm{Var}(u), \qquad \text{attained at } \log Z = \mathbb{E}[u].$$

So $\mathrm{Var}(u)$ *is* the TB objective with its normaliser concentrated out, and the two share a minimiser over the policy. $\mathrm{Var}(u)=0$ almost surely forces $\log(P_F/P^*)$ constant, and two normalised densities with constant ratio are equal, so $P_F = P^*$ and the terminal marginal is proportional to $R$. This is a derivation, not a property of the implementation: it holds for any group whose members share one target.

Two consequences. VarGrad is *level-free*, so `log_Z_learned` can be a pure readout rather than a training signal for the sampler. And a group must hold at least two rows: a singleton's centre is its own $u$, its residual is identically zero, and it carries no gradient. Nothing raises at the tensor level, so `config_invariants.py::vargrad_needs_groups` refuses at load instead, and `gflownet_losses.py::condition_group_stats` publishes `vg_live_frac`, the fraction of rows in a group of at least `min_group_count` (2).

## The estimator

A group of $g$ rows gives the plug-in $\hat{c} = \tfrac1g\sum_j u_j$, $d_i = \hat{c} - u_i$, $\hat{L} = \tfrac1g\sum_i \rho(d_i)$, with $\rho$ the loss shape of the next section. Three properties, all derivations.

**The centre is the Jensen centre.** $\hat c$ estimates $\mathbb{E}[u]$, the Jensen lower bound on the group's $\log Z$, not $\log Z$ itself. A log-mean-exp centre (`vg_lme`, `lme=True`) estimates $\log Z$ proper; the two flavours are mutually exclusive, asserted so in both branch builders.

**The plug-in is biased low by a factor depending on $g$.** In the quadratic regime $\mathbb{E}\big[\tfrac1g\sum_i(u_i-\bar u)^2\big] = \sigma^2(1 - 1/g)$: there is no Bessel correction on this path. At fixed $g$ that is a constant multiple and does not move the minimiser; it matters because $g$ is not fixed, so the loss level, and every statistic centred the same way, drifts with a quantity nobody set. `utils.py::quick_tb_stats` computes `logw_std_within` as the root mean square of the loss's own residual about the same per-group Jensen centre, so it reads $\sigma\sqrt{1-1/\bar g}$, not $\sigma$.

**Group size enters the gradient's variance twice, unequally.** A variance estimate from $g$ samples has relative standard deviation $\sqrt{2/(g-1)}$, so per-group noise falls fast with $g$; but at a fixed row budget $B$ there are $G = B/g$ groups and the batch-averaged loss pools roughly $B - G$ degrees of freedom either way, so the pooled trace barely moves. What does not average away is the *centre* error: $\hat c$ misses the group's true centre by about $\sigma/\sqrt{g}$, and every row shares that error, so it is a coherent per-group push, fresh at each visit. The derivation therefore predicts that small $g$ appears as slow per-group wander rather than as a noisier pooled loss.

## The Huber influence

The shipped shape is not a square. Every VarGrad term in the tree is `beta * F.smooth_l1_loss(centre, log_ratio, reduction='none', beta=beta)`; PyTorch's `smooth_l1` returns $d^2/(2\beta)$ inside the knee and $|d| - \beta/2$ outside, and the leading $\beta$ cancels the $1/\beta$, giving

$$\rho(d) = \begin{cases} \tfrac12 d^2, & |d| \le \beta\\[2pt] \beta|d| - \tfrac12\beta^2, & |d| > \beta\end{cases} \qquad\Longrightarrow\qquad \psi(d) \equiv \rho'(d) = \mathrm{clip}(d,\ \pm\beta).$$

The core has unit curvature, independent of $\beta$, so $\beta$ is purely the knee location and, under a magnitude-normalising optimiser like Adam, not a gain. The restoring force on a row is $\min(|d|,\beta)$ and saturates: past the knee the error grows without bound while the force stays at $\beta$. Two further consequences. Above the knee the loss is $\beta\sum_i|d_i| + \text{const}$, so the objective controls a Winsorised *first* moment and leaves the second free within a level set. And because $\psi$ is nonlinear and bounded, asymmetric residuals are *rectified*: the force balances at a pseudo-median displaced toward the short tail by an amount set by the tail mass beyond $\beta$. That displacement is a bias, not a variance, and no averaging removes it.

## Condition-grouped VarGrad, and where group size comes from

On the conditional route the estimand is per condition, so the group key is the condition and only the condition. `condition_grouped_empirical_z` forms groups by `torch.unique(condition_id, return_inverse=True)` and scatter-reduces, pooling every row of the batch sharing a condition and assuming nothing about row order. The branch selects this path with `cfg:fwd_loss_coeffs.vg_by_condition` and its `bwd`/`replay` counterparts; off, the loss falls back to the repeats-grouped path, which slices the batch into contiguous `repeats`-sized tiles.

**The group size is emergent.** It is roughly (rows per batch) over (distinct conditions drawn), and nothing in the config sets it: it moves with batch size, library size, the condition-sampling weights, and anything that grows the batch inside a run. `condition_group_stats` reports `vg_n_groups` and `vg_group_size_mean`; the latter is *row-weighted*, reporting $\mathbb{E}[n^2]/\mathbb{E}[n]$, the size of the group the average row sits in, which exceeds the unweighted mean whenever sizes vary.

**Two knobs buy group size, and they are not the same object.** `repeats` is $K$-tiling of the draw. On the forward branch the $K$ copies share a condition and roll out to *distinct* terminals, the cross-terminal group the estimator wants, but each tile is a further rollout and energy evaluation. On the backward branch the copies share the *terminal*, so $\log R$ cancels in the contrast and the result is trajectory-balance consistency (`gflownet_losses.py::get_tbc_loss`), a different objective. `condition_block_m` is the backward and replay alternative: the draw samples conditions first, then up to $M$ *distinct* stored rows for each (`buffer.py::CrystalBuffer._sample_condition_blocked_indices` returns exactly `batch_size` rows, topping up uniformly if the buffer is thin), so it manufactures cross-terminal groups at no energy cost. It shapes the draw for one loss, lives in the loss-coefficient block rather than on the buffer, and is read only where that branch's VarGrad coefficients are live (`train.py::_runs_grouped_vargrad`). Both knobs are paid for in distinct conditions per batch: the batch is a fixed $B$ rows, so tiling by $K$ or blocking by $M$ divides the number of distinct conditions the step sees by that factor.

## The coefficients sum to zero

Let a group have centre $\hat c = \tfrac1g\sum_j u_j$ *live in the graph*, as shipped. Differentiating $L = \sum_i \rho(\hat c - u_i)$,

$$\frac{\partial L}{\partial \theta} = \sum_i \psi(d_i)\Big(\frac{1}{g}\sum_j \nabla u_j - \nabla u_i\Big) = \sum_j \big[\bar\psi - \psi(d_j)\big]\,\nabla u_j, \qquad \bar\psi = \frac1g\sum_i \psi(d_i),$$

and the coefficients $\bar\psi - \psi(d_j)$ sum to $g\bar\psi - g\bar\psi = 0$ **exactly**, at any group size and any $\beta$. With the centre detached, $\partial L/\partial\theta = -\sum_i \psi(d_i)\nabla u_i$ and the coefficients sum to $-g\bar\psi$, nonzero as soon as the residual tails are asymmetric, leaving a common-mode force along the batch-mean score. At $g = 2$ the two forms coincide, since $d_2 = -d_1$ and $\psi$ is odd, so $\bar\psi = 0$; they diverge only on groups of three or more, which this estimator produces whenever a condition lands in a batch more often than `repeats`.

The zero sum is what makes the *level* of a group's log-weights invisible: adding a constant to every $u$ in a group leaves the loss value unchanged and contributes nothing to the gradient, so the level is an unpenalised direction in $u$-space. Unpenalised is not fixed, and a statement about $u$-space is not automatically one about parameter space: Adam scales each parameter's step by that parameter's own gradient magnitude, so a direction the objective does not restore is still moved at full step size by whatever the other loss terms project onto it. Each branch centring on its own rows is why a forward/backward level offset survives, the neighbouring page's subject.

Two implementation notes, distinct from the derivation. The form is selected by `getattr(loss_coeffs, 'vg_detach_center', 0) > 0.5`, and `mk_dev.yaml` carries no such key, so the shipped form is the live-centre, zero-sum one. And the returned empirical $\log Z$ is detached on the way out, after the loss centre is taken, so the `emp_z` regression target cannot push back on the policy while `vg_loss` stays bit-identical either way.

## Where the notes and the code disagree

The docstring of `gflownet_losses.py::condition_grouped_empirical_z` says the live (un-detached) centre is the form whose Huber weights fail to cancel, and that detaching removes the leftover. The algebra above gives the opposite, and `tests/losses/test_vg_detach_center.py` states the closed form $\psi(d_j) - \tfrac1g\sum_i\psi(d_i)$: code and test agree, only the prose is inverted, and docs/design/vargrad_objective.md L2 records the same inversion.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:fwd_loss_coeffs.{vg_lb, vg_lme, vg_by_condition, repeats, beta, emp_z}`, and the same keys on `cfg:bwd_loss_coeffs` and `cfg:replay_loss_coeffs`, which additionally carry `condition_block_m`; `cfg:bwd_loss_coeffs.tbc`; `cfg:batch_size`. `vg_detach_center` is read from a branch's block by `getattr` and is absent from `mk_dev.yaml`.

Code: `gflownet_losses.py::condition_grouped_empirical_z`, `::vg_lb`, `::vg_lme`, `::condition_group_stats`, `::get_tb_loss`, `::get_tbc_loss`, `::pooled_condition_vargrad`, `::combine_branch_terms`; `config_invariants.py::vargrad_needs_groups`, `::runs_grouped_vargrad`; `train.py::_runs_grouped_vargrad`, `::Modeller.mode_repeats`, `::Modeller.draw_bwd_sample`; `buffer.py::CrystalBuffer._sample_condition_blocked_indices`; `utils.py::quick_tb_stats`.

## Sources

Repo: docs/design/vargrad_objective.md, vargrad_convergence_theory.md, vargrad_program.md (theory sections only), and the code above at the stamped commit. Memory files project_vargrad_level_is_in_the_null_space, project_vargrad_group_size_is_emergent, project_blocks_vs_repeats_vargrad, project_conditional_instability_is_huber_basin_escape, project_vargrad_shelf_is_floor_not_noise, project_condition_block_m_is_a_loss_coeff located the code and were not used as evidence.
