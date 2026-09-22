# Gradient clipping

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

Between a loss and an optimizer step this codebase bounds four quantities: the gradient of a loss with respect to the reward (`reward_grad_clip`, `reward_grad_force_clip`), the per-row loss value (`loss_clip`), a residual's influence through the Huber knee (`beta`), and the global norm of the assembled parameter gradient (`gradient_norm_clip`, or the adaptive bar in `grad_clip_guard.py`). This page covers each, and the arithmetic relating a binding norm clip to the learning rate. The loss coefficients these sit inside belong to [loss-composition](loss-composition.md), the reward-path routes to [gradient-routes-and-terminal-force](gradient-routes-and-terminal-force.md), and the rate to [learning-rate](learning-rate.md).

## The parameter-gradient clip sites

`train.py::Modeller.step_loss` is the only place in `train.py` where a policy gradient is clipped; `train_conformer.py::run` holds the second site, driving the same guard object on the channels `fwd` and `bwd` ([conformer-training](conformer-training.md)). `step_loss` calls `loss.backward()`, returns early when `do_step` is false (mid-accumulation, so the clip sees the summed gradient of the whole window), takes a bar from `grad_guard.threshold(step_type)`, calls `torch.nn.utils.clip_grad_norm_(self.gfn_model.parameters(), bar)`, and feeds the returned norm to `grad_guard.observe(step_type, pre_clip)`. `step_type` is one of `('fwd', 'bwd', 'replay', 'fused')` (`grad_clip_guard.py::CHANNELS`), each reaching its own optimizer step. `clip_grad_norm_` reports the norm *before* rescaling and rescales the whole parameter set by $\min(1, \tau/\lVert g\rVert)$, so the delivered gradient keeps its direction at norm $\min(\lVert g\rVert, \tau)$. `observe` runs before the finiteness gate, so a non-finite norm is counted rather than lost down the early return, and is not folded into the bar.

## The guard: a quantile bar per branch

With `cfg:grad_clip_guard.enabled` true, `threshold` returns a per-branch tracked bar $\tau$. One `_Branch` exists per channel; an unrecognised `step_type` raises rather than falling back, and an unknown key in the config block raises at construction.

A branch begins *warming*: `_accumulate` collects $\log\lVert g\rVert$ for `cfg:grad_clip_guard.warmup_steps` observations, fits two moments, and seeds $\tau = \exp(\mu + z_p\sigma)$ with $z_p$ the standard normal quantile at `cfg:grad_clip_guard.p`, storing it also as `baseline`. Until then the bar is `cfg:gradient_norm_clip` under `warmup_clip: static`, or infinity under `warmup_clip: off`. In steady state each observation applies the multiplicative stochastic-approximation quantile step

$$\tau \leftarrow \tau \, \exp\!\Big(\eta\,\big(\mathbb{1}[\lVert g\rVert > \tau] - (1-p)\big)\Big),$$

whose fixed point is $P(\lVert g\rVert > \tau) = 1-p$, so $\tau$ is the branch's $p$-th quantile. The input is the indicator alone: an exceeding step moves $\log\tau$ by $+\eta p$ and a quiet one by $-\eta(1-p)$, whatever the size of the exceedance, and adaptation is asymmetric by $p/(1-p)$. `_apply_cap` then clamps $\tau$ into $[\text{baseline}/M,\ \text{baseline}\cdot M]$, $M$ = `cfg:grad_clip_guard.max_ratio`, counting each clamp as a saturation. A zero or negative norm contributes no log and is counted as `gradclip/nonpositive`.

`GradClipGuard.refresh` runs from `protocol.py::StageProtocol.advance` at every stage transition when `cfg:grad_clip_guard.refresh_on_stage` is set: every branch returns to warming with its accumulators reset, `tau` stays in place, so the outgoing bar is applied while the new one is measured, and `_accumulate` prints the before/after ratio when it lands.

`report` drains windowed counters every ten steps into `gradclip/<channel>_tau`, `_n`, `_fire_rate` and `_saturated`, omitting a channel with `n_total == 0` rather than publishing a zero. `fire_rate` is the window fraction with $\lVert g\rVert > \tau$, fixed point $1-p$. `is_calibrated` says whether a branch is out of warming, and returns false outright when the guard is disabled; its one caller was `step_loss`, which used it to withhold the clip ratio from the hypergradient sensor while the bar was the static fallback. That sensor was removed on 2026-09-20 (`protocol.py::RETIRED_LR_SENSOR_KINDS`, a stage declaring `kind: hyper` raises at load), so nothing in the package calls `is_calibrated` now. The tracker is object state, persisted under `grad_guard` by `checkpointing.py::Checkpointer.get_state_dict`.

## Where the static clip survives

`cfg:gradient_norm_clip: auto` resolves in `utils.py::resolve_derived_config` to

$$\text{clip} = 250 \cdot \frac{\text{grad\_median}(T)}{6.6\times 10^3} \cdot \sqrt{W/512},$$

with $T$ = `cfg:integrator.T`, $W$ = `cfg:model.policy_hidden_dim`, and `grad_median` log-log interpolated over `utils.py::_GRAD_MEDIAN` $= \{10: 1.0\times10^3,\ 25: 6.6\times10^3,\ 100: 1.7\times10^4\}$, extrapolated past the ends by the nearest segment. Because the reference $T = 25$ is itself the table's $6.6\times10^3$ entry, the ratio of clip to tabulated median is $250/6600 = 0.0379$ at $W = 512$ for every $T$. The resolver applies the table for every route.

The resolved static clip survives as the bar `threshold` returns when the guard is disabled or a branch is warming under `warmup_clip: static`, and as the clip on `gfn_model.flow_model.parameters()` in the three `z_calibration` sidecar steps, `train.py::Modeller._z_rollout_step`, `._z_replay_step` and `._z_calibration_step`, which always use it ([z-calibration](z-calibration.md)).

## Clips upstream of the parameter gradient

`gflownet_losses.py::soft_clip` bounds a per-row loss *term* at `loss_clip`: below the cutoff the value passes through, above it it is $\mathrm{sign}(x)\big(c + \log(1 + \max(|x| - c,\ 10^{-3}))\big)$, whose derivative $1/(1 + |x| - c)$ decays to zero. `combine_branch_terms` applies it elementwise on the `[n_terms, B]` stack before summing, so each term is bounded individually and the combined row at `n_active` times the cutoff; `-1` disables it, and the canonical config sets `1.0E+9` on all three branches ([loss-composition](loss-composition.md)).

The Huber knee bounds the derivative rather than the value: every TB-family term is `beta * F.smooth_l1_loss(..., beta=beta)`, so its derivative in the residual is $\mathrm{clip}(r, \pm\beta)$ ([trajectory-balance](trajectory-balance.md)).

`gflownet_losses.py::arm_reward_grad_hook` registers a backward hook on `log_r` that runs `nan_to_num` unconditionally, then clamps elementwise at `reward_grad_clip` when that is above zero; `reward_grad_force_clip` separately caps the norm of the per-row $\partial \log R/\partial x_T$ before it reaches the live terminal ([gradient-routes-and-terminal-force](gradient-routes-and-terminal-force.md)).

## A binding norm clip and the learning rate

Let the clip bind, $\lVert g\rVert > \tau$. The optimizer receives $\tau\,\hat g$ with $\hat g = g/\lVert g\rVert$: the gradient's magnitude has been discarded.

Under plain gradient descent the update is then $\Delta\theta = -\eta\tau\hat g$, of fixed norm $\eta\tau$ whatever $g$ is. On a locally quadratic direction of curvature $L$ with error $e$, the unclipped recursion $e \leftarrow (1-\eta L)e$ is geometric and diverges above $\eta = 2/L$, precisely because the step grows in proportion to $e$. With the clip binding the step no longer grows with $e$, and the recursion $|e| \leftarrow \big||e| - \eta\tau\big|$ cannot leave $[0, \eta\tau]$ once inside it. The cliff in $\eta$ is replaced by a residual oscillation of amplitude linear in $\eta$, and the algorithm is normalized gradient descent with step length $\eta\tau$.

Every optimizer constructed in `train.py` is `torch.optim.Adam`, which is invariant to a constant rescale: $g \to cg$ gives $m \to cm$, $v \to c^2 v$, so $\hat m/\sqrt{\hat v}$ is unchanged and a clip binding by the same factor every step changes nothing. What a step-varying binding clip changes is the weight each step carries into the moment estimates: a spike enters $v$ with weight $(1-\beta_2)\lVert g\rVert^2$, and clipping equalises that contribution across steps.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:gradient_norm_clip`, `cfg:grad_clip_guard.enabled`, `cfg:grad_clip_guard.p`, `cfg:grad_clip_guard.eta`, `cfg:grad_clip_guard.warmup_steps`, `cfg:grad_clip_guard.warmup_clip`, `cfg:grad_clip_guard.max_ratio`, `cfg:grad_clip_guard.refresh_on_stage`, `cfg:fwd_loss_coeffs.loss_clip`, `cfg:fwd_loss_coeffs.beta`, `cfg:fwd_loss_coeffs.pooled_beta`, `cfg:fwd_loss_coeffs.reward_grad_clip`, `cfg:fwd_loss_coeffs.reward_grad_force_clip`, `cfg:bwd_loss_coeffs.loss_clip`, `cfg:bwd_loss_coeffs.beta`, `cfg:replay_loss_coeffs.loss_clip`, `cfg:replay_loss_coeffs.beta`, `cfg:replay_loss_coeffs.reward_grad_clip`, `cfg:integrator.T`, `cfg:model.policy_hidden_dim`.

Code: `grad_clip_guard.py::GradClipGuard.from_config`, `.threshold`, `.observe`, `._accumulate`, `._apply_cap`, `.refresh`, `.report`, `.is_calibrated`, `.state_dict`, `.load_state_dict`, `grad_clip_guard.py::CHANNELS`; `train.py::Modeller.step_loss`, `._submodel_grad_norms`, `._z_rollout_step`, `._z_replay_step`, `._z_calibration_step`; `train_conformer.py::run`; `utils.py::resolve_derived_config`, `utils.py::_grad_median`, `utils.py::_GRAD_MEDIAN`; `gflownet_losses.py::soft_clip`, `.combine_branch_terms`, `.arm_reward_grad_hook`; `protocol.py::StageProtocol.advance`; `checkpointing.py::Checkpointer.get_state_dict`, `.set_state_dict`.

## Could be tooling

An AST walk over every call to `torch.nn.utils.clip_grad_norm_` in the package, resolving the first argument to the parameter set it names and the second to its config key, would produce the clip-site inventory above mechanically: a new call site appears as a row with no paragraph, and a call whose bar is a literal appears with no key. `resolve_derived_config` already computes the resolved static clip, so a generator can print it beside `_GRAD_MEDIAN[T]` and the flow head's parameter count, stating the implied ratio and which parameter set the key governs under this config's `grad_clip_guard.enabled`. And `fire_rate` has a fixed reference, $1-p$, so a run-health pass can read it with `_n` beside it on the channels `is_calibrated` reports out of warming.

## Sources

The code above, read at the stamped commit, and the `gradient_norm_clip`, `grad_clip_guard` and three `*_loss_coeffs` blocks of `configs/mk_dev.yaml`, with the module docstrings of `grad_clip_guard.py` and `utils.py`. Memory files located the code and were not used as evidence: project_grad_clip_auto_rule_foundations, project_equilibration_toy_fidelity_gap.
