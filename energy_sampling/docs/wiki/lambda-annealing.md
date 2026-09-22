# Lambda annealing

*Drift: **T** (theory). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

On the conditional route the sampler can be trained against a target that is not the physical one: a fitted density proxy for the policy's own draw supplies a second energy, and a scalar `cfg:energy_config.lambda_mix` interpolates between the two. This page covers the mixture the code forms, the geometry of the family of targets it indexes, the gate that moves the scalar, and what the residuals see when it moves. Left out: the VarGrad objective those residuals feed, [vargrad](vargrad.md); how the density proxy is built and validated, [prior-density-models](prior-density-models.md); everything the lexicographic balance block does other than fire the anneal event, [balance-controllers](balance-controllers.md); and the distinction between fresh and stored rows, [on-policy-and-off-policy-training](on-policy-and-off-policy-training.md).

## The mixture the code forms

`energies/molecular_crystal.py::MolecularCrystal.generator_energy` composes a per-row total from three quantities it names and publishes. The *flow leg* is `flow_energy`, present only when `cfg:energy_config.prior_flow_path` is set, computed as `self.prior_flow.energy(latents) * temperature`, where `energies/prior_flow.py::PriorFlow.energy` returns $-\log q(x)$ up to a constant for a flow `build_prior_flow.py` fitted to one policy's draws at one trajectory length. The multiplication by $T$ cancels `::MolecularCrystal.energy`'s later division by $T = 10^{\log T}$; the flow is a log-density, not a Boltzmann energy, and is not tempered. The *physical leg* is `physical_energy`, the crystal energy plus `reduction_energy * reduction_coeff` plus the latent-to-physical Jacobian. The *bounding leg* is `bounding_energy * bounding_coeff`, on the raw pre-clamp latents.

With $\lambda$ = `lambda_mix`, the total is

$$E_\lambda = (1-\lambda)\,E_{\text{flow}} + \lambda\,E_{\text{phys}} + E_{\text{bound}},$$

and the log-reward the losses consume is $\log R = -E_\lambda/T$. Two facts about the composition. The endpoints are taken *exactly* rather than arithmetically: at $\lambda = 0$ the code returns `flow_energy` untouched, at $\lambda = 1$ `physical_energy`, so a non-finite physical leg cannot poison a $\lambda = 0$ total through $0 \times \infty$, and $\lambda = 1$ is bitwise the pre-existing target. And bounding sits outside the mix, carrying weight $1$ at every $\lambda$, so a stored pair of legs re-mixes to a row's live total by a plain weighted sum.

Two things the constructor and the energy call refuse. `::MolecularCrystal.__init__` raises when `lambda_mix` differs from 1 and no `prior_flow_path` is set. `generator_energy` raises when a flow and the `energy_clip` attribute are both active, that clip being set from `cfg:energy_config.reward_range` by `::MolecularCrystal.set_reward_clip` and being a nonlinear rescale of the physical energy, which would make the $\lambda = 0$ endpoint something other than the flow. `cfg:energy_config.physical_energy_clip` is a separate absolute cutoff that log-compresses the crystal term inside the physical leg alone and publishes `physical_clip_active_frac`; the $\lambda = 0$ endpoint is still exactly the flow under it.

`latent_knn` is a different object: a value of the top-level `cfg:energy_function` scoring `crystal_energy` through `energies/prior_knn.py::PriorKNN` from a stored reference draw (`cfg:energy_config.prior_knn_path`). It fills the physical slot, not the flow slot.

## The path is a one-parameter exponential family

Group the $\lambda$-independent terms into a base density $p_0(x) \propto \exp\!\big(-(E_{\text{flow}} + E_{\text{bound}})/T\big)$ and define

$$\Delta(x) \;=\; \frac{E_{\text{phys}}(x) - E_{\text{flow}}(x)}{T}.$$

Then $E_\lambda/T = (E_{\text{flow}}+E_{\text{bound}})/T + \lambda \Delta$, so

$$p_\lambda(x) \;=\; \frac{p_0(x)\, e^{-\lambda \Delta(x)}}{Z(\lambda)}, \qquad Z(\lambda) = \mathbb{E}_{p_0}\big[e^{-\lambda\Delta}\big].$$

This is an exponential family in the natural parameter $-\lambda$ with sufficient statistic $\Delta$ and log-partition function $\psi(\lambda) = \log Z(\lambda)$. $E_\lambda/T$ is exactly linear in $\lambda$, so $\Delta$ is sufficient: nothing else about a row is needed to move it along the path, and stored legs re-mix without a rescore.

Differentiating under the integral,

$$\psi'(\lambda) = -\,\mathbb{E}_{p_\lambda}[\Delta], \qquad \psi''(\lambda) = \mathrm{Var}_{p_\lambda}(\Delta) \;\ge\; 0,$$

so $\psi$ is convex in $\lambda$, strictly unless $\Delta$ is $p_\lambda$-almost surely constant, and $\mathbb{E}_{p_\lambda}[\Delta]$ is monotone non-increasing along the path.

**Neighbouring rungs.** For a step $d\lambda$,

$$\log\frac{p_\lambda(x)}{p_{\lambda + d\lambda}(x)} = d\lambda\,\Delta(x) + \psi(\lambda + d\lambda) - \psi(\lambda),$$

and taking the expectation under $p_\lambda$ with the second-order expansion $\psi(\lambda+d\lambda)-\psi(\lambda) = -d\lambda\,\mathbb{E}_\lambda[\Delta] + \tfrac12 \mathrm{Var}_\lambda(\Delta)\,d\lambda^2 + O(d\lambda^3)$ gives

$$\mathrm{KL}\big(p_\lambda \,\|\, p_{\lambda+d\lambda}\big) \;=\; \tfrac12\,\mathrm{Var}_\lambda(\Delta)\,d\lambda^2 + O(d\lambda^3),$$

symmetric in its two arguments to this order. Constant KL per rung therefore means

$$d\lambda \;=\; \frac{\sqrt{2\varepsilon}}{\mathrm{sd}_\lambda(\Delta)},$$

i.e. rung size inversely proportional to the sufficient statistic's standard deviation under the current target.

## How the code moves lambda

`lambda_mix` is a numeric `energy_config` key, so it is eligible for the `anneal_coeffs` machinery that also ramps `bounding_coeff` and `reduction_coeff`. The mechanics:

- A stage's `balance` block must have `kind: lexicographic`; `anneal_coeffs` on any other kind raises. Each entry is `{target, rate}`, no other keys.
- `::StageProtocol._balance_tick` runs on the 10-step protocol tick (`::StageProtocol.tick`), evaluating the stage's rules in order: if none is violated the clean streak increments, any violation resets it to 0.
- When the streak reaches `cfg:controller.anneal_patience`, `::StageProtocol._anneal` fires once: every annealed absolute rule's threshold is tightened by its own `rate` or `cfg:controller.decay_rate` down to its `min` (a number, or a live metric name), then every `anneal_coeffs` entry still below its `target` is ramped by `val <- min(val / rate, target)` (an entry at or above its target is left alone). The ramp runs only after every rule has gone quiet.
- `balance.anneal_cooldown_steps` spaces the events, measured from the later of the stage's first tick and its last anneal event (`last_anneal_step`). Before the first event the rules are not evaluated at all; after an event they are, so running bests keep tracking. The streak is held at 0 in both windows.
- `train.py::Modeller.set_energy_coeffs` writes the live values from `::StageProtocol.energy_coeffs` onto the energy function every tenth step. The base `energy_config` value is in effect from run start; only a stage naming the key moves it.

The ramp is multiplicative and cannot leave exactly zero, so a mixing run starts at a small positive `lambda_mix`. The `configs/mk_dev.yaml` terminal conditional stage carries `anneal_coeffs: {lambda_mix: {target: 1.0, rate: 0.5}}`, i.e. $\lambda \leftarrow \min(2\lambda, 1)$, with `anneal_cooldown_steps: 2000` and two pacing rules on `gates/delta_mean` and `gates/delta_worst`, each `relative: best` with `drift: 0.0` and `if_missing: violated`; those gates come from the per-condition log-$Z$ tracker, logged as `zmatch/delta_mean` and `zmatch/delta_worst`. A geometric rung, $d\lambda = \lambda$, is the constant-KL schedule exactly when $\mathrm{sd}_\lambda(\Delta) \propto 1/\lambda$; the config comment instead records the settled within-condition spread as growing like $\lambda\,\sigma_\Delta$ above a $\lambda=0$ floor, under which a geometric rung is a constant *relative* injection rather than a constant KL. The live scalar is logged as `energy/lambda_mix`, read off the energy function, beside `protocol/coeff_lambda_mix` and `protocol/anneal_events`.

## What a rung does at the instant it moves

The target changes between one step and the next while the policy, the backward policy and every stored $\log Z$ are where the old target left them. For a row $x$, $\log R$ moves by $-d\lambda\,\Delta(x)$, and so does the trajectory log-weight $u = \log R + \log P_B - \log P_F$, neither policy term containing $\lambda$. A rung therefore injects the sufficient statistic itself, scaled: a common shift $-d\lambda\,\mathbb{E}[\Delta]$ plus a row-to-row spread $d\lambda\,\mathrm{sd}(\Delta)$. Against a scalar or persistent $\log Z$ all of that lands in the trajectory-balance residual ([trajectory-balance](trajectory-balance.md)); under condition-grouped VarGrad the group mean absorbs whatever is common within a condition, leaving the within-condition spread of $\Delta$ ([vargrad](vargrad.md)). The Huber knee `beta` is a per-branch loss coefficient, written from the stage's `loss_coeffs` by `train.py::Modeller.set_loss_coeffs` and carrying no $\lambda$ dependence, so the knee a rung moves residuals against does not move with $\lambda$.

Stored rows are re-scored, not carried at their admission $\lambda$. A replay draw goes through `::MolecularCrystal.prebuilt_sample_to_reward`, which calls `generator_energy` again on the stored batch, so its `log_reward` is composed at the current $\lambda$, with the stored trajectory's pre-clamp terminal passed as `raw_latents` (`train.py::Modeller._finish_replay_draw`) so bounding is scored live. The stored terminal force is kept per leg in the buffer's `force_legs` column (`buffer.py::N_FORCE_LEGS` = 3, in `::MolecularCrystal.REWARD_LEGS` order flow, phys, bound, from `gflownet_losses.py::terminal_force_legs`) and re-mixed at the current $\lambda$ by `::remix_force_legs`. `train.py::Modeller._prior_row_energy` likewise composes a prior-buffer row's energy from its two stored legs rather than reading a frozen scalar.

**Every force leg stored before 2026-09-21 has an identically zero flow column.** `::terminal_force_legs` differentiates each of `REWARD_LEGS` in turn, and `energies/prior_flow.py::PriorFlow.energy` wrapped its forward in `torch.no_grad()` until that date, so the flow leg's gradient was structurally zero rather than small. `::remix_force_legs` then recombines $(1-\lambda)\cdot 0 + \lambda F_{\text{phys}}$, i.e. a stored force that is the physical leg alone at any $\lambda$, including $\lambda \to 0$ where the flow is the entire target. Buffers restored from a run that predates the change carry that zero column with no stamp distinguishing it, and nothing re-derives the legs on restore.

What does not move with $\lambda$ is the anchor currency: `train.py::Modeller._anchor_energy_phys` defines $E_{\text{anchor}}$ as the total at $\lambda = 1$, `physical_energy + bounding_energy * bounding_coeff`, and an anchor store restored without the `ANCHOR_ENERGY_CURRENCY` stamp raises `BufferCurrencyError` on a run with a `prior_flow`.

At a rung, every row's reward and force are recomposed at the new $\lambda$, while which rows are present and where the per-condition $\log Z$ estimates sit were fixed under the old one.

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:energy_function`; `cfg:energy_config.{lambda_mix, prior_flow_path, prior_knn_path, reward_range, physical_energy_clip, bounding_coeff, reduction_coeff, temperature}`; `cfg:controller.{anneal_patience, decay_rate}`; per stage, `balance.{kind, anneal_coeffs, anneal_cooldown_steps, default_boost, rules}` with rule keys `{metric, relative, margin, drift, if_missing, boost, anneal}` and an `anneal` spec's `{rate, min}`.

Code: `energies/molecular_crystal.py::MolecularCrystal.generator_energy`, `::MolecularCrystal.prebuilt_sample_to_reward`, `::MolecularCrystal.energy`, `::MolecularCrystal.__init__`, `::MolecularCrystal.set_reward_clip`, `::MolecularCrystal.REWARD_LEGS`; `energies/prior_flow.py::PriorFlow.energy`; `energies/prior_knn.py::PriorKNN`; `gflownet_losses.py::remix_force_legs`, `::terminal_force_legs`; `protocol.py::StageProtocol._anneal`, `::StageProtocol._balance_tick`, `::StageProtocol.energy_coeffs`, `::StageProtocol.tick`; `train.py::Modeller.set_energy_coeffs`, `::Modeller.set_loss_coeffs`, `::Modeller._lambda_metrics`, `::Modeller._prior_row_energy`, `::Modeller._anchor_energy_phys`, `::Modeller._finish_replay_draw`; `buffer.py::N_FORCE_LEGS`; `build_prior_flow.py`, `build_prior_knn_reference.py`.

Metrics: `energy/lambda_mix`, `protocol/coeff_lambda_mix`, `protocol/anneal_streak`, `protocol/anneal_cooling`, `protocol/anneal_events`, `zmatch/delta_mean`, `zmatch/delta_worst`.

## Sources

Repo: `configs/mk_dev.yaml` (the `energy_config` lambda-path block and the terminal conditional stage's `balance`) and the code above, at the stamped commit. No file under `docs/design/` derives this path; the sufficient-statistic statement appears in the `mk_dev.yaml` and `generator_energy` comments, and the derivation above is this page's. Memory files project_lambda_anneal_path_geometry, project_lambda_overstep_response, project_lambda_zero_null_test_passes, project_latent_knn_prior_energy, project_qm9c_anneal_r2gate_arm and project_var_conditioning_pooled_anneal_default located the code and were not used as evidence.
