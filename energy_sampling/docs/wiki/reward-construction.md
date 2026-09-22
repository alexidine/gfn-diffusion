# Reward construction

*Drift: **M** (mixed). Verified against commit `a637e70`, 2026-09-20. Sources at the end.*

Every training branch consumes one per-row scalar, $\log R$. This page is about how a scored structure becomes that number: which energy terms are summed, which temperature divides the sum, what units the result is in, and which clips act on it. It is route-agnostic. Which backend produced the physical energy is [mlip-energy-routes](mlip-energy-routes.md); how the reduction penalty's walls are derived is [cell-fundamental-domains](cell-fundamental-domains.md); what a lattice sum contains is [crystal-force-fields](crystal-force-fields.md); where $\log R$ enters each branch's residual is [loss-composition](loss-composition.md).

## The chain from structure to log R

Three functions, in order. `energies/molecular_crystal.py::MolecularCrystal.generator_energy` composes a per-row total energy $E$ out of the terms below. `::MolecularCrystal.energy` forms $T = 10^{\log T}$ from its `log_temperature` argument and returns $E / T$. `energies/base_set.py::BaseSet.log_reward` negates that. So

$$\log R = -\,E(x_T) \,/\, T ,$$

with no additive constant on the path. The stored-row path is the same arithmetic in one function: `::MolecularCrystal.prebuilt_sample_to_reward` calls `generator_energy` and returns `-energy / sample_temperature`.

$\log T$ is per row. With `cfg:temperature_conditioning` false, `::MolecularCrystal.condition_samples` sets it to $\log_{10}$ of `cfg:energy_config.temperature` for every row; with it true, $\log T$ is drawn uniformly on `cfg:energy_config.log_temperature_range` and appended to the condition vector, so the temperature is both the divisor and a policy input.

## What the total energy is made of

On the physical routes (`lj`, `qlj`, `elj`, `silu`, `uma`, `mace`) `generator_energy` builds

$$E \;=\; \underbrace{E_{\text{mol}} + c_\rho\,E_\rho + E_{PV}}_{\texttt{crystal\_energy}} \;+\; c_{\text{red}}\,T\,\mathrm{relu}(\texttt{reduction\_en}) \;+\; E_{\text{jac}} \;+\; c_{\text{bnd}}\,E_{\text{bnd}} .$$

- $E_{\text{mol}}$ is the attribute named by `cfg:energy_function`, read off the batch after `crystal_batch.analyze`, divided by `z_prime` on every route except `uma` and `mace`, and multiplied by `cfg:energy_config.lj_rescale` on the three LJ-family routes when that key is set.
- $E_\rho$ is `molecular_crystal.py::density_penalty` on the packing coefficient: a hinge on $\mathrm{relu}(\log 0.55 - \log c_p)$ that is quadratic to a turnover of 1 and linear beyond it, plus $\mathrm{relu}(c_p - 0.95)^2$. Its coefficient is `cfg:energy_config.density_coeff`.
- $E_{PV}$ is `cfg:energy_config.pressure` in atmospheres, converted to kJ/mol, times `cell_volume / sym_mult / z_prime`. The canonical config carries neither `pressure` nor `lj_rescale`, so both take the constructor defaults, 1 atm and `None`.
- The reduction term is the stored `reduction_en`, rectified and multiplied by $T$ before its coefficient `cfg:energy_config.reduction_coeff`, so its weight in the Boltzmann exponent does not move with $T$.
- $E_{\text{jac}}$ is the change of measure from box latents to physical coordinates, `::MolecularCrystal.compute_jacobian`, itself premultiplied by $T$ in each of its three components: the two rotational terms $-2T\log\sin(r/2)$ and $-T\log\sin\theta$, and $-T z' \log(V/\text{sym\_mult})$.
- $E_{\text{bnd}}$ is the bounding term.

### Bounding

`generator_energy` reads the policy's pre-clamp terminal as `raw_latents` and forms $\sum_i \mathrm{relu}(x_i - 1)^2 + \mathrm{relu}(-(x_i+1))^2$, a pure quadratic hinge on the $[-1,1]$ latent box; the quartic terms beside it are commented out. When `max_z_prime > 1` the Z'-ordering penalty `::MolecularCrystal.compute_zp_order_penalty` is added into it. The sum is multiplied by $T$, then by `cfg:energy_config.bounding_coeff`, and added to the total once, outside the lambda mix. Without `raw_latents` the term is zero, which is the prior-buffer path; the replay path supplies the stored trajectory's last state. On the live forward path `::MolecularCrystal.analyze_crystal_batch` passes the policy output `x` itself. `models/gfn.py::GFN.predict_next_state` clips the emitted next state to $\pm$ `cfg:model.gfn_clip` when `cfg:model.clipping` is true, which caps the reachable latent and therefore the reachable bounding penalty.

## Units and the lj_coeff conversion

Energies are in kJ/mol and `cfg:energy_config.temperature` is a $kT$ in kJ/mol. On the eLJ route the raw lattice sum is in reduced units, and the conversion is `lj_coeff`: `::MolecularCrystal.stamp_lj_coeff`, called from `analyze_crystal_batch`, writes the run's value onto every freshly built batch as a per-graph attribute, `mxtaltools/dataset_utils/data_class_methods/crystal_analysis.py::MolCrystalAnalysis.compute_eLJ_energy` multiplies its own output by it (defaulting to 1.0 when the attribute is absent), and `generator_energy` consumes `mol_energy` unscaled. `::MolecularCrystal.assert_lj_coeff_stamped` raises on a batch carrying no `lj_coeff` attribute and on any row whose stamped value differs from the run's by more than `LJ_COEFF_RTOL` (1e-6, relative), so a stored row in raw units and a batch mixing two calibrations both raise.

`train.py::Modeller.init_prior_dataset` reads `thermal_scaling_factor` off the prior `.pt` and assigns it to `self.energy_function.lj_coeff`, overriding `cfg:energy_config.lj_coeff` for the whole run, before the re-analysis pass. The key's presence is what triggers it; priors written without it leave the config value in force. A factor differing from 1.0 by more than 1e-9 with `cfg:energy_function` other than `elj` raises, since the multiply happens only inside `compute_eLJ_energy`. `calibrate_qm9_energy.py` writes the key into a prior file in place.

## The two clips on the energy

`cfg:energy_config.reward_range` is a width in reward units. `::MolecularCrystal.set_reward_clip` takes the maximum of the rewards it is handed, sets `reward_clip` to that maximum minus `reward_range`, and `energy_clip` to $-\,$`reward_clip`$\,\times$ `self.temperature`, the configured temperature rather than a per-row one. It is called from `init_prior_dataset`, after the prior re-analysis pass and on that pass's own rewards $-E/T$ formed at that same temperature, so the threshold is measured against unclipped prior energies. The arming prints the range, the clip and the reward clip, then the gap between the maximum reward and the 99.9th percentile; `init_prior_dataset` raises when that gap exceeds $0.1 \times$ `reward_range`.

When `energy_clip` is set, `generator_energy` takes a different branch: the crystal energy is passed through `mxtaltools/common/utils.py::log_rescale_positive` (which is $\text{cutoff} + \log(1 + y - \text{cutoff})$ above the cutoff and the identity below, not a clamp), the bounding total and the reduction term are added to it, that sum is passed through the same rescale at a cutoff of `energy_clip` $+\ 0.1\,|$`energy_clip`$|$, the jacobian is added after, and the bounding total is then zeroed so it is not added twice. `ens_dict['energy_clip_active_frac']` reports the fraction of rows above the cutoff; the key passes `train.py::Modeller.log_thermo_properties`, which filters `ens_dict` on `'energy' in key or 'pot' in key` and excludes `gfn_energy`. This branch and a prior flow are mutually exclusive: `generator_energy` raises when both are configured.

`cfg:energy_config.physical_energy_clip` is the separate, absolute form: it log-compresses the crystal energy alone above a fixed cutoff, leaving bounding and the lambda mix untouched, and reports `physical_clip_active_frac`. Neither clip absorbs a non-finite energy. `::MolecularCrystal._assert_finite_energy` raises on any non-finite total, naming the offending component, except on the `uma` and `mace` routes listed in `_TOLERATES_NONFINITE`.

Downstream of $\log R$, `cfg:buffers.replay_buffer.admit_reward_min` is a hard exclusion on the replay admission path in `train.py::Modeller.manage_replay_buffer`: rows with $\log R$ below it, and rows with non-finite $\log R$ or residual, are removed before any prioritisation, and the count is tallied to `replay_churn['reward_rejected']`.

## Physical energy against the composite

Two per-row quantities are carried. The composite is the total above, which under lambda mixing is $(1-\lambda)$ times the flow leg plus $\lambda$ times the physical leg, plus bounding, with the two endpoints taken exactly rather than arithmetically. `ens_dict['physical_energy']` is the physical leg alone: crystal energy, reduction times its coefficient, and jacobian, with bounding outside it. `train.py::Modeller._anchor_energy_phys` forms the anchor currency as `physical_energy + bounding_energy * bounding_coeff` when a prior flow exists, and returns `None` otherwise; `::Modeller._anchor_energy` is what call sites use to read it off a scored batch. `buffer.py::ConditionLogZTracker.update_best_energy` takes both, keeping `best_energy` over the composite and `best_energy_phys` over the physical total. Four sites in `train.py` and one in `gflownet_losses.py` pass `energy_phys`, whose value is `None` on a flow-free run; while every supplied value is `None` the tracker's `phys_is_alias` stays true, `best_energy_phys` is a bit-for-bit alias of `best_energy`, and `::ConditionLogZTracker.lookup_best_energy(physical=True)` raises. With `requires_phys_energy` set, which the trainer does whenever the energy function carries a prior flow, an implicit alias raises instead.

## A latent-scored target

`latent_harmonic`, `latent_multiharmonic`, `latent_gaussian` and `latent_knn` set `latent_energy` True, which skips the whole physical block: no packing coefficient, no pressure term, no `mol_energy`, no `lj_coeff` assertion. The reduction energy is a structural zero on that path rather than a config-disabled term, and the jacobian is zero whenever `latent_energy` is true or `is_crystal` is false. The reward is then the analytic value plus bounding: for the first three, the attribute `crystal_batch.analyze` produced; for `latent_knn`, `energies/prior_knn.py::PriorKNN.energy` on the gauge-fixed latents rather than on `raw_latents`. `is_crystal` is separate and is False only for the two names in `molecular_crystal.py::TOY_ENERGY_FUNCTIONS`, mirrored for name-only callers by `::is_crystal_energy`.

## The reward gradient

`cfg:fwd_loss_coeffs.reward_grads` is the switch that lets $d\log R / dx_T$ reach the policy: `gflownet_losses.py::get_loss_reward` is called with `no_grad=loss_coeffs.reward_grads == 0`, and under `no_grad=False` the terminal is left attached and the reward call runs with `keep_grads=True`. The gate and force-clip keys beside it reshape that gradient onto the live terminal. Those routes are [gradient-routes-and-terminal-force](gradient-routes-and-terminal-force.md).

## Owner choices

*Left for the owner. Three buckets: bullets bitten, design choices, priorities.*

## Config keys

`cfg:energy_function`, `cfg:temperature_conditioning`, `cfg:energy_config.temperature`, `.log_temperature_range`, `.density_coeff`, `.bounding_coeff`, `.reduction_coeff`, `.lj_coeff`, `.lj_rescale`, `.pressure`, `.reward_range`, `.physical_energy_clip`, `.prior_flow_path`, `.lambda_mix`, `.prior_knn_path`; `cfg:model.clipping`, `cfg:model.gfn_clip`; `cfg:buffers.replay_buffer.admit_reward_min`; `cfg:fwd_loss_coeffs.reward_grads`.

Code, gfn_diffusion: `energies/molecular_crystal.py::MolecularCrystal.generator_energy`, `.energy`, `.prebuilt_sample_to_reward`, `.analyze_crystal_batch`, `.set_reward_clip`, `.stamp_lj_coeff`, `.assert_lj_coeff_stamped`, `.compute_jacobian`, `.compute_zp_order_penalty`, `.condition_samples`, `._assert_finite_energy`, `::density_penalty`, `::is_crystal_energy`, `::TOY_ENERGY_FUNCTIONS`; `energies/base_set.py::BaseSet.log_reward`; `energies/prior_knn.py::PriorKNN.energy`; `energies/prior_flow.py::PriorFlow.energy`; `models/gfn.py::GFN.predict_next_state`; `gflownet_losses.py::get_loss_reward`; `train.py::Modeller.init_prior_dataset`, `._anchor_energy_phys`, `._anchor_energy`, `.log_thermo_properties`, `.manage_replay_buffer`; `buffer.py::ConditionLogZTracker.update_best_energy`, `.lookup_best_energy`; `calibrate_qm9_energy.py`.

Code, mxtaltools: `mxtaltools/common/utils.py::log_rescale_positive`; `mxtaltools/dataset_utils/data_class_methods/crystal_analysis.py::MolCrystalAnalysis.compute_eLJ_energy`.

## Could be tooling

Two checks are mechanical. First, the term inventory: `generator_energy` writes every component into `ens_dict` under a fixed key before composing the total, so an AST walk over its assignments to `ens_dict[...]` yields the published term list, and a diff against the keys `log_thermo_properties` forwards under its `'energy' in key or 'pot' in key` filter names every component computed but never logged, and every key the filter drops.

Second, the unit currency. A loader-time pass over each prior `.pt` referenced by `cfg:prior_path` could report the presence and value of `thermal_scaling_factor` beside the config's `lj_coeff` and the resolved `cfg:energy_function`, which is the triple the override rule in `init_prior_dataset` decides on.

## Sources

The code listed above, read at the stamped commit (mxtaltools as checked out beside it), and the `energy_config`, `model` and `buffers.replay_buffer` blocks of the canonical config `configs/mk_dev.yaml`.
