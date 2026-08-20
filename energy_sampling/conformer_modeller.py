"""``ConformerModeller(Modeller)`` -- the conformer track on train.py's own machinery.

WHY A SUBCLASS RATHER THAN THE STRIPPED LOOP. train_conformer.py ran a parallel training
loop with no protocol, no buffer controllers and no stage machinery, on the stated grounds
that "none of them earn their keep before the sampler is shown to work at all". The sampler
is now shown to work: propanol at `torsion` reached log Z 5.935 against an exact 5.9365.

WHAT THE OVERNIGHT `full` RUN DID AND DID NOT SHOW. It ran 6000 steps in ~16 minutes with
train/loss falling steadily, and it did NOT diverge. The forward and backward branches
reported opposite-signed residuals, but that is a REPORTING ARTIFACT of alternating losses
-- the stripped loop takes turns between branches and logs whichever one just ran, so the
bimodal residual and grad-norm traces are two interleaved series, not one unstable one.
Anyone reading those traces as instability (as this docstring previously did) is reading a
sampling artifact. Owner correction, 2026-08-20.

So the case for this subclass is NOT a rescue. It is that the stripped loop has no MLE warm
start, no stage machinery, no buffer controllers and no batch sizer, and re-deriving them
beside a working implementation is the expensive way to get them.

WHY IT IS A SMALL FILE. Almost everything was already built for exactly this call:

  * ``ConformerTorsions`` implements train.py's 15-member energy protocol, and says in its
    own comment that this is what lets ConformerModeller subclass Modeller.
  * ``buffer.ConformerBuffer`` subclasses CrystalBuffer with the three crystal-specific
    hooks overridden, and names "the same call made for ``ConformerModeller(Modeller)``".
  * ``energies/conformer_data`` supplies condition_from_energy / attach_states /
    bake_energies.

So what remains here is the data-init seam and one mandatory GFN-config correction. The
protocol, LR controllers, batch sizer, buffers, checkpointing and OOM handling are all
inherited unmodified -- which is the entire point of the exercise.

THE PRIOR IS NOT RETRAINED. The crystal route's train_prior stage ends with
``snapshot_prior``, freezing the MLE-trained policy as THE prior model. The conformer
protocol deliberately omits that action: a fitted InternalPrior already exists and
benchmarks 32x-87000x over uniform-on-box on median energy excess, so phase 1 runs only to
broaden the policy space, and must not displace a prior that is already good.
"""
from __future__ import annotations

import numpy as np
import torch

from buffer import ConformerAnchorBuffer, ConformerBuffer
from energies.conformer_torsions import ConformerTorsions
from train import BULKY_ATTR_EXCLUDE_KEYS, Modeller

#: energy_config keys that describe the PROBLEM and are consumed here rather than passed
#: to ConformerTorsions, which takes no **kwargs and would raise on any of them.
_NON_ENERGY_KEYS = ('internal_prior_path', 'prior_sample_size', 'reward_range',
                    'density_coeff', 'reduction_coeff', 'analyze_kwargs',
                    'internal_oom_recovery')


class ConformerModeller(Modeller):

    #: every churned store is a ConformerBuffer. The three crystal-specific hooks it
    #: overrides (``_as_batch``, ``_orient_stored_batch``, ``_compute_xy``) are the whole
    #: difference; row draws, EMA bookkeeping, admission, purge, TTL and persistence are
    #: graph-agnostic and inherited.
    buffer_cls = ConformerBuffer

    #: and the anchor store, which is NOT buffer_cls -- AnchorBuffer has its own signature
    #: and its own reward/energy/surprise state. ConformerAnchorBuffer is that class with
    #: the same three graph hooks mixed in ahead of it in the MRO.
    anchor_buffer_cls = ConformerAnchorBuffer

    def _buffer_kwargs(self):
        """No ``max_z_prime``: a conformer graph has no asymmetric unit.

        It is not merely unused -- ``MXtalBase.__getattr__`` defers unknown attributes to
        the PyG store, so passing it RAISES rather than being ignored.
        """
        return {}

    def _buffer_y_fn(self):
        """``conformer_energy``, not the energy_function name.

        On the crystal route those coincide -- the analysis attaches its term under the
        backend's own name -- so the base class can use one for the other. Here they do
        not: the scalar is baked as ``conformer_energy`` by ``bake_energies``.
        """
        return 'conformer_energy'

    def _batch_latents(self, batch):
        """The conformer state is the stored ``torsion_state``, not a cell latent.

        ``batch_states`` is the same reader ``ConformerBuffer._compute_xy`` uses, so a row
        drawn from a buffer and a row read here give the identical vector -- which is what
        makes backward/replay training on stored rows sound at all.
        """
        from energies.conformer_data import batch_states

        return batch_states(batch)

    # ------------------------------------------------------------- eval figures

    #: DoF classes, in ``_free_block`` order. The axis differs per class and that is the
    #: whole reason they are not plotted together: r and theta are LINEAR box coordinates,
    #: phi is an ANGLE on a circle. torsion_latent_figure scaled every column by 180 and
    #: ranged it as degrees, which draws a bond-length distribution as though it were an
    #: angle (train_conformer.torsion_latent_figure, scaling problem 3).
    _DOF_CLASSES = ((0, 'r (bond length)', 1.0), (1, 'theta (angle)', 1.0),
                    (2, 'phi (torsion, deg)', 180.0))

    def _domain_figs(self, fig_dict, sample_batch, prior_latent_params, anchor_latents):
        """Sampler vs prior over the latent box, GROUPED BY DoF CLASS.

        One panel per class rather than one per column. train_conformer's
        torsion_latent_figure put one panel per state dimension at width 300*k, which is
        9000 px at propanol/`full` (d = 30) and worse on anything real -- the first of the
        three scaling problems its own docstring records. Grouping by class is the option
        that docstring names as scaling to any d, and it also fixes the third problem
        (the axis is now per class, so bond lengths are no longer drawn as degrees).

        What it does NOT do is preserve per-column identity, so a single bad column is
        averaged into its class. That is deliberate for now: per the agreed framing
        (2026-08-20) the distribution plot is the DRILL-DOWN, and the primary diagnostic is
        the scalar that trips -- see log_physical_properties. The worst-K-by-divergence
        variant is the next step and is not built.
        """
        from plotly.subplots import make_subplots

        from energies.conformer_data import batch_states
        from mxtaltools.reporting.utils import lightweight_one_sided_violin

        def host(t):
            """-> numpy on the host. prior_latent_params arrives as a CUDA tensor when
            buffer_device is 'cuda', and np.asarray on one raises rather than copying."""
            if t is None:
                return None
            return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

        samples = host(batch_states(sample_batch))
        reference = host(prior_latent_params)
        block = host(self.energy_function._free_block)
        present = [(b, title, scale) for b, title, scale in self._DOF_CLASSES
                   if (block == b).any()]
        if not present:
            return

        fig = make_subplots(rows=1, cols=len(present),
                            subplot_titles=[f'{t}  (n={int((block == b).sum())})'
                                            for b, t, _ in present])
        for col, (b, _title, scale) in enumerate(present):
            cols = np.flatnonzero(block == b)
            dists = [('sampler', samples[:, cols], 'rgba(60,120,216,0.55)')]
            if reference is not None and reference.shape[1] == block.shape[0]:
                dists.append(('prior', reference[:, cols], 'rgba(235,104,52,0.45)'))
            lo, hi = -scale, scale
            for name, data, color in dists:
                x_v, y_v = lightweight_one_sided_violin(
                    torch.as_tensor(scale * data.reshape(-1)), 200,
                    bandwidth_factor=0.05, data_min=lo, data_max=hi)
                fig.add_scatter(x=x_v, y=y_v, mode='lines', fill='toself',
                                fillcolor=color, line=dict(color=color, width=1.2),
                                name=name, legendgroup=name, showlegend=(col == 0),
                                row=1, col=col + 1)
            # a hair past the box so mass piled ON the wall is visible as mass at the edge
            # rather than being clipped flush against the frame
            fig.update_xaxes(range=[lo * 1.03, hi * 1.03], row=1, col=col + 1)
            fig.update_yaxes(showticklabels=False, row=1, col=col + 1)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          height=300, width=340 * len(present),
                          margin=dict(l=30, r=20, t=44, b=30))
        fig_dict['Latent Distribution by DoF Class'] = fig

    # ------------------------------------------------------------ eval metrics

    def _eval_extra_stats(self, mol_batch):
        """No packing coefficient: a conformer has no cell.

        Returns nothing rather than a NaN column. The quantity is ABSENT, not unknown, and
        a column of NaNs would be averaged into a published metric that reads as a number.
        """
        return {}

    def _in_box(self, state):
        """Per-sample: every NON-PERIODIC state column inside the latent box.

        Only the linear blocks are bounded -- ``bounding_energy`` walls exactly
        ``_lin_free_idx`` and the phi block WRAPS, so applying |x| <= 1 to it would mark
        perfectly ordinary torsions as out of bounds.
        """
        idx = self.energy_function._lin_free_idx.to(state.device)
        if idx.numel() == 0:
            return torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
        return (state.index_select(-1, idx).abs() <= 1.0).all(dim=-1)

    def _reasonable_sample_mask(self, sample_batch):
        """The conformer's 'is this physically reasonable', on the same absolute footing.

        The crystal bar is a hand-set window: bound energy plus packing in 0.55-0.95. The
        conformer equivalent has to be equally absolute rather than relative to the current
        batch, or 'Reasonable Sample Fraction' stops being comparable across a run:

          * the potential is FINITE -- a blown-up geometry gives inf/nan, and
          * every non-periodic DoF is INSIDE the box, i.e. bond lengths within delta_r_max
            and angles within delta_theta_max of the reference. Outside it the geometry is
            nonphysical, which is exactly what the wall term is there to penalise.
        """
        from energies.conformer_data import batch_states

        e = getattr(sample_batch, 'conformer_energy', None)
        if e is None:
            raise AttributeError(
                "_reasonable_sample_mask needs `conformer_energy`; every batch that "
                "reaches eval is written by set_batch_states, which attaches it")
        state = batch_states(sample_batch)
        return torch.isfinite(e.flatten()) & self._in_box(state.to(e.device))

    @property
    def _e_min(self):
        """The tier's own minimum energy, the ZERO every excess is measured against.

        Multi-start local search (prior_baselines.tier_minimum), so it is an UPPER bound on
        the true minimum and every excess built on it is a lower bound -- a uniform shift,
        which leaves within-run comparisons intact. Computed ONCE: it is a property of the
        molecule and the force field, not of the model, so recomputing per eval would burn
        150 Rprop steps to get the same number and would let the reference drift.
        """
        if getattr(self, '_e_min_cache', None) is None:
            from energies.prior_baselines import tier_minimum

            # STARTS COME FROM THE PRIOR, not from a uniform box draw. ConformerTorsions
            # has no closed-form sampler (`sample` raises on purpose), and more to the
            # point a multi-start minimum is only as good as its basin coverage -- prior
            # draws already spread over the rotamer modes, a box draw does not.
            src = self.prior_dataset.x
            take = torch.randperm(src.shape[0])[:256]
            starts = src[take].detach().clone().to(self.device)
            best, worst, n = tier_minimum(self.energy_function, starts)
            self._e_min_cache = best
            print(f'energy zero: tier minimum {best:.3f} kcal/mol over {n} starts '
                  f'(worst start {worst:.3f}) -- an UPPER bound, so every excess is a '
                  f'lower bound')
        return self._e_min_cache

    @property
    def _basin_ref(self):
        """The target's accessible rotamer basins. Sampler-independent, so computed once."""
        if getattr(self, '_basin_ref_cache', None) is None:
            from energies.prior_diagnostics import basin_reference

            self._basin_ref_cache = basin_reference(self.energy_function)
            r = self._basin_ref_cache
            if 'skipped' in r:
                print(f'basin coverage UNAVAILABLE: {r["skipped"]}')
            else:
                print(f'basin coverage: {int(r["accessible"].sum())} accessible of '
                      f'{len(r["combos"])} rotamer modes')
        return self._basin_ref_cache

    @property
    def _target_coupling(self):
        """How coupled the TARGET's rotamer landscape is. Cached: sampler-independent."""
        if getattr(self, '_target_coupling_cache', None) is None:
            from energies.conformer_eval_metrics import target_coupling

            tc = target_coupling(self._basin_ref)
            self._target_coupling_cache = tc
            import math
            if math.isnan(tc):
                print('target coupling UNAVAILABLE (fewer than 2 rotamer groups)')
            else:
                print(f'target coupling: {tc:.4f} nats over the rotamer groups -- '
                      f'{"essentially UNCOUPLED, so basin coverage is trustworthy here" if tc < 0.05 else "COUPLED: basin coverage over-counts reachable conformers on this molecule"}')
        return self._target_coupling_cache

    def _molecule_features(self, sample_batch):
        """Per-sample molecule descriptors for the correlation block.

        Constant on this unconditional run, so ``feature_correlations`` refuses rather than
        reporting a 0/0 correlation -- and the same call becomes live unchanged the moment
        the conditional route trains on a library, which is the point of wiring it now.
        """
        n = sample_batch.num_graphs
        en = self.energy_function
        n_atoms = float(np.asarray(en.spec.z).shape[0])
        n_rings = float(len(getattr(en, 'ring_cycles_cache', []) or []))
        try:
            from energies.ring_metrics import ring_cycles
            n_rings = float(len(ring_cycles(en)))
        except Exception:
            pass
        return {'size': np.full(n, n_atoms), 'n_rings': np.full(n, n_rings)}

    def log_physical_properties(self, metrics, sample_batch, val, arr):
        """Publish the conformer's OWN physical reading, not an empty block.

        Packing coefficient and reduction energy are cell properties. The conformer
        analogues are the two halves of the reasonableness bar, reported separately so a
        drop in the combined fraction can be attributed: geometry leaving the box is a
        sampler problem, a non-finite potential is an energy problem, and pooling them
        would hide which.
        """
        from energies.conformer_data import batch_states

        e = getattr(sample_batch, 'conformer_energy').flatten()
        state = batch_states(sample_batch).to(e.device)
        finite = torch.isfinite(e)
        metrics['In Box Fraction'] = val(self._in_box(state).float().mean())
        metrics['Finite Energy Fraction'] = val(finite.float().mean())
        if bool(finite.any()):
            metrics['Mean Conformer Energy'] = val(e[finite].mean())
            metrics['Conformer Energy'] = arr(e[finite])

        # ---- the sample-side statistics: energy composition, geometry, coverage --------
        # Everything below is a function of the SAMPLES and the force field only, so none
        # of it can be satisfied by the policy and the flow head agreeing with each other
        # -- which is the one thing the whole TB metric family cannot rule out.
        import energies.conformer_eval_metrics as cm

        en = self.energy_function
        prior_x = getattr(getattr(self, 'prior_dataset', None), 'x', None)
        prior_y = getattr(getattr(self, 'prior_dataset', None), 'y', None)

        metrics.update(cm.energy_component_stats(en, state))
        metrics.update(cm.geometry_stats(en, state))
        metrics.update(cm.dof_class_stats(en, state, reference=prior_x))
        metrics.update(cm.dof_element_stats(en, state))
        metrics.update(cm.ring_stats(en, state))
        # per-CYCLE ring torsion distributions against the prior. The pooled phi histogram
        # averages two rings that can fail in OPPOSITE directions into one blob -- on
        # phenyl-THP the aromatic ring runs too WIDE while the saturated ring COLLAPSES,
        # and pooled they partly cancel. corr_dist is the term that sees a ring matching
        # every marginal while never closing.
        metrics.update(cm.ring_torsion_stats(en, state, reference=prior_x))
        metrics.update(cm.basin_coverage(en, state, self._basin_ref))
        # the QUALIFIER on the line above: coverage's documented false pass is on
        # molecules whose groups are coupled, so this says whether that caveat bites on
        # THIS molecule. Same call block as n_missed, and not optional -- read alone,
        # coupling reports 0 for a collapsed sampler.
        metrics.update(cm.basin_coupling(en, state, self._basin_ref,
                                         target_tc=self._target_coupling))
        # the non-thermal tail regrouped by rotamer basin -- train.py's per-CONDITION
        # version correctly abstains here (one molecule = one condition). Emitted from the
        # same call as basin_coverage on purpose: read alone it rewards mode collapse,
        # because an abandoned basin drops out of the grouping instead of failing it, so
        # n_missed has to be on the same panel. See basin_nonthermal.
        metrics.update(cm.basin_nonthermal(
            en, state, e, self._e_min, self._basin_ref,
            float(getattr(self.args, 'nonthermal_entropy_per_dim', 4.0) or 0.0)
            * int(en.ndim)))
        metrics.update(cm.thermal_stats(en, e[finite], self._e_min))
        if prior_y is not None:
            metrics.update(cm.energy_vs_reference(e[finite], prior_y))
        metrics.update(cm.feature_correlations(
            e, self._molecule_features(sample_batch), 'corr_E/'))

    # ---------------------------------------------------------------- energy

    def init_energy_function(self):
        """``ConformerTorsions`` in place of ``MolecularCrystal``.

        ``level`` is passed through WITHOUT a fallback and ConformerTorsions takes no
        ``**kwargs``, so a config that omits it fails here rather than silently running
        `torsion` -- the failure mode that class was explicitly built to prevent.
        """
        import inspect

        cfg = {k: v for k, v in vars(self.args.energy_config).items()
               if k not in _NON_ENERGY_KEYS}
        cfg['device'] = str(self.device)
        cfg['temperature_conditioning'] = self.args.temperature_conditioning

        # FILTERED AGAINST THE SIGNATURE, AND THE DROPS ARE ANNOUNCED. energy_config is
        # shared with the crystal route and carries keys this energy has no concept of
        # (`temperature` -- the conformer derives kT from log_temperature). Passing them
        # raises, because ConformerTorsions deliberately has no **kwargs. But dropping
        # them SILENTLY would reintroduce exactly the swallowing that decision prevents,
        # so anything discarded is named at startup.
        accepted = set(inspect.signature(ConformerTorsions.__init__).parameters)
        dropped = sorted(k for k in cfg if k not in accepted)
        for k in dropped:
            cfg.pop(k)
        if dropped:
            print(f'energy_config: ignoring {dropped} -- not parameters of '
                  f'ConformerTorsions (crystal-route keys)')
        self.energy_function = ConformerTorsions(**cfg)
        print(self.energy_function.describe())
        # the fitted prior, loaded once and reused by init_prior_dataset and by any later
        # prior top-up. Held on the modeller rather than the energy: it is a sampling
        # device for the trainer, not part of the reward.
        path = getattr(self.args.energy_config, 'internal_prior_path', None)
        self.internal_prior = (torch.load(path, weights_only=False) if path else None)
        if self.internal_prior is not None:
            ver = vars(self.internal_prior).get('ring_sig_version', 1)
            if ver < 2:
                raise SystemExit(
                    f'{path} has ring_sig_version {ver} (pre-fix): no ring key resolves, '
                    f'so every ring falls through to the hold. Rebuild it with '
                    f'build_ring_banks.py before training on a ring molecule.')

    # ------------------------------------------------------------------ GFN

    def _build_gfn_config(self):
        """Base config plus ``angular_mask``, which is NOT optional here.

        ``GFN.get_periodic_dimensions``' non-crystal branch writes ``[False] * dim``. On a
        conformer that is not a degraded layout, it is a silently unnormalizable target:
        the phi block is 2-periodic and a reward with no wrap has no finite log Z at all.
        The energy declares the truth via ``periodic_dims``; passing it is what retired the
        old TorsionGFN subclass, and omitting it here would reintroduce the same bug
        through the back door.
        """
        cfg = super()._build_gfn_config()
        cfg['angular_mask'] = self.energy_function.periodic_dims
        return cfg

    def _resolve_periodic_centroid_axes(self):
        """No cell, so no centroids to wrap. config_invariants refuses the flag outright."""
        return None

    def _resolve_dead_latent_rows(self, quiet: bool = False):
        """No dead rows: every conformer state column is driven by construction.

        The crystal route holds rows that a space group pins. ``ConformerTorsions`` has
        already dropped any column driving nothing (see the `keep` mask in its
        constructor), so by the time a state exists every column is live.
        """
        return None

    # ----------------------------------------------------------- anchor paths

    def _noise_and_condition(self, batch, noise_log_range):
        """The conformer form: jitter ``torsion_state``, then condition.

        Mirrors the crystal noiser's convention exactly -- a unit-norm random direction
        with magnitude ``10 ** U(log_min, log_max)`` -- so ``noise_log_range`` means the
        same size of perturbation on both routes and the config comment stays true.

        The ONE difference is the boundary. The crystal clips every latent to [-1, 1];
        here only the non-periodic block is clipped, because the phi block WRAPS. Clipping
        a torsion would pile probability onto +/-pi and turn a rotation through the
        boundary into a hard stop against it.

        No orientation step: the stored orientation of a conformer graph is never read
        (see ConformerGraphHooks._orient_stored_batch), and no sg_ind/z_prime, which do
        not exist on this graph.
        """
        from energies.conformer_data import batch_states, set_batch_states

        log_min, log_max = float(noise_log_range[0]), float(noise_log_range[1])
        state = batch_states(batch)
        direction = torch.randn_like(state)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = torch.rand(state.shape[0], device=state.device)
        magnitude = 10 ** (log_min + (log_max - log_min) * u)
        noised = state + direction * magnitude[:, None]

        lin = self.energy_function._lin_free_idx.to(noised.device)
        if lin.numel():
            noised[:, lin] = noised[:, lin].clip(min=-1, max=1)
        set_batch_states(batch, noised, periodic=self.energy_function.periodic_dims)

        batch, log_T_tensor, condition, condition_id =             self.energy_function.condition_samples(batch)
        return batch, log_T_tensor, condition, condition_id

    # ----------------------------------------------------------- prior draws

    def _has_prior_sampler(self):
        """The fitted InternalPrior IS the prior, so phase-2 churn has a source.

        WITHOUT THIS the port has a silent hole. The crystal route's prior is a frozen GFN
        produced by train_prior's ``snapshot_prior``, and this protocol deliberately omits
        that action (the fitted prior is already good and must not be displaced). So
        ``hasattr(self, 'prior_model')`` is False forever, _prior_churn_cycle skips its
        draw, and the phase-2 prior buffer degrades to 100% anchors -- reported once as a
        WARNING and thereafter invisible, since an anchor-only buffer is a legal
        composition. Observed in the transition probe, 2026-08-20.
        """
        return self.internal_prior is not None

    def sample_from_prior(self, num_samples):
        """Draw from the fitted InternalPrior instead of rolling out a frozen GFN.

        Returns the same ``(metrics, sample_batch)`` contract fwd_eval_sampling gives the
        crystal route, restricted to the three keys _prior_churn_cycle actually reads:
        ``log_r``, ``log_T_tensor`` and ``condition_id``. There is exactly one condition on
        this route, so condition_id is all zeros rather than something to look up.

        log_r comes from ``prebuilt_sample_to_reward`` rather than being recomputed, so a
        churned row carries the SAME reward the buffer will later read back off it.
        """
        from energies.conformer_data import (attach_states, bake_energies,
                                             condition_from_energy)

        n = max(int(num_samples), 2)      # attach_states refuses a single row
        states, _ = self.energy_function.sample_prior_states(
            self.internal_prior, n, self._prior_rng, report=False)
        energies = bake_energies(self.energy_function, states)
        cond = condition_from_energy(self.energy_function,
                                     identifier=self.energy_function.smiles)
        batch = attach_states(cond, states.cpu(), energies.cpu(),
                              identifier=self.energy_function.smiles,
                              periodic=self.energy_function.periodic_dims)
        batch = batch.to(self.device)
        if hasattr(self, 'identifier_registry'):
            batch.add_graph_attr(
                torch.full((batch.num_graphs,),
                           self.identifier_registry[self.energy_function.smiles],
                           dtype=torch.long, device=batch.device), 'mol_id')

        # THROUGH condition_samples, not around it. It is what attaches `conditions` and
        # `condition_id` to the batch, and the rows admitted here land in prior_buffer
        # where _expire_stale_prior_rows reads batch.condition_id directly. Hand-rolling
        # the returned condition_id would satisfy this function's caller and still leave
        # the admitted ROWS without the attribute.
        batch, log_T_tensor, condition, condition_id =             self.energy_function.condition_samples(batch)
        log_r = self.energy_function.prebuilt_sample_to_reward(batch, 10 ** log_T_tensor)
        metrics = {
            'log_r': log_r.detach(),
            'log_T_tensor': log_T_tensor,
            'condition_id': condition_id,
        }
        return metrics, batch

    # -------------------------------------------------------------- datasets

    def init_mol_dataset(self):
        """One condition: the molecule itself, carrying no state.

        ``ConformerBuffer._compute_xy`` falls back to the reference conformer for a
        conditions-only batch, which is the honest value -- the reference conformer IS the
        zero of this parameterisation.
        """
        from energies.conformer_data import collate_conditions, condition_from_energy

        cond = condition_from_energy(self.energy_function,
                                     identifier=self.energy_function.smiles)
        # collate refuses a single row: a one-graph batch's per-graph tensors have
        # size(0) == 1, which every batch op reads as shared metadata and passes through
        # unindexed. Two copies of one condition is the minimum honest batch.
        self.mol_dataset = ConformerBuffer(collate_conditions([cond, cond.__copy__()]),
                                           device=self.buffer_device,
                                           **self._buffer_kwargs(),
                                           exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)
        self.test_mol_dataset = None

    @property
    def _prior_rng(self):
        """ONE rng for every prior draw in the run, created on first use.

        Persistent rather than re-seeded per call: a fresh default_rng(seed) at each churn
        cycle would redraw the SAME states every time, so the phase-2 prior buffer would
        churn against a fixed 1000-row set while reporting a healthy admit rate.
        """
        if getattr(self, '_prior_rng_state', None) is None:
            self._prior_rng_state = np.random.default_rng(int(self.args.seed))
        return self._prior_rng_state

    def init_prior_dataset(self):
        """Phase 1's dataset: draws from the fitted prior, scored at init.

        NO FILE IS LOADED. The crystal route reads a prepared prior dataset off disk; the
        conformer track has a fitted InternalPrior instead, so the equivalent is to sample
        it. Owner decision 2026-08-20: sample ``prior_sample_size`` rows, score them once
        here, and use that as the phase-1 dataset.

        Scored at init for the same reason the crystal route re-analyses its prior at init:
        ``prebuilt_sample_to_reward`` reads a baked ``conformer_energy`` off the graph and
        REFUSES to recompute, because a silent rescore there would hide a prep bug behind
        plausible numbers.

        The draw uses ``sample_prior_states`` at its defaults, so joint ring sampling is
        ON -- the path that benchmarks 32x-87000x over uniform-on-box.
        """
        from energies.conformer_data import attach_states, bake_energies, condition_from_energy

        if self.internal_prior is None:
            raise SystemExit(
                'energy_config.internal_prior_path is unset and prior_path is null, so '
                'there is nothing to seed phase 1 from. Point one of them at a prior.')

        n = int(getattr(self.args.energy_config, 'prior_sample_size', 50000))
        rng = self._prior_rng
        states, stats = self.energy_function.sample_prior_states(
            self.internal_prior, n, rng, report=True)
        energies = bake_energies(self.energy_function, states)

        cond = condition_from_energy(self.energy_function,
                                     identifier=self.energy_function.smiles)
        batch = attach_states(cond, states.cpu(), energies.cpu(),
                              identifier=self.energy_function.smiles,
                              periodic=self.energy_function.periodic_dims)
        # SAME CONSTRUCTION ARGS AS THE CRYSTAL prior_dataset. y_fn in particular is not
        # optional: log_buffer_stats reads `buff.y` for the energy readout, and during the
        # warm-start stage (bwd_sampling_mode 'dataset') the buffer it reads is THIS one,
        # not prior_buffer. Without it every prior energy metric is silently absent.
        self.prior_dataset = ConformerBuffer(batch,
                                             device=self.buffer_device,
                                             **self._buffer_kwargs(),
                                             x_fn=None,
                                             y_fn=self._buffer_y_fn(),
                                             exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                             )
        e = energies.detach().cpu().numpy()
        print(f'prior dataset: {n} states sampled from the fitted InternalPrior and scored '
              f'at init -- median {np.median(e):.1f}, p10 {np.percentile(e, 10):.1f}, '
              f'p90 {np.percentile(e, 90):.1f} kcal/mol')
        if stats.get('n_closure_bonds'):
            print(f'  ring closure {stats["closure_err"]:.4f} A = '
                  f'{stats["closure_sigma"]:.2f} bond-sigma over {stats["n_rings"]} '
                  f'system(s); {stats["n_ring_banked"]} banked, '
                  f'{stats["n_ring_thermal"]} held')


if __name__ == '__main__':
    # Mirrors train.py's own entrypoint rather than branching it. The crystal main stays a
    # straight line to Modeller(); the only thing that differs here is which class is
    # constructed, and that is not worth a dispatch in the file every crystal run goes
    # through.
    #
    #   python -u conformer_modeller.py --config configs/conformer_mk.yaml
    import torch as _torch

    from utils import get_train_args

    # float32 EVERYWHERE for now (owner decision 2026-08-20). Set before the config is
    # read, because buffer/state tensors are allocated at get_default_dtype() and a later
    # switch leaves a mixed-precision batch that only fails at the first matmul.
    _torch.set_default_dtype(_torch.float32)

    _args = get_train_args()

    # GPU pre-flight BEFORE anything touches CUDA -- same reason as train.py: two runs on
    # one card BSOD'd this machine, the driver does not politely OOM, and there is nothing
    # to catch after the fact. Override with GFN_ALLOW_GPU_SHARING=1.
    from gpu_guard import GPUBusy, require_free_gpu

    try:
        require_free_gpu()
    except GPUBusy as _e:
        raise SystemExit(str(_e))

    modeller = ConformerModeller(args=_args)
    modeller.train()
