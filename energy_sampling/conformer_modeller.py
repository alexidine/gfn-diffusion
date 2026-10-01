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

import time

import numpy as np
import torch

from buffer import ConformerAnchorBuffer, ConformerBuffer
from energies.conformer_torsions import ConformerTorsions
from train import BULKY_ATTR_EXCLUDE_KEYS, Modeller

#: energy_config keys that describe the PROBLEM and are consumed here rather than passed
#: to ConformerTorsions, which takes no **kwargs and would raise on any of them.
_NON_ENERGY_KEYS = ('internal_prior_path', 'prior_sample_size', 'reward_range',
                    'density_coeff', 'reduction_coeff', 'analyze_kwargs',
                    'internal_oom_recovery', 'prior_relax_steps', 'prior_dataset_path')


#: Re-exported so existing imports and tests keep resolving; the definitions live in
#: progress_metrics because nothing in them is conformer-specific.
from progress_metrics import (_PROGRESS_GATE, _column_w1, _column_w1_ratio,  # noqa: F401
                              progress_gate)

#: Angstrom. The largest |stored - rebuilt| reference coordinate a conditions or graph-form
#: prior file may carry against the member this run rebuilds for its identifier
#: (`_refuse_reference_mismatch`). A rebuild under the same RDKit reproduces the reference to
#: float32 storage rounding (~1e-6 A); another RDKit re-embeds 45 of 88 small-rung members,
#: by up to 3.6 A (2025.03.5 against 2025.09.4).
REFERENCE_POS_TOL = 1e-3



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

        ``statics`` once the conditions table exists (`_init_conformer_statics`): every store
        built from then on keeps its rows compact against it (buffer.py::ConformerCompactRows).
        """
        statics = getattr(self, 'conformer_statics', None)
        return {} if statics is None else {'statics': statics}

    def _init_conformer_statics(self):
        """The per-condition table compact stores join their rows to: `mol_dataset`'s batch,
        when the conditions come from `molecules_path` and carry one identifier per graph.
        None otherwise (the single-molecule route), and every store stays a full one."""
        from buffer import ConformerStatics

        self.conformer_statics = None
        batch = getattr(getattr(self, 'mol_dataset', None), 'batch', None)
        idents = None if batch is None else batch._store.get('identifier', None)
        if not getattr(self.args, 'molecules_path', None) or not isinstance(idents, (list, tuple)) \
                or len(idents) != int(batch.num_graphs) or int(batch.num_graphs) < 2:
            return
        self.conformer_statics = ConformerStatics(batch)
        print(f'compact conformer stores: {batch.num_graphs} condition graph(s) in the '
              f'conditions table; rows keep their state, energy and per-row fields only')

    def _derive_condition_fields(self):
        """Put `conditions` and `condition_id`, as `condition_samples` attaches them, on the
        conditions table (`ConformerStatics.derive`), so a conditioned row reads them from
        there rather than storing a copy of its molecule's embedding. Skipped under
        temperature conditioning, where `conditions` carries a per-row temperature draw."""
        statics = getattr(self, 'conformer_statics', None)
        if statics is None or getattr(self.energy_function, 'temperature_conditioning', False):
            return
        probe = statics.batch.subsample_new_batch(torch.arange(statics.batch.num_graphs))
        _, _, condition, condition_id = self.energy_function.condition_samples(probe)
        statics.derive('conditions', condition.detach())
        statics.derive('condition_id', condition_id.detach())

    def _bind_restored_stores(self):
        """Join every store restored from a sidecar (init_gfn) to the conditions table: compact
        rows are bound by identifier, a full-row sidecar is compacted. Runs after
        init_identifiers has put `mol_id` on the table, which full rows are keyed by."""
        statics = getattr(self, 'conformer_statics', None)
        for name in ('prior_buffer', 'replay_buffer', 'anchor_buffer'):
            buf = getattr(self, name, None)
            if buf is None:
                continue
            if statics is None:
                if getattr(buf, 'is_compact', False):
                    raise SystemExit(
                        f'{name} was restored as compact conformer rows, and this run has no '
                        f'conditions table to join them to (molecules_path unset)')
                continue
            print(f'{name}: {buf.bind_statics(statics)} to the conditions table '
                  f'({len(buf)} rows)')

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
                    (2, 'phi (torsion, deg)', 180.0),
                    (3, 'transverse (linear bend u, v)', 1.0))

    def _domain_figs(self, fig_dict, sample_batch, prior_latent_params, anchor_latents):
        """The 12 worst state columns, drawn with the CRYSTAL latent-parameter panel code.

        Same figure the crystal route logs -- overlaid one-sided violins, three to a row,
        transparent background, horizontal legend on top -- via MolCrystalOps' own
        ``_add_violin`` and ``_get_color_set``. Neither touches ``self``, and _add_violin
        already lays out on a THREE-column grid (``// 3``, ``% 3``), so a 4x3 panel of the
        worst 12 reuses it verbatim rather than reimplementing the idiom. What this route
        supplies is only what is genuinely different: which columns, their labels and their
        ranges.

        WHY WORST-12 RATHER THAN ALL. A crystal has 12-ish cell parameters and can show them
        all; a conformer at `full` has 72 columns and on phenyl-THP exactly three of them
        carry the entire ring-flip problem. Ranking by 1-D Wasserstein against the prior puts
        the columns that are actually wrong on the page and drops the ~60 that already match,
        which is what keeps one fixed panel readable at any molecule size.
        """
        from mxtaltools.dataset_utils.data_classes import MolCrystalData
        from plotly.subplots import make_subplots

        from energies.conformer_data import batch_states

        def host(t):
            if t is None:
                return None
            return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)

        samples = host(batch_states(sample_batch))
        reference = host(prior_latent_params)
        if reference is None or reference.shape[1] != samples.shape[1]:
            return
        anchors = host(anchor_latents)
        periodic = np.asarray(host(self.energy_function.periodic_dims)).astype(bool)
        block = host(self.energy_function._free_block)
        cls_name = {0: 'r', 1: 'theta', 2: 'phi', 3: 'transverse'}
        # on a CARRIER `_free_block` is the REGION code, which files a linear bend's u and v
        # under theta; the layout's per-member kinds say what the column holds for whom
        carrier = getattr(self.energy_function, 'carrier', None)
        label = ((lambda j: carrier.column_label(j)) if carrier is not None
                 else (lambda j: cls_name.get(int(block[j]), '?')))

        w1 = _column_w1(samples, reference, periodic)
        order = np.argsort(-w1)[:12]
        self._last_w1 = w1

        # the distributions to overlay, in the crystal panel's own (name, data) form
        dists = [('sampler', samples), ('prior', reference)]
        if anchors is not None and anchors.ndim == 2 and anchors.shape[1] == samples.shape[1]:
            dists.append(('anchors', anchors))
        colors = MolCrystalData._get_color_set(None, len(dists))

        titles = [f'col {int(j)} · {label(j)}'
                  f'{" (wraps)" if periodic[j] else ""}  |  W1 {w1[j]:.4f}' for j in order]
        fig = make_subplots(rows=4, cols=3, subplot_titles=titles)
        for i, j in enumerate(order):
            lo = min(float(d[:, j].min()) for _, d in dists)
            hi = max(float(d[:, j].max()) for _, d in dists)
            pad = 0.05 * max(hi - lo, 1e-6)
            rng = (lo - pad, hi + pad)
            for k, (name, data) in enumerate(dists):
                MolCrystalData._add_violin(None, fig, data[:, j], name, colors[k], i, rng,
                                           200, 0.05)
            fig.update_xaxes(range=list(rng), row=i // 3 + 1, col=i % 3 + 1)

        # styling copied from plot_batch_cell_params so the two panels read as one family
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          violinmode='overlay',
                          legend=dict(orientation='h', yanchor='bottom', y=1.05,
                                      xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
                          margin=dict(l=40, r=20, t=50, b=50),
                          font=dict(family='Helvetica', size=12, color='black'))
        fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside',
                         tickwidth=1, mirror=True)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, ticks='',
                         mirror=True)
        if len(dists) > 1:
            fig.update_traces(opacity=0.5)
        fig_dict['Latent Params (worst 12)'] = fig


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
        if getattr(en, 'is_carrier', False):
            # per ROW, from each row's own member -- the case this block was wired for
            from energies.ring_metrics import ring_cycles
            idents = en._row_identifiers(sample_batch, n)
            size = {i: float(np.asarray(m.spec.z).shape[0]) for i, m in en._members.items()}
            rings = {}
            for i, m in en._members.items():
                try:
                    rings[i] = float(len(ring_cycles(m)))
                except Exception:
                    rings[i] = 0.0
            return {'size': np.array([size[i] for i in idents]),
                    'n_rings': np.array([rings[i] for i in idents])}
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
        if getattr(en, 'is_carrier', False):
            # EVERY call below reads ONE chart (spec, _M, the force field, the basin table)
            # against the whole state block. On a carrier there is no such chart; the
            # per-member versions are not written yet, so the block is ABSENT rather than
            # computed against the wrong molecule.
            if not getattr(self, '_carrier_eval_notice', False):
                self._carrier_eval_notice = True
                print('eval: CARRIER set -- per-molecule physical stats (energy components, '
                      'geometry, dof classes, rings, basin coverage) are not computed yet')
            return
        prior_x = getattr(getattr(self, 'prior_dataset', None), 'x', None)
        prior_y = getattr(getattr(self, 'prior_dataset', None), 'y', None)

        metrics.update(cm.energy_component_stats(en, state))
        metrics.update(cm.geometry_stats(en, state))
        metrics.update(cm.dof_class_stats(en, state, reference=prior_x))
        # SAME reference as dof_class_stats, so the per-element ratios are the drill-down
        # for that group's sd_ratio_max rather than a second, unreferenced set of spreads.
        metrics.update(cm.dof_element_stats(en, state, reference=prior_x))
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
        u_star = (float(getattr(self.args, 'nonthermal_entropy_per_dim', 4.0) or 0.0)
                  * int(en.ndim))
        metrics.update(cm.basin_nonthermal(
            en, state, e, self._e_min, self._basin_ref, u_star))
        # THE DENOMINATORS FOR THE THREE LIVE cover/ KEYS, once. n_missed = 0 is
        # meaningless without n_accessible, worst_frac is only readable against the uniform
        # expectation, and coupling_tc_debiased only against the target's own coupling --
        # but none of them move within a run, so they are a startup line rather than three
        # more flat traces on the dashboard.
        if not getattr(self, '_cover_constants_logged', False):
            self._cover_constants_logged = True
            const = cm.cover_constants(en, self._basin_ref, u_star=u_star,
                                       target_tc=self._target_coupling)
            print('cover/ constants (fixed for this run, not logged): '
                  + ', '.join(f'{k.split("/")[-1]}={v:.4g}' if isinstance(v, float)
                              else f'{k.split("/")[-1]}={v}' for k, v in const.items()))
        # Marginal fit, energy-marginal overlap and the exit verdict are emitted by
        # Modeller.progress_metrics (train.py), which is ROUTE-AGNOSTIC -- it reads
        # everything through _batch_latents, so the crystal route gets the identical
        # measurement. Duplicating it here would double-log and let the two drift.

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
        # FROM THE TOP LEVEL, like temperature_conditioning and for the same reason: these
        # live beside it in the config because `train.get_conditioning_dim` reads them there
        # to size the conditioner. Forwarding them here is what keeps the two halves of the
        # contract agreeing -- without it the conditioner is built 256 wide while the energy
        # still emits a 1-wide zeros column, and the mismatch surfaces as a bare
        # `mat1 and mat2 shapes cannot be multiplied (8x1 and 256x512)` inside the MLP.
        cfg['embedding_conditioning'] = bool(getattr(self.args, 'embedding_conditioning', False))
        cfg['embedding_conditioning_dim'] = getattr(
            self.args, 'embedding_conditioning_dim', None)

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
        # A MOLECULE SET GETS A SET ENERGY. `ConformerTorsions` is ONE chart -- `energy`
        # reads its own `_M`, `r0` and Jacobian constants -- so a batch drawn from a
        # multi-molecule condition file was scored entirely against `energy_config.smiles`.
        # The shapes agree whenever k agrees, so it returned numbers rather than raising:
        # measured 3.33 nats of per-row error across three ordinary QM9 molecules, which is
        # a per-condition log Z error and therefore aimed straight at what the conditional
        # route is trying to learn.
        #
        # The molecule list is read from the CONDITIONS FILE rather than from a new config
        # key, so it cannot disagree with the set actually being trained on.
        set_smiles, set_idents = self._condition_set_molecules()
        print(f'condition set: {0 if not set_idents else len(set(set_idents))} distinct '
              f'molecule(s) read from molecules_path')
        # EVERY MEMBER FROM THE REFERENCE THE FILE STORES, not from a fresh embedding. A
        # stored state is a displacement from its member's reference, and the same tagged
        # SMILES embeds to another reference under another RDKit build; the file's `pos` is
        # the reference its states were written against. No ETKDG embedding or MMFF
        # relaxation runs for such a member.
        if 'reference_positions' in cfg:
            raise SystemExit('energy_config.reference_positions: not a config key. Each member '
                             'takes its reference conformer from the conditions file')
        refs = self._stored_references(set_smiles, set_idents, cfg.get('level'))
        t0 = time.perf_counter()
        if set_smiles and len(set(set_idents)) > 1:
            from energies.multi_conformer import MultiConformerTorsions
            cfg.pop('smiles', None)
            self.energy_function = MultiConformerTorsions(
                set_smiles, identifiers=set_idents,
                reference_positions={i: rd for i, (rd, _, _) in refs.items()}, **cfg)
        else:
            # one chart: the file's reference only when the file's one member IS the
            # configured molecule
            one = (refs.get(set_idents[0]) if set_idents and set_smiles[0] == cfg.get('smiles')
                   else None)
            refs = {set_idents[0]: one} if one is not None else {}
            self.energy_function = ConformerTorsions(
                reference_positions=None if one is None else one[0], **cfg)
        self._check_stored_references(refs, time.perf_counter() - t0)
        print(self.energy_function.describe())
        # THE BASE METHOD ALSO BUILDS THE TRACE WINDOW, and this override does not call
        # super(). Dropping it left profiling.trace silently INERT on the whole conformer
        # route -- the config key read as enabled and nothing was ever written, which is
        # the quiet kind of failure. Rebuilt here rather than by calling super(), because
        # super() would construct a MolecularCrystal first.
        import profiling
        self._trace_window = profiling.trace_from_config(
            self.args, cuda=str(self.device).startswith('cuda'),
            tag=str(getattr(self.args, 'run_name', 'run')))
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
        # SINGLE-CONDITION RUNS KEEP THE SCALAR FLOW HEAD. A conditional model normally
        # gets a scalarMLP over the condition embedding, but log Z is then a function
        # rather than a number -- and `z_level_fill`, the only thing pinning it once the
        # servo is off and fwd carries no loss weight, writes `flow_model.scalar.data`.
        # With one condition there is exactly one log Z, so the scalar is not a
        # simplification, it is the correct object. DERIVED from the conditions file,
        # never configured: a flag could disagree with the data it describes.
        _, _idents = self._condition_set_molecules()
        n_cond = len(set(_idents)) if _idents else 0
        cfg['scalar_flow'] = bool(cfg.get('conditional')) and n_cond == 1
        if cfg['scalar_flow']:
            print('flow head: LearnableScalar despite conditional=True -- the condition set '
                  'has exactly one member, so log Z is a single number')
        # POPPED, not passed. The base builder splats **vars(args.model) straight into GFN,
        # whose signature is explicit -- an unrecognised key is a TypeError at construction.
        self._policy_spec = {k: cfg.pop(k) for k in self._SET_POLICY_KEYS if k in cfg}
        return cfg

    #: `model:` keys selecting and sizing the SET policy over per-coordinate tokens. Kept
    #: out of the GFN constructor's arguments on purpose (see above). What they BUILT is
    #: stamped into gfn_config['conformer'] instead (_install_set_policy), which is what
    #: lets the checkpointer rebuild the head (gfn_from_config).
    #: NAMES ARE PREFIXED DELIBERATELY. `policy_layers` and `policy_hidden_dim` already
    #: exist in the model block as the FLAT policy's own GFN arguments; popping either
    #: would silently unbuild the flat path.
    _SET_POLICY_KEYS = ('policy_kind', 'set_policy_hidden', 'set_policy_layers',
                    'set_policy_corr_dim')

    #: the set head's sizes when the config omits them. ONE definition, read by the fresh
    #: build and by the reload check alike, so the two cannot disagree about a default and
    #: refuse a checkpoint the config never changed.
    _SET_POLICY_DEFAULTS = {'set_policy_hidden': 64, 'set_policy_layers': 4,
                            'set_policy_corr_dim': 32}

    def _condition_set_molecules(self):
        """`(smiles, identifiers)` of the condition set, or `(None, None)`.

        Read straight off `molecules_path` because that file IS the set the run trains over;
        a separate config list could disagree with it, and the failure would be a chart
        mismatch discovered rows later.

        READ ONCE PER PATH: the lists, and each identifier's stored reference (`pos` and `z`
        of its first graph, placement order), are kept on `_condition_set_cache`; this is
        called more than once at startup.
        """
        path = getattr(self.args, 'molecules_path', None)
        if not path:
            return None, None
        cached = getattr(self, '_condition_set_cache', None)
        if cached is not None and cached['path'] == str(path):
            return list(cached['smiles']), list(cached['identifiers'])
        try:
            blob = torch.load(path, weights_only=False, map_location='cpu')
        except Exception as exc:                              # noqa: BLE001 - reported
            print(f'condition set: could not read {path} ({type(exc).__name__}); '
                  f'falling back to the single energy_config.smiles')
            return None, None
        batch = blob['prior'] if isinstance(blob, dict) and 'prior' in blob else blob
        smis = getattr(batch, 'smiles', None)
        idents = getattr(batch, 'identifier', None)
        if smis is None or idents is None:
            return None, None
        smis, idents = list(smis), list(idents)
        ptr = getattr(batch, 'ptr', None)
        pos, z = getattr(batch, 'pos', None), getattr(batch, 'z', None)
        stored = ptr is not None and pos is not None and z is not None
        seen, out_s, out_i, refs = set(), [], [], {}
        for j, (smi, ident) in enumerate(zip(smis, idents)):
            if ident in seen:
                continue
            seen.add(ident)
            out_s.append(smi)
            out_i.append(ident)
            if stored:
                at = slice(int(ptr[j]), int(ptr[j + 1]))
                refs[ident] = (pos[at].detach().cpu().double().numpy().copy(),
                               z[at].detach().cpu().numpy().copy())
        self._condition_set_cache = {'path': str(path), 'smiles': out_s,
                                     'identifiers': out_i, 'references': refs}
        return list(out_s), list(out_i)

    def _stored_references(self, smiles, identifiers, level):
        """`{identifier: (RDKit-order positions, perm, placement-order pos)}` for every
        condition-set member whose reference the conditions file stores (`pos`, `z`).

        `energies/conformer_torsions.py::rdkit_order_reference` derives each member's placement
        order from its bond graph; `_check_stored_references` then confirms each built member
        against the stored `pos`. A file with no stored reference gives `{}`, and every member
        is embedded.
        """
        from energies.conformer_torsions import rdkit_order_reference

        cache = getattr(self, '_condition_set_cache', None) or {}
        stored = cache.get('references') or {}
        if not identifiers or not stored:
            if identifiers:
                print('condition set: the conditions file stores no reference geometry; every '
                      'member is EMBEDDED (seeded ETKDG + MMFF), which another RDKit build can '
                      'resolve to another reference')
            return {}
        out, bad = {}, []
        for smi, ident in zip(smiles, identifiers):
            if ident not in stored or ident in out:
                continue
            pos, z = stored[ident]
            try:
                rd, perm = rdkit_order_reference(smi, pos, level=level, z=z)
            except ValueError as exc:
                bad.append(f'{ident}: {exc}')
                continue
            out[ident] = (rd, perm, pos)
        if bad:
            raise SystemExit(f'{len(bad)} condition(s) of molecules_path store a reference that '
                             f'is not their molecule in any atom order, e.g.:\n  '
                             + '\n  '.join(bad[:10]))
        return out

    def _check_stored_references(self, refs, seconds: float):
        """Refuse a member built from a stored reference that is not that stored member.

        Each must reproduce the stored placement order and the stored `pos` exactly
        (`stored_reference_mismatch`); a derived placement order that differed would put every
        stored state on other atoms.
        """
        from energies.conformer_torsions import stored_reference_mismatch

        en = self.energy_function
        members = getattr(en, '_members', None) or {}
        if not members and refs:
            members = {next(iter(refs)): en}
        bad = []
        for ident, (_, perm, pos) in refs.items():
            why = stored_reference_mismatch(members[ident], pos, perm)
            if why:
                bad.append(f'{ident}: {why}')
        if bad:
            raise SystemExit(f'{len(bad)} member(s) built from the stored reference of '
                             f'molecules_path do not reproduce it:\n  ' + '\n  '.join(bad[:10]))
        n = max(len(members), 1)
        print(f'condition set: {len(refs)} of {n} member(s) built from the stored reference '
              f'(no embedding), {n - len(refs)} embedded; members built in {seconds:.1f} s')

    def init_identifiers(self):
        """Base registry, checked against the stamped condition set, then handed to the energy.

        Without the hand-over a buffered row -- which carries `mol_id` but not `identifier` --
        cannot be matched to its chart, and the energy would have to guess. The check
        (`_assert_registry_is_the_stamp`) is what makes the stamp's identifier list mol_id
        order rather than an assumption about it.
        """
        super().init_identifiers()
        self._assert_registry_is_the_stamp()
        binder = getattr(self.energy_function, 'bind_identifier_registry', None)
        if binder is not None:
            binder(self.identifier_registry)
        self._derive_condition_fields()
        self._bind_restored_stores()

    def init_gfn(self):
        """Base build or checkpoint load, then install -- or, on a reload, VERIFY -- the policy.

        THE RELOAD FLAG IS RECORDED ON THE PATH, not inferred from the stored block. It is
        cleared here and set by `gfn_from_config`, which only the checkpointer's two load
        paths call. Testing for gfn_config['conformer'] instead would read a flat checkpoint
        written before the block existed as a FRESH build: a set config would then swap a new
        head over the loaded trunk and reset the EMA and the Adam state load_full had just
        restored -- worse than refusing.
        """
        self._refuse_inert_prior_model()
        self._refuse_compiled_set_policy()
        self._gfn_reloaded = False
        super().init_gfn()
        self._install_set_policy()

    def _refuse_inert_prior_model(self):
        """`prior_model_name` is read by nothing on this route; refuse it rather than ignore it.

        Modeller.init_prior_dataset is the only reader, and this class overrides it without
        calling super(). So the frozen prior never loads, `skip_if: prior_loaded` can never
        hold, and a crystal-style phase-2 arm silently re-runs phase 1 from its loaded weights.
        Refused here, before any checkpoint is read.
        """
        name = getattr(self.args, 'prior_model_name', None)
        if name is not None:
            raise ValueError(
                f"prior_model_name={name!r} on the conformer route. Only "
                f"Modeller.init_prior_dataset reads it, and ConformerModeller overrides that "
                f"without calling super(), so no prior model would load and "
                f"`skip_if: prior_loaded` could never hold -- the arm would re-run phase 1. "
                f"Seed phase 2 with checkpoint_name + load_weights_only and "
                f"`skip_if: weights_loaded`; this route's prior is "
                f"energy_config.internal_prior_path.")

    def _config_policy_spec(self):
        """The set-policy keys as THIS config states them, on every path.

        `_policy_spec` is written by `_build_gfn_config`, which only a fresh build runs; a
        reload never calls it, so reading it there sees {} and calls every reload flat.
        """
        model = getattr(getattr(self, 'args', None), 'model', None)
        model = vars(model) if model is not None else {}
        return {k: model[k] for k in self._SET_POLICY_KEYS if k in model}

    def _refuse_compiled_set_policy(self):
        """A compiled set head would run on a resumed leg and not on a fresh one.

        maybe_compile_policy runs INSIDE the base init_gfn. On a fresh build that is before
        the swap, so it compiles the flat head _install_set_policy throws away and the set
        head runs eager. On a reload the checkpointer has already built the set head, so it
        IS compiled. The two legs of one run would then execute different code, and the
        ragged head's shapes change with every batch's molecule mix.
        """
        kind = str(self._config_policy_spec().get('policy_kind', 'flat')).lower()
        setting = getattr(self.args, 'compile_policy', False)
        # MIRRORS train.py maybe_compile_policy, which compiles on 'auto'/'step' (platform-
        # dependent) and otherwise on bool(setting) -- so the STRINGS 'off' and 'false' compile
        # there, and letting them through here would be exactly the split this refuses.
        compiles = setting in ('auto', 'step') or bool(setting)
        if kind == 'set' and compiles:
            raise ValueError(
                f"compile_policy={setting!r} with model.policy_kind: set. The base init_gfn "
                f"compiles before the set head is swapped in on a fresh build and after it "
                f"is rebuilt on a reload, so a fresh and a resumed leg would run different "
                f"code. Set compile_policy: false until a cluster measurement shows "
                f"compiling the ragged set head pays.")

    def _needs_ragged_policy(self) -> bool:
        """The set head must be the RAGGED one: a carrier state, or more than one molecule.

        The identity-layout case is the one that bites. A multi-molecule set whose members
        share their block counts is not a carrier (`is_carrier` False, K = k), and the dense
        `conditional_set_policy_for(self.energy_function, ...)` then bakes the REFERENCE
        member's static features into every row -- finite, plausible and wrong for every
        other molecule. The ragged head reads `dof_static` and `state_mask` off each row,
        and `build_conformer_conditions.py --carrier` writes both even for an identity
        layout. A set of ONE molecule repeated under several identifiers keeps the dense
        head: its reference features are every row's features.
        """
        en = getattr(self, 'energy_function', None)
        return (bool(getattr(en, 'is_carrier', False))
                or int(getattr(en, 'distinct_smiles', 1) or 1) > 1)

    def _state_block_width(self):
        """``[r, theta, phi]`` column counts of this run's state -- the carrier layout.

        Read off `_free_block`, which on a carrier IS the layout's (CarrierLayout.free_block),
        so this equals `carrier.block_width` there and gives the same numbers for an identity
        layout, which keeps no CarrierLayout.
        """
        fb = np.asarray(self.energy_function._free_block).reshape(-1)
        return [int((fb == b).sum()) for b in (0, 1, 2)]

    def _ragged_policy_stamp(self, spec, mol_dim):
        """Every argument the ragged set head is built from, as stored in the checkpoint.

        Written EXPLICITLY, defaults included (frame_size, norm, dropout), so the head a
        checkpoint rebuilds does not depend on RaggedConditionalSetPolicy's defaults staying
        what they were when it was trained.
        """
        from energies.dof_features import MAX_FRAME, state_feature_names
        if mol_dim % 2:
            raise ValueError(f'mol_dim {mol_dim} is odd; the pooled readout is two '
                             f'equal blocks')
        sizes = {k: int(spec.get(k, d)) for k, d in self._SET_POLICY_DEFAULTS.items()}
        return {'policy_kind': 'set', **sizes,
                'n_static': len(state_feature_names()),
                'enc_dim': mol_dim // 2, 'mol_dim': int(mol_dim),
                'frame_size': int(MAX_FRAME), 'norm': None, 'dropout': 0,
                'carrier': True, 'block_width': self._state_block_width()}

    @staticmethod
    def _ragged_policy_from_stamp(stamp, angular_mask, t_dim, zero_init: bool = False):
        # zero_init is an INITIALISATION, not architecture, so it is not in the stamp: the
        # fresh build passes model.zero_init (_install_set_policy); a reload leaves it False,
        # because the checkpoint's weights overwrite the init
        from models.ragged_set_policy import RaggedConditionalSetPolicy
        return RaggedConditionalSetPolicy(
            int(stamp['n_static']), angular_mask, int(t_dim),
            enc_dim=int(stamp['enc_dim']), mol_dim=int(stamp['mol_dim']),
            corr_dim=int(stamp['set_policy_corr_dim']), frame_size=int(stamp['frame_size']),
            hidden_dim=int(stamp['set_policy_hidden']),
            layers=int(stamp['set_policy_layers']), out_per_token=2,
            dropout=stamp['dropout'], norm=stamp['norm'], zero_init=bool(zero_init))

    def _set_policy_plan(self, spec):
        """`(head, stamp, mol_dim)` for a `policy_kind: set` config against THIS energy.

        `head` is 'ragged', 'conditional' (dense, bound to the energy's chart) or 'dense'.
        Shared by the fresh build and the reload check, so a reload compares against exactly
        what a fresh build of this config would construct.
        """
        conditional = bool(getattr(self.args, 'embedding_conditioning', False))
        ragged = self._needs_ragged_policy()
        if ragged and not conditional:
            raise NotImplementedError(
                'policy_kind: set on a CARRIER state or a multi-molecule set needs '
                'embedding_conditioning: the per-column features come from the molecule, and '
                'an unconditional set head bakes one molecule\'s features in at construction')
        mol_dim = None
        if conditional:
            mol_dim = int(getattr(self.args, 'embedding_conditioning_dim', 0) or 0)
            if not mol_dim:
                raise ValueError(
                    'embedding_conditioning is on but embedding_conditioning_dim is unset; '
                    'the set policy needs the width to build its context input')
        if ragged:
            return 'ragged', self._ragged_policy_stamp(spec, mol_dim), mol_dim
        sizes = {k: int(spec.get(k, d)) for k, d in self._SET_POLICY_DEFAULTS.items()}
        # the dense heads bind the energy's chart at construction, so there is nothing to
        # rebuild them from; the stamp names them so a reload is refused BY NAME
        stamp = {'policy_kind': 'set', **sizes, 'mol_dim': mol_dim, 'carrier': False}
        return ('conditional' if conditional else 'dense'), stamp, mol_dim

    def gfn_from_config(self, cfg):
        """The checkpointer's builder seam (Checkpointer._build_gfn), called by both loads.

        Builds the model a STORED gfn_config describes -- class, set head and `_carrier` --
        before any weights load, so the strict load_state_dict meets the architecture that
        was saved. load_full then deep-copies the EMA from it (class and head included),
        restores the P_B snapshot through set_pb_freeze, and restores the optimizers over the
        same parameter groups: reassigning `forward_policy` keeps its slot in `_modules`, so
        parameter order matches the fresh path's.

        `cfg` is NOT mutated. m.gfn_config keeps the 'conformer' block, so the next save
        writes it again and the leg after that can load too. The head is built from the
        STORED arguments, never from this config: the architecture follows the file, and
        _install_set_policy then checks the config against it field by field.

        THE EARLIEST POINT A CHECKPOINT MEETS THIS RUN. load_full calls this before it restores
        the modeller state, the buffer sidecar or condition_log_z, and init_energy_function
        has already built this run's members off molecules_path -- so the layout and the
        condition set are both judged here, ahead of every restore.
        """
        from models.conformer_gfn import ConformerGFN
        from models.gfn import GFN

        cfg = dict(cfg)
        stamp = cfg.pop('conformer', None)
        self._assert_checkpoint_layout(cfg, stamp)
        self._assert_condition_set(stamp)
        self._gfn_reloaded = True
        kind = 'flat' if stamp is None else str(stamp.get('policy_kind'))
        carrier = bool(stamp.get('carrier', False)) if stamp is not None else False
        if kind == 'flat':
            if not carrier:
                return GFN(**cfg)
            model = ConformerGFN(**cfg)
            model._carrier = True
            return model
        if kind == 'set' and carrier:
            model = ConformerGFN(**cfg)
            model.forward_policy = self._ragged_policy_from_stamp(
                stamp, cfg['angular_mask'], cfg['t_dim'])
            model._carrier = True
            return model
        raise NotImplementedError(
            f"this checkpoint stores model policy_kind {kind!r} with carrier={carrier}. Only a "
            f"flat policy or the RAGGED carrier set head can be rebuilt from a checkpoint: "
            f"the dense set heads bind one molecule's chart at construction, so there is "
            f"nothing stored to rebuild them from. Retrain on the ragged head.")

    def _assert_checkpoint_layout(self, cfg, stamp):
        """Refuse a checkpoint whose policy WIDTH or carrier LAYOUT is not this run's.

        Both P_F and P_B are built at the checkpoint's `dim` (models/gfn.py init_policies),
        and the ragged head raises on a K mismatch only at its first forward. So a rung
        trained at K = 21 handed to a K = 75 run would fail its strict load at best, and at
        worst build a 21-wide model against a 75-wide energy. Equal K is not enough either:
        the r | theta | phi split decides which columns wrap and which member column lands
        where, and two splits can share K.
        """
        en = self.energy_function
        want_dim, got_dim = int(en.data_ndim), int(cfg['dim'])
        if got_dim != want_dim:
            raise ValueError(
                f"checkpoint policy width {got_dim} does not match this run's state width "
                f"{want_dim}. Both policies are built at the checkpoint's width, so this "
                f"load would fail its strict state_dict or build a {got_dim}-wide model "
                f"against a {want_dim}-wide energy. Pin the carrier layout across the runs "
                f"that share weights, or start this run fresh.")
        stored_mask = cfg.get('angular_mask')
        if stored_mask is not None:
            got = [bool(b) for b in stored_mask]
            want = [bool(b) for b in en.periodic_dims]
            if got != want:
                raise ValueError(
                    f"checkpoint periodic columns {[i for i, b in enumerate(got) if b]} do "
                    f"not match this run's {[i for i, b in enumerate(want) if b]} (width "
                    f"{want_dim} both). The policy would wrap the wrong columns.")
        if stamp is not None and stamp.get('block_width') is not None:
            want_bw = self._state_block_width()
            got_bw = [int(w) for w in stamp['block_width']]
            if got_bw != want_bw:
                raise ValueError(
                    f"checkpoint carrier layout r|theta|phi {got_bw} (K = {sum(got_bw)}) "
                    f"does not match this run's {want_bw} (K = {sum(want_bw)}). Member "
                    f"columns would land in the wrong blocks.")
        elif getattr(en, 'is_carrier', False):
            print(f"WARNING: checkpoint carries no carrier-layout stamp (written before it "
                  f"existed). Width {want_dim} and the periodic columns match; the "
                  f"r|theta split cannot be verified from the file.")

    # ------------------------------------------------------ the condition set in the stamp

    #: how `_member_signature` digests a member. Stored beside the digests, so a checkpoint
    #: signed by another recipe has its identifiers compared and its members left unjudged,
    #: rather than every member read as changed.
    _MEMBER_SIGNATURE = 'blake2b-64(smiles|block codes|placement z)'

    @staticmethod
    def _member_signature(smiles, member) -> str:
        """16 hex characters for one member, from three fields the built member already holds.

        Nothing is parsed, embedded or read from disk. The SMILES it was built from (the
        conditions file's string, stereo marks included) names the molecule where the
        identifier does not. Its per-column block codes in state order (`_free_block`, a
        transverse u or v as 3) fix, with the stamped `block_width`, the carrier columns it
        owns. Its placement-order atomic numbers (`spec.z`) are the atoms `_resolve_rows`
        compares against every stored row. Another molecule under the same identifier, or the
        same SMILES built into another chart, changes the digest.
        """
        import hashlib
        codes = ''.join(str(int(c)) for c in np.asarray(member._free_block).reshape(-1))
        z = ','.join(str(int(a)) for a in np.asarray(member.spec.z).reshape(-1))
        return hashlib.blake2b(f'{smiles}|{codes}|{z}'.encode(), digest_size=8).hexdigest()

    def _condition_set_identity(self):
        """This run's condition set as `gfn_config['conformer']['condition_set']` stores it.

        None on a single chart, which has no member set. `identifiers` are in mol_id order:
        init_identifiers numbers the SORTED identifiers of the conditions and prior files, and
        the prior's are the members' (`_assert_registry_is_the_stamp` checks that once the
        registry exists), so identifier i is mol_id i -- and, with one space group and one Z',
        condition_id i. Built from the energy's members because they exist before a checkpoint
        loads, and the registry does not.
        """
        en = getattr(self, 'energy_function', None)
        members = getattr(en, '_members', None)
        if not members:
            return None
        smiles = getattr(en, '_member_smiles', None) or {}
        idents = sorted(members)
        return {'identifiers': idents,
                'signatures': [self._member_signature(smiles.get(i, members[i].smiles),
                                                      members[i]) for i in idents],
                'signature_recipe': self._MEMBER_SIGNATURE}

    def _with_condition_set(self, block):
        """`block` plus this run's `condition_set`, when the energy holds a member set."""
        ident = self._condition_set_identity()
        return dict(block) if ident is None else {**block, 'condition_set': ident}

    def _loading_weights_only(self) -> bool:
        """Whether the load in progress is `Checkpointer.load_weights_only` rather than `load_full`.

        gfn_from_config runs inside both, before either records anything that says which
        (load_full's `resume_step`, load_weights_only's `weights_only_loaded`), so this reads the
        branch train.py's Modeller.init_gfn takes: `checkpoint_name` with `load_weights_only` is
        the weights-only load; `checkpoint_name` alone, and `continue_from_checkpoint` whatever
        `load_weights_only` says, are full loads.
        """
        args = getattr(self, 'args', None)
        return (getattr(args, 'checkpoint_name', None) is not None
                and bool(getattr(args, 'load_weights_only', False)))

    @staticmethod
    def _name_some(items, quote: bool = True, limit: int = 10) -> str:
        shown = [repr(i) if quote else str(i) for i in list(items)[:limit]]
        more = f', +{len(items) - limit} more' if len(items) > limit else ''
        return '[' + ', '.join(shown) + more + ']'

    def _assert_condition_set(self, stamp):
        """Refuse a FULL resume onto another condition set; on a weights-only load, say so.

        mol_id is an identifier's position in the sorted condition set (init_identifiers), and
        a full resume restores state keyed by it: condition_log_z's rows and the `mol_id` on
        every stored buffer row. Against another set those rows are read as other molecules.
        The tracker's size check (init_condition_log_z) sees an added or removed member only
        after the sidecar is restored, and a same-size substitution not at all: that one
        surfaced at the first backward draw, in MultiConformerTorsions._resolve_rows. So every
        difference is refused here, ahead of every restore -- an identifier added or removed,
        a mol_id moved, a member whose signature changed under the same identifier.

        A weights-only load restores none of that state, and no weight is indexed by mol_id
        (the heads read each molecule off its condition graph), so another set is legitimate
        there and is named on one line. A checkpoint written before the stamp carried the set
        is warned about on a full resume, not refused. A single chart has no member set, and
        nothing is compared.
        """
        mine = self._condition_set_identity()
        if mine is None:
            return
        weights_only = self._loading_weights_only()
        stored = stamp.get('condition_set') if isinstance(stamp, dict) else None
        if not isinstance(stored, dict) or stored.get('identifiers') is None:
            if not weights_only:
                print(f"WARNING: checkpoint carries no condition-set stamp (written before it "
                      f"existed). This run's {len(mine['identifiers'])} identifiers cannot be "
                      f"checked against the set it was trained on, so a different set of the "
                      f"same size is not refused here; this leg's checkpoints carry the stamp.")
            return
        old, new = list(stored['identifiers']), list(mine['identifiers'])
        old_pos = {i: k for k, i in enumerate(old)}
        new_pos = {i: k for k, i in enumerate(new)}
        removed = [i for i in old if i not in new_pos]
        added = [i for i in new if i not in old_pos]
        moved = [f'{i!r} {old_pos[i]} -> {new_pos[i]}' for i in new
                 if i in old_pos and old_pos[i] != new_pos[i]]
        comparable = (stored.get('signature_recipe') == mine['signature_recipe']
                      and stored.get('signatures') is not None)
        changed = []
        if comparable:
            old_sig = dict(zip(old, stored['signatures']))
            changed = [i for i, s in zip(new, mine['signatures'])
                       if i in old_sig and old_sig[i] != s]
        diffs = [f'{label} {self._name_some(items, quote)}' for label, items, quote in (
            ('only in the checkpoint', removed, True),
            ('only in this run', added, True),
            ('same identifier, another member (SMILES, block codes or placement-order atoms)',
             changed, True),
            ('mol_id moved', moved, False)) if items]
        if not diffs:
            print(f"condition set: {len(new)} identifiers in mol_id order"
                  + (" and their member signatures" if comparable else
                     f" (member signatures by another recipe, "
                     f"{stored.get('signature_recipe')!r}, not compared)")
                  + " match the checkpoint's")
            return
        what = '; '.join(diffs)
        if weights_only:
            print(f"condition set: weights-only load onto another set ({len(old)} -> {len(new)} "
                  f"identifiers; {what}) -- allowed: no weight is indexed by mol_id, the "
                  f"tracker and buffers start on this set, and this run's checkpoints carry it")
            return
        raise ValueError(
            f"condition set: this full resume's conditions ({len(new)} identifiers) are not the "
            f"set the checkpoint was trained on ({len(old)}) -- {what}. mol_id is an "
            f"identifier's position in the sorted set, and a full resume restores "
            f"condition_log_z's rows and the stored buffer rows keyed by it, so they would be "
            f"read as other molecules. Start this run fresh (checkpoint_name: null, "
            f"continue_from_checkpoint: false), or take the weights alone (checkpoint_name: "
            f"this checkpoint, load_weights_only: true): no weight is indexed by mol_id, and "
            f"the tracker and buffers then start on this set.")

    def _restamp_condition_set(self):
        """After a load, the block this run's saves write names THIS run's condition set.

        A full resume has passed `_assert_condition_set`, so this rewrites the set that was
        read. A weights-only load onto another set replaces the checkpoint's, to which this
        run's tracker and buffers are not keyed. A block written before the stamp gains one,
        and a pre-stamp flat checkpoint on a carrier, which carries no block, gets the block a
        fresh build of this run writes. A flat policy off a carrier writes none, as fresh.
        """
        ident = self._condition_set_identity()
        if ident is None:
            return
        block = self.gfn_config.get('conformer')
        if block is None:
            if not getattr(self.energy_function, 'is_carrier', False):
                return
            block = {'policy_kind': 'flat', 'carrier': True,
                     'block_width': self._state_block_width()}
        self.gfn_config['conformer'] = {**block, 'condition_set': ident}

    def _assert_registry_is_the_stamp(self):
        """The stamped identifier list must BE the mol_id registry init_identifiers built.

        The stamp is computed from the energy's members, sorted, before any dataset loads; the
        registry is the sorted union of every loaded dataset's identifiers. They differ only
        when the prior file names an identifier the conditions file does not -- a row with no
        member, which `_resolve_rows` refuses at its first draw -- or if registration stops
        sorting. Either way the stamp would not be mol_id order, so it is refused at init.
        """
        block = (getattr(self, 'gfn_config', None) or {}).get('conformer') or {}
        stamped = (block.get('condition_set') or {}).get('identifiers')
        if stamped is None:
            return
        reg = self.identifier_registry
        registered = sorted(reg, key=reg.get)
        if registered == list(stamped):
            return
        only_reg = [i for i in registered if i not in set(stamped)]
        only_stamp = [i for i in stamped if i not in reg]
        raise ValueError(
            f"the mol_id registry ({len(registered)} identifiers, over the conditions and prior "
            f"files) is not the stamped condition set ({len(stamped)}, this run's energy "
            f"members, sorted): only in the registry {self._name_some(only_reg)}, only among "
            f"the members {self._name_some(only_stamp)}"
            + ("" if only_reg or only_stamp else
               "; the same identifiers in another order, so registration no longer sorts and "
               "_condition_set_identity must follow it")
            + ". A prior row whose identifier has no member cannot be scored; build the prior "
              "and the conditions from one molecule list.")

    def _install_set_policy(self):
        """`model.policy_kind: set` -> swap the flat scalarMLP for a per-coordinate set head.

        WHY A POST-CONSTRUCTION SWAP rather than a GFN constructor argument. `models/gfn.py`
        is shared with crystal and, per the owner decision of 2026-08-19, takes no changes
        beyond the raw-state passthrough without a further decision. The cost is paid here:
        what the swap built is STAMPED into gfn_config['conformer'], and a checkpoint load
        rebuilds it through `gfn_from_config` before the weights go in.

        ON A RELOAD THIS IS A CHECKED NO-OP (`_check_reloaded_policy`). The model, its EMA and
        the optimizers already hold the checkpoint's state; swapping here would replace the
        loaded head with a fresh init, reset the EMA and discard the Adam moments. Only the
        stamp's `condition_set` is rewritten to this run's (`_restamp_condition_set`).

        THE STAMP ALSO NAMES THE CONDITION SET (`_with_condition_set`) on an energy that holds
        members: the identifiers in mol_id order and a signature per member, which
        `gfn_from_config` compares before a full resume restores anything keyed by mol_id.

        On a fresh build THREE THINGS HAVE TO HAPPEN IN THIS ORDER and the last is the one
        that bites: the base `init_gfn` has already deep-copied the EMA model and already
        built the optimizers over the OLD policy's parameters. Swapping without rebuilding
        both leaves a run that trains nothing in the new head and reports a perfectly
        plausible loss.
        """
        if getattr(self, '_gfn_reloaded', False):
            self._check_reloaded_policy()
            self._restamp_condition_set()
            return
        spec = getattr(self, '_policy_spec', {})
        kind = str(spec.get('policy_kind', 'flat')).lower()
        carrier = bool(getattr(getattr(self, 'energy_function', None), 'is_carrier', False))
        if kind == 'flat':
            if carrier:
                self._flat_on_carrier()
                self.gfn_config['conformer'] = self._with_condition_set(
                    {'policy_kind': 'flat', 'carrier': True,
                     'block_width': self._state_block_width()})
            return
        if kind != 'set':
            raise ValueError(
                f"model.policy_kind must be 'flat' or 'set', got "
                f"{spec.get('policy_kind')!r}")

        rank = int(self.gfn_config.get('dplr_rank', 0) or 0)
        if rank > 0:
            raise NotImplementedError(
                f"model.policy_kind: set with dplr_rank {rank} is refused. SetPolicy._to_blocks "
                f"emits the low-rank factor as rank-major blocks while GFN.split_params "
                f"reads it .view(-1, dim, rank), i.e. dim-major, so u_raw would be silently "
                f"TRANSPOSED -- finite, plausible and wrong. Set dplr_rank: 0 for this run.")

        from copy import deepcopy
        from models.set_policy import conditional_set_policy_for, set_policy_for

        head, stamp, mol_dim = self._set_policy_plan(spec)
        # model.zero_init REACHES THE SET HEAD: it zeroes the head's output layer, which has no
        # bias, so the untrained head emits exactly 0 whatever its inputs. Not passed, the key
        # was inert under policy_kind set, and an untrained set head was a random function
        # that any change in its per-coordinate feature width re-drew
        zero_init = bool(self.gfn_config.get('zero_init', False))
        common = dict(hidden_dim=stamp['set_policy_hidden'], layers=stamp['set_policy_layers'],
                      out_per_token=2, zero_init=zero_init)
        t_dim = int(self.gfn_config['t_dim'])
        if head == 'ragged':
            # RAGGED OVER VALID COLUMNS, dense [B, 2K] out -- see RaggedConditionalSetPolicy.
            # Per-column static features ride on the batch (`dof_static`), so nothing here is
            # bound to one member.
            policy = self._ragged_policy_from_stamp(
                stamp, self.energy_function.periodic_dims, t_dim,
                zero_init=zero_init).to(self.device)
        elif head == 'conditional':
            policy = conditional_set_policy_for(
                self.energy_function, t_dim, mol_dim,
                corr_dim=stamp['set_policy_corr_dim'], **common).to(self.device)
        else:
            policy = set_policy_for(self.energy_function, t_dim, **common).to(self.device)
        self.gfn_model.forward_policy = policy

        conditional = mol_dim is not None
        if conditional:
            # RE-CLASS, in the same post-construction spirit as the policy swap above and for
            # the same reason: `models/gfn.py` is shared with crystal. `ConformerGFN` adds no
            # constructor state beyond `_mol_cond`, which is set here, so rebinding __class__
            # is exactly equivalent to having built one -- and it keeps the conformer route's
            # only structural need (carrying the molecule into the policy call) out of the
            # shared file. The EMA copy is taken AFTER, so it inherits the class.
            # `_carrier` follows the HEAD, not the energy: the ragged head needs `state_mask`
            # on every batch, identity layout included, so a batch without one is refused at
            # binding rather than at the policy's first gather.
            from models.conformer_gfn import ConformerGFN
            self.gfn_model.__class__ = ConformerGFN
            self.gfn_model._mol_cond = None
            self.gfn_model._state_mask = None
            self.gfn_model._carrier = head == 'ragged'

        self.ema_model = deepcopy(self.gfn_model)
        self.init_schedulers_optimizers()
        # the ARCHITECTURE, for the checkpoint: Checkpointer.save stores gfn_config whole.
        # `stamp` itself stays the plan _check_reloaded_policy compares field by field; the
        # stored copy adds the condition set, which that comparison must not judge
        self.gfn_config['conformer'] = self._with_condition_set(stamp)

        n_new = sum(p.numel() for p in policy.parameters())
        print(f"policy: {'CONDITIONAL ' if conditional else ''}SET over {policy.dim} "
              f"coordinate tokens, {n_new:,} params, width independent of dim; the flat "
              f"scalarMLP is replaced and the optimizers were rebuilt over it")
        if conditional:
            print(f"        f_j is LEARNED from per-atom embeddings (DoFCorrelator); the "
                  f"pooled {mol_dim}-d molecular embedding joins rho's context")

    def _flat_on_carrier(self):
        """A flat policy on the carrier: refuse DPLR, then re-class both copies.

        The flat head runs on the carrier unchanged, but the log-probs must be masked per
        row, which only ConformerGFN does -- and only in gauss_logprob. With dplr_rank > 0
        fwd_gauss_logprob takes the Woodbury path and never calls it, so the pad columns
        (pinned to 0, residual -drift) would enter the forward density unmasked.
        """
        rank = int(self.gfn_config.get('dplr_rank', 0) or 0)
        if rank > 0:
            raise NotImplementedError(
                f"a flat policy on a CARRIER state with dplr_rank {rank} is refused: the "
                f"DPLR forward density bypasses ConformerGFN's per-row mask, so pad columns "
                f"would be scored as real coordinates. Set dplr_rank: 0 for this run.")
        from models.conformer_gfn import ConformerGFN
        # both copies: the EMA model was already deep-copied by the base init_gfn (or by
        # load_full, which re-classing leaves loaded)
        for m in (self.gfn_model, self.ema_model):
            m.__class__ = ConformerGFN
            m._mol_cond, m._state_mask, m._carrier = None, None, True
        print(f'policy: FLAT on a {self.energy_function.data_ndim}-wide CARRIER '
              f'state; log-probs masked per row')

    def _check_reloaded_policy(self):
        """After a checkpoint load: the config must describe the architecture the FILE holds.

        The architecture follows the file (the same rule as Checkpointer.
        _assert_dead_rows_match), so a disagreement is refused, naming the field, rather than
        resolved either way. A checkpoint with no 'conformer' block predates the stamp and
        was trained flat. Nothing is swapped, copied or rebuilt: the EMA and the optimizer
        state are the ones the load restored.
        """
        stored = self.gfn_config.get('conformer')
        stored_kind = 'flat' if stored is None else str(stored.get('policy_kind'))
        spec = self._config_policy_spec()
        want_kind = str(spec.get('policy_kind', 'flat')).lower()
        if want_kind not in ('flat', 'set'):
            raise ValueError(f"model.policy_kind must be 'flat' or 'set', got "
                             f"{spec.get('policy_kind')!r}")
        if want_kind != stored_kind:
            raise ValueError(
                f"model.policy_kind: this config asks for {want_kind!r}, the checkpoint "
                f"holds a {stored_kind!r} policy"
                + (" (it carries no conformer block: written before the stamp, i.e. flat)"
                   if stored is None else "")
                + ". The architecture follows the file; point checkpoint_name at a "
                  "checkpoint of this kind, or set policy_kind to match it.")
        if stored_kind == 'flat':
            if getattr(self.energy_function, 'is_carrier', False):
                self._flat_on_carrier()
            print("policy: FLAT, rebuilt from the checkpoint; nothing swapped")
            return
        _, expected, _ = self._set_policy_plan(spec)
        for field, want in expected.items():
            got = stored.get(field, '<absent>')
            if got != want:
                raise ValueError(
                    f"set policy field {field!r}: the checkpoint was built with {got!r}, "
                    f"this config and energy give {want!r}. The architecture follows the "
                    f"file -- restore the config value, or start this run fresh.")
        print(f"policy: SET head rebuilt from the checkpoint's stamp (hidden "
              f"{stored['set_policy_hidden']}, layers {stored['set_policy_layers']}, K = "
              f"{self.energy_function.data_ndim}); config agrees field by field -- nothing "
              f"swapped, EMA and optimizer state as restored")

    def scramble_applicable(self):
        """Never on the conditional set head, which does not read the scrambled seam.

        The scramble permutes the conditioner output at the conditioner->trunk seam. On the
        set route that seam feeds only s_model, which P_B and the flow head read; the forward
        policy takes the TRUE molecule through its own bindings (mol_emb, atom_embedding,
        dof_static) and never sees s_emb. A scrambled stage would train a conditional P_F
        against a P_B shown a shuffled condition -- and the scramble DETACHES the conditioner
        (GFN._maybe_scramble_condition_embedding), so the Z head's input would stop training
        too. Said once, so a stage flag asking for it does not read as having run.
        """
        if getattr(self.gfn_model.forward_policy, 'wants_molecular_conditioning', False):
            if not getattr(self, '_scramble_refusal_said', False):
                self._scramble_refusal_said = True
                print('scramble_conditions: NOT APPLIED -- the forward policy is the '
                      'conditional SET head, which reads the molecule through its own '
                      'bindings, not the scrambled s_emb seam')
            return False
        return super().scramble_applicable()

    def init_condition_log_z(self):
        """The base tracker, then -- on a FULL RESUME -- this config's tracker settings.

        load_full restores the tracker through from_state_dict, which takes min_visits,
        half_life_visits, trim_frac and max_batch_weight from the checkpoint, and the base
        method then returns early. A leg that retunes them (a ladder rung sized for a new M)
        would run the old values while its config read as applied. The rewind path
        (Modeller.fire_loss_spike) already writes the config's values back; this is the same
        rule on the conformer's resume, kept conformer-local so a crystal resume is unchanged.
        clip_beta is NOT adopted: z_grad_ema's history is denominated in it.
        """
        restored = hasattr(self, 'condition_log_z')
        super().init_condition_log_z()
        tracker = self.condition_log_z
        want = int(self.energy_function.condition_library_size)
        if int(tracker.library_size) != want:
            raise ValueError(
                f"condition_log_z holds {int(tracker.library_size)} conditions but this "
                f"run's energy has {want}. A restored table sized for another condition set "
                f"would index its rows against the wrong molecules.")
        cz = getattr(self.args, 'condition_log_z', None)
        if not restored or cz is None:
            return
        for key in ('min_visits', 'half_life_visits', 'trim_frac', 'max_batch_weight'):
            val = getattr(cz, key, None)
            if val is None:
                continue
            old = getattr(tracker, key)
            if old != val:
                print(f"condition_log_z.{key}: checkpoint {old!r} -> config {val!r}")
            setattr(tracker, key, val)

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

    def _noise_and_condition(self, batch, noise_log_range, anchor_inds=None):
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

        ``tile: 'thermal'`` replaces the isotropic kick with `_thermal_displace`: each
        coordinate at its own thermal width (energies/thermal_tile.py), scaled per row by
        ``10 ** U(buffers.anchor_buffer.thermal_noise_log_range)``. ``noise_log_range`` is
        then unread. The clip, the zeroed pads and the phi wrap are the same for both tiles.
        """
        from energies.conformer_data import batch_states, set_batch_states

        # `anchor_inds` is the crystal shaped tile's key into its sidecar; that tile is
        # built over CELL latents and has no conformer form, so it is refused here
        # rather than dropped (which would silently run the isotropic draw instead).
        tile = getattr(self.args.buffers.anchor_buffer, 'tile', 'iso')
        if tile == 'shaped':
            raise NotImplementedError(
                "buffers.anchor_buffer.tile: 'shaped' is a crystal-latent tile "
                "(x_min/evals/evecs over cell parameters) and has no conformer form.")
        if tile not in ('iso', 'thermal'):
            raise ValueError(f"buffers.anchor_buffer.tile must be 'iso' or 'thermal' on the "
                             f"conformer route, got {tile!r}")
        state = batch_states(batch)
        if tile == 'thermal':
            noised = self._thermal_displace(batch, state)
        else:
            log_min, log_max = float(noise_log_range[0]), float(noise_log_range[1])
            direction = torch.randn_like(state)
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            u = torch.rand(state.shape[0], device=state.device)
            magnitude = 10 ** (log_min + (log_max - log_min) * u)
            noised = self._clip_and_pad(batch, state + direction * magnitude[:, None])
        set_batch_states(batch, noised, periodic=self.energy_function.periodic_dims)

        batch, log_T_tensor, condition, condition_id =             self.energy_function.condition_samples(batch)
        return batch, log_T_tensor, condition, condition_id

    def _clip_and_pad(self, batch, noised):
        """The box clip on the non-periodic columns, then the carrier pads back to exactly 0."""
        lin = self.energy_function._lin_free_idx.to(noised.device)
        if lin.numel():
            noised[:, lin] = noised[:, lin].clip(min=-1, max=1)
        # CARRIER PADS STAY 0. They are not coordinates, and the energy refuses a row whose
        # pads are nonzero, so jittering them would fail the first anchor scan.
        smask = getattr(batch, 'state_mask', None)
        if smask is not None:
            smask = smask.reshape(noised.shape).bool().to(noised.device)
            noised = torch.where(smask, noised, torch.zeros_like(noised))
        return noised

    @property
    def _thermal_tile(self):
        """ONE `ThermalTile` for the run, built on first use; its widths are cached per member."""
        if getattr(self, '_thermal_tile_obj', None) is None:
            from energies.thermal_tile import ThermalTile
            self._thermal_tile_obj = ThermalTile(self.energy_function)
        return self._thermal_tile_obj

    def _thermal_displace(self, batch, state, c=None):
        """``state`` displaced by the thermal tile, clipped and padded; not written back.

        ``c`` None: each row's multiplier is ``c = 10 ** U(log_min, log_max)`` from
        ``buffers.anchor_buffer.thermal_noise_log_range`` (default [-0.5, 0.5]); c = 1 puts
        a row one thermal width out in every coordinate. A number fixes c for every row.
        Periodic columns are left unwrapped here; `set_batch_states` wraps them.
        """
        if c is None:
            rng = getattr(self.args.buffers.anchor_buffer, 'thermal_noise_log_range', None)
            log_min, log_max = (-0.5, 0.5) if rng is None else (float(rng[0]), float(rng[1]))
            u = torch.rand(state.shape[0], device=state.device)
            c = 10 ** (log_min + (log_max - log_min) * u)
        else:
            c = torch.full((state.shape[0],), float(c), device=state.device)
        return self._clip_and_pad(batch, state + self._thermal_tile.draw(batch, state, c))

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

    def _draw_carrier_prior(self, n, rng, report=False, steps=None):
        """``(batch, energies)`` -- ``n`` prior rows split evenly over a CARRIER set's members.

        Each member draws in ITS OWN chart (sample_prior_states, the optional relax and the
        bake all go through the member), and the states are then placed in the carrier. The
        condition graph is the file's carrier-padded one, so `state_mask`, the embeddings and
        the remapped reconstruction map ride along. Equal counts per member: the buffer is
        supposed to represent the SET, not whichever molecule drew first.
        """
        from energies.conformer_data import attach_states, bake_energies

        en = self.energy_function
        lay = en.carrier
        idents = list(en._members)
        per = max(-(-int(n) // len(idents)), 2)       # ceil, and attach_states needs >= 2
        parts, es = [], []
        for ident in idents:
            member = en._members[ident]
            states, _ = self._draw_prior_states(per, rng, report=report, steps=steps,
                                                en=member)
            e = bake_energies(member, states)
            x = lay.to_carrier(ident, torch.as_tensor(states).cpu())
            part = attach_states(self._condition_template(ident), x, e.cpu(),
                                 identifier=ident, periodic=en.periodic_dims)
            if hasattr(self, 'identifier_registry'):
                part.add_graph_attr(torch.full((part.num_graphs,),
                                               self.identifier_registry[ident],
                                               dtype=torch.long), 'mol_id')
            parts.append(part)
            es.append(e.cpu())
            if report:
                print(f'  {ident}: {per} prior rows, median E {float(e.median()):.2f} '
                      f'kcal/mol, k = {lay.k(ident)} of K = {lay.K}')
        batch = parts[0]
        for part in parts[1:]:
            batch = batch.append_batch(part)
        return batch, torch.cat(es)

    def _draw_prior_states(self, n, rng, report=False, chunk: int = 4096, steps=None,
                           en=None):
        """Draw from the fitted prior, then REPAIR steric clashes if configured.

        A product-of-marginals prior draws each torsion independently, so a long chain
        walks through itself. Measured on the glycine series at level `full`/mmff, the LJ
        term is 36% of the median energy at Gly3, 59% at Gly4 and 91% at Gly6 (2397 of
        2526 kcal/mol on Ala4), while EVERY bonded term scales linearly and sanely --
        angle 20.7 -> 52.9, bond 12.0 -> 24.0, electrostatic 10.2 -> 22.5 across the whole
        series. So the draw is not wrong, it is UNACCEPTED: the best decile is fine and the
        bulk self-intersects. Oversampling barely helps (keeping the best 1/100 of 20,000
        only reaches T_eff/T 3.0 on Gly6); a few descent steps fix it outright.

        CALIBRATED, NOT GUESSED. ``T_eff/T`` reads **2.0** for a correctly thermal sample,
        because equipartition puts the median excess at d/2 -- which is exactly what
        ``frac_within_equipartition`` tests against, and it independently agrees here
        (0.44 / 0.51 against its ideal 0.5). Measured over 4096 draws:

            molecule     d   0 steps     8     10     12     15     20
            phenyl-THP  72      2.05  1.15   1.09   1.06   1.04   1.02
            Gly4        87      7.03  2.20   2.00   1.89   1.79   1.70
            Gly6       129     29.19  2.47   2.21   2.06   1.92   1.80
            Ala4       123     41.84  2.43   2.15   1.99   1.86   1.74

        10-12 steps lands the peptides ON 2.0. phenyl-THP needs NONE -- its raw draw is
        already thermal and relaxing it only over-cools, which is why the default is 0:
        a molecule whose prior is already right must not be "repaired", and every existing
        config keeps the behaviour it was tuned with.

        Rprop in STATE space via ``descend``, which returns the best point SEEN, so a step
        that overshoots cannot land worse than the draw. Chunked because ``descend`` holds
        an autograd graph over the whole batch, and the seed draw is 50,000 rows.
        """
        # `en` is a carrier MEMBER when drawing per member (see _draw_carrier_prior); a draw
        # always happens in one molecule's own chart
        en = self.energy_function if en is None else en
        states, stats = en.sample_prior_states(
            self.internal_prior, n, rng, report=report)
        if steps is None:
            steps = getattr(self.args.energy_config, 'prior_relax_steps', 0)
        steps = int(steps or 0)
        if steps <= 0:
            return states, stats

        from energies.prior_baselines import descend

        x = torch.as_tensor(states, dtype=en.dtype, device=en.device)
        before = en.potential_energy(x, float(en.temperature)).detach()
        out = []
        # enable_grad EXPLICITLY: the churn path reaches here through
        # rebuild_prior_by_churn, which is decorated @torch.no_grad(), and `descend` needs
        # a graph to step. The seed path is not under no_grad, so this fails ONLY on churn
        # -- late, and long after any short smoke test has passed.
        with torch.enable_grad():
            for i in range(0, x.shape[0], chunk):
                best_x, _ = descend(en, x[i:i + chunk], steps)
                out.append(best_x.detach())
        x = torch.cat(out, 0)
        if report:
            after = en.potential_energy(x, float(en.temperature)).detach()
            print(f'  prior relax: {steps} Rprop steps in state space, median energy '
                  f'{before.median().item():.1f} -> {after.median().item():.1f} kcal/mol '
                  f'over {x.shape[0]} rows')
        return x, stats

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

        if getattr(self.energy_function, 'is_carrier', False):
            # PER-MEMBER draws, which is what the refusal below asks for on a heterogeneous
            # set; the carrier makes it possible because every member's rows share one width
            batch, _ = self._draw_carrier_prior(n, self._prior_rng, report=False)
            batch = batch.to(self.device)
            batch, log_T_tensor, condition, condition_id = \
                self.energy_function.condition_samples(batch)
            log_r = self.energy_function.prebuilt_sample_to_reward(batch, 10 ** log_T_tensor)
            return {'log_r': log_r.detach(), 'log_T_tensor': log_T_tensor,
                    'condition_id': condition_id}, batch

        # LABEL BY IDENTIFIER, NOT BY SMILES. Everything downstream -- the buffers, the
        # mol_id registry, the per-molecule energy table -- keys on the identifier, and a
        # conditions file is free to make that distinct from the SMILES (and must, when one
        # molecule appears more than once). On the single-molecule route the two coincide,
        # which is why `self.energy_function.smiles` worked there and raised KeyError the
        # moment a real condition set appeared.
        ident = getattr(self.energy_function, 'reference_identifier', None) \
            or self.energy_function.smiles

        # THIS DRAW IS FROM THE REFERENCE MEMBER ALONE. `_draw_prior_states` goes through the
        # energy's own chart, and a multi-molecule energy's chart is its reference member's.
        # Churning those rows into a buffer that is supposed to represent a SET would skew it
        # toward one molecule -- silently, because every row is individually valid. That is
        # harmless when the set is one molecule repeated (the paired-benchmark case) and is a
        # real bias otherwise, so it is refused rather than left to be discovered in a result.
        distinct = int(getattr(self.energy_function, 'distinct_smiles', 1) or 1)
        if distinct > 1:
            raise NotImplementedError(
                f'prior-buffer churn draws from the reference member only, but this energy '
                f'holds {distinct} distinct molecules -- the churned rows would all be '
                f'{self.energy_function.smiles!r} and would skew the buffer toward it. Draw '
                f'per member before enabling churn on a heterogeneous set, or disable churn '
                f'(prior_buffer.mean_lifetime) for this run.')

        states, _ = self._draw_prior_states(n, self._prior_rng, report=False)
        energies = bake_energies(self.energy_function, states)
        # THE CONDITION IS THE FILE'S, when there is one. `condition_from_energy` rebuilds a
        # BARE graph: no `embedding`, no `atom_embedding`, no `dof_atoms`. Those are baked by
        # the conditions builder from the encoder and cannot be recovered from the energy, so
        # a churned row built that way is a row the conditional policy cannot read --
        # `condition_samples` refuses the whole batch, at the first churn, which is inside
        # the first evaluation. Exactly the defect `init_mol_dataset` already documents for
        # the eval batch; this is the same mistake on the churn path.
        cond = self._condition_template(ident)
        batch = attach_states(cond, states.cpu(), energies.cpu(),
                              identifier=ident,
                              periodic=self.energy_function.periodic_dims)
        batch = batch.to(self.device)
        if hasattr(self, 'identifier_registry'):
            batch.add_graph_attr(
                torch.full((batch.num_graphs,), self.identifier_registry[ident],
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

    @staticmethod
    def _as_run_dtype(batch):
        """Cast a batch's float tensors to the run's dtype, in place.

        Condition and prior files are written under float64 (the builder sets it for its
        geometry checks) while the run is float32. Storage precision is only free where the
        tensor never meets a parameter, and these do -- states feed the policy, embeddings
        feed the conditioner -- so a double batch raises at the first matmul.
        """
        want = torch.get_default_dtype()
        for key, val in list(batch._store.items()):
            if torch.is_tensor(val) and val.is_floating_point() and val.dtype != want:
                batch[key] = val.to(want)
        return batch

    @staticmethod
    def _require_graph_file(blob, path, what):
        if not isinstance(blob, dict) or 'prior' not in blob:
            raise SystemExit(
                f'{path} is not a graph-form {what} file; expected a dict with a `prior` key '
                f'as written by build_conformer_conditions.py')
        return blob

    @staticmethod
    def _read_graph_file(path, what):
        blob = torch.load(path, weights_only=False, map_location='cpu')
        return ConformerModeller._require_graph_file(blob, path, what)

    def _refuse_stereo_mismatch(self, batch, path, what):
        """SystemExit when any graph of a file was built under another stereo-lock coefficient.

        At LOAD, before anything reads the file: its baked `conformer_energy` holds clip(U + P)
        at the builder's coefficient and the `prior_path` branch of `init_prior_dataset` takes
        it as it stands (only the single-molecule `_load_prior_dataset` re-scores), and its
        `dof_static` marks the prior's held double-bond rows only when the builder locked.
        `MultiConformerTorsions._resolve_rows` refuses the same rows at scoring; this names the
        FILE, before the first policy call has read them.
        """
        from energies.conformer_data import stereo_coeff_mismatch
        want = float(getattr(self.energy_function, 'stereo_coeff', 0.0))
        bad = stereo_coeff_mismatch(batch, want)
        if bool(bad.any()):
            rec = getattr(batch, 'ctree_stereo_coeff', None)
            got = (['absent (built before the field existed, read as 0)'] if rec is None
                   else sorted({float(v) for v in rec.reshape(-1)[bad].tolist()}))
            raise SystemExit(
                f'{what} file {path}: {int(bad.sum())} of {int(bad.numel())} graphs were built '
                f'under stereo_coeff {got}, but this run locks at {want:g} '
                f'(energy_config.stereo_coeff). Their baked energies and per-coordinate features '
                f'belong to the other target. Rebuild it with build_conformer_conditions.py '
                f'--stereo-coeff {want:g}.')

    def _refuse_reference_mismatch(self, batch, path, what):
        """SystemExit when any graph's stored reference conformer (`pos`) is not the one this
        run rebuilt for its identifier.

        A stored state is a DELTA from its member's reference, and each member's reference is
        re-embedded here from the SMILES (RDKit ETKDG + MMFF). Another RDKit embeds another
        reference for many members, and the file's `z` and chart flags -- all
        `MultiConformerTorsions._resolve_rows` compares -- still agree, so every stored state
        would be read against a geometry it was not built from, with no error. Compared on
        every graph, atom by atom in placement order (`condition_from_energy` writes
        `pos = energy.ref_pos`), against REFERENCE_POS_TOL.
        """
        en = self.energy_function
        members = getattr(en, '_members', None) or {}
        pos = getattr(batch, 'pos', None)
        idents = getattr(batch, 'identifier', None)
        if pos is None:
            raise SystemExit(f'{what} file {path}: its graphs carry no `pos`, so their reference '
                             f'conformers cannot be checked against this run\'s members')
        n_graphs = int(batch.num_graphs)
        if idents is None:
            if members:
                raise SystemExit(f'{what} file {path}: its graphs carry no `identifier`, so '
                                 f'their reference conformers cannot be matched to a member')
            idents = [None] * n_graphs
        idents = list(idents)
        refs, lib_of = [], {}
        missing = []
        for ident in dict.fromkeys(idents):
            member = members.get(ident) if members else en
            if member is None:
                missing.append(ident)
                continue
            lib_of[ident] = len(refs)
            refs.append(member.ref_pos.detach().to('cpu', torch.float64).reshape(-1, 3))
        if missing:
            raise SystemExit(f'{what} file {path}: {len(missing)} identifier(s) have no member '
                             f'in this run\'s energy, e.g. {ConformerModeller._name_some(missing)}')
        n_ref = torch.tensor([r.shape[0] for r in refs], dtype=torch.long)
        ref_ptr = torch.cat([torch.zeros(1, dtype=torch.long), n_ref.cumsum(0)])
        ref_cat = torch.cat(refs)
        lib = torch.tensor([lib_of[i] for i in idents], dtype=torch.long)
        ptr = batch.ptr.detach().cpu().long()
        n_atoms = ptr[1:] - ptr[:-1]
        dev = torch.zeros(n_graphs, dtype=torch.float64)
        bad_count = n_atoms != n_ref[lib]
        dev[bad_count] = float('inf')
        ok = ~bad_count
        if bool(ok.any()):
            graph = torch.repeat_interleave(torch.arange(n_graphs), n_atoms)
            slot = torch.arange(int(ptr[-1])) - ptr[:-1][graph]
            keep = ok[graph]
            at = ref_ptr[lib[graph[keep]]] + slot[keep]
            gap = (pos.detach().cpu().double().reshape(-1, 3)[keep]
                   - ref_cat[at]).abs().amax(-1)
            dev.scatter_reduce_(0, graph[keep], gap, reduce='amax')
        bad = dev > REFERENCE_POS_TOL
        if bool(bad.any()):
            worst = {}
            for g in torch.nonzero(bad).reshape(-1).tolist():
                worst[idents[g]] = max(worst.get(idents[g], 0.0), float(dev[g]))
            ranked = sorted(worst.items(), key=lambda kv: -kv[1])
            listed = ', '.join(f'{i!r} ({"atom count differs" if d == float("inf") else f"{d:.3g} A"})'
                               for i, d in ranked[:10])
            raise SystemExit(
                f'{what} file {path}: {len(worst)} of {len(lib_of)} condition(s) store a '
                f'reference conformer that is not the one this run rebuilt (max deviation '
                f'{ranked[0][1]:.3g} A, tolerance {REFERENCE_POS_TOL:g} A): {listed}'
                f'{" ..." if len(ranked) > 10 else ""}. Every stored state is a delta from its '
                f'member\'s reference, so these rows would be read against another geometry. '
                f'The file was built under another RDKit or builder version than this '
                f'environment; rebuild the rung IN THIS ENVIRONMENT (build_conformer_set.py, '
                f'then build_conformer_references.py).')
        print(f'{what} file {path}: reference conformers match the rebuilt members on all '
              f'{n_graphs:,} graph(s) of {len(lib_of)} condition(s) (max deviation '
              f'{float(dev.max()) if n_graphs else 0.0:.2g} A)')

    def _prior_row_energy(self):
        """The prior buffer's current per-row training energy, conformer currency.

        The base implementation mixes `physical_energy` with `flow_energy` -- both written by
        `analyze_crystal_batch`, which no conformer row ever passes through. Its error even
        diagnoses the absence as a stale checkpoint, which is wrong on this route: the field
        was never produced here at all.

        The conformer analogue is the baked `conformer_energy`, the same scalar
        `prebuilt_sample_to_reward` reads, so expiry compares rows in the currency they were
        admitted in. There is no lambda leg on this route, so there is nothing to mix.
        """
        e = getattr(self.prior_buffer.batch, 'conformer_energy', None)
        if e is None:
            raise AttributeError(
                'prior_buffer rows carry no `conformer_energy`; the prior prep must attach '
                'it (see build_conformer_conditions.py --prior-out)')
        return e.detach().cpu().flatten()

    def init_mol_dataset(self):
        """One condition: the molecule itself, carrying no state.

        ``ConformerBuffer._compute_xy`` falls back to the reference conformer for a
        conditions-only batch, which is the honest value -- the reference conformer IS the
        zero of this parameterisation.
        """
        from energies.conformer_data import collate_conditions, condition_from_energy

        # THE CONDITION SET IS THE FILE, when there is one. Rebuilding it from
        # `self.energy_function` gives two copies of ONE molecule carrying no embeddings, so
        # on the conditional route every evaluation drew a batch the policy could not be
        # conditioned on -- `condition_samples` then refused it. Same shape as the prior
        # intake, and the same fix: read the set that the run actually trains over.
        path = getattr(self.args, 'molecules_path', None)
        if path:
            batch = self._as_run_dtype(self._read_graph_file(path, 'conditions')['prior'])
            self._refuse_stereo_mismatch(batch, path, 'conditions')
            self._refuse_reference_mismatch(batch, path, 'conditions')
            self.mol_dataset = ConformerBuffer(batch,
                                               device=self.buffer_device,
                                               **self._buffer_kwargs(),
                                               exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)
            self.test_mol_dataset = None
            n_mols = len(set(batch.identifier)) if hasattr(batch, 'identifier') else 1
            has_emb = getattr(batch, 'embedding', None) is not None
            print(f'mol_dataset: {batch.num_graphs} condition(s) over {n_mols} molecule(s) '
                  f'from {path}' + ('; embeddings present' if has_emb else
                                    '; NO embeddings -- eval cannot be conditioned'))
            return

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

    def _condition_template(self, identifier):
        """One condition graph for `identifier`, carrying whatever the file baked onto it.

        Read from `mol_dataset` -- the set the run actually trains over -- rather than
        rebuilt, so the embedding, the per-atom embedding and the DoF atom frames ride along.
        Cached: the churn path calls this on every prior-buffer cycle and `to_data_list` on a
        32-graph batch is not free.

        Falls back to `condition_from_energy` only when there is no condition set at all,
        which is the unconditional route -- where there is nothing baked to lose.
        """
        from energies.conformer_data import condition_from_energy

        cache = getattr(self, '_cond_template_cache', None)
        if cache is None:
            cache = self._cond_template_cache = {}
        if identifier in cache:
            return cache[identifier]

        ds = getattr(self, 'mol_dataset', None)
        batch = getattr(ds, 'batch', None) if ds is not None else None
        if batch is not None and getattr(batch, 'identifier', None) is not None:
            idents = list(batch.identifier)
            if identifier in idents:
                graph = batch.to_data_list()[idents.index(identifier)]
                cache[identifier] = graph
                return graph
        cache[identifier] = condition_from_energy(self.energy_function, identifier=identifier)
        return cache[identifier]

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

    def init_anchor_buffer_seed(self):
        """Seed the anchor buffer from HARDER-RELAXED prior draws, not the prior dataset.

        The base method's `seed_source: prior_dataset` calls that dataset "the real
        high-quality dataset". On the crystal route it is -- a curated set of real
        crystals. HERE THERE IS NO DATASET: init_prior_dataset SAMPLES the fitted prior,
        and `prior_relax_steps` is tuned to land it at T_eff/T = 2.0, i.e. deliberately
        THERMAL. Seeding anchors from it gives 50,000 average states: measured on
        tetraglycine, mean 96.8, median 84.4, and a MINIMUM of 45.9 against a tier minimum
        of 28.3 -- not one good conformer in the set, and `anchor_admitted_last_n` stayed
        0 for the whole run, so it never improved on that.

        `seed_relax_steps` fixes the seed at its source. Measured over 800 draws, median
        energy by descent depth: 0 -> 253.1, 10 -> 69.3, 30 -> 51.9, 60 -> 47.1,
        120 -> 44.5, 250 -> 43.4. SIXTY IS THE KNEE, and its median (47.1) already beats
        the thermal seed's minimum (45.9); past it the gain is 3 kcal for 4x the cost.

        The prior dataset itself is left at `prior_relax_steps`. The two want different
        things: backward terminals should be thermal, anchors should mark where the good
        states are.

        RE-DRAWN AND RE-BAKED, never relaxed in place. `prebuilt_sample_to_reward` reads
        the baked `conformer_energy` off the batch and REFUSES to recompute, and those
        energies then warm condition_log_z's best_energy -- so moving the states without
        rescoring would seed the buffer, and the admission gate, with new geometry
        carrying old energies.

        Under ``prior_dataset_noise: thermal`` with no relax steps, the base method is handed
        the prior rows as they were BEFORE the noise (`_maybe_noise_prior_rows`), by the same
        stand-in.
        """
        import types

        from energies.conformer_data import (attach_states, bake_energies,
                                             condition_from_energy)

        # the UNNOISED prior rows under prior_dataset_noise (`_maybe_noise_prior_rows`), popped
        # whether or not they are used, so a restored anchor buffer does not keep them alive
        raw = self.__dict__.pop('_prior_dataset_raw', None)
        if hasattr(self, 'anchor_buffer'):
            return
        cfg = self.args.buffers.anchor_buffer
        steps = int(getattr(cfg, 'seed_relax_steps', 0) or 0)
        if getattr(cfg, 'seed_source', 'generated') != 'prior_dataset' or steps <= 0:
            if getattr(cfg, 'seed_source', 'generated') != 'prior_dataset':
                return super().init_anchor_buffer_seed()
            # anchors mark where the good states are: under prior_dataset_noise they are
            # seeded from the rows BEFORE the noise. A COMPACT source hands the base method
            # its bounded selection (_anchor_seed_row_batches) through `row_batches`, which
            # train.py::_dataset_row_batches reads; a graph batch is handed over whole. Both
            # by the same stand-in as the relaxed seed below.
            source = self.prior_dataset if raw is None else raw
            if getattr(source, 'is_compact', False):
                stand_in = types.SimpleNamespace(
                    batch=source.batch,
                    row_batches=lambda limit=None: self._anchor_seed_row_batches(source))
            elif raw is None:
                return super().init_anchor_buffer_seed()
            else:
                stand_in = types.SimpleNamespace(batch=raw)
            saved = self.prior_dataset
            self.prior_dataset = stand_in
            try:
                return super().init_anchor_buffer_seed()
            finally:
                self.prior_dataset = saved

        n = int(getattr(self.args.energy_config, 'prior_sample_size', 50000))
        states, _ = self._draw_prior_states(n, self._prior_rng, report=False, steps=steps)
        energies = bake_energies(self.energy_function, states)
        cond = condition_from_energy(self.energy_function,
                                     identifier=self.energy_function.smiles)
        batch = attach_states(cond, torch.as_tensor(states).cpu(), energies.cpu(),
                              identifier=self.energy_function.smiles,
                              periodic=self.energy_function.periodic_dims)
        e = energies.detach().cpu().numpy()
        print(f'anchor seed: {n} prior draws relaxed {steps} steps -- median '
              f'{np.median(e):.1f}, p10 {np.percentile(e, 10):.1f}, min {e.min():.1f} '
              f'kcal/mol (the prior dataset itself stays thermal)')

        # hand the RELAXED set to the base method by standing in for prior_dataset: every
        # step after the seed batch is chosen -- condition_samples, the reward read,
        # update_best_energy, key parity, buffer construction -- is identical and worth
        # reusing rather than duplicating.
        saved = getattr(self, 'prior_dataset', None)
        self.prior_dataset = types.SimpleNamespace(batch=batch)
        try:
            super().init_anchor_buffer_seed()
        finally:
            self.prior_dataset = saved

    def _load_prior_dataset(self, path):
        """Read a prebuilt conformer state set off disk, and PROVE it belongs to this run.

        A state tensor carries no self-description: rows built for another molecule, another
        `level`, or another `energy_clip` load without complaint and train against energies
        that mean something else. So two checks, and the second is the one that bites.

        1. Metadata equality on the keys that change what an energy MEANS -- smiles, level,
           force_field, energy_clip, data_ndim.
        2. RE-SCORE the loaded states and compare against the stored energies. Metadata can
           be right while the file is stale (rebuilt force field, changed spanning tree),
           and only recomputation catches that. `prebuilt_sample_to_reward` reads the baked
           energy and REFUSES to recompute, so a mismatch that slips through here is never
           caught later -- it just trains on wrong rewards.
        """
        from energies.conformer_data import bake_energies

        blob = torch.load(path, weights_only=False)
        want = {'smiles': self.energy_function.smiles,
                'level': getattr(self.energy_function, 'level', None),
                'force_field': getattr(self.energy_function, 'force_field', None),
                'energy_clip': getattr(self.args.energy_config, 'energy_clip', None),
                'data_ndim': int(self.energy_function.data_ndim)}
        bad = {k: (blob.get(k), v) for k, v in want.items()
               if v is not None and k in blob and blob[k] != v}
        if bad:
            raise SystemExit(
                f'prior_dataset_path={path!r} was not built for this run:\n' +
                '\n'.join(f'  {k}: file has {a!r}, run wants {b!r}' for k, (a, b) in bad.items()))

        states = torch.as_tensor(blob['states'], dtype=self.energy_function.dtype,
                                 device=self.device)
        stored = torch.as_tensor(blob['energies']).to(states.device).double()
        fresh = bake_energies(self.energy_function, states).detach().double()
        gap = (fresh - stored).abs()
        finite = torch.isfinite(gap)
        worst = float(gap[finite].max()) if bool(finite.any()) else float('inf')
        if worst > 1e-2:
            raise SystemExit(
                f'prior_dataset_path={path!r} re-scores differently from its stored '
                f'energies (worst |delta| = {worst:.4g} kcal/mol over {len(states):,} rows). '
                'The file is stale with respect to the current energy definition; rebuild '
                'it rather than training on rewards that no longer describe these states.')
        print(f'prior dataset: {len(states):,} states loaded from {path} and re-scored '
              f'(worst |delta| {worst:.2e} kcal/mol) -- provenance verified')
        return states, fresh.to(self.energy_function.dtype)

    def _maybe_noise_prior_rows(self, batch):
        """``prior_dataset_noise: thermal`` -- every prior-dataset row replaced by a noised copy.

        'none' (the default, and absent) returns ``batch`` untouched. 'thermal' displaces
        every row by the thermal tile at c = 1 (`_thermal_displace`: one thermal width in
        every coordinate, NOT the anchor top-ups' multiplier range) and RE-BAKES its
        ``conformer_energy``, which `prebuilt_sample_to_reward` reads and never recomputes.
        One noised copy per row, drawn once at init: the dataset stays the same size.

        What reads the prior dataset reads the noised rows: the warm-up's backward draws
        (bwd_sampling_mode 'dataset'), the prior-buffer seed and `reseed_prior_from_dataset`
        (both `_prior_dataset_seed_batches`), and the eval reads that sample it (the
        sliced-Wasserstein reference, the eval figures, `_e_min`'s starts, the single-chart
        `dof_class_stats` reference). The ANCHOR seed does not: the unnoised rows are kept
        as `_prior_dataset_raw` and `init_anchor_buffer_seed` seeds from them, since anchors
        mark where the good states are.
        """
        mode = getattr(self.args, 'prior_dataset_noise', None) or 'none'
        if mode not in ('none', 'thermal'):
            raise ValueError(f"prior_dataset_noise must be 'none' or 'thermal', got {mode!r}")
        if mode == 'none':
            return batch
        from energies.conformer_data import batch_states, set_batch_states, wrap_state

        self._prior_dataset_raw = batch
        noised = batch.clone()
        state = batch_states(noised).to(self.device)
        x = self._thermal_displace(noised, state, c=1.0)
        x = wrap_state(x, self.energy_function.periodic_dims)
        e = self._bake_rows(noised, x)
        raw_e = torch.as_tensor(batch.conformer_energy).reshape(-1).double().cpu()
        excess = (e.double().cpu() - raw_e) / float(self.energy_function.temperature)
        set_batch_states(noised, x.cpu(), energies=e.cpu(),
                         periodic=self.energy_function.periodic_dims)
        print(f'prior dataset: prior_dataset_noise THERMAL -- {batch.num_graphs:,} rows '
              f'replaced by noised copies, median energy {float(raw_e.median()):.1f} -> '
              f'{float(e.median()):.1f} kcal/mol, median excess '
              f'{float(excess.median()):.1f} kT (the anchor seed keeps the unnoised rows)')
        return noised

    def _prior_noise_mode(self):
        mode = getattr(self.args, 'prior_dataset_noise', None) or 'none'
        if mode not in ('none', 'thermal'):
            raise ValueError(f"prior_dataset_noise must be 'none' or 'thermal', got {mode!r}")
        return mode

    def _compact_prior_store(self, keys, columns):
        """A compact prior-dataset store from per-row columns and each row's slot in the
        conditions table (buffer.py::ConformerCompactRows.from_columns)."""
        from buffer import ConformerCompactRows

        rows = ConformerCompactRows.from_columns(self.conformer_statics, keys, columns,
                                                 exclude_keys=BULKY_ATTR_EXCLUDE_KEYS)
        return ConformerBuffer(rows,
                               device=self.buffer_device,
                               **self._buffer_kwargs(),
                               x_fn=None,
                               y_fn=self._buffer_y_fn(),
                               exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                               )

    def _init_compact_prior_dataset(self, blob, path):
        """The prior dataset from a COMPACT prior file (`energies/conformer_data.py::
        compact_prior`): each row's state and energy, joined to the conditions table by its
        condition's identifier. No graph is built per row; the steps that need graphs (the
        thermal noise) take `COMPACT_CHUNK_ROWS` rows at a time.

        Refused: a run without a conditions table (`molecules_path` unset or without
        identifiers); an identifier the table lacks; a state width, a condition's
        free-column mask or its atom count that is not the table's; a non-zero pad column.
        The reference-conformer and stereo-lock checks the graph form runs per row are the
        conditions file's own here (`init_mol_dataset`): a compact row has no graph of its
        own to differ from its condition's.

        Under `prior_dataset_noise: thermal` the rows as stored are kept as a second compact
        store, `_prior_dataset_raw`, for the anchor seed, as `_maybe_noise_prior_rows` keeps
        the graph form's.
        """
        from buffer import COMPACT_CHUNK_ROWS, chunk_bounds
        from energies.conformer_data import (PRIOR_COMPACT_FORMAT, batch_states,
                                             check_compact_prior, state_dim, wrap_state)

        statics = getattr(self, 'conformer_statics', None)
        if statics is None:
            raise SystemExit(
                f'{path} is a compact prior file ({PRIOR_COMPACT_FORMAT}): its rows carry a '
                f'state, an energy and a condition, and read everything else from the '
                f'conditions table, which this run does not have (molecules_path must name '
                f'the conditions file the prior was built with, one identifier per graph)')
        try:
            check_compact_prior(blob, path)
        except ValueError as err:
            raise SystemExit(str(err)) from None
        idents = list(blob['identifiers'])
        missing = [i for i in idents if i not in statics.slot_of]
        if missing:
            raise SystemExit(
                f'{path}: {len(missing)} of its {len(idents)} condition(s) are not in the '
                f'conditions table ({self.args.molecules_path}), e.g. '
                f'{self._name_some(missing)}. The prior file and the conditions file must '
                f'come from one build_conformer_set.py run')
        slots = torch.tensor([statics.slot_of[i] for i in idents], dtype=torch.long)
        table = statics.batch
        x, e = blob['torsion_state'], blob['conformer_energy'].reshape(-1)
        ci = blob['condition_index'].long()
        n, K = int(x.shape[0]), int(x.shape[1])
        if n < 2:
            raise SystemExit(f'{path}: {n} prior row(s); a prior dataset needs at least 2')
        table_mask = table._store.get('state_mask', None)
        if K != state_dim(table) or table_mask is None:
            raise SystemExit(
                f'{path}: states are {K} wide; the conditions table '
                f'({self.args.molecules_path}) is {state_dim(table)} wide'
                + ('' if table_mask is not None else ' and carries no state_mask'))
        want_mask = table_mask.reshape(int(table.num_graphs), -1).bool()[
            slots.to(table_mask.device)].cpu()
        bad = (want_mask != blob['state_mask'].bool()).any(1) \
            | (statics.atom_counts(slots) != blob['n_atoms'].long())
        if bool(bad.any()):
            raise SystemExit(
                f'{path}: {int(bad.sum())} of its {len(idents)} condition(s) record a '
                f'free-column mask or an atom count that is not the conditions table\'s '
                f'({self.args.molecules_path}), e.g. '
                f'{self._name_some([idents[j] for j in torch.nonzero(bad).reshape(-1).tolist()])}'
                f'. The prior file and the conditions file must come from one '
                f'build_conformer_set.py run')
        mask = blob['state_mask'].bool()
        for i in range(0, n, 16 * COMPACT_CHUNK_ROWS):
            j = slice(i, i + 16 * COMPACT_CHUNK_ROWS)
            if bool((x[j][~mask[ci[j]]] != 0).any()):
                raise SystemExit(f'{path}: a prior row has a non-zero PAD column (rows '
                                 f'{i}..{min(n, j.stop) - 1}); pads are 0 in every stored state')

        # CAST TO THE RUN'S DTYPE, as the graph form is: states feed the policy
        want = torch.get_default_dtype()
        columns = {'torsion_state': x.to(want), 'conformer_energy': e.to(want)}
        for name, val in (blob.get('rows') or {}).items():
            columns[name] = val.to(want) if val.is_floating_point() else val
        keys = slots[ci]
        store = self._compact_prior_store(keys, columns)

        if self._prior_noise_mode() == 'thermal':
            # every row replaced by ONE noised copy, re-baked, a chunk of graphs at a time
            self._prior_dataset_raw = store
            periodic = self.energy_function.periodic_dims
            xs, es = [], []
            for i, j in chunk_bounds(n, COMPACT_CHUNK_ROWS):
                sub = store.batch.subsample_new_batch(torch.arange(i, j))
                state = batch_states(sub).to(self.device)
                xn = wrap_state(self._thermal_displace(sub, state, c=1.0), periodic)
                es.append(self._bake_rows(sub, xn).detach().cpu().reshape(-1).to(want))
                xs.append(xn.detach().cpu().to(want))
            noised = dict(columns, torsion_state=torch.cat(xs), conformer_energy=torch.cat(es))
            raw_e = columns['conformer_energy'].double()
            excess = (noised['conformer_energy'].double() - raw_e) \
                / float(self.energy_function.temperature)
            print(f'prior dataset: prior_dataset_noise THERMAL -- {n:,} rows replaced by '
                  f'noised copies, median energy {float(raw_e.median()):.1f} -> '
                  f'{float(noised["conformer_energy"].double().median()):.1f} kcal/mol, '
                  f'median excess {float(excess.median()):.1f} kT (the anchor seed keeps the '
                  f'unnoised rows)')
            columns = noised
            store = self._compact_prior_store(keys, columns)
        self.prior_dataset = store

        en = columns['conformer_energy'].double().numpy()
        teff = 1 + 2 * (float(np.median(en)) - float(en.min())) / self.energy_function.ndim
        print(f'prior dataset: {n:,} rows over {len(idents)} MOLECULES from {path} '
              f'(compact form, {store.batch.nbytes() / max(n, 1):.0f} B per row) -- median '
              f'{np.median(en):.1f}, p10 {np.percentile(en, 10):.1f}, '
              f'p90 {np.percentile(en, 90):.1f} kcal/mol, T_eff/T = {teff:.2f}')
        if 'embedding' in table._store:
            print(f'               molecular embeddings present on the conditions table '
                  f'({tuple(table.embedding.shape)}), so the backward branch is '
                  f'condition-aware')

    def _anchor_seed_row_batches(self, source):
        """The anchor seed's rows from a COMPACT prior-dataset store `source`: at most
        `buffers.anchor_buffer.max_size` of them, each condition's lowest-energy rows
        (buffer.py::lowest_rows_per_condition), a chunk of graphs at a time.

        `buffers.anchor_buffer.seed_rows_per_condition` sets the rows per condition; null
        or absent takes every row when the dataset fits `max_size`, and otherwise the most
        rows per condition that fit. Refused when even that does not fit: more conditions
        than `max_size`, or the configured count times the conditions above it. A full
        (graph-row) prior dataset does not come through here (`init_anchor_buffer_seed`):
        it is handed over whole, as on the crystal route, and is the single-molecule draw
        or a relaxed re-draw, `energy_config.prior_sample_size` rows.
        """
        from buffer import COMPACT_CHUNK_ROWS, chunk_bounds, lowest_rows_per_condition

        cfg = self.args.buffers.anchor_buffer
        max_size = int(cfg.max_size)
        per_condition = getattr(cfg, 'seed_rows_per_condition', None)
        keys = source.batch.keys
        energy = torch.as_tensor(source.batch.conformer_energy).detach().reshape(-1).cpu()
        try:
            rows, k, n_cond = lowest_rows_per_condition(keys, energy, max_size, per_condition)
        except ValueError as err:
            raise SystemExit(
                f'anchor seed from the prior dataset ({len(source):,} rows): {err}. Set '
                f'buffers.anchor_buffer.max_size and '
                f'buffers.anchor_buffer.seed_rows_per_condition so that the seed fits') \
                from None
        print(f'anchor seed: {len(rows):,} of {len(source):,} prior-dataset rows, the '
              f'lowest-energy {k} per condition over {n_cond:,} condition(s) '
              f'(buffers.anchor_buffer.max_size {max_size:,}, seed_rows_per_condition '
              f'{per_condition})')
        return (source.batch.subsample_new_batch(rows[i:j])
                for i, j in chunk_bounds(int(rows.numel()), COMPACT_CHUNK_ROWS))

    def _bake_rows(self, batch, states, chunk: int = 2048):
        """Baked ``conformer_energy`` (T = 1) for ``states`` on ``batch``'s rows, ``[n]``.

        One chart: `bake_energies`. A molecule SET: the one-pass energy in chunks, each row
        through its own member, reading back the ``conformer_energy`` it writes.
        """
        from energies.conformer_data import bake_energies

        en = self.energy_function
        if not getattr(en, '_members', None) or en.n_charts <= 1:
            return bake_energies(en, states).detach()
        out = []
        n = int(states.shape[0])
        for i in range(0, n, chunk):
            idx = torch.arange(i, min(i + chunk, n))
            sub = batch.subsample_new_batch(idx).to(en.device)
            _, sub = en.energy(states[i:i + chunk].to(en.device), sub,
                               torch.zeros(len(idx), device=en.device), return_exp=True)
            out.append(sub.conformer_energy.reshape(-1).detach())
        return torch.cat(out)

    def init_prior_dataset(self):
        """Phase 1's dataset: a prebuilt set off disk, or draws from the fitted prior.

        `energy_config.prior_dataset_path` loads a set built by
        `build_conformer_prior_dataset.py`. With it unset the conformer track has no file to
        read -- unlike the crystal route -- so the equivalent is to sample the fitted
        InternalPrior: `prior_sample_size` rows, scored once here (owner decision
        2026-08-20).

        Scored at init for the same reason the crystal route re-analyses its prior at init:
        ``prebuilt_sample_to_reward`` reads a baked ``conformer_energy`` off the graph and
        REFUSES to recompute, because a silent rescore there would hide a prep bug behind
        plausible numbers.

        The draw uses ``sample_prior_states`` at its defaults, so joint ring sampling is
        ON -- the path that benchmarks 32x-87000x over uniform-on-box.

        With a conditions table (`_init_conformer_statics`, built here first) the dataset is a
        compact store: each row keeps its state and energy, and joins the rest from the table.
        """
        from energies.conformer_data import attach_states, bake_energies, condition_from_energy

        self._init_conformer_statics()

        # MULTI-MOLECULE GRAPH-FORM PRIOR, and it is the only intake that can carry a
        # molecule SET. Everything below this branch rebuilds the batch from
        # `self.energy_function`, which is ONE molecule (problem.smiles) -- so the states may
        # vary but the graph, and therefore the condition, never does. A conditional run needs
        # per-row molecular identity to survive into the backward branch, and the frozen
        # embeddings ride on the graph, so the batch has to be taken off disk WHOLE rather
        # than reconstructed. `prior_path` was previously named only in an error message and
        # read by nothing.
        graph_prior = getattr(self.args, 'prior_path', None)
        if graph_prior:
            from energies.conformer_data import PRIOR_GRAPH_FORMAT, is_compact_prior
            blob = torch.load(graph_prior, weights_only=False, map_location='cpu')
            if is_compact_prior(blob):
                # COMPACT FORM: state, energy and condition per row. Joined to the conditions
                # table without a graph per row ever being built.
                self._init_compact_prior_dataset(blob, graph_prior)
                return
            self._require_graph_file(blob, graph_prior, 'prior')
            batch = blob.get('equalized_prior', None) or blob['prior']
            print(f'prior file {graph_prior}: GRAPH form ({PRIOR_GRAPH_FORMAT}), one full '
                  f'condition graph per row, read WHOLE ({batch.num_graphs:,} rows) before it '
                  f'is compacted. build_conformer_set.py now writes the compact form, which '
                  f'is joined to the conditions table without that.')
            # CAST TO THE RUN'S DTYPE. build_conformer_conditions runs under float64 for its
            # geometry checks, so every float tensor on the file is double while the run is
            # float32. Storage precision is only free where the tensor never meets a
            # parameter, and these do: states feed the policy and the embeddings feed the
            # conditioner, so a double batch raises `expected m1 and m2 to have the same
            # dtype` at the first matmul rather than merely wasting memory.
            batch = self._as_run_dtype(batch)
            self._refuse_stereo_mismatch(batch, graph_prior, 'prior')
            self._refuse_reference_mismatch(batch, graph_prior, 'prior')
            e_t = getattr(batch, 'conformer_energy', None)
            if e_t is None:
                raise SystemExit(
                    f'{graph_prior} carries no `conformer_energy`; prebuilt_sample_to_reward '
                    f'reads it off the graph and REFUSES to recompute, so a file without it '
                    f'would train on rewards that were never scored')
            energies = torch.as_tensor(e_t).reshape(-1).to(self.energy_function.dtype)
            batch = self._maybe_noise_prior_rows(batch)
            self.prior_dataset = ConformerBuffer(batch,
                                                 device=self.buffer_device,
                                                 **self._buffer_kwargs(),
                                                 x_fn=None,
                                                 y_fn=self._buffer_y_fn(),
                                                 exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                                 )
            e = energies.detach().cpu().numpy()
            n_mols = len(set(batch.identifier)) if hasattr(batch, 'identifier') else 1
            teff = 1 + 2 * (float(np.median(e)) - float(e.min())) / self.energy_function.ndim
            print(f'prior dataset: {batch.num_graphs:,} rows over {n_mols} MOLECULES from '
                  f'{graph_prior} -- median {np.median(e):.1f}, '
                  f'p10 {np.percentile(e, 10):.1f}, p90 {np.percentile(e, 90):.1f} kcal/mol, '
                  f'T_eff/T = {teff:.2f}')
            if getattr(batch, 'embedding', None) is not None:
                print(f'               molecular embeddings present '
                      f'({tuple(batch.embedding.shape)}), so the backward branch is '
                      f'condition-aware')
            return

        path = getattr(self.args.energy_config, 'prior_dataset_path', None)
        n = int(getattr(self.args.energy_config, 'prior_sample_size', 50000))
        if getattr(self.energy_function, 'is_carrier', False):
            if path:
                raise SystemExit('prior_dataset_path is a single-molecule state file; a '
                                 'CARRIER set draws its prior per member from the fitted '
                                 'InternalPrior. Unset prior_dataset_path.')
            if self.internal_prior is None:
                raise SystemExit('a CARRIER set draws its prior per member from the fitted '
                                 'InternalPrior, and energy_config.internal_prior_path is unset')
            print(f'prior dataset: CARRIER set, {n} rows split over '
                  f'{self.energy_function.n_charts} members')
            batch, energies = self._draw_carrier_prior(n, self._prior_rng, report=True)
            batch = self._maybe_noise_prior_rows(batch)
            self.prior_dataset = ConformerBuffer(batch,
                                                 device=self.buffer_device,
                                                 **self._buffer_kwargs(),
                                                 x_fn=None,
                                                 y_fn=self._buffer_y_fn(),
                                                 exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                                 )
            return
        if path:
            states, energies = self._load_prior_dataset(path)
            stats = {}
        else:
            if self.internal_prior is None:
                raise SystemExit(
                    'energy_config.internal_prior_path and prior_dataset_path are both '
                    'unset and prior_path is null, so there is nothing to seed phase 1 '
                    'from. Point one of them at a prior.')
            rng = self._prior_rng
            states, stats = self._draw_prior_states(n, rng, report=True)
            energies = bake_energies(self.energy_function, states)

        # THE CONDITION IS THE FILE'S, when there is one -- the THIRD site of this same
        # defect, after init_mol_dataset (eval batch) and sample_from_prior (churn path).
        # `condition_from_energy` rebuilds a BARE graph: no `embedding`, no
        # `atom_embedding`, no `dof_atoms`. The prior dataset seeds the prior BUFFER
        # (init_prior_buffer_seed -> _prior_dataset_seed_batches -> condition_samples), so on
        # a conditional route a bare condition here is refused before the first train step.
        # `_condition_template` falls back to condition_from_energy when there is no
        # condition set, so the unconditional route is unchanged.
        ident = getattr(self.energy_function, 'reference_identifier', None)             or self.energy_function.smiles
        cond = self._condition_template(ident)
        batch = attach_states(cond, states.cpu(), energies.cpu(),
                              identifier=ident,
                              periodic=self.energy_function.periodic_dims)
        # SAME CONSTRUCTION ARGS AS THE CRYSTAL prior_dataset. y_fn in particular is not
        # optional: log_buffer_stats reads `buff.y` for the energy readout, and during the
        # warm-start stage (bwd_sampling_mode 'dataset') the buffer it reads is THIS one,
        # not prior_buffer. Without it every prior energy metric is silently absent.
        batch = self._maybe_noise_prior_rows(batch)
        self.prior_dataset = ConformerBuffer(batch,
                                             device=self.buffer_device,
                                             **self._buffer_kwargs(),
                                             x_fn=None,
                                             y_fn=self._buffer_y_fn(),
                                             exclude_keys=BULKY_ATTR_EXCLUDE_KEYS,
                                             )
        e = energies.detach().cpu().numpy()
        src = f'loaded from {path}' if path else \
            f'{n} states sampled from the fitted InternalPrior and scored at init'
        # T_eff/T is the reading that says whether these terminals are THERMAL (2.0) or a
        # curated low-energy set (well below it). Both are legitimate -- TB is off-policy,
        # so the fixed point does not depend on the backward terminal distribution -- but
        # which one is in play changes what the backward branch is teaching, and it should
        # never be a surprise. Measured on the filtered tetraglycine set: 1.18.
        teff = 1 + 2 * (float(np.median(e)) - float(e.min())) / self.energy_function.ndim
        print(f'prior dataset: {src} -- median {np.median(e):.1f}, '
              f'p10 {np.percentile(e, 10):.1f}, p90 {np.percentile(e, 90):.1f} kcal/mol, '
              f'T_eff/T = {teff:.2f} (2.0 = thermal, lower = curated/cold)')
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
