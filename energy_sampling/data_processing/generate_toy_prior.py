"""
Config-driven generator for toy (latent_harmonic / latent_multiharmonic)
condition sets and priors.

Produces two independent artifacts, matching the two roles these files play
in training (see train.py's init_mol_dataset / init_prior_dataset):

  - condition set (molecules_path-type): one entry per configured condition,
    carrying `identifier` / `c` / `width` plus ordinary crystal-structure
    scaffolding borrowed from an existing template file. For its
    molecules_path role only the identifier/c/width metadata and graph
    content matter -- forward sampling always recomputes energy fresh from
    whatever latents the GFN actually produces (see analyze_crystal_batch's
    per-sample `c` override). But by default
    (condition_set.sample_latents: true) each condition's replicas ALSO get
    latent positions genuinely sampled from that condition's own target
    distribution, with energy baked in, so the same file doubles as a set of
    on-mode, per-condition seeds -- e.g. buffers.anchor_buffer.seed_source in
    training, to hand the anchor set every mode explicitly instead of
    relying on mode discovery. Set sample_latents: false to recover the old
    behavior (template's leftover cell params, no baked energy).

  - prior (prior_path-type): backward training draws directly from this file
    and gets its reward via prebuilt_sample_to_reward, which reads a STATIC,
    pre-baked energy attribute rather than recomputing -- so unlike the
    condition set, these samples need real latent positions and a correctly
    baked-in energy. Two modes (prior.mode):
      'shared' (default) -- ONE zero-centered (prior.shared_center) Gaussian,
        wide enough (prior.shared_width) to cover every configured
        condition, with NO knowledge of which condition a given draw will
        end up labeled with. This is the actual "broad unconditional prior"
        -- q_target << q_prior in the sense that the single shared prior's
        support comfortably contains every narrow target.
      'per_condition' -- for later: each condition gets its own Gaussian
        CENTERED on that condition's own `c` but scaled up by
        prior.width_multiplier, so q_target << q_prior holds locally,
        per-condition, by construction, rather than via one shared prior.
    Either way, energy is baked under each labeled sample's own (narrow)
    condition `width`, and every analyze() call only ever contains one
    condition at a time (see the ASYMMETRY note below) -- coverage_check
    reports an empirical q_target-vs-q_prior ESS check appropriate to
    whichever mode is active.

IMPORTANT ASYMMETRY between the two toy energy functions, discovered by
direct testing against mxtaltools -- not a design choice made here:
  - latent_harmonic_en(c, width, x) does `x - c[0]` internally: if `c` is a
    per-sample [n, cond_dim] tensor it silently uses only row 0 for every
    sample. It is NOT safe to bake more than one condition into a single
    analyze() call for this energy function.
  - latent_multiharmonic_en broadcasts a per-sample [n, cond_dim] `c`
    correctly (confirmed via its 'kdc,...c->...kd' einsum) and likewise a
    per-sample [n] `width`; analyze() forwards both untouched. Its
    SAMPLING-side helpers (sample_latent_multiharmonic,
    log_partition_latent) remain single-shared-c / scalar-width only.
  _build_prior_shared exploits this for multiharmonic (one multi-condition
  analyze() call for the whole prior); everything else here stays one
  condition per analyze()/sample_* call, which keeps latent_harmonic safe.
  latent_harmonic remains unsafe if you batch multiple conditions into one
  analyze() call anywhere else.

Also note: multiharmonic's condition vectors are NOT tied to the latent
space's dimensionality (data_ndim) -- they control mxtaltools'
_build_latent_field's Gaussian-mixture field and default to 8-dim
(_build_latent_field(cond_dim=8)) regardless of data_ndim. Match your `c`
vectors' length to whatever cond_dim you're using (8, if left at the
default).

Optional top-level `latent_field` config block forwards its keys straight
through to mxtaltools' _build_latent_field (cond_dim, n_core, n_ghost,
sigma_range, aniso_scale, depth_range, mu_scale, disp_max, logsig_scale,
gate_steep, edge_sigmas, max_temperature, max_width, seed) -- see that
function's docstring in crystal_analysis.py for what each knob does. Only
relevant for latent_multiharmonic (latent_harmonic doesn't use the GMM
field at all). Every latent_multiharmonic-touching call in this script
(sampling, density, analyze()) forwards these same kwargs, and
mxtaltools only actually consumes them the FIRST time the field is built
on a given batch object (_ensure_latent_field is a no-op once `_field` is
cached) -- so as long as `latent_field` stays fixed for a whole run, every
call sees the same field regardless of which one happened to trigger the
build. Leave the block out (or empty) to use mxtaltools' own defaults.

Optional condition_manifold (see expand_condition_manifold): instead of N
discrete named conditions, expands the `conditions` list (treated as
anchors) into a much larger set of draws spanning the connected region
between them -- random convex combinations of the anchors' own (c, width),
optionally jittered -- for testing generalization over a continuous
manifold rather than at fixed modes. Every draw gets its own unique
identifier (train.py's condition_id resolves purely through the identifier
string, so shared identifiers would collapse distinct conditions into one
bookkeeping slot); coverage_check compacts its reporting/plots past a
handful of conditions rather than printing one row per draw.

Usage:
    python generate_toy_prior.py [path/to/config.yaml]
    (defaults to configs/toy_prior_config.yaml next to this script)
"""
import math
import os
import sys

import numpy as np
import plotly.graph_objects as go
import torch

# no sys.path manipulation here, matching train.py -- run with gfn_diffusion/
# on PYTHONPATH so `energy_sampling` resolves as a package, same as training
from energy_sampling.utils import dict2namespace, load_yaml


def load_template(path):
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict):
        for key in ('prior', 'equalized_prior'):
            if key in data:
                data = data[key]
                break
    data.reset_sg_info(1)
    return data


def replicate(base_batch, n):
    """
    Subsamples n graphs (with replacement) from an existing, already-valid
    batch to use as crystal-structure scaffolding for n new samples.

    Deliberately NOT built by repeating a single index
    (base_batch.subsample_new_batch([0]*n)) -- confirmed by direct testing
    that some per-graph fields (aunit_handedness in particular) don't
    replicate correctly under a same-index-repeated subsample and blow up
    downstream in latent_to_cell_params/analyze; a genuinely-varied index
    array (even drawn with replacement from a small base) doesn't hit this.

    n is clamped to >= 2: mxtaltools' append_batch (used later to
    concatenate per-condition sub-batches) treats a field whose leading
    dimension is exactly 1 (e.g. T_cf, shape [1,3,3]) as shared/global
    metadata rather than genuine one-graph data, and raises if two such
    single-graph groups' values differ -- confirmed by direct testing: fails
    at group size 1, fine at 2+. A condition_set/prior built with only 1
    replica per condition therefore can't be concatenated across conditions.
    """
    if n < 2:
        print(f"generate_toy_prior: n={n} replicas requested, clamping to 2 -- "
              f"append_batch can't safely concatenate single-graph groups (see replicate() docstring)")
        n = 2
    idx = np.random.choice(base_batch.num_graphs, n, replace=True)
    rep = base_batch.subsample_new_batch(idx)
    # crystal-system-dependent caches need refreshing after subsampling,
    # same as the original script's reset_sg_info(1) call
    rep.reset_sg_info(torch.full((n,), int(rep.sg_ind[0]), dtype=torch.long))
    return rep


def latent_field_kwargs(cfg):
    """Extract the optional `latent_field` config block as a plain kwargs
    dict for mxtaltools' _build_latent_field, forwarded through every
    latent_multiharmonic call site below. Empty if the block isn't
    configured (mxtaltools' own defaults apply)."""
    block = getattr(cfg, 'latent_field', None)
    if block is None:
        return {}
    return {k: v for k, v in vars(block).items() if v is not None}


def sample_condition_latents(base_batch, energy_function, c, width, n, target_temperature,
                             field_kwargs=None):
    c_t = torch.as_tensor(c, dtype=torch.float32)
    if energy_function == 'latent_harmonic':
        x = base_batch.sample_latent_harmonic(
            n_samples=n, c=c_t[None], width=width, target_temperature=target_temperature)
    elif energy_function == 'latent_multiharmonic':
        x = base_batch.sample_latent_multiharmonic(
            n_samples=n, c=c_t, width=width, target_temperature=target_temperature,
            **(field_kwargs or {}))
    else:
        raise ValueError(f"'{energy_function}' is not a toy energy function")
    eps = 1e-3
    return x.clamp(min=-1 + eps, max=1 - eps)


def condition_log_partition(base_batch, energy_function, c, width, target_temperature, d, field_kwargs=None):
    """log-normalizer log Z of the target distribution defined by (c, width, T)
    -- the x-independent term factored out of condition_log_density, reused
    directly by print_partition_functions."""
    c_t = torch.as_tensor(c, dtype=torch.float32)
    if energy_function == 'latent_harmonic':
        return 0.5 * d * math.log(2 * math.pi * target_temperature) + d * math.log(width)
    else:
        return base_batch.log_partition_latent(c=c_t, target_temperature=target_temperature, width=width,
                                                **(field_kwargs or {}))


def condition_log_density(base_batch, energy_function, c, width, x, target_temperature,
                          field_kwargs=None):
    """Normalized log-density of the target/prior distribution defined by
    (c, width) at points x -- both toy energy functions have a closed-form
    normalizer, so no self-normalization/MC estimate is needed here."""
    c_t = torch.as_tensor(c, dtype=torch.float32)
    if energy_function == 'latent_harmonic':
        energy = base_batch.latent_harmonic_en(c=c_t[None], width=width, x=x)
    else:
        energy = base_batch.latent_multiharmonic_en(c=c_t, width=width, x=x, **(field_kwargs or {}))
    log_z = condition_log_partition(base_batch, energy_function, c, width, target_temperature, x.shape[-1],
                                    field_kwargs=field_kwargs)
    return -energy / target_temperature - log_z


def print_partition_functions(cfg, template, conditions, field_kwargs=None):
    """Prints each condition's target-distribution partition function
    (log Z and Z, evaluated at that condition's own (c, width) under
    prior.target_temperature) -- callers pass the originally configured
    `conditions` list, not any condition_manifold-expanded draws."""
    d = template.latent_params().shape[-1]
    T = cfg.prior.target_temperature
    print(f"Partition functions per condition (target_temperature={T}):")
    for cond in conditions:
        log_z = float(condition_log_partition(template, cfg.energy_function, cond.c, float(cond.width), T, d,
                                              field_kwargs=field_kwargs))
        try:
            z = f'{math.exp(log_z):.6g}'
        except OverflowError:
            z = 'inf' if log_z > 0 else '0'
        print(f"  [{cond.identifier}] log Z = {log_z:.6g}   Z = {z}")


def shared_prior_center_width(cfg, d):
    center = torch.zeros(d) if getattr(cfg.prior, 'shared_center', None) is None \
        else torch.as_tensor(cfg.prior.shared_center, dtype=torch.float32)
    return center, float(cfg.prior.shared_width)


def draw_prior_samples(cfg, template, cond, n, field_kwargs=None):
    """Mode-dispatching prior draw -- see build_prior's docstrings for what
    each mode means. Shared/dispatched with coverage_check so its ESS check
    always matches whatever prior.mode actually produced the saved file."""
    mode = getattr(cfg.prior, 'mode', 'shared')
    if mode == 'shared':
        d = template.latent_params().shape[-1]
        center, width = shared_prior_center_width(cfg, d)
        eps = 1e-3
        return (center[None] + width * torch.randn(n, d)).clamp(min=-1 + eps, max=1 - eps)
    else:
        prior_width = float(cond.width) * cfg.prior.width_multiplier
        return sample_condition_latents(
            template, cfg.energy_function, cond.c, prior_width, n, cfg.prior.target_temperature,
            field_kwargs=field_kwargs)


def prior_log_density(cfg, template, cond, x, field_kwargs=None):
    """Mode-dispatching prior log-density at points x, paired with
    draw_prior_samples -- see its docstring."""
    mode = getattr(cfg.prior, 'mode', 'shared')
    if mode == 'shared':
        d = x.shape[-1]
        center, width = shared_prior_center_width(cfg, d)
        sq = ((x - center[None]) / width).pow(2).sum(-1)
        log_z = 0.5 * d * math.log(2 * math.pi) + d * math.log(width)
        return -0.5 * sq - log_z
    else:
        prior_width = float(cond.width) * cfg.prior.width_multiplier
        return condition_log_density(
            template, cfg.energy_function, cond.c, prior_width, x, cfg.prior.target_temperature,
            field_kwargs=field_kwargs)


def expand_condition_manifold(cfg, anchors):
    """
    Replaces a small set of named anchor conditions with a much larger set of
    draws -- for testing generalization over a continuous conditional region
    rather than at N fixed discrete modes.

    Every draw gets its OWN unique identifier ({condition_manifold.identifier}
    _{i}), because train.py resolves condition identity purely through the
    identifier string: init_identifiers() maps distinct identifiers to dense
    mol_ids, and condition_samples() builds condition_id from mol_id alone --
    the c vector itself never enters the ID. Draws sharing one identifier
    would therefore collapse into a single condition_id and smear every
    manifold point's logZ/best-energy bookkeeping (ConditionLogZTracker,
    anchor buffers) into one slot. The cost of unique identifiers is sparse
    per-condition statistics -- each tracker slot only accumulates evidence
    when its exact condition is drawn -- which we accept for now (revisit if
    the manifold grows to many thousands of draws). Anchors keep their own
    configured identifiers.

    Three modes (condition_manifold.mode):
      'interpolate' (default) -- each draw's (c, width) is a random convex
        combination of the anchors' own (c, width) -- weights ~
        Dirichlet(1,...,1) over the K anchors, which is uniform over the
        simplex and reduces to uniform-on-a-line-segment for K=2 --
        optionally broadened by isotropic Gaussian noise
        (condition_manifold.noise_std) to also probe just outside the
        anchors' convex hull, not only strictly between them.
      'gaussian' -- c ~ center + scale * N(0, I): plain isotropic Gaussian
        noise, no connectivity structure at all.
      'uniform' -- c ~ Uniform over the axis-aligned hypercube of full side
        length `scale` centered on `center`.
    In the noise modes anchors are only used to infer cond_dim / the default
    draw width, so a single dummy anchor entry in `conditions` is enough
    (build_condition_set/build_prior only ever see the expanded list, plus
    the anchors themselves iff include_anchors).
    """
    m = cfg.condition_manifold
    mode = getattr(m, 'mode', 'interpolate')
    n = int(m.n_conditions)
    K = len(anchors)

    anchor_c = torch.stack([torch.as_tensor(a.c, dtype=torch.float32) for a in anchors])
    anchor_w = torch.as_tensor([float(a.width) for a in anchors], dtype=torch.float32)

    if mode == 'interpolate':
        if K < 2:
            raise ValueError("condition_manifold mode 'interpolate' needs >= 2 anchors in `conditions`")
        weights = torch.distributions.Dirichlet(torch.ones(K)).sample((n,))  # [n, K], uniform on simplex
        c = weights @ anchor_c      # [n, cond_dim]
        width = weights @ anchor_w  # [n]

        noise_std = float(getattr(m, 'noise_std', 0.0) or 0.0)
        if noise_std > 0:
            c = c + noise_std * torch.randn_like(c)
    elif mode in ('gaussian', 'uniform'):
        d = int(getattr(m, 'cond_dim', None) or anchor_c.shape[-1])
        center = torch.zeros(d) if getattr(m, 'center', None) is None \
            else torch.as_tensor(m.center, dtype=torch.float32)
        scale = float(m.scale)
        if mode == 'gaussian':
            c = center[None] + scale * torch.randn(n, d)  # scale = std
        else:
            c = center[None] + scale * (torch.rand(n, d) - 0.5)  # hypercube of full width `scale`
        # noise modes have no per-draw width structure -- one shared width for
        # every draw, defaulting to the anchors' mean if not configured
        draw_width = float(getattr(m, 'draw_width', None) or anchor_w.mean())
        width = torch.full((n,), draw_width)
    else:
        raise ValueError(f"condition_manifold.mode must be 'interpolate', 'gaussian' or 'uniform', got {mode!r}")

    base = m.identifier
    draws = [dict2namespace({'identifier': f'{base}_{i:04d}', 'c': c[i].tolist(), 'width': width[i].item()})
             for i in range(n)]

    if getattr(m, 'include_anchors', True):
        # also keep the exact anchor points themselves in the set (under their
        # own configured identifiers), so the manifold's boundary is guaranteed
        # to be covered/reported, not just its randomly-sampled interior
        anchor_draws = [dict2namespace({'identifier': a.identifier, 'c': a.c, 'width': float(a.width)})
                         for a in anchors]
        draws = anchor_draws + draws

    print(f"condition_manifold ({mode}): expanded {K} anchors -> {len(draws)} conditions, "
          f"each with its own unique identifier ('{base}_NNNN' + anchor names)")
    return draws


def append_all(parts):
    batch = parts[0]
    for part in parts[1:]:
        batch = batch.append_batch(part)
    return batch


def build_condition_set(cfg, template, field_kwargs=None):
    """
    See the module docstring: with condition_set.sample_latents (default
    true) each condition's replicas carry latents drawn from that condition's
    OWN target distribution, plus baked energy, so the file is directly
    usable as a per-condition on-mode seed set (anchor_buffer seed_source)
    and not just molecules_path metadata.

    For latent_multiharmonic the whole build is vectorized like
    _build_prior_shared: ONE replicate() for every graph + ONE
    latent_to_cell_params/analyze(), with per-condition c/width/identifier
    laid out by repeat_interleave. The old per-condition replicate() +
    append_all(parts) was O(n_conditions^2) (each append_batch
    re-concatenates the growing batch) and dominated runtime on big
    condition_manifold runs -- this removes it. Only the target SAMPLING
    itself stays a per-condition loop: multiharmonic's SAMPLING-side helpers
    are single-shared-c (see the module docstring's ASYMMETRY note), but that
    loop is linear and cheap. latent_harmonic keeps the fully per-condition
    build + append_all -- its energy silently uses only c[0], so it can't be
    baked more than one condition per analyze() call.
    """
    sample_latents = getattr(cfg.condition_set, 'sample_latents', True)
    n_rep = cfg.condition_set.n_replicas_per_condition

    if cfg.energy_function == 'latent_multiharmonic':
        # ONE replicate() + ONE latent_to_cell_params/analyze for the whole
        # set, mirroring _build_prior_shared's vectorized path -- the old
        # per-condition replicate + append_all(parts) was O(n_conditions^2)
        # in append_batch (each call re-concatenates the growing batch) and
        # dominated runtime on big condition_manifold runs. Metadata is laid
        # out per condition by repeat_interleave; multiharmonic's energy bake
        # broadcasts per-sample c/width in one analyze() call (see the module
        # docstring's ASYMMETRY note). Only the target SAMPLING itself stays a
        # per-condition loop (its single-shared-c constraint), but that's
        # linear and cheap next to what append_all cost.
        n_rep = max(2, n_rep)  # replicate()'s own clamp -- keep the layout exact
        n_conditions = len(cfg.conditions)
        n_total = n_rep * n_conditions
        c = torch.stack([torch.as_tensor(cond.c, dtype=torch.float32) for cond in cfg.conditions]) \
            .repeat_interleave(n_rep, dim=0)
        widths = torch.as_tensor([float(cond.width) for cond in cfg.conditions], dtype=torch.float32) \
            .repeat_interleave(n_rep)
        batch = replicate(template, n_total)
        batch.add_graph_attr(c, 'c')
        batch.add_graph_attr(widths, 'width')
        batch.identifier = [cond.identifier for cond in cfg.conditions for _ in range(n_rep)]
        if sample_latents:
            x_all = torch.cat([
                sample_condition_latents(template, cfg.energy_function, cond.c, float(cond.width),
                                         n_rep, cfg.prior.target_temperature, field_kwargs=field_kwargs)
                for cond in cfg.conditions])
            batch.latent_to_cell_params(x_all)
            batch.analyze([cfg.energy_function], c=batch.c, width=batch.width, assign_outputs=True,
                          **(field_kwargs or {}))
        return batch

    # latent_harmonic: c[0]-only energy forces a per-condition analyze()
    # (see the module docstring's ASYMMETRY note), so it keeps the original
    # per-condition build + append_all.
    parts = []
    for cond in cfg.conditions:
        # replicate() may return more graphs than requested (see its
        # docstring -- group size is clamped to >= 2), so every subsequent
        # per-graph tensor here must be sized off sub.num_graphs, not the
        # originally-requested count
        sub = replicate(template, n_rep)
        n = sub.num_graphs
        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        if sample_latents:
            x = sample_condition_latents(
                template, cfg.energy_function, cond.c, float(cond.width), n, cfg.prior.target_temperature)
            sub.latent_to_cell_params(x)
            sub.analyze([cfg.energy_function], c=sub.c, width=float(cond.width), assign_outputs=True)
        parts.append(sub)
    return append_all(parts)


def prior_samples_per_condition(cfg, total_key, legacy_per_condition_key):
    """
    Resolves the per-condition sample count build_prior actually needs from a
    TOTAL budget (total_key, split evenly across cfg.conditions -- the natural
    knob now that a condition_manifold run multiplies len(cfg.conditions) by
    hundreds), falling back to the old per-condition key if the total isn't
    configured. Per-condition counts are floored at 2 to match replicate()'s
    append_batch clamp (see its docstring), so the realized total can exceed
    the requested one when total_key < 2 * len(cfg.conditions).
    """
    n_conditions = len(cfg.conditions)
    total = getattr(cfg.prior, total_key, None)
    if total is not None:
        n = max(2, math.ceil(int(total) / n_conditions))
        print(f"prior: {total_key}={total} over {n_conditions} conditions -> "
              f"{n} samples per condition ({n * n_conditions} realized total)")
        return n
    legacy = getattr(cfg.prior, legacy_per_condition_key, None)
    if legacy is None:
        raise ValueError(f"prior config needs either {total_key} (preferred, a total split "
                         f"across conditions) or {legacy_per_condition_key}")
    print(f"prior: {total_key} not set, using legacy {legacy_per_condition_key}={legacy} "
          f"-> {int(legacy) * n_conditions} total over {n_conditions} conditions")
    return int(legacy)


def build_prior(cfg, template, n_per_condition, field_kwargs=None):
    mode = getattr(cfg.prior, 'mode', 'shared')
    if mode == 'shared':
        return _build_prior_shared(cfg, template, n_per_condition, field_kwargs=field_kwargs)
    elif mode == 'per_condition':
        return _build_prior_per_condition(cfg, template, n_per_condition, field_kwargs=field_kwargs)
    else:
        raise ValueError(f"prior.mode must be 'shared' or 'per_condition', got {mode!r}")


def _build_prior_shared(cfg, template, n_per_condition, field_kwargs=None):
    """
    ONE zero-centered (prior.shared_center) Gaussian, wide enough
    (prior.shared_width) to cover every configured condition, with no
    knowledge of which condition a given draw will end up labeled with --
    the actual "broad unconditional prior".

    Still has to bake energy under exactly one condition per analyze() call
    (see module docstring's ASYMMETRY note), so after drawing everything
    from the one shared Gaussian, this splits the draws into n_conditions
    equal chunks and labels each chunk with a different condition for
    baking -- the LATENT POSITIONS are shared/unconditional; only the
    identifier/c/width label and resulting baked energy differ per chunk.

    n_per_condition is clamped to >= 2 upfront (matching replicate()'s own
    clamp -- see its docstring) so each chunk's sample count always exactly
    matches what replicate() actually returns for that chunk.

    Since the latent positions are label-independent by construction, the
    per-condition loop here is pure bookkeeping -- so for
    latent_multiharmonic the whole build collapses to ONE replicate() + ONE
    analyze() call: latent_multiharmonic_en broadcasts per-sample
    [n, cond_dim] `c` (see the module docstring's ASYMMETRY note) AND
    per-sample [n] `width` (its own docstring: "width may be scalar or a
    per-sample tensor [B]"), and analyze() passes both straight through
    (compute() just forwards **kwargs). The single-shared-c / scalar-width
    restriction lives only in sample_latent_multiharmonic /
    log_partition_latent, which shared mode never calls. latent_harmonic
    keeps the per-condition loop (it silently uses only c[0] for the whole
    batch).
    """
    n_per_condition = max(2, n_per_condition)
    n_conditions = len(cfg.conditions)
    d = template.latent_params().shape[-1]
    center, width = shared_prior_center_width(cfg, d)

    eps = 1e-3
    x_all = (center[None] + width * torch.randn(n_per_condition * n_conditions, d)).clamp(min=-1 + eps, max=1 - eps)

    if cfg.energy_function == 'latent_multiharmonic':
        n_total = n_per_condition * n_conditions
        c = torch.stack([torch.as_tensor(cond.c, dtype=torch.float32) for cond in cfg.conditions]) \
            .repeat_interleave(n_per_condition, dim=0)
        widths = torch.as_tensor([float(cond.width) for cond in cfg.conditions], dtype=torch.float32) \
            .repeat_interleave(n_per_condition)
        print(f"prior (shared, vectorized): {n_conditions} conditions baked in one analyze() call")
        sub = replicate(template, n_total)
        sub.add_graph_attr(c, 'c')
        sub.add_graph_attr(widths, 'width')
        sub.identifier = [cond.identifier for cond in cfg.conditions for _ in range(n_per_condition)]
        sub.latent_to_cell_params(x_all)
        # bake energy under each condition's own (narrow) width -- not the
        # shared prior's width -- prebuilt_sample_to_reward needs the real
        # target energy
        sub.analyze([cfg.energy_function], c=sub.c, width=sub.width, assign_outputs=True,
                    **(field_kwargs or {}))
        return sub

    parts = []
    for i, cond in enumerate(cfg.conditions):
        x = x_all[i * n_per_condition:(i + 1) * n_per_condition]
        sub = replicate(template, n_per_condition)
        n = sub.num_graphs
        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        sub.latent_to_cell_params(x)
        # bake energy under the condition's own (narrow) width -- see above
        sub.analyze([cfg.energy_function], c=sub.c, width=float(cond.width), assign_outputs=True)
        parts.append(sub)
    return append_all(parts)


def _build_prior_per_condition(cfg, template, n_per_condition, field_kwargs=None):
    """
    For later: each condition gets its own Gaussian CENTERED on that
    condition's own `c` but scaled up by prior.width_multiplier, so
    q_target << q_prior holds locally, per-condition, by construction.
    """
    parts = []
    for cond in cfg.conditions:
        sub = replicate(template, n_per_condition)
        n = sub.num_graphs  # see build_condition_set's comment on why this, not n_per_condition
        prior_width = float(cond.width) * cfg.prior.width_multiplier
        x = sample_condition_latents(
            template, cfg.energy_function, cond.c, prior_width, n, cfg.prior.target_temperature,
            field_kwargs=field_kwargs)

        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        sub.latent_to_cell_params(x)
        # bake energy under the condition's own (narrow) width, not the
        # broadened prior_width used only to position these samples --
        # prebuilt_sample_to_reward needs the real target energy
        sub.analyze([cfg.energy_function], c=sub.c, width=float(cond.width), assign_outputs=True,
                    **(field_kwargs or {}))
        parts.append(sub)
    return append_all(parts)


def coverage_check(cfg, template, field_kwargs=None):
    """
    Direct, quantitative form of "q_target << q_prior": for each condition,
    draws prior samples (mode-dispatched -- see draw_prior_samples/build_prior
    for what 'shared' vs 'per_condition' mean) and computes the
    self-normalized importance-sampling effective sample size (ESS) fraction
    of the TARGET density under that prior. Low ESS means the prior rarely
    produces samples where the target actually has mass -- backward training
    seeded from it will struggle.

    n_is_samples is a TOTAL budget, split evenly across all conditions --
    every condition now carries its own unique identifier (see
    expand_condition_manifold), so a per-condition budget would scale the
    check's cost linearly with manifold size. Raise n_is_samples if you want
    more IS samples per condition on a big manifold. Reporting/plotting is
    also compacted past MANY_CONDITIONS conditions: a summary line
    (min/median ESS, worst offenders) instead of one line each, and two
    pooled histogram traces instead of two per condition.

    The histograms plot, for both prior-drawn and genuine target-drawn
    samples, the negative log target density ("energy under target") -- NOT
    a Euclidean distance to `c`, since `c` doesn't live in latent space for
    latent_multiharmonic (it's a separate, independently-dimensioned
    condition-embedding input to mxtaltools' Gaussian-mixture field, not a
    mode center itself), so this is the one comparison that's meaningful and
    dimension-safe for both toy energy functions. Prior-drawn samples should
    sit at much higher (more atypical) energy under the target than genuine
    target samples if coverage is healthy but not degenerate.
    """
    MANY_CONDITIONS = 8
    T = cfg.prior.target_temperature
    # the per-condition pipeline below (2 sampling calls + 2 density
    # evaluations, each with fixed mxtaltools overhead) makes the check's cost
    # linear in condition count, so on big manifold/noise runs check a random
    # subset -- a few hundred conditions with a real per-condition IS budget
    # says the same thing about coverage as all of them with 2 samples each
    conditions = list(cfg.conditions)
    n_check = getattr(cfg.coverage_check, 'n_check_conditions', None)
    if n_check is not None and len(conditions) > int(n_check):
        idx = np.random.choice(len(conditions), int(n_check), replace=False)
        conditions = [conditions[i] for i in idx]
        print(f"coverage_check: subsampling {len(conditions)} of {len(cfg.conditions)} conditions "
              f"(coverage_check.n_check_conditions)")
    n_conditions = len(conditions)
    n_per = max(2, cfg.coverage_check.n_is_samples // n_conditions)
    compact = n_conditions > MANY_CONDITIONS
    results = {}
    fig = go.Figure()
    all_prior_x, all_target_x = [], []
    neg_log_target_prior, neg_log_target_target = [], []

    for cond in conditions:
        x = draw_prior_samples(cfg, template, cond, n_per, field_kwargs=field_kwargs)
        log_prior = prior_log_density(cfg, template, cond, x, field_kwargs=field_kwargs)
        log_target = condition_log_density(template, cfg.energy_function, cond.c, float(cond.width), x, T,
                                           field_kwargs=field_kwargs)
        all_prior_x.append(x)
        neg_log_target_prior.append(-log_target)

        target_x = sample_condition_latents(template, cfg.energy_function, cond.c, float(cond.width), n_per, T,
                                            field_kwargs=field_kwargs)
        log_target_at_target = condition_log_density(template, cfg.energy_function, cond.c, float(cond.width),
                                                      target_x, T, field_kwargs=field_kwargs)
        all_target_x.append(target_x)
        neg_log_target_target.append(-log_target_at_target)

        log_w = log_target - log_prior
        n = log_w.shape[0]
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        ess_frac = (1.0 / (n * (log_w.exp() ** 2).sum())).item()
        results[cond.identifier] = ess_frac

        if not compact:
            status = 'OK' if ess_frac >= cfg.coverage_check.min_ess_frac \
                else 'WARNING -- prior may not cover this condition'
            print(f"[{cond.identifier}] q_target ESS fraction under prior: {ess_frac:.4g}  ({status})")
            fig.add_trace(go.Histogram(x=(-log_target).numpy(), nbinsx=60, opacity=0.5,
                                       histnorm='probability density', name=f'prior samples: {cond.identifier}'))
            fig.add_trace(go.Histogram(x=(-log_target_at_target).numpy(), nbinsx=60, opacity=0.5,
                                       histnorm='probability density', name=f'target samples: {cond.identifier}'))

    if compact:
        ess = torch.tensor(list(results.values()))
        n_below = int((ess < cfg.coverage_check.min_ess_frac).sum())
        worst = sorted(results.items(), key=lambda kv: kv[1])[:5]
        print(f"coverage_check: {n_conditions} conditions, {n_per} IS samples each -- "
              f"ESS fraction min {ess.min():.4g} / median {ess.median():.4g} / max {ess.max():.4g}; "
              f"{n_below} below min_ess_frac={cfg.coverage_check.min_ess_frac}")
        if n_below:
            print("  worst: " + ", ".join(f"{k}={v:.3g}" for k, v in worst))
        fig.add_trace(go.Histogram(x=torch.cat(neg_log_target_prior).numpy(), nbinsx=60, opacity=0.5,
                                   histnorm='probability density', name=f'prior samples ({n_conditions} conditions)'))
        fig.add_trace(go.Histogram(x=torch.cat(neg_log_target_target).numpy(), nbinsx=60, opacity=0.5,
                                   histnorm='probability density', name=f'target samples ({n_conditions} conditions)'))
        # pool the latent-space aux plots the same way -- one prior cloud, one
        # target cloud, instead of 2 * n_conditions separate distributions
        all_prior_x = [torch.cat(all_prior_x)]
        all_target_x = [torch.cat(all_target_x)]
    fig.update_layout(
        barmode='overlay',
        title="Prior-drawn vs. target-drawn samples, scored by -log(target density)",
        xaxis_title='-log q_target(x)', yaxis_title='density')

    '''
    
    clats = torch.cat(all_target_x)
    cbatch = template.subsample_new_batch(torch.tensor([0 for _ in range(len(clats))]))
    cbatch.reset_sg_info(1)
    cbatch.latent_to_cell_params(clats)
    cbatch.plot_batch_cell_params(space='latent', ref_dist=all_prior_x[0])
    cbatch.plot_batch_staircase(space='latent')
    
    '''

    if cfg.coverage_check.show_figures:
        fig.show()
    if cfg.coverage_check.save_figures:
        out_path = os.path.join(cfg.output_dir, f'{cfg.tag}_coverage_check.html')
        fig.write_html(out_path)
        print(f'Saved coverage figure -> {out_path}')

    return results


def main(config_path):
    cfg = dict2namespace(load_yaml(config_path))
    # dict2namespace only recurses into nested dicts, not dicts inside a
    # list -- cfg.conditions is a list, so its entries stay plain dicts
    # unless converted here
    cfg.conditions = [dict2namespace(c) if isinstance(c, dict) else c for c in cfg.conditions]
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    anchors = list(cfg.conditions)  # preserved for print_partition_functions below, pre-manifold-expansion

    # optional: replace the discrete anchor conditions above with a much
    # larger set of draws spanning the connected region between them -- must
    # run before build_condition_set/build_prior/coverage_check below, which
    # all just iterate cfg.conditions and don't otherwise know anchors from
    # manifold draws -- see expand_condition_manifold's docstring for why
    # these share ONE identifier rather than getting their own
    if getattr(cfg, 'condition_manifold', None) is not None and getattr(cfg.condition_manifold, 'enabled', False):
        cfg.conditions = expand_condition_manifold(cfg, cfg.conditions)

    os.makedirs(cfg.output_dir, exist_ok=True)

    template = load_template(cfg.template_path)
    field_kwargs = latent_field_kwargs(cfg)

    print_partition_functions(cfg, template, anchors, field_kwargs=field_kwargs)

    if getattr(cfg.condition_set, 'generate', True):
        cond_batch = build_condition_set(cfg, template, field_kwargs=field_kwargs)
        out_name = getattr(cfg.condition_set, 'output_name', None) or f'{cfg.tag}_conditions.pt'
        out_path = os.path.join(cfg.output_dir, out_name)
        del cond_batch.fingerprint
        torch.save({'prior': cond_batch}, out_path)
        cond_batch.plot_batch_cell_params(space='latent')
        cond_batch.plot_batch_staircase(space='latent')
        print(f"Saved condition set -> {out_path} "
              f"({cond_batch.num_graphs} graphs, {len(cfg.conditions)} unique identifiers)")

    if getattr(cfg.prior, 'generate', True):
        prior_batch = build_prior(
            cfg, template, prior_samples_per_condition(cfg, 'n_samples_total', 'n_samples_per_condition'),
            field_kwargs=field_kwargs)
        eq_batch = build_prior(
            cfg, template,
            prior_samples_per_condition(cfg, 'equalized_n_samples_total', 'equalized_n_samples_per_condition'),
            field_kwargs=field_kwargs)
        out_name = getattr(cfg.prior, 'output_name', None) or f'{cfg.tag}_prior.pt'
        out_path = os.path.join(cfg.output_dir, out_name)
        del prior_batch.fingerprint, eq_batch.fingerprint
        torch.save({'prior': prior_batch, 'equalized_prior': eq_batch}, out_path)
        print(f"Saved prior -> {out_path} ({prior_batch.num_graphs} / {eq_batch.num_graphs} graphs)")

    if getattr(cfg.coverage_check, 'run', True):
        coverage_check(cfg, template, field_kwargs=field_kwargs)


if __name__ == '__main__':
    default_config = os.path.join(os.path.dirname(__file__), 'configs', 'toy_prior_config.yaml')
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    main(config_path)
