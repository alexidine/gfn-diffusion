"""
Config-driven generator for toy (latent_harmonic / latent_multiharmonic)
condition sets and priors.

Produces two independent artifacts, matching the two roles these files play
in training (see train.py's init_mol_dataset / init_prior_dataset):

  - condition set (molecules_path-type): one entry per configured condition,
    carrying `identifier` / `c` / `width` plus ordinary crystal-structure
    scaffolding borrowed from an existing template file. No baked energy --
    forward sampling always recomputes energy fresh from whatever latents
    the GFN actually produces (see analyze_crystal_batch's per-sample `c`
    override), so the scaffolding's starting cell params don't matter, only
    the identifier/c/width metadata and graph content do.

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
    correctly (confirmed via its 'kdc,...c->...kd' einsum).
  Both build_condition_set and build_prior sidestep this by construction --
  every analyze()/sample_* call here only ever contains one condition, then
  the per-condition batches are concatenated -- so this is safe either way,
  but latent_harmonic remains unsafe if you batch multiple conditions into
  one analyze() call anywhere else.

Also note: multiharmonic's condition vectors are NOT tied to the latent
space's dimensionality (data_ndim) -- they control mxtaltools'
_build_latent_field's Gaussian-mixture field and default to 8-dim
(_build_latent_field(cond_dim=8)) regardless of data_ndim. Match your `c`
vectors' length to whatever cond_dim you're using (8, if left at the
default).

Optional condition_manifold (see expand_condition_manifold): instead of N
discrete named conditions, expands the `conditions` list (treated as
anchors) into a much larger set of unnamed draws spanning the connected
region between them -- random convex combinations of the anchors' own
(c, width), optionally jittered -- for testing generalization over a
continuous manifold rather than at fixed modes. All draws share one
identifier, so coverage_check reports/plots the whole manifold as a single
pooled distribution rather than one row per draw.

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


def sample_condition_latents(base_batch, energy_function, c, width, n, target_temperature):
    c_t = torch.as_tensor(c, dtype=torch.float32)
    if energy_function == 'latent_harmonic':
        x = base_batch.sample_latent_harmonic(
            n_samples=n, c=c_t[None], width=width, target_temperature=target_temperature)
    elif energy_function == 'latent_multiharmonic':
        x = base_batch.sample_latent_multiharmonic(
            n_samples=n, c=c_t, width=width, target_temperature=target_temperature)
    else:
        raise ValueError(f"'{energy_function}' is not a toy energy function")
    eps = 1e-3
    return x.clamp(min=-1 + eps, max=1 - eps)


def condition_log_density(base_batch, energy_function, c, width, x, target_temperature):
    """Normalized log-density of the target/prior distribution defined by
    (c, width) at points x -- both toy energy functions have a closed-form
    normalizer, so no self-normalization/MC estimate is needed here."""
    c_t = torch.as_tensor(c, dtype=torch.float32)
    if energy_function == 'latent_harmonic':
        energy = base_batch.latent_harmonic_en(c=c_t[None], width=width, x=x)
        d = x.shape[-1]
        log_z = 0.5 * d * math.log(2 * math.pi * target_temperature) + d * math.log(width)
    else:
        energy = base_batch.latent_multiharmonic_en(c=c_t, width=width, x=x)
        log_z = base_batch.log_partition_latent(c=c_t, target_temperature=target_temperature, width=width)
    return -energy / target_temperature - log_z


def shared_prior_center_width(cfg, d):
    center = torch.zeros(d) if getattr(cfg.prior, 'shared_center', None) is None \
        else torch.as_tensor(cfg.prior.shared_center, dtype=torch.float32)
    return center, float(cfg.prior.shared_width)


def draw_prior_samples(cfg, template, cond, n):
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
            template, cfg.energy_function, cond.c, prior_width, n, cfg.prior.target_temperature)


def prior_log_density(cfg, template, cond, x):
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
            template, cfg.energy_function, cond.c, prior_width, x, cfg.prior.target_temperature)


def expand_condition_manifold(cfg, anchors):
    """
    Replaces a small set of named anchor conditions with a much larger set of
    unnamed draws spanning the connected region between them -- for testing
    generalization over a continuous conditional manifold rather than at N
    fixed discrete modes.

    Every draw shares ONE identifier (condition_manifold.identifier) instead
    of getting its own: train.py's init_identifiers() builds an
    identifier_registry sized to however many distinct identifiers exist
    across molecules_path/prior_path, and buffer.py's ConditionLogZTracker
    preallocates its EMA/best-energy tables to that same size, one slot per
    identifier -- so giving every manifold draw a unique identifier would
    both explode that library and leave each slot with ~1 visit, defeating
    the tracker's whole point. One shared identifier keeps condition_id
    small and lets per-condition bookkeeping accumulate evidence across the
    whole manifold, same as it would for a single ordinary discrete
    condition. coverage_check pools its ESS check and histograms by
    identifier for the same reason -- the manifold is reported as ONE
    distribution ("does the prior cover this connected region at all"),
    not as many separate per-point checks.

    Each draw's (c, width) is a random convex combination of the anchors'
    own (c, width) -- weights ~ Dirichlet(1,...,1) over the K anchors, which
    is uniform over the simplex and reduces to uniform-on-a-line-segment for
    K=2 -- optionally broadened by isotropic Gaussian noise
    (condition_manifold.noise_std) to also probe just outside the anchors'
    convex hull, not only strictly between them.
    """
    m = cfg.condition_manifold
    K = len(anchors)
    if K < 2:
        raise ValueError("condition_manifold needs >= 2 anchors in `conditions` to interpolate between")

    anchor_c = torch.stack([torch.as_tensor(a.c, dtype=torch.float32) for a in anchors])
    anchor_w = torch.as_tensor([float(a.width) for a in anchors], dtype=torch.float32)

    n = int(m.n_conditions)
    weights = torch.distributions.Dirichlet(torch.ones(K)).sample((n,))  # [n, K], uniform on simplex
    c = weights @ anchor_c      # [n, cond_dim]
    width = weights @ anchor_w  # [n]

    noise_std = float(getattr(m, 'noise_std', 0.0) or 0.0)
    if noise_std > 0:
        c = c + noise_std * torch.randn_like(c)

    identifier = m.identifier
    draws = [dict2namespace({'identifier': identifier, 'c': c[i].tolist(), 'width': width[i].item()})
             for i in range(n)]

    if getattr(m, 'include_anchors', True):
        # also keep the exact anchor points themselves in the set, so the
        # manifold's boundary is guaranteed to be covered/reported, not just
        # its randomly-sampled interior
        anchor_draws = [dict2namespace({'identifier': identifier, 'c': a.c, 'width': float(a.width)})
                         for a in anchors]
        draws = anchor_draws + draws

    print(f"condition_manifold: expanded {K} anchors -> {len(draws)} conditions "
          f"under shared identifier '{identifier}'")
    return draws


def append_all(parts):
    batch = parts[0]
    for part in parts[1:]:
        batch = batch.append_batch(part)
    return batch


def build_condition_set(cfg, template):
    parts = []
    for cond in cfg.conditions:
        # replicate() may return more graphs than requested (see its
        # docstring -- group size is clamped to >= 2), so every subsequent
        # per-graph tensor here must be sized off sub.num_graphs, not the
        # originally-requested count
        sub = replicate(template, cfg.condition_set.n_replicas_per_condition)
        n = sub.num_graphs
        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        parts.append(sub)
    return append_all(parts)


def build_prior(cfg, template, n_per_condition):
    mode = getattr(cfg.prior, 'mode', 'shared')
    if mode == 'shared':
        return _build_prior_shared(cfg, template, n_per_condition)
    elif mode == 'per_condition':
        return _build_prior_per_condition(cfg, template, n_per_condition)
    else:
        raise ValueError(f"prior.mode must be 'shared' or 'per_condition', got {mode!r}")


def _build_prior_shared(cfg, template, n_per_condition):
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
    """
    n_per_condition = max(2, n_per_condition)
    n_conditions = len(cfg.conditions)
    d = template.latent_params().shape[-1]
    center, width = shared_prior_center_width(cfg, d)

    eps = 1e-3
    x_all = (center[None] + width * torch.randn(n_per_condition * n_conditions, d)).clamp(min=-1 + eps, max=1 - eps)

    parts = []
    for i, cond in enumerate(cfg.conditions):
        x = x_all[i * n_per_condition:(i + 1) * n_per_condition]
        sub = replicate(template, n_per_condition)
        n = sub.num_graphs
        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        sub.latent_to_cell_params(x)
        # bake energy under the condition's own (narrow) width -- not the
        # shared prior's width -- prebuilt_sample_to_reward needs the real
        # target energy
        sub.analyze([cfg.energy_function], c=sub.c, width=float(cond.width), assign_outputs=True)
        parts.append(sub)
    return append_all(parts)


def _build_prior_per_condition(cfg, template, n_per_condition):
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
            template, cfg.energy_function, cond.c, prior_width, n, cfg.prior.target_temperature)

        sub.add_graph_attr(torch.as_tensor(cond.c, dtype=torch.float32)[None].repeat(n, 1), 'c')
        sub.add_graph_attr(torch.full((n,), float(cond.width)), 'width')
        sub.identifier = [cond.identifier] * n
        sub.latent_to_cell_params(x)
        # bake energy under the condition's own (narrow) width, not the
        # broadened prior_width used only to position these samples --
        # prebuilt_sample_to_reward needs the real target energy
        sub.analyze([cfg.energy_function], c=sub.c, width=float(cond.width), assign_outputs=True)
        parts.append(sub)
    return append_all(parts)


def coverage_check(cfg, template):
    """
    Direct, quantitative form of "q_target << q_prior": for each IDENTIFIER
    (not each cfg.conditions entry -- see below), draws n_is_samples from the
    prior (mode-dispatched -- see draw_prior_samples/build_prior for what
    'shared' vs 'per_condition' mean) and computes the self-normalized
    importance-sampling effective sample size (ESS) fraction of the TARGET
    density under that prior. Low ESS means the prior rarely produces samples
    where the target actually has mass -- backward training seeded from it
    will struggle.

    Pooled by identifier rather than iterated per cfg.conditions entry: an
    ordinary discrete run has exactly one entry per identifier, so pooling is
    a no-op and this reduces to the original per-condition check. A
    condition_manifold run (see expand_condition_manifold) has MANY entries
    sharing one identifier, each a different point in the manifold -- pooling
    their prior/target draws together before the ESS calc is exactly "treat
    the whole conditional set as a single distribution and report whether the
    prior covers it", rather than reporting (or plotting) hundreds of
    near-identical single-point rows.

    Also plots, for both prior-drawn and genuine target-drawn samples, the
    negative log target density ("energy under target") -- NOT a Euclidean
    distance to `c`, since `c` doesn't live in latent space for
    latent_multiharmonic (it's a separate, independently-dimensioned
    condition-embedding input to mxtaltools' Gaussian-mixture field, not a
    mode center itself), so this is the one comparison that's meaningful and
    dimension-safe for both toy energy functions. Prior-drawn samples should
    sit at much higher (more atypical) energy under the target than genuine
    target samples if coverage is healthy but not degenerate.
    """
    T = cfg.prior.target_temperature
    n_total = cfg.coverage_check.n_is_samples
    results = {}
    fig = go.Figure()
    all_prior_x, all_target_x = [], []

    groups = {}
    for cond in cfg.conditions:
        groups.setdefault(cond.identifier, []).append(cond)

    for identifier, group in groups.items():
        # split the identifier's sample budget evenly across however many
        # manifold points share it (1, for an ordinary discrete condition)
        n_per = max(1, n_total // len(group))

        log_w_parts, neg_log_target_prior, neg_log_target_target = [], [], []
        group_prior_x, group_target_x = [], []
        for cond in group:
            x = draw_prior_samples(cfg, template, cond, n_per)
            log_prior = prior_log_density(cfg, template, cond, x)
            log_target = condition_log_density(template, cfg.energy_function, cond.c, float(cond.width), x, T)
            log_w_parts.append(log_target - log_prior)
            neg_log_target_prior.append(-log_target)
            group_prior_x.append(x)

            target_x = sample_condition_latents(template, cfg.energy_function, cond.c, float(cond.width), n_per, T)
            log_target_at_target = condition_log_density(template, cfg.energy_function, cond.c, float(cond.width),
                                                          target_x, T)
            neg_log_target_target.append(-log_target_at_target)
            group_target_x.append(target_x)

        log_w = torch.cat(log_w_parts)
        n = log_w.shape[0]
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        ess_frac = (1.0 / (n * (log_w.exp() ** 2).sum())).item()
        results[identifier] = ess_frac

        status = 'OK' if ess_frac >= cfg.coverage_check.min_ess_frac else 'WARNING -- prior may not cover this condition'
        pooled_note = f"  (pooled over {len(group)} manifold draws)" if len(group) > 1 else ""
        print(f"[{identifier}] q_target ESS fraction under prior: {ess_frac:.4g}  ({status}){pooled_note}")

        fig.add_trace(go.Histogram(x=torch.cat(neg_log_target_prior).numpy(), nbinsx=60, opacity=0.5,
                                   histnorm='probability density', name=f'prior samples: {identifier}'))
        fig.add_trace(go.Histogram(x=torch.cat(neg_log_target_target).numpy(), nbinsx=60, opacity=0.5,
                                   histnorm='probability density', name=f'target samples: {identifier}'))
        all_prior_x.append(torch.cat(group_prior_x))
        all_target_x.append(torch.cat(group_target_x))
    fig.update_layout(
        barmode='overlay',
        title="Prior-drawn vs. target-drawn samples, scored by -log(target density)",
        xaxis_title='-log q_target(x)', yaxis_title='density')

    template.plot_batch_cell_params(space='latent', aux_dists=all_prior_x + all_target_x)
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

    if getattr(cfg.condition_set, 'generate', True):
        cond_batch = build_condition_set(cfg, template)
        out_name = getattr(cfg.condition_set, 'output_name', None) or f'{cfg.tag}_conditions.pt'
        out_path = os.path.join(cfg.output_dir, out_name)
        torch.save({'prior': cond_batch}, out_path)
        print(f"Saved condition set -> {out_path} "
              f"({cond_batch.num_graphs} graphs, {len(cfg.conditions)} unique identifiers)")

    if getattr(cfg.prior, 'generate', True):
        prior_batch = build_prior(cfg, template, cfg.prior.n_samples_per_condition)
        eq_batch = build_prior(cfg, template, cfg.prior.equalized_n_samples_per_condition)
        out_name = getattr(cfg.prior, 'output_name', None) or f'{cfg.tag}_prior.pt'
        out_path = os.path.join(cfg.output_dir, out_name)
        torch.save({'prior': prior_batch, 'equalized_prior': eq_batch}, out_path)
        print(f"Saved prior -> {out_path} ({prior_batch.num_graphs} / {eq_batch.num_graphs} graphs)")

    if getattr(cfg.coverage_check, 'run', True):
        coverage_check(cfg, template)


if __name__ == '__main__':
    default_config = os.path.join(os.path.dirname(__file__), 'configs', 'toy_prior_config.yaml')
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    main(config_path)
