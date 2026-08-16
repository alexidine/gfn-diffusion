"""
cond_workup -- matched (toy-generation, training) config pairs for working up
the conditional workflow.

Each experiment produces TWO files that must stay coupled:
  {i}_toy.yaml   -> fed to data_processing/generate_toy_prior.py, bakes the
                    {tag}_prior.pt / {tag}_conditions.pt artifacts
  {i}_train.yaml -> fed to train.py --config, points at those artifacts

The whole point of generating them together is that a handful of values MUST
be identical across the two files or the run is silently wrong:

  * the GMM field block: toy `latent_field` == training
    `energy_config.analyze_kwargs`. The prior/conditions are BAKED with this
    field; the training energy RE-SCORES against it. n_core / n_ghost / cond_dim
    (and every other field knob) therefore land in both places from one source.
  * cond_dim appears in FOUR spots (2 per file): toy latent_field.cond_dim +
    condition_manifold.cond_dim; training vector_conditioning_dim +
    analyze_kwargs.cond_dim. The anchor `c` vectors are also length cond_dim.
  * filenames chain off the toy `tag`: {tag}_prior.pt / {tag}_conditions.pt ->
    training prior_path / molecules_path / anchor_buffer.seed_source.

The TRAINING PROTOCOL (stages, LRs, controller, buffers) is held FIXED across
the battery -- it's the method under test. Only the conditioning/dataset knobs
below vary, so a failure is attributable to a knob rather than a method change.

data_ndim (latent space dimensionality) is NOT a knob here -- it's fixed by the
template .pt (template.latent_params().shape[-1]). For latent_multiharmonic the
condition vectors are independently dimensioned from the latent space (see
generate_toy_prior.py's module docstring), so cond_dim moves freely of it.

Experimental knobs (all forwarded to make_pair):
  cond_dim       latent_field.cond_dim / manifold cond_dim / vector_conditioning_dim / anchor length
  n_core         latent_field.n_core           (GMM core-mode count)
  n_ghost        latent_field.n_ghost          (GMM ghost-mode count)
  mode           condition_manifold.mode       'interpolate' | 'uniform' | 'gaussian'
  scale          condition_manifold.scale      uniform side length / gaussian std (noise modes only)
  n_conditions   condition_manifold.n_conditions
  noise_std      condition_manifold.noise_std  (interpolate only)
  n_replicas     condition_set.n_replicas_per_condition

Usage (from energy_sampling/, with the csd_mxt_gfn venv):
    python configs/cond_workup/generate_configs.py
Then run each pair sequentially -- see the emitted run_all.ps1.
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent

# repo roots for PYTHONPATH (no editable installs -- the IDE supplies these)
PYTHONPATH = (r"C:\Users\mikem\Projects\mxt_gfn\mxtaltools;"
              r"C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion")


def load_base(name):
    with (OUTDIR / name).open('r') as f:
        return yaml.safe_load(f)


def make_pair(ind, base_toy, base_train, note='',
              cond_dim=2, n_core=1, n_ghost=0,
              mode='interpolate', scale=2.0, n_conditions=1000,
              noise_std=0.1, n_replicas=4, seed=None):
    """Emit one coupled (toy, train) config pair. Returns a log dict."""
    tag = f"cw{ind:02d}"

    toy = deepcopy(base_toy)
    train = deepcopy(base_train)
    outdir = toy['output_dir']  # single source for the .pt artifact location

    # --- the shared GMM field: written to BOTH files from one dict so they
    #     can never drift (baked-vs-scored consistency) ---
    field = deepcopy(toy['latent_field'])
    field['cond_dim'] = cond_dim
    field['n_core'] = n_core
    field['n_ghost'] = n_ghost
    toy['latent_field'] = field
    train['energy_config']['analyze_kwargs'] = deepcopy(field)

    # --- anchors: two well-separated cond_dim-length modes (generalizes the
    #     baseline's [+1,+1]/[-1,-1]); interpolate draws convex combos of these,
    #     uniform/gaussian only use them for cond_dim / default width ---
    toy['conditions'] = [
        {'identifier': 'mode_a', 'c': [1.0] * cond_dim, 'width': 1.0},
        {'identifier': 'mode_b', 'c': [-1.0] * cond_dim, 'width': 1.0},
    ]

    cm = toy['condition_manifold']
    cm['enabled'] = True
    cm['mode'] = mode
    cm['n_conditions'] = n_conditions
    cm['cond_dim'] = cond_dim  # inferred from anchors in interpolate; explicit for noise modes
    cm['include_anchors'] = True
    cm['noise_std'] = noise_std  # interpolate only (ignored by noise modes)
    cm['scale'] = scale  # noise modes only (ignored by interpolate)

    toy['condition_set']['n_replicas_per_condition'] = n_replicas
    toy['tag'] = tag
    if seed is not None:
        toy['seed'] = seed

    # batteries run headless/sequential -- master switch over every figure
    # (condition-set plot_batch calls + coverage_check histogram)
    toy['show_figures'] = False

    # --- training side: point at this pair's artifacts + match cond_dim ---
    train['prior_path'] = f"{outdir}\\{tag}_prior.pt"
    train['molecules_path'] = f"{outdir}\\{tag}_conditions.pt"
    train['buffers']['anchor_buffer']['seed_source'] = f"{outdir}\\{tag}_conditions.pt"
    train['vector_conditioning_dim'] = cond_dim
    train['run_name'] = tag
    train['tag'] = tag
    if seed is not None:
        train['seed'] = seed

    with (OUTDIR / f'{ind}_toy.yaml').open('w') as f:
        yaml.dump(toy, f, sort_keys=False, default_flow_style=False)
    with (OUTDIR / f'{ind}_train.yaml').open('w') as f:
        yaml.dump(train, f, sort_keys=False, default_flow_style=False)

    return {'index': ind, 'tag': tag, 'note': note,
            'cond_dim': cond_dim, 'n_core': n_core, 'n_ghost': n_ghost,
            'mode': mode, 'scale': scale, 'n_conditions': n_conditions,
            'noise_std': noise_std, 'n_replicas': n_replicas,
            'seed': seed if seed is not None else toy['seed']}


def write_runner(log):
    lines = [
        "# cond_workup battery -- run sequentially from energy_sampling/.",
        "# Comment out lines to skip runs. Each pair: bake artifacts, then train.",
        f'$env:PYTHONPATH = "{PYTHONPATH}"',
        "",
    ]
    for e in log:
        i = e['index']
        lines += [
            f"# --- {e['tag']}: {e['note']} ---",
            f"python data_processing\\generate_toy_prior.py configs\\cond_workup\\{i}_toy.yaml",
            f"python train.py --config configs\\cond_workup\\{i}_train.yaml",
            "",
        ]
    (OUTDIR / 'run_all.ps1').write_text("\n".join(lines))


if __name__ == '__main__':
    toy = load_base('base_toy.yaml')
    train = load_base('base_train.yaml')
    log = []

    # Every knob is spelled out on every call so the whole grid is visible and
    # tunable right here. Baseline values (experiment 0) = current
    # toy_prior_config.yaml; each later run perturbs one axis off that baseline.
    # Columns: cond_dim n_core n_ghost | mode scale n_conditions noise_std | n_replicas seed
    log.append(make_pair(0, toy, train, note='baseline: 2d interpolate, n1000, r4',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=1000, noise_std=0.1,
                         n_replicas=10, seed=None))

    # ---- condition count (generalization + tracker sparsity) ----
    log.append(make_pair(1, toy, train, note='cond count: sparse',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=50, noise_std=0.1,
                         n_replicas=10, seed=None))
    log.append(make_pair(2, toy, train, note='cond count: dense (tracker-sparsity probe)',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=10000, noise_std=0.1,
                         n_replicas=10, seed=None))

    # ---- manifold noise (off-manifold spread on the conditioner) ----
    log.append(make_pair(3, toy, train, note='noise: clean manifold',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=1000, noise_std=0.0,
                         n_replicas=10, seed=None))
    log.append(make_pair(4, toy, train, note='noise: broad',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=1000, noise_std=0.3,
                         n_replicas=10, seed=None))

    # ---- per-condition replica count (biased-seed discovery) ----
    log.append(make_pair(5, toy, train, note='replicas: minimal (floor)',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=1000, noise_std=0.1,
                         n_replicas=2, seed=None))

    # ---- dimensionality (conditioner capacity) ----
    log.append(make_pair(6, toy, train, note='cond_dim 8',
                         cond_dim=8, n_core=1, n_ghost=0,
                         mode='interpolate', n_conditions=1000, noise_std=0.1,
                         n_replicas=4, seed=None))

    # ---- field structure (energy curvature toward molecules) ----
    log.append(make_pair(7, toy, train, note='richer GMM field',
                         cond_dim=2, n_core=4, n_ghost=4,
                         mode='interpolate', n_conditions=1000, noise_std=0.1,
                         n_replicas=10, seed=None))

    # ---- geometry: uniform hypercube of conditions instead of a manifold ----
    log.append(make_pair(8, toy, train, note='uniform conditions',
                         cond_dim=2, n_core=1, n_ghost=0,
                         mode='uniform', n_conditions=1000, noise_std=0.1,
                         n_replicas=10, seed=None))

    write_runner(log)
    with (OUTDIR / 'experiment_log.yaml').open('w') as f:
        yaml.dump(log, f, sort_keys=False)
    print(f"Generated {len(log)} config pairs + run_all.ps1 + experiment_log.yaml in {OUTDIR}")
