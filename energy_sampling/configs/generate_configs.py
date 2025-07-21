from itertools import product

import yaml
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    path = Path(path)
    with path.open('r') as f:
        return yaml.safe_load(f), path.parent  # return both content and its directory


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


def generate_oneoffs(spec_path='experiments.yaml', output_dir='configs'):
    spec, spec_dir = load_yaml(spec_path)
    base_path = spec['defaults']['base']
    base_config_path = spec_dir / base_path  # <- this makes it relative to the spec
    base, _ = load_yaml(base_config_path)
    experiments = spec['experiments']

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    log_entries = []

    for i, exp in enumerate(experiments):
        tag = exp['tag']
        config = deepcopy(base)
        config['run_name'] = tag
        overwrite_nested_dict(config, exp['update'])

        config_path = outdir / f"{i}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        log_entries.append({
            "index": i,
            "tag": tag,
            "update_keys": list(exp['update'].keys()),
            "notes": exp.get("notes", "")
        })

    log_path = outdir / 'experiment_log.yaml'
    with open(log_path, 'w') as f:
        yaml.dump(log_entries, f, sort_keys=False)

    print(f"Generated {len(experiments)} configs with log at {log_path}")


def generate_grid(spec_path='experiments.yaml', output_dir='configs'):
    spec, spec_dir = load_yaml(spec_path)
    base_path = spec['defaults']['base']
    base_config_path = spec_dir / base_path  # <- this makes it relative to the spec
    base, _ = load_yaml(base_config_path)

    grid_axes = spec['grid']
    presets = spec.get('presets', {})
    grid_overrides = spec.get('grid_overrides', {})

    keys, values = zip(*grid_axes.items())  # e.g., keys = ['direction', 'model', 'variance']

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for i, combo in enumerate(product(*values)):
        tag_parts = [str(i)] + [f"{k}-{v}" for k, v in zip(keys, combo)]
        tag = "_".join(tag_parts)
        config = deepcopy(base)
        config['run_name'] = tag

        for k, v in zip(keys, combo):
            # Apply named preset if available
            if v in presets:
                overwrite_nested_dict(config, presets[v])
            # Apply axis-specific overrides if available
            if k in grid_overrides and v in grid_overrides[k]:
                overwrite_nested_dict(config, grid_overrides[k][v])

            elif isinstance(v, (int, float, str, bool)):
                config[k] = v

        with open(outdir / f"{i}.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        experiments.append({
            "index": i,
            "tag": tag,
            "combo": dict(zip(keys, combo)),
            "notes": ""
        })

    log_path = outdir / 'experiment_log.yaml'
    with open(log_path, 'w') as f:
        yaml.dump(experiments, f)

    print(f"Generated {len(experiments)} configs with log at {log_path}")


if __name__ == "__main__":
    # for if the experiments are a list of single runs
    generate_oneoffs(spec_path='qm9/test1/experiments1.yaml', output_dir='qm9/test1/')

    # for if the experiments are run on a parameter grid
    #generate_grid(spec_path='nicotinamide/experiments2.yaml', output_dir='nicotinamide/')
