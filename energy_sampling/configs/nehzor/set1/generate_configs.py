from copy import deepcopy
import os
from itertools import product

import yaml
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    path = Path(path)
    with path.open('r') as f:
        return yaml.safe_load(f), path.parent  # return both content and its directory


def apply_experiment(base_config, experiment):
    config = deepcopy(base_config)

    for k, v in experiment.items():
        if k == "name":
            continue
        if v is not None:
            config[k] = v
        if v == 'none':
            config[k] = None

    return config


EXPERIMENT_TEMPLATE = {
    "name": None,
    "energy_function": None

}

if __name__ == "__main__":
    base_path = 'base.yaml'
    base, spec_dir = load_yaml(base_path)

    experiments = [
        {"name": "uma", "energy_function": 'uma', 'buffer_path': "/scratch/mk8347/data/crystal_datasets/nehzor/nehzor_uma_prior_dataset.pt"
},
        {"name": "elj", "energy_function": 'elj', 'buffer_path': "/scratch/mk8347/data/crystal_datasets/nehzor/nehzor_elj_prior_dataset.pt"},

    ]

    for ind, exp in enumerate(experiments):
        config = apply_experiment(base, exp)

        run_name = f"{base['run_name']}_{exp['name']}_{ind}"
        config['run_name'] = run_name

        config_path = f"{ind}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
