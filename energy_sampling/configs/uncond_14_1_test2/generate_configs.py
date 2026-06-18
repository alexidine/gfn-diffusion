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


if __name__ == "__main__":
    ind = 0
    base_path = 'base.yaml'
    for repeats in [1, 3, 5, 7]:
        base, spec_dir = load_yaml(base_path)
        config = deepcopy(base)

        run_name = f"{base['run_name']}_{repeats}"
        config['repeats'] = repeats
        config['run_name'] = run_name
        config['tag'] = 'uncond_2'
        config_path = f"{ind}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        ind += 1
