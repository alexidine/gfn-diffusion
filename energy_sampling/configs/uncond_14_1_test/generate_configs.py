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
    for log_var_range in [2, 6]:
        for pb_var_range in [2, 6]:
            for t_scale in [0.025, 0.05, 0.75]:
                base, spec_dir = load_yaml(base_path)
                config = deepcopy(base)

                run_name = f"{base['run_name']}_{log_var_range}_{pb_var_range}_{t_scale}"
                config['run_name'] = run_name
                config['tag'] = 'uncond_1'
                config_path = f"{ind}.yaml"

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                ind += 1
