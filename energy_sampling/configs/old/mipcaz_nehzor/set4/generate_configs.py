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
    ind = 0
    for base_path, molname in zip(['mipcas.yaml'], ['MIPCAS']):
        base, spec_dir = load_yaml(base_path)
        for energy_function in ['elj']:
            config = deepcopy(base)
            config['energy_function'] = energy_function


            vrange = [0.25, 0.5, 0.75, 1.0]
            for var_boost in vrange:
                config['p3_widevar_prob'] = 0.5
                config['p3_widevar_var'] = var_boost
                config['continue_from_checkpoint'] = True

                run_name = f"{base['run_name']}_{energy_function}_{var_boost:.2f}_{ind}"
                config['run_name'] = run_name
                config['tag'] = '4'
                config_path = f"{ind}.yaml"#{molname}_{energy_function}_{ind}.yaml"

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                ind += 1