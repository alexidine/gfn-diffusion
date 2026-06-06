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
    for base_path, molname in zip(['mipcas.yaml','nehzor.yaml'], ['MIPCAS','NEHZOR']):
        base, spec_dir = load_yaml(base_path)
        for energy_function in ['elj','uma']:
            config = deepcopy(base)
            config['energy_function'] = energy_function

            if energy_function == 'elj':
                vrange = [0]
            else:
                vrange = [0, 0.69, 0.76, 0.82]
            for var_boost in vrange:
                config['p3_widevar_prob'] = 0.25
                config['p3_widevar_var'] = var_boost

                run_name = f"{base['run_name']}_{energy_function}_{var_boost:.2f}_{ind}"
                config['run_name'] = run_name
                config['tag'] = 'PV'
                config_path = f"{ind}.yaml"#{molname}_{energy_function}_{ind}.yaml"

                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                ind += 1