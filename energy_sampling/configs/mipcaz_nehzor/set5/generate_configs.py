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
    config_paths = [
        r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\configs\mipcas\set2\0.yaml',
        r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\configs\mipcas\set2\10.yaml',
        r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\configs\nehzor\set2\1.yaml',
        r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\configs\nehzor\set2\0.yaml'
    ]
    model_prepends = [
        'mipcas_2',
        'mipcas_2',
        'nehzor_2',
        'nehzor_2',
    ]
    for base_path, tag in zip(config_paths, model_prepends):
        base, spec_dir = load_yaml(base_path)
        config = deepcopy(base)

        if 'nehzor' in base_path:
            mol = 'nehzor'
        else:
            mol = 'mipcas'


        config['p3_widevar_prob'] = 0
        config['p3_widevar_var'] = 0
        config['continue_from_checkpoint'] = True

        #run_name = f"{base['run_name']}_{mol}_{config['energy_function']}_{ind}"
        #config['run_name'] = run_name
        config['tag'] = tag
        config_path = f"{ind}.yaml"#{molname}_{energy_function}_{ind}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        ind += 1