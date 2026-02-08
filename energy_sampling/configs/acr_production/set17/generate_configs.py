import os
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


if __name__ == "__main__":
    base_path = 'base.yaml'
    base, spec_dir = load_yaml(base_path)

    ind = 0
    sg = 14
    zp = 1
    temp = 2.5
    en = 'elj'
    mx = 10
    nbs = 20000

    ind = 0
    for s_err in [0.05, 0.5]:
        for i_err in [1, 10]:
            config = deepcopy(base)
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
            config['tag'] = f'acr17'
            config['noised_buffer_length'] = nbs
            config['noised_max_steps'] = mx
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['energy_function'] = en
            config['energy_static_temperature'] = temp
            config['thermalization_slope_err'] = s_err
            config['thermalization_intercept_err'] = i_err

            config['molecules_path'] = '/scratch/mk8347/data/crystal_datasets/acridine/acridine_conformer.pt'
            config['buffer_path'] = f'/scratch/mk8347/data/crystal_datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

            config_path = f'{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1

