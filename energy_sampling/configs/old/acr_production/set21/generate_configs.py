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
    en = 'elj'

    ind = 0
    for lr_policy in [1e-4, 5e-4]:
        for trajlen in [100, 150]:

            config = deepcopy(base)
            config['T'] = trajlen
            config['eval_T'] = trajlen
            config['min_traj_len'] = trajlen
            config['max_traj_len'] = trajlen
            config['lr_policy'] = lr_policy
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
            config['tag'] = f'acr20'
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['energy_function'] = en

            config['molecules_path'] = '/scratch/mk8347/data/crystal_datasets/acridine/acridine_conformer.pt'
            config['buffer_path'] = f'/scratch/mk8347/data/crystal_datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

            config_path = f'{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1
