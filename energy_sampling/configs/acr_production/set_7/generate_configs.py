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
    for zp in [1, 2]:
        for sg in [2, 4, 9, 14, 15, 19, 33, 61, 1, 5, 7, 12, 13, 18, 29, 33, 60, 62, 146]:
            config = deepcopy(base)
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['run_name'] = f'sg{sg}_zp{zp}'
            config['tag'] = f'acr7'
            config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/acridine/acridine_conformer.pt'
            config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

            config_path = f'{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1
