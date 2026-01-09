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
    sg = 61
    zp = 1
    en = 'elj'
    rew = 100
    temp = 2.5
    layers = 4
    norm = 'layer'

    for eps in [0.2, 0.3, 0.4, 0.5]:
        for hidden in [1024, 2048]:
            config = deepcopy(base)
            config['norm'] = norm
            config['hidden_dim'] = hidden
            config['layers'] = layers
            config['s_emb_dim'] = hidden
            config['space_groups'] = [sg]
            config['z_primes'] = [zp]
            config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
            config['tag'] = f'xul4'
            config['energy_function'] = en
            config['energy_static_temperature'] = temp
            config['reward_range'] = rew
            config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/xuldud/xuldud.pt'
            config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/xuldud/xuldud_sg{sg}_zp{zp}.pt'

            config_path = f'{ind}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1
