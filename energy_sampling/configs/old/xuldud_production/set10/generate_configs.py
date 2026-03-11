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
    rew = 100
    temp = 2.5
    layers = 4
    norm = 'layer'
    hidden = 1024
    en = 'uma'
    eps = 0.2
    mx = 25
    nbs = 10000
    bs = 100000

    config = deepcopy(base)
    config['noised_buffer_length'] = nbs
    config['buffer_size'] = bs
    config['noised_max_steps'] = mx
    config['max_batch_size'] = 1000
    config['norm'] = norm
    config['hidden_dim'] = hidden
    config['layers'] = layers
    config['s_emb_dim'] = hidden
    config['space_groups'] = [sg]
    config['z_primes'] = [zp]
    config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
    config['tag'] = f'xul10'
    config['energy_function'] = en
    config['energy_static_temperature'] = temp
    config['reward_range'] = rew
    config['thermalization_conv_eps'] = eps
    config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/xuldud/xuldud.pt'
    config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/xuldud/xuldud_sg{sg}_zp{zp}.pt'

    config_path = f'{ind}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    ind += 1