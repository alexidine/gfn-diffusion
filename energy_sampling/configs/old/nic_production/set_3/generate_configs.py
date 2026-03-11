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
    param_deltas = [
        ('energy_function', ['elj', 'uma']),
        ('t_scale', [0.05, 0.1, 0.2]),
        ('log_var_range', [4, 6]),
        ('pb_var_range', [2, 4, 6]),
        ('noised_buffer_length', [50000, 200000]),
        ('anchor_noise_fraction', [0.95, 1.0]),
        ('btb_threshold', [1.5, 2.0]),
        ('thermalization_conv_eps', [0.5, 0.75]),
    ]

    ind = 0
    for key, values in param_deltas:
        for v in values:
            config = deepcopy(base)

            config.update({
                'space_groups': [14],
                'z_primes': [1],
                'run_name': f'sg14_zp1_{ind}',
                'tag': 'nic3',
                'molecules_path': '/scratch/mk8347/csd_runs/datasets/nicotinamide/protonated_nicotinamide.pt',
                'buffer_path': '/scratch/mk8347/csd_runs/datasets/nicotinamide/nic_sg14_zp1.pt',
            })

            config[key] = v

            if (
                    key == 'pb_var_range'
                    and config['pb_var_range'] > config['log_var_range']
            ):
                continue

            with open(f'{ind}.yaml', 'w') as f:
                yaml.dump(config, f)

            ind += 1