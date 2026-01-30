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
    temp = 2.5
    en = 'uma'
    eps = 0.2
    mx = 25
    nbs = 10000

    ind = 0
    for zp in [1]:
        for sg in [14]:
            for mx in [25, 50]:
                for nbs in [10000, 20000]:
                    for bs in [10000, 20000]:
                        config = deepcopy(base)
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
                        config['tag'] = f'acr13'
                        config['noised_buffer_length'] = nbs
                        config['buffer_size'] = bs
                        config['noised_max_steps'] = mx
                        config['max_batch_size'] = 1000
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['energy_function'] = en
                        config['energy_static_temperature'] = temp
                        config['thermalization_conv_eps'] = eps

                        config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/acridine/acridine_conformer.pt'
                        config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

                        config_path = f'{ind}.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)

                        ind += 1

    en = 'elj'
    for zp in [1]:
        for sg in [14]:
            for mx in [25, 50]:
                for nbs in [10000, 20000]:
                    for bs in [10000, 20000]:
                        config = deepcopy(base)
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
                        config['tag'] = f'acr13'
                        config['noised_buffer_length'] = nbs
                        config['buffer_size'] = bs
                        config['noised_max_steps'] = mx
                        config['max_batch_size'] = 1000
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['energy_function'] = en
                        config['energy_static_temperature'] = temp
                        config['thermalization_conv_eps'] = eps

                        config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/acridine/acridine_conformer.pt'
                        config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

                        config_path = f'{ind}.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)

                        ind += 1

    temp = 1.25

    for zp in [1]:
        for sg in [14]:
            for mx in [25, 50]:
                for nbs in [10000, 20000]:
                    for bs in [10000, 20000]:
                        config = deepcopy(base)
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
                        config['tag'] = f'acr13'
                        config['noised_buffer_length'] = nbs
                        config['buffer_size'] = bs
                        config['noised_max_steps'] = mx
                        config['max_batch_size'] = 1000
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['energy_function'] = en
                        config['energy_static_temperature'] = temp
                        config['thermalization_conv_eps'] = eps

                        config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/acridine/acridine_conformer.pt'
                        config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

                        config_path = f'{ind}.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)

                        ind += 1

    temp = 5
    en = 'elj'
    for zp in [1]:
        for sg in [14]:
            for mx in [25, 50]:
                for nbs in [10000, 20000]:
                    for bs in [10000, 20000]:
                        config = deepcopy(base)
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['run_name'] = f'sg{sg}_zp{zp}_{ind}'
                        config['tag'] = f'acr13'
                        config['noised_buffer_length'] = nbs
                        config['buffer_size'] = bs
                        config['noised_max_steps'] = mx
                        config['max_batch_size'] = 1000
                        config['space_groups'] = [sg]
                        config['z_primes'] = [zp]
                        config['energy_function'] = en
                        config['energy_static_temperature'] = temp
                        config['thermalization_conv_eps'] = eps

                        config['molecules_path'] = '/scratch/mk8347/csd_runs/datasets/acridine/acridine_conformer.pt'
                        config['buffer_path'] = f'/scratch/mk8347/csd_runs/datasets/acridine/acridine_sg{sg}_zp{zp}.pt'

                        config_path = f'{ind}.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)

                        ind += 1
