from pathlib import Path
import yaml
from copy import copy
import os


def load_yaml(path):
    """
    Safely load yaml file as dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


base_config = load_yaml('base.yaml')

# conditional and unconditional runs
config_list = [
    {
        'anneal_energy': False,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'latent_harmonic',
    },
    {
        'anneal_energy': False,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'crystal_harmonic',
    },
    {
        'anneal_energy': False,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'latent_multiharmonic',
    },
    {
        'anneal_energy': False,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'crystal_multiharmonic',
    },
    {
        'anneal_energy': True,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'latent_harmonic',
    },
    {
        'anneal_energy': True,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'crystal_harmonic',
    },
    {
        'anneal_energy': True,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'latent_multiharmonic',
    },
    {
        'anneal_energy': True,
        'energy_min_temperature': 0.01,
        'energy_static_temperature': 1,
        'energy_function': 'crystal_multiharmonic',
    },
]

def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1


ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['run_name'] = config['run_name'] + '_' + str(ind)

    run_config = config_list[ix1]
    overwrite_nested_dict(config, run_config)

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1
