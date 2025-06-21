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

# a bunch of essentially unconditional runs with forward-only training
config_list = []
config_list.append(
    {
        'energy_min_temperature': 0.01,
        'energy_max_temperature': 0.01,
        'mode_fwd': 'tb',
        'mode_bwd': 'tb'
    }
)
config_list.append(
    {
        'energy_min_temperature': 0.1,
        'energy_max_temperature': 0.1,
        'mode_fwd': 'tb',
        'mode_bwd': 'tb'
    }
)

config_list.append(
    {
        'energy_min_temperature': 1,
        'energy_max_temperature': 1,
        'mode_fwd': 'tb',
        'mode_bwd': 'tb'
    }
)
config_list.append(
    {
        'energy_min_temperature': 0.01,
        'energy_max_temperature': 0.01,
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg'
    }
)
config_list.append(
    {
        'energy_min_temperature': 0.1,
        'energy_max_temperature': 0.1,
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg'
    }
)

config_list.append(
    {
        'energy_min_temperature': 1,
        'energy_max_temperature': 1,
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg'
    }
)




"""
1) TB was unstable and did weird stuff with very wide variance distributions
2) vargrad seemed more stable
3) buffer poisoning looks like a real potential problem
"""

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
