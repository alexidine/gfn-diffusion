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

config_list = [
    {
        'lr_policy': 0.0001,
        'lr_flow': 0.01,
        'lr_back': 0.0001,
        'gradient_norm_clip': 1.0,
        'gfn_clip': 10000.0,
        'clipping': True,
        'weight_decay': 1e-7,
        'use_weight_decay': False,

        'joint_layers': 4,
        'hidden_dim': 256,
        'dropout': 0,
        'norm': None,

        'batch_size': 25,
        'grow_batch_size': True,
        'max_batch_size': 500,
        'eval_period': 250,

        'local_search': False,
        'both_ways': False,
        'learn_pb': False,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.5,
        'exploration_wd': True,
        'learned_variance': True,

    },  # 0 baseline
    {
        'lr_policy': 0.00005,
        'lr_flow': 0.005,
        'lr_back': 0.00005,
        'gradient_norm_clip': 1.0,
        'gfn_clip': 10000.0,
        'clipping': True,
        'weight_decay': 1e-7,
        'use_weight_decay': False,

        'joint_layers': 4,
        'hidden_dim': 256,
        'dropout': 0,
        'norm': None,

        'batch_size': 25,
        'grow_batch_size': True,
        'max_batch_size': 500,
        'eval_period': 250,

        'local_search': False,
        'both_ways': False,
        'learn_pb': False,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.5,
        'exploration_wd': True,
        'learned_variance': True,

    },  # 1 low lr
    {
        'lr_policy': 0.0001,
        'lr_flow': 0.01,
        'lr_back': 0.0001,
        'gradient_norm_clip': 1.0,
        'gfn_clip': 10000.0,
        'clipping': True,
        'weight_decay': 1e-7,
        'use_weight_decay': False,

        'joint_layers': 4,
        'hidden_dim': 256,
        'dropout': 0,
        'norm': None,

        'batch_size': 25,
        'grow_batch_size': True,
        'max_batch_size': 500,
        'eval_period': 250,

        'local_search': False,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.5,
        'exploration_wd': True,
        'learned_variance': True,

    },  # 2 both ways training
]

"""
Notes

-x high LR makes weight update trigger unusable
- making sheets which are a bit too diffuse
- still doing mode collapse on a single positional packing
-x indeed mode collapse in general here
-x add full ranges to lattice feats viz
-x even wider steps between reporting please

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
