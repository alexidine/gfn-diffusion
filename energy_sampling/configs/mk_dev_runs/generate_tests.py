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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': True,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 0 baseline
    {
        'lr_policy': 0.001,
        'lr_flow': 0.1,
        'lr_back': 0.001,
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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': True,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 1 high LR
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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': True,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.5,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 2 high exploration factor
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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': True,
        'both_ways': True,
        'learn_pb': False,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 3 not learned pb
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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': False,
        'both_ways': False,
        'learn_pb': False,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 4 no LS, no bwd
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
        'max_batch_size': 1000,
        'eval_period': 500,

        'local_search': False,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 5 no LS
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

        'batch_size': 5,
        'grow_batch_size': True,
        'max_batch_size': 100,
        'eval_period': 500,

        'local_search': True,
        'both_ways': True,
        'learn_pb': True,
        'exploratory': True,
        'pb_scale_range': 0.1,
        'exploration_factor': 0.2,
        'exploration_wd': True,
        'learned_variance': True,

        'anneal_energy': True,
        'energy_annealing_threshold': 10,
        'energy_temperature': 1.0
    },  # 6 small batches
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

    # automate tagging
    run_name = os.path.basename(os.getcwd())
    config['run_name'] = run_name

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1
