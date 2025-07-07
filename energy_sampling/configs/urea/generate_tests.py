from pathlib import Path
import yaml
from copy import copy
import os


def overwrite_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            assert k in d1.keys()
            d1[k] = overwrite_nested_dict(d1[k], v)
        else:
            d1[k] = v
    return d1

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


"""
to-test

- fwd, fwd-bwd, fwd-bwd with preload
- model size
- learnable pb
- learnable variance
- overall variance level
- traj norming
- greedy alternating
- temperature conditioning
"""

base_config = load_yaml('base.yaml')

# baseline
base_config.update({'energy_function': 'silu_energy',
                    'energy_static_temperature': 1,
                    'anneal_energy': False,
                    'temperature_conditioning': False,
                    'conditional_flow_model': False,
                    'min_traj_length': 10,
                    'max_traj_length': 15,
                    'discretizer_max_ratio': 10,
                    'mode_fwd': 'tb',
                    'mode_bwd': 'tb',
                    'both_ways': False,
                    'buffer_path': None,
                    'energy_density_coeff': 2.0,
                    'exploratory': True,
                    'exploration_factor': 0.35,
                    'exploration_wd': True,
                    'wd_max_steps': 10000,
                    't_scale': 1.0,
                    'log_var_range': 4.0,
                    'pb_scale_range': 0.1,
                    'learn_pb': False,
                    'learned_variance': False,
                    'repeats': 10,
                    'joint_layers': 2,
                    'hidden_dim': 128,
                    's_emb_dim': 128,
                    'dropout': 0,
                    'norm': None,
                    'lr_anneal_time': 10000,
                    'max_batch_size': 500,
                    'buffer_size': 25000,
                    'reweight_T': None})

config_list = []
tags = []
"""
to-test

- fwd, fwd-bwd, fwd-bwd with preload
- model size
- learnable pb
- learnable variance
- overall variance level
- traj norming
- greedy alternating
- temperature conditioning
"""
ind = 0
for direction in ['fwd', 'fwd-bwd', 'fwd-bwd-preload']:
    for special in ['big_model', 'learn_pb', 'learn_variance',
                    'low_variance', 'norm_traj', 'greedy', 'temp_cond']:
        cc = {}

        if direction == 'fwd':
            cc['both_ways'] = False
            cc['buffer_path'] = None
        elif direction == 'fwd-bwd':
            cc['both_ways'] = True
            cc['buffer_path'] = None
        elif direction == 'fwd-bwd-preload':
            cc['both_ways'] = True
            cc['buffer_path'] = '/scratch/mk8347/csd_runs/datasets/urea_gfn_dataset.pt'
        if special == 'big_model':
            cc['joint_layers'] = 8
            cc['hidden_dim'] = 512
            cc['s_emb_dim'] = 512
            cc['norm'] = 'layer'
            cc['dropout'] = 0.5
        elif special == 'learn_pb':
            cc['learn_pb'] = True
        elif special == 'learn_variance':
            cc['learned_variance'] = True
        elif special == ['low_variance']:
            cc['t_scale'] = 0.5
        elif special == ['norm_traj']:
            cc['reweight_T'] = 1
        elif special == 'greedy':
            cc['mode_fwd'] = 'tb_greedy'
        elif special == ['temp_cond']:
            cc['mode_fwd'] = 'vg'
            cc['conditional_flow_model'] = True
            cc['anneal_energy'] = True
            cc['temperature_conditioning'] = True
            cc['reweight_T'] = 10

        config_list.append(cc)
        tag = f"{ind}_{direction}_{special}"
        tags.append(tag)
        print(tag)
        ind += 1
"""


0 fwd big_model - too dense, but nice dists. Not sharp at all
1 fwd learn_pb  - very slightly better
2 fwd learn_variance - similar
3 fwd low_variance - similar !! did not have a lower variance, actually
4 fwd norm_traj - similar
5 fwd greedy
6 fwd temp_cond
7 fwd-bwd big_model
8 fwd-bwd learn_pb
9 fwd-bwd learn_variance
10 fwd-bwd low_variance
11 fwd-bwd norm_traj
12 fwd-bwd greedy
13 fwd-bwd temp_cond
14 fwd-bwd-preload big_model
15 fwd-bwd-preload learn_pb
16 fwd-bwd-preload learn_variance
17 fwd-bwd-preload low_variance
18 fwd-bwd-preload norm_traj
19 fwd-bwd-preload greedy
20 fwd-bwd-preload temp_cond

"""


ind = 0
for ix1 in range(len(config_list)):
    config = copy(base_config)
    config['run_name'] = config['run_name'] + '_' + tags[ix1]

    run_config = config_list[ix1]

    overwrite_nested_dict(config, run_config)

    with open(str(ind) + '.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    ind += 1
