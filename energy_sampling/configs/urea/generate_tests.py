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
        elif special == 'low_variance':
            cc['t_scale'] = 0.5
        elif special == 'norm_traj':
            cc['reweight_T'] = 1
        elif special == 'greedy':
            cc['mode_fwd'] = 'tb_greedy'
        elif special == 'temp_cond':
            cc['mode_fwd'] = 'vg'
            cc['mode_bwd'] = 'vg'
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
0 fwd big_model - blew up distributions hard to edges. Looks like mode collapse, but on straight trash
1 fwd learn_pb - kindof fine. too-wide dists. log Z unconverged. packing coeff all over the place
2 fwd learn_variance - too dense, more unstable learning. High loss. Wide dists
3 fwd low_variance - still too-wide dists - FAILED TO GO INTO CONFIG
4 fwd norm_traj - too-wide dists. Exactly identical to 3 - FAILED TO GO INTO CONFIG
5 fwd greedy - unspectacular. Narrow dists
6 fwd temp_cond - pinned wide dists. Super diffuse
7 fwd-bwd big_model - better behaved. Maybe starting to learn something. Good fit. Bit diffuse
8 fwd-bwd learn_pb - bit worse
9 fwd-bwd learn_variance - diverging, despite better losses. LMAO. Fitting to crap
10 fwd-bwd low_variance - FAILED TO GO INTO CONFIG
11 fwd-bwd norm_traj - FAILED TO GO INTO CONFIG
12 fwd-bwd greedy - bit worse than baseline. Very, very wide dists
13 fwd-bwd temp_cond. A bit wide. Could be worse honestly. Excellent, excellent fit. Was maybe doing ok before i cancelled it.
# the rest below with the preloaded sample all over-boosted log Z, and the forward training stalled out
14 fwd-bwd-preload big_model
15 fwd-bwd-preload learn_pb
16 fwd-bwd-preload learn_variance
17 fwd-bwd-preload low_variance
18 fwd-bwd-preload norm_traj
19 fwd-bwd-preload greedy
20 fwd-bwd-preload temp_cond

none of the above fit the modes at all or found very many bound states

next step: try for vicious overfit / mode collapse
to-try:
:: overall
> big buffer
> moderate buffer temperature

-: tb_greedy
-: vg_greedy
-: greedy only
-: cond again
-: both ways and preload only
-: aggressive traj norming
-: low baseline variance
"""
new_tags = [
    '21_baseline',
    '22_tb_greedy',
    '23_vg_greedy',
    '24_cond_vg',
    '25_traj_norm',
    '26_low_var',
    '27_big_model',
    '28_high_var_range',
    '29_low_var_range',
    '30_high_pb_range',
    '31_pre_baseline',
    '32_pre_tb_greedy',
    '33_pre_vg_greedy',
    '34_pre_cond_vg',
    '35_pre_traj_norm',
    '36_pre_low_var',
    '37_pre_big_model',
    '38_pre_high_var_range',
    '39_pre_low_var_range',
    '40_pre_high_pb_range',
]
tags.extend(new_tags)

new_base = \
    {'lr_policy': 0.001,
     'lr_back': 0.001,
     'lr_flow': 0.1,
     'lr_anneal_time': 10000,
     'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'tb',
     'mode_bwd': 'tb',
     'both_ways': True,
     'buffer_path': None,  #'/scratch/mk8347/csd_runs/datasets/urea_gfn_dataset.pt'
     'energy_density_coeff': 2.0,
     'exploratory': True,
     'exploration_factor': 0.35,
     'exploration_wd': True,
     'wd_max_steps': 5000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.5,
     'learn_pb': False,
     'learned_variance': False,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 128,
     's_emb_dim': 128,
     'dropout': 0,
     'norm': 'layer',
     'max_batch_size': 500,
     'buffer_size': 25000,
     'reweight_T': None,
     'beta': 0.01}

for tag in new_tags:
    cc = copy(new_base)
    if 'pre' in tag:
        cc['buffer_path'] = '/scratch/mk8347/csd_runs/datasets/urea_gfn_dataset.pt'

    if 'baseline' in tag:
        pass
    elif 'tb_greedy' in tag:
        cc['mode_fwd'] = 'tb_greedy'
        cc['mode_bwd'] = 'tb'
    elif 'vg_greedy' in tag:
        cc['mode_fwd'] = 'vg_greedy'
        cc['mode_bwd'] = 'vg'
    elif 'cond_vg' in tag:
        cc['mode_fwd'] = 'vg_greedy'
        cc['mode_bwd'] = 'vg'
        cc['conditional_flow_model'] = True
        cc['anneal_energy'] = True
        cc['temperature_conditioning'] = True
        cc['reweight_T'] = 10
    elif 'traj_norm' in tag:
        cc['rweight_T'] = 10
    elif 'low_var' in tag:
        cc['t_scale'] = 0.5
    elif 'big_model' in tag:
        cc['joint_layers'] = 8
        cc['hidden_dim'] = 512
        cc['s_emb_dim'] = 512
        cc['norm'] = 'layer'
        cc['dropout'] = 0.5
    elif 'high_var_range' in tag:
        cc['log_var_range'] = 10.0
    elif 'low_var_range' in tag:
        cc['log_var_range'] = 1.0
    elif 'high_pb_range' in tag:
        cc['pb_scale_range'] = 5
    else:
        assert False

    config_list.append(cc)

"""
General notes
-: forward loss crazy noisy. Losses in general quite noisy. Possibly split LR is not good
-: log Z training lags way behind

21_baseline
22_tb_greedy
23_vg_greedy
24_cond_vg
25_traj_norm
26_low_var
27_big_model
28_high_var_range
29_low_var_range
30_high_pb_range
31_pre_baseline
32_pre_tb_greedy
33_pre_vg_greedy
34_pre_cond_vg
35_pre_traj_norm
36_pre_low_var
37_pre_big_model
38_pre_high_var_range
39_pre_low_var_range
40_pre_high_pb_range

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
