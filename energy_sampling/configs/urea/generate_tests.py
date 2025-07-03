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
config_list = []
for energy in ['ellipsoid_overlap', 'silu_energy']:
    for temps in ['hot', 'conditioned']:
        for both_ways in [True, False]:
            if temps == 'hot':
                anneal = False
                condition = False
                static_T = 1
                mode_fwd = 'tb'
            elif temps == 'conditioned':
                anneal = True
                condition = True
                static_T = 1
                mode_fwd = 'vg'
            else:
                assert False

            config_list.append(
                {'energy_function': energy,
                 'energy_static_temperature': static_T,
                 'anneal_energy': anneal,
                 'temperature_conditioning': condition,
                 'conditional_flow_model': condition,
                 'mode_fwd': mode_fwd,
                 'mode_bwd': mode_fwd,
                 'both_ways': both_ways,
                 }
            )

""" 2 batches done with density coeff 0.01 and 10
density coeff 0.01
0 - both, hot, ellipsoid : fit seems good, but density is low. Distributions are edgy
1 - fwd, hot, ellipsoid : fit ok, also a bit edgy, and low density
2 - both, conditioned, ellipsoid : crashed out early from eigh
3 - fwd, conditioned, ellipsoid : very edgy, very diffuse. E vs T failure
4 - both, hot, silu : OK fit. Not too edgy. Density low but actually improving. Not focusing great on low modes
5 - fwd, hot, silu : bit edgy, OK fit, diffuse and not really improving
6 - both, conditioned, silu : very diffuse, bad fit. Appears to basically be exploding
7 - fwd, conditioned, silu : very diffuse, multiple edgy dimensions. OK but not great fit. May also be mid-explosion

density_coeff 10
0 - both, hot, ellipsoid : oom kill. Was off to an OK start I guess
1 - fwd, hot, ellipsoid : Edgy in pose dims. Dense with some overlaps
2 - both, conditioned, ellipsoid : crashed out early from eigh
3 - fwd, conditioned, ellipsoid : crashed out early from eigh
4 - both, hot, silu : good densities, very poor energies, poor policy support
5 - fwd, hot, silu : ultra-dense, very good fit, very edgy
6 - both, conditioned, silu : oom kill
7 - fwd, conditioned, silu : super edgy, very dense, very bad

to-do:
-: tune density --> tune range
-: control variance --> play with ranges
-:x fix OOM --> more ram
-:x time regularization --> add a smoothness penalty
-: address policy support --> more policy variance for longer, or TBM/MLE training
"""

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 8 new baseline config
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 8,
     'hidden_dim': 512,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 9 - bigger model
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 2,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 10 - higher density cutoff
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 2.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 11 - higher std variance
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 20000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 12 - more exploration time
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.2,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 13 - more pb range
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 2.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 14 - less pf range
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0.5,
     'norm': None,
     }  # 15 - lots of dropout
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': 'layer',
     }  # 16 - norming
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': False,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 17 fixed Pb
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 1,
     'exploratory': True,
     'exploration_factor': 0.5,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': False,
     'learned_variance': False,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     }  # 18 fixed variance and pb
)



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
