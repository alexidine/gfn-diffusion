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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 9 - bigger model : exploded a bit and then oomed very early
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
     'energy_density_coeff': 5,
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
     'lr_anneal_time': 20000,
     }  # 10 - lower density cutoff : identical except diffuse samples
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 11 - higher std variance : worse losses but better exploration. var expl in rot dims. pretty good training tbh
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 12 - more exploration time : basically the same, bit worse honestly
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 13 - more pb range : atrociously bad loss behavior
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 14 - less pf range : slightly worse than baseline, bad parity plots
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 15 - lots of dropout : actually way better sample performance. Good loss
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 16 - norming : meh
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 17 fixed Pb : backward training gets worse over time
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
     'energy_density_coeff': 10,
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
     'lr_anneal_time': 20000,
     }  # 18 fixed variance and pb : actually kindof wellish behaved. Beautiful parity plots
)

"""
-: dropout seems good
-: combo loss seems like a bit of a fail
-: learned Z not converging to empirical estimate
-: high density cap seems good
-: models systematically fail to match ref distribution
-: models systematically fail to find good minima
-: models often explode rotation dimension
-: more variance budget might be good, but too much freedom is unhealthy
-: fixed policies are bad
-: saturating dimensions might be bad
-: trajectories may need to be longer

trying: 
dropout, new new combo loss, high density cap, 
less variance freedom, longer training trajs, tigher ts ratios
"""


config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
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
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 19 new baseline config, the most basic shit imaginable
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 0.5,
     'pb_scale_range': 0.1,
     'learn_pb': False,
     'learned_variance': True,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 20 with very small learned variance
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 0.5,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': False,
     'learned_variance': False,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 21 with lower base variance
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
     'exploration_wd': True,
     'wd_max_steps': 10000,
     't_scale': 1.0,
     'log_var_range': 4.0,
     'pb_scale_range': 0.1,
     'learn_pb': True,
     'learned_variance': False,
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 22 with very small learned back mean
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 0.5,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
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
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 23 with lower static T
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 8,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
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
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 24 with shorter training trajs
)

config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
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
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 25 with both ways training
)
config_list.append(
    {'energy_function': 'silu_energy',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 20,
     'max_traj_length': 30,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'energy_density_coeff': 10,
     'exploratory': True,
     'exploration_factor': 0.35,
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
     'dropout': 0.25,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 26 with smaller dropout
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
