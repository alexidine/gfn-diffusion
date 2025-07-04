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
config_list = []
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'buffer_path': None,
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
     }  # 0 - density loss wins out up to 1. Moderate overlaps. Overall variance explosion.
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
     'buffer_path': None,
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
     }  # 1 both ways - looking ok but memory leak crash
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': True,
     'temperature_conditioning': True,
     'conditional_flow_model': True,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'buffer_path': None,
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
     }  # 2 with T conditioning - better overlaps. OK Z learning
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'buffer_path': None,
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
     'joint_layers': 8,
     'hidden_dim': 512,
     's_emb_dim': 512,
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 1000,
     }  # 3 huge model - wacky explision
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 40,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'buffer_path': None,
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
     }  # 4 more train steps - identical to 1
)

"""
trying smaller density coeff
fixed both_ways
trying big expl but small baseline variance
trying small learnable variance
"""

config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
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
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 5 - new baseline
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
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
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0.5,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 6 - with dropout
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': False,
     'buffer_path': None,
     'energy_density_coeff': 2.0,
     'exploratory': True,
     'exploration_factor': 1,
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
     'dropout': 0,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 7 - lo-hi variance
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'combo',
     'mode_bwd': 'combo',
     'both_ways': True,
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
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 8 - both ways
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
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
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 9 - pure tb
)
config_list.append(
    {'energy_function': 'ellipsoid_overlap',
     'energy_static_temperature': 1,
     'anneal_energy': False,
     'temperature_conditioning': False,
     'conditional_flow_model': False,
     'min_traj_length': 10,
     'max_traj_length': 15,
     'discretizer_max_ratio': 10,
     'mode_fwd': 'vg',
     'mode_bwd': 'vg',
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
     'repeats': 5,
     'joint_layers': 4,
     'hidden_dim': 256,
     's_emb_dim': 256,
     'dropout': 0,
     'norm': None,
     'lr_anneal_time': 20000,
     'max_batch_size': 2000,
     'buffer_size': 50000,
     }  # 10 - pure vg
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
