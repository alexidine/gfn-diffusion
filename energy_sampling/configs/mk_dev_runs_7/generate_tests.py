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
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #0
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 50,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #1
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.55,
        'wd_max_steps': 2000,
    } #2
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 5000,
    } #3
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': .25,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #4
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': 'layer',
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #5
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 512,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #6
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 2,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #7
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 1.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 8,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #8
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 5.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': None,
        'dropout': 0,
        'T': 50,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } # 9
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'bwd': False,
        'both_ways': False,
        'train_pb': False,
        'anneal_energy': True,  # harden intermolecular repulsion over time
        'energy_annealing_threshold': 5.0e-2,
        'convergence_history': 500,
        'energy_density_coeff': 1,  # how much to weight the density penalty term in the energy function
        'temperature_conditioning': True,
        'energy_min_temperature': 0.99999,
        'energy_max_temperature': 1.0,
        'energy_static_temperature': 1,
        'temperature_scaling_factor': 1,
        'joint_layers': 4,
        'hidden_dim': 256,
        'norm': 'layer',
        'dropout': 0,
        'T': 10,
        'pb_scale_range': 0.1,
        't_scale': 1.0,
        'exploration_factor': 0.35,
        'wd_max_steps': 2000,
    } #5
)


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
