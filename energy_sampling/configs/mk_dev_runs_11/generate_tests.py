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
        'pb_scale_range': 0.1,
        'max_batch_size': 2000,
        't_scale': 1.0

    }  # 0, tb small pb
)
config_list.append(
    {
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg',
        'pb_scale_range': 0.1,
        'max_batch_size': 500,
        't_scale': 1.0

    }  # 1, vargrad small pb
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'pb_scale_range': 0.2,
        'max_batch_size': 2000,
        't_scale': 1.0

    }  # 2, tb large pb
)
config_list.append(
    {
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg',
        'pb_scale_range': 0.2,
        'max_batch_size': 500,
        't_scale': 1.0

    }  # 3, vargrad large pb
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'pb_scale_range': 1.0,
        'max_batch_size': 2000,
        't_scale': 2.0

    }  # 4, tb large pf and pb
)
config_list.append(
    {
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg',
        'pb_scale_range': 1.0,
        'max_batch_size': 500,
        't_scale': 2.0

    }  # 5, vargrad large pf and pb
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'pb_scale_range': 0.1,
        'max_batch_size': 2000,
        't_scale': 0.5

    }  # 6, tb small t
)
config_list.append(
    {
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg',
        'pb_scale_range': 0.1,
        'max_batch_size': 500,
        't_scale': 0.5

    }  # 7, vargrad small t
)
config_list.append(
    {
        'mode_fwd': 'tb',
        'mode_bwd': 'tb',
        'pb_scale_range': 0.1,
        'max_batch_size': 2000,
        't_scale': 1.0,
        'T': 50,

    }  # 8, tb more steps
)
config_list.append(
    {
        'mode_fwd': 'cond-tb-avg',
        'mode_bwd': 'cond-tb-avg',
        'pb_scale_range': 0.1,
        'max_batch_size': 500,
        't_scale': 1.0,
        'T': 50

    }  # 9, vargrad more steps
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
