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
    'mode_fwd': "cond-tb-avg",
    'mode_bwd': "cond-tb-avg"
    },  # 0 VarGrad
    {
        'mode_fwd': "tb",
        'mode_bwd': "tb"
    },  # 1 TB

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
