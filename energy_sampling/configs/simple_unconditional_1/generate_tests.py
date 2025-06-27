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
temps = [0.1, 10]
losses = ['tb', 'vg']
lengths = [10, 100]
both_wayses = [True, False]
t_scales = [0.5, 1, 2]
for temp in temps:
    for loss in losses:
        for length in lengths:
            for both_ways in both_wayses:
                for t_scale in t_scales:
                    config_list.append(
                        {"T": length,
                         "energy_static_temperature": temp,
                         "mode_fwd": loss,
                         "both_ways": both_ways,
                         "t_scale": t_scale,
                         "max_batch_size": 2000,
                         }
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
