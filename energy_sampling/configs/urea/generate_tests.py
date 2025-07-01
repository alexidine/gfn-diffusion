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

#
# Latent unimodal hot
# Latent unimodal cold
# Latent unimodal conditional T
# Crystal unimodal hot
# Crystal unimodal cold
# Crystal unimodal conditional T
# Latent Multimodal hot
# Latent Multimodal cold
# Latent Multimodal conditional T
# Crystal Multimodal hot
# Crystal Multimodal cold
# Crystal Multimodal conditional T

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

"""
0: good energies, very odd high variance behavior. Not enough eploration
1: worse overlaps, higher density, wacky high variance behavior in fewer dimensions
2: crashed
3: same high-variance behavior. Still training.
4: not enough good modes exploration. Distributions are actually kindof nice, but not sharp enough
5: variance explosion issue in several dimensions. C dimension is extremely saturated, causing wild density fluctuations
6: crashed
7: variance issue
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
