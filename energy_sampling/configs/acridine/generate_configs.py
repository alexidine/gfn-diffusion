from copy import deepcopy
import os
from itertools import product

import yaml
from pathlib import Path
from copy import deepcopy


def load_yaml(path):
    path = Path(path)
    with path.open('r') as f:
        return yaml.safe_load(f), path.parent  # return both content and its directory


def apply_experiment(base_config, experiment):
    config = deepcopy(base_config)

    for k, v in experiment.items():
        if k == "name":
            continue
        if v is not None:
            config[k] = v
        if v == 'none':
            config[k] = None

    return config


zps = []
sgs = []
zps.append(1)  # Form II & IX
sgs.append(14)
zps.append(2)  # Form III & VII
sgs.append(14)
zps.append(2)  # Form VI
sgs.append(9)
zps.append(3)  # Form IV
sgs.append(19)

if __name__ == "__main__":
    ind = 0
    base_path = 'base.yaml'
    for zp, sg in zip(zps, sgs):
        for efunc in ['mace', 'uma']:

            base, spec_dir = load_yaml(base_path)
            config = deepcopy(base)

            run_name = f"{base['run_name']}_{sg}_{zp}"
            config['run_name'] = run_name
            config['energy_function'] = efunc
            if efunc == 'mace':
                config[
                    'mlip_path'] = r"scratch/mk8347/data/acr_112025_mh1_stagetwo.model"
            elif efunc == 'uma':
                config['mlip_path'] = r"/scratch/mk8347/models/uma/esen_s.pt"
            config['tag'] = 'run_1'
            config_path = f"{ind}.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            ind += 1
