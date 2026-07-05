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


def make_config(sg, zp, mol, efunc, ind):
    base, spec_dir = load_yaml(base_path)
    config = deepcopy(base)

    run_name = f"{base['run_name']}_{mol}_{sg}_{zp}_{efunc}"
    config['run_name'] = run_name
    config['energy_function'] = efunc
    if efunc == 'mace':
        config[
            'mlip_path'] = r"scratch/mk8347/data/acr_112025_mh1_stagetwo.model"
    elif efunc == 'uma':
        config['mlip_path'] = r"/scratch/mk8347/models/uma/esen_s.pt"
    config[
        'prior_path'] = rf"/scratch/mk8347/data/crystal_datasets/conditional/priors/{mol}_sg{sg}_zp{zp}_{efunc}_prior_dataset.pt"
    config['molecules_path'] = config['prior_path']
    config['tag'] = 'uncond_july_1'
    config_path = f"{ind}.yaml"

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    ind += 1


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
    for mol in ['acridine', 'mipcas', 'nehzor']:
        if mol == 'mipcas':
            sg = 2
            zp = 1
            for efunc in ['uma','elj']:
                make_config(sg, zp, mol, efunc, ind)
                ind +=1
        elif mol == 'nehzor':
            sg = 14
            zp = 1
            for efunc in ['uma','elj']:
                make_config(sg, zp, mol, efunc, ind)
                ind +=1
        elif mol == 'acridine':
            efunc = 'mace'
            for sg, zp in zip(sgs, zps):
                make_config(sg, zp, mol, efunc, ind)
                ind +=1
