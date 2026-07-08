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


def make_config(sg, zp, mol, efunc, ind, T, traj_len):
    base, spec_dir = load_yaml(base_path)
    config = deepcopy(base)

    run_name = f"{base['run_name']}_{mol}_{sg}_{zp}_{efunc}"
    config['run_name'] = run_name
    config['energy_function'] = efunc
    if efunc == 'mace':
        config[
            'mlip_path'] = r"/scratch/mk8347/data/acr_112025_mh1_stagetwo.model"
    elif efunc == 'uma':
        config['mlip_path'] = r"/scratch/mk8347/models/uma/esen_s.pt"
    config[
        'prior_path'] = rf"/scratch/mk8347/data/crystal_datasets/conditional/priors/{mol}_sg{sg}_zp{zp}_{efunc}_prior_dataset.pt"
    config['molecules_path'] = config['prior_path']
    config['tag'] = 'uncond_july_1'
    config['energy_config']['temperature'] = T
    config['integrator']['T'] = traj_len
    config['eval_T'] = traj_len
    config_path = f"{ind}.yaml"

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    ind += 1


if __name__ == "__main__":
    ind = 0

    base_path = 'base.yaml'
    traj_len = 100
    for T in [2.5]:
        for mol in ['mipcas']:
            for rank in [0, 2, 4, 6, 8, 10]:
                sg = 2
                zp = 1
                for efunc in ['elj']:

                    base, spec_dir = load_yaml(base_path)
                    config = deepcopy(base)

                    run_name = f"{ind}_{base['run_name']}_{mol}_{sg}_{zp}_{efunc}"
                    config['run_name'] = run_name
                    config['energy_function'] = efunc
                    if efunc == 'mace':
                        config[
                            'mlip_path'] = r"/scratch/mk8347/data/acr_112025_mh1_stagetwo.model"
                    elif efunc == 'uma':
                        config['mlip_path'] = r"/scratch/mk8347/models/uma/esen_s.pt"
                    config[
                        'prior_path'] = rf"/scratch/mk8347/data/crystal_datasets/conditional/priors/{mol}_sg{sg}_zp{zp}_{efunc}_prior_dataset.pt"
                    config['molecules_path'] = config['prior_path']
                    config['tag'] = 'cov_1'
                    config['energy_config']['temperature'] = T
                    config['integrator']['T'] = traj_len
                    config['eval_T'] = traj_len
                    config['model']['dplr_rank'] = rank
                    config_path = f"{ind}.yaml"

                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    ind +=1



