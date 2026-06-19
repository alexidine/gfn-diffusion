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


if __name__ == "__main__":
    ind = 0
    base_path = 'base.yaml'
    for zp, sg in zip(zps, sgs):
        for efunc in ['mace']:#, 'uma']:
            for traj_grad in [1]:
                for reward_grad in [0, 1]:

                    base, spec_dir = load_yaml(base_path)
                    config = deepcopy(base)

                    run_name = f"{base['run_name']}_{sg}_{zp}"
                    config['run_name'] = run_name
                    config['energy_function'] = efunc
                    config['z_primes'] = [zp]
                    config['space_groups'] = [sg]
                    config['fwd_loss_coeffs']['traj_grads'] = traj_grad
                    config['fwd_loss_coeffs']['reward_grads'] = reward_grad
                    if efunc == 'mace':
                        config[
                            'mlip_path'] = r"/scratch/mk8347/data/acr_112025_mh1_stagetwo.model"
                    config['buffer_path'] = rf"/scratch/mk8347/data/crystal_datasets/acridine/may_acridine_sg{sg}_zp{zp}_prior_dataset.pt"

                    config_path = f"{ind}.yaml"

                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    ind += 1
