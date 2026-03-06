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

    """
    Goal: throw everything at the wall and see what gets us the best TB convergence
    Baseline model, and then areas to probe:
        - SDE discretization, SDE flexibility, and model size need to be tested individually and in combination
            - flexibility of forward AND backward policy, something like
                - forward: 2, 6, 10
                - backward: 2, 6, 10
            - model size (couple layers & hidden dim)
                - normal: 1024x4
                - large: 1536x6
                - small: 512x3
            - 100 vs 25 vs 10 integration steps
                - I want to see specifically is a high capacity, high flexibility model can converge in few SDE steps
        - check also the grad norm clip - particularly on large models could be important

    Probably after this set we then check with the optimal model
        - small temperature linear scan 
        - small lr linear scan 
        - one run without the layer norm

    """


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


EXPERIMENT_TEMPLATE = {
    "name": None,

    # SDE
    "T": None,
    "eval_T": None,
    "t_scale": None,
    "log_var_range": None,
    "pb_drift_range": None,
    "pb_var_range": None,

    # OPT
    "lr_policy": None,
    "lr_flow": None,
    "lr_back": None,
    "gradient_norm_clip": None,

    # MODEL
    "joint_layers": None,
    "hidden_dim": None,
    "s_emb_dim": None,
    "condition_emb_dim": None,
    "norm": None,

    # PHYSICS
    "energy_function": None,
    "energy_static_temperature": None,
}

if __name__ == "__main__":
    base_path = 'base.yaml'
    base, spec_dir = load_yaml(base_path)

    experiments = [
        {
            "name": "baseline"
        },
        {
            "name": 'large',
            'lr_policy': 5e-5,
            'lr_back': 5e-5,
            "hidden_dim": 1536,
            "s_emb_dim": 1536,
            "joint_layers": 6,
        },
        {
            "name": 'large_T25',
            'lr_policy': 5e-5,
            'lr_back': 5e-5,
            "hidden_dim": 1536,
            "s_emb_dim": 1536,
            "joint_layers": 6,
            'T': 25,
            'eval_T': 25
        },
        {
            "name": 'var10_10',
            'pb_var_range': 10,
            'log_var_range': 10,
        },
        {
            "name": 'var14_14',
            'pb_var_range': 14,
            'log_var_range': 14,
        },
        {
            "name": 'nonorm',
            "norm": 'none'
        },
        {
            "name": 'T150',
            "T": 150,
            "eval_T": 150,
        },
        {
            "name": 'large_lowlr',
            'lr_policy': 1e-5,
            'lr_back': 1e-5,
            "hidden_dim": 1536,
            "s_emb_dim": 1536,
            "joint_layers": 6,
        },
        {
            "name": 'hot',
            'energy_static_temperature': 5,
        }
    ]
    """
    """

    for ind, exp in enumerate(experiments):
        config = apply_experiment(base, exp)

        run_name = f"{base['run_name']}_{exp['name']}_{ind}"
        config['run_name'] = run_name

        config_path = f"{ind}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
