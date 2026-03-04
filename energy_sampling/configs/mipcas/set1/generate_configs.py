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
            "name": "T25",
            "T": 25,
            "eval_T": 25,
        },
        {
            "name": "small",
            "hidden_dim": 512,
            "joint_layers": 3,

        },
        {
            "name": "large_T25",
            "hidden_dim": 1536,
            "joint_layers": 6,
            "T": 25,
            "eval_T": 25,
        },
        {
            "name": "large",
            "hidden_dim": 1536,
            "joint_layers": 6,
        },
        {
            "name": "var10_1",
            "log_var_range": 10,
        },
        {
            "name": "drift_4",
            "pb_drift_range": 0.4,
        },
        {
            "name": "var6_6",
            "log_var_range": 6,
            "pb_var_range": 6,
        },
        {
            "name": "var6_10",
            "log_var_range": 6,
            "pb_var_range": 10,
        },
        {
            "name": "var10_6",
            "log_var_range": 10,
            "pb_var_range": 6,
        },
        {
            "name": "large_flex",
            "hidden_dim": 1536,
            "joint_layers": 6,
            "T": 100,
            "eval_T": 100,
            "log_var_range": 10,
            "pb_var_range": 6,
        },
        {
            "name": "large_T25_flex",
            "hidden_dim": 1536,
            "joint_layers": 6,
            "T": 25,
            "eval_T": 25,
            "log_var_range": 10,
            "pb_var_range": 6,
        },
        {
            "name": "hiclip",
            "gradient_norm_clip": 1.0,
        },
        {
            "name": "large_hiclip",
            "gradient_norm_clip": 0.5,
            "hidden_dim": 1536,
            "joint_layers": 6,
        },
        {
            "name": "large_T25_hiclip",
            "hidden_dim": 1536,
            "joint_layers": 6,
            "T": 25,
            "eval_T": 25,
            "gradient_norm_clip": 1.0,
        },
        {
            "name": "T_25_low_t",
            "t_scale": 0.025,
            "T": 25,
            "eval_T": 25,
        },
        {
            "name": "T_25_high_t",
            "t_scale": 0.1,
            "T": 25,
            "eval_T": 25,
        },
        {
            "name": "T25_tscale_scaled",
            "T": 25,
            "eval_T": 25,
            "t_scale": 0.05 * (100 / 25),  # preserve integrated variance
        }

    ]
    """
    """

    for ind, exp in enumerate(experiments):
        config = apply_experiment(base, exp)

        run_name = f"{base['run_name']}_{exp['name']}_{ind}"
        config['run_name'] = run_name
        config['tag'] = run_name

        config_path = f"{ind}_{exp['name']}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
