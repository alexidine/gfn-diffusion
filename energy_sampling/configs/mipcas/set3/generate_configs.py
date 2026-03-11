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
    "lr_warmup_ratio": None,
    "lr_warmup_time": None,


}

if __name__ == "__main__":
    base_path = 'base.yaml'
    base, spec_dir = load_yaml(base_path)

    experiments = [
        # fixed time, varying ratio (starting LR sensitivity)
        {"name": "t1k_r10", "lr_warmup_ratio": 10, "lr_warmup_time": 1000},
        {"name": "t1k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 1000},
        {"name": "t1k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 1000},

        # fixed ratio, varying time (ramp slope sensitivity)
        {"name": "t3k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 3000},
        {"name": "t10k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 10000},

        # low starting LR + long ramp (most conservative)
        {"name": "t3k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 3000},
        {"name": "t10k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 10000},

        # baseline for comparison
        {"name": "t1k_r25_baseline", "lr_warmup_ratio": 25, "lr_warmup_time": 1000},
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
