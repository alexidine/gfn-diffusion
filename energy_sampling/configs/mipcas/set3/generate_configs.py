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
    # notes
    # lr_flow is inconsistently affected by the warmup ratio - fix in code. Hopefully doesn't break experiments too much
    # A lot of these are getting an LR cut at almost the same exact moment - possibly the witch from MLE to TB losses? That would make a lot of sense.
    # I think I give it a grace period then
    # actually - bad theory. The cuts come before the training phase update. Curious

    experiments = [
        # fixed time, varying ratio (starting LR sensitivity)
        {"name": "t1k_r10", "lr_warmup_ratio": 10, "lr_warmup_time": 1000}, # LR cut before peak, then another during LR decrease (visible loss explosion, with immediate recovery). Similar otherwise to baseline (almost the same LR in the end)
        {"name": "t1k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 1000}, # Very similar profile to baseline. LR cut at almost the same moment
        {"name": "t1k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 1000},  # Same policy LR profile, but due to vaguery above the LR flow was about 1/4 as large? Backward losses are much worse, forward losses are vastly better. It's overall rather odd.

        # fixed ratio, varying time (ramp slope sensitivity)
        {"name": "t3k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 3000}, # crashed, node failed after 6k steps. One LR cut late (loss explosion) at a high LR so definitely more stable.Loss profile overall similar, though slope is maybe developing a bit better? Hard to tell.
        {"name": "t10k_r25", "lr_warmup_ratio": 25, "lr_warmup_time": 10000},  # Still warming up. Stage 1 training slower. Overall similar profiles though. Maybe again developing a better slope?
        # suspect from these two that maintaining a higher LR in to later training might be advantageous - if one can get through the unstable earlier sections
        # we need at least a moderate LR to converge phase 1, and it is in general quite stable

        # low starting LR + long ramp (most conservative)
        {"name": "t3k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 3000},  # due to above artifact, the flow lr was low for these. LR cut at 5e-4 (later and higher than baseline). Loss profile kindof similar? Slope not promising
        {"name": "t10k_r100", "lr_warmup_ratio": 100, "lr_warmup_time": 10000}, # very slow phase 1, no cuts yet but still hasn't reached the cut range (usually around 4e-4 or higher).

        # baseline for comparison
        {"name": "t1k_r25_baseline", "lr_warmup_ratio": 25, "lr_warmup_time": 1000},  # LR cut near 1e-4. Reasonable looking convergence behavior.
        # vs our default model baseline: larger models above are LR cutting earlier, driving a hiccup which destabilizes early training a bit, driving slightly higher losses. We don't have terminal convergence data though.
    ]

    for ind, exp in enumerate(experiments):
        config = apply_experiment(base, exp)

        run_name = f"{base['run_name']}_{exp['name']}_{ind}"
        config['run_name'] = run_name

        config_path = f"{ind}.yaml"

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
