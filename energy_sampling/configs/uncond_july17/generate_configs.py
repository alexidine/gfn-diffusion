"""
uncond_july17 -- unconditional toy_2harm battery on the new protocol.py stage
system (base.yaml = mk_dev as of 2026-07-17 with cluster paths). Follows
uncond_july16's structure; sweeps architecture depth & width, off-diagonal
(DPLR) covariance, trajectory rollout length, and LR level.

16 runs:
  runs 0-7   : main grid  width {512, 1024} (4 layers) x dplr_rank {0, 6} x T {10, 20}
  runs 8-11  : depth probes  layers {6, 8} x width {512, 1024} at the mk_dev
               operating corner (dplr 6, T 10) -- compare to runs 2 (512x4) and
               6 (1024x4)
  runs 12-13 : rollout probes  T 40 at width {512, 1024}, dplr 6 -- extends the
               dplr-6 T series {10, 20, 40} (runs 2/3 and 6/7)
  runs 14-15 : LR probes  x3 and x0.33 on lr_policy/lr_back/lr_replay/lr_fused
               together (lr_flow untouched -- Z head is pinned anyway) at
               1024x4, dplr 6, T 10 -- compare to run 6

max_batch_size is 50000 flat: the A100s kill under-utilized jobs, and the
adaptive batch growth / OOM backoff finds each architecture's real ceiling on
its own, so the cap just needs to be out of the way.
"""

from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = Path(__file__).parent / 'base.yaml'
OUTDIR = Path(__file__).parent

MAX_BATCH = 50000
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')


def load_base():
    with BASE_PATH.open('r') as f:
        return yaml.safe_load(f)


def make_config(ind, width, layers, dplr_rank, traj_len, lr_mult, base):
    config = deepcopy(base)

    for key in WIDTH_KEYS:
        config['model'][key] = width
    for key in LAYER_KEYS:
        config['model'][key] = layers
    config['model']['dplr_rank'] = dplr_rank

    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len

    for key in LR_KEYS:
        config[key] = round(float(base[key]) * lr_mult, 12)

    config['max_batch_size'] = MAX_BATCH

    name = (f"{base['run_name']}_h{width}x{layers}_dplr{dplr_rank}"
            f"_T{traj_len}_lr{lr_mult:g}")
    config['run_name'] = name

    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {'index': ind, 'run_name': name, 'width': width, 'layers': layers,
            'dplr_rank': dplr_rank, 'T': traj_len, 'lr_mult': lr_mult,
            'lr_policy': round(float(base['lr_policy']) * lr_mult, 12),
            'max_batch_size': MAX_BATCH}


if __name__ == '__main__':
    base = load_base()
    log = []
    ind = 0

    # main grid: 2 x 2 x 2 = 8
    for width in [512, 1024]:
        for dplr_rank in [0, 6]:
            for traj_len in [10, 20]:
                log.append(make_config(ind, width, 4, dplr_rank, traj_len, 1.0, base))
                ind += 1

    # depth probes at the operating corner (dplr 6, T 10): 2 x 2 = 4
    for width in [512, 1024]:
        for layers in [6, 8]:
            log.append(make_config(ind, width, layers, 6, 10, 1.0, base))
            ind += 1

    # rollout probes, T 40 at dplr 6: 2
    for width in [512, 1024]:
        log.append(make_config(ind, width, 4, 6, 40, 1.0, base))
        ind += 1

    # LR probes at 1024x4, dplr 6, T 10: 2
    for lr_mult in [3.0, 0.33]:
        log.append(make_config(ind, 1024, 4, 6, 10, lr_mult, base))
        ind += 1

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs with log at {OUTDIR / "experiment_log.yaml"}')
