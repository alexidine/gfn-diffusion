"""
uncond_july16 -- unconditional mipcas/sg2/zp1/elj architecture + rollout sweep.

16 runs:
  runs 0-11  : full grid  hidden_dim {512, 1024} x dplr_rank {0, 5, 10} x T {10, 20}
  runs 12-15 : gradient_norm_clip {0.1, 10.0} x hidden_dim {512, 1024}, at the
               grid's baseline corner (dplr_rank 0, T 10) so they're directly
               comparable to runs 0 and 6.

max_batch_size is set per hidden dim (5000 @512, 2500 @1024) -- the A100s kill us
for under-utilization, and the 512-dim models need a bigger cap to saturate.
"""

from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = Path(__file__).parent / 'base.yaml'
OUTDIR = Path(__file__).parent

# hidden dim -> max_batch_size
MAX_BATCH = {512: 5000, 1024: 2500}


def load_base():
    with BASE_PATH.open('r') as f:
        return yaml.safe_load(f)


def make_config(ind, hidden_dim, dplr_rank, traj_len, grad_clip, base):
    config = deepcopy(base)

    for key in ('s_emb_dim', 't_hidden_dim', 's_hidden_dim', 'policy_hidden_dim',
                'flow_hidden_dim', 'cond_hidden_dim'):
        config['model'][key] = hidden_dim
    config['model']['dplr_rank'] = dplr_rank

    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len

    config['gradient_norm_clip'] = grad_clip
    config['max_batch_size'] = MAX_BATCH[hidden_dim]

    name = (f"{base['run_name']}_h{hidden_dim}_dplr{dplr_rank}"
            f"_T{traj_len}_clip{grad_clip}")
    config['run_name'] = name

    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {'index': ind, 'run_name': name, 'hidden_dim': hidden_dim,
            'dplr_rank': dplr_rank, 'T': traj_len, 'gradient_norm_clip': grad_clip,
            'max_batch_size': MAX_BATCH[hidden_dim]}


if __name__ == '__main__':
    base = load_base()
    default_clip = base['gradient_norm_clip']
    log = []
    ind = 0

    # main grid: 2 x 3 x 2 = 12
    for hidden_dim in [512, 1024]:
        for dplr_rank in [0, 5, 10]:
            for traj_len in [10, 20]:
                log.append(make_config(ind, hidden_dim, dplr_rank, traj_len,
                                       default_clip, base))
                ind += 1

    # grad-clip probes at the baseline corner: 2 x 2 = 4
    for hidden_dim in [512, 1024]:
        for grad_clip in [0.1, 10.0]:
            log.append(make_config(ind, hidden_dim, 0, 10, grad_clip, base))
            ind += 1

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs with log at {OUTDIR / "experiment_log.yaml"}')
