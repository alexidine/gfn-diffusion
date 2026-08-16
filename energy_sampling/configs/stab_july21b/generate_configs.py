"""
stab_july21b -- refresh of the stab_july21 elj battery (0-7 for slurm arrays),
mipcas sg2 zp1 elj. Two changes from stab_july21:

1. eval_T = train T everywhere. The july21 battery evaluated at 2x train T,
   which floored wass_debiased at ~0.065 (ratio 2.0) / ~0.022 (1.43) against
   the 0.015 phase-1 exit bar and inflated mean sample energy -- controlled
   pair fch63ngx (eval_T=40, wass < 0.015 by ~2k steps) vs y510keot
   (eval_T=80, flat 0.062-0.069 for 51k steps), identical configs otherwise.
   The learned policy does not transfer across integration dt.

2. Shared pretrained prior + warmstart stage. Battery members auto-load any
   compatible prior via reuse_prior's shared-prior scan (checkpointing.
   find_shared_prior: any *_prior.pt in checkpoints_dir -- any run_name/tag --
   with an exactly matching problem_def), skip train_prior (skip_if:
   prior_loaded), and instead run the short 'warmstart' stage -- MLE+TBC at
   the member's own T/arch, exiting on matched-eval wass_debiased < 0.03 +
   tbc < 2.0 (~2-5k steps at these T) -- so the policy never enters buildout
   cold and the loaded prior is never overwritten (no snapshot_prior on the
   warmstart exit).
   As of 2026-07-21 three stab_july21-tagged runs already exited train_prior
   and left matching priors on disk (hash efd05c: T10/T100 512x4, T100
   1024x4) -- the array will pick one up (newest mtime) with no extra step.
   pretrain.yaml (below) is kept only as a from-scratch fallback for a fresh
   checkpoints_dir with no matching prior yet; run it first ONLY in that
   case, to completion of train_prior.

Grid unchanged from stab_july21 runs 8-15. Known 512x4 points: 7.5e-5
flat-stable, ~3e-4 the edge. dplr_rank stays 6 everywhere. max_batch_size
50000 flat (A100 utilization policy; the adaptive growth finds each config's
real ceiling).
  0 : 512x4  T40  lr 7.5e-5  -- anchor at the known-stable point
  1 : 512x4  T70  lr 7.5e-5  -- T transfer
  2 : 512x4  T100 lr 7.5e-5  -- T transfer
  3 : 1024x4 T40  lr 2e-5    -- width transfer at the toy-scaled peak
  4 : 1024x4 T100 lr 2e-5    -- width x long-T corner
  5 : 512x4  T40  lr 1.5e-4  -- ceiling probe (between stable and the edge)
  6 : 1024x4 T40  lr 6e-5    -- ceiling probe at 3x the scaled guess
  7 : 512x6  T40  lr 4e-5    -- depth transfer

pretrain.yaml: 512x4 T40 at the base LRs (lr_back 1e-4; MLE tolerates it),
full protocol -- kill it or let it run on after the phase-1 exit saves the
prior.
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent
MAX_BATCH = 50000
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')


def load_base(name):
    with (OUTDIR / name).open('r') as f:
        return yaml.safe_load(f)


def fmt_lr(x):
    return f"{x:.1e}".replace('e-0', 'e-')


def set_arch(config, width, layers, traj_len):
    for key in WIDTH_KEYS:
        config['model'][key] = width
    for key in LAYER_KEYS:
        config['model'][key] = layers
    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len
    config['max_batch_size'] = MAX_BATCH


def make_config(ind, base, width, layers, traj_len, peak_lr, note=''):
    config = deepcopy(base)
    set_arch(config, width, layers, traj_len)

    for key in LR_KEYS:
        config[key] = round(float(peak_lr), 12)

    name = f"elj_h{width}x{layers}_T{traj_len}_lr{fmt_lr(peak_lr)}"
    config['run_name'] = name

    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {'index': ind, 'run_name': name, 'width': width, 'layers': layers,
            'T': traj_len, 'eval_T': config['eval_T'], 'seed': config['seed'],
            'peak_lr': float(peak_lr), 'note': note}


if __name__ == '__main__':
    base = load_base('base_elj.yaml')
    log = []

    log.append(make_config(0, base, 512, 4, 40, 7.5e-5,
                           note='anchor at known-stable point'))
    log.append(make_config(1, base, 512, 4, 70, 7.5e-5, note='T transfer'))
    log.append(make_config(2, base, 512, 4, 100, 7.5e-5, note='T transfer'))
    log.append(make_config(3, base, 1024, 4, 40, 2.0e-5,
                           note='width transfer at toy-scaled peak'))
    log.append(make_config(4, base, 1024, 4, 100, 2.0e-5,
                           note='width x long-T corner'))
    log.append(make_config(5, base, 512, 4, 40, 1.5e-4,
                           note='512 ceiling probe (overshoot expected)'))
    log.append(make_config(6, base, 1024, 4, 40, 6.0e-5,
                           note='1024 ceiling probe (overshoot expected)'))
    log.append(make_config(7, base, 512, 6, 40, 4.0e-5, note='depth transfer'))

    # one-time shared-prior job, run BEFORE the array; not an array member.
    # Base LRs kept (lr_back 1e-4 for the MLE phase).
    pretrain = deepcopy(base)
    set_arch(pretrain, 512, 4, 40)
    pretrain['run_name'] = 'elj_prior_h512x4_T40'
    with open(OUTDIR / 'pretrain.yaml', 'w') as f:
        yaml.dump(pretrain, f, default_flow_style=False)

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs + pretrain.yaml with log at '
          f'{OUTDIR / "experiment_log.yaml"}')
