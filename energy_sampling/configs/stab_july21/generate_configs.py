"""
stab_july21 -- two 8-run batteries in one directory (0-15 for slurm arrays).

Runs 0-7 (base_toy.yaml = uncond_july20 toy_2harm base): stabilization + fast
convergence. Peak LRs sit at ~half the per-arch phase-3 breach scales measured
in uncond_july20 (first genuine tripwire fire, ramp resets excluded):
512x4 breached at scale ~0.5-0.6, 1024x4 at ~0.12-0.20, T40 runs at/never
below 1.0. Pass criterion: zero genuine cuts per run.
  0 : 1024x4 T40 lr 1e-4    -- anchor, reproduces uncond_july20's best run
  1 : 1024x4 T40 lr 2e-4    -- T40 ceiling probe from above (overshoot expected)
  2 : 1024x4 T40 lr 1e-4, warmup_steps 300  -- faster per-stage ramp
  3 : 512x4  T40 lr 5e-5    -- half its measured ceiling
  4 : 1024x4 T20 lr 1e-5    -- T-deconfound: T20 at a safe peak vs T40
  5 : 512x4  T20 lr 2.5e-5  -- same at 512
  6 : 512x4  T10 lr 3e-5    -- fastest-per-step candidate
  7 : 512x4  T10 lr 3e-5, bounding_coeff 0  -- soft-wall A/B pair with run 6

Runs 8-15 (base_elj.yaml = mk_dev as of 2026-07-21 with cluster paths,
fresh-run checkpoint fields): real-problem transfer, mipcas sg2 zp1 elj.
Known 512x4 points: 7.5e-5 flat-stable, ~3e-4 the edge.
  8  : 512x4  T25 lr 7.5e-5  -- anchor at the known-stable operating point
  9  : 512x4  T40 lr 7.5e-5  -- does the T40 win transfer
  10 : 512x4  T10 lr 7.5e-5  -- or does short-T suffice on the real problem
  11 : 1024x4 T25 lr 2e-5    -- width transfer at the toy-scaled peak
  12 : 1024x4 T40 lr 2e-5    -- the toy battery's best corner, real problem
  13 : 512x4  T25 lr 1.5e-4  -- ceiling probe (between stable and the edge)
  14 : 1024x4 T25 lr 6e-5    -- ceiling probe at 3x the scaled guess
  15 : 512x6  T25 lr 4e-5    -- depth transfer

dplr_rank stays 6 everywhere. max_batch_size 50000 flat (A100 utilization
policy; the adaptive growth finds each config's real ceiling).
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


def make_config(ind, base, family, width, layers, traj_len, peak_lr,
                warmup_steps=None, wall_off=False, note=''):
    config = deepcopy(base)

    for key in WIDTH_KEYS:
        config['model'][key] = width
    for key in LAYER_KEYS:
        config['model'][key] = layers

    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len

    for key in LR_KEYS:
        config[key] = round(float(peak_lr), 12)

    if warmup_steps is not None:
        config['adaptive_lr']['warmup_steps'] = warmup_steps

    if wall_off:
        config['energy_config']['bounding_coeff'] = 0.0
        for stage in config['protocol']['stages']:
            anneal = stage.get('balance', {}).get('anneal_coeffs', {})
            anneal.pop('bounding_coeff', None)

    config['max_batch_size'] = MAX_BATCH

    name = f"{family}_h{width}x{layers}_T{traj_len}_lr{fmt_lr(peak_lr)}"
    if warmup_steps is not None:
        name += f"_w{warmup_steps}"
    if wall_off:
        name += "_nowall"
    config['run_name'] = name

    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return {'index': ind, 'run_name': name, 'family': family,
            'width': width, 'layers': layers, 'T': traj_len,
            'peak_lr': float(peak_lr), 'warmup_steps': warmup_steps,
            'wall_off': wall_off, 'note': note}


if __name__ == '__main__':
    toy = load_base('base_toy.yaml')
    elj = load_base('base_elj.yaml')
    log = []

    # ---- toy stabilization battery: 0-7 ----
    log.append(make_config(0, toy, 'toy', 1024, 4, 40, 1.0e-4,
                           note='anchor: reproduce uncond_july20 best'))
    log.append(make_config(1, toy, 'toy', 1024, 4, 40, 2.0e-4,
                           note='T40 ceiling probe from above'))
    log.append(make_config(2, toy, 'toy', 1024, 4, 40, 1.0e-4,
                           warmup_steps=300, note='fast per-stage ramp'))
    log.append(make_config(3, toy, 'toy', 512, 4, 40, 5.0e-5,
                           note='half measured 512x4 ceiling'))
    log.append(make_config(4, toy, 'toy', 1024, 4, 20, 1.0e-5,
                           note='T-deconfound at safe peak'))
    log.append(make_config(5, toy, 'toy', 512, 4, 20, 2.5e-5,
                           note='T-deconfound at safe peak'))
    log.append(make_config(6, toy, 'toy', 512, 4, 10, 3.0e-5,
                           note='fastest-per-step candidate'))
    log.append(make_config(7, toy, 'toy', 512, 4, 10, 3.0e-5, wall_off=True,
                           note='soft-wall A/B pair with run 6'))

    # ---- real-problem transfer battery: 8-15 ----
    log.append(make_config(8, elj, 'elj', 512, 4, 25, 7.5e-5,
                           note='anchor at known-stable point'))
    log.append(make_config(9, elj, 'elj', 512, 4, 40, 7.5e-5,
                           note='T40 transfer'))
    log.append(make_config(10, elj, 'elj', 512, 4, 10, 7.5e-5,
                           note='short-T transfer'))
    log.append(make_config(11, elj, 'elj', 1024, 4, 25, 2.0e-5,
                           note='width transfer at toy-scaled peak'))
    log.append(make_config(12, elj, 'elj', 1024, 4, 40, 2.0e-5,
                           note='toy best corner on real problem'))
    log.append(make_config(13, elj, 'elj', 512, 4, 25, 1.5e-4,
                           note='512 ceiling probe (overshoot expected)'))
    log.append(make_config(14, elj, 'elj', 1024, 4, 25, 6.0e-5,
                           note='1024 ceiling probe (overshoot expected)'))
    log.append(make_config(15, elj, 'elj', 512, 6, 25, 4.0e-5,
                           note='depth transfer'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(log, f, sort_keys=False)

    print(f'Generated {len(log)} configs with log at {OUTDIR / "experiment_log.yaml"}')
