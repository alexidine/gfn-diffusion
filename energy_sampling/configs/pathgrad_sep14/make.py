"""pathgrad_sep14 -- forward-seat path/reward gradient ladder, LOCAL (RTX 5080).

QUESTION. Does the forward branch learn faster with the reparameterised path
gradient and the reward gradient than with the detached density route alone?
The record never separated the two: the historical 'very destabilizing' result
was full-T, unclipped LJ, all-or-nothing; path_grad_last_k 1 alone was inert
(no reward gradient, so nothing directional reaches the path); the one arm that
enabled reward_grads died at step 1 on NaN through the clamp hook; and
configs/local_aug02's separating arms were never run.

DESIGN. Unconditional MIPCAS ELJ off mk_dev's own phase-1 exit (step 430, T=10,
full resume, same as a plain mk_dev launch). The equilibration stage is put on
the FORWARD SEAT: rollout every step, fwd trains the policy (freeze_policy 0)
at a pinned 0.3 share, bwd/replay split the rest with the ramp near-frozen.
One knob per arm on the GLOBAL fwd_loss_coeffs (stage overrides reject these
keys):

    ctrl      k 0,  rg 0    the shipped density route
    k1_rg1    k 1,  rg 1    reward gradient through the last step only
    k3_rg1    k 3,  rg 1    three steps of Jacobian product
    k10_rg1   k 10, rg 1    full T -- the historical blow-up, now sanitised
    k1_rg0    k 1,  rg 0    path term with NO reward gradient (expected inert)

Every arm carries the source sanitiser (nan_to_num on d log R / d x_T) and
reports rewardgrad/nonfinite_frac, rewardgrad/clipped_frac and the per-step
pathgrad/state_grad_step{i} norms, so a Jacobian blow-up is SEEN, not inferred.
reward_grad_clip 10 on the rg 1 arms.

LR is held CONSTANT: burn_in_scale == fixed_scale == 0.2 (2.5e-5), repeat_every
0, fire_cut_factor 1 -- the same shape prod_sep12 uses -- so an arm that fires
and dies is the measurement.

SCORE. fwd/tb_err vs the control at matched steps, the fwd/bwd tb_err RATIO
(pinned ~2.7x everywhere; a real forward fix must close it), Mean Sample Energy
/ Excess Energy at eval, and the pathgrad/rewardgrad channels.
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'pg14'
RESUME_STEP = 430          # mk_dev's phase-1 exit (modeller_state.step_ind)
STEPS = 3000
FWD_SHARE = 0.3

# SEEDS. pg14seed = mk_dev's own Aug-10 phase-1 exit (step 430, an under-trained
# MLE: the buffer is not absorbed, under_coverage crawls, the ramp starves
# replay -- owner 2026-09-15). pg15seed = the Aug-23 dev_elj_p2_cruise phase-1
# exit (step 2910, same problem identity under normalize_problem_def), copied and
# lj-migrated the same way. Both are COPIES; the originals are untouched.
SEEDS = {
    'pg14seed': ('pg14seed_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_phase1_exit.pt', 430),
    'pg15seed': ('pg15seed_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1_phase1_exit.pt', 2910),
}


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fwd_seat(cfg):
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'equilibration']
    eq = stages[1]
    eq['fwd_rollout_every'] = 0          # rollout every step: the forward seat
    eq['z_pin_rollout_every'] = 0
    eq.pop('fwd_rollout_triggers', None)  # dead (refused) without a cadence
    eq['fracs'] = {'fwd': FWD_SHARE, 'bwd': 0.5, 'replay': 0.2}
    eq['balance']['pinned'] = {'fwd': FWD_SHARE}
    # the ramp must be in (0, 1]; 1e-4 per tick is a frozen split for 3000 steps
    eq['balance']['up'] = 1.0e-4
    eq['balance']['down'] = 1.0e-4
    # bounds must sit inside the split pair's mass (1 - FWD_SHARE = 0.7)
    eq['balance']['bounds'] = {'bwd': [0.25, 0.6], 'replay': [0.1, 0.45]}
    eq['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 0.0}
    return cfg


def common(cfg, name, seed='pg14seed', steps=STEPS):
    seed_file, resume_step = SEEDS[seed]
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75   # desktop co-tenancy on this box
    cfg['epochs'] = resume_step + steps
    cfg['eval_period'] = 250
    cfg['eval_num_samples'] = 2500       # ELJ eval is cheap; 10000 set the VRAM peak
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed'
    lc['burn_in_scale'] = lc['fixed_scale']   # constant rate from step one
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    # The working-tree mk_dev carries energy_config.physical_energy_clip (null),
    # an UNCOMMITTED key that enters the problem hash, so mk_dev's own phase-1
    # exit no longer matches it (load_full compares problem_def with no
    # exemption). Absent key = the code default None, the same behaviour.
    cfg['energy_config'].pop('physical_energy_clip', None)
    # full resume off a COPY of mk_dev's phase-1 exit: the Aug-10 sidecar predates
    # the lj_coeff currency stamp and is refused as-is; the copy was migrated
    # once with migrate_buffer_sidecar.py (--lj-coeff 0.3635836825, mipcas) so
    # the owner's original stays untouched.
    assert cfg['checkpoint_name'].startswith('dev_mk_dev_') and cfg['checkpoint_name'].endswith('_phase1_exit.pt')
    cfg['checkpoint_name'] = seed_file
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    return cfg


def arm(name, k, rg, clip):
    cfg = common(fwd_seat(base()), name)
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = int(k)
    fc['reward_grads'] = float(rg)
    fc['reward_grad_clip'] = float(clip)
    fc['traj_grads'] = 0.0               # truncation is the only path-live switch
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} k={k} reward_grads={rg} clip={clip}')
    return out


def replay_arm(name, k, rg, clip, seed='pg14seed'):
    """REPLAY SEAT: mk_dev's stock equilibration stage (rare rollouts, 1 in 20,
    fwd Z-only) with the LAST k steps of every replayed trajectory re-sampled
    live and the reward re-scored on the fresh terminal (replay_loss_coeffs
    .resample_last_k); reward_grads lets d log R / d x_T reach the policy through
    that tail. rs_ctrl is the stock stage untouched -- the rare-rollout design
    as it ships. rs_k1_rg0 re-samples the tail WITHOUT the reward gradient, so
    the tail's own entropy/bridge term is separated from the reward payload."""
    cfg = common(base(), name, seed=seed)
    rc = cfg['replay_loss_coeffs']
    rc['resample_last_k'] = int(k)
    rc['reward_grads'] = float(rg)
    rc['reward_grad_clip'] = float(clip)
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} replay resample_last_k={k} reward_grads={rg} clip={clip}')
    return out


def stored_force_arm(name, k, mode='implied', seed='pg14seed'):
    """REPLAY SEAT + STORED FORCE: rare rollouts as shipped; every reporting
    rollout records d log R / d x_T (fwd terminal_force) into the buffer's force
    column, and replay re-propagates the last k stored steps through their
    implied noise and pushes x_T along that force. NO energy call at replay
    time -- the production form of the last-step reward gradient."""
    cfg = common(base(), name, seed=seed)
    rc = cfg['replay_loss_coeffs']
    rc['stored_force_k'] = int(k)
    rc['stored_force_mode'] = str(mode)
    rc['resample_last_k'] = 0
    rc['reward_grads'] = 0.0
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} replay stored_force_k={k} mode={mode}')
    return out


def pin_replay_share(path, name, replay=0.3):
    """Re-emit an existing replay-seat arm with the replay share PINNED (ramp
    frozen at 1e-4/tick), so the dose of the replay-side term is a setting
    rather than the gated ramp's response to the bwd sensor."""
    cfg = yaml.safe_load(open(HERE / f'{path}.yaml', encoding='utf-8'))
    cfg['run_name'] = name
    eq = cfg['protocols']['unconditional_tb']['stages'][1]
    eq['fracs'] = {'fwd': 0.0, 'bwd': round(1.0 - replay, 3), 'replay': replay}
    eq['balance']['pinned'] = {'fwd': 0.0}
    eq['balance']['up'] = 1.0e-4
    eq['balance']['down'] = 1.0e-4
    eq['balance']['bounds'] = {'bwd': [0.25, 0.9], 'replay': [0.1, 0.75]}
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:18s} from {path} with replay share pinned at {replay}')


if __name__ == '__main__':
    stored_force_arm('rs_sf_k1', 1)
    stored_force_arm('rs_sf_k1_v2', 1)   # forces on EVERY admission path
    stored_force_arm('rs_sf_k1_mu', 1, mode='mean')
    stored_force_arm('rs_sf_k3', 3)
    replay_arm('rs_ctrl', 0, 0.0, 0.0)
    replay_arm('rs_k1_rg1', 1, 1.0, 10.0)
    replay_arm('rs_k1_rg0', 1, 0.0, 0.0)
    replay_arm('rs_k3_rg1', 3, 1.0, 10.0)
    arm('ctrl', 0, 0.0, 0.0)
    arm('k1_rg1', 1, 1.0, 10.0)
    arm('k3_rg1', 3, 1.0, 10.0)
    arm('k10_rg1', 10, 1.0, 10.0)
    arm('k1_rg0', 1, 0.0, 0.0)
    pin_replay_share('rs_ctrl', 'rs_ctrl_pin')
    pin_replay_share('rs_k1_rg1', 'rs_k1_rg1_pin')
    pin_replay_share('rs_sf_k1_mu', 'rs_sf_k1mu_pin')
    pin_replay_share('rs_sf_k1', 'rs_sf_k1_pin')
    # second seed (the step-2910 cruise phase-1 exit): the stored-force baseline pair
    replay_arm('s2_rs_ctrl', 0, 0.0, 0.0, seed='pg15seed')
    stored_force_arm('s2_rs_sf_k1', 1, seed='pg15seed')
