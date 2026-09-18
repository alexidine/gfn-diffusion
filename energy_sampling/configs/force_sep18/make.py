"""force_sep18 -- reshaped terminal-force ladder on the forward seat, LOCAL (RTX 5080),
P_B frozen, off the pg14_p1_cont MLE run's BEST checkpoint (step 80050, still in the
train_prior stage -- it never met the phase-1 exit gate -- T=10, batch 1000).

WHY. dose_sep16's two force arms lost after an early gain. Read against their controls
and against per-sample measurements on the frozen mip checkpoint (2026-09-18): the
truncated terminal-force path lowers the residual by RELOCATING samples -- over-dense
rows slide downhill (search, the early gain), under-dense rows slide uphill away from
states that should gain probability (the loss keeps improving while the backward
sensors on fixed prior rows degrade). 30% of the raw push was that flight and ~1% of
rows (clash forces 100x the median) set its direction; the shipped reward_grad_clip
clips d loss/d log R, which the Huber already bounds, so it was inert. Scored offline,
gating the term to residuals above ~1 nat and clipping |F| per row put ~75% of the push
on rows where descent is the right move (raw: ~25%).

ARMS (fwd_loss_coeffs; everything else identical):
  ctrl_pb   k 0, reward_grads 0          the density route alone
  raw_pb    k 1, reward_grads 1          the dose16_fs_k1 form (clip inert), P_B frozen
  gc_pb     k 1, gate 1.0, |F| clip 250  reshaped: descent only, tail bounded
  gcs_pb    gc + path_grad_scale 0       ... and the noise-scale channel detached
  g0c_pb    k 1, gate 0.0, |F| clip 250  the gate threshold's sensitivity
  g_pb      k 1, gate 1.0, no clip       added 14:10 after gc_pb: the clip cost half the search gain
                                          and all of the clash-tail cleanup; this isolates gate from clip

SEAT. Rollout every step, fwd trains the policy at a pinned 0.3 share, bwd 0.5 /
replay 0.2 with the ramp frozen; constant LR (burn_in_scale == fixed_scale); P_B
frozen by the top-level key (snapshotted from the MLE weights at load). RESUME
MECHANICS: the seed is a train_prior-stage frame, so that stage's exit gate
(gates/progress_done) is replaced by an always-true term; at the first eval (250 steps
of MLE at the fixed rate) the run exits train_prior (snapshot:phase1_exit and
snapshot_prior fire), enters equilibration (rebuild_prior_by_churn, bootstrap_z) and the
forward seat takes over. The ELJ energy runs under MXT_LEGACY_TRICLINIC_WALLS=1 at
launch (the seed predates the Niggli walls).

READ. Z and eval_fwd/tb_err at matched steps (the gain); bwd/tb_err, bwd/tb_resid,
bwd/under_coverage and zmatch/delta_mean (the damage sensors -- forward improving while
these degrade is the relocation fingerprint); rewardgrad/gate_frac, force_clipped_frac,
force_norm_max, push_norm_mean; Excess Energy P50/P90/P99 at eval.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'fs18'
SEED_FILE = 'pg14_p1_cont_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1_best.pt'
RESUME_STEP = 80050
STUB_STEPS = 250   # one eval period of MLE before the always-true exit fires
STEPS = 3000
FWD_SHARE = 0.3
FORCE_CLIP = 250.0


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fwd_seat(cfg):
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'equilibration'], [s['name'] for s in stages]
    tp = stages[0]
    assert tp['exit'] == [{'metric': 'gates/progress_done', 'above': 0.5, 'patience': 1}], tp['exit']
    tp['exit'] = [{'metric': 'bwd/mle', 'above': -1.0e9, 'patience': 1}]
    eq = stages[1]
    eq['fwd_rollout_every'] = 0
    eq['z_pin_rollout_every'] = 0
    eq.pop('fwd_rollout_triggers', None)
    eq['fracs'] = {'fwd': FWD_SHARE, 'bwd': 0.5, 'replay': 0.2}
    eq['balance']['pinned'] = {'fwd': FWD_SHARE}
    eq['balance']['up'] = 1.0e-4
    eq['balance']['down'] = 1.0e-4
    eq['balance']['bounds'] = {'bwd': [0.25, 0.6], 'replay': [0.1, 0.45]}
    eq['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 0.0}
    return cfg


def common(cfg, name, steps=STEPS):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75
    cfg['epochs'] = RESUME_STEP + STUB_STEPS + steps
    cfg['eval_period'] = 250
    cfg['eval_num_samples'] = 2500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['freeze_backward_policy'] = 'full'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed'
    lc['burn_in_scale'] = lc['fixed_scale']
    lc['repeat_every'] = 0
    lc['fire_cut_factor'] = 1.0
    # the seed's stored problem_def predates energy_config.physical_energy_clip; absent = the code default
    cfg['energy_config'].pop('physical_energy_clip', None)
    assert cfg['checkpoint_name'].startswith('dev_mk_dev_') and cfg['checkpoint_name'].endswith('_phase1_exit.pt')
    cfg['checkpoint_name'] = SEED_FILE   # a _best.pt: full resume, stage train_prior
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    assert int(cfg['integrator']['T']) == 10, cfg['integrator']['T']
    return cfg


def arm(name, k, rg, gate=None, force_clip=0.0, scale=1, steps=STEPS):
    cfg = common(fwd_seat(base()), name, steps=steps)
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = int(k)
    fc['reward_grads'] = float(rg)
    fc['reward_grad_clip'] = 0.0
    fc['reward_grad_gate'] = None if gate is None else float(gate)
    fc['reward_grad_force_clip'] = float(force_clip)
    fc['path_grad_scale'] = int(scale)
    fc['traj_grads'] = 0.0
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:16s} k={k} reward_grads={rg} gate={gate} force_clip={force_clip} path_grad_scale={scale} steps={steps}')
    return out


if __name__ == '__main__':
    arm('ctrl_pb', 0, 0.0)
    arm('raw_pb', 1, 1.0)
    arm('gc_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP)
    arm('gcs_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP, scale=0)
    arm('g0c_pb', 1, 1.0, gate=0.0, force_clip=FORCE_CLIP)
    arm('g_pb', 1, 1.0, gate=1.0, force_clip=0.0)
    arm('smoke_gc_pb', 1, 1.0, gate=1.0, force_clip=FORCE_CLIP, scale=0, steps=60)   # crosses the stub exit at +250, then 60 fused steps
