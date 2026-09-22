"""rr_sep19 -- the rare-rollout convergence ladder on the T=100 mipcas archive, LOCAL (RTX 5080).

GOAL (owner 2026-09-19): a rare-rollout config that converges to the identified endpoint, the N=1
frozen-P_B arm of dose_sep16 (n1_t6_pbfrozen: Z 35.5-35.9 monotone, fwd tb_err 2.9, held-out gap
~0.07, mean sample energy -128.7 kJ/mol), starting from the step-160000 archive (Z 28.8) so the
run has the climb to make. Build up from basics: N 5 or 10, tau 600, P_B frozen, NO reward-gradient
forces anywhere, and the gated ramp UNPINNED with its ceiling raised to replay 0.95 / bwd 0.05.

BASE: dose_sep16d's n5_t600_pb, the cluster arm that reached 35.25 at 13.5k steps (batch 1600,
replay pinned 0.3), localised the way force_sep18c does (same archive, same problem hash a8b328).

  n5         fwd_rollout_every 5,  seed_lr 1.25e-4 (rate 1.0), tau 600. Differs from the cluster's
             n5_t600_pb in the unpinned ramp, the 95:5 bounds, batch 400 and anchors-only churn.
  n10        fwd_rollout_every 10, seed_lr 0.625e-4 (rate 0.5): the same dose per row (passes x
             rate) at half the energy calls. Next rung if n5 memorises: rate/4.
  smoke_n5   n5 at 30 steps.

DOSE. Under store-all every replay row is trained on N times before it expires (passes = N), the
LR is the pressure per pass, and the share sets how much of the update is replay vs bwd (fracs are
loss weights on full batches; Adam sees the relative weight). tau does not enter the dose: it sets
the pool (B x tau / N = 48k rows at N=5, 24k at N=10; the cluster n5 pool was 192k) and staleness.

CONTROLLER ON THIS RESUME. The archive sits at replay 0.1 / bwd 0.9 with the ratchet TRIPPED:
gr_best 1.20 was set at a log-Z maximum (step 124670) and the level bwd/under_coverage is 2.39.
The level FALLS with log Z (slope -0.14 per nat on fs18c's climb, -0.10 on n1_t6_pbfrozen's), so
it crosses the release line best + 0.25 = 1.45 near Z ~33, after which the ramp moves: +0.004
share per 10-step tick while the 150-step rise sensor (bar 0) is quiet, -0.006 when it fires, so
the share climbs only while the sensor is quiet on more than 60% of ticks. No stage_ctrl reset is
needed for THIS seed; a seed already at high Z would carry a stale reference (code, not config).
The cooldown does not re-run on a resume (gr_entry_step 101820 rides the checkpoint).

READ at matched steps against the endpoint: replay/log_Z_learned (level, last-2k sd),
eval_fwd/tb_err, replay/val_gap_nats (held-out gap), mean sample energy, Replay Frac (where the
controller parks), protocol/gr_tripped and gr_fired. Under the level with a widening held-out gap
= over-dosed, cut the rate. At the level = the production candidate at that N.

IDENTITY. As force_sep18c: prior_path keeps the cluster string (part of the stored problem hash;
a copy sits at C:\scratch\mk8347\...), molecules_path is the D: copy, batch 400 clamps the
restored 1600, freeze_backward_policy is top-level so it applies on resume. ELJ runs under
MXT_LEGACY_TRICLINIC_WALLS=1 (the archive predates the Niggli walls).
"""
import copy
import importlib.util
import pathlib
import sys
from argparse import Namespace

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent
TAG = 'rr19'
BASE = ROOT / 'dose_sep16d' / 'dose16d_n5_t600_pb.yaml'
CKPT_DIR = 'D:\\crystal_datasets\\checkpoints\\'
SEED = 'p12_p12_mip_lr1_elj-mipcas_sg2_zp1_elj_200k_prior_dataset-T2.5-a8b328_step160000.pt'
PRIOR_MODEL = 'p12_p12_mip_lr1_elj-mipcas_sg2_zp1_elj_200k_prior_dataset-T2.5-a8b328_prior.pt'
SEED_HASH = 'a8b328'
RESUME_STEP = 160000
STEPS = 3000   # owner 2026-09-19: 2-3k steps max per test arm (5 tau at tau 600; ~1.8 h at ~2 s/step on the 5080)
BATCH = 400
TAU = 600
BASE_LR = 1.25e-4
LOCAL_MOLECULES = 'D:\\crystal_datasets\\conditional\\priors\\mipcas_sg2_zp1_elj_200k_prior_dataset.pt'
BOUNDS = {'bwd': [0.05, 0.9], 'replay': [0.1, 0.95]}   # owner: let it go up to 95:5
ENTRY_FRACS = {'fwd': 0.0, 'bwd': 0.9, 'replay': 0.1}   # where the archive sits; entry fracs are not re-applied on a resume

_spec = importlib.util.spec_from_file_location('dosemake', ROOT / 'dose_sep16' / 'make.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


def _ns(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def problem_hash_of(cfg):
    sys.path.insert(0, str(ES.parent))
    from energy_sampling.utils import get_problem_definition, normalize_problem_def, problem_hash
    return problem_hash(normalize_problem_def(get_problem_definition(_ns(cfg))))


#: the HOLD test (owner 2026-09-19 18:40: "converging from a known good but expensive regime into a healthy
#: rare rollout one"): resume the CONVERGED N=1 frozen-P_B arm (dose16e_n1_pbfrozen best, step 162250,
#: Z 34.85, replay pinned 0.3, tau 120) and switch it into N=10 through a real transition (copied stage).
HOLD_BASE = ROOT / 'dose_sep16e' / 'dose16e_n1_pbfrozen.yaml'
HOLD_SEED = 'dose16e_dose16e_n1_pbfrozen_elj-mipcas_sg2_zp1_elj_200k_prior_dataset-T2.5-a8b328_best.pt'
HOLD_RESUME_STEP = 162250


def base(path=BASE):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def common(cfg, name, steps, seed=SEED, resume_step=RESUME_STEP):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    cfg['cuda_memory_fraction'] = 0.75
    cfg['checkpoints_dir'] = CKPT_DIR
    cfg['checkpoint_name'] = seed
    cfg['prior_model_name'] = PRIOR_MODEL
    assert cfg['load_weights_only'] is False and cfg['continue_from_checkpoint'] is False
    assert cfg['prior_path'].startswith('/scratch/mk8347/'), cfg['prior_path']
    cfg['molecules_path'] = LOCAL_MOLECULES
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH
    assert cfg['grow_batch_size'] is False and cfg['batch_sizer_retest_steps'] == 0
    cfg['epochs'] = resume_step + steps
    cfg['eval_period'] = 250
    cfg['eval_num_samples'] = 2500
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['freeze_backward_policy'] = 'full'
    # owner 2026-09-18: no prior-MODEL samples in the backward branch; churn from noised anchors only
    cfg['buffers']['prior_buffer']['source'] = 'anchors'
    assert int(cfg['integrator']['T']) == 100
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == 1.0
    return cfg


def unpin(cfg):
    eq = dm._eq(cfg)
    bal = eq['balance']
    assert bal['kind'] == 'gated_ramp' and bal['ramp'] == 'replay' and bal['guard'] == 'bwd', bal
    assert bal['pinned'] == {'fwd': 0.0}, bal['pinned']
    bal['bounds'] = {k: list(v) for k, v in BOUNDS.items()}
    eq['fracs'] = dict(ENTRY_FRACS)
    # the parser refuses a floor under the stage's deactivate_threshold or under min_fracs
    assert float(eq['deactivate_threshold']) <= BOUNDS['bwd'][0], eq['deactivate_threshold']
    assert eq.get('min_fracs') is None, eq.get('min_fracs')
    return cfg


def no_forces(cfg):
    fc = cfg['fwd_loss_coeffs']
    assert fc['reward_grads'] == 0.0 and fc['traj_grads'] == 0.0 and fc.get('path_grad_last_k', 0) == 0, fc
    rc = cfg['replay_loss_coeffs']
    assert rc.get('stored_force_k', 0) == 0 and rc.get('reward_grads', 0.0) == 0.0, rc
    eq = dm._eq(cfg)
    # the forward branch is inert in this shape: rollouts are admission events, not training
    assert eq['loss_coeffs']['fwd']['freeze_policy'] == 1.0 and ENTRY_FRACS['fwd'] == 0.0
    return cfg


def pin(cfg, replay):
    # LEG 2 (owner 2026-09-19 18:30, after n5's first 500 steps): the archive's held state (replay 0.1 /
    # bwd 0.9, ratchet tripped on the stale 1.20 reference) is a DEADLOCK under rare rollouts -- log Z
    # flat at 28.6, bwd/under_coverage RISING 2.13 -> 2.28 while every pinned-0.3 cluster arm from the
    # same archive fell 1.8 -> 1.46 and climbed 2 nats. The level only falls when Z climbs and Z only
    # climbs with replay off the floor, so the ramp can never earn its release. Pinned points, as flk_sep14.
    eq = dm._eq(cfg)
    bwd = round(1.0 - replay, 3)
    eq['fracs'] = {'fwd': 0.0, 'bwd': bwd, 'replay': replay}
    eq['balance']['bounds'] = {'bwd': [bwd, bwd], 'replay': [replay, replay]}
    return cfg


def fresh_entry(cfg, replay):
    # The cluster battery enters equilibration through a REAL transition: fresh stage_ctrl (no stale
    # reference; 500-step cooldown with the fracs frozen at entry, then the running minimum is seeded),
    # entry fracs applied. Reproduced here with the dose_sep16f copied-stage trick: the resumed stage gets
    # an always-true exit (fires at the first eval, ~eval_period steps in) and a copy of itself with the
    # wanted entry fracs and the 95:5 bounds is appended; no on_enter (buffers stay as restored).
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = dm._eq(cfg)
    eq['exit'] = [{'metric': 'fwd/log_Z_learned', 'above': -1.0e9, 'patience': 1}]
    new = copy.deepcopy(eq)
    new['name'] = 'equilibration_ramp'
    new.pop('exit')
    new.pop('on_enter', None)
    new.pop('on_exit', None)
    bwd = round(1.0 - replay, 3)
    new['fracs'] = {'fwd': 0.0, 'bwd': bwd, 'replay': replay}
    new['balance']['bounds'] = {k: list(v) for k, v in BOUNDS.items()}
    stages.append(new)
    return cfg


def arm(name, n, rate, steps=STEPS, pin_replay=None, entry_replay=None):
    cfg = common(base(), name, steps)
    dm._every(n)(cfg)
    dm._rate(rate)(cfg)
    dm._tau(TAU)(cfg)
    dm._freeze_pb(cfg)
    dm._holdout(cfg)
    unpin(cfg)
    if pin_replay is not None:
        pin(cfg, pin_replay)
    if entry_replay is not None:
        fresh_entry(cfg, entry_replay)
    no_forces(cfg)
    eq = dm._eq(cfg)
    rb = cfg['buffers']['replay_buffer']
    assert eq['fwd_rollout_every'] == n and rb['mean_residence_steps'] == TAU and rb['val_frac'] == 0.05
    assert abs(cfg['lr_control']['seed_lr'] - BASE_LR * rate) < 1e-12
    assert cfg['freeze_backward_policy'] == 'full' and cfg['batch_size'] == cfg['max_batch_size'] == BATCH
    h = problem_hash_of(cfg)
    assert h == SEED_HASH, f'{name}: problem hash {h} != the archive\'s {SEED_HASH}'
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} N={n:2d} rate={rate} seed_lr={cfg["lr_control"]["seed_lr"]:.3e} tau={TAU} '
          f'bounds={eq["balance"]["bounds"]} steps={steps} batch={BATCH} T=100 hash ok')
    return out


def hold_arm(name, n_rare, rate, steps):
    """The converged N=1 frozen arm switched into rare rollouts: resumed stage (every 1, pinned 0.3, tau 120)
    exits at its first eval; the copy 'equilibration_rare' runs every n_rare at tau 600 with the same pinned
    split, and the transition re-stamps the rate at fixed_scale (= burn_in_scale) = rate."""
    cfg = common(base(HOLD_BASE), name, steps, seed=HOLD_SEED, resume_step=HOLD_RESUME_STEP)
    dm._tau(TAU)(cfg)
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = dm._eq(cfg)
    assert eq['fwd_rollout_every'] == 1 and eq['balance']['bounds'] == {'bwd': [0.7, 0.7], 'replay': [0.3, 0.3]}, name
    eq['exit'] = [{'metric': 'fwd/log_Z_learned', 'above': -1.0e9, 'patience': 1}]
    new = copy.deepcopy(eq)
    new['name'] = 'equilibration_rare'
    new.pop('exit')
    new.pop('on_enter', None)
    new.pop('on_exit', None)
    new['fwd_rollout_every'] = int(n_rare)
    stages.append(new)
    lc = cfg['lr_control']
    lc['fixed_scale'] = float(rate)
    lc['burn_in_scale'] = float(rate)
    assert cfg['freeze_backward_policy'] == 'full', name
    no_forces(cfg)
    h = problem_hash_of(cfg)
    assert h == SEED_HASH, f'{name}: problem hash {h} != the archive\'s {SEED_HASH}'
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'wrote {out.name:14s} HOLD: N=1 -> N={n_rare} at the first eval, rate {rate} on the transition, tau={TAU}, '
          f'pinned 0.3, seed step {HOLD_RESUME_STEP}, steps={steps} batch={BATCH} hash ok')
    return out


if __name__ == '__main__':
    hold_arm('hold10', 10, 0.5, steps=1250)
    arm('n5', 5, 1.0)
    arm('n10', 10, 0.5)
    arm('smoke_n5', 5, 1.0, steps=30)
    # leg 2: the share question (see pin / fresh_entry). n5 shape throughout.
    arm('pin30', 5, 1.0, steps=1000, pin_replay=0.3)
    arm('pin70', 5, 1.0, steps=1000, pin_replay=0.7)
    arm('ramp50', 5, 1.0, steps=1250, entry_replay=0.5)
