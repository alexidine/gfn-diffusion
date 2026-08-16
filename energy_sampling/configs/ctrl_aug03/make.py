"""ctrl_aug03 -- controller battery for the `naive` stage (local, RTX 5080).

Derived from mk_dev.yaml, which is USER-OWNED and only ever READ here. Every
arm gets its own run_name, so its checkpoint prefix ({tag}_{run_name}_{problem})
is distinct and nothing here can touch the dev_mk_dev_* set.

WHAT IS HELD FIXED ACROSS ALL FOUR ARMS
  - pure TB, scalar log Z, lr_flow 0.1 (mk_dev's base state; the SubTB /
    full_flow / lr_flow 1e-4 / z_calibration-off cluster from local_aug02 is
    reverted by construction, since that was config-level only -- no code
    change implemented SubTB).
  - fwd pinned at 0.05 and training BOTH Z and the policy (freeze_policy 0).
    In mk_dev's naive stage fwd is Z-ONLY, which makes fwd/tb_err a held-out
    metric. Here it is a training branch, which is what makes the
    replay-vs-fwd scatter ratio a clean generalization gap: both branches now
    carry the same loss with the same policy gradient, and differ only in
    their sampler.
  - resume from the mk_dev phase-1 exit (step 6680, stage train_prior with
    request_eval stamped), so every arm enters `naive` from bit-identical
    weights, optimizers and buffers. Phase 1 is never re-run and cannot
    contribute variance to the comparison.
  - z_calibration left ON at mk_dev's settings.
    CORRECTED 2026-08-03 after reading the runs: an earlier version of this note
    claimed the sensor "reads identically zero on an unconditional run, so it is
    expected never to fire". That is wrong. mk_dev sets `sensor: pooled`, which
    is |EMA fwd/tb_resid_clipped| -- perfectly well defined with one condition.
    Only the `rms`/`worst` sensors are per-condition dispersion and degenerate
    unconditionally. Measured: the sensor reads 6.67 nats at the train_prior ->
    naive transition (the bwd->fwd Z handoff, since phase 1 trains Z through
    mle/tbc and naive switches it to forward TB), z_cal spends 8-30 extra Z
    steps per train step for ~300 steps absorbing it, and then goes nearly
    dormant (p = 0 on 72-86% of ticks thereafter).

THE DESIGN
  arm 0  fx_static   fixed mix                 buffer servo OFF
  arm 1  fx_servo    fixed mix                 buffer servo ON
  arm 2  cs_servo    constraint integrator     buffer servo ON
  arm 3  pp_servo    proportional (incumbent)  buffer servo ON

  0 vs 1  isolates the buffer servo, on the simplest possible plant -- nothing
          else in the run is moving, so any difference is the servo's.
  1 vs 2 vs 3  isolates the frac controller at a matched buffer servo.
  0 is the absolute reference: every knob static.

  arms 2 and 3 are given the SAME two bars (bwd 2.0, replay 10.0) so the
  comparison is of the control LAW, not of two calibrations. The incumbent
  proportional arm has never been run at fwd 0.05 with fwd policy gradients,
  so its own historical numbers are not a matched control for this route --
  it has to be re-run here to be comparable.

BARS
  bwd/relative_under_wcen 2.0 is the user-stated absorption criterion ("<2, or
  <1 in the limit"): no mode of the prior buffer badly under-weighted.
  fwd/over_coverage 10.0 is an admitted guess -- its reachable floor is
  unknown, and the metric settled 17.0-17.8 on the fixed-mix dev run. Under
  the constraint law a guessed best-effort bar is cheap: if it is unreachable
  its drive never zeroes and the actuator walks until the constraint pushes
  back or it hits its bound, which is exactly "as good as we can get provided
  the constraint holds". Under the proportional law the same guess biases the
  equilibrium directly. That difference is the main thing arms 2 and 3 test.

BOUNDS
  replay is bounded [0.10, 0.25] absolute. The ceiling sits just under the
  0.29-0.40 band in which every ratcheted dev run rang; the floor keeps a real
  replay dose so that even total constraint domination degrades to a viable
  fixed mix rather than to a starved one.
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'ctrl_aug03'

# mk_dev's own phase-1 exit: step_ind 6680, stage train_prior, request_eval
# stamped True, gfn_config full_flow False / dplr_rank 6 / periodic_centroids
# True -- i.e. the pure-TB architecture these arms want, verified by reading
# the file rather than inferred.
PHASE1_EXIT = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92'
               '_phase1_exit.pt')
RESUME_STEP = 6680
STEP_BUDGET = 2400            # train steps per arm past the resume (see run_all.ps1 budget)

ENTRY = {'fwd': 0.05, 'bwd': 0.75, 'replay': 0.20}
BARS = {'bwd': 2.0, 'replay': 10.0}
REPLAY_BOUNDS = [0.10, 0.25]

# replay/scatter_err over fwd/scatter_err. Replay is a |resid|-prioritized
# resample of stored fwd rollouts, so it is the HARD tail of the forward
# distribution and its residual spread should EXCEED fresh forward's (observed
# healthy ~2x). Below 1.0 the policy fits reused stored trajectories better
# than the fresh draws they were selected from: memorization. The deadband to
# 1.5 keeps the servo off during ratio jitter and leaves the healthy ~2x point
# comfortably inside the released region.
BUFFER_SERVO = {
    'numerator': 'replay/scatter_err',
    'denominator': 'fwd/scatter_err',
    'bar': 1.0,
    'release': 1.5,
    'scale': 0.1,
    'gain': 0.02,
    'relax': 0.25,
    'max_boost': 12.0,
    'max_step': 0.03,
}


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def naive_stage(cfg):
    for st in cfg['protocol']['stages']:
        if st['name'] == 'naive':
            return st
    raise KeyError("mk_dev.yaml has no 'naive' stage")


def common(cfg):
    """Everything the four arms share."""
    cfg['checkpoint_name'] = PHASE1_EXIT
    cfg['load_weights_only'] = False        # full resume: stage, step, optimizers, buffers
    cfg['continue_from_checkpoint'] = False
    cfg['reuse_prior'] = False
    cfg['epochs'] = RESUME_STEP + STEP_BUDGET
    cfg['eval_period'] = 500
    cfg['figs_period'] = 2000
    cfg['archive_period'] = 0               # 910MB/archive, and a 5k-step arm has nothing to archive
    cfg['terminal_frozen_steps'] = 2000
    cfg['max_reloads'] = 4

    st = naive_stage(cfg)
    st['fracs'] = dict(ENTRY)
    st['min_fracs'] = {'fwd': 0.05, 'bwd': 0.05, 'replay': 0.02}
    st['deactivate_threshold'] = 0.01
    # fwd now trains Z AND the policy; bwd keeps freeze_z 1 from the base block
    st['loss_coeffs'] = {'fwd': {'tb': 1.0, 'freeze_policy': 0.0, 'freeze_z': 0.0},
                         'bwd': {'tb': 1.0}}
    st.pop('balance', None)
    st.pop('buffer_servo', None)
    return cfg


def write(cfg, name):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
    print(f'wrote {out.name:20s} run_name={name}')
    return out


LOG = []


def record(index, name, controller, servo, note):
    LOG.append({'index': index, 'run_name': name, 'tag': TAG,
                'controller': controller, 'buffer_servo': servo,
                'resume_from': PHASE1_EXIT, 'resume_step': RESUME_STEP,
                'epochs': RESUME_STEP + STEP_BUDGET,
                'fwd_frac': ENTRY['fwd'], 'fwd_trains_policy': True,
                'entry_fracs': dict(ENTRY), 'note': note})


# ------------------------------------------------------------------ arm 0
# Everything static. The reference every other arm is read against, and the
# control for arm 1's buffer servo.
cfg = common(base())
write(cfg, 'fx_static')
record(0, 'fx_static', 'fixed', False,
       'reference cell: fixed mix 0.05/0.75/0.20, no controller, no servo')

# ------------------------------------------------------------------ arm 1
# Buffer servo only. Isolates freshness from allocation: the fracs never move,
# so anything that changes is the buffer's doing. This is the cleanest test of
# whether replay overfitting is worth actuating on at all.
cfg = common(base())
naive_stage(cfg)['buffer_servo'] = copy.deepcopy(BUFFER_SERVO)
write(cfg, 'fx_servo')
record(1, 'fx_servo', 'fixed', True,
       'buffer freshness servo on a static mix; A/B against arm 0')

# ------------------------------------------------------------------ arm 2
# The proposed controller. One-sided integral (dual ascent) on the bwd
# constraint against the replay objective, in the logit of replay's share of
# the pair, with priority 3 giving the constraint a soft lexicographic win and
# max_step bounding slew.
#
# gain 0.02 / max_step 0.03 at a 10-step tick: a typical contest (drive ~0.8)
# moves theta ~0.016/tick, traversing the whole bounded range in ~1000 steps,
# while a constraint breach at priority 3 saturates max_step and restores in
# ~400. Slower than the proportional default (~200 steps, far quicker than the
# plant) in the direction that starves bwd; faster in the direction that
# restores it -- matching the failure asymmetry, since starving bwd collapses
# coverage fast but recovers fast, whereas starving replay degrades the policy
# slowly and recovers slowly.
cfg = common(base())
st = naive_stage(cfg)
st['balance'] = {
    'kind': 'constraint',
    'pinned': {'fwd': ENTRY['fwd']},
    'metrics': {'bwd': 'bwd/relative_under_wcen', 'replay': 'fwd/over_coverage'},
    'constrain': 'bwd',
    'bars': dict(BARS),
    'bounds': {'replay': list(REPLAY_BOUNDS)},
    'gain': 0.02,
    'priority': 3.0,
    'max_step': 0.03,
}
st['buffer_servo'] = copy.deepcopy(BUFFER_SERVO)
write(cfg, 'cs_servo')
record(2, 'cs_servo', 'constraint', True,
       'constraint integrator, constrain bwd at 2.0, best-effort replay at 10.0, '
       'replay bounded [0.10, 0.25]')

# ------------------------------------------------------------------ arm 3
# The incumbent, at the same bars and the same ceiling, so the only difference
# from arm 2 is the control law: a static tilt of the idle mix by relative
# unmet need, versus an integrator. default_boost reproduces the entry split as
# shares of the pair (0.75/0.95, 0.20/0.95), so both arms idle at the same mix
# and start from the same point. alpha 0.005 is mk_dev's own value, the top of
# the range its docstring recommends.
cfg = common(base())
st = naive_stage(cfg)
st['balance'] = {
    'kind': 'proportional',
    'pinned': {'fwd': ENTRY['fwd']},
    'metrics': {'bwd': 'bwd/relative_under_wcen', 'replay': 'fwd/over_coverage'},
    'drive': 'relative',
    'targets': dict(BARS),
    'max_fracs': {'replay': REPLAY_BOUNDS[1]},
    'default_boost': {'bwd': ENTRY['bwd'] / 0.95, 'replay': ENTRY['replay'] / 0.95},
    'alpha': 0.005,
    'floor': 0.03,
}
st['buffer_servo'] = copy.deepcopy(BUFFER_SERVO)
write(cfg, 'pp_servo')
record(3, 'pp_servo', 'proportional', True,
       'incumbent proportional tilt at the same bars/ceiling as arm 2; '
       'controller-law A/B')

with (HERE / 'experiment_log.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(LOG, f, default_flow_style=False, sort_keys=False)
print(f'wrote experiment_log.yaml ({len(LOG)} arms)')
