"""
press_july29 -- convergence-acceleration press, built off uncond_july28/4.yaml
(prop_fwd20, the battery's best arm) as the single baseline.

TIER A (0-14): short screens. epochs 16000 = ~3030 phase-2 steps (~3 h) on top
of the shared T=60 phase1_exit warm start at step 12970. Every Tier-A arm sets
adaptive_lr.decay_halflife_steps 0 so the LR is CONSTANT and a detonation read
is unambiguous. The matched control is uncond_july28 arm 17 (prop_fwd20_nodecay)
read at step 16000 -- same seed, same baseline, halflife 0 already.

  0-3   LR ladder x2 / x4 / x8 / x16 off lr 5e-5 (all four lr_* moved together)
  4-5   x4 / x8 with ema_decay 0.95
  6-7   x4 with gradient_norm_clip x3 / x0.33 (cut_grad_abs and reset_grad_abs
        ride the explicit clip at 30x / 300x via resolve_derived_config)
  8     x4 with reward_range 500 -> 100
  9     ema_decay 0.95 at baseline LR
  10-12 in-rollout variance schedule: t_scale_ratio/power. Checkpoint-safe --
        GFN registers _var_accum with persistent=False
  13-14 replay residence 100 / 50, churn raised in lockstep to hold occupancy:
        residence = min(max_size/churn_rate, max_residence_steps)

TIER C (15-18): architecture/rollout changes that are NOT checkpoint-compatible
(width, dplr rank, periodic_centroids, T all change the weight layout or the
problem identity, and Checkpointer.assert_problem_match hard-fails). These run
the full protocol from scratch: checkpoint_name null, reuse_prior false so
stage 'train_prior' is not skipped by skip_if: prior_loaded. epochs 40000.

HOLD TIER C until the Tier-A LR ladder reports -- they are day-plus runs and the
ladder is expected to move the LR they should be run at.
"""

from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = Path(__file__).parent.parent / 'uncond_july28' / '4.yaml'
OUTDIR = Path(__file__).parent
TAG = 'press_july29'

TIER_A_EPOCHS = 16000
TIER_C_EPOCHS = 40000

BASE_LR = 5.0e-05          # arm 4's hand LR at T=60 (2.4x the resolver anchor)
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

# resolve_derived_config at (W=512, T=60); recomputed here so the overrides are
# exact multiples of what the resolver would otherwise have produced
CLIP_T60_W512 = 454.80


def load_base():
    with BASE_PATH.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_lrs(cfg, lr):
    for k in LR_KEYS:
        cfg[k] = lr


def tier_a(cfg):
    cfg['epochs'] = TIER_A_EPOCHS
    cfg['adaptive_lr']['decay_halflife_steps'] = 0
    return cfg


def tier_c(cfg):
    """Full protocol from scratch: no warm start, phase 1 must actually run."""
    cfg['epochs'] = TIER_C_EPOCHS
    cfg['adaptive_lr']['decay_halflife_steps'] = 0
    cfg['checkpoint_name'] = None
    cfg['continue_from_checkpoint'] = True
    cfg['reuse_prior'] = False
    cfg['prior_model_name'] = None
    return cfg


def set_T(cfg, T, lr):
    cfg['integrator']['T'] = T
    cfg['integrator']['min_traj_length'] = T
    cfg['integrator']['max_traj_length'] = T
    cfg['eval_T'] = T
    set_lrs(cfg, lr)


def set_width(cfg, w):
    for key in ('s_emb_dim', 's_hidden_dim', 't_hidden_dim', 'policy_hidden_dim',
                'flow_hidden_dim', 'cond_hidden_dim'):
        cfg['model'][key] = w


ARMS = []


def arm(index, name, note, tier, build):
    ARMS.append(dict(index=index, run_name=name, note=note, tier=tier, build=build))


# ---------------------------------------------------------------- TIER A -----
for i, (mult, nm) in enumerate([(2, 'lr2x'), (4, 'lr4x'), (8, 'lr8x'), (16, 'lr16x')]):
    arm(i, f'A_{nm}', f'LR ladder: all lr_* x{mult} = {BASE_LR * mult:.1e}, constant (nodecay)',
        'A', (lambda m: (lambda c: set_lrs(c, BASE_LR * m)))(mult))

arm(4, 'A_lr4x_ema', 'LR x4 + ema_decay 0.95 (target-network A/B vs arm 1)', 'A',
    lambda c: (set_lrs(c, BASE_LR * 4), c.update(ema_decay=0.95)))
arm(5, 'A_lr8x_ema', 'LR x8 + ema_decay 0.95 (target-network A/B vs arm 2)', 'A',
    lambda c: (set_lrs(c, BASE_LR * 8), c.update(ema_decay=0.95)))

arm(6, 'A_lr4x_clip3x', f'LR x4 + gradient_norm_clip {CLIP_T60_W512:.1f} -> {CLIP_T60_W512 * 3:.1f}; '
                        f'cut_grad_abs/reset_grad_abs follow at 30x/300x', 'A',
    lambda c: (set_lrs(c, BASE_LR * 4), c.update(gradient_norm_clip=round(CLIP_T60_W512 * 3, 1))))
arm(7, 'A_lr4x_clip033x', f'LR x4 + gradient_norm_clip {CLIP_T60_W512:.1f} -> 150.0', 'A',
    lambda c: (set_lrs(c, BASE_LR * 4), c.update(gradient_norm_clip=150.0)))

arm(8, 'A_lr4x_rr100', 'LR x4 + reward_range 500 -> 100. CHANGES THE TARGET: log Z is not '
                       'comparable across this arm; score on fwd/r2, EffDim, fwd/logw_std', 'A',
    lambda c: (set_lrs(c, BASE_LR * 4), c['energy_config'].update(reward_range=100)))

arm(9, 'A_ema', 'ema_decay 0.95 at baseline LR (separates EMA-helps from EMA-enables)', 'A',
    lambda c: c.update(ema_decay=0.95))

arm(10, 'A_var_r10_p4', 't_scale_ratio 0.1 power 4: late sharp variance taper, budget preserved', 'A',
    lambda c: c['model'].update(t_scale_ratio=0.1, t_scale_power=4.0))
arm(11, 'A_var_r01_p4', 't_scale_ratio 0.01 power 4: aggressive taper', 'A',
    lambda c: c['model'].update(t_scale_ratio=0.01, t_scale_power=4.0))
arm(12, 'A_var_r10_p1', 't_scale_ratio 0.1 power 1: taper spread through the rollout', 'A',
    lambda c: c['model'].update(t_scale_ratio=0.1, t_scale_power=1.0))

arm(13, 'A_replay_r100', 'replay residence 200 -> 100: churn_rate 50 -> 100, '
                         'max_residence_steps 250 -> 125 (occupancy held at max_size)', 'A',
    lambda c: c['buffers']['replay_buffer'].update(churn_rate=100, max_residence_steps=125))
arm(14, 'A_replay_r50', 'replay residence 200 -> 50: churn_rate 50 -> 200, '
                        'max_residence_steps 250 -> 60 (occupancy held at max_size)', 'A',
    lambda c: c['buffers']['replay_buffer'].update(churn_rate=200, max_residence_steps=60))

# ---------------------------------------------------------------- TIER C -----
arm(15, 'C_w1024', 'width 512 -> 1024 (six *_hidden_dim + s_emb_dim); gradient_norm_clip stays '
                   'auto so the resolver scales it by sqrt(2)', 'C',
    lambda c: set_width(c, 1024))
arm(16, 'C_dplr12', 'dplr_rank 6 -> 12', 'C',
    lambda c: c['model'].update(dplr_rank=12))
arm(17, 'C_noperiodic', 'periodic_centroids true -> false', 'C',
    lambda c: c['model'].update(periodic_centroids=False))
arm(18, 'C_T100', 'T 60 -> 100 with its OWN phase 1. lr_* = 3.0e-5 = resolver anchor at T=100 '
                  '(1.25e-5) x 2.4, the same multiplier arm 4 uses at T=60', 'C',
    lambda c: set_T(c, 100, 3.0e-05))


def build_all():
    base = load_base()
    log = []
    for a in ARMS:
        cfg = deepcopy(base)
        cfg = tier_a(cfg) if a['tier'] == 'A' else tier_c(cfg)
        a['build'](cfg)
        cfg['tag'] = TAG
        cfg['run_name'] = a['run_name']

        with (OUTDIR / f"{a['index']}.yaml").open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

        log.append({'index': a['index'], 'run_name': a['run_name'], 'tier': a['tier'],
                    'note': a['note'], 'epochs': cfg['epochs'],
                    'lr_fused': cfg['lr_fused'], 'T': cfg['integrator']['T'],
                    'from_scratch': cfg['checkpoint_name'] is None})

    with (OUTDIR / 'experiment_log.yaml').open('w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
    return log


if __name__ == '__main__':
    for entry in build_all():
        print(f"{entry['index']:>2}  {entry['run_name']:<18} tier {entry['tier']}  "
              f"epochs {entry['epochs']:<6} lr {entry['lr_fused']:.2e}  T{entry['T']}"
              f"{'  [from scratch]' if entry['from_scratch'] else ''}")
