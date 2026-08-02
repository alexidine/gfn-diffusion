"""
aug02 battery -- 16 arms, generated from configs/tw_july31/1.yaml.

WHY THIS BATTERY. tw_july31 was designed as a T x W x LR ceiling grid and
returned almost none of that, for two reasons that this battery fixes rather
than repeats.

  (1) Half the grid died in phase 1 and then burned the clock. 8 of 16 arms
      converged phase 1 normally (wass_debiased -> 0.012-0.024), detonated,
      and then sat emitting bitwise-identical numbers for up to 56k steps --
      ~65 GPU-hours of frozen compute. _terminal_policy_state read only the
      'fwd' tracker slot, which phase 1 never populates, so nothing was
      watching. Fixed in train.py (_frozen_training_state + a branch-scanning
      _terminal_policy_state); that fix is a PREREQUISITE for this battery,
      not part of it.

  (2) The LR ladder collapsed to one rung. The controller's cut tier fired on
      nearly every arm around step 1000-1700, so a grid of set LRs
      4e-4..6.4e-3 ran at LIVE LRs of only 2e-4, 4e-4 or 8e-4, with three
      arms landing BELOW the resolver's own anchor. The set LR was never the
      independent variable.

So: every LR here is a LIVE LR, held live by cut_ratio = 1.0 (below), and
every arm is labelled by it.

WHAT tw_july31 DID RETURN, and what this battery does with it. Its two
controlled clip comparisons both favoured the TIGHT clip, which is the
opposite of its own premise:

  T60_anchor vs postfix_lr8x -- identical T/W/LR/seed/batch, differing ONLY in
    gradient_norm_clip (3864 vs 454) and the two grad tripwires. The loose arm
    was worse at every matched wall point (8h: 11.60 vs 10.10; 20h: 6.74 vs
    6.30; 31h: 6.21 vs 4.83).
  T10_clip_resolver (clip 37.9) vs T10_lr16 (clip 1011), same LR -- the tight
    arm reached phase 2 at step 6,640 against 12,640, and tracked ahead of the
    T=60 baseline at matched wall for its whole 13.2h life.

Both are single-seed and neither was a designed comparison, so clip is
promoted here to a first-class 2x3 factor (arms 0-5) rather than left as a
byproduct.

CUT_RATIO = 1.0 -- the load-bearing change. adaptive_lr.enabled: false does
NOT stop the cut tier (controller.py: "FIRE runs regardless of the flag"), so
the only way to hold a live LR is to make the cut itself a no-op. Fires are
still counted and logged (lr_ctrl/fires_*), so "this arm was hot" survives as
a diagnostic even though the LR no longer moves. The reset tier still rewinds
on a true explosion, and _frozen_training_state is now the real containment:
a detonating arm costs ~2000 steps instead of 8-20 hours, which is what makes
the deliberately-over-ceiling rung (arm 9) affordable to run at all.

TRIPWIRE BARS ARE HELD CONSTANT PER T, derived from the LOOSE clip at that T,
so that the clip A/B varies the clip alone. Bars derived from a tight clip
would sit at ~1x the measured healthy grad median and fire continuously,
measuring the tripwire instead of the clipping regime (tw_july31's arm 8 made
the same choice for the same reason). With cut_ratio = 1.0 the cut bar is
inert anyway; the reset bar still matters.

CLIP LEVELS are anchored to runs that actually happened, not to a formula:
  tight  T=10 = 37.9   (44gt5whr / tw_july31 arm 8)
         T=60 = 454.4  (postfix_july30, incl. the lr8x baseline)
         T=25 = 135.0  (log-log interpolation between them -- NOT measured)
  loose  3x the measured post-fix grad median x sqrt(W/512), i.e. the
         tw_july31 values: 1011 / 2006.9 / 3864.

MATCHED-WALL, NOT MATCHED-STEP. T=10 takes ~6x more steps per hour than T=60,
so epochs is left high and wall clock is the budget. Every cross-T comparison
must be read at equal wall time; tw_july31's T=10 arms were killed at 2-3h
against a 48h baseline and their raw tb_err looked terrible for that reason
alone.

TIER 2 (arms 13-15) uses the in-rollout variance schedule, which already
exists and is currently off (model.t_scale_ratio: null). t_scale_ratio is
sigma^2(1)/sigma^2(0) in (0, 1], so ratio < 1 narrows the noise rate toward
the end of the rollout -- a finer terminal resolution -- and
t_scale_preserve_budget: true holds total accumulated variance fixed so this
reshapes the schedule without also changing its scale. This is the actuator
for the one thing no LR/T/W setting in tw_july31 touched: across every healthy
arm the FIT ripple damps (tb_err amp ratio 0.63-0.94) while step_var and
terminal_var GROW (1.31-1.58). Score these arms on the terminal_var/step_var
amplitude ratio, not on tb_err.
"""
import math
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent
PARENT = OUTDIR.parent / 'tw_july31' / '1.yaml'
TAG = 'aug02'

LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

CLIP_TIGHT = {10: 37.9, 60: 454.4}          # measured, from real runs
CLIP_LOOSE = {10: 1011.0, 25: 2006.9, 60: 3864.0}   # 3x measured median (tw_july31)
CUT_OVER_CLIP = 10.0
RESET_OVER_CUT = 10.0

SEEDS = {'A': 12345, 'B': 20260802, 'C': 20260803}
BASE_EPOCHS = 100000
FROZEN_PATIENCE = 2000


def clip_tight(T):
    if T in CLIP_TIGHT:
        return CLIP_TIGHT[T]
    lo, hi = 10, 60
    f = (math.log(T) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return round(math.exp(math.log(CLIP_TIGHT[lo])
                          + f * (math.log(CLIP_TIGHT[hi]) - math.log(CLIP_TIGHT[lo]))), 1)


def load_parent():
    with PARENT.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_T(cfg, T):
    cfg['integrator']['T'] = T
    cfg['eval_T'] = T
    return cfg


def set_lrs(cfg, lr):
    for key in LR_KEYS:
        cfg[key] = lr
    return cfg


def set_bars(cfg, T, clip):
    """Bars from the LOOSE clip at this T (held constant across the clip A/B);
    gradient_norm_clip is the only thing the clip factor moves."""
    loose = CLIP_LOOSE[T]
    cfg['adaptive_lr']['cut_grad_abs'] = round(CUT_OVER_CLIP * loose, 1)
    cfg['adaptive_lr']['reset_grad_abs'] = round(RESET_OVER_CUT * CUT_OVER_CLIP * loose, 1)
    cfg['gradient_norm_clip'] = clip
    return cfg


def hold_lr_live(cfg):
    """cut_ratio 1.0 => on_explosion multiplies the cut factor by 1.0, so a
    fire is recorded but the LR does not move. recovery_target_frac stays
    <= cut_ratio, which keeps the re-ramp path disengaged."""
    cfg['adaptive_lr']['cut_ratio'] = 1.0
    cfg['adaptive_lr']['recovery_target_frac'] = 0.5
    cfg['adaptive_lr']['decay_halflife_steps'] = 0
    return cfg


def set_containment(cfg):
    cfg['terminal_frozen_steps'] = FROZEN_PATIENCE
    return cfg


def set_var_schedule(cfg, ratio, power=4.0):
    cfg['model']['t_scale_ratio'] = ratio
    cfg['model']['t_scale_power'] = power
    cfg['model']['t_scale_preserve_budget'] = True
    return cfg


def from_scratch(cfg):
    """find_shared_prior matches on problem_def, which excludes T and
    architecture, so a T=60 prior would match a T=10 arm; a loaded prior also
    SKIPS train_prior and leaves the policy random."""
    cfg['checkpoint_name'] = None
    cfg['prior_model_name'] = None
    cfg['reuse_prior'] = False
    cfg['continue_from_checkpoint'] = True
    return cfg


# (index, run_name, T, live_lr, clip_kind, seed_key, t_scale_ratio, tier, note)
ARMS = [
    # --- Tier 1: T ladder at matched LIVE LR, tight clip -----------------
    (0, 'T10_lr2_tight', 10, 2e-4, 'tight', 'A', None, 1,
     'T ladder rung; tight clip = 44gt5whr/arm8 value'),
    (1, 'T25_lr2_tight', 25, 2e-4, 'tight', 'A', None, 1,
     'REFERENCE CELL: T ladder rung, LR ladder rung, Tier-2 control, seed-set anchor'),
    (2, 'T60_lr2_tight', 60, 2e-4, 'tight', 'A', None, 1,
     'T ladder rung; replicates the postfix_lr8x baseline at its post-cut live LR'),
    # --- Tier 1: clip A/B, loose leg at each T ---------------------------
    (3, 'T10_lr2_loose', 10, 2e-4, 'loose', 'A', None, 1,
     'clip A/B vs arm 0'),
    (4, 'T25_lr2_loose', 25, 2e-4, 'loose', 'A', None, 1,
     'clip A/B vs arm 1'),
    (5, 'T60_lr2_loose', 60, 2e-4, 'loose', 'A', None, 1,
     'clip A/B vs arm 2; repeats the tw_july31 T60_anchor cell with the LR held live'),
    # --- Tier 1: live-LR ladder at T=25 ----------------------------------
    (6, 'T25_lr1_tight', 25, 1e-4, 'tight', 'A', None, 1, 'LR ladder rung'),
    (7, 'T25_lr4_tight', 25, 4e-4, 'tight', 'A', None, 1, 'LR ladder rung'),
    (8, 'T25_lr8_tight', 25, 8e-4, 'tight', 'A', None, 1, 'LR ladder rung'),
    (9, 'T25_lr16_tight', 25, 1.6e-3, 'tight', 'A', None, 1,
     'LR ladder top rung; above the observed phase-1 ceiling, cheap under the frozen abort'),
    # --- Tier 1: seed replicates on the reference cell --------------------
    (10, 'T25_lr2_tight_s2', 25, 2e-4, 'tight', 'B', None, 1, 'seed replicate of arm 1'),
    (11, 'T25_lr2_tight_s3', 25, 2e-4, 'tight', 'C', None, 1, 'seed replicate of arm 1'),
    # --- Tier 1: T=60 second LR rung --------------------------------------
    (12, 'T60_lr4_tight', 60, 4e-4, 'tight', 'A', None, 1,
     'T=60 second rung: is the optimum still 2e-4 once the cut cannot move the LR'),
    # --- Tier 2: in-rollout variance schedule (control = arm 1) -----------
    (13, 'T25_vr30', 25, 2e-4, 'tight', 'A', 0.30, 2, 't_scale_ratio 0.30'),
    (14, 'T25_vr10', 25, 2e-4, 'tight', 'A', 0.10, 2, 't_scale_ratio 0.10'),
    (15, 'T25_vr03', 25, 2e-4, 'tight', 'A', 0.03, 2, 't_scale_ratio 0.03'),
]


def build():
    log = []
    for (idx, name, T, lr, clip_kind, seed_key, ratio, tier, note) in ARMS:
        cfg = load_parent()
        clip = clip_tight(T) if clip_kind == 'tight' else CLIP_LOOSE[T]
        set_T(cfg, T)
        set_lrs(cfg, lr)
        set_bars(cfg, T, clip)
        hold_lr_live(cfg)
        set_containment(cfg)
        from_scratch(cfg)
        if ratio is not None:
            set_var_schedule(cfg, ratio)
        cfg['seed'] = SEEDS[seed_key]
        cfg['epochs'] = BASE_EPOCHS
        cfg['tag'] = TAG
        cfg['run_name'] = f'a2_{name}'
        with (OUTDIR / f'{idx}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
        log.append({
            'index': idx, 'run_name': f'a2_{name}', 'tier': tier, 'T': T,
            'eval_T': T, 'live_lr': lr, 'clip_kind': clip_kind,
            'gradient_norm_clip': clip,
            'cut_grad_abs': cfg['adaptive_lr']['cut_grad_abs'],
            'reset_grad_abs': cfg['adaptive_lr']['reset_grad_abs'],
            'cut_ratio': 1.0, 't_scale_ratio': ratio,
            'seed': SEEDS[seed_key], 'terminal_frozen_steps': FROZEN_PATIENCE,
            'epochs': BASE_EPOCHS, 'from_scratch': True, 'note': note,
        })
    with (OUTDIR / 'experiment_log.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
    return log


if __name__ == '__main__':
    rows = build()
    hdr = (f"{'idx':>3s} {'run_name':22s} {'tier':>4s} {'T':>3s} {'live_lr':>8s} "
           f"{'clip':>8s} {'kind':>6s} {'cut':>9s} {'reset':>10s} {'vr':>5s} {'seed':>9s}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f"{r['index']:3d} {r['run_name']:22s} {r['tier']:4d} {r['T']:3d} "
              f"{r['live_lr']:8.1e} {r['gradient_norm_clip']:8.1f} {r['clip_kind']:>6s} "
              f"{r['cut_grad_abs']:9.0f} {r['reset_grad_abs']:10.0f} "
              f"{str(r['t_scale_ratio']):>5s} {r['seed']:9d}")
