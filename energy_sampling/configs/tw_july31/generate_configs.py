"""
tw_july31 -- rollout-length x width x LR grid for the "fastest and tightest
unconditional mipcas ELJ" goal.

WHY THIS BATTERY. postfix_july30 established that (a) 4e-4 is safe at T=60,
4-8x the pre-fix TB edge, and (b) fwd/tb_err descends as a POWER LAW in
phase-2 steps with exponent k ~ -0.25, and k is the same at every rung of the
LR ladder when fitted over a matched error window (nodecay -0.233, lr2x
-0.225, lr4x -0.264, lr8x -0.253). LR moves the prefactor, not the exponent.
At k = -0.25, going from lr8x's current 7 nats to 2 nats costs ~150x more
steps -- 31-45 days at the T=60 step rate. Two days buys 5.4x. So the goal
needs ~20-30x that LR alone cannot supply, from either:
  - THROUGHPUT (prefactor): T is the only lever with the right structure --
    ~6x cheaper per step at T=10, plus up to ~6x more LR if the resolver's
    lr ~ 1/T shape survives the fix. 36x is the whole budget.
  - SHAPE (the exponent k): only width and batch can plausibly bend it. If
    the descent is capacity-limited, W moves k; if optimisation-limited, it
    does not, and we learn that for 3 arms.

WHAT THE RESOLVER WOULD HAVE DONE, AND WHY EVERY BAR HERE IS EXPLICIT.
`grad_norm_pre_clip` measured on the postfix_july30 arms (phase 2, >=1k steps
past entry) against the resolver's own gradient_norm_clip:

    T=60 W=512  clip 454.4    ref    median 1288   100.0% of steps CLIPPED
                              lr2x   median 1056   100.0%
                              lr4x   median  475    53.6%
                              lr8x   median  239     7.0%
    T=10 W=512  clip  37.9    44gt5whr median 337  100.0%

utils._GRAD_MEDIAN is {10: 1.0e3, 25: 6.6e3, 100: 1.7e4}, measured PRE-FIX;
log-log interpolation gives 1.19e4 at T=60 against a measured 1289, so the
table is ~9x high at T=60 and ~3x high at T=10. Left on `auto` this grid
would hand every low-LR cell a clip ~3-9x below its own working gradient
norm -- 100% saturation -- and scale that error by sqrt(W/512) across the
width axis, confounding W with a clip regime change. So:

    gradient_norm_clip   = 3 x the MEASURED median at that T, x sqrt(W/512)
    adaptive_lr.cut_grad_abs   = 10 x that clip      (~13-16x the healthy p99)
    adaptive_lr.reset_grad_abs = 10 x cut_grad_abs

The medians above were read at the 1x LR rung, and grad norm FALLS with LR
(1288 -> 239 across the ladder), so a bar set from the 1x median is
deliberately loose for the high-LR cells: an outlier guard, not a normaliser.
Healthy p99/median ran 1.9-2.7 in every surviving arm; r70 (dead) ran 108.

RECOVERY STAYS INERT ON PURPOSE. controller.py only re-ramps when
recovery_target_frac > cut_ratio, and both are 0.5 here (inherited). For a
CEILING measurement a fire must be terminal and unambiguous; the new 0.85
recovery in mk_dev would let marginal arms re-ramp and smear the ceiling.
Enable it for production runs, not for this one.

BATCH STAYS ADAPTIVE (parent's batch_size 1000 / grow true / max 50000).
An earlier draft pinned it to remove the postfix arms' drift (lr8x sat at
3438, nodecay at 5064). That was wrong twice over:
  - 2831 is the throughput knee on the ~17 GB LOCAL card. On the ~86 GB
    cluster card a pinned 2831 at T=10 holds ~15% of VRAM and, more to the
    point, drops GPU utilisation and power draw -- at T=10 each SDE step is a
    small kernel, so phase 2 goes launch-bound and more samples per kernel is
    nearly free. Adaptive batching is the mechanism built for exactly that.
  - Above the critical batch size the gradient-noise reduction saturates, so
    two arms at 3438 and 5064 make the same per-step progress; the drift
    biases WALL-CLOCK, not the step-matched readouts this grid is built on
    (the ceiling, and k fitted against phase-2 steps).
That premise is assumed, not measured, and if it is false every step-matched
comparison in BOTH batteries is biased -- so arm 6 pins a deliberately small
batch against arm 1's adaptive one. Matching per-step progress means we are
past the knee and the drift never mattered.

FROM SCRATCH, AND reuse_prior MUST STAY FALSE. checkpointing.find_shared_prior
matches ANY run's *_prior.pt in checkpoints_dir on problem_def alone, and its
docstring is explicit that problem_def "carries only the target identity ...
never architecture/T/lr". A T=60 prior therefore matches a T=10 arm, and a
loaded prior SKIPS train_prior and leaves the policy random. On a grid that
varies T and W this is a silent-catastrophe path, not a hygiene preference.

PARENT is postfix_july30/1.yaml (postfix_nodecay): constant LR
(decay_halflife_steps 0), the battery's own mixture and controller settings
kept byte-identical so arm 9 anchors this grid to that one.

ARMS (16 = one cluster submission)

  T=10 LR LADDER (0-2, +12) -- W512. The ceiling at T=10,
  measured rather than inferred. The lr ~ 1/T rule predicts 2.4e-3 from
  T=60's measured 4e-4; 0-2 bracket it and 12 is insurance in case 3.2e-3
  rides.
   0 : T10_lr08     8.0e-4    (= the 1/T rule's T=60 value, carried across)
   1 : T10_lr16     1.6e-3    MID CELL -- the reference for every other block
   2 : T10_lr32     3.2e-3    (just above the 1/T prediction)
  12 : T10_lr64     6.4e-3

  WIDTH (3-5, +15) -- T=10. Does capacity bend k? W1024 gets TWO
  LRs so "worse" cannot be confused with "detonated"; W256 gets the mid and
  the high rung because a narrower net should tolerate MORE LR (MLE tolerance
  shrinks with width) and "smaller and much faster" is a live candidate for
  a speed goal, not just "bigger and better".
   3 : T10_w256_lr16    W 256, 1.6e-3
  15 : T10_w256_lr32    W 256, 3.2e-3
   4 : T10_w1024_lr08   W 1024, 8.0e-4
   5 : T10_w1024_lr16   W 1024, 1.6e-3

  CRITICAL-BATCH CHECK (6) -- T=10, W512, 1.6e-3, batch PINNED at 1000 with
  growth off; the only arm in the grid that does not batch adaptively. Its
  treatment leg is arm 1. Matching per-step progress means we are past the
  knee where extra samples stop reducing gradient noise -- which is the
  premise that makes the postfix arms' batch drift (3438 vs 5064) harmless
  for every step-matched comparison in both batteries.
   6 : T10_b_fixed1000

  CLIP A/B (8) -- T=10, W512, 1.6e-3, clip left at the resolver's
  37.9 (100% saturation). Its treatment leg is arm 1, which is identical
  except for the raised clip. If clipping is inert under Adam (2z5oo55f) the
  pair reads flat and we have bought certainty for one slot; if the low-LR
  arms were clip-trapped, part of the LR ladder's gain is available without
  spending stability margin.
   8 : T10_clip_resolver

  ANCHOR + T=60 CEILING RECHECK (9, 14) -- W512.
   9 : T60_anchor   4.0e-4  replicates lr8x's T/W/LR/batching exactly; the
                    only difference is this grid's explicit clip and tripwire
                    bars, so it is both the cross-battery anchor and the
                    correction term for the bar change.
  14 : T60_lr08     8.0e-4  lr16x fired at 8e-4 against cut_grad_abs = 30 x
                    clip = 13632, on a WOBBLE (tb_err 10.2 -> 11.4, max
                    grad 9.5e5 in one spike). utils._CUT_GRAD_OVER_CLIP is
                    now 100, and this grid's bar is 10 x an explicit clip.
                    So "8e-4 breaks at T=60" is currently bar-dependent, not
                    physics. This arm settles it.

  T=25 (10-11, +7) -- W512. Three rungs, bracketing the T=25 ceiling as
  densely as T=10's, which is what a ceiling-vs-T exponent fit needs; T=60
  already has four rungs from postfix_july30. The 1/T rule predicts a 9.6e-4
  ceiling here.
  10 : T25_lr08     8.0e-4
  11 : T25_lr16     1.6e-3
   7 : T25_lr32     3.2e-3

  REPLICATE (13) -- arm 1 at a different seed. Every conclusion in the
  postfix analysis rests on single arms; one replicate at the mid cell
  calibrates how much of a between-arm gap is noise.
  13 : T10_lr16_s2  seed 20260731

READING GUIDE. Primary outcome is NOT wall-clock-to-threshold. It is
  (1) the highest surviving LR rung at each T -> does the ceiling scale as
      1/T, 1/sqrt(T), or not at all;
  (2) the power-law exponent k of fwd/tb_err vs phase-2 steps, fitted over a
      MATCHED error window (the postfix fits needed ~one factor-3.5 of error,
      ~9k phase-2 steps, to be stable) -> does anything move k;
  (3) clip saturation (grad_norm_pre_clip vs gradient_norm_clip) logged per
      arm, so no result is read without knowing which regime produced it.
epochs stays at the parent's 100000 -- deliberately not capped. Read the grid
at MATCHED PHASE-2 STEPS, not at whatever step each arm happened to reach:
that comparison can always be truncated in analysis, but an arm stopped early
cannot be extended, and a T=10 arm capped at 40k would idle out in ~8 h on
exactly the cluster time this battery exists to spend. Fitting k needs ~one
factor-3.5 of error (~9k phase-2 steps in postfix_july30); everything past
that is free signal on the tail. Wall-clock is recoverable from _runtime.
"""

import math
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent
PARENT = OUTDIR.parent / 'postfix_july30' / '1.yaml'
TAG = 'tw_july31'

LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')
WIDTH_KEYS = ('policy_hidden_dim', 's_emb_dim', 's_hidden_dim',
              't_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')

# MEASURED post-fix pre-clip grad-norm medians (phase 2, 1x-LR rung).
# T=60: postfix_july30 ref/nodecay.  T=10: 44gt5whr.
# T=25 is log-log interpolated between them (exponent 0.747), not measured --
# flagged in the experiment log so it is never quoted as data.
GRAD_MEDIAN_MEASURED = {10: 337.0, 60: 1288.0}
CLIP_OVER_MEDIAN = 3.0
CUT_OVER_CLIP = 10.0
RESET_OVER_CUT = 10.0

PINNED_BATCH = 1000      # arm 6 only: the critical-batch check
BASE_EPOCHS = 100000     # parent's value, kept: see the epochs note in the docstring
W_REF = 512


def grad_median(T):
    if T in GRAD_MEDIAN_MEASURED:
        return GRAD_MEDIAN_MEASURED[T]
    lo, hi = 10, 60
    f = (math.log(T) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(GRAD_MEDIAN_MEASURED[lo])
                    + f * (math.log(GRAD_MEDIAN_MEASURED[hi])
                           - math.log(GRAD_MEDIAN_MEASURED[lo])))


def clip_for(T, W):
    """3x the measured median at this T, x sqrt(W/512) -- the resolver's own
    width scaling, applied to a measured rather than a pre-fix-tabulated
    median."""
    return round(CLIP_OVER_MEDIAN * grad_median(T) * math.sqrt(W / W_REF), 1)


def load_parent():
    with PARENT.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_T(cfg, T):
    """Rollout length. eval_T tracks T -- the stab_july21 elj battery's
    eval_T = 2T scoring artifact floored wass_debiased on an integration-dt
    mismatch, so these must never diverge."""
    cfg['integrator']['T'] = T
    cfg['eval_T'] = T
    return cfg


def set_width(cfg, W):
    for key in WIDTH_KEYS:
        cfg['model'][key] = W
    return cfg


def set_lrs(cfg, lr):
    for key in LR_KEYS:
        cfg[key] = lr
    return cfg


def pin_batch(cfg, batch):
    """Pin the batch (growth off). Used by ONE arm only -- the critical-batch
    check. Every other arm inherits the parent's adaptive batching."""
    cfg['batch_size'] = batch
    cfg['grow_batch_size'] = False
    cfg['max_batch_size'] = batch
    return cfg


def set_bars(cfg, clip, clip_override=None):
    """Explicit clip + tripwire bars. `auto` would rebuild them from
    utils._GRAD_MEDIAN, which is pre-fix and ~3-9x high (see docstring).

    clip_override changes gradient_norm_clip ALONE and leaves the tripwire
    bars on the grid's own scale. That is what makes arm 8 a clean A/B: bars
    derived from its 37.9 clip would be cut_grad_abs 379, BELOW the measured
    healthy median of 337, so the arm would fire on its first steps and
    measure the tripwire instead of the clipping regime."""
    cfg['adaptive_lr']['cut_grad_abs'] = round(CUT_OVER_CLIP * clip, 1)
    cfg['adaptive_lr']['reset_grad_abs'] = round(RESET_OVER_CUT * CUT_OVER_CLIP * clip, 1)
    cfg['gradient_norm_clip'] = clip if clip_override is None else clip_override
    return cfg


def keep_recovery_inert(cfg):
    """recovery_target_frac <= cut_ratio => controller.py's re-ramp path never
    engages, so a fire is a terminal, unambiguous ceiling marker. Stamped
    explicitly so a future parent-config change cannot silently enable it
    mid-grid."""
    cfg['adaptive_lr']['recovery_target_frac'] = 0.5
    cfg['adaptive_lr']['cut_ratio'] = 0.5
    return cfg


def from_scratch(cfg):
    """No pre-existing artifact may enter. reuse_prior false is load-bearing
    twice over here: find_shared_prior matches on problem_def, which excludes
    T and architecture, so a T=60 prior would match a T=10 arm -- and a
    loaded prior SKIPS train_prior, leaving the policy random."""
    cfg['checkpoint_name'] = None
    cfg['prior_model_name'] = None
    cfg['reuse_prior'] = False
    cfg['continue_from_checkpoint'] = True   # own-run requeue only
    return cfg


# (index, run_name, T, W, lr, batch, seed_override, clip_override, note)
ARMS = [
    (0,  'tw_T10_lr08',        10,  512, 8.0e-4, None, None, None,
     'T=10 ladder: 8e-4 (T=60 ceiling carried across unscaled)'),
    (1,  'tw_T10_lr16',        10,  512, 1.6e-3, None, None, None,
     'T=10 ladder: 1.6e-3. MID CELL -- reference for width/batch/clip blocks'),
    (2,  'tw_T10_lr32',        10,  512, 3.2e-3, None, None, None,
     'T=10 ladder: 3.2e-3, just above the lr~1/T prediction of 2.4e-3'),
    (3,  'tw_T10_w256_lr16',   10,  256, 1.6e-3, None, None, None,
     'width 256 at the mid rung; narrower should tolerate MORE LR'),
    (4,  'tw_T10_w1024_lr08',  10, 1024, 8.0e-4, None, None, None,
     'width 1024, low rung -- the safe leg of the capacity pair'),
    (5,  'tw_T10_w1024_lr16',  10, 1024, 1.6e-3, None, None, None,
     'width 1024, mid rung -- pairs with arm 4 so worse != detonated'),
    (6,  'tw_T10_b_fixed1000', 10,  512, 1.6e-3, 1000,       None, None,
     'CRITICAL-BATCH CHECK: batch pinned at 1000 vs arm 1 adaptive. Matching '
     'per-step progress => past the knee, so the postfix batch drift was benign'),
    (7,  'tw_T25_lr32',        25,  512, 3.2e-3, None,       None, None,
     'T=25 third rung: brackets the T=25 ceiling as densely as T=10, which is '
     'what the ceiling-vs-T exponent fit actually needs'),
    (8,  'tw_T10_clip_resolver', 10, 512, 1.6e-3, None, None, 37.9,
     'clip A/B control leg: resolver clip 37.9 = 100% saturation. Treatment = arm 1'),
    (9,  'tw_T60_anchor',      60,  512, 4.0e-4, None, None, None,
     'anchor to postfix_july30 lr8x: same T/W/LR, fixed batch + explicit bars'),
    (10, 'tw_T25_lr08',        25,  512, 8.0e-4, None, None, None,
     'T=25: 8e-4, just below the 1/T prediction of 9.6e-4'),
    (11, 'tw_T25_lr16',        25,  512, 1.6e-3, None, None, None,
     'T=25: 1.6e-3, above the 1/T prediction'),
    (12, 'tw_T10_lr64',        10,  512, 6.4e-3, None, None, None,
     'T=10 ladder insurance rung, in case 3.2e-3 rides'),
    (13, 'tw_T10_lr16_s2',     10,  512, 1.6e-3, None, 20260731, None,
     'seed replicate of arm 1: calibrates between-arm noise'),
    (14, 'tw_T60_lr08',        60,  512, 8.0e-4, None, None, None,
     'T=60 at 8e-4 under the new bars: was lr16x fire physics or a 30x-clip bar?'),
    (15, 'tw_T10_w256_lr32',   10,  256, 3.2e-3, None, None, None,
     'width 256 at the high rung -- the "smaller and much faster" candidate'),
]


def build_all():
    log = []
    for index, run_name, T, W, lr, batch, seed, clip_override, note in ARMS:
        cfg = from_scratch(load_parent())
        set_T(cfg, T)
        set_width(cfg, W)
        set_lrs(cfg, lr)
        if batch is not None:
            pin_batch(cfg, batch)
        clip = clip_for(T, W)
        set_bars(cfg, clip, clip_override)
        keep_recovery_inert(cfg)
        cfg['epochs'] = BASE_EPOCHS
        if seed is not None:
            cfg['seed'] = seed
        cfg['tag'] = TAG
        cfg['run_name'] = run_name

        with (OUTDIR / f'{index}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)

        log.append({
            'index': index, 'run_name': run_name, 'note': note,
            'T': T, 'eval_T': cfg['eval_T'], 'W': W,
            'lr_fused': lr, 'batch_size': cfg['batch_size'],
            'batch_adaptive': bool(cfg['grow_batch_size']), 'seed': cfg['seed'],
            'gradient_norm_clip': cfg['gradient_norm_clip'],
            'clip_source': ('resolver value, A/B control leg (bars kept on the '
                            'grid scale so the clip is the only variable)'
                            if clip_override
                            else ('3 x MEASURED median' if T in GRAD_MEDIAN_MEASURED
                                  else '3 x INTERPOLATED median (T=25 unmeasured)')),
            'cut_grad_abs': cfg['adaptive_lr']['cut_grad_abs'],
            'epochs': BASE_EPOCHS,
            'decay_halflife_steps': cfg['adaptive_lr']['decay_halflife_steps'],
            'recovery_inert': True,
            'from_scratch': True,
        })

    with (OUTDIR / 'experiment_log.yaml').open('w', encoding='utf-8', newline='\n') as f:
        yaml.safe_dump(log, f, default_flow_style=False, sort_keys=False)
    return log


if __name__ == '__main__':
    for r in build_all():
        print(f"{r['index']:>2}  {r['run_name']:<22} T={r['T']:<3} W={r['W']:<5} "
              f"lr={r['lr_fused']:.1e}  batch={r['batch_size']:<5} "
              f"clip={r['gradient_norm_clip']:<7} cut={r['cut_grad_abs']:<8} "
              f"seed={r['seed']}")
