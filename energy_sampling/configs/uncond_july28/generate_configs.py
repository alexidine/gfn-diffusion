"""
uncond_july28 -- 15 runs (slurm array 0-14) on the UNCONDITIONAL mipcas ELJ
problem (sg2/zp1, all four conditioning flags false), single-phase naive
protocol, driven by the 2026-07-27 PROPORTIONAL controller rewrite.

Lineage: base_elj.yaml is replay_july26's verbatim (BOM-stripped). Every arm
warm-starts from the same T=60 phase1_exit snapshot at step 12970, so all 15
enter the naive stage with the same trained policy and nothing pays the ~13k-step
(~7.5h) phase-1 tax. The external control is replay_july26_naive_fixed
(1xz7zd9n), still running: fracs pinned 0.065/0.869/0.065, fwd/tb_err_worst
36.7 -> 20.9 monotone over 19.5k steps, EffDim flat at 5.87-5.99.

WHY THIS BATTERY EXISTS. replay_july26 ran the SAME naive protocol under the
lexicographic ("gentle") controller and under a fixed mix. Measured 2026-07-27:

  naive_fixed  (1xz7zd9n)  fracs never move   tb_err_worst 36.7 -> 20.9 monotone
                                              EffDim 5.97 stable
  naive_gentle (3koghy7g)  replay 0.05 -> 0.90  tb_err_worst LIMIT-CYCLES
                           fwd crushed to its   37/88/58/37/69/40/67/39/49/18.7/41.7
                           0.02 floor           EffDim decaying 5.99 -> 5.25

Four of five gentle arms died; both fixed arms are alive at 32.5k steps. The
same failure was reproduced independently on the toy under the PROPORTIONAL
controller (replay 0.05 -> 0.40 in 4/4 dev runs). Two different controllers,
one failure: replay share runs away and the fit roughly doubles.

Root cause, measured open-loop on the toy fixed-mix run (zcdgtwfv, rules: []):
the old split was a ratio of RAW drives s_i = max(metric_i - target_i, 0), so
targets outside a metric's operating band weld one side off and the other on,
and the ratio pins at its floor. Recalibrating alone does not fix it -- the two
metrics differ in units and dynamic range (over_coverage ~20 vs relative_under
~3.4), so replay wins on scale even with both targets inside their bands.

The 2026-07-27 rewrite (protocol.py) adds, both opt-in, default absolute:
  drive: relative   s_i = max(metric_i/target_i - 1, 0)          dimensionless
  max_fracs         ABSOLUTE per-mode ceiling (floor bounds shares
                    symmetrically and cannot cap one side alone)
and combines by TILTING the idle split rather than equalizing drives:
      share_a = w_a(1+s_a) / [w_a(1+s_a) + w_b(1+s_b)],  w = default_boost
so both sides at target reproduces default_boost EXACTLY. Equalizing drives was
tried first and rejected in simulation: at the healthy point it lands on replay
~0.43 of the batch, inside the range where every ratcheted run rang.

KEY SAFETY PROPERTY, and the reason targets can be set for a problem whose
bands are still moving: if targets are too HIGH both drives are 0 and the split
is default_boost exactly. Miscalibration degrades to the known-good FIXED MIX
instead of ratcheting. Arm 12 tests exactly that claim.

TARGET CALIBRATION comes from 1xz7zd9n at 32.5k steps -- the only settled
unconditional ELJ numbers we have, and both still slowly descending:
    bwd/relative_under   4.43 (peak, 19.6k) -> 3.37 and falling
    fwd/over_coverage   32.0 (13k)          -> 20.3 and falling
Reference targets are set just under the current values so both drives are live
but modest: bwd 3.0 (drive 0.12), replay 18.0 (drive 0.13). Near-balanced, so
the split sits near default_boost and the controller expresses only sustained
imbalance -- by construction, not by luck.

  ARMS (all T=60 / LR 5.0e-5 / warm-start unless stated; single-knob deltas
  vs arm 0 except where noted):

  controller (0-4, 11-12) -- the primary question
   0 : prop_ref        reference proportional controller
   1 : fixed_null      fixed mix, rules []: the zero-controller null that
                       prices the controller under IDENTICAL code (1xz7zd9n
                       is the same mix but predates the rewrite)
   2 : prop_alpha01    alpha 0.002 -> 0.01 (5x faster; ~1k-step time constant,
                       deliberately INSIDE the 1-2k absorption cycle to price
                       the "must be slower than the plant" rule)
   4 : prop_fwd20      fwd pinned 0.1 -> 0.2: more on-policy exposure. NB fwd
                       is Z's ONLY trainer (bwd/replay carry freeze_z)

  CONTROLLED REPLAY DOSE-RESPONSE (3, 11, 12) -- the question the ratcheting
  batteries could never answer. Replay share has only ever been observed at
  the good fixed level (0.05-0.11) or at runaway (0.29-0.90), never HELD in
  between, so we cannot say whether the ratchet's damage came from the LEVEL
  replay reached or from the fact that it MOVED. These hold it steady at
  three intermediate levels via the idle split (default_boost sets the
  operating point directly; targets and caps cannot -- see below):
   3 : prop_r15        idle 0.85/0.15 -> replay ~0.135, cap 0.25
  11 : prop_r30        idle 0.70/0.30 -> replay ~0.27,  cap 0.40
  12 : prop_r50        idle 0.50/0.50 -> replay ~0.45,  cap 0.55. This is
                       INSIDE the range where every ratcheted arm rang. If a
                       HELD 0.45 is stable, the damage was the movement (and
                       the limit cycle is the disease); if it rings anyway,
                       the level itself is toxic and the cap is the cure.

  WHY NOT SWEEP TARGETS OR THE CAP: with drive: relative the split depends on
  the RATIO of (1+s), so proportionally-similar targets give proportionally-
  similar drives and the tilt stays ~1. Checked at the settled operating point
  (relative_under 3.367, over_coverage 20.28): targets 3.0/18.0, 2.5/15.0 and
  3.6/21.0 ALL settle to bwd 0.846 / replay 0.054 -- identical runs. The cap at
  0.2 never binds there either. Target and cap arms would have burned four
  slots on replicates; the idle split is the knob that actually moves the
  operating point.

  rollout length (5-8, 13) -- LRs via resolve_derived_config ('auto'), which
  is the executable source of truth for the (W,T) rescale. Arm 5 is the
  matched T=60 anchor so the T axis is read at a consistent LR policy, NOT
  against the hand-set 5e-5 arms. Warm-starting a T=60 policy into a longer
  rollout is a discretization change -- the policy is time-conditioned so it
  loads and adapts, but arm 5 is the honest control for it.
   5 : prop_T60_auto   T=60,  lr_fused 2.08e-5 (also the LR-axis anchor)
   6 : prop_T100       T=100, lr_fused 1.25e-5
   7 : prop_T150       T=150, lr_fused 8.3e-6
   8 : prop_T200       T=200, lr_fused 6.25e-6
  13 : prop_T100_ckpt  T=100 + traj_checkpoint: activation memory O(1) in T
                       instead of O(T) (~33.6x VRAM at T=100, ~1.7x time,
                       bitwise-identical incl. RNG). NOT yet cluster-validated
                       -- this arm validates it against arm 6, which is the
                       same config with checkpointing off

  learning rate (9-10), around the arm-5 anchor
   9 : prop_lr2x       2x auto (4.17e-5): approaches the hand-set 5e-5 the
                       adaptive controller cut to 1.42e-5 on 1xz7zd9n
  10 : prop_lr_half    0.5x auto (1.04e-5)

  14 : prop_compound   best-guess next default: T=100 auto + alpha 0.005 +
                       cap 0.3. The only multi-knob arm; uninterpretable
                       alone, informative if the single-knob arms agree

NOT SWEPT, deliberately: model width/depth. The phase1_exit checkpoint is
architecture-locked, so any width change forces a fresh ~13k-step phase 1
(~7.5h of an overnight window) and leaves almost no naive-stage signal. Width
needs its own battery with a pretrained per-width prior, not slots here.

READING GUIDE. Primary outcome is step-matched fwd/tb_err_worst AT MATCHED
EffDim -- the staged arms reach ~7 nats at EffDim 1.75 (collapse), so a low
error at low EffDim is not a win. Then: Replay Frac trajectory (the failure
signature is monotone climb; the design predicts it stays inside default_boost
+- the cap), fwd/logw_std flat vs sawtooth, and whether the adaptive LR
controller ever fires. protocol/prop_drive_bwd and prop_drive_replay are now
logged in the stage's DECLARED drive form (they share _drive with the tick;
report() previously hardcoded the absolute form and the two silently diverged),
so a welded drive is directly visible as a channel pinned at 0.
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent

# same phase1_exit snapshot as replay_july26 (step 12970), schema_v1-restamped
CKPT_FILE = ('stab_july21c_elj_h512x4_T60_lr5.0e-5_elj-mipcas_sg2_zp1_'
             'elj_prior_dataset-T2.5-68890c_phase1_exit.pt')

WIDTH, LAYERS, TRAJ_LEN, HAND_LR = 512, 4, 60, 5.0e-5
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

# resolve_derived_config anchors (utils.py) -- mirrored here ONLY to compute
# explicit multiples for the LR-axis arms. 'auto' arms never touch these.
LR_ANCHORS = {'lr_policy': 1.0e-4, 'lr_replay': 1.0e-4,
              'lr_back': 1.0e-4, 'lr_fused': 5.0e-5}
T_REF = 25

# reference targets: just under 1xz7zd9n's settled values (relative_under 3.37,
# over_coverage 20.3 at 32.5k, both still falling)
REF_TARGETS = {'bwd': 3.0, 'replay': 18.0}
REF_IDLE = {'bwd': 0.94, 'replay': 0.06}
REF_FWD = 0.1


def make_base(traj_len=TRAJ_LEN, auto_lr=False, lr_scale=None,
              traj_checkpoint=False):
    """
    Battery base. auto_lr leaves the four rollout LRs as 'auto' so
    resolve_derived_config scales them by T_REF/T at load (the documented and
    executable rescale path); lr_scale multiplies those resolved values by an
    explicit factor. gradient_norm_clip and the adaptive_lr bars are ALWAYS
    left 'auto' -- they carry the grad_median(T) and sqrt(W) terms, and
    hand-scaling them is what the checklist exists to prevent.
    """
    with (OUTDIR / 'base_elj.yaml').open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for key in WIDTH_KEYS:
        config['model'][key] = WIDTH
    for key in LAYER_KEYS:
        config['model'][key] = LAYERS

    config['integrator']['T'] = traj_len
    config['integrator']['min_traj_length'] = traj_len
    config['integrator']['max_traj_length'] = traj_len
    config['eval_T'] = traj_len          # eval_T == train T (discretization-locked)

    if auto_lr or lr_scale is not None:
        for key in LR_KEYS:
            if lr_scale is None:
                config[key] = 'auto'
            else:
                config[key] = LR_ANCHORS[key] * T_REF / traj_len * lr_scale
    else:
        for key in LR_KEYS:
            config[key] = HAND_LR

    # let the resolver own every (W,T)-derived guard rail
    config['gradient_norm_clip'] = 'auto'
    config['adaptive_lr']['cut_grad_abs'] = 'auto'
    config['adaptive_lr']['reset_grad_abs'] = 'auto'
    config['adaptive_lr']['reset_loss_abs'] = 'auto'

    config['traj_checkpoint'] = traj_checkpoint
    config['max_batch_size'] = 50000

    # variance schedule OFF everywhere (identical to the constant-rate path)
    config['model']['t_scale_ratio'] = None
    config['model']['t_scale_power'] = 4.0
    config['model']['t_scale_preserve_budget'] = True

    config['tag'] = 'uncond_july28'
    config['checkpoint_name'] = CKPT_FILE
    config['load_weights_only'] = False
    config['epochs'] = 100000
    config['eval_period'] = 500
    config['figs_period'] = 1000
    return config


def naive_stage(fracs, balance):
    return {
        'name': 'naive',
        'train_mode': 'fused',
        'bwd_sampling_mode': 'prior',
        'flags': {'buffers_active': True, 'update_log_z': True},
        'fracs': fracs,
        # INERT under kind: proportional (floors live in _nudge_mode_fracs,
        # which only the lexicographic path calls) -- kept so a switch back to
        # lexicographic doesn't silently lose them. The proportional `floor`
        # does this job.
        'min_fracs': {'fwd': 0.02, 'bwd': 0.05, 'replay': 0.02},
        'deactivate_threshold': 0.01,
        'loss_coeffs': {
            'fwd': {'tb': 1.0},   # Z + policy grads (fwd is Z's only trainer)
            'bwd': {'tb': 1.0},   # freeze_z: 1 from base -> policy only
        },
        'balance': balance,
    }


def prop_balance(targets=None, idle=None, fwd=REF_FWD, alpha=0.002,
                 cap_replay=0.2, floor=0.03):
    targets = dict(targets or REF_TARGETS)
    idle = dict(idle or REF_IDLE)
    return {
        'kind': 'proportional',
        'pinned': {'fwd': fwd},          # must equal fracs.fwd
        'metrics': {'bwd': 'bwd/relative_under', 'replay': 'fwd/over_coverage'},
        'drive': 'relative',             # s = metric/target - 1
        'targets': targets,
        'max_fracs': {'replay': cap_replay},
        'default_boost': idle,           # the split when both sides are at target
        'alpha': alpha,                  # time constant ~10/alpha train steps
        'floor': floor,                  # share of the PAIR's mass
    }


def prop_arm(config=None, fwd=REF_FWD, idle=None, **balance_kwargs):
    """Naive stage under the proportional controller. Entry fracs put the
    pair's mass at the idle ratio so the run starts where it would settle."""
    config = config if config is not None else make_base()
    idle = dict(idle or REF_IDLE)
    pair = round(1.0 - fwd, 6)
    fracs = {'fwd': fwd,
             'bwd': round(pair * idle['bwd'], 4),
             'replay': round(pair * idle['replay'], 4)}
    balance = prop_balance(fwd=fwd, idle=idle, **balance_kwargs)
    train_prior = next(s for s in config['protocol']['stages']
                       if s['name'] == 'train_prior')
    config['protocol']['stages'] = [train_prior, naive_stage(fracs, balance)]
    return config


def fixed_arm(config=None, fracs=None):
    """
    Zero-controller null: lexicographic with an EMPTY rule list = pure idle
    nudge toward default_boost.

    NB under lexicographic the entry fracs are a TARGET, not the literal split:
    _nudge_mode_fracs settles at floors + (1 - sum floors) * target, so with
    min_fracs summing to 0.09 the entry (0.05, 0.9, 0.05) settles at
    (0.0655, 0.869, 0.0655) -- which is EXACTLY where 1xz7zd9n sits. Entering
    at the already-settled numbers instead would land somewhere else and quietly
    stop being a replicate.
    """
    config = config if config is not None else make_base()
    fracs = fracs or {'fwd': 0.05, 'bwd': 0.9, 'replay': 0.05}
    balance = {'kind': 'lexicographic',
               'default_boost': dict(fracs),
               'rules': []}
    train_prior = next(s for s in config['protocol']['stages']
                       if s['name'] == 'train_prior')
    config['protocol']['stages'] = [train_prior, naive_stage(fracs, balance)]
    return config


def emit(ind, config, name, note):
    config['run_name'] = name
    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    naive = next(s for s in config['protocol']['stages'] if s['name'] == 'naive')
    bal = naive['balance']
    return {'index': ind, 'run_name': name, 'note': note,
            'kind': bal['kind'],
            'T': config['integrator']['T'],
            'lr_fused': config['lr_fused'],
            'traj_checkpoint': config['traj_checkpoint'],
            'fracs': naive['fracs'],
            'targets': bal.get('targets'),
            'alpha': bal.get('alpha'),
            'cap_replay': (bal.get('max_fracs') or {}).get('replay')}


if __name__ == '__main__':
    e = []

    # ---- controller axis (T=60, hand LR 5e-5, matched to replay_july26) ----
    e.append(emit(0, prop_arm(), 'prop_ref',
                  'reference proportional controller: targets 3.0/18.0, cap 0.2, alpha 0.002'))
    e.append(emit(1, fixed_arm(), 'fixed_null',
                  'fixed mix, entry 0.05/0.9/0.05 -> settles 0.0655/0.869/0.0655 = 1xz7zd9n exactly'))
    e.append(emit(2, prop_arm(alpha=0.01), 'prop_alpha01',
                  'alpha 0.002 -> 0.01: response INSIDE the absorption cycle'))
    e.append(emit(3, prop_arm(idle={'bwd': 0.85, 'replay': 0.15}, cap_replay=0.25),
                  'prop_r15',
                  'idle replay share 0.06 -> 0.15 (replay ~0.135): dose-response step 1'))
    e.append(emit(4, prop_arm(fwd=0.2), 'prop_fwd20',
                  'fwd pinned 0.1 -> 0.2: more on-policy exposure / faster Z'))

    # ---- rollout length (auto LRs; arm 5 is the matched anchor) ----
    e.append(emit(5, prop_arm(make_base(auto_lr=True)), 'prop_T60_auto',
                  'T=60 at resolver LRs (lr_fused 2.08e-5): T-axis + LR-axis anchor'))
    e.append(emit(6, prop_arm(make_base(traj_len=100, auto_lr=True)), 'prop_T100',
                  'T=100, resolver LRs'))
    e.append(emit(7, prop_arm(make_base(traj_len=150, auto_lr=True)), 'prop_T150',
                  'T=150, resolver LRs'))
    e.append(emit(8, prop_arm(make_base(traj_len=200, auto_lr=True)), 'prop_T200',
                  'T=200, resolver LRs'))

    # ---- learning rate, around the arm-5 anchor ----
    e.append(emit(9, prop_arm(make_base(lr_scale=2.0)), 'prop_lr2x',
                  '2x resolver LR (lr_fused 4.17e-5)'))
    e.append(emit(10, prop_arm(make_base(lr_scale=0.5)), 'prop_lr_half',
                  '0.5x resolver LR (lr_fused 1.04e-5)'))

    # ---- target placement ----
    e.append(emit(11, prop_arm(idle={'bwd': 0.70, 'replay': 0.30}, cap_replay=0.40),
                  'prop_r30',
                  'idle replay share 0.30 (replay ~0.27): dose-response step 2'))
    e.append(emit(12, prop_arm(idle={'bwd': 0.50, 'replay': 0.50}, cap_replay=0.55),
                  'prop_r50',
                  'idle replay share 0.50 (replay ~0.45): LEVEL-vs-MOVEMENT test'))

    # ---- checkpointing validation + compound ----
    e.append(emit(13, prop_arm(make_base(traj_len=100, auto_lr=True,
                                         traj_checkpoint=True)), 'prop_T100_ckpt',
                  'arm 6 + traj_checkpoint: O(1)-in-T activations, validates vs arm 6'))
    e.append(emit(14, prop_arm(make_base(traj_len=100, auto_lr=True),
                               alpha=0.005, cap_replay=0.3), 'prop_compound',
                  'compound best-guess: T=100 + alpha 0.005 + cap 0.3'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(e, f, sort_keys=False)

    print(f'Generated {len(e)} configs with log at {OUTDIR / "experiment_log.yaml"}')
    hdr = (f'{"idx":>3s} {"run":>16s} {"kind":>13s} {"T":>4s} {"lr_fused":>10s} '
           f'{"ckpt":>5s} {"fwd":>5s} {"bwd":>6s} {"rep":>5s} {"targets":>16s} '
           f'{"alpha":>6s} {"cap":>4s}')
    print(hdr)
    for r in e:
        t = r['targets']
        ts = f"{t['bwd']}/{t['replay']}" if t else '-'
        lr = r['lr_fused']
        lrs = lr if isinstance(lr, str) else f'{lr:.3g}'
        print(f'{r["index"]:>3d} {r["run_name"]:>16s} {r["kind"]:>13s} {r["T"]:>4d} '
              f'{lrs:>10s} {str(r["traj_checkpoint"]):>5s} '
              f'{r["fracs"]["fwd"]:>5.3f} {r["fracs"]["bwd"]:>6.3f} '
              f'{r["fracs"]["replay"]:>5.3f} {ts:>16s} '
              f'{str(r["alpha"]):>6s} {str(r["cap_replay"]):>4s}')
