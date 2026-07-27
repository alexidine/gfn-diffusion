"""
replay_july26 -- 16 runs (slurm array 0-15) scanning the replay-buffer
redesign's knobs, on the tsched_july24 lineage: same stamped phase1_exit
snapshot (stab_july21c index 6, o06f3c2z, elj_h512x4_T60_lr5.0e-5), same
geometry and T-scaled tripwires, variance schedule OFF everywhere (the
tsched battery parked it: physically active but not the TB-convergence
lever, and low terminal noise embrittles the pinned-path replay anchor).
base_elj.yaml = tsched_july24's verbatim except admit_temperature 5 -> 2:
the REFERENCE arm here is sw2v8yas's winning config (T 2, TTL 250, cap
8-30 @ h0 10, churn 50, buildout replay share 0.1), which beat the T=5
control to a LOWER fwd/tb_err_worst floor (6.9 vs 7.4) ~20k steps sooner.

Code prerequisites: the 2026-07-25 train.py (SUPPLY-side replay admission
-- the v1 demand-side pacing deadlocked 10/14 tsched arms) + 2026-07-24
buffer.py (birth_step/birth_loss). Parent checkpoints already carry the
schema_version restamp (see tsched_july24/restamp_schema_v1.py).

WHAT THE BATTERY ASKS, by axis (all arms single-knob deltas vs arm 0
unless noted):

  T continuation (1-3): tsched found admit_temperature MONOTONE toward
  sharp over {2, 5, 20}. Does it keep paying below 2, or does near-argmax
  admission re-poison the buffer now that the cap+TTL belt exists? t5
  doubles as the cross-battery bridge to tsched's control.
  Prediction if the belt is the whole story: 0.5 <= 1 <= 2 all fine.

  Replay share (4-6): the trailing-anchor theory says share is the
  anchor's STIFFNESS. Never scanned (0.1 fixed everywhere so far).
  Dose-response of the buildout default_boost replay share.

  TTL fine bracket (7-8): tsched bracketed coarsely -- 100 ok, 250 good,
  1000 bad (stale anchor: rows drawn ~360x, expired_delta positive).

  Mechanism falsifiers (9-11): replay EXCISED from buildout (share 0,
  buffer still fills -- supply-side admission runs on fwd steps -- only
  the training draw is removed), with bwd raised as the alternative
  spread-owner. If bwd 0.1-0.3 stabilizes buildout without replay, replay
  isn't special -- it was just the only spread-owner awake (bwd owns
  SPREAD per the stationarity result, and buildout pins bwd at 0.001).
  If bwd can't substitute, the pinned-path anchor does something backward
  rollouts can't (fixed paths expose policy drift that fresh bwd rollouts
  re-absorb). norep = designed replication of the accidental
  fwd-alone-detonates observation (dead-buffer arms, logw_std sawtooth).
  NB the excision is BUILDOUT-scoped: terminal's config is untouched, but
  no norep arm is expected to exit buildout anyway.

  Extras (12-15): t1_share02 = compound best-guess (next-default
  candidate if T and share both pay). fixedcap re-runs the
  health-modulation ablation AT the new T=2 reference (tsched's was T=5
  and died young). churn25/churn100 scan intake rate: equilibrium length
  ~ min(max_size, churn_rate x TTL), so 25 runs a deliberately THIN
  buffer (~6.2k) and 100 a displacement-churned one (mean age ~50).

   0 : control_t2     -- reference; replicates sw2v8yas fresh
   1 : t05            -- admit_temperature 0.5 (near-argmax)
   2 : share03        -- buildout default_boost replay 0.3
   3 : norep          -- replay share 0, bwd left dormant (0.001)
   4 : bwd03_norep    -- replay 0, bwd pinned at 0.3 (if 0.3 can't
                         substitute for replay's anchoring, 0.1 can't)
   5 : fixedcap       -- admit_cap_min = admit_cap_max = 30 (cap(h) off)
  6-15 : single-phase naive protocol -- see REVISION v2/v3 below

READING GUIDE: outcome = step-matched fwd/tb_err_worst (floor LEVEL and
arrival step), fwd/r2, fwd/logw_std stability (sawtooth = lost anchor).
replay_buffer_expired_delta SIGN is the anchor-health readout (negative =
resident rows improving; positive on every diverging tsched arm).
absorbed_frac stays near zero by design (calibration mandate: rows cycle).
Watch replay_buffer_admit_cap/admit_health around the ~15k transition
kick -- if cap never leaves its healthy end, the fixedcap arm is
uninformative on the health axis. The best tsched arms plateaued at ~7
nats independent of terminal noise -- if ANY arm here goes well below 7
(t1/share03 are the candidates), the plateau was anchor-limited, NOT the
target's own within-condition spread, and the buildout exit bar (5.0) may
finally be reachable.

REVISION v2 (2026-07-26, arms 0-1 already running and byte-untouched;
2-15 reorganized before launch). Six refinement arms cut (t5, share005,
ttl150, ttl400, churn25, churn100 -- bridges and fine brackets, not
direction-setters) to make room for the SINGLE-PHASE NAIVE protocol
(arms 10-15): one always-on stage after train_prior, no other stages, no
transitions, no on_enter surgery. fwd (tb: Z + policy grads), bwd
(tb, freeze_z from base: policy only), replay (tb, freeze_z from base:
policy only) all permanently active; buffers_active + churn from step
one; entry fracs = idle default_boost, bwd-DOMINANT (spread owned from
step one, per the bwd-owns-spread stationarity result -- buildout's
instability traced to spread being unowned at bwd 0.001). Zero code: the
declarative stage engine makes this a config-only protocol variant.

The gentle controller (naive_gentle_* arms), v3 per user direction --
replay is the workhorse for everything downstream of Z: it tames fwd
scatter_err at the extremes and combats mode concentration, while
boosting FWD for its own calibration is inefficient (fwd is bad at
converging fwd r2) and unmanaged fwd share costs mode forgetting:
  1. fwd/tb_resid_clipped > 0.5 -> boost fwd. The Z ruler -- fwd is Z's
     only trainer, the one job that genuinely needs fwd share. NB at
     phase1_exit entry Z sits at the MLE warm-start level, so this rule
     fires hard early and boosts fwd until Z walks down to the on-policy
     level -- an EMERGENT, continuous z_match with no stage boundary.
  2. bwd/tb_err_worst > 25 -> boost bwd, annealing toward 5 at per-rule
     rate 0.98 (slow: ~4k+ clean steps to close).
  3. fwd/tb_err_worst relative-to-own-best (margin 1.5, drift 0.003) ->
     boost REPLAY: on-policy calibration degrading = spread/retention
     work, replay's job -- not fwd's.
  4. fwd/over_coverage > 8 -> boost replay: mode-concentration tamp,
     egregious-outliers-only, UNANNEALED (the c8utdn8q lesson: annealing
     this bar ratchets onto the metric's natural level).
Still deliberately NO replay-side-metric rule: replay error metrics are
admission-policy artifacts and may never GROW replay's share (the
buildout rule-3 lesson).
naive_fixed_* arms have NO rules at all -- fracs pinned at entry values
(the user's 'constant low replay level' variant), the zero-controller
null that prices the gentle controller itself.

   6 : naive_fixed        -- (fwd 0.05, bwd 0.9, replay 0.05), no rules
   7 : naive_fixed_b80    -- (0.1, 0.8, 0.1), no rules
   8 : naive_gentle       -- (0.05, 0.9, 0.05) + gentle controller
   9 : naive_gentle_b80   -- (0.1, 0.8, 0.1)
  10 : naive_gentle_b70   -- (0.2, 0.7, 0.1)
  11 : naive_gentle_r25   -- (0.05, 0.7, 0.25): replay-heavy anchor
  12 : naive_gentle_r40   -- (0.05, 0.55, 0.4): where does replay
                             dominance saturate or hurt?
  13 : naive_gentle_fwd30 -- (0.3, 0.6, 0.1): faster Z / more on-policy
  14 : naive_gentle_repidle -- entry (0.05, 0.9, 0.05) but IDLE
                             default_boost (0.05, 0.65, 0.3): the literal
                             'boost replay on default' -- idle drifts
                             replay-ward, rules unchanged
  15 : naive_gentle_lr1e4 -- (0.05, 0.9, 0.05) at DOUBLE LR (1e-4 on
                             policy/back/replay/fused). Forward-first
                             detonates at 1e-4 (the TB LR ceiling); if
                             bwd-dominant naive survives it, the
                             stability claim has teeth AND the protocol
                             buys wall-clock

Naive reading guide: stability first (fwd/logw_std flat vs sawtooth, LR
controller quiet, no terminal rewinds -- lr1e4 is the acid test), then
time-to-plateau and floor LEVEL vs the staged arms 0-5. Watch
gates/delta_worst as a passive diagnostic of the emergent z_match (no
gate reads it here); early fwd-frac elevation then relaxation = rule 1
doing the handoff. naive_fixed vs naive_gentle (x2 mixes) prices the
controller; b90->b70 + r25/r40 scan how the anchor share trades against
bwd dominance; repidle isolates idle-attractor placement from rule
response.
"""

from copy import deepcopy
from pathlib import Path

import yaml

OUTDIR = Path(__file__).parent

# o06f3c2z's phase1_exit snapshot -- already schema_version-restamped
CKPT_FILE = ('stab_july21c_elj_h512x4_T60_lr5.0e-5_elj-mipcas_sg2_zp1_'
             'elj_prior_dataset-T2.5-68890c_phase1_exit.pt')

# parent run geometry: stab_july21c index 6 (identical to tsched_july24)
WIDTH, LAYERS, TRAJ_LEN, PEAK_LR = 512, 4, 60, 5.0e-5
WIDTH_KEYS = ('s_emb_dim', 't_hidden_dim', 's_hidden_dim',
              'policy_hidden_dim', 'flow_hidden_dim', 'cond_hidden_dim')
LAYER_KEYS = ('s_layers', 'policy_layers', 'flow_layers', 'cond_layers')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')

# stab_july21c's T=25 baseline tripwire bars, scaled linearly by T/25 --
# identical to tsched_july24/jacob_july24 so every battery shares the
# parent's protection exactly
T_BASELINE = 25
CLIP_BASELINE = 250.0
CUT_LOSS_BASELINE = 2.5e+3
CUT_GRAD_BASELINE = 7.5e+3
RESET_LOSS_BASELINE = 2.5e+4
RESET_GRAD_BASELINE = 7.5e+4


def make_base():
    with (OUTDIR / 'base_elj.yaml').open('r') as f:
        config = yaml.safe_load(f)

    for key in WIDTH_KEYS:
        config['model'][key] = WIDTH
    for key in LAYER_KEYS:
        config['model'][key] = LAYERS
    config['integrator']['T'] = TRAJ_LEN
    config['integrator']['min_traj_length'] = TRAJ_LEN
    config['integrator']['max_traj_length'] = TRAJ_LEN
    config['eval_T'] = TRAJ_LEN
    for key in LR_KEYS:
        config[key] = PEAK_LR

    factor = TRAJ_LEN / T_BASELINE
    config['gradient_norm_clip'] = round(CLIP_BASELINE * factor, 1)
    config['adaptive_lr']['cut_loss_abs'] = round(CUT_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['cut_grad_abs'] = round(CUT_GRAD_BASELINE * factor, 1)
    config['adaptive_lr']['reset_loss_abs'] = round(RESET_LOSS_BASELINE * factor, 1)
    config['adaptive_lr']['reset_grad_abs'] = round(RESET_GRAD_BASELINE * factor, 1)
    config['max_batch_size'] = 50000

    # variance schedule OFF for the whole battery (identical to the
    # constant-rate parent code path, bitwise-verified in tsched_july24)
    config['model']['t_scale_ratio'] = None
    config['model']['t_scale_power'] = 4.0
    config['model']['t_scale_preserve_budget'] = True

    config['tag'] = 'replay_july26'
    config['checkpoint_name'] = CKPT_FILE
    config['load_weights_only'] = False
    config['epochs'] = 60000
    config['eval_period'] = 500
    config['figs_period'] = 1000
    return config


def buildout_stage(config):
    return next(s for s in config['protocol']['stages'] if s['name'] == 'buildout')


def replay_arm(**overrides):
    """Reference config with replay-buffer knobs overridden."""
    config = make_base()
    config['buffers']['replay_buffer'].update(overrides)
    return config


def share_arm(replay_share, config=None):
    """Buildout idle replay share (default_boost); fwd takes the rest."""
    config = config or make_base()
    stage = buildout_stage(config)
    stage['balance']['default_boost'] = {'fwd': round(1.0 - replay_share, 3),
                                         'replay': replay_share}
    return config


NAIVE_RULES = [
    # Z is the ruler -- fires hard at entry (Z at the MLE warm-start level)
    # and quiets once Z reaches the on-policy level: the emergent z_match.
    # The one job that genuinely needs fwd share (fwd is Z's only trainer)
    {'metric': 'fwd/tb_resid_clipped', 'abs': True, 'above': 0.5, 'boost': 'fwd'},
    # bwd fit floor: high start, slow per-rule anneal toward the staged
    # protocol's working bar
    {'metric': 'bwd/tb_err_worst', 'above': 25.0, 'boost': 'bwd',
     'anneal': {'min': 5.0, 'rate': 0.98}},
    # on-policy calibration degrading vs own best -> REPLAY, not fwd:
    # replay tames fwd scatter extremes; boosting fwd is inefficient at
    # converging fwd's own fit and unmanaged fwd costs mode forgetting
    {'metric': 'fwd/tb_err_worst', 'relative': 'best', 'margin': 1.5,
     'drift': 0.003, 'boost': 'replay'},
    # mode-concentration tamp: egregious-outliers-only, UNANNEALED
    # (c8utdn8q: annealing ratchets this bar onto the metric's natural level)
    {'metric': 'fwd/over_coverage', 'above': 8.0, 'boost': 'replay'},
]


def naive_arm(fwd, bwd, replay, rules=True, default_boost=None, peak_lr=None):
    """
    Single-phase naive protocol: [train_prior, naive] -- one always-on
    fused stage, no transitions, no on_enter surgery. Entry fracs double
    as the idle default_boost unless default_boost overrides (the
    'boost replay on default' variant). min_fracs all sit above
    deactivate_threshold: nothing may ever switch off. peak_lr overrides
    the four rollout LRs (lr_flow untouched).
    """
    config = make_base()
    train_prior = next(s for s in config['protocol']['stages']
                       if s['name'] == 'train_prior')
    naive = {
        'name': 'naive',
        'train_mode': 'fused',
        'bwd_sampling_mode': 'prior',
        'flags': {'buffers_active': True, 'update_log_z': True},
        'fracs': {'fwd': fwd, 'bwd': bwd, 'replay': replay},
        'min_fracs': {'fwd': 0.02, 'bwd': 0.05, 'replay': 0.02},
        'deactivate_threshold': 0.01,
        'loss_coeffs': {
            'fwd': {'tb': 1.0},   # Z + policy grads
            'bwd': {'tb': 1.0},   # freeze_z: 1 from base -> policy only
        },
        'balance': {
            'kind': 'lexicographic',
            'default_boost': default_boost or {'fwd': fwd, 'bwd': bwd, 'replay': replay},
            'rules': deepcopy(NAIVE_RULES) if rules else [],
        },
    }
    config['protocol']['stages'] = [train_prior, naive]
    if peak_lr is not None:
        for key in LR_KEYS:
            config[key] = peak_lr
    return config


def falsifier_arm(bwd_frac):
    """
    Replay excised from buildout (share 0; the buffer still FILLS via
    supply-side admission -- only the training draw is removed), bwd
    raised to bwd_frac as the alternative spread-owner. bwd_frac below
    the 0.01 deactivate threshold leaves bwd dormant (pure excision).
    min_fracs pin bwd so the lexicographic layer can't starve it back to
    the floor (s706frkh).
    """
    config = make_base()
    stage = buildout_stage(config)
    bwd = max(bwd_frac, 0.001)
    stage['fracs'] = {'fwd': round(0.999 - bwd, 3), 'bwd': bwd, 'replay': 0.0}
    stage['min_fracs'] = {'fwd': 0.02, 'bwd': bwd, 'replay': 0.0}
    if bwd_frac >= 0.01:
        stage['balance']['default_boost'] = {'fwd': round(1.0 - bwd, 3), 'bwd': bwd}
    else:
        stage['balance']['default_boost'] = {'fwd': 1.0}
    return config


def emit(ind, config, name, note):
    config['run_name'] = name
    with open(OUTDIR / f'{ind}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    rb = config['buffers']['replay_buffer']
    stage_names = [s['name'] for s in config['protocol']['stages']]
    protocol = 'naive' if 'naive' in stage_names else 'forward_first'
    active = next(s for s in config['protocol']['stages']
                  if s['name'] in ('naive', 'buildout'))
    boost = active['balance']['default_boost']
    return {'index': ind, 'run_name': name, 'note': note,
            'protocol': protocol,
            'n_rules': len(active['balance']['rules']),
            'admit_temperature': rb['admit_temperature'],
            'max_residence_steps': rb['max_residence_steps'],
            'cap_min_max_h0': [rb['admit_cap_min'], rb['admit_cap_max'],
                               rb['admit_cap_health_h0']],
            'churn_rate': rb['churn_rate'],
            'fwd_share': boost.get('fwd', 0.0),
            'bwd_share': boost.get('bwd', 0.0),
            'replay_share': boost.get('replay', 0.0)}


if __name__ == '__main__':
    entries = []

    entries.append(emit(0, make_base(), 'control_t2',
                    'reference: T 2, TTL 250, share 0.1 -- fresh replication of sw2v8yas'))

    entries.append(emit(1, replay_arm(admit_temperature=0.5), 't05',
                    'admit_temperature 0.5: near-argmax admission'))

    entries.append(emit(2, share_arm(0.3), 'share03',
                    'buildout replay share 0.1 -> 0.3'))
    entries.append(emit(3, falsifier_arm(0.0), 'norep',
                    'replay excised, bwd dormant: designed replication of fwd-alone instability'))
    entries.append(emit(4, falsifier_arm(0.3), 'bwd03_norep',
                    'replay excised, bwd pinned 0.3: can bwd own spread instead?'))
    entries.append(emit(5, replay_arm(admit_cap_min=30.0), 'fixedcap',
                    'cap_min = cap_max = 30: health modulation off, at the T=2 reference'))

    entries.append(emit(6, naive_arm(0.05, 0.9, 0.05, rules=False), 'naive_fixed',
                    'single-phase naive, fracs pinned (0.05/0.9/0.05), NO controller'))
    entries.append(emit(7, naive_arm(0.1, 0.8, 0.1, rules=False), 'naive_fixed_b80',
                    'single-phase naive, fracs pinned (0.1/0.8/0.1), NO controller'))
    entries.append(emit(8, naive_arm(0.05, 0.9, 0.05), 'naive_gentle',
                    'single-phase naive (0.05/0.9/0.05) + gentle 4-rule controller'))
    entries.append(emit(9, naive_arm(0.1, 0.8, 0.1), 'naive_gentle_b80',
                    'single-phase naive (0.1/0.8/0.1)'))
    entries.append(emit(10, naive_arm(0.2, 0.7, 0.1), 'naive_gentle_b70',
                    'single-phase naive (0.2/0.7/0.1)'))
    entries.append(emit(11, naive_arm(0.05, 0.7, 0.25), 'naive_gentle_r25',
                    'single-phase naive (0.05/0.7/0.25): replay-heavy anchor'))
    entries.append(emit(12, naive_arm(0.05, 0.55, 0.4), 'naive_gentle_r40',
                    'single-phase naive (0.05/0.55/0.4): replay-dominance saturation probe'))
    entries.append(emit(13, naive_arm(0.3, 0.6, 0.1), 'naive_gentle_fwd30',
                    'single-phase naive (0.3/0.6/0.1): faster Z, more on-policy exposure'))
    entries.append(emit(14, naive_arm(0.05, 0.9, 0.05,
                                      default_boost={'fwd': 0.05, 'bwd': 0.65, 'replay': 0.3}),
                    'naive_gentle_repidle',
                    'entry (0.05/0.9/0.05), idle default_boost (0.05/0.65/0.3): boost replay on default'))
    entries.append(emit(15, naive_arm(0.05, 0.9, 0.05, peak_lr=1.0e-4), 'naive_gentle_lr1e4',
                    'single-phase naive (0.05/0.9/0.05) at LR 1e-4: the TB-LR-ceiling acid test'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(entries, f, sort_keys=False)

    print(f'Generated {len(entries)} configs with log at {OUTDIR / "experiment_log.yaml"}')
    print(f'{"idx":>3s} {"run":>18s} {"proto":>13s} {"rules":>5s} {"T":>5s} '
          f'{"ttl":>5s} {"cap":>16s} {"fwd":>5s} {"bwd":>5s} {"rep":>5s}')
    for row in entries:
        print(f'{row["index"]:>3d} {row["run_name"]:>18s} {row["protocol"]:>13s} '
              f'{row["n_rules"]:>5d} {row["admit_temperature"]:>5.1f} '
              f'{row["max_residence_steps"]:>5d} {str(row["cap_min_max_h0"]):>16s} '
              f'{row["fwd_share"]:>5.2f} {row["bwd_share"]:>5.2f} {row["replay_share"]:>5.2f}')
