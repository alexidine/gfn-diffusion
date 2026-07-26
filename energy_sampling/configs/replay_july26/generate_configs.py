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
   2 : t1             -- admit_temperature 1
   3 : t5             -- admit_temperature 5 (tsched-control bridge)
   4 : share005       -- buildout default_boost replay 0.05
   5 : share02        -- 0.2
   6 : share03        -- 0.3
   7 : ttl150         -- max_residence_steps 150
   8 : ttl400         -- max_residence_steps 400
   9 : norep          -- replay share 0, bwd left dormant (0.001)
  10 : bwd01_norep    -- replay 0, bwd pinned at 0.1
  11 : bwd03_norep    -- replay 0, bwd pinned at 0.3
  12 : t1_share02     -- T 1 + share 0.2 compound
  13 : fixedcap       -- admit_cap_min = admit_cap_max = 30 (cap(h) off)
  14 : churn25        -- churn_rate 25 (thin buffer ~6.2k)
  15 : churn100       -- churn_rate 100 (fast displacement, age ~50)

READING GUIDE: outcome = step-matched fwd/tb_err_worst (floor LEVEL and
arrival step), fwd/r2, fwd/logw_std stability (sawtooth = lost anchor).
replay_buffer_expired_delta SIGN is the anchor-health readout (negative =
resident rows improving; positive on every diverging tsched arm).
absorbed_frac stays near zero by design (calibration mandate: rows cycle).
Watch replay_buffer_admit_cap/admit_health around the ~15k transition
kick -- if cap never leaves its healthy end, arm 13 is uninformative on
the health axis. The best tsched arms plateaued at ~7 nats independent of
terminal noise -- if ANY arm here goes well below 7 (t1/share03 are the
candidates), the plateau was anchor-limited, NOT the target's own
within-condition spread, and the buildout exit bar (5.0) may finally be
reachable.
"""

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
    boost = buildout_stage(config)['balance']['default_boost']
    return {'index': ind, 'run_name': name, 'note': note,
            'admit_temperature': rb['admit_temperature'],
            'max_residence_steps': rb['max_residence_steps'],
            'cap_min_max_h0': [rb['admit_cap_min'], rb['admit_cap_max'],
                               rb['admit_cap_health_h0']],
            'churn_rate': rb['churn_rate'],
            'buildout_replay_share': boost.get('replay', 0.0),
            'buildout_bwd_share': boost.get('bwd', 0.0)}


if __name__ == '__main__':
    entries = []

    entries.append(emit(0, make_base(), 'control_t2',
                    'reference: T 2, TTL 250, share 0.1 -- fresh replication of sw2v8yas'))

    entries.append(emit(1, replay_arm(admit_temperature=0.5), 't05',
                    'admit_temperature 0.5: near-argmax admission'))
    entries.append(emit(2, replay_arm(admit_temperature=1.0), 't1',
                    'admit_temperature 1'))
    entries.append(emit(3, replay_arm(admit_temperature=5.0), 't5',
                    'admit_temperature 5: cross-battery bridge to the tsched control'))

    entries.append(emit(4, share_arm(0.05), 'share005',
                    'buildout replay share 0.1 -> 0.05'))
    entries.append(emit(5, share_arm(0.2), 'share02',
                    'buildout replay share 0.1 -> 0.2'))
    entries.append(emit(6, share_arm(0.3), 'share03',
                    'buildout replay share 0.1 -> 0.3'))

    entries.append(emit(7, replay_arm(max_residence_steps=150), 'ttl150',
                    'TTL 250 -> 150'))
    entries.append(emit(8, replay_arm(max_residence_steps=400), 'ttl400',
                    'TTL 250 -> 400'))

    entries.append(emit(9, falsifier_arm(0.0), 'norep',
                    'replay excised, bwd dormant: designed replication of fwd-alone instability'))
    entries.append(emit(10, falsifier_arm(0.1), 'bwd01_norep',
                    'replay excised, bwd pinned 0.1: can bwd own spread instead?'))
    entries.append(emit(11, falsifier_arm(0.3), 'bwd03_norep',
                    'replay excised, bwd pinned 0.3'))

    entries.append(emit(12, share_arm(0.2, replay_arm(admit_temperature=1.0)), 't1_share02',
                    'compound best-guess: T 1 + share 0.2'))
    entries.append(emit(13, replay_arm(admit_cap_min=30.0), 'fixedcap',
                    'cap_min = cap_max = 30: health modulation off, at the T=2 reference'))
    entries.append(emit(14, replay_arm(churn_rate=25), 'churn25',
                    'churn_rate 25: thin equilibrium buffer ~6.2k'))
    entries.append(emit(15, replay_arm(churn_rate=100), 'churn100',
                    'churn_rate 100: fast displacement churn, mean age ~50'))

    with open(OUTDIR / 'experiment_log.yaml', 'w') as f:
        yaml.dump(entries, f, sort_keys=False)

    print(f'Generated {len(entries)} configs with log at {OUTDIR / "experiment_log.yaml"}')
    print(f'{"idx":>3s} {"run":>14s} {"T":>5s} {"ttl":>5s} {"cap":>14s} '
          f'{"churn":>5s} {"share":>6s} {"bwd":>5s}')
    for row in entries:
        print(f'{row["index"]:>3d} {row["run_name"]:>14s} {row["admit_temperature"]:>5.1f} '
              f'{row["max_residence_steps"]:>5d} {str(row["cap_min_max_h0"]):>14s} '
              f'{row["churn_rate"]:>5d} {row["buildout_replay_share"]:>6.2f} '
              f'{row["buildout_bwd_share"]:>5.2f}')
