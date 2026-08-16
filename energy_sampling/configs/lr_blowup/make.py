"""
lr_blowup -- a deliberately unsafe LR ramp, run to watch the guards catch it.

WHAT THIS IS FOR. lrdisc/v1 (2026-08-10) ramped 55x and detonated with nothing
stopping it: the alpha* quorum trigger could not fire (bar 3.0 sat above probe
span 2.0, and 'beyond' censoring capped the statistic under the quorum at every
LR), and every other guard in controller.py is an EXPLOSION detector with bars
at 1e9. Two guards were added in response -- `discovery.max_ramp_gain` (a
deterministic bound on the ramp episode) and `adaptive_lr.damage` (a relative
tripwire on fwd/tb_err against its own trailing median). This arm exists to
show them firing on a live run rather than only in replay.

DESIGN. Same checkpoint and same route as v1, so the numbers are comparable.
The ramp is faster than production (1.3x per 40 steps rather than 1.25x per 50)
so the run reaches the known damage knee -- ~3.1e-4 on this route, 2.5x the
1.25e-4 it trains at -- in a few hundred steps instead of a thousand.

WHICH GUARD SHOULD FIRE. The damage tripwire, and that is the point: it is the
one that DETECTS something. max_ramp_gain is set to 8.0 rather than the
production 6.0 precisely so it does not pre-empt the sensor -- it is the net
under the sensor, not the thing under test. Expected order:

  ~step 5440   damage baseline has enough history to judge (needs `window`
               populated back to `guard`, ~140 steps at controller cadence)
  ~step 5500   lr passes ~2.6e-4, fwd/tb_err starts climbing off its baseline
  ~step 5560   damage trips: peak_scale /3, ramp -> cruise, ceiling recorded
  after        fwd/tb_err should flatten or recover; lr_ctrl/damage_trips = 1

READ IT WITH: lr_ctrl/damage_ratio (vs the 1.15 bar), lr_ctrl/damage_trips,
lr_ctrl/peak_scale, lr_ctrl/disc_ramping, lr_fused, fwd/tb_err.

WRITES NOTHING. checkpoint_read_only suppresses save/archive/link, so this
cannot touch the shared 573c92 checkpoints it loads from.
"""
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'mk_dev.yaml')
CKPT = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt')
START = 5300
N_STEPS = 800


def build(hard=False):
    with open(BASE) as f:
        cfg = yaml.safe_load(f)

    cfg['run_name'] = 'blowup'
    cfg['tag'] = 'lrblow'
    cfg['checkpoint_name'] = CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    cfg['checkpoint_read_only'] = True      # never write: this run is a probe
    cfg['epochs'] = START + N_STEPS
    cfg['eval_period'] = 500
    cfg['figs_period'] = 1000

    al = cfg['adaptive_lr']
    al['servo']['seed_lr'] = 1.25e-4        # the rate this route trains at

    d = al['discovery']
    d['enabled'] = True
    d['ramp_per_tick'] = 1.3                # faster than production, on purpose
    d['ramp_period'] = 40
    d['max_ramp_gain'] = 8.0                # the NET, set clear of the sensor

    # Production damage settings, unmodified -- the point is to test THESE.
    dmg = al['damage']
    assert dmg['enabled'] and dmg['ratio'] == 1.15 and dmg['window'] == 400, dmg

    if hard:
        # DAMAGE-LED variant. blowup.yaml never blew up: with span fixed the
        # alpha* trigger fires at ~3.7x the operating LR, long before anything
        # degrades, so the damage tripwire and the backstop were never reached
        # and neither was tested. Lowering the bar keeps the ramp and every
        # other mechanism identical while making the quorum unreachable in
        # practice -- the trigger now only fires if the optimal step drops
        # under a THIRD of the one taken, i.e. deep overshoot.
        #
        # alpha_bar: 0 does NOT do this. _advance_servo routes to
        # _trigger_tick only when bar > 0, and the whole discovery ramp lives
        # inside _trigger_tick, so a bar of 0 falls through to the legacy
        # median servo and the LR never ramps at all (verified: peak_scale
        # flat at 1.0 over 1200 steps). A low POSITIVE bar is the way.
        cfg['run_name'] = 'blowup_hard'
        al['trigger']['alpha_bar'] = 0.3
        d['max_ramp_gain'] = 12.0           # net well clear of the damage bar
        cfg['epochs'] = START + 1200

    out = os.path.join(HERE, 'blowup_hard.yaml' if hard else 'blowup.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    seed = al['servo']['seed_lr']
    print(f'wrote {out}')
    print(f"  resume {CKPT} @{START} -> {cfg['epochs']}  ({N_STEPS} steps), read-only")
    print(f"  ramp x{d['ramp_per_tick']} per {d['ramp_period']} steps from {seed:.2e}")
    print(f"  damage: {dmg['metric']} >= {dmg['ratio']}x trailing median "
          f"({dmg['window']}-step window, {dmg['guard']}-step guard), "
          f"patience {dmg['patience']} -> cut /{dmg['cut']}")
    print(f"  backstop: max_ramp_gain {d['max_ramp_gain']} -> hard stop at "
          f"{seed * d['max_ramp_gain']:.2e}")
    print(f"  known knee on this route: ~3.1e-4 (2.5x), v1 detonated at 2.9e-3 (23x)")
    return out


if __name__ == '__main__':
    import sys
    build(hard='--hard' in sys.argv)
