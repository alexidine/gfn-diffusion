"""
lr_discovery v1 -- first live run of the ramp / cruise / re-ramp LR loop.

WHAT THE LOOP DOES. peak_scale is climbed deliberately (ramp_per_tick every
ramp_period steps) until the quorum trigger fires, then cut once and held. The
climb repeats every cruise_steps, because the operating point genuinely drifts:
alpha* rises over a run as the landscape flattens, so a single calibration goes
stale. Between ramps the same trigger is a pure brake -- it fires if the
landscape sharpens under the fixed rate, which is exactly "this LR just became
too hot". One mechanism, both jobs.

WHY THESE NUMBERS, measured on this route (T=10, mipcas-elj, phase 2) by
sweeping the LR 1000x over 1000 steps and recording the probe against a
held-out loss:

  - alpha* falls MONOTONICALLY with LR across the whole sweep (as lr^-0.67, not
    the 1/lr a fixed-curvature model predicts), and the windowed statistic is
    smooth even though single probes scatter over two decades. So it is
    directly gateable.
  - `alpha_bar 3.0` at `quorum 0.6` fires at ~3x the operating LR, while the
    held-out loss does not leave its plateau until ~7.6x. The trigger leads the
    damage by ~2.5x, so a ramp that stops on the trigger never enters the
    damage zone -- which is why this needs no checkpoint/rewind machinery.
  - alpha* never reaches 1.0 before the run dies of non-finite gradients, so any
    bar at or below 1 never fires. The bar must be > 1.
  - 'downward' fits are NOT a hot-end signal: they appear only at ~12x, after
    the damage, and in normal runs only at the LOW-LR end during warmup. The
    calibration-free detector hoped for there does not exist; the bar is a
    chosen number and that is accepted.

  - quorum 0.6 rather than the median (0.5): a proportion is bounded with
    binomial variance, whereas alpha* has no finite moments. At n=25 with 30%
    of readings genuinely below the bar, 0.6 gives 0.18% false triggers against
    the median's 4.4% -- and under a one-sided cut every false trigger is
    permanent until the next re-ramp.

  - cut_on_trigger 3.0 lands the cruise at roughly the rate this route has been
    running at in practice (trigger ~3x / 3), i.e. the loop should REDISCOVER
    the working LR rather than move it. That is the result to look for.

WHAT TO WATCH (wandb, online):
  lr_ctrl/disc_ramping  1 while climbing, 0 while cruising
  lr_ctrl/trigger_frac  the quorum statistic vs lr_ctrl/trigger_bar
  lr_ctrl/peak_scale    the sawtooth: climb, cut, hold, climb
  lr_fused              same sawtooth in real units; should cruise near 1.25e-4
  fwd/tb_err, Effective Dimension -- must not degrade across the ramps

Run:  python configs/lr_discovery/make.py
      python train.py --config configs/lr_discovery/v1.yaml
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'mk_dev.yaml')
CKPT = 'dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
START, N_STEPS = 5300, 8000     # ~2 full ramp/cruise cycles at cruise_steps 4000


def build():
    with open(BASE) as f:
        cfg = yaml.safe_load(f)

    cfg['run_name'] = 'v1'
    cfg['tag'] = 'lrdisc'                 # distinct identity: its own checkpoints
    cfg['checkpoint_name'] = CKPT
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['epochs'] = START + N_STEPS
    cfg['eval_period'] = 500              # never 250/250 locally
    cfg['figs_period'] = 1000

    al = cfg['adaptive_lr']
    s = al['servo']
    # 5.21e-5, not the 1.25e-4 v1 used. That 1.25e-4 came from lr_aug08, a
    # DIFFERENT route. On elj-mipcas the reference arm (jahe2jvv) cruised at
    # 5.21e-5 and beat the 1.25e-4 restart (emcf4ye6) by ~1.5 nats of
    # fwd/tb_err at every matched window, with the gap not closing:
    #   5300-5600  17.4 vs 18.8     5600-6100  16.1 vs 17.9     6100-6800  15.7 vs 17.2
    # Seeding at the reference rate makes the opening of this run identical to
    # the arm it is measured against, so the ramp is the only difference.
    s['seed_lr'] = 5.2083e-5
    # trigger/discovery sit beside `servo`, not inside it: the trigger both ends
    # a ramp and brakes during cruise, so it belongs to neither leg
    al['trigger'] = {'alpha_bar': 2.0, 'quorum': 0.6}
    al['discovery'] = {
        'enabled': True,
        'ramp_per_tick': 1.25,
        'ramp_period': 50,
        'ramp_cadence': 2,
        'cruise_cadence': 20,
        'cut_on_trigger': 3.0,
        'cruise_steps': 4000,
        # Hold at the seed before the first climb. v1's restart ramped from
        # step 0 and was above its resume rate within 40 steps; this run trains
        # 1000 steps at the reference rate first, which both establishes the
        # damage tripwire's baseline at a STABLE LR and gives a clean
        # like-for-like stretch against jahe2jvv before anything moves.
        'start_in_cruise': True,
        'initial_cruise_steps': 1000,
        # Backstop. The ramp is supposed to end on the sensor; v1 established
        # that it may not, so bound the episode too. 6x the LR the episode
        # started from is just past where v1's damage began (2.5x) and far
        # short of where it detonated (55x).
        'max_ramp_gain': 6.0,
    }
    # The ramp reads a moving target, so the window must be short enough that
    # the LR moves < ~1.5x across it: min_readings x ramp_cadence = 20 steps,
    # over which peak_scale moves 1.25^(20/50) = 1.09x. Comfortable.
    s['min_readings'] = 10

    out = os.path.join(HERE, 'v1.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'wrote {out}')
    print(f"  resume {CKPT} @{START} -> {cfg['epochs']}  ({N_STEPS} steps)")
    print(f"  trigger: >= {al['trigger']['quorum']:.0%} of window below "
          f"alpha* {al['trigger']['alpha_bar']}")
    print(f"  ramp x{al['discovery']['ramp_per_tick']} per {al['discovery']['ramp_period']} steps"
          f" | cut /{al['discovery']['cut_on_trigger']} | cruise "
          f"{al['discovery']['cruise_steps']} steps")
    print(f"  expected: climb {s['seed_lr']:.2e} -> ~{s['seed_lr']*3:.2e}, "
          f"cut to ~{s['seed_lr']*3/al['discovery']['cut_on_trigger']:.2e}, repeat")
    return out


if __name__ == '__main__':
    build()
