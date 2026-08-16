"""
lr_aug08 pair C -- REPOINTED 2026-08-08 after a_climb read out.

WHAT a_climb SETTLED, and why pair C's original question is dead. It was built
to ask "can the servo find a fixed point, and does calibrating `target` put that
fixed point somewhere good?" The first half answered itself:

  * the servo climbed 1.25e-5 -> 3.15e-4 and PARKED there
  * `alpha_median` at the landing point is 0.998 against a target of 1.0
  * `lr x alpha*` is constant at 3.1e-4 +-12% across the whole 8.6x sweep

So the loop is not the problem. It converges from below, holds its setpoint to
0.2%, and its sensor obeys the 1/lr law it was designed around. **The problem is
that it lands 20x above the best LR the same sweep measured**, and following it
there costs 4.51 nats of `bwd/tb_err` (13.50 at 1.56e-5 -> 18.01 at 3.28e-4,
rising monotonically at every one of 20 rungs in between).

WHICH LEAVES THE ACTUAL QUESTION UNANSWERED: **where is the optimum?** The sweep
is monotonic all the way down to its lowest rung, so it never bracketed a
minimum -- 1.56e-5 is simply the smallest LR that was visited, not a measured
best. And every rung got only ~200 steps at a still-moving LR, so the sweep
identifies a direction, not a value.

PAIR C IS THEREFORE A FIXED-LR LADDER BELOW THE REFERENCE, run properly:
constant LR, servo off, full budget, same resume as every other arm.

  c_low      1.56e-5   the sweep's lowest rung, held for 5400 steps
  c_verylow  5.0e-6    3x lower again -- does it keep improving, or turn over?

Read against `a_fixed` (1.25e-4, already run). Three outcomes:

  c_low < a_fixed and c_verylow < c_low   -> optimum is below 5e-6; the whole
                                             route has been running ~25x too hot
  c_low < a_fixed and c_verylow > c_low   -> optimum bracketed near 1.5e-5, and
                                             alpha* there is ~18, so `target`
                                             is calibratable at ~18 (route-specific)
  c_low >= a_fixed                        -> the sweep's monotonic rise was
                                             ELAPSED TIME, not LR, and a_climb's
                                             result is void. b_descend is the
                                             independent check on this.

⚠ THE CONFOUND THIS PAIR DOES NOT FIX. In `a_climb`, LR and elapsed time rise
together, so the sweep alone cannot separate them. Two things already argue
against a pure time effect -- `a_fixed` at constant LR is flat over the same
window, and the sweep's interpolation at 1.25e-4 (~15.5-15.8) independently
reproduces `a_fixed`'s 15.89 -- but the decisive arm is `b_descend`, where LR
falls while time rises. **Read b_descend before trusting anything here.**
"""
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'lrc0808'

P2_STEPS = 2650
P2_CKPT = 'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
CKPT_STAGE = 'naive'
PAIR_BATCH = 1000
ARMS = []


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def stage2(cfg):
    return cfg['protocol']['stages'][1]


def arm_cfg(lr, budget):
    cfg = base()
    cfg['eval_period'] = 250
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['checkpoint_read_only'] = True
    stage2(cfg)['name'] = CKPT_STAGE
    cfg['adaptive_lr']['warmup_steps'] = 200
    cfg['batch_size'] = PAIR_BATCH
    cfg['max_batch_size'] = PAIR_BATCH
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    # same buffer revert as the rest of lr_aug08 -- vary the LR axis only
    rb = cfg['buffers']['replay_buffer']
    rb.pop('prioritise', None)
    rb['max_size'] = 4000
    stage2(cfg).pop('buffer_servo', None)
    stage2(cfg)['balance']['bounds'] = {'replay': [0.05, 0.45]}
    stage2(cfg)['min_fracs'] = {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.05}
    stage2(cfg)['balance']['pinned'] = {'fwd': 0.2}
    stage2(cfg)['fracs'] = {'fwd': 0.2, 'bwd': 0.45, 'replay': 0.35}
    # FIXED: no `auto` anywhere, so lr_servo_managed is empty and the servo
    # reads and logs alpha* while actuating nothing. That is deliberate -- these
    # arms are also the alpha*(lr) calibration points, held long enough that the
    # reading is not contaminated by a moving LR the way a_climb's rungs were.
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = lr
    cfg['adaptive_lr']['servo']['enabled'] = False
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['epochs'] = P2_STEPS + budget
    return cfg


def main():
    for name, lr, ask in (
        ('c_low', 1.563e-5,
         'the sweep lowest rung, held for a full run. Beats a_fixed (15.89 at 1.25e-4) '
         'or the sweep was elapsed time. Also gives a CLEAN alpha* reading at this LR -- '
         'a_climb only passed through it in ~200 steps at a moving LR.'),
        ('c_verylow', 5.0e-6,
         '3x below the sweep lowest rung. Turning over here brackets the optimum near '
         '1.5e-5; still improving means the route has been running >25x too hot and the '
         'ladder needs another rung down.'),
    ):
        cfg = arm_cfg(lr, 5400)
        cfg['run_name'] = name
        cfg['tag'] = TAG
        ARMS.append((name, cfg, ' '.join(ask.split())))


def emit():
    main()
    seen = {}
    for i, (name, cfg, ask) in enumerate(ARMS):
        (HERE / f'c{i}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        c = copy.deepcopy(cfg)
        c.pop('run_name'), c.pop('epochs')
        k = json.dumps(c, sort_keys=True, default=str)
        if k in seen:
            raise SystemExit(f'DUPLICATE: {seen[k]} and {name}')
        seen[k] = name
        print(f'c{i}.yaml  {name:11s} lr={cfg["lr_fused"]:.3g}  {ask[:62]}')


if __name__ == '__main__':
    emit()
