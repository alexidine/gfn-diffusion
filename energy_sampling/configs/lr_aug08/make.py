"""
lr_aug08 -- does the alpha* LR servo work? T=10 laptop battery, 2026-08-08.

WHY THIS EXISTS. controller.py v7 replaced the hand-set peak + cut/latch/recovery
middle layer with a servo on alpha*: peak <- peak * clip(median alpha* / target).
Two halves of that are validated and one is not.

  VALIDATED (local_aug08 pair D, and F0 across 16 batt0807 runs):
    * the sensor is not precision-limited -- second_diff_rel median 3.6e-2 vs a
      1e-6 floor, 0.28% of probes below it
    * the scaling law holds -- raising lr 1.72x divided alpha_median by 1.73
  NOT VALIDATED:
    * that FOLLOWING alpha* lands anywhere good. The only direct test says it
      does not: the probe read ~1.7 at base LR and taking that step cost 0.6-0.9
      nats of bwd/tb_err in BOTH rows of the freeze x LR 2x2.

That last point is the whole risk of v7. A loop with a correct sensor and a wrong
setpoint tracks confidently to the wrong place, which is a failure mode the old
hand-tuned peak did not have. This battery measures it directly, at 5400 steps
rather than the 800 the original reading came from.

FOUND BY THE SMOKE ARM, BEFORE ANY OF THIS RAN: at lr ~1.4e-6 the probe returned
'downward' on 100% of fits, so alpha* was always nan and the servo could never
actuate at all. The step was too short to bracket the basin -- concave, but
DESCENDING (loss_delta_rel < 0), which is "your step is too small" and not "the
model is wrong". step_probe.py now splits those into 'beyond' and 'downward'.
Without that fix every servo arm below would have sat at its seed and the
battery would have measured nothing.

THE DESIGN. Four arms, two pairs, all resuming from ONE post-transient
checkpoint, so every arm starts at bitwise the same state and the only thing that
differs is what owns the learning rate.

  a_fixed   lr_fused pinned at 1.25e-4, servo OFF        <- the reference
  a_climb   servo ON, seeded at 1e-5 (12.5x BELOW ref)   <- can it find the band?
  b_climbB  a_climb at another seed                      <- does it land twice?
  b_descend servo ON, seeded at 4.0e-4 (3.2x ABOVE ref)  <- does it cut back?

`a_fixed` is configured exactly as local_aug08's `a_frz`, off the same resume at
the same length -- so its final-window bwd/tb_err should reproduce 15.14, and if
it does not, something in the v7 rewrite changed training rather than just the
LR path, and the whole battery is void. Check that first.

WHAT EACH ARM CAN SHOW, stated before the run:

  a_climb reaching ~1.25e-4 and PARKING there is the strong result: the servo
  found the hand-tuned optimum with no prior knowledge, from 12.5x below.
  Reaching it and continuing past it is F5's failure -- the growth half is
  wrong and `clip: [lo, 1.0]` (one-sided brake) is the posture to ship.
  Never reaching it says the climb rate (clip 1.25 / period 200) is too slow to
  pay for itself inside a run.

  b_descend is the half F5's measurement supports, so it is the CONTROL on the
  servo's own machinery: if the descent does not work either, the problem is
  the implementation, not the setpoint.

  b_climbB vs a_climb is the only reproducibility read available. The converged
  LR is the quantity that has to be stable -- alpha_median's per-probe spread is
  known to be wide (IQR 0.5-1.0) and that is FINE if the median it feeds is
  stable, which is exactly what a seed replicate tests.

BUFFERS ARE HELD AT THE OLD CONFIGURATION ON PURPOSE. mk_dev now ships
prioritise.enabled + buffer_servo (the B7b package). Both are live design
changes with a measured cost on fwd/tb_err (B0b: +4.4 nats), so leaving them on
would confound every number here against every prior baseline. This battery
varies the LR and nothing else. mk_dev's own configuration is smoke-tested
separately (arm `smoke`).

SIZING (measured 2026-08-07/08). Two arms share the GPU at batch 1000 /
cuda_memory_fraction 0.45; paired throughput ~1.0 step/s at T=10, so budget ~=
seconds: 5400 steps ~ 1.5 h per pair, 3 h for the battery.

5400 rather than local_aug08's 3600 because the climb arms have to do two
things in one run. From 1e-5 to the 1.25e-4 reference is 12.5x, and the servo's
per-tick ceiling is clip_hi 1.25 every period 200 steps, so the fastest possible
climb is ln(12.5)/ln(1.25) = 11.3 ticks = ~2270 steps. At 3600 that leaves ~1100
steps to settle in and be read, which is not enough to separate "converged
here" from "still moving". a_fixed runs 5400 too so every arm is read at matched
steps; its 3600-step bin is still directly comparable to local_aug08's 15.14.

max_batch_size is pinned EQUAL to batch_size -- an OOM cut in one arm and not
the other gives the two different lambda*tau and voids the pair.

Every arm writes lr0808_* prefixes and runs checkpoint_read_only, so this is
read-only with respect to every checkpoint on disk.
"""
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'lr0808'

# Phase-2 resume, POST log-Z transient (local_aug07: log_Z_learned settles by
# ~25% of phase 2 and tb_resid_clipped is inside D29's +-0.5 after it). Anything
# read before this step is reading the transient.
P2_STEPS = 2650
P2_CKPT = 'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'

# The stage this checkpoint was trained under. mk_dev renamed it to
# 'equilibration' on 2026-08-08 and StageProtocol.stage resolves BY NAME and
# raises on a miss, so a resume needs the old name back. Renaming here rather
# than editing the checkpoint keeps the checkpoint untouched.
CKPT_STAGE = 'naive'

# local_aug08's base LR at T=10 -- what `a_frz` ran and what pair D measured the
# optimum to be at or below (2.15e-4 was measurably WORSE on both rows).
REF_LR = 1.25e-4

PAIR_BATCH = 1000
ARMS = []


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def stage2(cfg):
    return cfg['protocol']['stages'][1]


def local(cfg):
    cfg['eval_period'] = 250
    cfg['figs_period'] = 1000        # must be a multiple of eval_period
    cfg['archive_period'] = 0        # throwaway arms
    cfg['checkpoint_read_only'] = True
    stage2(cfg)['name'] = CKPT_STAGE
    # WARMUP IS SHORTENED ON A RESUME, and the reason is not budget. The
    # warmup exists so rebuilt optimizers do not land at the peak on COLD Adam
    # moments; these arms resume with load_optimizers=True, so the moments are
    # warm and that rationale does not apply. At mk_dev's 1000 the servo (which
    # holds through warmup, since the envelope is below 1 there and alpha* would
    # read the shrunken step) would sit idle for 1000 of 5400 steps.
    cfg['adaptive_lr']['warmup_steps'] = 200
    return cfg


def paired(cfg):
    cfg['batch_size'] = PAIR_BATCH
    cfg['max_batch_size'] = PAIR_BATCH
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    return cfg


def old_buffers(cfg):
    """Revert mk_dev's 2026-08-08 buffer changes, so this battery is comparable
    to local_aug08 and reads only the LR axis. See the module docstring."""
    rb = cfg['buffers']['replay_buffer']
    rb.pop('prioritise', None)
    rb['max_size'] = 4000
    stage2(cfg).pop('buffer_servo', None)
    # mk_dev widened replay's ceiling to 0.60 and raised bwd's floor to 0.25
    # alongside the prioritised draw; both belong to that change, not this one.
    stage2(cfg)['balance']['bounds'] = {'replay': [0.05, 0.45]}
    stage2(cfg)['min_fracs'] = {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.05}
    stage2(cfg)['balance']['pinned'] = {'fwd': 0.2}
    stage2(cfg)['fracs'] = {'fwd': 0.2, 'bwd': 0.45, 'replay': 0.35}
    return cfg


def resume(cfg, budget):
    """Pin the resume point EXPLICITLY.

    mk_dev defaults to continue_from_checkpoint: true + checkpoint_name: null,
    which resolves to {tag}_{run_name}_{problem}_running.pt. run_name is unique
    per arm so that file never exists -- a generator that forgets this does not
    chain arms, it silently RETRAINS PHASE 1 in every one, which is invisible in
    the results and costs the whole day.

    `epochs` is an ABSOLUTE ceiling on a resumed run (the loop is
    trange(init_step, epochs+1)), not a budget -- so it must be P2_STEPS + N.
    """
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False   # checkpoint_name takes precedence
    cfg['epochs'] = P2_STEPS + budget
    assert cfg['checkpoint_name'] == P2_CKPT, 'resume point not pinned'
    assert cfg['continue_from_checkpoint'] is False, 'would resolve to _running.pt'
    assert budget > 0, 'zero budget runs no steps and verifies nothing'
    return cfg


def servo(cfg, seed_lr, target=1.0, clip=(0.8, 1.25)):
    """Hand the fused LR to the alpha* loop, seeded at `seed_lr`.

    Only lr_fused is set to `auto`: this is a fused stage, so lr_policy /
    lr_back / lr_replay are not read at all ([[fused-mode-dead-lr-knobs]]), and
    writing `auto` on them would put keys in lr_servo_managed that nothing
    consumes -- harmless, but it would make the log claim the servo is driving
    four groups when it drives one.
    """
    cfg['lr_fused'] = 'auto'
    for k in ('lr_policy', 'lr_back', 'lr_replay'):
        cfg[k] = REF_LR                       # explicit floats: fixed, unread here
    s = cfg['adaptive_lr']['servo']
    s['enabled'] = True
    s['seed_lr'] = seed_lr
    s['target'] = target
    s['clip'] = list(clip)
    return cfg


def fixed(cfg, lr):
    """No `auto` key anywhere -> lr_servo_managed is empty -> the servo reads
    and logs alpha* but actuates nothing. The probe still runs, so this arm also
    supplies the alpha* trace at a KNOWN-GOOD fixed LR, which is the reference
    the servo arms' readings are compared against."""
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = lr
    cfg['adaptive_lr']['servo']['enabled'] = False
    return cfg


def arm(name, pair, budget, cfg, asks):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    ARMS.append((name, pair, budget, cfg, ' '.join(asks.split())))


# ===========================================================================
def main():
    # ---- SMOKE: mk_dev exactly as shipped, 150 steps. Not an experiment. ---
    # Everything new at once: servo + probe + prioritised replay + buffer_servo
    # + the new bounds. Its only job is to prove the configuration RUNS before
    # two hours of GPU go into arms that share most of that code.
    c = paired(local(base()))
    arm('smoke', 'S', 150, resume(c, 150),
        'does mk_dev as shipped survive 150 steps? Checks the servo tick, the '
        'prioritised draw, the buffer servo and the new frac bounds together.')

    # ---- PAIR A: the reference, and the climb. ---------------------------
    c = fixed(old_buffers(paired(local(base()))), REF_LR)
    arm('a_fixed', 'A', 5400, resume(c, 5400),
        'local_aug08 a_frz re-run under v7. bwd/tb_err final window MUST come back near '
        '15.14 -- if it does not, the v7 rewrite changed training and not just the LR '
        'path, and every other arm here is void. Also gives the alpha* trace at a '
        'known-good fixed LR, which is what the servo arms are read against.')

    c = servo(old_buffers(paired(local(base()))), seed_lr=1.0e-5)
    arm('a_climb', 'A', 5400, resume(c, 5400),
        'servo from 12.5x BELOW the reference LR. Does it reach ~1.25e-4, and how long '
        'does the climb take? Parking at the reference is the strong result; sailing '
        'past it is F5 (the growth half is wrong, ship clip [lo, 1.0]); never arriving '
        'says clip 1.25 / period 200 is too slow to pay for itself in a run.')

    # ---- PAIR B: reproducibility, and the descent. ------------------------
    c = servo(old_buffers(paired(local(base()))), seed_lr=1.0e-5)
    c['seed'] = 20260808
    arm('b_climbB', 'B', 5400, resume(c, 5400),
        'seed replicate of a_climb. The CONVERGED LR is the quantity that has to be '
        'stable -- per-probe alpha* spread is known to be wide and that is fine if the '
        'median is stable. Two different landing points would mean the loop is '
        'integrating noise.')

    c = servo(old_buffers(paired(local(base()))), seed_lr=4.0e-4)
    arm('b_descend', 'B', 5400, resume(c, 5400),
        'servo from 3.2x ABOVE the reference. The DOWN direction is the half pair D '
        'supports, so this is the control on the servo machinery itself: if the descent '
        'does not work either, the fault is the implementation and not the setpoint.')


def emit():
    main()
    rows = []
    for i, (name, pair, budget, cfg, asks) in enumerate(ARMS):
        # a duplicate arm is a wasted GPU-hour that looks like a result -- see
        # the rb0808 post-mortem in decisions.md D30
        (HERE / f'{i}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        rows.append((i, name, pair, budget, asks))
    keys = []
    for i, (name, _, _, cfg, _) in enumerate(ARMS):
        c = copy.deepcopy(cfg)
        c.pop('run_name'), c.pop('epochs')
        keys.append((json.dumps(c, sort_keys=True, default=str), name))
    seen = {}
    for k, name in keys:
        if k in seen:
            raise SystemExit(f'DUPLICATE ARMS: {seen[k]} and {name} differ only in '
                             f'run_name/epochs. Fix the generator.')
        seen[k] = name
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8') as f:
        f.write('idx\tarm\tpair\tsteps\tasks\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    for r in rows:
        print(f'{r[0]:>3}  {r[1]:<10} pair {r[2]}  {r[3]:>5} steps  {r[4][:80]}')


if __name__ == '__main__':
    emit()
