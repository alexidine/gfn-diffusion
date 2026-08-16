"""
lr_aug08 pair CAL -- the servo with a CALIBRATED target, from both sides.

NAMED 'CAL', NOT 'pair D'. `local_aug08` already has a pair D (the freeze x LR
2x2 that measured alpha* ~ 1/lr), it is cited throughout docs/, and two things
with one name in a doc set that already warns about ID collisions is how a
reading goes wrong six months from now.

THE CLAIM THIS TESTS. Pair A refuted `target: 1.0`: the loop converged perfectly
(alpha_median 1.006, peak_scale 32.2 against a 200 bound, servo_hold 0) and
landed at 3.2e-4, which cost 2.19 nats of bwd/tb_err and 2.89 of fwd/tb_err
against a hand-set 1.25e-4. The natural reading of that was "alpha* is useless as
a controller." Pair C says otherwise, and the reason is that the two branch
metrics disagree about what "better" means:

  arm       lr        bwd/tb_err   fwd/tb_err
  c_low     1.56e-5      13.5         20.8      <- bwd best, fwd BAD
  a_fixed   1.25e-4      15.04        17.91     <- fwd best
  a_climb   3.2e-4       17.23        20.80     <- both bad

`bwd/tb_err` is monotone in LR -- lower is always better, which is close to
saying "a policy that barely moves fits the fixed buffer well" and is not by
itself a result. `fwd/tb_err` is **U-shaped, with its minimum AT the hand-set
1.25e-4**, and fwd is the on-policy branch: it is the one that says whether the
samples are good, not whether the buffer is being fitted.

So the LR optimum is 1.25e-4 on this route, and the question becomes: **what does
alpha* read there?** `a_fixed` answers it directly -- ~2.0 in steady state. Which
makes `target` a CALIBRATABLE quantity rather than a physical constant:

    target := alpha* measured at an LR you are happy with

TARGET below is computed from a_fixed's own second half, so this generator IS
the calibration procedure rather than a description of one.

THE TEST. Two arms, same calibrated target, seeded on opposite sides of it:

  d_cal_below   seed 1.0e-5   (8x below the expected landing point)
  d_cal_above   seed 4.0e-4   (3.2x above)

**Both landing at ~1.25e-4, and reproducing a_fixed's fwd/tb_err ~17.9, is the
servo validated.** Convergence from both sides rules out drift, bound-parking and
seed-dependence in one shot -- none of which a single arm stopping somewhere
plausible could rule out.

WHAT WOULD FALSIFY IT
  * they land in different places      -> the loop integrates noise; ship the
                                          one-sided brake, clip [0.8, 1.0]
  * they land together but off 1.25e-4 -> alpha* is not a function of LR alone;
                                          the policy state it also depends on is
                                          uncontrolled, and calibration will not
                                          transfer across runs either
  * they land right but score worse    -> getting to an LR by servo is not the
                                          same as running at it, i.e. the PATH
                                          matters and a servo cannot be judged
                                          by its endpoint

⚠ b_descend is the reason `d_cal_above` is worth running rather than assumed.
It was launched BEFORE the F7 window-flush fix and overshot 4.0e-4 -> 5.36e-4
before descending, because the probe kept buffering through warmup. These arms
carry the fix, so `d_cal_above` is also the check that F7 actually cured it: a
clean descent with no initial excursion is the pass.
"""
import copy
import glob
import json
import os
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'lrd0808'

P2_STEPS = 2650
P2_CKPT = 'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
CKPT_STAGE = 'naive'
REF_LR = 1.25e-4
PAIR_BATCH = 1000
TARGET_FALLBACK = 2.0
ARMS = []


def measure_target():
    """alpha* at the LR pair C identifies as the fwd/tb_err optimum.

    Second half of a_fixed only: the first bins carry the resume transient and
    the probe climbing into `ok` fits (fit_ok_rate runs 0.80 -> 0.98 over that
    arm), so an early median is taken over a different population of fits.
    """
    try:
        from read import run_name_of, scan, series
    except Exception:
        return TARGET_FALLBACK, 'fallback (read.py unavailable)'
    root = HERE.parent.parent
    for d in sorted(glob.glob(os.path.join(root, 'wandb', 'offline-run-*')),
                    key=os.path.getmtime, reverse=True):
        if run_name_of(d) != 'a_fixed':
            continue
        hist, _ = scan(d)
        v = series(hist, 'lrprobe/alpha_median')
        if len(v) < 100:
            continue
        half = v[len(v) // 2:]
        return round(statistics.median(half), 2), f'a_fixed second half, {len(half)} evals'
    return TARGET_FALLBACK, 'fallback (no completed a_fixed found)'


TARGET, TARGET_SRC = measure_target()


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def stage2(cfg):
    return cfg['protocol']['stages'][1]


def arm_cfg(seed_lr, budget):
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
    rb = cfg['buffers']['replay_buffer']
    rb.pop('prioritise', None)
    rb['max_size'] = 4000
    stage2(cfg).pop('buffer_servo', None)
    stage2(cfg)['balance']['bounds'] = {'replay': [0.05, 0.45]}
    stage2(cfg)['min_fracs'] = {'fwd': 0.02, 'bwd': 0.02, 'replay': 0.05}
    stage2(cfg)['balance']['pinned'] = {'fwd': 0.2}
    stage2(cfg)['fracs'] = {'fwd': 0.2, 'bwd': 0.45, 'replay': 0.35}
    cfg['lr_fused'] = 'auto'
    for k in ('lr_policy', 'lr_back', 'lr_replay'):
        cfg[k] = REF_LR
    s = cfg['adaptive_lr']['servo']
    s['enabled'] = True
    s['seed_lr'] = seed_lr
    s['target'] = TARGET
    # SET EXPLICITLY, do not inherit. mk_dev shipped the one-sided brake
    # (clip [0.8, 1.0]) on 2026-08-08 in response to F8, and these arms are the
    # experiment that asks whether the GROWTH servo is salvageable with a
    # calibrated target -- which needs a two-sided clip. Inheriting would have
    # silently turned both arms into brakes, so d_cal_below could never climb
    # off its 1e-5 seed and the pair would have measured nothing.
    s['clip'] = [0.8, 1.25]
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['epochs'] = P2_STEPS + budget
    return cfg


def main():
    for name, seed_lr, ask in (
        ('d_cal_below', 1.0e-5,
         f'servo at the CALIBRATED target {TARGET}, climbing from 8x below. Landing at '
         f'~1.25e-4 with fwd/tb_err ~17.9 reproduces a_fixed without anyone having set '
         f'an LR.'),
        ('d_cal_above', 4.0e-4,
         f'same target, descending from 3.2x above. Landing where d_cal_below lands is '
         f'the validation. Also the check that F7 (window flush) cured b_descends 34% '
         f'wrong-direction overshoot -- a clean descent with no initial excursion.'),
    ):
        cfg = arm_cfg(seed_lr, 5400)
        cfg['run_name'] = name
        cfg['tag'] = TAG
        ARMS.append((name, cfg, ' '.join(ask.split())))


def emit():
    main()
    seen = {}
    print(f'calibrated servo target = {TARGET}  ({TARGET_SRC})')
    for i, (name, cfg, ask) in enumerate(ARMS):
        (HERE / f'd{i}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        c = copy.deepcopy(cfg)
        c.pop('run_name'), c.pop('epochs')
        k = json.dumps(c, sort_keys=True, default=str)
        if k in seen:
            raise SystemExit(f'DUPLICATE: {seen[k]} and {name}')
        seen[k] = name
        print(f'd{i}.yaml  {name:12s} seed_lr={cfg["adaptive_lr"]["servo"]["seed_lr"]:.2g}  {ask[:58]}')


if __name__ == '__main__':
    emit()
