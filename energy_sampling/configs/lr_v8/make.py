"""
lr_v8 -- first runs of the periodic ray-calibration LR controller.

  python configs/lr_v8/make.py            # writes all three
  python train.py --config configs/lr_v8/<arm>.yaml

ARMS
  smoke    400 steps, period 100, read-only. Does it fire, restore, and cost
           what we think? Not a result, a wiring check.
  explode  Deliberately seeded ~12x hot, at the PRODUCTION period. Asks the
           question honestly: can a periodic controller pull back a rate that is
           already damaging, or is the between-calibration exposure fatal?
  ramp     The headline. Seeded ~6x LOW so the controller has to find the
           operating rate from below, then hold it. What we want to see is a
           monotone climb that decelerates into alpha_target and then a long
           stable stretch with fwd/tb_err flat or falling.

All three resume the same checkpoint at step 5300 (equilibration, fused steps),
so the calibration path under test is the one production uses. `explode` and
`smoke` are checkpoint_read_only; `ramp` writes under its own tag.
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'mk_dev.yaml')
CKPT = 'dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
START = 5300

# The rate this route was last observed training stably at (tuphwfkm, alpha* 3.6-5.0).
HEALTHY = 1.59e-4


def _ray_off(cfg):
    """Withdraw every `lr_sensor: {kind: ray}` declaration -- the only way there
    now is to turn the probe off.

    `ray_calibration.enabled: False` used to do this and is retired
    (utils._RETIRED_KEYS): a flag and a stage declaration are two mechanisms for
    one decision and could disagree. train.py arms the probe on
    `bool(self._ray_askers())`, so absence of the declaration IS the off switch.
    A MAPPING, not the bare string -- Stage._parse_lr_sensor raises TypeError on
    a str."""
    off = []
    for st in cfg['protocols'][cfg['protocol']]['stages']:
        if (st.get('lr_sensor') or {}).get('kind') == 'ray':
            st['lr_sensor'] = {'kind': 'none'}
            off.append(st['name'])
    return off


def _base(run_name, steps, read_only):
    with open(BASE) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = run_name
    cfg['tag'] = 'lrv8'
    cfg['checkpoint_name'] = CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['load_weights_only'] = False
    cfg['checkpoint_read_only'] = bool(read_only)
    cfg['epochs'] = START + steps
    cfg['eval_period'] = 500
    cfg['figs_period'] = 1000
    return cfg


def build():
    out = []

    # ---- smoke -------------------------------------------------------------
    cfg = _base('smoke', 400, read_only=True)
    cfg['adaptive_lr']['seed_lr'] = HEALTHY
    cfg['adaptive_lr']['warmup_steps'] = 1     # no envelope: we want the sensor, not the ramp
    cfg['adaptive_lr']['ray_calibration']['period'] = 100
    out.append(('smoke', cfg))

    # ---- explode -----------------------------------------------------------
    # 12x the healthy rate. v1 detonated at ~2.9e-3 having crossed 1.5e-3 on the
    # way, so this starts inside the regime that killed it. eta_down 0.5 means a
    # resolved 'too hot' reading cuts by up to 0.354 per calibration, so the
    # recovery path is ~3 calibrations IF the readings resolve.
    # 4x, NOT 12x. 12x was tried first (run mvwsu5d5) and is not a recovery test:
    # fwd/tb_err went 18.5 -> 2.2e6 in THIRTY steps, gradients went non-finite,
    # and the calibrator correctly refused to read (raycal/skipped=1) because
    # there was no optimizer step left to measure. The exposure window, not the
    # controller, is the binding constraint there. 4x is the realistic drifted-hot
    # case: above the ~2.5x damage threshold measured on this route, but slow
    # enough that a calibration lands while there is still a run to save.
    cfg = _base('explode', 2000, read_only=True)
    cfg['adaptive_lr']['seed_lr'] = 4 * HEALTHY
    cfg['adaptive_lr']['warmup_steps'] = 1     # start hot immediately -- that is the test
    # period 200, NOT the production 500. v1 went from damage onset to non-finite
    # gradients in ~300 steps at this rate, so at period 500 the run would very
    # likely die before its first calibration -- which would measure the exposure
    # window, not the recovery. Shortened deliberately so the question under test
    # is "can it pull back", and the exposure question is recorded separately.
    cfg['adaptive_lr']['ray_calibration']['period'] = 200
    out.append(('explode', cfg))

    # ---- ramp --------------------------------------------------------------
    # Seeded 6x low. Convergence is geometric with ratio (1 - eta_up) = 0.75 per
    # calibration, so ln(6)/-ln(0.75) ~ 6 calibrations ~ 3000 steps to arrive,
    # then it should sit. Long tail on purpose: the claim under test is not
    # "it arrives" but "it arrives and STAYS".
    cfg = _base('ramp', 12000, read_only=False)
    cfg['adaptive_lr']['seed_lr'] = HEALTHY / 6.0
    out.append(('ramp', cfg))

    # ---- ablate --------------------------------------------------------------
    # CONTROL ARM for the replay-noise question. Matches reference run jybhxgxl
    # in every respect that could plausibly matter -- same resume point
    # (_phase1_exit.pt, i.e. the START of equilibration, not mid-stage), same
    # fixed LR 1.25e-4, same eval/figs cadence -- and differs from it ONLY in
    # carrying the current working tree.
    #
    # The LRs are explicit floats, not `auto`, so nothing actuates: with the ray
    # declaration withdrawn below, `auto` would be refused at load (and rightly
    # -- the sensor check is per stage, so equilibration would be left with
    # nothing to move its rate and would pin the LR at the seed while claiming
    # to be adaptive).
    #
    # READ: jitter of replay/tb_err = median|dx|/median|x|. jybhxgxl 0.0018,
    # the v8 ramp 0.0087. If this arm lands near 0.0018 the cause is NOT in the
    # working tree; if near 0.0087 it is, and the diff is where to look.
    cfg = _base('ablate', 3000, read_only=True)
    cfg['checkpoint_name'] = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset'
                              '-T2.5-573c92_phase1_exit.pt')
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 1.25e-4
    _ray_off(cfg)
    cfg['eval_period'] = 250      # match the reference runs, NOT the 500 house rule
    cfg['figs_period'] = 500
    # phase1_exit sits at step ~430 (both reference runs start there), so epochs
    # is set ABSOLUTELY here rather than via START.
    cfg['epochs'] = 3500
    out.append(('ablate', cfg))

    # ---- ablate_mid ----------------------------------------------------------
    # The decisive follow-up. IDENTICAL to `ablate` in every field except the
    # checkpoint: this one resumes MID-equilibration from _running.pt (the point
    # the v8 ramp started from, ~step 4149, a policy and replay buffer inherited
    # through tuphwfkm and its ancestors) instead of entering the stage fresh
    # from _phase1_exit.pt.
    #
    # `ablate` already reproduced reference jybhxgxl on the current working tree,
    # so the code is not the cause. This isolates the one remaining variable.
    # If replay/tb_err jitter lands near 0.008 rather than 0.0018, the answer is
    # the resumed STATE, and the question becomes what in that lineage did it.
    cfg = dict(cfg)
    cfg['run_name'] = 'ablate_mid'
    cfg['checkpoint_name'] = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset'
                              '-T2.5-573c92_running.pt')
    cfg['epochs'] = 7200          # ~3050 steps from ~4149, matching ablate's length
    out.append(('ablate_mid', cfg))

    for name, c in out:
        path = os.path.join(HERE, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)
        al, rc = c['adaptive_lr'], c['adaptive_lr']['ray_calibration']
        # The probe's armed/inert state is DERIVED from the stages now, so it is
        # printed: there is no longer a flag in the file to read it off.
        askers = [st['name'] for st in c['protocols'][c['protocol']]['stages']
                  if (st.get('lr_sensor') or {}).get('kind') == 'ray']
        print(f"wrote {path}")
        print(f"   seed {al['seed_lr']:.3g} ({al['seed_lr'] / HEALTHY:.2g}x healthy) "
              f"| {c['epochs'] - START} steps | period {rc['period']} | "
              f"target alpha* {al['calibration']['alpha_target']} | "
              f"read_only {c['checkpoint_read_only']} | "
              f"ray {','.join(askers) if askers else 'INERT (no stage asks)'}")
    return [n for n, _ in out]


if __name__ == '__main__':
    build()
