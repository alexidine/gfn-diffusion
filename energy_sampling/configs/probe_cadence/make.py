"""
probe_cadence -- measure the CADENCE DEPENDENCE of the step-probe's statistics.

THE QUESTION. `step_probe` currently runs at cadence 20 and the servo medians
over a window of 25, so a reading costs 500 train steps to fully refresh. That
latency is what makes a stepwise LR climb cost ~600 steps/rung, which is too
slow to be useful. The fix on the table is BURSTS: probe at a much higher
cadence for a short window, sparse cadence otherwise.

Bursts are only worth anything if readings taken CLOSE TOGETHER are still
statistically independent. Measured on the live runs, alpha* is essentially
white noise around a slowly-varying mean -- the variogram V(b) tracks the white
prediction sigma^2/b out to blocks of 320-640 train steps. But that whiteness
was measured AT CADENCE 20, i.e. it only establishes independence at lag >= 20
train steps. It says nothing about lag 1-5, and Adam's beta1 = 0.9 is a concrete
reason to expect correlation there: successive step DIRECTIONS share momentum
state, and alpha* is a property of the step direction.

  If readings 2 steps apart are independent -> a 50-step burst delivers a
  full-precision reading and the climb loop gets a ~10x latency cut.
  If they are ~0.8 correlated -> that burst is worth ~3 independent samples and
  buys nothing; the useful floor is cadence 4-5.

So: run the probe at CADENCE 1 and measure the autocorrelation of the raw
alpha* stream against lag. One number decides the burst design.

WHY THIS ARM. Resumes the current dev checkpoint at step 5300 -- phase 2
(equilibration), past warmup (envelope 1.0), peak_scale 1.0, batch 1000 -- which
is a well-behaved stationary operating point rather than a transient.

STATIONARITY IS THE POINT, so this arm deliberately pins the LR. All four policy
LR keys become explicit floats at 1.25e-4, which is EXACTLY what the servo-seeded
`auto` path was already producing (seed_lr 1.25e-4 x envelope 1.0 x peak_scale
1.0). Explicit floats sit outside `lr_servo_managed` (utils.py), so peak_scale is
never applied (controller.py::_apply_lrs) and `servo_enabled` goes False for want
of managed keys -- the LR cannot move under us while we measure the noise it
feeds. Nothing about the measured statistics changes; only the guarantee does.

WRITES ARE OFF. `checkpoint_read_only: true` (checkpointing.py:94) suppresses
save/save_buffers/archive while leaving loading active, so this cannot clobber
the dev run's checkpoints -- which a normal launch would, via train.py's
save('final') outside the loop and save('running') every 50 steps.

Run:  python configs/probe_cadence/make.py
Then: the driver in the session scratchpad patches StepProbe.measure to record
      every individual reading (report() only surfaces the latest one per 10
      steps, which is far too coarse to see short-lag structure).
"""
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..', 'mk_dev.yaml')

# resume point: phase 2, step 5300, envelope 1.0, peak_scale 1.0
CKPT = 'dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
LR = 1.25e-4        # what the servo-seeded auto path was already running
N_STEPS = 500       # readings; enough for autocorrelation out to lag ~50


def build():
    with open(BASE) as f:
        cfg = yaml.safe_load(f)

    cfg['run_name'] = 'probe_cad1'
    cfg['tag'] = 'cadmeas'

    # resume the dev checkpoint, full state, and write NOTHING back
    cfg['checkpoint_name'] = CKPT
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['checkpoint_read_only'] = True
    cfg['archive_period'] = 0

    # epochs is ABSOLUTE and the loop is trange(init_step, epochs+1): an epochs
    # below the checkpoint's step silently runs zero steps and verifies nothing
    cfg['epochs'] = 5300 + N_STEPS

    # every step probed
    cfg['step_probe']['cadence'] = 1

    # pin the LR: explicit floats are not servo-managed, so peak_scale never
    # applies and the servo has nothing to own
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = LR

    out = os.path.join(HERE, 'cad1.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'wrote {out}')
    print(f'  resume {CKPT} @5300 -> {cfg["epochs"]}  ({N_STEPS} steps)')
    print(f'  step_probe.cadence = {cfg["step_probe"]["cadence"]}, '
          f'window = {cfg["step_probe"]["window"]}, span = {cfg["step_probe"]["span"]}')
    print(f'  lr pinned at {LR} (explicit -> unmanaged), checkpoint writes SUPPRESSED')
    return out


def build_ramp():
    """THE EDGE-FINDING ARM. Sweep the LR 3 orders of magnitude (10x below the
    known-good rate to 100x above it) over 1000 steps and record what the sensor
    does as training goes from healthy to destroyed.

    The question this answers: what does 'too hot' look like to the probe, and
    does it look like anything that transfers WITHOUT per-problem calibration?
    An absolute alpha* bar needs calibrating (alpha* is ~4.9 at the known-good
    rate here, nowhere near the textbook 1.0). Fit VALIDITY might not: today
    100% of fits are usable, and a 'downward' fit means the parabola model is
    simply wrong, which is a qualitative transition rather than a tuned number.

    The driver owns the LR schedule and records the probe's held-out loss (l0)
    alongside it, so the detector's verdict can be scored against the actual
    onset of damage on the same axis.
    """
    with open(BASE) as f:
        cfg = yaml.safe_load(f)

    cfg['run_name'] = 'probe_ramp'
    cfg['tag'] = 'cadmeas'
    cfg['checkpoint_name'] = CKPT
    cfg['load_weights_only'] = False
    cfg['continue_from_checkpoint'] = False
    cfg['checkpoint_read_only'] = True
    cfg['archive_period'] = 0
    cfg['epochs'] = 5300 + 1000
    cfg['step_probe']['cadence'] = 2      # measured independent at lag 2
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = LR                       # overridden per step by the driver

    out = os.path.join(HERE, 'ramp.yaml')
    with open(out, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'wrote {out}  (1000 steps, cadence 2, LR ramp driven externally)')
    return out


if __name__ == '__main__':
    build()
    build_ramp()
