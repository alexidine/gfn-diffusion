"""local_pb_freeze: P_B trainable vs P_B frozen after phase 1, one variable.

    python configs/local_pb_freeze/make.py

Both arms are configs/local_prod_sep02/lp02.yaml (the local ELJ mipcas rig,
T=10, batch 400, fixed fracs 0.05/0.475/0.475, fixed LR scale 0.125, warm
start from dev_elj_p2_cruise's phase-1 exit) with:

  * the same seed, so the two runs share every draw until the weights diverge;
  * grad_geometry on every 20 fused steps, which now also reports the
    per-submodel split of each branch's gradient (fused_grad/{branch}_norm_
    backward_policy etc.) -- this is the (f) measurement, bwd vs replay as the
    source of P_B's phase-2 gradient, and it only exists on the trainable arm;
  * `freeze_backward_policy` false / 'head' / 'full' -- the ONLY difference.
    'head' is the cheap "zero the backward_policy param group" freeze, which
    leaves P_B drifting through the shared s_model/t_model trunk; 'full'
    evaluates P_B on a snapshot of trunk + head, so P_B is genuinely the
    phase-1 exit's. gradnorm/backward_policy must read 0 on both.

The train_prior stage of prod_eq exits at the first eval (bwd/mle > -1e9), so
equilibration engages at ~step 200 and runs to `epochs`. T=10 is the rig's
length, NOT the ship length (T=100); that caveat is in the write-up.
"""
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'local_prod_sep02' / 'lp02.yaml'
EPOCHS = 3200
GEOM_EVERY = 20

ARMS = {
    'pbab_train': dict(freeze_backward_policy=False),
    'pbab_headfrozen': dict(freeze_backward_policy='head'),   # requires_grad off on the head only
    'pbab_frozen': dict(freeze_backward_policy='full'),       # P_B on a snapshot of trunk + head
}


def build(name, freeze_backward_policy):
    cfg = yaml.safe_load(BASE.read_text())
    cfg['run_name'] = name
    cfg['tag'] = 'pbab'
    cfg['epochs'] = EPOCHS
    cfg['freeze_backward_policy'] = freeze_backward_policy
    cfg['grad_geometry'] = dict(enabled=True, every=GEOM_EVERY)
    assert cfg['seed'] == 12345 and cfg['load_weights_only'] is True
    assert cfg['checkpoint_name'].endswith('_phase1_exit.pt')
    assert cfg['integrator']['T'] == cfg['eval_T'] == 10
    assert cfg['lr_control']['mode'] == 'fixed' and cfg['lr_control']['fixed_scale'] == 0.125
    return cfg


if __name__ == '__main__':
    for name, kw in ARMS.items():
        cfg = build(name, **kw)
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        print('wrote', name)
