"""
lrsensor_aug11 -- functional test of the per-stage lr_sensor (ReduceLROnPlateau).

    python configs/lrsensor_aug11/make.py

Two scenarios, deliberately NOT matched to each other -- they are different
tempos. Each is FRESH, so each passes through train_prior (MLE channel) and then
var_conditioning (the two VarGrad channels), exercising both in one run.

  blowup   seed_lr 1.0e-3, ~8x the T=40 optimum, with warmup compressed to 200
           steps so it arrives hot almost immediately. patience 10 (100 steps) so
           it reacts inside the excursion.
           PASS = a cut fires, peak_scale falls, the loss turns back over.

  normal   stock seed_lr 1.25e-4 and warmup 1000, patience 60 (600 steps).
           PASS = quiet through the healthy descent, and a cut once progress
           stalls. The earlier slope-only sensor sat clean for 3000 steps here
           while the run was stalled at 8x the healthy LR, which is what the
           plateau rule exists to catch.

patience is in CHECKS, one per 10 train steps. 600 steps is where healthy (arm A)
and stalled (lrs_normal) separate; at 300 they do not separate at all.

Both leave ray_calibration.enabled TRUE while declaring kind: plateau, so
lr_ctrl/calibrations staying 0 is a live assertion that the per-stage gate
suppresses the ray probe.

NB warmup re-arms per stage, so each stage's sensor is blocked for `warmup` steps
after that stage is ENTERED. In lrs_normal phase 1 exits around step 570, before
its 1000-step warmup elapses, so its MLE sensor never activates -- by design. MLE
coverage comes from lrs_blowup, which uses warmup 200.
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from configs.generate_configs import overwrite_nested_dict  # noqa: E402

# (name, seed_lr, warmup_steps, patience_checks, epochs)
SCENARIOS = [
    ('lrs_blowup', 1.0e-3, 200, 10, 2000),
    ('lrs_normal', 1.25e-4, 1000, 60, 3000),
]

# Named per stage, not derived from the active coefficients: bwd level_gap has a
# coefficient but is sign-indefinite and not a convergence signal, and
# vg_by_condition has one while being a boolean switch with no series behind it.
STAGE_METRICS = {
    'train_prior': ['bwd/mle'],
    'var_conditioning': ['fwd/vg_lb', 'bwd/vg_lb'],
}


def _stage(cfg, name):
    for st in cfg['protocol']['stages']:
        if st['name'] == name:
            return st
    raise KeyError(f'stage {name!r} not in protocol')


def build(base, name, seed_lr, warmup, patience, epochs):
    cfg = overwrite_nested_dict(yaml.safe_load(yaml.safe_dump(base)),
                                {'run_name': name, 'epochs': epochs})
    cfg['adaptive_lr']['seed_lr'] = seed_lr
    cfg['adaptive_lr']['warmup_steps'] = warmup
    for stage_name, metrics in STAGE_METRICS.items():
        _stage(cfg, stage_name)['lr_sensor'] = {
            'kind': 'plateau',
            'metrics': list(metrics),
            'factor': 0.5,
            'patience': patience,
            'cooldown': 10,
        }
    return cfg


def assert_test_shape(cfg, name):
    """Everything the test depends on, asserted rather than trusted."""
    assert cfg['checkpoint_name'] is None and cfg['prior_model_name'] is None, \
        f'{name}: must be FRESH or train_prior is skipped and the MLE sensor never runs'
    assert cfg['continue_from_checkpoint'] is False, f'{name}: must not resume'
    assert cfg['ray_calibration']['enabled'] is True, \
        f'{name}: leave ray enabled -- calibrations staying 0 is the gate assertion'
    for stage_name, metrics in STAGE_METRICS.items():
        ls = _stage(cfg, stage_name).get('lr_sensor')
        assert ls and ls['kind'] == 'plateau', f'{name}/{stage_name}: needs a plateau sensor'
        assert ls['metrics'] == metrics, f'{name}/{stage_name}: metrics drifted'
    # blocked through warmup, then one patience per decision -- leave room for a
    # few. warmup re-arms per stage, so this budget is per stage, not global.
    warmup = cfg['adaptive_lr']['warmup_steps']
    patience = _stage(cfg, 'var_conditioning')['lr_sensor']['patience']
    need = warmup + patience * 10 * 3
    assert cfg['epochs'] >= need, \
        (f'{name}: epochs {cfg["epochs"]} < {need} = warmup {warmup} + 3x patience '
         f'({patience} checks). Too few chances to act.')


def main():
    argparse.ArgumentParser().parse_args()
    base = yaml.safe_load((HERE / 'base.yaml').read_text())
    rows = ['name\tseed_lr\twarmup\tpatience_checks\tepochs']
    for name, seed_lr, warmup, patience, epochs in SCENARIOS:
        cfg = build(base, name, seed_lr, warmup, patience, epochs)
        assert_test_shape(cfg, name)
        (HERE / f'{name}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        rows.append(f'{name}\t{seed_lr:g}\t{warmup}\t{patience}\t{epochs}')
        print(f'wrote {HERE / f"{name}.yaml"}')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n')
    print(f'wrote {HERE / "INDEX.tsv"}')


if __name__ == '__main__':
    main()
