"""dose_sep16e -- P_B frozen along the N ladder and the pressure ladder. Same seed
archive, same 8 h wall, same generator contract as dose_sep16 / 16b / 16c / 16d.

The 16d battery holds the production candidates (freeze + rate/4 at N=20, the
same + tau 600, and n5 tau 600 frozen). These three isolate what the freeze
interacts with:

  n1_pbfrozen   n1 (tau 120, pool 192k, the arm with the 1700-step Z cycle) with
                P_B frozen. n1_t6_pbfrozen removed the cycle on FRESH rows; this
                asks whether the freeze alone removes it on a stale pool, i.e.
                whether tau still matters once P_B is frozen.
  n5_pb         n5 (tau 120) frozen: dose 12 at 5 passes. Against n20_lr025_pb
                (dose 12 at 20 passes) it is the N ladder at fixed pressure per
                row under the freeze.
  n20_lr05_pb   n20 at rate/2 frozen: dose 23. With n20_lr025_pb it is the
                pressure ladder under the freeze -- is the level still set by the
                pressure once P_B cannot move?
"""
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = 'dose16e'

_spec = importlib.util.spec_from_file_location('dosemake', ROOT / 'dose_sep16' / 'make.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

ARMS = {
    'n1_pbfrozen': dm.COMMON + [dm._every(1), dm._freeze_pb],
    'n5_pb':       dm.COMMON + [dm._every(5), dm._freeze_pb],
    'n20_lr05_pb': dm.COMMON + [dm._every(20), dm._rate(0.5), dm._freeze_pb],
}
EXPECT = {'n1_pbfrozen': (1, 0.3, 1.0, 120, 1600), 'n5_pb': (5, 0.3, 1.0, 120, 1600),
          'n20_lr05_pb': (20, 0.3, 0.5, 120, 1600)}


def build():
    base = yaml.safe_load(dm.BASE.read_text(encoding='utf-8'))
    out = {}
    for arm, deltas in ARMS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))
        name = TAG + '_' + arm
        cfg['run_name'] = name
        cfg['tag'] = TAG
        cfg['checkpoint_name'] = dm.PLACEHOLDER
        cfg['prior_model_name'] = dm.PRIOR_PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        for d in deltas:
            d(cfg)
        check(cfg, name, arm)
        out[name] = cfg
    return out


def check(cfg, name, arm):
    n, w, scale, tau, batch = EXPECT[arm]
    eq = dm._eq(cfg)
    assert cfg['batch_size'] == cfg['max_batch_size'] == batch and cfg['grow_batch_size'] is False, name
    assert cfg['batch_util_target'] == 0.0 and cfg['batch_sizer_retest_steps'] == 0, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['val_frac'] == 0.05 and rb['val_cap'] == 1024 and rb['churn_rate'] == 0, name
    assert rb['mean_residence_steps'] == tau and rb['max_size'] >= 3 * batch * tau / n, name
    assert eq['fwd_rollout_every'] == n, name
    bwd = round(1 - w, 3)
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': bwd, 'replay': w} and eq['balance']['bounds'] == {'bwd': [bwd, bwd], 'replay': [w, w]}, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == 1.0, name
    assert abs(lc['seed_lr'] - 1.25e-4 * scale) < 1e-12, name
    assert cfg['freeze_backward_policy'] == 'full', name
    assert not cfg['replay_loss_coeffs'].get('stored_force_k'), name
    dm._scan_local_paths(cfg, name) if hasattr(dm, '_scan_local_paths') else None


SBATCH = dm.SBATCH.replace('#SBATCH --job-name=dose16', '#SBATCH --job-name=dose16e') \
    .replace('configs/dose_sep16/joblogs/%x_%A_%a.out', 'configs/dose_sep16e/joblogs/%x_%A_%a.out') \
    .replace('ARMS=${{WORKDIR}}/configs/dose_sep16', 'ARMS=${{WORKDIR}}/configs/dose_sep16e') \
    .replace('# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.',
             '# dose_sep16e: P_B frozen along the N and pressure ladders off the SAME frozen archive dose_sep16 and flk_sep14 used.')
assert 'job-name=dose16e' in SBATCH and 'configs/dose_sep16e/joblogs' in SBATCH and 'ARMS=${{WORKDIR}}/configs/dose_sep16e' in SBATCH and 'dose_sep16/' not in SBATCH.replace('dose_sep16e', '')


def main(argv):
    dirty = dm.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted files the arms execute:\n  ' + '\n  '.join(dirty))
    arms = build()
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text('SLURM cannot create --output; SEED.txt is inherited from flk_sep14 at launch\n', encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\n')
        for name in arms:
            f.write('%s\t%s\n' % (name, dm.PARENT))
    with (HERE / 'submit_dose_sep16e.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=dm.WALL, last=len(arms) - 1, placeholder=dm.PLACEHOLDER,
                              prior_placeholder=dm.PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        n, w, scale, tau, batch = EXPECT[name[len(TAG) + 1:]]
        print('%-26s N=%-3d replay=%.2f scale=%.2f tau=%-4d pool=%7d dose~%5.1f freeze_pb %s' % (
            name, n, w, scale, tau, batch * tau // n, dm.dose(n, scale, w, batch), cfg['freeze_backward_policy']))


if __name__ == '__main__':
    main(sys.argv[1:])
