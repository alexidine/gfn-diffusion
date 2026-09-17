"""dose_sep16b -- two companion arms for the two spare GPUs (2026-09-16 evening),
same seed archive, same 8 h wall, same generator contract as dose_sep16.

WHAT THE FIRST 3k STEPS OF dose_sep16 SHOWED (read from wandb 2026-09-16):
log Z is monotone in the reuse per row -- n20 29.4, n5 32.7, n1 34.0 -- and
n1_t6 (reuse 1 with n20's buffer shape) sits at 33.8 beside n1, so the lever is
the dose, not the pool size or the fresh fraction. The held-out replay gap says
the same: ~1.4 nats on every N=20 arm, ~0.1 on every N=1 arm. n20_pbfrozen
matched n20 exactly, which validates nothing: at N=20 there was no gain to
attribute. The validity check has to sit on a winning arm.

  n1_t6_pbfrozen   n1_t6 with P_B frozen (freeze_backward_policy full). If the
                   5-nat gain survives, it is P_F improving; if it vanishes, the
                   N=1 arms were moving P_B toward P_F.
  n20_lr025        n20 at a quarter of the base rate: dose 12, the same as n5.
                   The rate is the lever the UMA production arms CAN pull
                   (they cannot afford N=5); if this matches n5 the dose
                   transfers through the rate and the UMA setting follows.
"""
import importlib.util
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent
TAG = 'dose16b'

_spec = importlib.util.spec_from_file_location('dosemake', ROOT / 'dose_sep16' / 'make.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

ARMS = {
    'n1_t6_pbfrozen': dm.COMMON + [dm._every(1), dm._tau(6), dm._freeze_pb],
    'n20_lr025':      dm.COMMON + [dm._every(20), dm._rate(0.25)],
}
EXPECT = {'n1_t6_pbfrozen': (1, 0.3, 1.0, 6, 1600), 'n20_lr025': (20, 0.3, 0.25, 120, 1600)}


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
    assert cfg['freeze_backward_policy'] == ('full' if arm == 'n1_t6_pbfrozen' else False), name
    assert not cfg['replay_loss_coeffs'].get('stored_force_k'), name
    dm._scan_local_paths(cfg, name) if hasattr(dm, '_scan_local_paths') else None


SBATCH = dm.SBATCH.replace('#SBATCH --job-name=dose16', '#SBATCH --job-name=dose16b') \
    .replace('configs/dose_sep16/joblogs/%x_%A_%a.out', 'configs/dose_sep16b/joblogs/%x_%A_%a.out') \
    .replace('ARMS=${{WORKDIR}}/configs/dose_sep16', 'ARMS=${{WORKDIR}}/configs/dose_sep16b') \
    .replace('# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.',
             '# dose_sep16b: two companion arms off the SAME frozen archive dose_sep16 and flk_sep14 used.')


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
    with (HERE / 'submit_dose_sep16b.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=dm.WALL, last=len(arms) - 1, placeholder=dm.PLACEHOLDER,
                              prior_placeholder=dm.PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        n, w, scale, tau, batch = EXPECT[name[len(TAG) + 1:]]
        print('%-24s N=%-3d replay=%.2f scale=%.2f tau=%-4d dose~%5.1f freeze_pb %s' % (
            name, n, w, scale, tau, dm.dose(n, scale, w, batch), cfg['freeze_backward_policy']))


if __name__ == '__main__':
    main(sys.argv[1:])
