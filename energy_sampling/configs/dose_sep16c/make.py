"""dose_sep16c -- the tau (pool size) arms at moderate and high N, same seed archive,
same 8 h wall, same generator contract as dose_sep16 / dose_sep16b.

WHAT dose_sep16 SHOWS AT ~3k STEPS (read from wandb 2026-09-16 22:10):
per ENERGY CALL n5 tracks n1 exactly to 500 rollouts (32.83 vs 32.80) and then
stalls at ~32.5 (held-out gap 0.53) while n1 keeps climbing (34.2 at 2300 steps,
gap 0.13). Within the N=20 family the level is monotone in the pressure per row
(n20 29.5 < n20_b2560 30.0 < n20_lr05 31.6), and at N=1 the pressure barely
matters (n1_w15 33.6 vs n1 34.2). So passes per row AND pressure per pass both set
the plateau. tau was varied only at N=1 (n1 vs n1_t6, tau 120 vs 6: a wash) --
i.e. only where there is no memorisation to cure. Under store-all, tau sets the
POOL (B*tau/N rows) and the row AGE but not the passes per row (always N) or the
pressure per pass. On the local ELJ rig (rr_hc2_n20 vs _t25, tau 100 -> 500 at
N=20) a 5x pool cut the held-out gap 41% and improved the held-out fit with log Z
flat. Whether a large pool lifts the PLATEAU at N=5 / N=20 on mip is untested.

  n5_t600     n5 with tau 600: pool 192k rows (n1's pool) at 5 passes per row.
              Does n1's pool lift n5 off its 32.5 shelf? The pool-matched test of
              passes per row.
  n20_t600    n20 with tau 600: pool 48k (5x n20's, the local rig's contrast).
              Does the held-out gap fall and does Z follow, at 20 passes per row?

Read both only after ~5 tau = 3000 steps (the pool is still filling before that;
at 2 tau the local rig read the OPPOSITE sign). N=20 arms run ~1300 steps/h, N=5
~1270, so the settled window opens ~2.5 h in.
"""
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = 'dose16c'

_spec = importlib.util.spec_from_file_location('dosemake', ROOT / 'dose_sep16' / 'make.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

ARMS = {
    'n5_t600':  dm.COMMON + [dm._every(5), dm._tau(600)],
    'n20_t600': dm.COMMON + [dm._every(20), dm._tau(600)],
}
EXPECT = {'n5_t600': (5, 0.3, 1.0, 600, 1600), 'n20_t600': (20, 0.3, 1.0, 600, 1600)}


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
    assert cfg['freeze_backward_policy'] is False, name
    assert not cfg['replay_loss_coeffs'].get('stored_force_k'), name
    dm._scan_local_paths(cfg, name) if hasattr(dm, '_scan_local_paths') else None


SBATCH = dm.SBATCH.replace('#SBATCH --job-name=dose16', '#SBATCH --job-name=dose16c') \
    .replace('configs/dose_sep16/joblogs/%x_%A_%a.out', 'configs/dose_sep16c/joblogs/%x_%A_%a.out') \
    .replace('ARMS=${{WORKDIR}}/configs/dose_sep16', 'ARMS=${{WORKDIR}}/configs/dose_sep16c') \
    .replace('# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.',
             '# dose_sep16c: the tau (pool size) arms off the SAME frozen archive dose_sep16 and flk_sep14 used.')
assert 'job-name=dose16c' in SBATCH and 'configs/dose_sep16c/joblogs' in SBATCH and 'ARMS=${{WORKDIR}}/configs/dose_sep16c' in SBATCH and 'dose_sep16/' not in SBATCH.replace('dose_sep16c', '')


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
    with (HERE / 'submit_dose_sep16c.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=dm.WALL, last=len(arms) - 1, placeholder=dm.PLACEHOLDER,
                              prior_placeholder=dm.PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        n, w, scale, tau, batch = EXPECT[name[len(TAG) + 1:]]
        print('%-20s N=%-3d replay=%.2f scale=%.2f tau=%-4d pool=%7d dose~%5.1f' % (
            name, n, w, scale, tau, batch * tau // n, dm.dose(n, scale, w, batch)))


if __name__ == '__main__':
    main(sys.argv[1:])
