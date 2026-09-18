"""dose_sep16g -- owner's side experiment (2026-09-18): wider anchor noise than any 16f arm
with a 10x looser prior-buffer energy window. Same seed archive, same shape
(dose16e_n1_pbfrozen: N=1, tau 120, P_B frozen, replay 0.3 pinned, batch 1600), same
anchors-only rebuild-on-resume mechanics and legacy-walls launch as dose_sep16f.

THE WINDOW. A noised anchor enters the prior buffer only if its composite energy sits
within buffers.prior_buffer.ramp_floor (100 kJ/mol) of its condition's best known
energy; the same test expires resident rows as that minimum ratchets down
(expire_max_frac 0.1 per call), and the coverage weight ramps from 0 at the floor to 1
at floor - ramp_width (50). The reach trigger (reach_quantile 0.9 / reach_threshold
0.75) reads the buffer's 90th-percentile excess against the SAME floor and tops up from
anchors when it covers less than 75% of it. At ramp_floor 1000 / ramp_width 500: rows up
to 1000 kJ/mol above the minimum are admitted and never expire, the coverage weight is
flat to 500 above the minimum, and the reach trigger cannot fire (excesses of tens of
kJ/mol never reach 75% of 1000).

  anch_n08_w10  noise 10^-1.4..10^-1.1 (0.04-0.08 latent), ramp_floor 1000, ramp_width 500

One arm (owner's call). Read after ~5 tau against dose16f_anch_n05 (0.03-0.06, window as
today) and the parent; the noise and the window are not separated by this arm alone.
"""
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = 'dose16g'

_spec = importlib.util.spec_from_file_location('dose16fmake', ROOT / 'dose_sep16f' / 'make.py')
df = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(df)
dm = df.dm


def _window(floor, width):
    def d(cfg):
        pb = cfg['buffers']['prior_buffer']
        pb['ramp_floor'] = float(floor)
        pb['ramp_width'] = float(width)
    return d


NOISE = (-1.4, -1.1)
ARMS = {   # owner 2026-09-18: just the one arm; the noise-only control is dose16f_anch_n05 (0.03-0.06)
    'anch_n08_w10': df.BASE_DELTAS + [df._anchors_only(*NOISE, 250_000, 0.25), df._rebuild_stage, _window(1000, 500)],
}
EXPECT = {'anch_n08_w10': (NOISE[0], NOISE[1], 1000, 500)}


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
    lo, hi, floor, width = EXPECT[arm]
    eq = dm._eq(cfg)
    assert cfg['batch_size'] == cfg['max_batch_size'] == 1600 and cfg['grow_batch_size'] is False, name
    assert cfg['batch_util_target'] == 0.0 and cfg['batch_sizer_retest_steps'] == 0, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['val_frac'] == 0.05 and rb['val_cap'] == 1024 and rb['churn_rate'] == 0 and rb['mean_residence_steps'] == 120, name
    assert eq['fwd_rollout_every'] == 1, name
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3} and eq['balance']['bounds'] == {'bwd': [0.7, 0.7], 'replay': [0.3, 0.3]}, name
    assert cfg['freeze_backward_policy'] == 'full', name
    pb = cfg['buffers']['prior_buffer']
    ab = cfg['buffers']['anchor_buffer']
    assert pb['source'] == 'anchors' and pb['max_size'] == 250_000 and pb['init_fraction'] == 0.25, name
    assert ab['noise_log_range'] == [lo, hi] and ab['frozen'] is True, name
    assert pb['ramp_floor'] == floor and pb['ramp_width'] == width and 0 < pb['ramp_width'] <= pb['ramp_floor'], name
    stages = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in stages][-2:] == ['equilibration', df.NEW_STAGE], name
    assert eq['exit'] == [{'metric': df.EXIT_METRIC, 'above': -1.0e9, 'patience': 1}], name
    assert stages[-1]['on_enter'] == ['rebuild_prior_by_churn'] and 'exit' not in stages[-1], name
    dm._scan_local_paths(cfg, name) if hasattr(dm, '_scan_local_paths') else None


SBATCH = df.SBATCH.replace('#SBATCH --job-name=dose16f', '#SBATCH --job-name=dose16g') \
    .replace('configs/dose_sep16f/joblogs/%x_%A_%a.out', 'configs/dose_sep16g/joblogs/%x_%A_%a.out') \
    .replace('ARMS=${{WORKDIR}}/configs/dose_sep16f', 'ARMS=${{WORKDIR}}/configs/dose_sep16g') \
    .replace('# dose_sep16f: anchors-only prior buffer at three noise radii,',
             '# dose_sep16g: wider anchor noise with / without a 10x looser prior-buffer energy window,')
assert 'job-name=dose16g' in SBATCH and 'configs/dose_sep16g/joblogs' in SBATCH \
    and 'ARMS=${{WORKDIR}}/configs/dose_sep16g' in SBATCH and SBATCH.count('export MXT_LEGACY_TRICLINIC_WALLS=1') == 1 \
    and 'dose_sep16f' not in SBATCH.replace('dose_sep16f and flk_sep14', '')


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
    with (HERE / 'submit_dose_sep16g.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=dm.WALL, last=len(arms) - 1, placeholder=dm.PLACEHOLDER,
                              prior_placeholder=dm.PRIOR_PLACEHOLDER))
    for name in arms:
        lo, hi, floor, width = EXPECT[name[len(TAG) + 1:]]
        print('%-22s noise %.4f..%.4f latent  ramp_floor %5d ramp_width %4d  (N=1, tau 120, P_B frozen, anchors-only rebuild to 62.5k)' % (
            name, 10 ** lo, 10 ** hi, floor, width))


if __name__ == '__main__':
    main(sys.argv[1:])
