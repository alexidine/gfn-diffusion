"""dose_sep16f -- is the shared optimum robust to how the prior is presented? Same seed
archive, same 8 h wall, same generator contract as dose_sep16 / 16b / 16c / 16d / 16e.

WHY (owner, 2026-09-17): every arm that reaches high Z stops visiting one lattice family
(bwd_outside_fwd_final rises ~4% of the prior per nat of Z, frozen and trainable arms on
the same line). Two very different training routes agree, so it is not a training
artefact; what is untested is whether the way the PRIOR is presented to the backward
branch decides it. The checkpoint probes found the sampler ~1.3 nats under-dense on every
basin's thermal shell (~+4 kT, latent radius ~0.02 in the soft directions) and ~0 at the
minima: the backward branch's noise tiles (isotropic, log-uniform 0.003-0.03) score the
floors and the stiff walls, never the shell. So: rebuild the backward branch's buffer from
NOISED ANCHORS ONLY (no prior model in the loop), at three noise radii and two sizes, on
the best-behaved shape (dose16e_n1_pbfrozen: N=1, tau 120, P_B frozen, Z 35.55 at 7.6k,
sd 0.01), and ask whether the endpoint moves.

MECHANICS. The arms resume INSIDE 'equilibration', so its on_enter never fires and the
restored 250k-row prior buffer (seeded from the prior dataset) would just stay. Each arm
therefore gives 'equilibration' an always-true exit and appends a copy of it named
'equilibration_anch' whose on_enter is rebuild_prior_by_churn: at the first eval the run
transitions, the prior buffer is discarded and refilled from noised anchors to
init_fraction x max_size (min_size=10k rows per cycle, cap 4x the needed cycles), and
training continues with the same fracs, bounds and loss coefficients. Every arm, the
control included, shares that transition, so the optimizer rebuild at the boundary is
common to all four; the reference without it is dose16e_n1_pbfrozen.

  anch_n003     noise 10^-2.5..10^-1.5 (today's), buffer 62.5k   the control for the
                anchors-only rebuild itself
  anch_n02      noise 10^-1.8..10^-1.6 (0.016-0.025 = the thermal shell), 62.5k
  anch_n05      noise 10^-1.5..10^-1.2 (0.03-0.06 = the walls), 62.5k
  anch_n02_big  the shell noise with the buffer rebuilt to the full 250k

READ: Z at matched steps against dose16e_n1_pbfrozen and against each other;
bwd_outside_fwd_final and w1r/median (comparable ACROSS these four, not to earlier arms:
the buffer composition differs); offline, the family split of each endpoint's replay
buffer. If the optimum is robust, all four end within ~0.3 nat with the same family share.
If the shell was the missing information, anch_n02 ends with a higher share of the
abandoned family (and possibly higher Z); anch_n05 should mostly pin walls and change
little.
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = 'dose16f'

_spec = importlib.util.spec_from_file_location('dosemake', ROOT / 'dose_sep16' / 'make.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

EXIT_METRIC = 'fwd/log_Z_learned'
NEW_STAGE = 'equilibration_anch'


def _anchors_only(lo, hi, max_size, init_fraction):
    def d(cfg):
        pb = cfg['buffers']['prior_buffer']
        pb['source'] = 'anchors'
        pb['max_size'] = int(max_size)
        pb['init_fraction'] = float(init_fraction)
        cfg['buffers']['anchor_buffer']['noise_log_range'] = [float(lo), float(hi)]
    return d


def _rebuild_stage(cfg):
    stages = cfg['protocols']['unconditional_tb']['stages']
    eq = dm._eq(cfg)
    eq['exit'] = [{'metric': EXIT_METRIC, 'above': -1.0e9, 'patience': 1}]
    new = copy.deepcopy(eq)
    new['name'] = NEW_STAGE
    new.pop('exit')
    new['on_enter'] = ['rebuild_prior_by_churn']
    stages.append(new)


BASE_DELTAS = dm.COMMON + [dm._every(1), dm._freeze_pb]
ARMS = {
    'anch_n003':    BASE_DELTAS + [_anchors_only(-2.5, -1.5, 250_000, 0.25), _rebuild_stage],
    'anch_n02':     BASE_DELTAS + [_anchors_only(-1.8, -1.6, 250_000, 0.25), _rebuild_stage],
    'anch_n05':     BASE_DELTAS + [_anchors_only(-1.5, -1.2, 250_000, 0.25), _rebuild_stage],
    'anch_n02_big': BASE_DELTAS + [_anchors_only(-1.8, -1.6, 250_000, 1.0), _rebuild_stage],
}
EXPECT = {  # (noise lo, noise hi, max_size, init_fraction)
    'anch_n003': (-2.5, -1.5, 250_000, 0.25), 'anch_n02': (-1.8, -1.6, 250_000, 0.25),
    'anch_n05': (-1.5, -1.2, 250_000, 0.25), 'anch_n02_big': (-1.8, -1.6, 250_000, 1.0),
}


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
    lo, hi, max_size, init_fraction = EXPECT[arm]
    eq = dm._eq(cfg)
    assert cfg['batch_size'] == cfg['max_batch_size'] == 1600 and cfg['grow_batch_size'] is False, name
    assert cfg['batch_util_target'] == 0.0 and cfg['batch_sizer_retest_steps'] == 0, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['val_frac'] == 0.05 and rb['val_cap'] == 1024 and rb['churn_rate'] == 0 and rb['mean_residence_steps'] == 120, name
    assert eq['fwd_rollout_every'] == 1, name
    assert eq['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3} and eq['balance']['bounds'] == {'bwd': [0.7, 0.7], 'replay': [0.3, 0.3]}, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == 1.0 and abs(lc['seed_lr'] - 1.25e-4) < 1e-12, name
    assert cfg['freeze_backward_policy'] == 'full', name
    pb = cfg['buffers']['prior_buffer']
    ab = cfg['buffers']['anchor_buffer']
    assert pb['source'] == 'anchors' and pb['max_size'] == max_size and pb['init_fraction'] == init_fraction, name
    assert ab['noise_log_range'] == [lo, hi] and ab['frozen'] is True, name
    assert pb['min_size'] == 10000, name + ': rebuild admits min_size rows per cycle'
    stages = cfg['protocols']['unconditional_tb']['stages']
    names = [s['name'] for s in stages]
    assert names[-2:] == ['equilibration', NEW_STAGE], names
    assert eq['exit'] == [{'metric': EXIT_METRIC, 'above': -1.0e9, 'patience': 1}], name
    new = stages[-1]
    assert 'exit' not in new and new['on_enter'] == ['rebuild_prior_by_churn'], name
    for k in ('balance', 'fracs', 'loss_coeffs', 'fwd_rollout_every', 'bwd_sampling_mode', 'train_mode', 'flags'):
        assert new[k] == eq[k], name + ': ' + k
    dm._scan_local_paths(cfg, name) if hasattr(dm, '_scan_local_paths') else None


SBATCH = dm.SBATCH.replace('#SBATCH --job-name=dose16', '#SBATCH --job-name=dose16f') \
    .replace('configs/dose_sep16/joblogs/%x_%A_%a.out', 'configs/dose_sep16f/joblogs/%x_%A_%a.out') \
    .replace('ARMS=${{WORKDIR}}/configs/dose_sep16', 'ARMS=${{WORKDIR}}/configs/dose_sep16f') \
    .replace('# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.',
             '# dose_sep16f: anchors-only prior buffer at three noise radii, off the SAME frozen archive dose_sep16 and flk_sep14 used.')
assert 'job-name=dose16f' in SBATCH and 'configs/dose_sep16f/joblogs' in SBATCH \
    and 'ARMS=${{WORKDIR}}/configs/dose_sep16f' in SBATCH and 'dose_sep16/' not in SBATCH.replace('dose_sep16f', '')


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
    with (HERE / 'submit_dose_sep16f.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=dm.WALL, last=len(arms) - 1, placeholder=dm.PLACEHOLDER,
                              prior_placeholder=dm.PRIOR_PLACEHOLDER))
    for name in arms:
        lo, hi, max_size, init_fraction = EXPECT[name[len(TAG) + 1:]]
        print('%-22s noise %.4f..%.4f latent  rebuild to %6d of %6d rows  (N=1, tau 120, P_B frozen, replay 0.3, batch 1600)' % (
            name, 10 ** lo, 10 ** hi, int(max_size * init_fraction), max_size))


if __name__ == '__main__':
    main(sys.argv[1:])
