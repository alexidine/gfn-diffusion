"""pool8sg4_next_nov22: three MLE arms that follow up pool8sg4_nov20's mle4 (owner 2026-11-22).

  v4_kick  mle4 from a fresh policy on the same four-group prior: vector embeddings only (atom_tokens split_pooled) and
           a fresh chart kick of each drawn row's parent (noised_prior.live_kernel chart) at KICK_LAW, the radius law
           P-1 v_kick used, set explicitly. STEPS. Batch fixed at 1000, mle4's landed batch (98-100% of its logged steps),
           so a cheaper model cannot make the occupancy sizer pick a different rows-per-step than the reference.
  p1_b250  the attribution control: P-1 only, the cross-attention model and stored children of pool8_smooth_nov21
           x_fixed, on the switched P-1 prior, fixed batch P1_BATCH (the ~250 P-1 rows per step mle4 trains on), STEPS.
           fused_grad_accum_min_samples follows the batch (inert on a bwd step; it is the denominator of updates_per_sec).
  x4_aug   rotation augmentation on the four-group cross-attention model, weights-only from mle4's step 125,000 archive:
           pool8_auglong_nov18's recipe (rotate_frames, mirror off, orient_grad_balance norm, rho 1, LR held at 0.2 x seed,
           exit on the held-out random-frame plateau, hard cap PLATEAU_CAP, then a geometric LR anneal to 0.02 x seed over
           ANNEAL_STEPS). Batch fixed at 1000: auglong's sizer swung 625-1600 through OOM cuts, and four groups leave
           about 7.6 GB less GPU room.

LR paths (x seed_lr 1.25e-4). v4_kick, p1_b250: mk_dev's schedule, 0.05 held for 3,000 steps, geometric to 0.2 by step
4,000, then 0.2 to the end. x4_aug: 0.2 from step 0 until the plateau exit or the cap, then 0.2 -> 0.02 over the anneal.

usage: python configs/pool8sg4_next_nov22/make.py     (three yamls + submit_pool8sg4_next_nov22.sbatch)
       python configs/pool8sg4_next_nov22/make.py --local DIR --prior P --conditions C --test T --ckdir CK --source S
                                                      (local smoke configs; the local prior is K=0, P-1 rows only)"""
import argparse
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel / 'make.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sg4 = _load('sg4', 'pool8sg4_nov20')
smooth = _load('smooth', 'pool8_smooth_nov21')
auglong_make = _load('auglong_make', 'pool8_auglong_nov18')

TAG = 'pool8sg4_next_nov22'
ARMS = ('v4_kick', 'p1_b250', 'x4_aug')
STEPS = 150000
KICK_LAW = [-2.0, -0.375]   # log10 Angstrom: pool8_smooth_nov21 v_kick's 10^U(-2, -0.375)
MLE4_BATCH = 1000
P1_BATCH = 250
X4_SOURCE = 'pool8sg4_nov20_mle4_elj-pool8_4sg_train_k4_metric-T2.5-1d2dd7_step125000.pt'
LOOP_MARGIN = 100           # epochs past the last stage's cap, so its on_exit (snapshot, stop) ends the run
FIXED = {'grow_batch_size': False, 'batch_util_target': 0.0}


def _stages(cfg):
    return cfg['protocols'][cfg['protocol']]['stages']


def v4_kick(files):
    cfg = sg4.mle4(files)
    nf, np_cfg = cfg['model']['nf'], cfg['noised_prior']
    assert nf['atom_tokens'] == 'split' and nf['sym_ops'] == 'pooled', (nf['atom_tokens'], nf['sym_ops'])
    assert np_cfg['train_kinds'] == ['child'] and np_cfg['live_kernel'] is None, np_cfg
    nf['atom_tokens'] = 'split_pooled'
    np_cfg.update(live_kernel='chart', live_log_range=list(KICK_LAW))
    stage = _stages(cfg)[0]
    assert stage['name'] == 'train_prior' and 'lr_schedule' not in stage, stage
    stage['max_steps'] = STEPS
    cfg.update(batch_size=MLE4_BATCH, epochs=STEPS, tag=TAG, run_name='v4_kick', **FIXED)
    return cfg


def p1_b250(files):
    cfg = smooth.arm('x_fixed', files)
    assert cfg['model']['nf']['atom_tokens'] == 'split' and cfg['noised_prior']['live_kernel'] is None
    stage = _stages(cfg)[0]
    assert stage['name'] == 'train_prior' and stage['train_mode'] == 'bwd' and 'lr_schedule' not in stage, stage
    stage['max_steps'] = STEPS
    cfg.update(batch_size=P1_BATCH, fused_grad_accum_min_samples=P1_BATCH, epochs=STEPS, tag=TAG, run_name='p1_b250',
               **FIXED)
    return cfg


def x4_aug(files, source=X4_SOURCE):
    cfg = auglong_make.auglong(sg4.mle4(files), source)
    nf = cfg['model']['nf']
    assert nf['atom_tokens'] == 'split' and nf['sym_ops'] == 'pooled' and nf['rotate_frames'] and not nf['mirror_frames']
    assert nf['orient_grad_balance'] == 'norm' and nf['orient_grad_rho'] == 1.0 and nf['canonical_frame'] == 'none'
    plateau, anneal = _stages(cfg)
    assert plateau['max_steps'] == auglong_make.PLATEAU_CAP and anneal['max_steps'] == auglong_make.ANNEAL_STEPS
    assert plateau['exit'] == [{'metric': 'gates/heldout_flat', 'above': 0.5, 'patience': 1}], plateau['exit']
    assert anneal['on_exit'][-1] == 'stop', anneal['on_exit']
    cfg.update(batch_size=MLE4_BATCH, archive_period=5000, tag=TAG, run_name='x4_aug',
               epochs=auglong_make.PLATEAU_CAP + auglong_make.ANNEAL_STEPS + LOOP_MARGIN, **FIXED)
    return cfg


def cluster_files():
    d = sg4.DATA
    four = {'prior': f'{d}/{sg4.PRIOR}', 'conditions': f'{d}/{sg4.CONDITIONS}', 'test': f'{d}/{sg4.TEST}'}
    b = smooth.big
    one = {'prior': f'{b.DATA}/{smooth.PRIOR}', 'conditions': f'{b.DATA}/{b.CONDITIONS}', 'test': f'{b.DATA}/{b.TEST}'}
    assert one['conditions'] == four['conditions'], (one['conditions'], four['conditions'])
    return four, one


def sbatch():
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(ARMS)).replace('@NARMS@', str(len(ARMS) - 1))


def write(name, cfg, out=HERE):
    (out / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')


def localize(cfg, ckdir, steps, run):
    """A short CPU-buffer, small-batch run on the local K=0 prior (anchors only, so train_kinds anchor)."""
    cfg['noised_prior']['train_kinds'] = ['anchor']
    cfg.update(checkpoints_dir=ckdir, tag=f'{TAG}_local', run_name=run, smoke=True, cuda_memory_fraction=0.2,
               buffer_device='cpu', batch_size=64, fused_grad_accum_min_samples=64, eval_num_samples=400,
               test_eval_num_samples=200, eval_period=max(10, steps // 4), archive_period=10 ** 6, epochs=steps + 20,
               **FIXED)
    if cfg.get('zsg_panel'):
        cfg['zsg_panel'].update(molecules=8, draws=4, every_n_evals=2)
    return cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write local smoke configs into this directory')
    p.add_argument('--prior', default=None)
    p.add_argument('--conditions', default=None)
    p.add_argument('--test', default=None)
    p.add_argument('--ckdir', default=None, help='local smoke: checkpoints directory (reads --source, writes there)')
    p.add_argument('--source', default=None, help='local smoke: a four-group checkpoint file name in --ckdir')
    p.add_argument('--steps', type=int, default=80)
    a = p.parse_args()
    if a.local:
        assert a.prior and a.conditions and a.test and a.ckdir and a.source, '--local needs every file argument'
        assert not a.ckdir.upper().startswith('D:'), 'local smoke checkpoints stay off D:'
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        files = {'prior': a.prior, 'conditions': a.conditions, 'test': a.test}
        c = localize(v4_kick(files), a.ckdir, a.steps, 'v4_kick')
        _stages(c)[0]['max_steps'] = a.steps
        write('v4_kick', c, out)
        c = localize(p1_b250(files), a.ckdir, a.steps, 'p1_b250')
        _stages(c)[0]['max_steps'] = a.steps
        write('p1_b250', c, out)
        # x4_flat: the plateau gate fires (any 10-nat change counts as flat), then the anneal runs and stops
        # x4_cap: the gate cannot reach a verdict (its window is longer than the stage), so the stage ends at its hard cap
        for run, gate in (('x4_flat', {'evals': 2, 'nats': 10.0, 'smooth': 1}),
                          ('x4_cap', {'evals': 1000, 'nats': 0.15, 'smooth': 1})):
            c = localize(x4_aug(files, a.source), a.ckdir, a.steps, run)
            c['heldout_gate'].update(gate)
            c['eval_period'] = 8    # the gate's first possible verdict is at eval smooth + evals = 3, inside the stage
            st = _stages(c)
            st[0]['max_steps'], st[1]['max_steps'] = a.steps // 2, a.steps // 4
            st[1]['lr_schedule'][1]['steps'] = a.steps // 4
            write(run, c, out)
        print(f'local smoke configs in {out}')
    else:
        four, one = cluster_files()
        write('v4_kick', v4_kick(four))
        write('p1_b250', p1_b250(one))
        write('x4_aug', x4_aug(four))
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(), encoding='utf-8', newline='\n')
        print(f"{', '.join(ARMS)} and submit_{TAG}.sbatch in {HERE}")
