"""pool8sg4_nov20: the first conditional run over four space groups at once, P-1 (2), P2_1/c (14), P2_12_12_1 (19) and
Pbca (61), on the pool8 molecules (owner 2026-11-20: start training, watch the usual metrics plus Z(sg) and per-group
statistics). Switched eLJ (mk_dev analyze_kwargs lj_envelope 1.0); the four K=4 priors merged into one file by
prep_pool8sg4_nov20.sbatch.

Arms (one per card, run in turn):
  smoke   mle4's config, 300 steps, evals every 100: load RAM and VRAM, the landed batch, the step rate, and that every
          sg<g>/ key appears. Run first; read it before mle4.
  smoke_cpu  smoke with buffer_device cpu (every buffer and prior_dataset in host RAM): the fallback if smoke runs out
          of GPU memory (prior_dataset plus the static prior buffer are two GPU copies of ~40M rows). Its step rate
          against smoke's prices the fallback.
  mle4    pool8_big_nov15's unaugmented conditional MLE (the big model, usage-targeted batch growth), from a fresh policy,
          capped at MLE_STEPS. A single-group checkpoint cannot warm-start it: the group slot layout and the group
          embedding change the policy (checkpointing refuses).
  tb4     (make.py --tb SOURCE, after mle4) bigbuf's TB stage (pool8_tbrate_nov17: bootstrap_z:offset, pooled_center
          head, LR hold 0.2, fwd_rollout_every 5, replay max_size 200k) from an mle4 archive, weights only, TB_STEPS.

Changes from the single-group recipe, each needed by four groups:
  space_groups [2, 14, 19, 61], P-1 first (the dead-row probe reads the first prior rows against space_groups[0]).
  model.nf.sym_ops pooled: the level head and the flow context read the row's group (a mixed-group FD policy refuses
          to build otherwise). The norbornane P-1 + P2_1/c run (nov09) used the same.
  test_eval_num_samples 4000: ~1,000 held-out-molecule forward draws per group.
  zsg_panel: the Z(sg) readouts (head panel every eval, importance-sampling panel every few evals).
Conditions file: P-1's (one row per molecule; rollouts draw each molecule's group uniformly). Test file: uniform over
the four groups, the same 125 held-out molecules in each, at most 8 crystals per (molecule, group)
(configs/pool8sg_prior_nov17/test4_uniform.sbatch, owner 2026-11-20: a uniform draw).

Read (per group, sg<g>/ splits): eval_bwd/mle_heldout_rows, eval_fwd/jensen_z, Excess Energy Nats P50, Reasonable
Sample Fraction, bwd/mle; zsg/* panels. Per-group levels live on different FD charts, so read each group's curve over
time, not levels across groups.

usage: python configs/pool8sg4_nov20/make.py              (smoke, smoke_cpu, mle4 yamls, submit_pool8sg4_nov20.sbatch)
       python configs/pool8sg4_nov20/make.py --tb SOURCE  (tb4.yaml from an mle4 checkpoint file name)
       python configs/pool8sg4_nov20/make.py --local DIR --prior P --conditions C --test T   (local smoke configs)"""
import argparse
import copy
import importlib.util
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel / 'make.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


big = _load('big', 'pool8_big_nov15')
tbrate = _load('tbrate', 'pool8_tbrate_nov17')

TAG = 'pool8sg4_nov20'
GROUPS = [2, 14, 19, 61]
MLE_STEPS = 60000
SMOKE_STEPS = 300
TB_STEPS = 10000
DATA = '/scratch/mk8347/data/crystal_datasets/conditional'
PRIOR = 'pool8sg4_nov20/priors/pool8_4sg_train_k4_metric.pt'
CONDITIONS = 'pool8_nov09/priors/pool8_p1_train_conditions.pt'
TEST = 'pool8sg4_nov20/priors/pool8_4sg_test_w5.pt'   # uniform over the groups (configs/pool8sg_prior_nov17/test4_uniform.sbatch)
ZSG_PANEL = {'enabled': True, 'molecules': 64, 'draws': 32, 'every_n_evals': 4}   # Z(sg) readouts (train.py::Modeller.log_zsg_panels)


def mle4(files):
    cfg = big.config(files)
    cfg['space_groups'] = list(GROUPS)
    nf = cfg['model']['nf']
    assert nf['sym_ops'] == 'none', nf['sym_ops']
    nf.update(sym_ops='pooled', sym_ops_init=None)
    assert cfg['energy_config']['crystal_coordinates'] == 'fd'
    assert cfg['energy_config']['analyze_kwargs'].get('lj_envelope') == 1.0, cfg['energy_config']['analyze_kwargs']
    # buffer_device cpu: the GPU placement ran out of memory seeding the anchor buffer (smoke poq5fwwy, mle4 qicjg8lf,
    # 2026-11-20: prior_dataset 33 GB live, 76.5 GB allocated); smoke_cpu q0c73xp4 ran 300 steps at 2.2 s/step, batch 2225
    cfg.update(test_eval_num_samples=4000, epochs=MLE_STEPS, archive_period=5000, tag=TAG, run_name='mle4',
               buffer_device='cpu')
    if ZSG_PANEL is not None:
        cfg['zsg_panel'] = dict(ZSG_PANEL)
    stage = cfg['protocols'][cfg['protocol']]['stages'][0]
    stage['max_steps'] = MLE_STEPS
    return cfg


def smoke(files):
    cfg = mle4(files)
    cfg.update(epochs=SMOKE_STEPS, eval_period=100, archive_period=10 ** 6, smoke=True, run_name='smoke',
               buffer_device='cuda')
    cfg['protocols'][cfg['protocol']]['stages'][0]['max_steps'] = SMOKE_STEPS
    return cfg


def smoke_cpu(files):
    """smoke with every buffer, prior_dataset included, in host RAM (buffer_device cpu): the fallback if the GPU cannot
    hold the four-group rows. Read its step rate against smoke's."""
    cfg = smoke(files)
    cfg.update(buffer_device='cpu', run_name='smoke_cpu')
    return cfg


def tb4(files, source):
    """bigbuf (pool8_tbrate_nov17) on the four-group base, from an mle4 checkpoint."""
    base = mle4(files)
    cfg = tbrate.control(base, source)
    vc = tbrate._stage(cfg)
    assert vc['fwd_rollout_every'] == 20, vc['fwd_rollout_every']
    vc['fwd_rollout_every'] = 5
    cfg['buffers']['replay_buffer']['max_size'] = tbrate.BIGBUF_ROWS
    cfg.update(epochs=TB_STEPS, tag=TAG, run_name='tb4')
    assert cfg['space_groups'] == GROUPS and cfg['model']['nf']['sym_ops'] == 'pooled'
    return cfg


def sbatch(arms):
    text = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    files = (PRIOR, CONDITIONS, TEST)
    return (text.replace('@TAG@', TAG).replace('@ARMS@', ' '.join(arms)).replace('@NARMS@', str(len(arms) - 1))
            .replace('@DATA@', DATA).replace('@FILES@', ' '.join(files)))


def write(name, cfg, out=HERE):
    (out / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tb', default=None, help='write tb4.yaml from this mle4 checkpoint file name')
    p.add_argument('--local', default=None, help='write local smoke configs into this directory')
    p.add_argument('--prior', default=None)
    p.add_argument('--conditions', default=None)
    p.add_argument('--test', default=None)
    p.add_argument('--ckdir', default='D:/crystal_datasets/cluster_pull_nov19/runs')
    p.add_argument('--steps', type=int, default=60)
    a = p.parse_args()
    if a.local:
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        files = {'prior': a.prior, 'conditions': a.conditions, 'test': a.test}
        cfg = smoke(files)
        cfg.update(checkpoints_dir=a.ckdir, tag=f'{TAG}_local', cuda_memory_fraction=0.4, epochs=a.steps,
                   batch_size=64, grow_batch_size=False, eval_period=max(20, a.steps // 2), eval_num_samples=400,
                   test_eval_num_samples=200)
        cfg['protocols'][cfg['protocol']]['stages'][0]['max_steps'] = a.steps
        write('smoke', cfg, out)
        print(f'local smoke config in {out}')
    else:
        files = {'prior': f'{DATA}/{PRIOR}', 'conditions': f'{DATA}/{CONDITIONS}', 'test': f'{DATA}/{TEST}'}
        if a.tb:
            write('tb4', tb4(files, a.tb))
            print(f'tb4.yaml in {HERE}')
        else:
            write('smoke', smoke(files))
            write('smoke_cpu', smoke_cpu(files))
            write('mle4', mle4(files))
            (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(['smoke', 'smoke_cpu', 'mle4', 'tb4']), encoding='utf-8',
                                                       newline='\n')
            print(f'smoke.yaml, mle4.yaml and submit_{TAG}.sbatch in {HERE}')
