"""pool8_mle_nov10: conditional MLE on the pool8 P-1 set -- every pool v1 molecule up to 8 heavy atoms with a standard-
frame fixed point (72,464), 32 eLJ minima each (banded farthest-point picks, MXtalTools configs/crystal_searches/
pool8_nov09), pre-noised by data_processing/build_noised_prior.py. One knob per arm against the base.

Base: the committed mk_dev plus the norbornane MLE recipe (configs/norb_mle_nov05/make.py BASE) -- NF policy, molecule
conditioning (atom tokens), prefix FD rows, the prior's training children inside a 10 kT window, batch 500, mk_dev's
LR schedule, MLE ONLY (no exit, STEPS steps, then stop). Readouts: eval_test/mle and eval_test/mle_gap (held-out
molecules, against the training anchors), eval_bwd/mle_heldout_gap (held-out crystals of training molecules).

Files (DATA), all built on the cluster by configs/pool8_prior_nov09/build_pool8_prior.sbatch:
  PRIOR       pool8_p1_train_k<K>_metric.pt (build_noised_prior.py; its children train, its anchors are the
              training anchors)
  CONDITIONS  data_processing/molecule_split_files.py conditions PRIOR CONDITIONS
  TEST        data_processing/molecule_split_files.py heldout pool8_p1_heldout_anchors.pt pool8_p1_test 5
              --per_molecule 8 --max_molecules 500
K is None until the owner sets it; make.py then refuses to write the sbatch.

usage: python configs/pool8_mle_nov10/make.py            (writes <arm>.yaml and submit_pool8_mle_nov10.sbatch here)
       python configs/pool8_mle_nov10/make.py --local DIR (every arm on the local norbornane files, into DIR: a
                                                           recipe check, not this battery)"""
import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/priors'
TAG = 'pool8_mle_nov10'
STEPS = 50000
# the prior's children per minimum (configs/pool8_prior_nov09/build_pool8_prior.sbatch); None until the owner sets it
K = 4
# arms per A100 job. Peak trainer RAM measured locally (2026-11-09, norbornane P-1 priors of 0.79 and 2.33 GB): about
# 2.9 GB + 2.0 x the prior file. pool8 files are ~1.1 KB per row, 2.32M minima x (K + 1) rows: K 4 -> ~28 GB per arm
# (4 per 192 GB node), K 10 -> ~58 GB (2 per node).
PER_GPU = None if K is None else (4 if K <= 4 else 2)
# the files every arm reads, all built on the cluster by build_pool8_prior.sbatch (-1: the sbatch checks each exists
# and is not empty)
PRIOR = None if K is None else f'pool8_p1_train_k{K}_metric.pt'
CONDITIONS = 'pool8_p1_train_conditions.pt'
TEST = 'pool8_p1_test_w5.pt'
LOCAL = {'prior': 'D:/crystal_datasets/conditional/norbornane_nov04/priors/norbornane_p1_train_k100_metric.pt',
         'conditions': 'D:/crystal_datasets/conditional/norbornane_nov04/priors/norbornane_p1_train_conditions.pt',
         'test': 'D:/crystal_datasets/conditional/norbornane_nov04/priors/norbornane_p1_test_w10.pt'}

BASE = {
    'energy_config.crystal_coordinates': 'fd', 'model.kind': 'nf', 'model.nf.atom_tokens': True,
    'model.nf.row_tokens': 'prefix', 'molecule_conditioning': True, 'batch_size': 500, 'gradient_norm_clip': 10.0,
    'condition_log_z.rollout_condition_draw': 'cycle',
    'noised_prior.window_kT': 10.0, 'noised_prior.live_kernel': None, 'noised_prior.train_kinds': ['child'],
    'epochs': STEPS, 'eval_period': 500, 'figs_every_n_evals': 4, 'archive_period': 5000, 'archive_buffers': False,
    'checkpoints_dir': f'{ROOT}/checkpoints', 'cuda_memory_fraction': 0.23, 'tag': TAG, 'smoke': False,
}
LR_BASE = [{'to': 0.05, 'steps': 3000, 'kind': 'hold'}]   # mk_dev's hold; an LR arm changes where the ramp lands


def lr(final):
    return {'stage.lr_schedule': LR_BASE + [{'to': final, 'steps': 1000, 'kind': 'geometric'}]}


ROTMIR = {'model.nf.rotate_frames': True, 'model.nf.mirror_frames': True}
CAP_UP = {'model.nf.tf_width': 512, 'model.nf.tf_layers': 6, 'cuda_memory_fraction': 0.30}
CAP_DOWN = {'model.nf.tf_width': 128, 'model.nf.tf_layers': 2, 'model.nf.tf_mlp_ratio': 1, 'model.nf.atom_convs': 2,
            'model.nf.context_hidden': 128, 'model.nf.level_hidden': 128}
# arm -> (overrides, one line saying what it changes). Groups of PER_GPU in this order share an A100 under MPS; their
# cuda_memory_fraction must sum to <= 1 (checked below).
ARMS = {
    'base': ({}, 'the base recipe'),
    'base_s2': ({'seed': 12346}, 'the base, another seed (the noise floor)'),
    'rotmir': (ROTMIR, 'rotation + mirror frame augmentation'),
    'cap_up': (CAP_UP, 'transformer width 512, 6 layers (base 256, 4)'),
    'cap_up_rotmir': ({**CAP_UP, **ROTMIR}, 'width 512, 6 layers, with rotation + mirror augmentation'),
    'lr_x04': (lr(0.4), 'final LR 0.4 x seed_lr (base 0.2 x)'),
    'steps_long': ({'epochs': 2 * STEPS}, f'{2 * STEPS} steps (base {STEPS})'),
    'cap_down': (CAP_DOWN, 'every width <= 128, 2 layers'),
}
# settings not yet in mk_dev, set without the key check (models/nf_policy.py NF_PENDING_DEFAULTS)
PENDING = ('model.nf.tf_mlp_ratio', 'model.nf.level_hidden', 'model.nf.rotate_frames', 'model.nf.mirror_frames')


def put(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split('.')
    for k in path:
        if k not in node:
            raise KeyError(f'{dotted}: {k!r} is not a key of the base config')
        node = node[k]
    if last not in node and dotted not in PENDING:
        raise KeyError(f'{dotted}: {last!r} is not a key of the base config (mk_dev carries every live key)')
    node[last] = value


def arm_config(arm, files):
    cfg = yaml.safe_load((HERE.parent / 'mk_dev.yaml').read_text(encoding='utf-8'))
    over, _ = ARMS[arm]
    paths = {'prior_path': files['prior'], 'molecules_path': files['conditions'],
             'test_molecules_path': files['test']}
    for k, v in {**BASE, **paths, **{k: v for k, v in over.items() if not k.startswith('stage.')}}.items():
        put(cfg, k, v)
    cfg['run_name'] = arm
    stages = cfg['protocols'][cfg['protocol']]['stages']
    assert stages[0]['name'] == 'train_prior', [s['name'] for s in stages]
    p1 = dict(stages[0])
    p1.pop('exit', None)                              # MLE only: no exit, ended at its cap
    p1['flags'] = dict(p1['flags'], scramble_conditions=False)   # conditional MLE trains the condition encoder
    p1['loss_coeffs'] = {'bwd': dict(p1['loss_coeffs']['bwd'], tb_z_source='persistent')}
    p1['max_steps'] = int(cfg['epochs'])
    p1['on_exit'] = ['snapshot:phase1_exit', 'stop']
    if 'stage.lr_schedule' in over:
        p1['lr_schedule'] = over['stage.lr_schedule']
    cfg['protocols'][cfg['protocol']]['stages'] = [p1]
    return cfg


def groups(per=None):
    per = per or PER_GPU or 4
    names = list(ARMS)
    out = [names[i:i + per] for i in range(0, len(names), per)]
    for g in out:
        share = sum(ARMS[a][0].get('cuda_memory_fraction', BASE['cuda_memory_fraction']) for a in g)
        assert share <= 1.0 + 1e-9, f'group {g}: cuda_memory_fraction sums to {share:.2f} > 1'
    return out


def sbatch(gs):
    files = {PRIOR: -1, CONDITIONS: -1, TEST: -1}
    cases = '\n'.join(f'    {i}) GROUP="{" ".join(g)}" ;;' for i, g in enumerate(gs))
    doc = '\n'.join(f'#   array {i}: ' + '; '.join(f'{a} ({ARMS[a][1]})' for a in g) for i, g in enumerate(gs))
    template = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (template.replace('@LAST@', str(len(gs) - 1)).replace('@TAG@', TAG)
            .replace('@N@', str(len(ARMS))).replace('@PER@', str(PER_GPU)).replace('@GROUPS_DOC@', doc)
            .replace('@DATA@', DATA).replace('@CASES@', cases).replace('@FILES@', ' '.join(files))
            .replace('@BYTES@', ' '.join(str(v) for v in files.values())))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--local', default=None, help='write every arm on the local norbornane files into this directory')
    a = p.parse_args()
    gs = groups()
    if a.local:
        out = Path(a.local)
        out.mkdir(parents=True, exist_ok=True)
        for arm in ARMS:
            cfg = arm_config(arm, LOCAL)
            cfg.update(checkpoints_dir='D:/crystal_datasets/gfn_checkpoints', tag=f'{TAG}_local', smoke=True)
            (out / f'{arm}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
        print(f'{len(ARMS)} local arms in {out}')
    else:
        if K is None:
            raise SystemExit('K is None: set it to the children per minimum of the prior once the owner chooses it')
        files = {'prior': f'{DATA}/{PRIOR}', 'conditions': f'{DATA}/{CONDITIONS}', 'test': f'{DATA}/{TEST}'}
        for arm in ARMS:
            (HERE / f'{arm}.yaml').write_text(yaml.safe_dump(arm_config(arm, files), sort_keys=False),
                                              encoding='utf-8', newline='\n')
        (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(gs), encoding='utf-8', newline='\n')
        print(f'{len(ARMS)} arms in {len(gs)} jobs of {PER_GPU}')
    for i, g in enumerate(gs):
        print(i, g)
