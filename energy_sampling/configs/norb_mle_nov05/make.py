"""norb_mle_nov05: conditional MLE knob sweep on the norbornane P-1 eLJ set (molecule mode, FD rows), one knob per arm.

Base (every arm unless its own line changes it): the committed mk_dev plus the norbornane MLE recipe -- NF policy,
molecule conditioning (atom tokens, split encoder), prefix FD rows, the K100 metric-kick pre-noised prior's training
children (live_kernel null) inside a 10 kT window, the 117-row condition file, the 35 held-out molecules' minima within
10 kT as test_molecules_path, batch 500, mk_dev's LR schedule, MLE ONLY (no exit, 20k steps, then stop). Readouts:
eval_bwd/mle_heldout_gap (within molecules), eval_test/mle and eval_test/mle_gap (between molecules, against the
training anchors). The local reference run is norbornane_nov05_mle_p1_noised_w10_c (held-out minima at ~6k steps).

Four arms per A100 under MPS at cuda_memory_fraction 0.23 (a local batch-500 arm peaked at 13.9 GB).
usage: python configs/norb_mle_nov05/make.py   (writes <arm>.yaml and submit_norb_mle_nov05.sbatch beside this file)"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = '/scratch/mk8347/projects/gfn_cond/gfn-diffusion/energy_sampling'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional/norbornane_nov04/priors'
TAG = 'norb_mle_nov05'
STEPS = 20000
PER_GPU = 4
# the files every arm reads, with their sizes (the sbatch refuses a partial copy)
FILES = {'norbornane_p1_train_k100_metric.pt': 2328032453, 'norbornane_p1_train_conditions.pt': 332153,
         'norbornane_p1_test_w10.pt': 4378129,
         'norbornane_p1_train_k100_metric.pt.fdchart-80b2454bf1fe9638.pt': 46210165}

BASE = {
    'energy_config.crystal_coordinates': 'fd', 'model.kind': 'nf', 'model.nf.atom_tokens': True,
    'model.nf.row_tokens': 'prefix', 'molecule_conditioning': True, 'batch_size': 500, 'gradient_norm_clip': 10.0,
    'condition_log_z.rollout_condition_draw': 'cycle',
    'noised_prior.window_kT': 10.0, 'noised_prior.live_kernel': None, 'noised_prior.train_kinds': ['child'],
    'prior_path': f'{DATA}/norbornane_p1_train_k100_metric.pt',
    'molecules_path': f'{DATA}/norbornane_p1_train_conditions.pt',
    'test_molecules_path': f'{DATA}/norbornane_p1_test_w10.pt',
    'epochs': STEPS, 'eval_period': 250, 'figs_every_n_evals': 2, 'archive_period': 2500, 'archive_buffers': True,
    'checkpoints_dir': f'{ROOT}/checkpoints', 'cuda_memory_fraction': 0.23, 'tag': TAG, 'smoke': False,
}
LR_BASE = [{'to': 0.05, 'steps': 3000, 'kind': 'hold'}]   # mk_dev's hold; an LR arm changes where the ramp lands


def lr(final):
    return {'stage.lr_schedule': LR_BASE + [{'to': final, 'steps': 1000, 'kind': 'geometric'}]}


# arm -> (overrides, one line saying what it changes); GPU groups of PER_GPU, in this order
ARMS = {
    # reference and noise floor, learning rate (mk_dev: 0.05 x seed_lr held 3k steps, then to 0.2 x = 2.5e-5)
    'base': ({}, 'the base recipe'),
    'base_s2': ({'seed': 12346}, 'the base, another seed (the noise floor)'),
    'lr_x01': (lr(0.1), 'final LR 0.1 x seed_lr (1.25e-5)'),
    'lr_x04': (lr(0.4), 'final LR 0.4 x seed_lr (5e-5)'),
    'lr_x08': (lr(0.8), 'final LR 0.8 x seed_lr (1e-4)'),
    'wd_1e4': ({'use_weight_decay': True, 'weight_decay': 1e-4}, 'Adam weight decay 1e-4 (L2, every group)'),
    'wd_1e3': ({'use_weight_decay': True, 'weight_decay': 1e-3}, 'Adam weight decay 1e-3'),
    'batch_250': ({'batch_size': 250}, 'batch 250 (accumulated to the same 1000-row update)'),
    # molecule conditioner, and data
    'pooled_mol': ({'model.nf.atom_tokens': 'split_pooled'},
                   'molecule as pooled vectors only, no cross-attention to atoms'),
    'atom_d64': ({'model.nf.atom_node_dim': 64}, 'atom encoder node width 64 (base 128)'),
    'atom_d256': ({'model.nf.atom_node_dim': 256}, 'atom encoder node width 256'),
    'atom_c1': ({'model.nf.atom_convs': 1}, 'atom encoder 1 message-passing round (base 3)'),
    'atom_c5': ({'model.nf.atom_convs': 5}, 'atom encoder 5 rounds'),
    'window_15': ({'noised_prior.window_kT': 15.0}, 'data: minima within 15 kT (more minima per molecule; base 10)'),
    # AR model
    'tf_w128': ({'model.nf.tf_width': 128, 'model.nf.tf_heads': 4}, 'transformer width 128 (base 256)'),
    'tf_w512': ({'model.nf.tf_width': 512}, 'transformer width 512'),
    'tf_l2': ({'model.nf.tf_layers': 2}, 'transformer 2 layers (base 4)'),
    'tf_l8': ({'model.nf.tf_layers': 8}, 'transformer 8 layers'),
    'norm_rms': ({'model.nf.tf_norm': 'rms'}, 'RMS norm in the transformer (base layer norm)'),
    'ctx_256': ({'model.nf.context_dim': 256}, 'condition context width 256 (base 128)'),
    # rows and spline
    'rows_typed': ({'model.nf.row_tokens': 'typed'}, 'typed FD rows without the prefix physics features'),
    'bins_16': ({'model.nf.n_bins': 16}, 'spline 16 bins (base 32)'),
    'bins_64': ({'model.nf.n_bins': 64}, 'spline 64 bins'),
    # (two flow blocks are refused under typed rows: one block only)
    'lr_nohold': ({'stage.lr_schedule': [{'to': 0.2, 'steps': 1, 'kind': 'hold'}]},
                  'final LR 0.2 x seed_lr from step 0, no 3k-step warm hold'),
}


def put(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split('.')
    for k in path:
        if k not in node:
            raise KeyError(f'{dotted}: {k!r} is not a key of the base config')
        node = node[k]
    if last not in node:
        raise KeyError(f'{dotted}: {last!r} is not a key of the base config (mk_dev carries every live key)')
    node[last] = value


def arm_config(arm):
    cfg = yaml.safe_load((HERE.parent / 'mk_dev.yaml').read_text(encoding='utf-8'))
    over, _ = ARMS[arm]
    for k, v in {**BASE, **{k: v for k, v in over.items() if not k.startswith('stage.')}}.items():
        put(cfg, k, v)
    cfg['run_name'] = arm
    stages = cfg['protocols'][cfg['protocol']]['stages']
    assert stages[0]['name'] == 'train_prior', [s['name'] for s in stages]
    p1 = dict(stages[0])
    p1.pop('exit', None)                              # watch MLE: no exit; MLE is the only stage, ended at its cap
    p1['flags'] = dict(p1['flags'], scramble_conditions=False)   # conditional MLE trains the condition encoder
    p1['loss_coeffs'] = {'bwd': dict(p1['loss_coeffs']['bwd'], tb_z_source='persistent')}
    p1['max_steps'] = STEPS
    p1['on_exit'] = ['snapshot:phase1_exit', 'stop']
    if 'stage.lr_schedule' in over:
        p1['lr_schedule'] = over['stage.lr_schedule']
    cfg['protocols'][cfg['protocol']]['stages'] = [p1]
    return cfg


def sbatch(groups):
    cases = '\n'.join(f'    {i}) GROUP="{" ".join(g)}" ;;' for i, g in enumerate(groups))
    doc = '\n'.join(f'#   array {i}: ' + '; '.join(f'{a} ({ARMS[a][1]})' for a in g) for i, g in enumerate(groups))
    template = (HERE / 'submit_template.sbatch').read_text(encoding='utf-8')
    return (template.replace('@LAST@', str(len(groups) - 1)).replace('@TAG@', TAG)
            .replace('@N@', str(len(ARMS))).replace('@PER@', str(PER_GPU)).replace('@GROUPS_DOC@', doc)
            .replace('@DATA@', DATA).replace('@CASES@', cases).replace('@FILES@', ' '.join(FILES))
            .replace('@BYTES@', ' '.join(str(v) for v in FILES.values())))


if __name__ == '__main__':
    names = list(ARMS)
    for arm in names:
        cfg = arm_config(arm)
        (HERE / f'{arm}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8', newline='\n')
    groups = [names[i:i + PER_GPU] for i in range(0, len(names), PER_GPU)]
    (HERE / f'submit_{TAG}.sbatch').write_text(sbatch(groups), encoding='utf-8', newline='\n')
    print(f'{len(names)} arms in {len(groups)} jobs of {PER_GPU}')
    for i, g in enumerate(groups):
        print(i, g)
