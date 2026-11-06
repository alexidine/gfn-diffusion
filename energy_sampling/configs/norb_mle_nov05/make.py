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


# SLOTS ARE FIXED BY THE QUEUED SBATCH (array groups of PER_GPU, in this order), so a revision keeps the slot names
# and gives each slot's yaml its arm's own run_name (the W&B name). Revised 2026-11-06 before any job started, after the
# local runs showed capacity mostly shifts the overfit timescale: the battery now spends its arms on data sharpness
# (kick scale), data learning curves, spline sharpness, molecule controls, and only two capacity points.
# slot -> (overrides, one line saying what it changes, run_name)
CAP128 = {'model.nf.tf_width': 128, 'model.nf.tf_layers': 2, 'model.nf.tf_mlp_ratio': 1, 'model.nf.atom_convs': 2,
          'model.nf.context_hidden': 128, 'model.nf.level_hidden': 128}
CAP64 = {'model.nf.tf_width': 64, 'model.nf.tf_heads': 4, 'model.nf.tf_layers': 2, 'model.nf.tf_mlp_ratio': 1,
         'model.nf.atom_node_dim': 64, 'model.nf.atom_convs': 2, 'model.nf.context_dim': 64,
         'model.nf.context_hidden': 64, 'model.nf.level_hidden': 64}
KICK075 = {'prior_path': f'{DATA}/norbornane_p1_train_k100_kick075.pt'}
KICK133 = {'prior_path': f'{DATA}/norbornane_p1_train_k100_kick133.pt'}
ARMS = {
    # reference and noise floor; LR (mk_dev: 0.05 x seed_lr held 3k steps, then to 0.2 x = 2.5e-5)
    'base': ({}, 'the base recipe', 'base'),
    'base_s2': ({'seed': 12346}, 'the base, another seed (the noise floor)', 'base_s2'),
    'lr_x01': (lr(0.1), 'final LR 0.1 x seed_lr (1.25e-5)', 'lr_x01'),
    'lr_x04': (lr(0.4), 'final LR 0.4 x seed_lr (5e-5)', 'lr_x04'),
    # regularisation and spline sharpness
    'lr_x08': ({'use_weight_decay': True, 'weight_decay': 1e-2}, 'Adam weight decay 1e-2', 'wd_1e2'),
    'wd_1e4': ({'model.nf.min_bin_height': 1e-4}, 'spline bin-height floor 1e-4 (base 1e-7)', 'floor_1e4'),
    'wd_1e3': ({'use_weight_decay': True, 'weight_decay': 1e-3}, 'Adam weight decay 1e-3', 'wd_1e3'),
    'batch_250': ({'model.nf.min_bin_height': 1e-3}, 'spline bin-height floor 1e-3', 'floor_1e3'),
    # molecule controls and capacity
    'pooled_mol': ({'model.nf.atom_tokens': 'split_pooled'},
                   'molecule as pooled vectors only, no cross-attention to atoms', 'pooled_mol'),
    'atom_d64': ({'molecule_conditioning': False, 'model.nf.atom_tokens': False, 'model.nf.row_tokens': 'plain'},
                 'MOLECULE-BLIND: no conditioning, plain rows (the flow sees nothing molecule-specific)', 'blind_plain'),
    'atom_d256': ({'model.nf.row_tokens': 'plain'}, 'conditional with plain rows (the pair of blind_plain)', 'cond_plain'),
    'atom_c1': (CAP128, 'every width <= 128, 2 layers (transformer MLP and level head 128 too): 734k params',
                'cap128_l2'),
    'atom_c5': (CAP64, 'every width <= 64, 2 layers', 'cap64_l2'),
    'window_15': ({'noised_prior.window_kT': 15.0}, 'data: minima within 15 kT (more minima per molecule; base 10)',
                  'window_15'),
    # data sharpness: kick scale (the K100 children re-drawn at a larger radius; same seed, same held-out minima)
    'tf_w128': (KICK075, 'children kicked up to 0.75 A (10^U(-2, -0.125); base up to 0.42 A)', 'kick_075'),
    'tf_w512': (KICK133, 'children kicked up to 1.33 A (10^U(-2, 0.125))', 'kick_133'),
    'tf_l2': ({'model.nf.tf_layers': 2}, 'transformer 2 layers (base 4)', 'tf_l2'),
    'tf_l8': ({**KICK133, **CAP128}, 'kick_133 and cap128_l2 together', 'kick133_cap128'),
    # data learning curves (nested subsets of the base prior; the 35 held-out molecules unchanged)
    'norm_rms': ({'prior_path': f'{DATA}/norbornane_p1_train_k100_mol50.pt',
                  'molecules_path': f'{DATA}/norbornane_p1_train_conditions_mol50.pt'},
                 'data: 60 of the 117 molecules (their own condition file)',
                 'mol_50'),
    'ctx_256': ({'prior_path': f'{DATA}/norbornane_p1_train_k100_mol25.pt',
                  'molecules_path': f'{DATA}/norbornane_p1_train_conditions_mol25.pt'},
                 'data: 36 of the 117 molecules (their own condition file)',
                'mol_25'),
    # rows and spline
    'rows_typed': ({'model.nf.row_tokens': 'typed'}, 'typed FD rows without the prefix physics features',
                   'rows_typed'),
    'bins_16': ({'model.nf.n_bins': 16}, 'spline 16 bins (base 32)', 'bins_16'),
    'bins_64': ({'prior_path': f'{DATA}/norbornane_p1_train_k100_min50.pt'},
                'data: half of the minima of each molecule (its lowest kept)', 'minima_50'),
    'lr_nohold': ({'stage.lr_schedule': [{'to': 0.2, 'steps': 1, 'kind': 'hold'}]},
                  'final LR 0.2 x seed_lr from step 0, no 3k-step warm hold', 'lr_nohold'),
}
# the policy settings not yet in mk_dev (models/nf_policy.py NF_PENDING_DEFAULTS): set without the key check
PENDING_NF = ('tf_mlp_ratio', 'level_hidden')


def put(cfg, dotted, value):
    node = cfg
    *path, last = dotted.split('.')
    for k in path:
        if k not in node:
            raise KeyError(f'{dotted}: {k!r} is not a key of the base config')
        node = node[k]
    if last not in node and not (dotted.startswith('model.nf.') and last in PENDING_NF):
        raise KeyError(f'{dotted}: {last!r} is not a key of the base config (mk_dev carries every live key)')
    node[last] = value


def arm_config(arm):
    cfg = yaml.safe_load((HERE.parent / 'mk_dev.yaml').read_text(encoding='utf-8'))
    over, _, run_name = ARMS[arm]
    for k, v in {**BASE, **{k: v for k, v in over.items() if not k.startswith('stage.')}}.items():
        put(cfg, k, v)
    cfg['run_name'] = run_name
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
