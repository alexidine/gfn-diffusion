"""nsym_sep30 -- fine-tuning on the normaliser-symmetrised MIPCAS eLJ prior at 2x, 5x and 10x the usual anchor noise
(owner 2026-09-30: "symmetrized prior runs on mip+ELJ, at 2x, 5x, 10x our usual anchor noise"; "a weights-only
restart from our best current model").

    python configs/nsym_sep30/make.py

THE ARMS: ns30_mip_x2 / _x5 / _x10 = p23_mip_ft's config (the prod_sep20 production shape: every 5th step, P_B frozen,
replay pinned 0.3 / bwd 0.7, tau 600, rate 0.5, batch 1600) with
  * prior_path and molecules_path -> PRIOR_NEW, the symmetrised prior (data_processing/symmetrize_prior.py, layout
    `orbits` thinned at 0.01: every row of `prior` and `equalized_prior` written as all the descriptions the trainer
    can build -- its four half-cell y/z shifts, plus the opposite x face for a row sitting on one -- then leader-
    clustered at radius 0.01 in the wrapped latent (0.4 of a thermal kick) so that no two rows sit closer than that:
    the over-represented dense spots are capped and the rest of the file is untouched (74% of rows had no neighbour
    at that radius). Handedness +1 throughout; every written row rescored through the trainer's analyze call against
    its source; y/z orbits asserted whole. Owner 2026-09-30: "thin out over-representation within very high density
    latent regions -- that's it").
  * a WEIGHTS-ONLY first launch from p23_mip_ft (prior_path is part of the problem identity;
    warm_start_ignore_problem_keys: [prior_path] exempts it on the weights-only path and nowhere else). It carries the
    weights and the frozen P_B snapshot; optimiser, buffers, log Z bootstrap and the step count start fresh. The yaml
    holds load_weights_only: WEIGHTS_ONLY_PLACEHOLDER and the sbatch writes true on the first launch and false on a
    resubmission, which then resumes the arm's own checkpoint in full.
  * buffers.anchor_buffer.noise_log_range shifted up by log10(factor): the churn's log-uniform isotropic latent noise,
    [-2.5, -1.5] in mk_dev, i.e. magnitudes 0.003-0.032 against a measured 1 kT kick of ~0.025.
  * buffers.anchor_buffer.max_size = the row count, so no path can trim the seed (the buffer is frozen and its thin
    pass is off, so nothing does today; the seed is every prior-dataset row).
  * the box and reduction penalties at p23's ramp target 1000 as plain config values, no coeff_schedule.
The entry is the production one: the train_prior stub exits at the first eval (step 50 of a fresh step count) and
equilibration's on_enter rebuilds the prior buffer by churn from the symmetrised anchors at the arm's noise, bootstraps
log Z from 4000 rollouts and keeps the restored P_B snapshot (freeze_pb:full on an installed snapshot is a no-op).

SYMMETRY UNDER NOISE: the images differ from their source by latent shifts in y/z and a swap of the two x faces, all
latent isometries, so isotropic latent noise on a symmetrised anchor set is symmetric in distribution at any amplitude.

SEED: the sbatch takes p23_mip_ft's newest 5000-step archive, or its _running.pt with SRC_RUNNING=1 (the arm must not
be running then). PRIOR_NEW must be on the cluster at the production priors directory with exactly PRIOR_NEW_BYTES.

READ: the trainlog's 'warm start EXEMPTING [prior_path]' line; prior_buffer_length (62.5k after the rebuild);
bwd/under_coverage and bwd/tb_err (the buffer law changed, so neither is comparable with p23's level);
fwd/log_Z_learned vs p23's 35.2; eval_fwd/tb_err; zmatch/delta_mean; Mean Sample Energy; fwd/box_contact_frac.
"""
import copy
import importlib.util
import math
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('p20make', ROOT / 'prod_sep20' / 'make.py')
p20 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p20)
fin = p20.fin

TAG = 'ns30'
BATTERY = 'nsym_sep30'
FAM = 'mip'
N = 5
SRC_ARM = 'p23_mip_ft'
SRC_YAML = 'prod_sep23_ft/p23_mip_ft.yaml'
PRIOR_OLD = 'mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2.pt'
PRIOR_NEW = 'mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2_nsym_t010.pt'
PRIOR_NEW_BYTES = 627_551_917   # from the 2026-09-30 build (the sbatch refuses a file of any other size)
PRIOR_NEW_ROWS = 791_068     # equalized_prior rows after the 0.01 thinning of the 8-image orbits
LOCAL_PRIOR = pathlib.Path('D:/crystal_datasets/conditional/priors') / PRIOR_NEW
NOISE_BASE = [-2.5, -1.5]
FACTORS = {'x2': 2.0, 'x5': 5.0, 'x10': 10.0}
PENALTY = 1000.0
WO_PLACEHOLDER = 'WEIGHTS_ONLY_PLACEHOLDER'
#: every path this battery means to differ from p23_mip_ft on
INTENDED = {'run_name', 'tag', 'prior_path', 'molecules_path', 'load_weights_only', 'warm_start_ignore_problem_keys',
            'energy_config.bounding_coeff', 'energy_config.reduction_coeff', 'buffers.anchor_buffer.noise_log_range',
            'buffers.anchor_buffer.max_size', 'protocols.unconditional_tb.stages[1].coeff_schedule'}


def noise_range(factor):
    return [round(v + math.log10(factor), 4) for v in NOISE_BASE]


def build(base, label, factor):
    p20.TAG = TAG
    name, cfg = p20.build_arm(base, FAM, N)
    name = f'{TAG}_{FAM}_{label}'
    cfg['run_name'] = name
    ec = cfg['energy_config']
    assert ec['bounding_coeff'] == 10.0 == ec['reduction_coeff'], name
    ec['bounding_coeff'] = PENALTY
    ec['reduction_coeff'] = PENALTY
    production = copy.deepcopy(cfg)          # the production shape, checked by p20.check below
    assert cfg['prior_path'].endswith('/' + PRIOR_OLD) and cfg['molecules_path'] == cfg['prior_path'], name
    cfg['prior_path'] = cfg['molecules_path'] = cfg['prior_path'][:-len(PRIOR_OLD)] + PRIOR_NEW
    cfg['load_weights_only'] = True
    cfg['warm_start_ignore_problem_keys'] = ['prior_path']
    ab = cfg['buffers']['anchor_buffer']
    assert ab['noise_log_range'] == NOISE_BASE and ab['tile'] == 'iso' and ab['seed_source'] == 'prior_dataset', name
    assert ab['frozen'] is True and not ab['thin_every_n_evals'], name + ': the anchor buffer can thin or churn'
    ab['noise_log_range'] = noise_range(factor)
    ab['max_size'] = PRIOR_NEW_ROWS
    return name, cfg, production


def _diff(a, b, path=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f'{path}.{k}' if path else str(k)
            if k not in a:
                out.append(('added', p))
            elif k not in b:
                out.append(('removed', p))
            else:
                out += _diff(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a and all(isinstance(x, dict) for x in a + b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f'{path}[{i}]')
    elif a != b:
        out.append(('changed', path))
    return out


def check(cfg, name, production, factor):
    p20.check(production, name, FAM, N)
    seed = fin.load(SRC_YAML)
    a, b = dict(fin.w3.problem_def(cfg)), dict(fin.w3.problem_def(seed))
    assert a.pop('prior_path') != b.pop('prior_path') and a == b, name + ': the identity differs from p23_mip_ft beyond prior_path'
    assert cfg['prior_path'] == cfg['molecules_path'] and cfg['prior_path'].rsplit('/', 1)[1] == PRIOR_NEW, name
    assert cfg['load_weights_only'] is True and cfg['warm_start_ignore_problem_keys'] == ['prior_path'], name
    assert cfg['continue_from_checkpoint'] is False and cfg['checkpoint_name'] == fin.PLACEHOLDER, name
    ec = cfg['energy_config']
    assert ec['bounding_coeff'] == PENALTY == ec['reduction_coeff'], name
    ab = cfg['buffers']['anchor_buffer']
    lo, hi = ab['noise_log_range']
    assert abs(10 ** lo / 10 ** NOISE_BASE[0] - factor) < 1e-3 and abs(10 ** hi / 10 ** NOISE_BASE[1] - factor) < 1e-3, name
    assert ab['max_size'] == PRIOR_NEW_ROWS and cfg['buffers']['prior_buffer']['source'] == 'anchors', name
    st = cfg['protocols']['unconditional_tb']['stages']
    assert [s['name'] for s in st] == ['train_prior', 'equilibration'] and not st[1].get('coeff_schedule'), name
    assert st[1]['on_enter'] == ['rebuild_prior_by_churn', 'bootstrap_z:rollout:4000', 'freeze_pb:full'], name
    # what differs from p23_mip_ft: the intended paths, and whatever the committed mk_dev gained or changed since
    d = _diff(seed, cfg)

    def intended(p):
        return any(p == q or p.startswith(q + '.') for q in INTENDED)

    missing = sorted(q for q in INTENDED if not any(p == q or p.startswith(q + '.') for _, p in d))
    assert not missing, f'{name}: meant to differ from {SRC_ARM} on {missing} and does not'
    return sorted((kind, p) for kind, p in d if not intended(p))


def main(argv):
    assert PRIOR_NEW_BYTES > 0, 'PRIOR_NEW_BYTES is not filled in: build the prior first (data_processing/symmetrize_prior.py)'
    if LOCAL_PRIOR.exists():
        assert LOCAL_PRIOR.stat().st_size == PRIOR_NEW_BYTES, (LOCAL_PRIOR.stat().st_size, PRIOR_NEW_BYTES)
    dirty = fin.w3.dirty_files()
    if dirty:
        print('WARNING: the working tree is dirty (base read from git HEAD; the cluster runs HEAD):\n  ' + '\n  '.join(dirty))
    base = fin.committed_mk_dev()
    arms, drift = {}, None
    for label, factor in FACTORS.items():
        name, cfg, production = build(base, label, factor)
        extra = check(cfg, name, production, factor)
        assert drift is None or drift == extra, name
        drift = extra
        for weights_only in (True, False):          # the first launch and a resubmission, as the sbatch resolves them
            probe = copy.deepcopy(cfg)
            probe['load_weights_only'] = weights_only
            fin.load_check(probe, name, ['train_prior', 'equilibration'])
        cfg['load_weights_only'] = WO_PLACEHOLDER
        arms[name] = (cfg, factor)
    for stale in HERE.glob(f'{TAG}_*.yaml'):
        stale.unlink()
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    for name, (cfg, _) in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    fin._write_index(HERE / 'INDEX_a.tsv', [(name, FAM, 'switch', SRC_ARM, PRIOR_NEW, str(PRIOR_NEW_BYTES)) for name in arms])
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(wall=fin.WALL, last=len(arms) - 1, tag=TAG, battery=BATTERY, leg='a',
                                  ckpts=fin.w3.CLUSTER_CKPTS, data=fin.w3.CLUSTER_DATA, seed_block=fin.SEED_B,
                                  what=f'fine-tuning on the normaliser-symmetrised MIPCAS eLJ prior ({PRIOR_NEW}, 8-image orbits thinned at 0.01) at 2x / 5x / 10x anchor noise: weights-only first launch from {SRC_ARM}, full resume afterwards.'))
    for i, (name, (cfg, factor)) in enumerate(arms.items()):
        lo, hi = cfg['buffers']['anchor_buffer']['noise_log_range']
        print(f"[{i}] {name:<12} anchor noise x{factor:g}: log10 range [{lo}, {hi}] = latent {10 ** lo:.4f} to {10 ** hi:.4f}; "
              f"penalties {PENALTY:g}; weights-only first launch from *{SRC_ARM}_*")
    print(f'differences from {SRC_ARM} beyond the intended ones (the committed mk_dev moved since it was generated): {len(drift)}')
    for kind, p in drift:
        print(f'    {kind:<8} {p}')
    print(f'the prior must be at {fin.w3.CLUSTER_DATA}/{PRIOR_NEW} with {PRIOR_NEW_BYTES:,} bytes before submitting')


if __name__ == '__main__':
    main(sys.argv[1:])
