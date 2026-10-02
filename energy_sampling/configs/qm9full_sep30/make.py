r"""qm9full_sep30 -- the conditional GFN on the full-QM9 prior, CLUSTER (owner 2026-09-30).

THE PRIOR. qm9full_{conditions,prior,test_conditions}.pt, written by build_qm9_full_prior.py from the qm9_full_sep29
search (P-1, Z'=1, raw eLJ): every standardized QM9 molecule, near-exact duplicates removed (envwise RDF < 0.01),
chunks 0-49 cut to their 10 most diverse crystals, a random 5% of molecules held out with all their crystals.

PHASE 1 (`python configs/qm9full_sep30/make.py p1`): train_prior only, ONE training the phase-2 arms share (two legs,
below) -- the Z fallback they differ in does not act in train_prior (tbc 0). Stopped by hand, as cl21_p1 was: the stage carries no
exit, and archive_period 5000 writes the candidate seeds. Built from cond_lam_sep21/p1.yaml, the first phase of the
chain the best conditional run (cond_tb_sep25 ctb25_extreme_l1) came from, with:
  - every *_hidden_dim 1024 except the log Z head (flow_hidden_dim 64, flow_layers 2: small, so a learned Z(c) can
    be tested without the 1.6M-parameter head that memorized); model.condition_embedding_dim 128 (was 16)
  - integrator.T 50 and eval_T 50 (was 25)
  - energy_config.energy_reference seed_min, so the head learns the level phase 2 is scored in
  - condition_log_z: untrusted_z global; half_life_visits 50 -- at ~124k molecules and 1000 rows a step a molecule is
    revisited every ~124 steps, so the default 200 visits is ~25k steps of memory; 50 keeps ~6k and stays above the
    28 that config_invariants records as the lowest value shown to survive
  - the cluster: /scratch paths, compile_policy false and cuda_memory_fraction 0.9 as the final_sep19 arms; a fresh
    first launch and self-resume afterwards (mle_nig_sep17's job script)

TWO LEGS, one INDEX row each; the job script's array is the second.
  [0] qf30_p1     from scratch at lr_control.fixed_scale 0.05 (the base's value: 0.05 x seed_lr 1.25e-4 = 6.25e-6).
  [1] qf30_p1lr2  the same config at fixed_scale 2 (2.5e-4, owner 2026-10-02), seeded weights-only from leg 0's
                  _best.pt. A NEW ARM, NOT A RESUBMIT OF LEG 0: a full resume restores the controller's scale from
                  the checkpoint (lr_ctrl), and fixed mode reads fixed_scale once, when burn-in ends
                  (LRController._open_bracket), so leg 0 resumed under a new fixed_scale keeps 0.05. A weights-only
                  load starts the step count, the optimizers and the controller fresh: burn_in_steps at
                  burn_in_scale, then the geometric ramp to fixed_scale.
                  A leg that promotes above its burn-in scale also takes the controller guard the hot MLE batteries
                  ran under (mle_sep09, mle_w3_sep16, mle_nig_sep17): hard_failure.loss_excursion_k 40 and
                  fire_cut_factor 1.0. The base's 10 and 0.5 are the pair prod_aug28/make.py measured: the bar is
                  fitted in burn-in, where the loss band is narrow, and held through the promotion, so the
                  promotion transient fires it and each fire halves the rate for the rest of the stage (5 of 20
                  prod_aug26 arms). And epochs 1,000,000: the step count restarts at 0 and the base's 100,000 would
                  end a stage that is stopped by hand.

PHASE 2, once the owner picks the seed: the extreme TB recipe from the seed, weights only, as two arms --
A untrusted_z global, B untrusted_z head (the small learned head). Not generated here yet.

    python configs/qm9full_sep30/make.py p1
"""
import copy
import importlib.util
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('nigmake', HERE.parent / 'mle_nig_sep17' / 'make.py')
nig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nig)
w3 = nig.w3

TAG = 'qf30'
BATTERY = 'qm9full_sep30'
P1_BASE = HERE.parent / 'cond_lam_sep21' / 'p1.yaml'
LOCAL_PRIORS = pathlib.Path(r'D:\crystal_datasets\conditional\priors')
PRIOR, CONDITIONS, TEST = 'qm9full_prior.pt', 'qm9full_conditions.pt', 'qm9full_test_conditions.pt'
WIDTH = 1024
HEAD_WIDTH, HEAD_LAYERS = 64, 2
COND_DIM = 128
T = 50
HALF_LIFE_VISITS = 50.0
PROTOCOL = 'conditional_vargrad'
WIDTH_KEYS = ('t_hidden_dim', 's_hidden_dim', 'policy_hidden_dim', 'cond_hidden_dim')
SEED_LR = 1.25e-4
BASE_SCALE = 0.05
# (run_name, lr_control.fixed_scale, run_name of the leg it seeds from weights-only or None)
LEGS = (('p1', BASE_SCALE, None),
        ('p1lr2', 2.0, 'p1'))
LIVE = 1    # the INDEX row the job script's array launches
# (hard_failure.loss_excursion_k, fire_cut_factor, epochs): the base's, and a promoting leg's
BASE_GUARD = (10.0, 0.5, 100_000)
HOT_GUARD = (40.0, 1.0, 1_000_000)


def _guard(cfg):
    lc = cfg['lr_control']
    return lc['hard_failure']['loss_excursion_k'], lc['fire_cut_factor'], cfg['epochs']


def build_p1(run_name, scale, warm):
    cfg = yaml.safe_load(P1_BASE.read_text(encoding='utf-8'))
    cfg['tag'], cfg['run_name'] = TAG, run_name
    cfg['checkpoints_dir'] = w3.CLUSTER_CKPTS
    cfg['checkpoint_name'] = w3.CK_PLACEHOLDER if warm else None
    cfg['prior_model_name'] = None
    cfg['load_weights_only'] = bool(warm)
    cfg['continue_from_checkpoint'] = w3.CONT_PLACEHOLDER
    lc = cfg['lr_control']
    assert (lc['fixed_scale'], lc['seed_lr']) == (BASE_SCALE, SEED_LR), (lc['fixed_scale'], lc['seed_lr'])
    assert _guard(cfg) == BASE_GUARD, _guard(cfg)
    lc['fixed_scale'] = scale
    if scale != lc['burn_in_scale']:
        lc['hard_failure']['loss_excursion_k'], lc['fire_cut_factor'], cfg['epochs'] = HOT_GUARD
    cfg['prior_path'] = f'{w3.CLUSTER_DATA}/{PRIOR}'
    cfg['molecules_path'] = f'{w3.CLUSTER_DATA}/{CONDITIONS}'
    cfg['test_molecules_path'] = f'{w3.CLUSTER_DATA}/{TEST}'
    cfg['compile_policy'] = False
    cfg['cuda_memory_fraction'] = 0.9
    m = cfg['model']
    for k in WIDTH_KEYS:
        m[k] = WIDTH
    m['flow_hidden_dim'], m['flow_layers'] = HEAD_WIDTH, HEAD_LAYERS
    m['condition_embedding_dim'] = COND_DIM
    cfg['integrator']['T'] = T
    cfg['eval_T'] = T
    cfg['energy_config']['energy_reference'] = 'seed_min'
    cl = cfg['condition_log_z']
    cl['untrusted_z'] = 'global'
    cl['global_half_life_updates'] = 50.0
    cl['half_life_visits'] = HALF_LIFE_VISITS
    # phase 1 only, run until stopped: the train_prior stage of the conditional protocol without its exit
    stages = cfg['protocols'][PROTOCOL]['stages']
    tp = [s for s in stages if s['name'] == 'train_prior']
    assert len(tp) == 1, [s['name'] for s in stages]
    tp = copy.deepcopy(tp[0])
    tp.pop('exit', None)
    tp.pop('on_exit', None)
    cfg['protocols'][PROTOCOL]['stages'] = [tp]
    cfg['protocol'] = PROTOCOL
    return cfg


def check_p1(cfg, name, scale, warm):
    m = cfg['model']
    assert all(m[k] == WIDTH for k in WIDTH_KEYS), {k: m[k] for k in WIDTH_KEYS}
    assert (m['flow_hidden_dim'], m['flow_layers'], m['condition_embedding_dim']) == (HEAD_WIDTH, HEAD_LAYERS, COND_DIM)
    assert cfg['integrator']['T'] == T == cfg['eval_T']
    assert cfg['embedding_conditioning'] is True and cfg['embedding_conditioning_dim'] == 192
    assert cfg['space_groups'] == [2] and cfg['energy_function'] == 'elj'
    assert cfg['energy_config']['energy_reference'] == 'seed_min'
    assert cfg['condition_log_z']['untrusted_z'] == 'global'
    for key, name in (('prior_path', PRIOR), ('molecules_path', CONDITIONS), ('test_molecules_path', TEST)):
        assert cfg[key] == f'{w3.CLUSTER_DATA}/{name}', (key, cfg[key])
    st = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in st] == ['train_prior'] and 'exit' not in st[0] and st[0]['train_mode'] == 'bwd'
    assert st[0]['loss_coeffs']['bwd']['tbc'] == 0.0, 'the shared phase 1 must not train through the Z fallback'
    assert cfg['continue_from_checkpoint'] == w3.CONT_PLACEHOLDER, name
    if warm:
        assert cfg['checkpoint_name'] == w3.CK_PLACEHOLDER and cfg['load_weights_only'] is True, name
    else:
        assert cfg['checkpoint_name'] is None and cfg['load_weights_only'] is False, name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['seed_lr'] == SEED_LR and lc['fixed_scale'] == scale, name
    # fixed_scale acts on the rate train_prior steps (lr_back, managed when 'auto') and no rail holds it
    assert st[0]['train_mode'] == 'bwd' and cfg['lr_back'] == 'auto' and cfg.get('max_lr') is None, name
    assert _guard(cfg) == (BASE_GUARD if scale == lc['burn_in_scale'] else HOT_GUARD), (name, _guard(cfg))
    w3._scan_local_paths(cfg, name)


def main(argv):
    if argv[:1] != ['p1']:
        sys.exit(__doc__)
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_local = LOCAL_PRIORS / PRIOR
    assert prior_local.exists(), f'{prior_local} missing: build it (build_qm9_full_prior.py) before generating'
    prior_bytes = prior_local.stat().st_size
    (HERE / 'joblogs').mkdir(exist_ok=True)
    (HERE / 'joblogs' / '.gitkeep').write_text('ships this directory to the cluster; SLURM cannot create --output\n',
                                               encoding='utf-8')
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS]
    # the job script finds a leg's files by `*<arm>_*`: no arm name may match another's
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    rows, identity = [], {}
    for i, (run_name, scale, warm) in enumerate(LEGS):
        name = names[i]
        cfg = build_p1(run_name, scale, warm)
        check_p1(cfg, name, scale, warm)
        w3.load_check(cfg, name)
        identity[run_name] = w3.problem_def(cfg)
        if warm:
            # a weights-only load refuses a seed saved under another problem identity
            assert names.index(f'{TAG}_{warm}') < i and identity[warm] == identity[run_name], name
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append(f"{name}\tqm9full\t{'warm' if warm else 'fresh'}\t{f'{TAG}_{warm}' if warm else '-'}\t{PRIOR}"
                    f"\t{prior_bytes}\n")
        print(f"[{i}]{' <- the array' if i == LIVE else ''} {name}: lr_control.fixed_scale {scale:g} "
              f"({scale * SEED_LR:g}), {f'weights-only from {TAG}_{warm} _best.pt' if warm else 'from scratch'}")
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        f.writelines(rows)
    array = '#SBATCH --array=0-__LAST__'
    seeds = 'seeds weights-only from the mle09 _best.pt (warm)'
    old = '# __BATTERY__: phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.'
    assert all(nig.SBATCH.count(s) == 1 for s in (array, seeds, old)), 'the mle_nig_sep17 job script moved'
    sb = (nig.SBATCH.replace(array, f'#SBATCH --array={LIVE}-{LIVE}')
          .replace(seeds, "seeds weights-only from the warm_src arm's _best.pt (warm)")
          .replace(old, f'# __BATTERY__: phase 1 (train_prior) of the conditional GFN on the full-QM9 prior; row 0 '
                        f'from scratch, row {LIVE} weights-only from row 0; stopped by hand, archives every 5000 '
                        f'steps are the phase-2 seeds.')
          .replace('__TAG__', TAG).replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    with (HERE / f'submit_{BATTERY}.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    print(f'every leg: width {WIDTH} (log Z head {HEAD_WIDTH} x {HEAD_LAYERS}), condition dim {COND_DIM}, T {T}, '
          f'energy_reference seed_min, untrusted_z global, half_life_visits {HALF_LIFE_VISITS:g}')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


if __name__ == '__main__':
    main(sys.argv[1:])
