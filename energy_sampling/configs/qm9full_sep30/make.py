r"""qm9full_sep30 -- the conditional GFN on the full-QM9 prior, CLUSTER (owner 2026-09-30).

THE PRIOR. qm9full_{conditions,prior,test_conditions}.pt, written by build_qm9_full_prior.py from the qm9_full_sep29
search (P-1, Z'=1, raw eLJ): every standardized QM9 molecule, near-exact duplicates removed (envwise RDF < 0.01),
chunks 0-49 cut to their 10 most diverse crystals, a random 5% of molecules held out with all their crystals.

PHASE 1 (`python configs/qm9full_sep30/make.py p1`): train_prior only, ONE training the phase-2 arms share (two legs,
below) -- the Z fallback they differ in does not act in train_prior (tbc 0). Stopped by hand, as cl21_p1 was: the
stage carries no exit, and archive_period 5000 writes the candidate seeds. Built from cond_lam_sep21/p1.yaml, the
first phase of the chain the best conditional run (cond_tb_sep25 ctb25_extreme_l1) came from, with:
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

PHASE 2 (`python configs/qm9full_sep30/make.py arms`; owner 2026-10-03, phase 1 "visually converged"): the extreme TB
recipe (cond_tb_sep25/ctb25_extreme_l1_cont3.yaml: lambda 1, tb 1 on every branch against the persistent per-condition
Z, forward and backward at 0.5 / 0.5 with no replay share, the forward branch's reward gradient on its last step, P_B
frozen at entry, the log Z head regressed onto the tracker's trusted estimates) seeded from phase-1 leg 1, weights only
on a first launch, as two arms that differ in one key:
  qf30_tbg  condition_log_z.untrusted_z global   a row below min_visits is scored against one all-condition level
  qf30_tbh  condition_log_z.untrusted_z head     ... against the small learned head, which the TB residual then trains
Held-out molecules are never visited, so their eval rows always take the fallback: eval_test/tb_err against
eval_fwd/tb_err is the test of whether a learned Z(c) generalises. On top of the recipe:
  - the model, integrator, data files and cluster keys are phase 1's (the seed's architecture and problem; the problem
    identity is asserted equal to the seed leg's)
  - the stub train_prior of final_sep19 (no skip_if, an exit that holds at the first metric write, on_exit
    snapshot_prior) and no prior model
  - energy_reference seed_min; half_life_visits 50; untrusted_z as the arm
  - lr_control.fixed_scale 0.2 (2.5e-5). At width 512 the recipe's first leg ran 0.2 and its last ran 0.5 to step
    28,590 with lr_ctrl/divergences 0 (their W&B summaries); 0.2 is that last rate halved for the doubled width, the
    rule that gave phase 1 its 2.5e-4. THE RATE IS FIXED AT LAUNCH: a resume keeps the checkpointed scale (TWO LEGS
    above), so a different rate is a new arm.
  - fire_cut_factor 1.0 (the canonical value; the recipe carried 0.5 from its base). The loss-excursion bar stays out
    of service (loss_excursion_k 1e6, cond_tb_sep25/make.py), so a fire is a gradient excursion or a non-finite step:
    it rewinds at the same rate in both arms, and an arm that keeps firing ends on the reload budget with a .dead file.
    A halving cut would leave the two arms at different rates.
  - epochs 1,000,000 (the recipe's 200,000 is a local run length; the step count starts at 0 and the wall ends a leg)
  - buffers.anchor_buffer.max_size 2,500,000, about twice the seed, which is every prior row (1,212,915). The
    recipe's 200,000 sat above its 52,181-row seed and its anchor count never moved (anchor_buffer_length at steps 2,600
    and 28,590); under that cap here the first admission would thin the seed down to it (the overflow thin in
    top_up_prior_from_anchors and screen_and_admit_anchors). The other buffer keys are the recipe's, which are the
    canonical config's "CONDITIONAL ARM:" values (anchor growth on, replay unprioritised).
  - cluster budgets: final_sep19's eLJ eval budget, phase 1's held-out sample count, and an archive with buffers every
    10,000 steps and not 5000: a buffers file holds the whole anchor buffer, 2.8 KB a row on the smoke run's sidecar,
    so about 3.5 to 4 GB an arm here against phase 1's 1.19 GB
THE SEED (final_sep19's job script, SEED_B): leg 1's newest 5000-step archive, or its _running.pt with SRC_RUNNING=1.
Each arm resolves it at its own first launch, so the two arms share a seed only if leg 1 is not writing meanwhile:
cancel it first. A resubmission resumes the arm's own _running.pt in full.

    python configs/qm9full_sep30/make.py p1
    python configs/qm9full_sep30/make.py arms [--dry]
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
_spec = importlib.util.spec_from_file_location('finmake', HERE.parent / 'final_sep19' / 'make.py')
fin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fin)

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

# ---- phase 2
RECIPE = HERE.parent / 'cond_tb_sep25' / 'ctb25_extreme_l1_cont3.yaml'
SEED_LEG = LEGS[LIVE][0]
ARMS = (('tbg', 'global'), ('tbh', 'head'))
P2_SCALE = 0.2
P2_FIRE_CUT = 1.0
P2_EPOCHS = 1_000_000
P2_ARCHIVE = 10_000
ANCHOR_MAX = 2_500_000
# (lr_control.fixed_scale, fire_cut_factor, hard_failure.loss_excursion_k, epochs, anchor_buffer.max_size) of the recipe
RECIPE_IS = (0.5, 0.5, 1.0e6, 200_000, 200_000)
WO_PLACEHOLDER = 'WEIGHTS_ONLY_PLACEHOLDER'
FROM_P1 = ('prior_path', 'molecules_path', 'test_molecules_path', 'checkpoints_dir', 'model', 'integrator', 'eval_T',
           'compile_policy', 'cuda_memory_fraction', 'test_eval_num_samples')
STUB_EXIT = [{'metric': 'bwd/mle', 'above': -1e9, 'patience': 1}]
STAGES = ['train_prior', 'var_conditioning']


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


def build_arm(run, untrusted, p1):
    cfg = yaml.safe_load(RECIPE.read_text(encoding='utf-8'))
    lc, ab = cfg['lr_control'], cfg['buffers']['anchor_buffer']
    recipe_is = (lc['fixed_scale'], lc['fire_cut_factor'], lc['hard_failure']['loss_excursion_k'], cfg['epochs'],
                 ab['max_size'])
    assert recipe_is == RECIPE_IS and lc['seed_lr'] == SEED_LR, recipe_is
    for k in FROM_P1:
        cfg[k] = copy.deepcopy(p1[k])
    cfg['tag'], cfg['run_name'] = TAG, run
    cfg['checkpoint_name'] = fin.PLACEHOLDER
    cfg['load_weights_only'] = True     # the first launch; main_arms writes the placeholder the job script fills
    cfg['continue_from_checkpoint'] = False
    cfg['prior_model_name'] = fin.PRIOR_PLACEHOLDER
    cfg['epochs'] = P2_EPOCHS
    cfg['archive_period'] = P2_ARCHIVE
    cfg['archive_buffers'] = True
    cfg.update(fin.ELJ_EVAL)
    lc['fixed_scale'] = P2_SCALE
    lc['fire_cut_factor'] = P2_FIRE_CUT
    cfg['energy_config']['energy_reference'] = 'seed_min'
    cl = cfg['condition_log_z']
    cl['untrusted_z'] = untrusted
    cl['global_half_life_updates'] = 50.0
    cl['half_life_visits'] = HALF_LIFE_VISITS
    ab['max_size'] = ANCHOR_MAX
    stages = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in stages] == STAGES, [s['name'] for s in stages]
    stub = stages[0]
    stub.pop('skip_if', None)
    stub['exit'] = copy.deepcopy(STUB_EXIT)
    stub['on_exit'] = ['snapshot_prior']
    cfg['protocol'] = PROTOCOL
    return cfg


def check_arm(cfg, name, untrusted, p1, n_prior_rows):
    assert cfg['model'] == p1['model'] and cfg['integrator'] == p1['integrator'], f'{name}: the seed\'s model moved'
    assert cfg['integrator']['T'] == cfg['eval_T'] == p1['eval_T'] == T, f'{name}: the trajectory length moved'
    mine, theirs = w3.problem_def(cfg), w3.problem_def(p1)
    moved = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity differs from the seed leg on {moved}; the seed would be refused'
    ec = cfg['energy_config']
    assert ec['energy_reference'] == 'seed_min' and ec['lambda_mix'] == 1.0 and ec['prior_flow_path'] is None, name
    cl = cfg['condition_log_z']
    assert (cl['untrusted_z'], cl['half_life_visits'], cl['min_visits']) == (untrusted, HALF_LIFE_VISITS, 20), name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['seed_lr'] == SEED_LR and lc['fixed_scale'] == P2_SCALE, name
    # the stage promotes above its burn-in scale: no cut on a fire, and the loss-excursion bar out of service
    assert lc['fixed_scale'] > lc['burn_in_scale'] and lc['fire_cut_factor'] == P2_FIRE_CUT, name
    assert lc['hard_failure']['loss_excursion_k'] == RECIPE_IS[2], name
    # fixed_scale acts on the rate var_conditioning steps (lr_fused, managed when 'auto') and no rail holds it
    assert cfg['lr_fused'] == 'auto' and cfg.get('max_lr') is None, name
    assert (cfg['epochs'], cfg['archive_period'], cfg['archive_buffers']) == (P2_EPOCHS, P2_ARCHIVE, True), name
    assert P2_ARCHIVE % cfg['eval_period'] == 0, f'{name}: an archive links the buffers file the last eval wrote'
    # the canonical config's "CONDITIONAL ARM:" globals (mk_dev.yaml), which the recipe carries
    assert (cfg['batch_size'], cfg['grow_batch_size'], cfg['max_batch_size'], cfg['batch_util_target']) == \
        (1000, False, 1000, 0.0), name
    assert cl['rollout_condition_draw'] == 'cycle' and cfg['z_calibration']['fill_from_eval'] == 'off', name
    rb, ab = cfg['buffers']['replay_buffer'], cfg['buffers']['anchor_buffer']
    assert (rb['churn_rate'], rb['mean_residence_steps'], rb['max_size']) == (0, 1200, 150000), name
    assert rb['val_frac'] == 0.0 and rb['prioritise']['enabled'] is False, name
    assert (ab['frozen'], ab['thin_every_n_evals'], ab['refresh_every_n_evals'], ab['topup_admit_record_breakers']) == \
        (False, 0, 0, True), name
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors' and ab['seed_source'] == 'prior_dataset', name
    assert ab['max_size'] == ANCHOR_MAX >= 2 * n_prior_rows, (name, n_prior_rows)
    st = cfg['protocols'][PROTOCOL]['stages']
    assert st[0]['exit'] == STUB_EXIT and 'skip_if' not in st[0] and st[0]['on_exit'] == ['snapshot_prior'], name
    vc = st[1]
    assert vc['train_mode'] == 'fused', name
    assert vc['on_enter'] == ['rebuild_prior_by_churn', 'set_lr_flow:1.0e-4', 'freeze_pb'], (name, vc['on_enter'])
    assert vc['fracs'] == {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0} and vc['fwd_rollout_every'] == 0, name
    for branch in ('fwd', 'bwd', 'replay'):
        c = vc['loss_coeffs'][branch]
        assert c['tb'] == 1.0 and c['tb_z_source'] == 'persistent', (name, branch)
    fwd = vc['loss_coeffs']['fwd']
    assert (fwd['emp_z_persistent'], fwd['freeze_z'], fwd['reward_grads']) == (1.0, 0.0, 1.0), \
        f'{name}: the head must learn the trusted estimates, and the forward branch carries the terminal force'
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['prior_model_name'] == fin.PRIOR_PLACEHOLDER, name
    assert cfg['continue_from_checkpoint'] is False, name
    w3._scan_local_paths(cfg, name)


def main_arms(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    seed_arm = f'{TAG}_{SEED_LEG}'
    seed_yaml = HERE / f'{seed_arm}.yaml'
    p1 = yaml.safe_load(seed_yaml.read_text(encoding='utf-8'))
    committed = w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{seed_arm}.yaml'], HERE)
    assert yaml.safe_load(committed) == p1, f'{seed_yaml} is not the committed file the seed leg ran'
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size
    import torch
    n_prior_rows = int(torch.load(LOCAL_PRIORS / TEST, map_location='cpu', weights_only=False)['n_structures'])
    arms = {}
    for run, untrusted in ARMS:
        name = f'{TAG}_{run}'
        cfg = build_arm(run, untrusted, p1)
        check_arm(cfg, name, untrusted, p1, n_prior_rows)
        for weights_only in (True, False):  # the first launch and a resubmission, as the job script resolves them
            probe = copy.deepcopy(cfg)
            probe['load_weights_only'] = weights_only
            fin.load_check(probe, name, STAGES)
        cfg['load_weights_only'] = WO_PLACEHOLDER
        arms[name] = cfg
    # the job script finds an arm's files, and the seed leg's, by `*<arm>_*`: no name may match another's
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + list(arms)
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    a, b = arms.values()
    b_as_a = copy.deepcopy(b)
    b_as_a['condition_log_z']['untrusted_z'] = a['condition_log_z']['untrusted_z']
    b_as_a['run_name'] = a['run_name']
    assert a == b_as_a, 'the two arms must differ in condition_log_z.untrusted_z alone'
    for i, (name, cfg) in enumerate(arms.items()):
        lc = cfg['lr_control']
        print(f"[{i}] {name}: untrusted_z {cfg['condition_log_z']['untrusted_z']}; lr_control.fixed_scale "
              f"{lc['fixed_scale']:g} ({lc['fixed_scale'] * SEED_LR:g}), fire_cut_factor {lc['fire_cut_factor']:g}; "
              f"anchor capacity {ANCHOR_MAX:,} for {n_prior_rows:,} prior rows; weights-only first launch from "
              f"*{seed_arm}_*")
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in arms.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    fin._write_index(HERE / 'INDEX_b.tsv',
                     [(name, 'qm9full', 'seeded', seed_arm, PRIOR, str(prior_bytes)) for name in arms])
    with (HERE / f'submit_{BATTERY}_b.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(
            wall=fin.WALL, last=len(arms) - 1, tag=TAG + 'b', battery=BATTERY, leg='b', ckpts=w3.CLUSTER_CKPTS,
            data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
            what=f'phase 2 of the conditional GFN on the full-QM9 prior, the extreme TB recipe: weights-only first '
                 f'launch from {seed_arm} (its newest step archive, or its _running.pt with SRC_RUNNING=1), full '
                 f'resume afterwards; the arms differ in condition_log_z.untrusted_z (global, head).'))
    print(f'wrote {len(arms)} arms, INDEX_b.tsv and submit_{BATTERY}_b.sbatch')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


def main(argv):
    if argv[:1] == ['p1']:
        return main_p1(argv)
    if argv[:1] == ['arms']:
        return main_arms(argv)
    sys.exit(__doc__)


def main_p1(argv):
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
