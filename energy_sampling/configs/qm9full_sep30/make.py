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
on a first launch. EIGHT ARMS (owner 2026-10-03: "a couple of learning rates, with and without forces, fwd and replay
seat ... maybe learned Z(c)"), each the baseline qf30_tbg with the named differences and no others (asserted leaf by
leaf). INDEX_b rows in this order; rows 0-1 are as first pushed (2126d9fa) and new arms are appended, never reordered:
  [0] qf30_tbg       THE BASELINE: forward seat, forward force, 2.5e-5, and a row below min_visits scored against
                     one all-condition level (condition_log_z.untrusted_z global)
  [1] qf30_tbh       untrusted_z head: ... against the small learned head, which the TB residual then trains
  [2] qf30_fwdF_lr5  lr_control.fixed_scale 0.5 (6.25e-5)
  [3] qf30_fwdN_lr2  no force (forward reward_grads 0, path_grad_last_k 0)
  [4] qf30_fwdN_lr5  no force, 6.25e-5
  [5] qf30_repN_lr2  the replay seat, no force
  [6] qf30_repF_lr2  the replay seat with the stored terminal force on replay rows (replay stored_force_k 1)
  [7] qf30_tbl       tb_z_source learned on every branch: the head is the Z in every row's residual, trained by the
                     forward TB residual and still regressed onto the tracker where a molecule is trusted.
                     config_invariants marks this BASELINE (every conditional battery that ran used persistent)
So: rate x force on the forward seat (0, 2, 3, 4), seat x force at 2.5e-5 (0, 3, 5, 6), and the Z of the residual at
the baseline (0, 1, 7).
THE REPLAY SEAT is prod_sep20's, on the recipe's stage: a rollout every 5th step that carries no loss (it admits to
the replay buffer and feeds the per-condition Z), the policy trained by backward and replay rows at 0.7 / 0.3 held by
the entry fracs (no balance block), tau 600, 5% of admitted rows held out for replay/val_gap_nats, and an extra
rollout while the buffer holds under two batches. It has not run on the conditional route under TB before. Nothing
trains the head there (the forward loss has no weight), which the global fallback does not need.
z_calibration.fill_threshold is 0.5 there only because config_invariants refuses a rollout cadence without an armed
level fill; the fill refuses a Z(c) head, so it does nothing, and what a rollout pins is the tracker. update_log_z is
true, as in the recipe's last legs, so backward and replay rows feed the per-condition Z as well: on this seat it is
fed mostly by stored rows (per step 1000 backward and 1000 replay rows against 200 forward).
Held-out molecules are never visited, so their eval rows always take the fallback (the head, under tbl):
eval_test/tb_err against eval_fwd/tb_err is the test of whether a learned Z(c) generalises. qf30_tbh differs from the
baseline only until a molecule reaches min_visits, and on held-out rows; qf30_tbl is the arm in which a learned Z(c)
is in every training row's residual. On top of the recipe, in every arm:
  - the model, integrator, data files and cluster keys are phase 1's (the seed's architecture and problem; the problem
    identity is asserted equal to the seed leg's)
  - the stub train_prior of final_sep19 (no skip_if, an exit that holds at the first metric write, on_exit
    snapshot_prior) and no prior model
  - energy_reference seed_min; half_life_visits 50; untrusted_z as the arm
  - lr_control.fixed_scale 0.2 (2.5e-5) or 0.5 (6.25e-5). At width 512 the recipe's first leg ran 0.2 and its last
    ran 0.5 to step 28,590 with lr_ctrl/divergences 0 (their W&B summaries), and prod_sep20's replay seat ran 0.5;
    0.2 is that rate halved for the doubled width, the rule that gave phase 1 its 2.5e-4. THE RATE IS FIXED AT
    LAUNCH: a resume keeps the checkpointed scale (TWO LEGS above), so a different rate is a new arm.
  - fire_cut_factor 1.0 (the canonical value; the recipe carried 0.5 from its base). The loss-excursion bar stays out
    of service (loss_excursion_k 1e6, cond_tb_sep25/make.py), so a fire is a gradient excursion or a non-finite step:
    it rewinds at the arm's own rate, and an arm that keeps firing ends on the reload budget with a .dead file.
    A halving cut would leave arms that share a rate at different ones.
  - epochs 1,000,000 (the recipe's 200,000 is a local run length; the step count starts at 0 and the wall ends a leg)
  - buffers.anchor_buffer.max_size 2,500,000, about twice the seed, which is every prior row (1,212,915). The
    recipe's 200,000 sat above its 52,181-row seed and its anchor count never moved (anchor_buffer_length at steps 2,600
    and 28,590); under that cap here the first admission would thin the seed down to it (the overflow thin in
    top_up_prior_from_anchors and screen_and_admit_anchors). The other buffer keys are the recipe's, which are the
    canonical config's "CONDITIONAL ARM:" values (anchor growth on, replay unprioritised).
  - cluster budgets: final_sep19's eLJ eval budget, phase 1's held-out sample count, and an archive with buffers every
    10,000 steps and not 5000: a buffers file holds the whole anchor buffer, 2.8 KB a row on the smoke run's sidecar,
    so about 3.5 to 4 GB an arm here against phase 1's 1.19 GB
LEG C, P_B LEFT TRAINABLE (owner 2026-10-03: "did we have unfrozen Pb on this battery? we should have"). Every
leg-b arm freezes P_B on entering the TB stage (on_enter freeze_pb, the recipe's setting). Leg c is three leg-b arms
with that action removed and nothing else changed (asserted), in their own INDEX_c.tsv and job script, so that
submitting it cannot relaunch a leg-b arm that is running:
  [0] qf30_upb_lr5   qf30_fwdF_lr5 with P_B live (6.25e-5; owner 2026-10-03, on forward Jensen: "qf30_fwdF_lr5 is
                     winning")
  [1] qf30_upb_lr2   qf30_tbg with P_B live (the baseline)
  [2] qf30_upb_tbl   qf30_tbl with P_B live (learned Z, the arm whose backward residual carries the whole level gap)
Forward seat only. Forward Jensen stays comparable across the two legs: E_PF[log R + log P_B - log P_F] is a lower
bound on log Z under any P_B.
THE SEED (final_sep19's job script, SEED_B): leg 1's newest 5000-step archive, or its _running.pt with SRC_RUNNING=1.
Each arm resolves it at its own first launch, so the arms share a seed only if leg 1 is not writing meanwhile:
cancel it first. A resubmission resumes the arm's own _running.pt in full.

    python configs/qm9full_sep30/make.py p1
    python configs/qm9full_sep30/make.py arms [--dry]      # legs b and c
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
# (run_name, seat, force, lr_control.fixed_scale, the Z of the residual). ROW ORDER IS THE ARRAY INDEX: append only.
ARMS = (('tbg', 'fwd', True, 0.2, 'global'),
        ('tbh', 'fwd', True, 0.2, 'head'),
        ('fwdF_lr5', 'fwd', True, 0.5, 'global'),
        ('fwdN_lr2', 'fwd', False, 0.2, 'global'),
        ('fwdN_lr5', 'fwd', False, 0.5, 'global'),
        ('repN_lr2', 'replay', False, 0.2, 'global'),
        ('repF_lr2', 'replay', True, 0.2, 'global'),
        ('tbl', 'fwd', True, 0.2, 'learned'))
# leg c: (run_name, the leg-b run it is with the stage's freeze_pb removed). ROW ORDER IS THE ARRAY INDEX: append only.
LIVE_PB = (('upb_lr5', 'fwdF_lr5'), ('upb_lr2', 'tbg'), ('upb_tbl', 'tbl'))
ON_ENTER = ['rebuild_prior_by_churn', 'set_lr_flow:1.0e-4', 'freeze_pb']
P2_SCALES = (0.2, 0.5)
BRANCHES = ('fwd', 'bwd', 'replay')
# prod_sep20's replay seat: rollout period, replay share, mean residence in steps, the held-out split of final_sep19
REPLAY_SEAT = dict(every=5, replay=0.3, tau=600, val_frac=0.05, val_min=64, val_cap=1024, occupancy_min_batches=2.0,
                   fill_threshold=0.5)
VC = f'protocols.{PROTOCOL}.stages[var_conditioning]'
# the leaves a factor may move away from the baseline, as path prefixes; an arm must move every one it names
MOVES = {
    'rate': ('lr_control.fixed_scale',),
    'no_force': (f'{VC}.loss_coeffs.fwd.reward_grads', f'{VC}.loss_coeffs.fwd.path_grad_last_k'),
    'head': ('condition_log_z.untrusted_z',),
    'learned': ('condition_log_z.untrusted_z',) + tuple(f'{VC}.loss_coeffs.{b}.tb_z_source' for b in BRANCHES),
    'replay': (f'{VC}.fracs', f'{VC}.min_fracs', f'{VC}.balance', f'{VC}.fwd_rollout_every',
               f'{VC}.fwd_rollout_triggers.occupancy_min_batches', f'{VC}.loss_coeffs.fwd.freeze_policy',
               f'{VC}.loss_coeffs.fwd.emp_z_persistent', f'{VC}.loss_coeffs.fwd.freeze_z',
               f'{VC}.loss_coeffs.fwd.reward_grads', f'{VC}.loss_coeffs.fwd.path_grad_last_k',
               'buffers.replay_buffer.mean_residence_steps', 'buffers.replay_buffer.val_frac',
               'buffers.replay_buffer.val_min', 'buffers.replay_buffer.val_cap', 'replay_loss_coeffs.stored_force_k',
               'replay_loss_coeffs.stored_force_mode', 'replay_loss_coeffs.resample_last_k',
               'replay_loss_coeffs.reward_grads', 'replay_loss_coeffs.force_chunk_rows',
               'z_calibration.fill_threshold'),
}
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


def _replay_seat(cfg, vc, force):
    """prod_sep20's replay seat on the recipe's stage (the module docstring, THE REPLAY SEAT)."""
    rs = REPLAY_SEAT
    vc['fracs'] = {'fwd': 0.0, 'bwd': round(1.0 - rs['replay'], 3), 'replay': rs['replay']}
    del vc['min_fracs'], vc['balance']      # no balance: the entry fracs hold, and replay trains (Stage.replay_trains)
    vc['fwd_rollout_every'] = rs['every']
    vc['fwd_rollout_triggers']['occupancy_min_batches'] = rs['occupancy_min_batches']
    # the rollout carries no loss, so nothing on it trains the policy or the head, and no live force is taken on it
    vc['loss_coeffs']['fwd'].update(freeze_policy=1.0, emp_z_persistent=0.0, freeze_z=1.0, reward_grads=0.0,
                                    path_grad_last_k=0)
    cfg['buffers']['replay_buffer'].update(mean_residence_steps=rs['tau'], val_frac=rs['val_frac'],
                                           val_min=rs['val_min'], val_cap=rs['val_cap'])
    # the stored terminal force on replay rows; the keys are absent from the recipe's base block, which a stage
    # override cannot add to (final_sep19 sets them in the base block as well)
    cfg['replay_loss_coeffs'].update(stored_force_k=1 if force else 0, stored_force_mode='implied', resample_last_k=0,
                                     reward_grads=0.0, force_chunk_rows=None)
    # config_invariants refuses a cadenced stage without an armed level fill (fwd_rollout_cadence_is_well_formed).
    # The fill itself refuses a Z(c) head (Modeller._z_fill_head_is_fillable), so it does nothing on this route:
    # what a rollout updates here is the per-condition tracker. mk_dev's value.
    cfg['z_calibration']['fill_threshold'] = rs['fill_threshold']


def build_arm(run, seat, force, scale, z, p1, live_pb=False):
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
    lc['fixed_scale'] = scale
    lc['fire_cut_factor'] = P2_FIRE_CUT
    cfg['energy_config']['energy_reference'] = 'seed_min'
    cl = cfg['condition_log_z']
    cl['untrusted_z'] = 'head' if z == 'learned' else z     # under a learned source no row takes the fallback
    cl['global_half_life_updates'] = 50.0
    cl['half_life_visits'] = HALF_LIFE_VISITS
    ab['max_size'] = ANCHOR_MAX
    stages = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in stages] == STAGES, [s['name'] for s in stages]
    stub, vc = stages
    stub.pop('skip_if', None)
    stub['exit'] = copy.deepcopy(STUB_EXIT)
    stub['on_exit'] = ['snapshot_prior']
    cfg['protocol'] = PROTOCOL
    if z == 'learned':
        for branch in BRANCHES:
            vc['loss_coeffs'][branch]['tb_z_source'] = 'learned'
    if seat == 'replay':
        _replay_seat(cfg, vc, force)
    elif not force:
        vc['loss_coeffs']['fwd'].update(reward_grads=0.0, path_grad_last_k=0)
    if live_pb:
        vc['on_enter'] = [a for a in vc['on_enter'] if a != 'freeze_pb']
    return cfg


def check_arm(cfg, name, seat, force, scale, z, p1, n_prior_rows, live_pb=False):
    assert seat in ('fwd', 'replay') and scale in P2_SCALES and z in ('global', 'head', 'learned'), name
    assert cfg['model'] == p1['model'] and cfg['integrator'] == p1['integrator'], f"{name}: the seed's model moved"
    assert cfg['integrator']['T'] == cfg['eval_T'] == p1['eval_T'] == T, f'{name}: the trajectory length moved'
    mine, theirs = w3.problem_def(cfg), w3.problem_def(p1)
    moved = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    assert not moved, f'{name}: problem identity differs from the seed leg on {moved}; the seed would be refused'
    ec = cfg['energy_config']
    assert ec['energy_reference'] == 'seed_min' and ec['lambda_mix'] == 1.0 and ec['prior_flow_path'] is None, name
    cl = cfg['condition_log_z']
    assert cl['untrusted_z'] == ('head' if z == 'learned' else z), name
    assert (cl['half_life_visits'], cl['min_visits']) == (HALF_LIFE_VISITS, 20), name
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['seed_lr'] == SEED_LR and lc['fixed_scale'] == scale, name
    # the stage promotes above its burn-in scale: no cut on a fire, and the loss-excursion bar out of service
    assert lc['fixed_scale'] > lc['burn_in_scale'] and lc['fire_cut_factor'] == P2_FIRE_CUT, name
    assert lc['hard_failure']['loss_excursion_k'] == RECIPE_IS[2], name
    # fixed_scale acts on the rate var_conditioning steps (lr_fused, managed when 'auto') and no rail holds it
    assert cfg['lr_fused'] == 'auto' and cfg.get('max_lr') is None, name
    assert (cfg['epochs'], cfg['archive_period'], cfg['archive_buffers']) == (P2_EPOCHS, P2_ARCHIVE, True), name
    assert P2_ARCHIVE % cfg['eval_period'] == 0, f'{name}: an archive links the buffers file the last eval wrote'
    # the canonical config's "CONDITIONAL ARM:" globals (mk_dev.yaml), which the recipe carries
    batch = cfg['batch_size']
    assert (batch, cfg['grow_batch_size'], cfg['max_batch_size'], cfg['batch_util_target']) == \
        (1000, False, 1000, 0.0), name
    assert cl['rollout_condition_draw'] == 'cycle' and cfg['z_calibration']['fill_from_eval'] == 'off', name
    rb, ab = cfg['buffers']['replay_buffer'], cfg['buffers']['anchor_buffer']
    assert (rb['churn_rate'], rb['max_size'], rb['prioritise']['enabled']) == (0, 150000, False), name
    assert (ab['frozen'], ab['thin_every_n_evals'], ab['refresh_every_n_evals'], ab['topup_admit_record_breakers']) == \
        (False, 0, 0, True), name
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors' and ab['seed_source'] == 'prior_dataset', name
    assert ab['max_size'] == ANCHOR_MAX >= 2 * n_prior_rows, (name, n_prior_rows)
    st = cfg['protocols'][PROTOCOL]['stages']
    assert st[0]['exit'] == STUB_EXIT and 'skip_if' not in st[0] and st[0]['on_exit'] == ['snapshot_prior'], name
    vc = st[1]
    assert vc['train_mode'] == 'fused' and vc['flags']['update_log_z'] is True, name
    assert vc['on_enter'] == (ON_ENTER[:-1] if live_pb else ON_ENTER), (name, vc['on_enter'])
    # P_B trains unless the stage freezes it: its head is learned and no load-time freeze is set
    assert cfg['model']['learn_pb'] is True and not cfg.get('freeze_backward_policy'), name
    assert not (live_pb and seat == 'replay'), f'{name}: a live P_B is for the forward seat'
    for branch in BRANCHES:
        c = vc['loss_coeffs'][branch]
        assert c['tb'] == 1.0 and c['tb_z_source'] == ('learned' if z == 'learned' else 'persistent'), (name, branch)
    fwd, rc = vc['loss_coeffs']['fwd'], cfg['replay_loss_coeffs']
    if seat == 'fwd':
        assert vc['fracs'] == {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0} and vc['fwd_rollout_every'] == 0, name
        assert (rb['mean_residence_steps'], rb['val_frac']) == (1200, 0.0) and 'stored_force_k' not in rc, name
        # the head learns the trusted estimates; the forward branch trains the policy, with or without the force
        assert (fwd['emp_z_persistent'], fwd['freeze_z'], fwd['freeze_policy']) == (1.0, 0.0, 0.0), name
        assert (fwd['reward_grads'], fwd['path_grad_last_k']) == ((1.0, 1) if force else (0.0, 0)), name
    else:
        rs = REPLAY_SEAT
        assert z == 'global', f'{name}: nothing trains the head on the replay seat, so no residual may read it'
        assert vc['fracs'] == {'fwd': 0.0, 'bwd': 0.7, 'replay': 0.3} and 'balance' not in vc, name
        assert 'min_fracs' not in vc and vc['fwd_z_sidecar'] is False and vc['replay_warmup_rows'] == 0, name
        assert vc['fwd_rollout_every'] == rs['every'] and vc['z_pin_rollout_every'] == 0, name
        assert vc['fwd_rollout_triggers']['occupancy_min_batches'] == rs['occupancy_min_batches'], name
        assert (fwd['freeze_policy'], fwd['emp_z_persistent'], fwd['freeze_z']) == (1.0, 0.0, 1.0), name
        assert (fwd['reward_grads'], fwd['path_grad_last_k']) == (0.0, 0), name
        assert (rb['mean_residence_steps'], rb['val_frac'], rb['val_cap']) == \
            (rs['tau'], rs['val_frac'], rs['val_cap']), name
        # the cap must not bind: occupancy is batch x tau / period (rollout-cadence-and-dose.md)
        assert rb['max_size'] >= 1.2 * batch * rs['tau'] / rs['every'], name
        assert (rc['stored_force_k'], rc['stored_force_mode'], rc['resample_last_k']) == \
            (1 if force else 0, 'implied', 0), name
        # the held-out split is refused beside a condition-blocked replay draw, and the stored force under
        # temperature conditioning
        assert rc['condition_block_m'] == 0 and cfg['temperature_conditioning'] is False, name
        assert cfg['z_calibration']['fill_threshold'] == rs['fill_threshold'] > 0, name
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['prior_model_name'] == fin.PRIOR_PLACEHOLDER, name
    assert cfg['continue_from_checkpoint'] is False, name
    w3._scan_local_paths(cfg, name)


_ABSENT = object()


def _leaves(node, pre=''):
    """{dotted path: value} of every leaf; a list of named dicts (the stages) is walked by name."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(_leaves(v, f'{pre}.{k}' if pre else str(k)))
        return out
    if isinstance(node, list) and node and all(isinstance(x, dict) and 'name' in x for x in node):
        out = {}
        for x in node:
            out.update(_leaves(x, f"{pre}[{x['name']}]"))
        return out
    return {pre: node}


def _moved(a, b):
    la, lb = _leaves(a), _leaves(b)
    return sorted(k for k in set(la) | set(lb) if k != 'run_name' and la.get(k, _ABSENT) != lb.get(k, _ABSENT))


def check_battery(arms):
    """Each arm is the baseline with its factors' leaves moved: all of them, and no others."""
    assert len({spec[1:] for spec in ARMS}) == len(ARMS), 'two arms share one specification'
    base, baseline = ARMS[0][1:], arms[f'{TAG}_{ARMS[0][0]}']
    assert base == ('fwd', True, P2_SCALES[0], 'global'), base
    for run, seat, force, scale, z in ARMS[1:]:
        name = f'{TAG}_{run}'
        factors = ((['rate'] if scale != base[2] else [])
                   + (['replay'] if seat == 'replay' else [] if force else ['no_force'])
                   + ([z] if z != 'global' else []))
        allowed = [p for f in factors for p in MOVES[f]]
        moved = _moved(baseline, arms[name])
        stray = [m for m in moved if not any(m == p or m.startswith(p + '.') for p in allowed)]
        missing = [p for p in allowed if not any(m == p or m.startswith(p + '.') for m in moved)]
        assert not stray, f'{name}: differs from the baseline outside its factors {factors}: {stray}'
        assert not missing, f'{name}: its factors {factors} name {missing}, which it does not move'
    seat_pair = [arms[f'{TAG}_{run}'] for run, seat, _, _, _ in ARMS if seat == 'replay']
    assert _moved(*seat_pair) == ['replay_loss_coeffs.stored_force_k'], 'the replay-seat pair differs beyond the force'


def _baseline_notices(cfg):
    """config_invariants' violations below ERROR for the arm as a first launch resolves it."""
    import config_invariants    # on the path once fin.load_check has run
    raw = copy.deepcopy(cfg)
    raw['checkpoint_name'], raw['prior_model_name'] = 'SEED_best.pt', None
    return [v for v in config_invariants.check(raw) if v.severity != config_invariants.ERROR]


def _vet(cfg, name, z):
    """Load the arm as the job script resolves it, both ways, account for its notices, and leave the placeholder."""
    for weights_only in (True, False):  # the first launch and a resubmission
        probe = copy.deepcopy(cfg)
        probe['load_weights_only'] = weights_only
        fin.load_check(probe, name, STAGES)
    notices = _baseline_notices(cfg)
    if z == 'learned':
        # the battery's one departure from the conditional persistent-Z baseline, made on purpose: the trainer
        # prints these at load and runs, and `config_snapshot --check` reports such an arm as contract FAILED
        assert notices and all(v.rule == 'conditional_z_settings_are_conditional' and 'tb_z_source' in v.detail
                               for v in notices), (name, [str(v) for v in notices])
    else:
        assert not notices, (name, [str(v) for v in notices])
    cfg['load_weights_only'] = WO_PLACEHOLDER
    return cfg


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
    for run, seat, force, scale, z in ARMS:
        name = f'{TAG}_{run}'
        cfg = build_arm(run, seat, force, scale, z, p1)
        check_arm(cfg, name, seat, force, scale, z, p1, n_prior_rows)
        arms[name] = _vet(cfg, name, z)
    check_battery(arms)
    live, spec = {}, {run: rest for run, *rest in ARMS}
    for run, base_run in LIVE_PB:
        name = f'{TAG}_{run}'
        cfg = build_arm(run, *spec[base_run], p1, live_pb=True)
        check_arm(cfg, name, *spec[base_run], p1, n_prior_rows, live_pb=True)
        live[name] = _vet(cfg, name, spec[base_run][3])
        assert _moved(arms[f'{TAG}_{base_run}'], live[name]) == [f'{VC}.on_enter'], \
            f'{name} differs from {TAG}_{base_run} beyond the freeze'
    # the job script finds an arm's files, and the seed leg's, by `*<arm>_*`: no name may match another's
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + list(arms) + list(live)
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    print(f'phase-2 arms, each the baseline [0] with the named differences. INDEX_b row = array index. Anchor capacity '
          f'{ANCHOR_MAX:,} for {n_prior_rows:,} prior rows; weights-only first launch from *{seed_arm}_*')
    z_text = {'global': 'tracker, global fallback', 'head': 'tracker, head fallback', 'learned': 'learned head'}
    for i, (run, seat, force, scale, z) in enumerate(ARMS):
        frc = ('stored force on replay rows' if seat == 'replay' else 'forward force') if force else 'no force'
        print(f"[{i}] {TAG}_{run:<9} {'forward' if seat == 'fwd' else 'replay '} seat | {frc:<27} | rate "
              f"{scale * SEED_LR:.3g} (fixed_scale {scale:g}) | Z: {z_text[z]}")
    print(f'leg c, P_B left trainable (INDEX_c row = array index):')
    for i, (run, base_run) in enumerate(LIVE_PB):
        print(f'[{i}] {TAG}_{run:<9} {TAG}_{base_run} without freeze_pb on entering the TB stage')
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in {**arms, **live}.items():
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
                 f'resume afterwards; {len(arms)} arms over the rate, the terminal force, the seat and the Z of the '
                 f'residual (make.py).'))
    fin._write_index(HERE / 'INDEX_c.tsv',
                     [(name, 'qm9full', 'seeded', seed_arm, PRIOR, str(prior_bytes)) for name in live])
    with (HERE / f'submit_{BATTERY}_c.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(fin.SBATCH.format(
            wall=fin.WALL, last=len(live) - 1, tag=TAG + 'c', battery=BATTERY, leg='c', ckpts=w3.CLUSTER_CKPTS,
            data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
            what=f'leg-b arms of phase 2 with P_B left trainable (no freeze_pb on entering the TB stage): '
                 f'weights-only first launch from {seed_arm} (its newest step archive, or its _running.pt with '
                 f'SRC_RUNNING=1), full resume afterwards (make.py).'))
    print(f'wrote {len(arms)} leg-b arms with INDEX_b.tsv and submit_{BATTERY}_b.sbatch, and {len(live)} leg-c arms '
          f'with INDEX_c.tsv and submit_{BATTERY}_c.sbatch')
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
