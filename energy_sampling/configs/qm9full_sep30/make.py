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

LEG D, CONTINUATION (`python configs/qm9full_sep30/make.py cont`; owner 2026-10-05, the leg-b and leg-c arms stopped
by hand near steps 110,000 and 72,000: "we should no matter [what] submit a couple of continuation jobs ... Pb and
forward forces and higher LR looked good"). INDEX_d rows, in the order to submit them in:
  [0] qf30_fwdF_lr5    the leg-b arm itself, resumed from its own _running.pt (its committed file, untouched)
  [1] qf30_fwdF_lr10   qf30_fwdF_lr5 continued at twice the rate (1.25e-4)
  [2] qf30_fwdF_k5m    qf30_fwdF_lr5 continued with the forward force reaching the last 5 steps
                       (path_grad_last_k 5 = 0.1 time units at T 50; 1 step is 0.02), through the steps' means only
                       (path_grad_scale 0: the sampled log-variances are detached)
  [3] qf30_fwdF_k5     the same reach with the scale channel live, as the 1-step force ran (path_grad_scale 1)
  [4] qf30_upb_lr5     the leg-c arm itself (P_B trainable), resumed from its own _running.pt
  [5] qf30_upb_lr10    qf30_upb_lr5 continued at twice the rate
ROWS BUILT AND NOT SHIPPED (local runs 2026-10-05 on a 300-molecule prior at batch 200, each a full load from one
step-900 archive of a qf30_fwdF_lr5-shaped run in cruise at scale 0.5, 150 steps): four times the rate ended
UNRECOVERABLE after 4 rewinds in 56 steps, and twice the rate with the 5-step force took 3 rewinds in 150 steps;
twice the rate alone, and the 5-step force alone in both forms, took none. Step time there: 1.20 s with the 1-step
force, 1.37 s with 5 steps through the means, 1.80 s with 5 steps and the scale channel live.
A CONTINUED ARM IS A NEW RUN NAME LOADED IN FULL from the arm it continues (load_weights_only false): weights, the
P_B snapshot, optimizers, step count, the per-condition Z table, the buffers and the controller's state, so it opens
inside var_conditioning at the source's step and no on_enter action runs again. It differs from its source in the
named keys and no others (asserted leaf by leaf):
  - THE RATE MOVES THROUGH lr_control.seed_lr, NOT fixed_scale. The controller's scale is restored from the checkpoint
    (0.5 in both sources, in cruise) and fixed_scale is read once, when burn-in ends (TWO LEGS above), so a new
    fixed_scale would be inert. The applied rate is base x scale (controller.py, _apply_lrs) and the base is this
    config's seed_lr (lr_fused auto), so seed_lr 2.5e-4 under the restored 0.5 trains at 1.25e-4. fixed_scale stays
    0.5, which keeps the pair consistent should the controller ever reopen.
  - the force's reach is the stage's forward path_grad_last_k and its scale channel the stage's forward
    path_grad_scale; loss coefficients are always read from the config.
The first launch of a continued arm seeds from ONE NAMED STEP ARCHIVE of its source and that archive's frozen
buffers (steps 100,000 and 70,000, INDEX column src_step), never the source's _running.pt and never "the newest
archive": rows 0 and 4 resume those runs, rewrite the first and add to the second, and array tasks do not start
together. So do not pass SRC_RUNNING to this leg. A resubmission resumes the arm's own _running.pt. Rows 0 and 4 name
no seed ('-', no step): a row whose own _running.pt is missing stops at the seed lookup instead of starting phase 2
again from phase 1.

LEG E, CONDITIONAL MLE (`python configs/qm9full_sep30/make.py cmle`; owner 2026-10-05). Phase 1 trains with
scramble_conditions, so the trunk learns to ignore the condition and the conditioner stays at its initial weights:
its forward draws sat at a median 68 kT above their molecule's best from step 24,000 to the end (qf30_p1lr2), and
everything conditional is learned inside the TB stage (qf30_fwdF_lr5: 25 kT at step 100,000, 3.2 kT a doubling of
steps). Leg e is phase 1's leg-1 config with that flag false and nothing else moved (asserted): the same backward
MLE on the stored search minima, the conditioner now in the graph. No TB, no rollouts in training; run until stopped.
  [0] qf30_cmle_seed  weights from qf30_p1lr2's step-45,000 archive: phase 1 carried on, conditionally (P_B live)
  [1] qf30_cmle_lead  weights from qf30_fwdF_lr5's step-100,000 archive: the TB-trained leader under MLE (P_B is the
                      frozen snapshot that archive carries, which a weights-only load installs)
Both are weights-only on a first launch (step count, optimizers and controller fresh: burn-in, then the ramp to
fixed_scale 2 = 2.5e-4, phase 1's rate) and resume themselves in full afterwards. The reading is the forward draws'
median excess against 68 (phase 1) and 25 (the leader). The rows are the stored minima as they are, at most 10 a
molecule, so a training molecule can be memorised (owner 2026-10-05: "not worried about MLE overfit for now").

LEG F, FORCE TERM (`python configs/qm9full_sep30/make.py force`; owner 2026-10-06: "an actual test run with this
whole new setup"). P_F's mean gains gate(t) * cap(step variance * force) (models/gfn.py GFN.init_force_drift), the
force being minus the gradient of a pre-trained atom trunk's energy on the latent (models/crystal_force.py; the trunk
is configs/atom_trunk_oct05's at_c2_ft_late). Each row is qf30_fwdF_lr5's step-FORCE_SEED_STEP archive loaded IN FULL
under a new name, as a leg-d continued row is, with the eight force leaves added and nothing else moved (asserted):
  [0] qf30_frc_ctl  no force term: the control, from the same archive under the same code
  [1] qf30_frc_g0   gate learned from 0: the arm starts as the control and the term has to earn its way in
  [2] qf30_frc_g1   gate learned from 0.1
The force acts on states from t = FORCE_T_MIN on (the last fifth of the trajectory), and one step's force displacement
is capped at FORCE_MAX_SIGMA of that step's noise standard deviations. The archive predates the gates: they start at
their configured values and every other parameter keeps its optimizer state (checkpointing.py _load_state,
_grown_optimizer_state). P_B is the frozen snapshot the archive carries and takes no force term. Every trajectory
that is scored, rolled out or not, pays one trunk call per state inside the window, so a step is slower than the
control's. Read against the control at equal steps: the eval_fwd quality numbers and the forward Jensen;
force/gate_fwd_mean_* (does the gate grow); force/cosine_*, force/err_sigma_* and force/agreement_failed (the trunk
against its own target on this run's rollout states); train_step_time.

    python configs/qm9full_sep30/make.py p1
    python configs/qm9full_sep30/make.py arms [--dry]      # legs b and c
    python configs/qm9full_sep30/make.py cont [--dry]      # leg d
    python configs/qm9full_sep30/make.py cmle [--dry]      # leg e
    python configs/qm9full_sep30/make.py force [--dry]     # leg f
    python configs/qm9full_sep30/make.py mleb [--dry]      # leg g
    python configs/qm9full_sep30/make.py scratch [--dry]   # leg h

LEG G, CONDITIONAL MLE OVER BATCH AND RATE (`python configs/qm9full_sep30/make.py mleb`; owner 2026-10-06: "MLE is
quite well-behaved and cheap. I wonder if we could get away with huge batches and larger LR, to accelerate
convergence"). Leg e's qf30_cmle_seed crossed the TB leader's excess at step 20,000 and ran at 2.5e-4 and batch 1000
with no rewind. Leg g is that arm with P_B frozen from the start, the batch pinned, and the two factors crossed:
  [0] qf30_cmf_b1k_lr20  batch 1000, 2.5e-4: the control (leg e's arm with P_B frozen and the batch pinned)
  [1] qf30_cmf_b1k_lr40  batch 1000, 5e-4
  [2] qf30_cmf_b4k_lr20  batch 4000, 2.5e-4
  [3] qf30_cmf_b4k_lr40  batch 4000, 5e-4
NO ROW AT 1e-3: in the local smoke run of this arm shape (300-molecule prior, batch 200, burn-in 50 and ramp 200
steps) the loss rose with the rate from 5e-5 on, and at 1e-3 the run took 4 rewinds in 31 steps and ended
UNRECOVERABLE; at 5e-4 it held 6 nats above its floor with no rewind. The generator refuses a scale above 4.
P_B FROZEN (freeze_backward_policy true: a snapshot of the seed's P_B, taken at start-up) because the batch is bound
by memory: at batch 1000 the training step peaked at 32.5 GB with P_B live (qf30_cmle_seed) and 16.4 GB with it
frozen (qf30_cmle_lead), which also ran 0.70 s a step against 0.98, and the two sat on one excess curve at equal
steps. THE BATCH IS PINNED (grow_batch_size false, max_batch_size the batch, batch_util_target 0): leg e carries the
occupancy sizer, which held its base rung. With growth off an out-of-memory cut is not regrown, so an arm that does
not fit runs on at a smaller batch: Batch Size and batch/oom_events say so. The rate is lr_control.fixed_scale, read
at the end of burn-in on these weights-only starts (2 and 4 on seed_lr 1.25e-4). Same seed as leg e: qf30_p1lr2's
step-45,000 archive, weights only on a first launch, full resume afterwards; no exit, stopped by hand. Compare at
equal steps, at equal rows seen and at equal wall clock.

LEG H, CONDITIONAL MLE FROM SCRATCH (`python configs/qm9full_sep30/make.py scratch`; owner 2026-10-06, the control
of the comparison of the existing architecture with and without the force term, and later of the graph policy).
Leg e's and leg g's arms start from other weights (phase 1's scrambled leg, or the TB leader); these start from nothing.
Each row is phase 1's leg-1 recipe (build_p1 at fixed_scale 2: burn-in, the ramp to 2.5e-4, the hot guard, run
until stopped) with no warm start, the condition scramble off and its own seed, and nothing else moved (asserted):
  [0] qf30_ed_cmle_s1   seed 12345 (the recipe's)
  [1] qf30_ed_cmle_s2   seed 23456
  [2] qf30_ed_cmle_s3   seed 34567
A fresh first launch and self-resume afterwards (phase 1's job script, reading INDEX_h.tsv). P_B is live and the
batch sizer is leg e's, as in phase 1: leg g's frozen P_B is a snapshot of trained weights, which a fresh run lacks.
The reading is the forward draws' median excess and the forward Jensen, on training and on held-out molecules, by
step and by hour, against qf30_cmle_seed's curve at equal MLE steps (leg e). ROWS ARE APPENDED, never reordered:
the rows that add the force term to this recipe take the next indices.
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
# leg d: (run_name, the run it continues or None for a run resumed as itself, multiple of the source's rate, the
# forward force's reach in steps, whether the force's scale channel is live). ROW ORDER IS THE ARRAY INDEX: append only.
CONT = (('fwdF_lr5', None, 1, 1, True),
        ('fwdF_lr10', 'fwdF_lr5', 2, 1, True),
        ('fwdF_k5m', 'fwdF_lr5', 1, 5, False),
        ('fwdF_k5', 'fwdF_lr5', 1, 5, True),
        ('upb_lr5', None, 1, 1, True),
        ('upb_lr10', 'upb_lr5', 2, 1, True))
CONT_SCALE = 0.5        # the controller scale both sources carry in their checkpoints (their fixed_scale)
CONT_REACH_MAX = 0.3    # time units: the longest reach of the forward force a continued arm may take (k / T)
NO_SEED = '-'           # INDEX warm_src of a row that only resumes its own run: matches no checkpoint file
# the archive a seeded row of legs d and e loads, by source run: the last one each stopped run wrote (W&B last steps
# 109,590, 72,150 and 47,940 at archive periods 10,000, 10,000 and 5,000)
SEED_STEP = {'fwdF_lr5': 100_000, 'upb_lr5': 70_000, 'p1lr2': 45_000}
# leg e: (run_name, the run whose weights it starts from). ROW ORDER IS THE ARRAY INDEX: append only.
CMLE = (('cmle_seed', 'p1lr2'), ('cmle_lead', 'fwdF_lr5'))
# leg h: (run_name, seed) of the from-scratch conditional-MLE rows. ROW ORDER IS THE ARRAY INDEX: append only.
SCRATCH = (('ed_cmle_s1', 12345), ('ed_cmle_s2', 23456), ('ed_cmle_s3', 34567))
SCRATCH_SCALE = 2.0     # lr_control.fixed_scale of phase 1's leg 1 and of leg e: 2.5e-4 after burn-in and the ramp
# leg g: (run_name, batch_size, lr_control.fixed_scale). ROW ORDER IS THE ARRAY INDEX: append only.
MLEB = (('cmf_b1k_lr20', 1000, 2.0), ('cmf_b1k_lr40', 1000, 4.0),
        ('cmf_b4k_lr20', 4000, 2.0), ('cmf_b4k_lr40', 4000, 4.0))
MLEB_SCALE_MAX = 4.0        # fixed_scale 8 (1e-3) diverged in the 2026-10-06 smoke run (the module docstring, LEG G)
MLEB_BASE = 'cmle_seed'     # the leg-e arm leg g is built from
# leg f: (run_name, the run it continues, the starting value of P_F's force gate or None for no force term). ROW
# ORDER IS THE ARRAY INDEX: append only.
FORCE = (('frc_ctl', 'fwdF_lr5', None), ('frc_g0', 'fwdF_lr5', 0.0), ('frc_g1', 'fwdF_lr5', 0.1))
FORCE_SEED_STEP = 150_000   # the newest archive qf30_fwdF_lr5 had written when leg f was generated (2026-10-06)
FORCE_T_MIN = 0.8           # trajectory time from which a state gets a force
FORCE_MAX_SIGMA = 2.0       # cap on one step's force displacement, in that step's noise standard deviations
FORCE_CHUNK = 1000          # crystal states per trunk call
# the trunk: its path under the cluster checkpoints directory, its bytes, and the energy it was fitted at
FORCE_TRUNK = ('atom_trunk_oct05/at_c2_ft_late.pt', 595_833)
FORCE_TRUNK_ENERGY = {'temperature': 6.9, 'lj_coeff': 1.0}
FORCE_LEAVES = ('drift_force.checkpoint', 'drift_force.chunk') + tuple(
    f'model.force_drift_{k}' for k in ('fwd', 'bwd', 'learned', 'max_sigma', 't_min', 'differentiable'))
PRIOR_GUARD = """if [ "${HAVE}" != "${PRIOR_BYTES}" ]; then
    echo "FATAL: ${DATA}/${PRIOR} is ${HAVE} bytes, expected ${PRIOR_BYTES}" >&2; exit 1
fi
"""
FORCE_GUARD = """
# THE FORCE TERM (leg f): the trunk an arm with a gate reads, and the code on both sides that reads it.
TRUNK=%(trunk)s
HAVE=$(stat -c %%s ${TRUNK} 2>/dev/null || echo 0)
if [ "${HAVE}" != "%(bytes)d" ]; then
    echo "FATAL: ${TRUNK} is ${HAVE} bytes, expected %(bytes)d" >&2; exit 1
fi
for f in ${PROJECT_ROOT}/MXtalTools/mxtaltools/crystal_building/image_pairs.py ${WORKDIR}/models/crystal_force.py; do
    if [ ! -s "${f}" ]; then echo "FATAL: ${f} is missing -- git pull BOTH repositories" >&2; exit 1; fi
done
if ! grep -q 'FORCE_DRIFT_GFN_KEYS' ${WORKDIR}/checkpointing.py; then
    echo "FATAL: ${WORKDIR}/checkpointing.py does not read the force keys on a reload -- git pull gfn-diffusion" >&2; exit 1
fi
"""
NIGGLI_CHECK = """'Niggli triclinic penalty is OFF'\\" || exit 1
"""
FORCE_IMPORT = """        python -c \\"from models.crystal_force import CrystalDriftForce, TrunkForce\\" || exit 1
"""
# the job script's seed lookup (final_sep19's SEED_B) and the pinned one legs d and e run in its place
SEED_NEWEST = """        CK=$(ls -t ${CKPTS}/*${SRC}_*_step[0-9]*.pt 2>/dev/null | grep -v '_buffers.pt$' | head -1)
        if [ -z "${CK}" ]; then
            echo "FATAL: leg A arm ${SRC} has no step archive yet in ${CKPTS} (needs archive_period steps)" >&2; exit 1
        fi
"""
SEED_PINNED = """        # THE SEED STEP IS NAMED (INDEX column 7), not "the newest": a resumed source writes more.
        SRC_STEP=$(awk -F'\\t' -v n=${ROW} 'NR==n {print $7}' ${INDEX})
        if [ -z "${SRC_STEP}" ]; then
            echo "FATAL: arm ${ARM} has no _running.pt of its own and INDEX names no seed step for it" >&2; exit 1
        fi
        NA=$(ls ${CKPTS}/*${SRC}_*_step${SRC_STEP}.pt 2>/dev/null | grep -v '_buffers.pt$' | wc -l)
        if [ "${NA}" -ne 1 ]; then
            echo "FATAL: ${NA} matches for *${SRC}_*_step${SRC_STEP}.pt in ${CKPTS} (need exactly 1)" >&2; exit 1
        fi
        CK=$(ls ${CKPTS}/*${SRC}_*_step${SRC_STEP}.pt | grep -v '_buffers.pt$')
"""
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


def _write_index_pinned(path, rows):
    """INDEX with a seventh column: the step of the source archive a seeded row loads ('' on a resume row)."""
    with path.open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\tsrc_step\n')
        for r in rows:
            assert len(r) == 7, r
            f.write('\t'.join(r) + '\n')


def _pinned_sbatch(**fields):
    """final_sep19's job script with its newest-archive seed lookup replaced by the named step."""
    sb = fin.SBATCH.format(**fields)
    assert sb.count(SEED_NEWEST) == 1, "final_sep19's seed lookup moved"
    return sb.replace(SEED_NEWEST, SEED_PINNED)


def build_cont(run, source, rate, reach, scale_live=True):
    """A leg-b or leg-c arm continued under a new name, loaded in full from that arm (the module docstring, LEG D)."""
    cfg = copy.deepcopy(source)
    cfg['run_name'] = run
    cfg['load_weights_only'] = False    # a literal: the job script's weights-only substitution passes it by
    cfg['lr_control']['seed_lr'] = SEED_LR * rate
    fwd = cfg['protocols'][PROTOCOL]['stages'][1]['loss_coeffs']['fwd']
    fwd['path_grad_last_k'] = reach
    fwd['path_grad_scale'] = 1 if scale_live else 0
    return cfg


def check_cont(cfg, name, source, source_name, rate, reach, scale_live, extra_allowed=()):
    lc, src_lc = cfg['lr_control'], source['lr_control']
    # the rate: base x the scale the checkpoint restores; nothing else in the controller's block moves
    assert lc['mode'] == 'fixed' and cfg['lr_fused'] == 'auto' and cfg.get('max_lr') is None, name
    assert src_lc['seed_lr'] == SEED_LR and src_lc['fixed_scale'] == CONT_SCALE == lc['fixed_scale'], name
    assert lc['seed_lr'] == SEED_LR * rate and rate >= 1, (name, lc['seed_lr'])
    assert lc['fire_cut_factor'] == P2_FIRE_CUT and lc['hard_failure']['loss_excursion_k'] == RECIPE_IS[2], name
    # a full load into the source's own stage, with its buffers
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['load_weights_only'] is False, name
    assert cfg['continue_from_checkpoint'] is False and cfg['prior_model_name'] == fin.PRIOR_PLACEHOLDER, name
    assert not cfg['buffers'].get('fresh_on_switch'), f'{name}: the source buffers must be restored'
    mine, theirs = w3.problem_def(cfg), w3.problem_def(source)
    assert mine == theirs, f'{name}: problem identity differs from {source_name}; its checkpoint would be refused'
    st = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in st] == STAGES, name
    assert st[1]['on_enter'] == source['protocols'][PROTOCOL]['stages'][1]['on_enter'], name
    fwd = st[1]['loss_coeffs']['fwd']
    assert fwd['reward_grads'] == 1.0 and fwd['path_grad_last_k'] == reach >= 1, (name, fwd)
    assert source['protocols'][PROTOCOL]['stages'][1]['loss_coeffs']['fwd']['path_grad_scale'] == 1, source_name
    assert fwd['path_grad_scale'] == (1 if scale_live else 0), (name, fwd)
    # the two rows the 2026-10-05 smoke runs lost (the module docstring): no rate above twice the source's, and
    # no longer reach at a raised rate
    assert rate <= 2 and not (rate > 1 and reach > 1), f'{name}: rate x{rate} with reach {reach} diverged in the smoke'
    assert reach / cfg['integrator']['T'] <= CONT_REACH_MAX, f'{name}: the force reaches past {CONT_REACH_MAX} time units'
    assert (cfg['epochs'], cfg['archive_period'], cfg['archive_buffers']) == (P2_EPOCHS, P2_ARCHIVE, True), name
    # the source with the named leaves moved: all of them, and no others
    allowed = ['load_weights_only'] + (['lr_control.seed_lr'] if rate != 1 else []) + \
              ([f'{VC}.loss_coeffs.fwd.path_grad_last_k'] if reach != 1 else []) + \
              ([] if scale_live else [f'{VC}.loss_coeffs.fwd.path_grad_scale']) + list(extra_allowed)
    assert _moved(source, cfg) == sorted(allowed), (name, _moved(source, cfg))
    w3._scan_local_paths(cfg, name)
    fin.load_check(copy.deepcopy(cfg), name, STAGES)
    assert not _baseline_notices(cfg), (name, [str(v) for v in _baseline_notices(cfg)])


def main_cont(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size

    def committed(run):
        """A leg-b or leg-c arm as it ran: the committed file, which must be the one on disk."""
        name = f'{TAG}_{run}'
        text = w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{name}.yaml'], HERE)
        cfg = yaml.safe_load(text)
        on_disk = yaml.safe_load((HERE / f'{name}.yaml').read_text(encoding='utf-8'))
        assert on_disk == cfg, f'{name}.yaml is not the committed file'
        return cfg

    ran = {f'{TAG}_{run}' for run, *_ in ARMS} | {f'{TAG}_{run}' for run, _ in LIVE_PB}
    rows, new = [], {}
    for run, src, rate, reach, scale_live in CONT:
        name = f'{TAG}_{run}'
        if src is None:
            # resumed as itself: its committed file is the config, and its own _running.pt the only start
            assert name in ran and (rate, reach, scale_live) == (1, 1, True), name
            cfg = committed(run)
            assert cfg['load_weights_only'] == WO_PLACEHOLDER and cfg['lr_control']['fixed_scale'] == CONT_SCALE, name
            probe = copy.deepcopy(cfg)
            probe['load_weights_only'] = False      # what the job script resolves on a resubmission
            fin.load_check(probe, name, STAGES)
            rows.append((name, 'qm9full', 'resume', NO_SEED, PRIOR, str(prior_bytes), ''))
            continue
        source_name = f'{TAG}_{src}'
        assert source_name in ran and name not in ran, name
        source = committed(src)
        cfg = build_cont(run, source, rate, reach, scale_live)
        check_cont(cfg, name, source, source_name, rate, reach, scale_live)
        new[name] = cfg
        rows.append((name, 'qm9full', 'continued', source_name, PRIOR, str(prior_bytes), str(SEED_STEP[src])))
    # the job script finds an arm's files, and its source's, by `*<arm>_*`: no name may match another's
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + sorted(ran | set(new))
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    print(f'leg d, continuation (INDEX_d row = array index). A continued arm loads its source in full and trains at '
          f'seed_lr x the restored scale {CONT_SCALE:g}:')
    for i, (run, src, rate, reach, scale_live) in enumerate(CONT):
        if src is None:
            print(f'[{i}] {TAG}_{run:<12} resumed from its own _running.pt, unchanged')
        else:
            print(f'[{i}] {TAG}_{run:<12} {TAG}_{src} continued | rate {SEED_LR * rate * CONT_SCALE:.3g} (seed_lr '
                  f'{SEED_LR * rate:.3g}) | forward force over the last {reach} step{"s" if reach > 1 else ""} '
                  f'({reach / T:g} time units), {"means and scales" if scale_live else "means only"}')
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in new.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    _write_index_pinned(HERE / 'INDEX_d.tsv', rows)
    with (HERE / f'submit_{BATTERY}_d.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(_pinned_sbatch(
            wall=fin.WALL, last=len(rows) - 1, tag=TAG + 'd', battery=BATTERY, leg='d', ckpts=w3.CLUSTER_CKPTS,
            data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
            what=f'continuation of the stopped leg-b and leg-c arms. A "resume" row continues its own _running.pt; a '
                 f'"continued" row is a new run name loaded IN FULL (weights, optimizers, Z table, buffers) from the '
                 f"src_step archive of its warm_src arm on a first launch, and resumes itself afterwards. DO NOT PASS "
                 f'SRC_RUNNING: the resume rows rewrite those files (make.py, LEG D).'))
    print(f'wrote {len(new)} continued arms with INDEX_d.tsv ({len(rows)} rows) and submit_{BATTERY}_d.sbatch')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


def build_force(run, source, gate, trunk):
    """A leg-b arm continued under a new name with the force leaves added: a gate in P_F starting at `gate`, or,
    for None, every leaf present and no force term (the module docstring, LEG F)."""
    cfg = build_cont(run, source, 1, 1, True)
    cfg['model'].update(force_drift_fwd=gate, force_drift_bwd=None, force_drift_learned=True,
                        force_drift_max_sigma=FORCE_MAX_SIGMA, force_drift_t_min=FORCE_T_MIN,
                        force_drift_differentiable=False)
    cfg['drift_force'] = {'checkpoint': None if gate is None else trunk, 'chunk': FORCE_CHUNK}
    return cfg


def check_force(cfg, name, source, source_name, gate, trunk):
    check_cont(cfg, name, source, source_name, 1, 1, True, extra_allowed=FORCE_LEAVES)
    m = cfg['model']
    assert m['force_drift_fwd'] == gate and (gate is None or 0 <= gate <= 1), (name, gate)
    assert (m['force_drift_learned'], m['force_drift_max_sigma'], m['force_drift_t_min'],
            m['force_drift_differentiable']) == (True, FORCE_MAX_SIGMA, FORCE_T_MIN, False), name
    assert cfg['drift_force'] == {'checkpoint': None if gate is None else trunk, 'chunk': FORCE_CHUNK}, name
    # P_B is frozen on entering the TB stage and the archive carries that snapshot, written without a gate: a
    # backward gate would be refused when the snapshot loads
    assert m['force_drift_bwd'] is None and 'freeze_pb' in cfg['protocols'][PROTOCOL]['stages'][1]['on_enter'], name
    # what train.py _build_drift_force refuses a force term without, and the energy the trunk was fitted at
    assert cfg['energy_function'] == 'elj' and list(cfg['z_primes']) == [1], name
    assert cfg['temperature_conditioning'] is False and cfg['compile_policy'] is False, name
    assert {k: cfg['energy_config'][k] for k in FORCE_TRUNK_ENERGY} == FORCE_TRUNK_ENERGY, name
    # the window opens on a grid time, so the count of states that reach the trunk is the same for every row
    assert 0 < FORCE_T_MIN < 1 and abs(FORCE_T_MIN * T - round(FORCE_T_MIN * T)) < 1e-9, FORCE_T_MIN


def main_force(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size
    trunk = f'{w3.CLUSTER_CKPTS}/{FORCE_TRUNK[0]}'

    def committed(run):
        """The source arm as it ran: the committed file, which must be the one on disk."""
        name = f'{TAG}_{run}'
        cfg = yaml.safe_load(w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{name}.yaml'], HERE))
        on_disk = yaml.safe_load((HERE / f'{name}.yaml').read_text(encoding='utf-8'))
        assert on_disk == cfg, f'{name}.yaml is not the committed file'
        return cfg

    ran = ({f'{TAG}_{run}' for run, *_ in ARMS} | {f'{TAG}_{run}' for run, _ in LIVE_PB}
           | {f'{TAG}_{run}' for run, *_ in CONT} | {f'{TAG}_{run}' for run, _ in CMLE})
    rows, new = [], {}
    for run, src, gate in FORCE:
        name, source_name = f'{TAG}_{run}', f'{TAG}_{src}'
        assert source_name in ran and name not in ran, name
        source = committed(src)
        cfg = build_force(run, source, gate, trunk)
        check_force(cfg, name, source, source_name, gate, trunk)
        new[name] = cfg
        rows.append((name, 'qm9full', 'continued', source_name, PRIOR, str(prior_bytes), str(FORCE_SEED_STEP)))
    assert [g for _, _, g in FORCE].count(None) == 1 and FORCE[0][2] is None, 'row 0 is the one control'
    # the job script finds an arm's files, and its source's, by `*<arm>_*`: no name may match another's
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + sorted(ran | set(new))
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    in_window = round(T * (1 - FORCE_T_MIN)) + 1
    print(f'leg f, force term (INDEX_f row = array index). Every row loads {TAG}_{FORCE[0][1]} step {FORCE_SEED_STEP:,} '
          f'in full; the force acts from t = {FORCE_T_MIN:g} ({in_window} of the {T + 1} states of a trajectory), '
          f'capped at {FORCE_MAX_SIGMA:g} noise std per step; trunk {trunk}:')
    for i, (run, src, gate) in enumerate(FORCE):
        print(f'[{i}] {TAG}_{run:<8} ' + ('no force term (control)' if gate is None
                                          else f'P_F gate learned from {gate:g}; no term in P_B (frozen)'))
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in new.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    _write_index_pinned(HERE / 'INDEX_f.tsv', rows)
    sb = _pinned_sbatch(
        wall=fin.WALL, last=len(rows) - 1, tag=TAG + 'f', battery=BATTERY, leg='f', ckpts=w3.CLUSTER_CKPTS,
        data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
        what=f'the force term in P_F. Every row is a new run name loaded IN FULL (weights, optimizers, Z table, '
             f'buffers) from the src_step archive of its warm_src arm on a first launch, and resumes itself '
             f'afterwards; row 0 is the control. DO NOT PASS SRC_RUNNING (make.py, LEG F).')
    # the trunk and the code that reads it are checked before the seed is resolved, and imported before the run
    assert sb.count(PRIOR_GUARD) == 1 and sb.count(NIGGLI_CHECK) == 1, "final_sep19's job script moved"
    sb = sb.replace(PRIOR_GUARD, PRIOR_GUARD + FORCE_GUARD % {'trunk': trunk, 'bytes': FORCE_TRUNK[1]})
    sb = sb.replace(NIGGLI_CHECK, NIGGLI_CHECK + FORCE_IMPORT)
    with (HERE / f'submit_{BATTERY}_f.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    print(f'wrote {len(new)} arms with INDEX_f.tsv ({len(rows)} rows) and submit_{BATTERY}_f.sbatch')
    print(f'the trunk must be at {trunk} with {FORCE_TRUNK[1]:,} bytes; the seed is {TAG}_{FORCE[0][1]} step '
          f'{FORCE_SEED_STEP:,} with its _buffers.pt beside it')


def build_scratch(run, seed):
    """Phase 1's leg-1 recipe from nothing, conditionally: no warm start, the scramble off, its own seed (the module
    docstring, LEG H)."""
    cfg = build_p1(run, SCRATCH_SCALE, None)
    cfg['seed'] = seed
    stages = cfg['protocols'][PROTOCOL]['stages']
    assert stages[0]['flags']['scramble_conditions'] is True, run
    stages[0]['flags']['scramble_conditions'] = False
    return cfg


def check_scratch(cfg, name, seed, p1lr2):
    check_p1(cfg, name, SCRATCH_SCALE, None)
    st = cfg['protocols'][PROTOCOL]['stages']
    assert st[0]['bwd_sampling_mode'] == 'dataset' and st[0]['loss_coeffs']['bwd']['mle'] == 1.0, name
    assert cfg['embedding_conditioning'] is True and not cfg.get('freeze_backward_policy'), name
    assert cfg['seed'] == seed and isinstance(seed, int), (name, cfg['seed'])
    # the committed phase-1 leg 1 (which leg e's arms are, scramble aside) with these leaves moved, and no others:
    # a fresh start in place of the weights-only seed, the scramble, and the seed where it is not the recipe's
    assert w3.problem_def(cfg) == w3.problem_def(p1lr2), f'{name}: not the problem of {TAG}_{LEGS[LIVE][0]}'
    allowed = ['checkpoint_name', 'load_weights_only',
               f'protocols.{PROTOCOL}.stages[train_prior].flags.scramble_conditions']
    allowed += ['seed'] if seed != p1lr2['seed'] else []
    assert _moved(p1lr2, cfg) == sorted(allowed), (name, _moved(p1lr2, cfg))
    w3.load_check(cfg, name)


def main_scratch(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size
    leg1 = f'{TAG}_{LEGS[LIVE][0]}'
    p1lr2 = yaml.safe_load(w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{leg1}.yaml'], HERE))
    assert yaml.safe_load((HERE / f'{leg1}.yaml').read_text(encoding='utf-8')) == p1lr2, f'{leg1}.yaml is not committed'
    ran = ({f'{TAG}_{run}' for run, *_ in ARMS} | {f'{TAG}_{run}' for run, _ in LIVE_PB}
           | {f'{TAG}_{run}' for run, *_ in CONT} | {f'{TAG}_{run}' for run, _ in CMLE}
           | {f'{TAG}_{run}' for run, *_ in FORCE} | {f'{TAG}_{run}' for run, *_ in MLEB})
    assert len({seed for _, seed in SCRATCH}) == len(SCRATCH), 'two rows share a seed'
    rows, new = [], {}
    for run, seed in SCRATCH:
        name = f'{TAG}_{run}'
        assert name not in ran, name
        cfg = build_scratch(run, seed)
        check_scratch(cfg, name, seed, p1lr2)
        new[name] = cfg
        rows.append(f'{name}\tqm9full\tfresh\t-\t{PRIOR}\t{prior_bytes}\n')
    names = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + sorted(ran | set(new))
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    rate = SCRATCH_SCALE * SEED_LR
    print(f'leg h, conditional MLE from scratch (INDEX_h row = array index): phase 1\'s leg-1 recipe at {rate:g}, '
          f'scramble off, no warm start:')
    for i, (run, seed) in enumerate(SCRATCH):
        print(f'[{i}] {TAG}_{run:<12} seed {seed}')
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in new.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    with (HERE / 'INDEX_h.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\tfamily\tstart\twarm_src\tprior\tprior_bytes\n')
        f.writelines(rows)
    array = '#SBATCH --array=0-__LAST__'
    old = '# __BATTERY__: phase-1 MLE on the Niggli P-1 priors, warm (mle09 best, weights-only) and fresh.'
    index, index_note = '${ARMS}/INDEX.tsv', '# Arm = row of INDEX.tsv'
    assert all(nig.SBATCH.count(t) == 1 for t in (array, old, index_note)) and nig.SBATCH.count(index) == 4, \
        'the mle_nig_sep17 job script moved'
    sb = (nig.SBATCH.replace(array, f'#SBATCH --array=0-{len(rows) - 1}')
          .replace(old, f'# __BATTERY__ leg h: conditional MLE from scratch (phase 1 with the condition scramble off), '
                        f'one row per seed; the control of the force-term and graph-policy comparisons. Stopped by '
                        f'hand; archives every 5000 steps.')
          .replace(index, '${ARMS}/INDEX_h.tsv').replace(index_note, '# Arm = row of INDEX_h.tsv')
          .replace('__TAG__', TAG + 'h').replace('__BATTERY__', BATTERY)
          .replace('__CKPTS__', w3.CLUSTER_CKPTS).replace('__DATA__', w3.CLUSTER_DATA))
    assert 'INDEX.tsv' not in sb, 'a reference to phase 1\'s INDEX survived'
    with (HERE / f'submit_{BATTERY}_h.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(sb)
    print(f'wrote {len(new)} arms with INDEX_h.tsv ({len(rows)} rows) and submit_{BATTERY}_h.sbatch')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


def build_cmle(run, p1):
    """Phase 1's leg-1 config with the condition scramble off, under the phase-2 job script's seed placeholders
    (the module docstring, LEG E)."""
    cfg = copy.deepcopy(p1)
    cfg['run_name'] = run
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['prior_model_name'] is None, run
    cfg['load_weights_only'] = True     # the first launch; main_cmle writes the placeholder the job script fills
    cfg['continue_from_checkpoint'] = False
    stages = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in stages] == ['train_prior'] and stages[0]['flags']['scramble_conditions'] is True
    stages[0]['flags']['scramble_conditions'] = False
    return cfg


def check_cmle(cfg, name, p1, source, source_name):
    check_p1(dict(cfg, continue_from_checkpoint=w3.CONT_PLACEHOLDER), name, 2.0, True)
    st = cfg['protocols'][PROTOCOL]['stages']
    assert st[0]['flags'] == dict(p1['protocols'][PROTOCOL]['stages'][0]['flags'], scramble_conditions=False), name
    assert st[0]['bwd_sampling_mode'] == 'dataset' and st[0]['loss_coeffs']['bwd']['mle'] == 1.0, name
    # an embedding condition is what the scramble acts on, and the conditioner is what leaving it off trains
    assert cfg['embedding_conditioning'] is True and not cfg.get('freeze_backward_policy'), name
    mine, theirs = w3.problem_def(cfg), w3.problem_def(source)
    assert mine == theirs, f'{name}: problem identity differs from {source_name}; its weights would be refused'
    assert cfg['model'] == source['model'], f'{name}: not the model of {source_name}'
    assert cfg['integrator'] == source['integrator'], f'{name}: not the trajectory of {source_name}'
    # phase 1's leg 1 with the scramble off; continue_from_checkpoint is that leg's job-script placeholder, which this
    # leg's job script does not fill
    moved = _moved(p1, cfg)
    assert moved == ['continue_from_checkpoint',
                     f'protocols.{PROTOCOL}.stages[train_prior].flags.scramble_conditions'], (name, moved)
    for weights_only in (True, False):  # the first launch and a resubmission
        fin.load_check(dict(copy.deepcopy(cfg), load_weights_only=weights_only), name, ['train_prior'])


def main_cmle(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size

    def committed(run):
        name = f'{TAG}_{run}'
        cfg = yaml.safe_load(w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{name}.yaml'], HERE))
        on_disk = yaml.safe_load((HERE / f'{name}.yaml').read_text(encoding='utf-8'))
        assert on_disk == cfg, f'{name}.yaml is not the committed file'
        return cfg

    p1 = committed(SEED_LEG)
    rows, new = [], {}
    for run, src in CMLE:
        name, source_name = f'{TAG}_{run}', f'{TAG}_{src}'
        cfg = build_cmle(run, p1)
        check_cmle(cfg, name, p1, committed(src), source_name)
        cfg['load_weights_only'] = WO_PLACEHOLDER
        new[name] = cfg
        rows.append((name, 'qm9full', 'weights', source_name, PRIOR, str(prior_bytes), str(SEED_STEP[src])))
    ran = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + [f'{TAG}_{run}' for run, *_ in ARMS] + \
          [f'{TAG}_{run}' for run, _ in LIVE_PB] + [f'{TAG}_{run}' for run, src, *_ in CONT if src is not None]
    names = ran + list(new)
    assert len(set(names)) == len(names), names
    # the job script finds an arm's files, and its source's, by `*<arm>_*`: no name may match another's
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    print(f'leg e, conditional MLE (INDEX_e row = array index): {TAG}_{SEED_LEG} with scramble_conditions false, '
          f'weights-only first launch from the named archive, rate {2.0 * SEED_LR:g} after burn-in and ramp:')
    for i, (run, src) in enumerate(CMLE):
        print(f'[{i}] {TAG}_{run:<10} weights of {TAG}_{src} at step {SEED_STEP[src]:,}')
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in new.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    _write_index_pinned(HERE / 'INDEX_e.tsv', rows)
    with (HERE / f'submit_{BATTERY}_e.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(_pinned_sbatch(
            wall=fin.WALL, last=len(rows) - 1, tag=TAG + 'e', battery=BATTERY, leg='e', ckpts=w3.CLUSTER_CKPTS,
            data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
            what=f'conditional MLE: phase 1 ({TAG}_{SEED_LEG}) with scramble_conditions false, weights-only first '
                 f'launch from the src_step archive of the warm_src arm, full resume afterwards; stopped by hand '
                 f'(make.py, LEG E).'))
    print(f'wrote {len(new)} arms with INDEX_e.tsv and submit_{BATTERY}_e.sbatch')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


def build_mleb(run, base, batch, scale):
    """Leg e's conditional-MLE arm with P_B frozen from start-up, the batch pinned at `batch` and the rate at
    `scale` x seed_lr (the module docstring, LEG G)."""
    cfg = copy.deepcopy(base)
    cfg['run_name'] = run
    cfg['freeze_backward_policy'] = True
    cfg.update(batch_size=batch, max_batch_size=batch, grow_batch_size=False, batch_util_target=0.0)
    cfg['lr_control']['fixed_scale'] = scale
    return cfg


def check_mleb(cfg, name, base, p1, batch, scale):
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['seed_lr'] == SEED_LR and lc['fixed_scale'] == scale, name
    assert cfg['lr_back'] == 'auto' and cfg.get('max_lr') is None, name     # the rate train_prior steps, unrailed
    assert _guard(cfg) == HOT_GUARD and lc['burn_in_scale'] < scale <= MLEB_SCALE_MAX, (name, _guard(cfg), scale)
    assert (cfg['batch_size'], cfg['max_batch_size'], cfg['grow_batch_size'], cfg['batch_util_target']) == \
        (batch, batch, False, 0.0), name
    # at or above the accumulation floor every step is one plain optimizer step (batch-size.md)
    assert batch >= cfg['fused_grad_accum_min_samples'], name
    assert cfg['freeze_backward_policy'] is True and cfg['model']['learn_pb'] is True, name
    st = cfg['protocols'][PROTOCOL]['stages']
    assert [s['name'] for s in st] == ['train_prior'] and st[0]['flags']['scramble_conditions'] is False, name
    assert st[0]['train_mode'] == 'bwd' and st[0]['bwd_sampling_mode'] == 'dataset', name
    assert w3.problem_def(cfg) == w3.problem_def(p1), f'{name}: problem identity differs from the seed leg'
    assert cfg['checkpoint_name'] == fin.PLACEHOLDER and cfg['load_weights_only'] == WO_PLACEHOLDER, name
    moved = _moved(base, cfg)
    want = sorted(['freeze_backward_policy', 'grow_batch_size', 'max_batch_size', 'batch_util_target']
                  + (['batch_size'] if batch != base['batch_size'] else [])
                  + (['lr_control.fixed_scale'] if scale != base['lr_control']['fixed_scale'] else []))
    assert moved == want, (name, moved)
    w3._scan_local_paths(cfg, name)
    for weights_only in (True, False):  # the first launch and a resubmission
        fin.load_check(dict(copy.deepcopy(cfg), load_weights_only=weights_only), name, ['train_prior'])


def main_mleb(argv):
    dry = '--dry' in argv
    dirty = w3.dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted:\n  ' + '\n  '.join(dirty))
    prior_bytes = (LOCAL_PRIORS / PRIOR).stat().st_size

    def committed(run):
        name = f'{TAG}_{run}'
        cfg = yaml.safe_load(w3._git(['show', f'HEAD:energy_sampling/configs/{BATTERY}/{name}.yaml'], HERE))
        on_disk = yaml.safe_load((HERE / f'{name}.yaml').read_text(encoding='utf-8'))
        assert on_disk == cfg, f'{name}.yaml is not the committed file'
        return cfg

    base, p1 = committed(MLEB_BASE), committed(SEED_LEG)
    seed = dict(CMLE)[MLEB_BASE]
    assert seed == SEED_LEG and base['batch_size'] == 1000 and base['lr_control']['fixed_scale'] == 2.0, MLEB_BASE
    assert len({(b, s) for _, b, s in MLEB}) == len(MLEB) and MLEB[0][1:] == (1000, 2.0), 'row 0 is the control'
    rows, new = [], {}
    for run, batch, scale in MLEB:
        name = f'{TAG}_{run}'
        cfg = build_mleb(run, base, batch, scale)
        check_mleb(cfg, name, base, p1, batch, scale)
        new[name] = cfg
        rows.append((name, 'qm9full', 'weights', f'{TAG}_{seed}', PRIOR, str(prior_bytes), str(SEED_STEP[seed])))
    ran = [f'{TAG}_{run_name}' for run_name, _, _ in LEGS] + [f'{TAG}_{run}' for run, *_ in ARMS] + \
          [f'{TAG}_{run}' for run, _ in LIVE_PB] + [f'{TAG}_{run}' for run, src, *_ in CONT if src is not None] + \
          [f'{TAG}_{run}' for run, _ in CMLE] + [f'{TAG}_{run}' for run, *_ in FORCE]
    names = ran + list(new)
    assert len(set(names)) == len(names), names
    # the job script finds an arm's files, and its source's, by `*<arm>_*`: no name may match another's
    assert not any(a != b and f'{a}_' in f'{b}_' for a in names for b in names), names
    print(f'leg g, conditional MLE over batch and rate (INDEX_g row = array index): {TAG}_{MLEB_BASE} with P_B frozen '
          f'and the batch pinned, weights of {TAG}_{seed} at step {SEED_STEP[seed]:,}:')
    for i, (run, batch, scale) in enumerate(MLEB):
        print(f'[{i}] {TAG}_{run:<13} batch {batch:>5,} | rate {scale * SEED_LR:g} (fixed_scale {scale:g}) | '
              f'{1_212_915 / batch:,.0f} steps a pass over the prior rows')
    if dry:
        print('--dry: checks passed, nothing written')
        return
    for name, cfg in new.items():
        path = HERE / f'{name}.yaml'
        with path.open('w', encoding='utf-8', newline='\n') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == cfg, f'{path} does not read back as written'
    _write_index_pinned(HERE / 'INDEX_g.tsv', rows)
    with (HERE / f'submit_{BATTERY}_g.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(_pinned_sbatch(
            wall=fin.WALL, last=len(rows) - 1, tag=TAG + 'g', battery=BATTERY, leg='g', ckpts=w3.CLUSTER_CKPTS,
            data=w3.CLUSTER_DATA, seed_block=fin.SEED_B,
            what=f'conditional MLE over batch size and rate, P_B frozen: {TAG}_{MLEB_BASE} with the batch pinned, '
                 f'weights-only first launch from the src_step archive of the warm_src arm, full resume afterwards; '
                 f'stopped by hand. With growth off an out-of-memory cut stands: read Batch Size (make.py, LEG G).'))
    print(f'wrote {len(new)} arms with INDEX_g.tsv and submit_{BATTERY}_g.sbatch')
    print(f'the prior must be at {w3.CLUSTER_DATA}/{PRIOR} with {prior_bytes:,} bytes, beside {CONDITIONS} and {TEST}')


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
    if argv[:1] == ['cont']:
        return main_cont(argv)
    if argv[:1] == ['cmle']:
        return main_cmle(argv)
    if argv[:1] == ['mleb']:
        return main_mleb(argv)
    if argv[:1] == ['force']:
        return main_force(argv)
    if argv[:1] == ['scratch']:
        return main_scratch(argv)
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
