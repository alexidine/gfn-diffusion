"""Local (RTX 5080 laptop, 16GB) experiment configs, derived from mk_dev.yaml.

mk_dev.yaml is USER-OWNED and is only ever READ here. Every config written by
this script uses its own run_name, so the checkpoint prefix
({tag}_{run_name}_{problem}) is distinct and no local run can clobber the
dev_mk_dev_* checkpoint set.

Resume mode is chosen per hypothesis, not by default:
  abort   -- partially-converged PHASE 2 (step30000, load_full). The hypothesis
             is about a phase-2 policy detonating, so phase 1 is not in scope
             and re-running it would only add an hour.
  ring    -- same phase-2 resume; the ripple only exists in a fused stage.
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
CKPT_P2 = ('dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92'
           '_step30000.pt')
LR_KEYS = ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resume_phase2(cfg, ckpt=CKPT_P2):
    cfg['checkpoint_name'] = ckpt
    cfg['load_weights_only'] = False      # full resume: stage, step, optimizers, buffers
    cfg['continue_from_checkpoint'] = False
    return cfg


def write(cfg, name, tag='local_aug02'):
    cfg['run_name'] = name
    cfg['tag'] = tag
    out = HERE / f'{name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
    print(f"wrote {out.name:24s} run_name={name}")
    return out


# ---------------------------------------------------------------- abort test
# HYPOTHESIS: a policy driven far past its LR ceiling saturates, freezes
# grad_norm_pre_clip bitwise, and _frozen_training_state aborts the run.
# Falsified if the run either (a) recovers via the reset-tier rewind and keeps
# training, or (b) freezes and is NOT caught.
cfg = resume_phase2(base())
for k in LR_KEYS:
    cfg[k] = 1.0e-2                       # ~50x over any observed ceiling
cfg['adaptive_lr']['cut_ratio'] = 1.0     # controller may not rescue it
cfg['terminal_frozen_steps'] = 500        # shortened so the test is minutes
cfg['epochs'] = 34000                     # resume is at 30k
cfg['eval_period'] = 250
write(cfg, 'abort_test')

# ------------------------------------------------- fix validation (all three)
# HYPOTHESIS: with the post-launch fixes, the SAME detonation that produced an
# unbounded runaway (abort_test / jgyk2lzl) and an infinite rewind loop
# (cluster d7z705wc) now terminates. Fresh run_name => no '<prefix>_best.pt'
# exists, so this exercises:
#   fix 2  fire_loss_spike's no-rewind-target guard -> warn + force_ratio 0.5
#   fix 3  max_reloads cap -> UNRECOVERABLE abort (set to 3 so it lands fast)
#   fix 1  the non-finite streak channel, if the gradients go all-non-finite
#          before the reload cap is reached
# Falsified if it runs on past ~step 31000 still training.
cfg = resume_phase2(base())
for k in LR_KEYS:
    cfg[k] = 1.0e-2
cfg['adaptive_lr']['cut_ratio'] = 1.0     # the aug02 setting that caused the loop
cfg['terminal_frozen_steps'] = 500
cfg['max_reloads'] = 3
cfg['epochs'] = 33000
cfg['eval_period'] = 250
write(cfg, 'fix_validation')

# ------------------------------------------ fix validation, REWIND-LOOP branch
# fix_validation at 1e-2 detonates straight into all-non-finite gradients, so
# channel 1 aborts first and the rewind path is never reached -- non-finite
# readings never get to check_spike (the stale-value hole). To exercise fixes 2
# and 3 the detonation has to stay FINITE and large, which is what the cluster
# arm d7z705wc actually did (2471 -> 1.3e5 -> 4.2e6 -> 9.2e7, all finite, 11
# tripwire fires). 3e-3 is ~2x over the T=25 ceiling rather than ~50x.
# EXPECT: repeated reset-tier fires -> no-rewind-target warning + forced cut ->
# and if it still re-detonates, UNRECOVERABLE at reload 4.
cfg = resume_phase2(base())
for k in LR_KEYS:
    cfg[k] = 3.0e-3
cfg['adaptive_lr']['cut_ratio'] = 1.0
cfg['terminal_frozen_steps'] = 500
cfg['max_reloads'] = 3
cfg['epochs'] = 33000
cfg['eval_period'] = 250
write(cfg, 'fix_validation_rewind')

# ---------------------------------------------------------- false-positive control
# HYPOTHESIS: the same detector, same patience, does NOT fire on a healthy run.
# This is the arm that matters -- an abort that also kills good runs is worse
# than no abort. Identical to abort_test except the LR is mk_dev's own.
cfg = resume_phase2(base())
cfg['terminal_frozen_steps'] = 500
cfg['epochs'] = 34000
cfg['eval_period'] = 250
write(cfg, 'abort_control')

# ------------------------------------------------------------- ringing probe
# HYPOTHESIS (the open degeneracy in flat-direction-limit-cycle-phase2): the
# ~550-step phase-2 ripple is either a real ~550-step mode or an ALIAS of a
# ~2-step edge-of-stability mode, because every scalar is logged 1-in-10.
# Discriminator = lag-1 autocorrelation of the DETRENDED per-step step_var:
#   real ~550-step mode -> rho_1 near +1
#   aliased 2-step mode -> rho_1 near -1
# Resumes into the FUSED stage because the ripple does not exist before it
# (phase 1 pins the backward policy to data and has no P_F/P_B trade), so the
# small-T fast-phase-1 route would not reproduce the phenomenon at all.
cfg = resume_phase2(base())
cfg['per_step_probe_steps'] = 2000
cfg['epochs'] = 32500                     # 2000 probe steps + margin
cfg['eval_period'] = 500                  # keep eval out of the probe window's way
write(cfg, 'ring_probe')

# --------------------------------------------- ringing probe, CALIBRATION panel
# The first probe (vlqklgmy) measured step_var/terminal_var only and came back
# near-white at T=10 -- ACF to zero by lag 39, the whole 100-1200-step band
# holding 1.7% of power. That is a null on an observable
# flat-direction-limit-cycle-phase2 already predicts is quiet at T=10, where the
# ~120 step-dims are ~6x stiffer and the mode surfaces in slope_err /
# intercept_err instead. _per_step_probe now also captures the calibration panel,
# so this rerun asks the slow-mode question on the coordinate that should carry
# it at this T.
#   H_slow confirmed  -> slope_err/intercept_err ACF shows a peak at 400-1000
#   H_slow refuted    -> calibration is white too, and no observable rings at T=10
cfg = resume_phase2(base())
cfg['per_step_probe_steps'] = 2000
cfg['epochs'] = 32500
cfg['eval_period'] = 500
write(cfg, 'ring_probe_cal')

# ----------------------------------------------------------------- SubTB tests
# WHY. TB's residual is log Z + sum log P_F - log R - sum log P_B, so with log Z
# at its optimum E[w] the loss IS Var(w) over trajectories: the objective is
# "flatten the log importance weight", and its gradient is a REINFORCE-style
# credit assignment where the advantage is a WHOLE-TRAJECTORY deviation. That is
# the nonlocality. SubTB/DB localize it, but only by replacing a single learned
# SCALAR (log Z) with a learned FIELD F(x,t) -- so they trade gradient variance
# for estimation burden, and if F is bad its error enters the policy gradient
# directly. HYPOTHESIS for the blow-ups: it is F's error, not the policy's.
#
# This matters now because [[invariant-convergence-rate]] found a 16x LR sweep
# moved the healthy-regime rate by ~2.2x with no systematic dependence. If the
# rate is set by credit-assignment variance, then no knob that leaves the credit
# STRUCTURE alone can move it -- and SubTB/DB are the only knobs on our list
# that change it. So this is the experiment that result points at.
#
# The naive stage already alternates fwd = {tb, freeze_policy} (Z only) against
# bwd = {tb} (policy only). Under SubTB that is flow-field-only vs policy-only,
# i.e. the field gets its own training signal with the policy detached -- the
# structure the hypothesis wants, for free.
def subtb_variant(cfg, lam, name, note_lr=None):
    cfg['model']['full_flow'] = True          # scalar log Z -> F(x,t) field
    for mode in ('fwd_loss_coeffs', 'bwd_loss_coeffs', 'replay_loss_coeffs'):
        cfg[mode]['tb'] = 0.0
        cfg[mode]['subtb'] = 1.0
        cfg[mode]['subtb_lambda'] = lam
    for st in cfg['protocol']['stages']:
        lc = st.setdefault('loss_coeffs', {})
        if st['name'] == 'naive':
            lc['fwd'] = {'subtb': 1.0, 'tb': 0.0, 'freeze_policy': 1.0}
            lc['bwd'] = {'subtb': 1.0, 'tb': 0.0}
            lc['replay'] = {'subtb': 1.0, 'tb': 0.0}
        else:
            # PHASE 1 MUST STAY CLEAN. A stage override MERGES onto the base
            # block, so base subtb: 1.0 leaks into train_prior, whose override
            # only names mle/repeats/tbc. Observed live (gqtl7l87): bwd/subtb
            # ran at ~60 through phase 1 while gradnorm/flow_model stayed 0 --
            # freeze_z detaches the flow, so the term had no way to TRAIN the
            # field and instead pushed the POLICY to satisfy sub-trajectory
            # balance against a random, never-updated critic. That is the exact
            # bad-critic-poisons-policy mechanism, running in the one phase that
            # was supposed to be identical to the TB baseline.
            for br in ('fwd', 'bwd', 'replay'):
                lc.setdefault(br, {})['subtb'] = 0.0
                lc[br].setdefault('tb', 0.0)
    if note_lr:
        for k in LR_KEYS:
            cfg[k] = note_lr

    # TWO full_flow INCOMPATIBILITIES, both found live on the first attempt and
    # both sufficient on their own to make SubTB "just blow up":
    #
    # (1) lr_flow 0.1 is calibrated for flow_model = LearnableScalar -- a convex
    #     1-D problem where O(lr) jitter is the only cost. full_flow makes it a
    #     4-layer MLP, and mk_dev's own comment says nets want 1.0e-4. Left
    #     alone, the field head trains at ~1000x its appropriate LR. This is
    #     almost certainly the historical blow-up, and it is not SubTB's fault.
    cfg['lr_flow'] = 1.0e-4
    #
    # (2) z_calibration's rollout asserts gradient touches flow_model ONLY, and
    #     under full_flow the head consumes the SHARED s_emb/t_emb so grad
    #     legitimately reaches t_model. The guard is right, not over-strict:
    #     owner.zero_grad() only clears flow params, so stray t_model grads
    #     would leak into the next real step. Proper fix is detaching the
    #     embeddings for that rollout; for an exploratory run just turn it off.
    cfg['z_calibration']['enabled'] = False

    # expect blow-ups: let them show their shape, but cap the waste
    cfg['max_reloads'] = 6
    cfg['terminal_frozen_steps'] = 2000
    cfg['epochs'] = 20000
    cfg['checkpoint_name'] = None      # flow head changes shape; no cross-run resume
    cfg['continue_from_checkpoint'] = True   # but DO resume this run's own phase 1
    cfg['reuse_prior'] = False
    return write(cfg, name)


# lambda 0.95 = long credit windows, nearly TB. lambda 0.5 = short windows,
# nearly DB. If the hypothesis is right, the SHORT-window arm should be the one
# that suffers more from a bad field (more sub-residuals, each needing F).
subtb_variant(base(), 0.95, 'subtb_l95')
subtb_variant(base(), 0.50, 'subtb_l50')

# -------------------------------------- FORWARD path/reward gradient (retry)
# The fwd branch currently runs traj_grads 0 AND reward_grads 0, so it learns
# ONLY from the scalar residual: the reward is a differentiable function of x_T
# and its derivative is discarded. Previous attempts to turn this on were "very
# destabilizing" -- but that was all-or-nothing, and there are TWO independent
# destabilizers with different fixes:
#   (i)  BPTT Jacobian product over all T reparameterized SDE steps
#        -> fixed by path_grad_last_k (truncate to the final k)
#   (ii) d log R / d x_T is near-singular for LJ whenever atoms clash, so a few
#        overlapping samples dominate the batch gradient
#        -> fixed by reward_grad_clip (clip at the SOURCE; the global grad clip
#           cannot separate the reward path from the density path once summed)
# Full-T with unclipped LJ gradients fails for reasons that say nothing about
# whether truncated-and-clipped works.
#
# All arms resume from the SAME phase-2 checkpoint as ring_probe, which is
# therefore a matched, already-collected CONTROL (same resume, same window,
# unmodified). Phase 1 is untouched, so nothing here can affect it.
#
# SCORE ON THE fwd/bwd tb_err RATIO, not the level. That ratio is pinned near
# 2.7x across every run and quartile ([[invariant-convergence-rate]]); if the
# fwd deficit really is the missing path/reward gradient, a fix must CLOSE it.
# A change that lowers fwd/tb_err but leaves the ratio at 2.7x improved the
# shared factor, not forward training.
def fwd_grad_variant(k, reward_grads, clip, name):
    cfg = resume_phase2(base())
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = k          # 0 = off (bitwise-identical to today)
    fc['reward_grads'] = reward_grads
    fc['reward_grad_clip'] = clip       # 0 = off
    cfg['epochs'] = 33000               # ~3000 steps past the resume
    cfg['eval_period'] = 250            # faster readout on a short window
    cfg['terminal_frozen_steps'] = 1000
    cfg['max_reloads'] = 4
    return write(cfg, name)


# k=1 clipped: the candidate. k=1 unclipped: does it reproduce the historical
# blow-up? If unclipped dies and clipped does not, stiff LJ gradients were the
# destabilizer and the diagnosis is confirmed. k=1 with reward_grads OFF
# isolates the path term from the reward term.
fwd_grad_variant(1, 1.0, 10.0, 'fpg_k1_rg1_clip10')
fwd_grad_variant(1, 1.0, 0.0, 'fpg_k1_rg1_noclip')
fwd_grad_variant(1, 0.0, 0.0, 'fpg_k1_rg0')

# ------------------------------ SubTB with the fwd branch training the POLICY
# In the stock `naive` stage the fwd branch is Z-ONLY (freeze_policy: 1), so the
# policy is trained exclusively by bwd (data-anchored) and replay (buffer) --
# which makes fwd/tb_err a HELD-OUT metric, not a training loss. That reframes
# the observed run: replay/tb_err crossing BELOW fwd/tb_err (0.70x, against a
# TB norm near 2.0x) with replay/r2 going positive while fwd/scatter_err and
# fwd/over_coverage DEGRADE is a train/test crossover -- replay-buffer
# overfitting, exactly as called. SubTB accelerates it because it imposes a
# constraint per SUB-trajectory rather than one per trajectory, so each
# replayed sample carries ~T times the memorizable constraint density.
#
# This arm lets the fwd branch train the policy, so fresh forward samples get a
# say instead of the policy being fit only to buffer + data. Deliberately NO new
# controller logic for fwd/replay balance -- just the one flag.
#
# Resumes from the MLE exit (phase1_exit), so phase 1 is not repeated and this
# is a clean A/B against the run just stopped, which shared that exact state.
PHASE1_EXIT = ('local_aug02_subtb_l95_elj-mipcas_sg2_zp1_elj_prior_dataset'
               '-T2.5-573c92_phase1_exit.pt')

cfg = base()
cfg['model']['full_flow'] = True
cfg['lr_flow'] = 1.0e-4
cfg['z_calibration']['enabled'] = False
for mode in ('fwd_loss_coeffs', 'bwd_loss_coeffs', 'replay_loss_coeffs'):
    cfg[mode]['tb'] = 0.0
    cfg[mode]['subtb'] = 1.0
    cfg[mode]['subtb_lambda'] = 0.95
for st in cfg['protocol']['stages']:
    lc = st.setdefault('loss_coeffs', {})
    if st['name'] == 'naive':
        # the one change under test: fwd now trains the policy as well as Z
        lc['fwd'] = {'subtb': 1.0, 'tb': 0.0, 'freeze_policy': 0.0}
        lc['bwd'] = {'subtb': 1.0, 'tb': 0.0}
        lc['replay'] = {'subtb': 1.0, 'tb': 0.0}
    else:
        for br in ('fwd', 'bwd', 'replay'):
            lc.setdefault(br, {})['subtb'] = 0.0
            lc[br].setdefault('tb', 0.0)
cfg['checkpoint_name'] = PHASE1_EXIT       # explicit tag beats 'running'
cfg['load_weights_only'] = False           # want optimizers + stage restored
cfg['continue_from_checkpoint'] = False
cfg['reuse_prior'] = False
cfg['max_reloads'] = 6
cfg['terminal_frozen_steps'] = 2000
cfg['epochs'] = 20000
write(cfg, 'subtb_fwdpol')

# ------------------------------------- SubTB, fwd-policy ON, replay frac LOW
# Isolates replay_frac against subtb_fwdpol (same phase1_exit, same
# freeze_policy: 0), so B vs C is a one-variable comparison.
#
# In a FUSED stage the fracs are LOSS WEIGHTS, not throughput shares
# (fused_train_step docstring): every branch runs at the full batch each step and
# draw_replay_sample takes batch_size=self.batch_size. So:
#
#   draws_per_row = batch_size / churn_rate = 2831 / 80 = 35.4
#
# (lifetime = occupancy/churn in steady state, so max_size and residence cancel
# exactly -- a bigger buffer does NOT reduce reuse.) Under SubTB that is ~35 x T
# ~ 354 constraint-impositions per stored trajectory. Batch growth drives it
# directly: 1000 -> 12.5, 2831 -> 35.4, 7410 -> 92.6, and replay has no
# churn_batch_ref (only prior_buffer does, train.py:3881).
#
# replay_frac therefore does NOT change EXPOSURE -- the same rows are drawn the
# same 35 times. It scales replay's LOSS WEIGHT, so 0.218 -> 0.05 cuts the
# memorization PRESSURE ~4.4x while perturbing no buffer dynamics at all. That
# makes it the cleanest single-variable test available. Raising churn would cut
# exposure instead (and is not supply-limited: fwd yields 2831 candidates/step
# against churn 80, a 2.8% admission rate), but at fixed max_size it buys that
# by collapsing residence, turning replay into an on-policy duplicate of fwd.
#
# Implemented via the EXISTING balance.max_fracs cap -- a static bound on the
# proportional controller, not new controller logic. The controller is currently
# driving replay UP (fwd/over_coverage ~24 against target 18), so a cap is what
# actually binds; changing entry fracs alone would be overridden.
cfg = yaml.safe_load(open(HERE / 'subtb_fwdpol.yaml', encoding='utf-8'))
for st in cfg['protocol']['stages']:
    if st['name'] == 'naive':
        st['fracs'] = {'fwd': 0.2, 'bwd': 0.75, 'replay': 0.05}
        st['balance']['max_fracs'] = {'replay': 0.05}
        st['balance']['default_boost'] = {'bwd': 0.94, 'replay': 0.06}
cfg['run_name'] = 'subtb_fwdpol_lowreplay'
with (HERE / 'subtb_fwdpol_lowreplay.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
print(f"{'wrote subtb_fwdpol_lowreplay.yaml':30s} run_name=subtb_fwdpol_lowreplay")

# ------------------------- SubTB, fwd-policy ON, FAST-CHURN / FRESH replay buffer
# Design goal: buffer LARGE and/or fast-changing, yet FRESH -- hard to overfit,
# and current to the policy (long lag is itself harmful, independent of
# memorization). The three quantities, in steady state:
#
#     lifetime (steps) = mean_residence_steps        (occupancy/churn)
#     occupancy        = churn_rate * mean_residence_steps   (capped at max_size)
#     draws_per_row    = batch_size / churn_rate
#
# so churn buys BOTH low reuse and (at fixed occupancy) freshness, residence
# sets lag directly, and max_size only has to be big enough not to bind.
#
#                      before          after
#   churn_rate            80            600
#   mean_residence        50              8
#   max_size            4000           6000
#   ------------------------------------------
#   draws_per_row       35.4            4.7      (7.5x less memorizable reuse)
#   policy lag         50 steps       8 steps    (6x fresher)
#   occupancy           4000           4800      (larger, TTL binds not the cap)
#
# Not supply-limited: the fwd branch yields a full batch (2831) of candidates
# per step, so churn 600 admits ~21% of what is already generated -- no extra
# rollouts, though admission work rises 7.5x and step time is worth watching.
# toxic_min_draws is rescaled to hold its meaning (it was 0.57x expected draws
# at 20/35.4; 0.57 x 4.7 ~ 3, rounded to 5 to stay above the mean and catch only
# genuinely long-lived rows).
#
# replay_frac deliberately left at the stock 0.2 so this isolates BUFFER
# DYNAMICS against subtb_fwdpol -- exposure, not loss weight.
cfg = yaml.safe_load(open(HERE / 'subtb_fwdpol.yaml', encoding='utf-8'))
rb = cfg['buffers']['replay_buffer']
rb['churn_rate'] = 600
rb['mean_residence_steps'] = 8
rb['max_size'] = 6000
rb['toxic_min_draws'] = 5
cfg['run_name'] = 'subtb_fwdpol_freshbuf'
with (HERE / 'subtb_fwdpol_freshbuf.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
print(f"wrote subtb_fwdpol_freshbuf.yaml run_name=subtb_fwdpol_freshbuf")

# ------------------- SubTB, fwd-policy ON, HARDER churn + rebalanced branch mix
# Two changes against subtb_fwdpol_freshbuf, both motivated by its readout at
# 1230 steps past transition:
#
# (1) CHURN HARDER. freshbuf took replay/fwd from 0.64 to 0.85 (scatter) and
#     0.94 (tb_err) -- the overfitting inversion is mostly gone but not fully;
#     replay scatter is still ~15% better than fwd.
#         churn 600 -> 1200   =>  draws_per_row 4.7 -> 2.4
#         residence  8 -> 5   =>  policy lag 8 -> 5 steps
#         occupancy = 1200*5 = 6000 = max_size (saturated, unchanged size)
#     Still not supply-limited: 1200 of ~2831 candidates/step = 42% admission.
#     Admission work doubles again, so watch step time (was 1.12 -> 1.57 s/it).
#
# (2) REBALANCE. bwd holds 58% of the loss weight while being fit 5.7x better
#     than fwd (scatter 4.29 vs 24.47) -- it is the easiest branch AND the
#     loudest. It is also, like replay, data-anchored, so its gradient cannot
#     move forward geometry. fwd is now the only branch that can (freeze_policy
#     is 0), so weight moves toward it.
#         fwd 0.20 -> 0.35   bwd 0.58 -> 0.40   replay 0.22 -> 0.25
#     pinned must equal fracs.fwd; default_boost re-splits the non-fwd mass
#     (0.65) as bwd 0.40/0.65 = 0.615, replay 0.25/0.65 = 0.385.
cfg = yaml.safe_load(open(HERE / 'subtb_fwdpol_freshbuf.yaml', encoding='utf-8'))
rb = cfg['buffers']['replay_buffer']
rb['churn_rate'] = 1200
rb['mean_residence_steps'] = 5
rb['toxic_min_draws'] = 3
for st in cfg['protocol']['stages']:
    if st['name'] == 'naive':
        st['fracs'] = {'fwd': 0.35, 'bwd': 0.40, 'replay': 0.25}
        st['balance']['pinned'] = {'fwd': 0.35}
        st['balance']['default_boost'] = {'bwd': 0.615, 'replay': 0.385}
cfg['run_name'] = 'subtb_bal_hardchurn'
with (HERE / 'subtb_bal_hardchurn.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
print('wrote subtb_bal_hardchurn.yaml  run_name=subtb_bal_hardchurn')

if __name__ == '__main__':
    pass

# ------------------------- SubTB, hard churn, bwd RAISED (mass-covering restored)
# Correction of direction: 'bwd overpowered' meant bwd is overpowered BY the
# other branches, i.e. it needs MORE weight, not less. E cut bwd 0.58 -> 0.38.
#
# The reason to expect that matters: bwd/MLE is the FORWARD-KL, mass-covering
# term; fwd TB is reverse-KL, mode-seeking. Cutting bwd removes mass-covering
# pressure, and wass_debiased degraded monotonically B 0.0074 -> D 0.0110 ->
# E 0.0138 while every TB metric improved and EffDim stayed flat. So the
# prediction here is specific: raising bwd should improve wass, likely at some
# cost in fwd/tb_err.
#
# fwd stays at 0.35 (raising it 0.20 -> 0.35 improved fwd, bwd AND replay
# tb_err); the increase comes out of replay, which had re-inverted against fwd
# (scatter ratio 0.879 -> 0.777).
#     fwd 0.35   bwd 0.38 -> 0.50   replay 0.27 -> 0.15
# Buffer left at the harder churn (draws_per_row 2.4, lag 5) so this isolates
# the branch mix against subtb_bal_hardchurn.
cfg = yaml.safe_load(open(HERE / 'subtb_bal_hardchurn.yaml', encoding='utf-8'))
for st in cfg['protocol']['stages']:
    if st['name'] == 'naive':
        st['fracs'] = {'fwd': 0.35, 'bwd': 0.50, 'replay': 0.15}
        st['balance']['pinned'] = {'fwd': 0.35}
        st['balance']['default_boost'] = {'bwd': 0.769, 'replay': 0.231}
cfg['run_name'] = 'subtb_bwdup'
with (HERE / 'subtb_bwdup.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=True)
print('wrote subtb_bwdup.yaml  run_name=subtb_bwdup')
