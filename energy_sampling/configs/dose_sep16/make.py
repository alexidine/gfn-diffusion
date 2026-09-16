"""dose_sep16 -- the replay-dose ladder and the tau ladder on mip, at a PINNED batch:
does a fresher buffer hold the high-Z state that N=20 only visits, and does a
larger buffer change the regime at the same per-row dose?

    python configs/dose_sep16/make.py            # the nine dose/tau arms
    python configs/dose_sep16/make.py --with-k1  # + n20_sf1 and fs_k1 (needs the stored-force / path-gradient code committed)

THE HYPOTHESIS (owner + 2026-09-16 readout of flk_sep14 and the four p12 arms).
Across the production arms the replay dose D = N * (lr/1e-4) * (w/0.1) * (1000/B)
orders everything: memorisation (replay/resid_vs_intake 0.70/0.75/0.81/0.90 at
D = 15.6/7.8/2.1/1.0 for mip/neh/nehu/mipu), whether log Z climbs or sits flat
with transient peaks, and whether the ratchet holds replay at the floor. mipu,
at D ~ 1 and memorisation 0.90, is the only arm that looks healthy. mip's
excursions to 31-32 were the policy broadening on a burst of fresh rollouts and
then memorising the same rows at 16x mipu's rate; pinned at 0.3 replay it holds
29.8 (flk14_pin30). Rare rollouts were bought for the MLIP routes; on ELJ a
rollout is nearly free, so N=20 buys nothing there and costs the freshness.

THE BATCH IS PINNED AT 1600 ON EVERY ARM, sizer retest off, because the sizer
walks 1600 <-> 2560 every 1000 steps on rung noise and under store-all that
moves occupancy 1.6x and the per-row dose 1/1.6x on each edge -- log Z followed
it in every arm measured, pinned or not. Here it is the instrument, not a
strategy. The held-out split (val_frac 0.05) is ON so replay/val_gap_nats can
separate learning from fitting the rows; resid_vs_intake cannot.

THE ARMS (2026-09-16 v3, after the adversarial review). Replay PINNED at 0.3
unless stated (point bounds, as flk_sep14), batch 1600 unless stated, all off the
SAME 160k archive flk_sep14 used (read from its SEED.txt if present, else the
parent's newest archive, recorded the same way). That archive sits at Z 28.73,
the 9th percentile of the parent's recent window (a local LOW), so every arm
relaxes upward at first: read levels and slopes from seed+4k on, at matched
step counts (N=1 and B=2560 arms run ~20-25% fewer steps in the wall).

  n20           D ~ 47   tau 120   the control: flk14_pin30's shape at a pinned batch
  n5            D ~ 12   tau 120
  n1            D ~ 2.3  tau 120   a rollout every step. NOT 'fresh replay': the hazard is
                per step, so the DRAW's age is exponential(tau) whatever N is -- n1's
                draw is as stale as n20's, only its intake is fresher. The N ladder
                moves occupancy (B*tau/N: 9.6k -> 38k -> 192k), the fresh fraction
                (N/tau) and the eval-origin share along with the dose
  n1_t6         D ~ 2.3  tau   6   occupancy 9.6k, fresh 1/6: the SAME buffer shape as n20
                with only the reuse per row changed, 20 -> 1. The true on-policy limit
  n1_w15        D ~ 1.2  tau 120   n1 at share 0.15: the same dose by the share lever
  n20_lr05      D ~ 23   tau 120   the control at half the base rate (seed_lr; the rate reaches
                all four branches, so this is a GLOBAL rate cut = the mip-vs-neh
                production gap, not a replay-only dose cut)
  n20_b2560     D ~ 29   tau 120   batch pinned at 2560: the one effect with a measured size on
                this run (+1.2 nats per e-fold of B on the parent), clean of the sizer
                and the ratchet. Bundles dose /1.6, pool x1.6 and gradient noise /1.6;
                the N and tau arms price the first two, the remainder is noise
  n10_w60_t60   D ~ 47   tau  60   N 10, share 0.6, tau 60: dose, occupancy (9.6k) and fresh
                fraction (1/6) IDENTICAL to n20; only reuse (10 vs 20) and the share
                move -- the share exponent the dose law assumes (never measured) and
                reuse vs admission rate at fixed dose
  n20_pbfrozen  D ~ 47   tau 120   the control with P_B FROZEN (freeze_backward_policy full,
                snapshotted at load): is the Z gain P_F improving or P_B deforming to
                explain stored rows? (replay_loss_coeffs.detach_pb is a dead key)
  (--with-k1)   n20_sf1: the control + replay stored-force k=1;  fs_k1: the forward seat
                with the last-step reward gradient (see the functions)

Dropped from v2 after review: t3/t12/t48 (a tau ladder at fixed dose; t48 needs
~4.8k steps to equilibrate, most of an 8 h wall) and n1_lr05 (the rate halves the
bwd dose too, so it was not the matched pair to n1_w15; n20_lr05 is the clean
contrast). Per-row dose is tau-invariant (draws per row = N under store-all).

ENDPOINT (pre-registered): Theil-Sen slope of fwd/log_Z_learned over
[seed+4k, end] at matched step counts; the Z level; max(fwd, bwd, replay
emp_z) as the best lower bound on log Z*; eval_fwd/tb_err; replay/val_gap_nats.
NO anchor-distribution veto (w1r, emarg, wass measure movement relative to the
prior sample, not convergence); n20_pbfrozen is the validity check instead.

fwd_rollout_every 1 keeps the cadenced code path (fill pins Z on every step);
0 would take the legacy every-step path and is deliberately not used, so the
only thing that changes down the ladder is N.

WALL 8 h (owner 2026-09-16). At the measured 3.1 s/step, ~9.3k steps at N=20 and
~7.5k on the N=1 and B=2560 arms. A winner shows in ~3k steps (the parent
reached 31-32 within ~500 steps of every favourable window); slope ranking of
slow climbers is marginal at this wall and is what a 12 h rerun of the winner
is for. Every buffer here equilibrates within 600 steps.

WHAT TO READ: fwd/log_Z_learned level and slope over the last 4k, its sd;
replay/resid_vs_intake (mipu's 0.90 is the target; the 1/e line is 0.37);
fwd/tb_err and eval_fwd/tb_err; fwd/emp_z minus learned (the ELBO gap);
bwd/logw_std_within; Excess Energy Nats Mean (breadth); protocol/gr_held (must
stay ~0 under a pin); replay/val_gap_nats on every arm, and its trend on the tau
arms. The n1_lr05 vs n1_w15 pair says whether dose is the whole story or the
share has a separate role.
"""
import copy
import importlib.util
import pathlib
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
ES = ROOT.parent
BASE = ROOT / 'prod_sep12' / 'p12_mip_lr1.yaml'
PARENT = 'p12_mip_lr1'
PLACEHOLDER = 'WARM_CHECKPOINT_PLACEHOLDER'
PRIOR_PLACEHOLDER = 'PRIOR_MODEL_PLACEHOLDER'
TAG = 'dose16'
WALL = '8:00:00'
EXECUTED = ('configs/mk_dev.yaml', 'train.py', 'protocol.py', 'buffer.py',
            'gflownet_losses.py', 'checkpointing.py', 'utils.py', 'config_invariants.py',
            'models/gfn.py', 'energies/molecular_crystal.py')

_spec = importlib.util.spec_from_file_location('flkmake', ROOT / 'flk_sep14' / 'make.py')
flk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flk)
_eq, _pin = flk._eq, flk._pin


def _pin_batch(cfg, b=1600):
    # THE BATCH IS THE INSTRUMENT'S ONLY CONFOUND. The sizer re-measures every
    # 1000 steps and walks 1600 <-> 2560 on rung noise; under store-all that moves
    # occupancy 1.6x and the per-row dose 1/1.6x on every step edge, and log Z
    # follows it in every arm measured (memory: batch_sizer_retest_forces_the_z_cycle).
    # Pinned here so N, w, eta, tau and (on one arm) B itself are the only movers.
    cfg['batch_size'] = int(b)
    cfg['max_batch_size'] = int(b)
    cfg['grow_batch_size'] = False
    cfg['batch_util_target'] = 0.0
    cfg['batch_sizer_retest_steps'] = 0


def _freeze_pb(cfg):
    # P_B FROZEN, every branch. The validity test for the endpoint: a Z gain that
    # survives this is P_F improving; one that vanishes was P_B deforming to explain
    # stored trajectories (u depends only on log P_F - log P_B, so a common shift is
    # free and lowers KL without moving the terminal marginal). On a resume with no
    # stored snapshot, 'full' snapshots the live t/s/backward weights at load and
    # the live trunk trains P_F only. NB replay_loss_coeffs.detach_pb (the per-branch
    # form) is a DEAD KEY: mk_dev carries it and no module reads it, in HEAD or in
    # the working tree (checked 2026-09-16). Do not use it.
    cfg['freeze_backward_policy'] = 'full'


def _holdout(cfg):
    # 5% of every admission batch flagged held-out and never trained on, so
    # replay/val_gap_nats separates learning (held-out falls too) from fitting the
    # rows (only the trained side falls). resid_vs_intake cannot: a better policy
    # and a memorised row both lower it. ~80 rows per rollout, ~480 resident at
    # tau 120, well above val_min 64.
    cfg['buffers']['replay_buffer']['val_frac'] = 0.05
    # val_gap_nats_se floors at ~2 nats with val_cap 256 -- too coarse to act on.
    cfg['buffers']['replay_buffer']['val_cap'] = 1024
    # THE CAP MUST NEVER BIND: under store-all O = B*tau/N, and the N=1 arm sits at
    # 1600*120 = 192k rows against the base cap of 250k, t48 at 77k. A binding cap
    # turns eviction into displacement and the arm measures the cap. ~1 KB/row on
    # the card, so a million-row ceiling costs nothing unless it is reached.
    cfg['buffers']['replay_buffer']['max_size'] = 1_000_000


def _tau(t):
    def f(cfg):
        cfg['buffers']['replay_buffer']['mean_residence_steps'] = int(t)
    return f


def _every(n):
    def f(cfg):
        _eq(cfg)['fwd_rollout_every'] = int(n)
    return f


def _rate(scale):
    # THE RATE ON A RESUME IS seed_lr, NOT fixed_scale. The controller comes back
    # from the parent's checkpoint in CRUISE with the parent's promoted scale
    # (1.0), and fixed mode stamps `fixed_scale` only at the end of a burn-in that
    # a resume never repeats -- so a config fixed_scale 0.5 would run at 1.0.
    # `_apply_lrs` computes lr = base x scale where every `auto` lr_* key resolves
    # to seed_lr, so halving seed_lr halves the applied rate whatever was restored.
    def f(cfg):
        cfg['lr_control']['seed_lr'] = 1.25e-4 * float(scale)
    return f


def _stored_force(cfg):
    # REPLAY STORED-FORCE, k=1: every admitted row records d log R / d x_T at
    # admission (one extra reward call on the admitted rows, cheap on ELJ) and the
    # replay loss re-propagates the last stored step through its implied noise and
    # pushes x_T along that force. No energy call at replay time. Rows restored
    # from the parent's sidecar carry no force and are skipped until turned over.
    rc = cfg['replay_loss_coeffs']
    rc['stored_force_k'] = 1
    rc['stored_force_mode'] = 'implied'
    rc['resample_last_k'] = 0
    rc['reward_grads'] = 0.0


FS_FWD, FS_BWD, FS_REPLAY = round(0.3 / 0.7, 4), round(0.5 / 0.7, 4), round(0.2 / 0.7, 4)


def _forward_seat_k1(cfg):
    # THE ON-POLICY ENDPOINT WITH THE REWARD GRADIENT (pathgrad_sep14 k1_rg1 shape):
    # a rollout every step, the forward branch trains the policy (freeze_policy 0)
    # at a pinned 0.3 share, replay 0.2 / bwd 0.5 pinned by point bounds, the
    # last-step path gradient live with the reward gradient through it, clip 10.
    # Under fwd_rollout_every 0 the fill stash is free on every step and the
    # trigger block is refused, so it is popped.
    # THE RESUMED PAIR CARRIES MASS 1.0. bwd_frac + replay_frac are modeller state
    # restored from the parent (0.9 + 0.1); the ramp re-splits that pair and never
    # renormalises it, and the pinned fwd share is ADDED on top. Bounds written as
    # absolute 0.5 / 0.2 therefore conflict at runtime (s_lo > s_hi) and collapse
    # to a midpoint. So the intended 0.3 : 0.5 : 0.2 is written as its ratios
    # against a pair of 1.0 -- fwd 0.4286, bwd 0.7143, replay 0.2857 -- the same
    # split up to a global loss scale that Adam does not see. The parser caps each
    # bound at 1 - pinned = 0.5714, so only the replay bound is written (0.2857
    # fits) and bwd is the remainder of the pair; the entry fracs carry the same
    # ratios and would normalise to 0.3/0.5/0.2 on a fresh transition.
    eq = _eq(cfg)
    eq['fwd_rollout_every'] = 0
    eq['z_pin_rollout_every'] = 0
    eq.pop('fwd_rollout_triggers', None)
    eq['fracs'] = {'fwd': FS_FWD, 'bwd': FS_BWD, 'replay': FS_REPLAY}
    eq['balance']['pinned'] = {'fwd': FS_FWD}
    eq['balance']['bounds'] = {'replay': [FS_REPLAY, FS_REPLAY]}
    eq['loss_coeffs']['fwd'] = {'tb': 1.0, 'freeze_policy': 0.0}
    fc = cfg['fwd_loss_coeffs']
    fc['path_grad_last_k'] = 1
    fc['reward_grads'] = 1.0
    fc['reward_grad_clip'] = 10.0
    fc['traj_grads'] = 0.0


COMMON = [lambda c: _pin(c, 0.3), _pin_batch, _holdout]
#: built only with --with-k1: both need the stored-force / path-gradient code,
#: which is not in the committed tree as of 2026-09-16.
K1_ARMS = {
    'n20_sf1':  COMMON + [_every(20), _stored_force],
    'fs_k1':    [_pin_batch, _holdout, _forward_seat_k1],
}
ARMS = {
    # --- the dose ladder by cadence, at share 0.3 ----------------------------------
    'n20':          COMMON + [_every(20)],
    'n5':           COMMON + [_every(5)],
    'n1':           COMMON + [_every(1)],
    # --- the single-factor cut: tau/N held at 6, only the reuse per row moves ----
    'n1_t6':        COMMON + [_every(1), _tau(6)],
    # --- the same dose by the share, at N=1 ------------------------------------------
    'n1_w15':       [lambda c: _pin(c, 0.15), _pin_batch, _holdout, _every(1)],
    # --- the global-rate lever on the control shape: the mip-vs-neh production gap --
    'n20_lr05':     COMMON + [_every(20), _rate(0.5)],
    # --- the batch effect, measured clean of the sizer: dose /1.6, pool x1.6, noise /1.6
    'n20_b2560':    [lambda c: _pin(c, 0.3), lambda c: _pin_batch(c, 2560), _holdout, _every(20)],
    # --- dose, occupancy and freshness IDENTICAL to n20; only reuse (10 vs 20) and the
    #     share (0.6 vs 0.3) move: the share exponent the dose law assumes, and reuse
    #     vs admission rate at fixed dose
    'n10_w60_t60':  [lambda c: _pin(c, 0.6), _pin_batch, _holdout, _every(10), _tau(60)],
    # --- the validity check: P_B frozen ------------------------------------------------
    'n20_pbfrozen': COMMON + [_every(20), _freeze_pb],
}
#: (N, replay share, rate scale, tau, batch)
EXPECT = {
    'n20':          (20, 0.3, 1.0, 120, 1600),
    'n5':           (5, 0.3, 1.0, 120, 1600),
    'n1':           (1, 0.3, 1.0, 120, 1600),
    'n1_t6':        (1, 0.3, 1.0, 6, 1600),
    'n1_w15':       (1, 0.15, 1.0, 120, 1600),
    'n20_lr05':     (20, 0.3, 0.5, 120, 1600),
    'n20_b2560':    (20, 0.3, 1.0, 120, 2560),
    'n10_w60_t60':  (10, 0.6, 1.0, 60, 1600),
    'n20_pbfrozen': (20, 0.3, 1.0, 120, 1600),
    'n20_sf1':      (20, 0.3, 1.0, 120, 1600),
    'fs_k1':        (0, round(0.2 / 0.7, 4), 1.0, 120, 1600),
}


def dirty_files():
    out = subprocess.run(['git', 'status', '--porcelain', '--'] + [str(ES / p) for p in EXECUTED],
                         capture_output=True, text=True, cwd=str(ES), check=True).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def dose(n, scale, w, batch=1600):
    return n * (1.25e-4 * scale / 1e-4) * (w / 0.1) * (1000.0 / batch)


def build(with_k1=False):
    base = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    out = {}
    arms = dict(ARMS, **K1_ARMS) if with_k1 else ARMS
    for arm, deltas in arms.items():
        cfg = copy.deepcopy(base)
        name = TAG + '_' + arm
        cfg['run_name'] = name
        cfg['tag'] = TAG
        cfg['checkpoint_name'] = PLACEHOLDER
        cfg['prior_model_name'] = PRIOR_PLACEHOLDER
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        for d in deltas:
            d(cfg)
        check(cfg, name, arm)
        out[name] = cfg
    return out


def check(cfg, name, arm):
    n, w, scale, tau, batch = EXPECT[arm]
    eq = _eq(cfg)
    # the instrument: batch pinned, sizer silent, held-out split on
    assert cfg['batch_size'] == cfg['max_batch_size'] == batch and cfg['grow_batch_size'] is False, name + ': batch must be pinned'
    assert cfg['batch_util_target'] == 0.0 and cfg['batch_sizer_retest_steps'] == 0, name + ': the sizer must be silent'
    assert cfg['buffers']['replay_buffer']['val_frac'] == 0.05, name + ': held-out split off'
    assert cfg['buffers']['replay_buffer']['val_cap'] == 1024, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['mean_residence_steps'] == tau, name
    assert rb['max_size'] >= 3 * batch * tau / max(n, 1), name + ': replay cap would bind'
    assert cfg['checkpoint_name'] == PLACEHOLDER and cfg['prior_model_name'] == PRIOR_PLACEHOLDER, name
    assert cfg['load_weights_only'] is False and cfg['epochs'] >= 500_000, name
    assert eq['fwd_rollout_every'] == n, name
    assert eq['flags']['z_calibration'] is False and float(cfg['z_calibration']['fill_threshold']) > 0, name
    b = eq['balance']
    if arm == 'fs_k1':
        assert n == 0 and 'fwd_rollout_triggers' not in eq, name + ': the forward seat rolls out every step'
        assert eq['fracs'] == {'fwd': FS_FWD, 'bwd': FS_BWD, 'replay': FS_REPLAY} and b['pinned'] == {'fwd': FS_FWD}, name
        assert b['bounds'] == {'replay': [FS_REPLAY, FS_REPLAY]}, name + ': the replay bound must be a point'
        assert abs(FS_BWD + FS_REPLAY - 1.0) < 1e-3 and FS_REPLAY <= 1.0 - FS_FWD, name + ': the resumed pair has mass 1.0 and the bound must parse'
        assert eq['loss_coeffs']['fwd'] == {'tb': 1.0, 'freeze_policy': 0.0}, name
        fc = cfg['fwd_loss_coeffs']
        assert fc['path_grad_last_k'] == 1 and fc['reward_grads'] == 1.0 and fc['reward_grad_clip'] == 10.0 and fc['traj_grads'] == 0.0, name
    else:
        assert n >= 1, name + ': N must stay on the cadenced path (>= 1)'
        bwd = round(1 - w, 3)
        assert eq['fracs'] == {'fwd': 0.0, 'bwd': bwd, 'replay': w}, name
        assert b['bounds'] == {'bwd': [bwd, bwd], 'replay': [w, w]}, name + ': bounds must be a point'
    rc = cfg['replay_loss_coeffs']
    assert cfg['freeze_backward_policy'] == ('full' if arm == 'n20_pbfrozen' else False), name + ': freeze_backward_policy'
    assert float(rc.get('detach_pb', 0.0)) == 0.0, name + ': detach_pb is a dead key; leave it 0'
    if arm == 'n20_sf1':
        assert rc['stored_force_k'] == 1 and rc['stored_force_mode'] == 'implied' and rc['resample_last_k'] == 0 and rc['reward_grads'] == 0.0, name
    else:
        assert not rc.get('stored_force_k') and not rc.get('resample_last_k'), name + ': no replay tail on a dose arm'
    lc = cfg['lr_control']
    assert lc['mode'] == 'fixed' and lc['fixed_scale'] == lc['burn_in_scale'] == 1.0, name + ': the scale is inherited; the rate moves through seed_lr'
    assert abs(lc['seed_lr'] - 1.25e-4 * scale) < 1e-12, name
    assert all(cfg[k] == 'auto' for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused')), name + ': seed_lr only reaches auto keys'
    assert cfg['buffers']['prior_buffer']['source'] == 'prior_model', name
    assert rb['churn_rate'] == 0, name
    for s in cfg['protocols']['unconditional_tb']['stages']:
        sensor = s.get('hot_lr_sensor')
        if isinstance(sensor, dict):
            assert sensor.get('action', 'report') == 'report', name


# The seed block reads flk_sep14's SEED.txt first so the two batteries share one
# archive; otherwise it records the parent's newest archive exactly as flk did.
SBATCH = flk.SBATCH.replace('#SBATCH --job-name=flk14', '#SBATCH --job-name=dose16') \
    .replace('configs/flk_sep14/joblogs/%x_%A_%a.out', 'configs/dose_sep16/joblogs/%x_%A_%a.out') \
    .replace('# flk_sep14: seven knobs off one frozen p12_mip_lr1 archive.',
             '# dose_sep16: the replay-dose ladder off the SAME frozen p12_mip_lr1 archive flk_sep14 used.') \
    .replace('ARMS=${{WORKDIR}}/configs/flk_sep14', 'ARMS=${{WORKDIR}}/configs/dose_sep16') \
    .replace('SEED_FILE=${{LOGS}}/SEED.txt',
             'SEED_FILE=${{LOGS}}/SEED.txt\nFLK_SEED=${{WORKDIR}}/configs/flk_sep14/joblogs/SEED.txt\n'
             'if [ ! -s ${{SEED_FILE}} ] && [ -s ${{FLK_SEED}} ]; then cp ${{FLK_SEED}} ${{SEED_FILE}}; echo "  seed inherited from flk_sep14: $(cat ${{SEED_FILE}})"; fi')


def main(argv):
    dirty = dirty_files()
    if dirty and '--allow-dirty' not in argv:
        sys.exit('REFUSING: uncommitted files the arms execute:\n  ' + '\n  '.join(dirty) +
                 '\nBuild from a clean worktree, or pass --allow-dirty for a LOCAL build.')
    arms = build(with_k1='--with-k1' in argv)
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text('SLURM cannot create --output; SEED.txt is written here at launch\n', encoding='utf-8')
    for name, cfg in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\n')
        for name in arms:
            f.write('%s\t%s\n' % (name, PARENT))
    with (HERE / 'submit_dose_sep16.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(SBATCH.format(wall=WALL, last=len(arms) - 1, placeholder=PLACEHOLDER,
                              prior_placeholder=PRIOR_PLACEHOLDER))
    for name, cfg in arms.items():
        n, w, scale, tau, batch = EXPECT[name[len(TAG) + 1:]]
        neff = max(n, 1)
        print('%-16s N=%-3d replay=%.2f scale=%.2f tau=%-4d B=%-5d occupancy~%6d fresh 1/%-4d reuse %-3d dose~%5.1f  freeze_pb %s'
              % (name, n, w, scale, tau, batch, batch * tau / neff, max(1, round(tau / neff)), neff, dose(neff, scale, w, batch),
                 cfg['freeze_backward_policy']))


if __name__ == '__main__':
    main(sys.argv[1:])
