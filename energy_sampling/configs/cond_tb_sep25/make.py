r"""cond_tb_sep25 -- conditional phase 2 on TRAJECTORY BALANCE with the per-condition ema_logw
as the normaliser, in place of pooled VarGrad. LOCAL test battery.

    python configs/cond_tb_sep25/make.py

WHY. VarGrad is TB with the normaliser profiled out (vargrad.md: min_logZ E[(u-logZ)^2] =
Var(u), attained at logZ = E[u]), so it needs no Z(c) -- which is why it was chosen after
learned Z(c) oscillated. But the group mean that stands in for Z is estimated from ~4 rows,
while `buffer.py::ConditionLogZTracker.ema_logw` estimates the SAME quantity from an
effective 414 samples. Measured on cl21c step20000: lag bias 0.081 nats against EMA noise
0.081 nats, RMSE ~0.12 against a 1.64-nat residual spread -- a 7% error. The case for TB is
not that we can improve that estimator; it is that VarGrad discards the level information the
estimator already provides.

The tracker is `.detach().cpu()` bookkeeping, so it cannot be in a gradient path: the
policy->Z->policy loop is a LAGGED EMA loop, not the gradient-coupled quadratic that
oscillated. A wrong centre mis-signs a clipped force; it does not diverge.

WHAT THIS DROPS, and it is most of the fiddly surface. TB's residual is per-row, so there are
no groups: no `repeats`, no `condition_block_m`, no `vg_live_frac`, no singleton groups, and
no pooled thinning (which silently took n_groups 500 -> 49 on the g16 arm). `pooled_vg` and
`level_gap` exist only to patch a flat direction that appears BECAUSE VarGrad is level-free,
so both go, and with them the `replay_seat_problems` constraints on pooled_source.

WHAT IT UNLOCKS. The reshaped terminal force (`reward_grad_gate` / `reward_grad_force_clip`)
reads d L/d log R by autograd against the BRANCH loss. On the VarGrad seat fwd carried tb 0
and the only log-R term was the pooled one assembled later in fused_train_step, so the call
returned None and the surrogate injected a zero push -- `rewardgrad/push_norm_mean` measured
exactly 0 there against 0.53-1.09 on force_sep18, whose fwd carries tb 1. With tb 1.0 on fwd
this is the first conditional seat where the gate and the clip can fire at all, so the
cl21 _fc / _fg nulls are artefacts and are not evidence about either key. Separately, the
gate thresholds on the TB residual, whose log Z is now the tracker's per-condition value, so
rows gate against their OWN condition's level for free.

BASE. The verified cl21 lambda=0 arms (same identity, same phase-1 seed, same flow path); the
only edit is the stage's coefficient block, so the diff IS the thing under test. The stage
keeps the name `var_conditioning` -- the label is now a misnomer, kept because cont.py and
analysis/tests/test_keys.py key on it and a first test should not move two things at once.

lambda stays at 0 and `coeff_schedule` stays empty: get TB-with-persistent-Z standing up
before adding the anneal. When the ramp goes on, the rate is what bounds the Z error --
measured d log R/d lambda = 61.6 nats per unit lambda, so a rung's level step must stay under
the ~1.64-nat residual spread, i.e. delta-lambda <~ 0.025.
"""
import argparse
import copy
import os
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
BASE_DIR = HERE.parent / 'cond_lam_sep21'
TAG = 'ctb25'

# keys that only exist because the loss was VarGrad; every one is dead or wrong under TB
VG_KEYS = ('vg_by_condition', 'vg_lb', 'vg_lme', 'vg_detach_center',
           'pooled_vg', 'pooled_beta', 'pooled_ratio', 'pooled_source', 'pooled_bridge_only',
           'pooled_thin_by_condition', 'condition_block_m', 'level_gap', 'level_gap_clamp',
           'level_gap_pf_only', 'emp_z', 'emp_z_persistent', 'z_level')
TB_BETA = 80.0


def build(base_name, out_name, force, freeze_pb, beta=TB_BETA,
          gate=None, force_clip=0.0, exc_k=40.0, lam=0.0,
          blend=False, src_ckpt=None, epochs=None, xcond_period=1000,
          archive_period=1000, emp_z_persistent=0.0, fixed_scale=None):
    cfg = yaml.safe_load((BASE_DIR / f'{base_name}.yaml').open(encoding='utf-8'))

    # --- the base must be what this generator thinks it is, or the diff is not the test
    assert float(cfg['energy_config']['lambda_mix']) == 0.0, cfg['energy_config']['lambda_mix']
    # lambda_mix is in _NON_IDENTITY_ENERGY_CONFIG_KEYS, so moving it keeps the problem hash
    # and the seed still loads. utils.py's comment there states the cost plainly: at lambda=0
    # the target IS a different distribution, so two checkpoints at different lambda look
    # interchangeable by hash and are not -- read lambda_mix off the run before comparing.
    cfg['energy_config']['lambda_mix'] = float(lam)
    if float(lam) == 1.0:
        # at lambda=1 the mix is (1-lam)*flow + lam*phys, so the flow leg carries weight 0 --
        # keeping the path would construct and evaluate it every step for nothing. Dropping it
        # is also what makes this arm identical in shape to the phase-1 target, whose tracker
        # was warmed at lambda=1 and is therefore already matched to this arm.
        cfg['energy_config']['prior_flow_path'] = None
    # molecular_crystal.py raises on lambda != 1 with no flow; at lambda == 1 the flow is
    # legitimately absent because the mix gives it weight 0.
    assert (float(lam) == 1.0 or cfg['energy_config'].get('prior_flow_path')), 'lambda != 1 needs a flow'
    assert cfg['buffers']['prior_buffer']['source'] == 'anchors', \
        "prior buffer must be fed from noised anchors, not the prior model"
    stages = cfg['protocols'][cfg['protocol']]['stages']
    tp, st = stages[0], stages[1]
    assert tp['name'] == 'train_prior' and tp.get('skip_if') == 'prior_loaded', \
        'the MLE stage must stay a skipped stub or the arm re-runs phase 1'
    assert st['name'] == 'var_conditioning', st['name']
    assert st['bwd_sampling_mode'] == 'prior', st['bwd_sampling_mode']

    cfg['run_name'] = out_name
    lc = st['loss_coeffs']

    # --- TB on the two branches that train. tb_z_source persistent is REQUIRED on a
    # conditional run (config_invariants::conditional_z_settings_are_conditional) and is
    # what puts ema_logw in the residual in place of the learned head.
    for mode in ('fwd', 'bwd', 'replay'):
        blk = lc.setdefault(mode, {})
        for k in VG_KEYS:
            blk.pop(k, None)
        blk['repeats'] = 1.0                 # TB is per-row: no grouping anywhere
        blk['tb'] = 1.0
        blk['beta'] = float(beta)
        blk['tb_z_source'] = 'persistent'
    lc['fwd']['freeze_policy'] = 0.0         # forward SEAT: fwd trains, it is not a Z sidecar
    lc['bwd']['freeze_z'] = 1.0              # nothing trains the learned head; it is a readout
    lc['replay']['freeze_z'] = 1.0
    # THE HEAD SIDECAR. Under tb_z_source 'persistent' the TB residual reads ema_logw, so
    # no branch trains log_Z_learned and it sits wherever init left it (-16.6 on
    # pr2i43we against branch levels in the +20s to +60s). That is harmless to training
    # but leaves no Z for a condition the tracker has never visited -- every held-out
    # condition -- because the tracker is a lookup table and the head is the only
    # estimator that generalises across the condition embedding. emp_z_persistent
    # regresses the head onto the tracker on trusted conditions (gflownet_losses: masked
    # by log_z_target_mask, target detached). It trains the head ALONE:
    # GFN._condition_flow detaches the conditioner structurally, after a 2026-08-17 run in
    # which the leaked Z gradient grew the conditioner 187x and NaN'd the policy.
    # Proof of use: the startup line 'no mode trains the flow (Z) head' must NOT print.
    if emp_z_persistent:
        lc['fwd']['emp_z_persistent'] = float(emp_z_persistent)
        lc['fwd']['freeze_z'] = 0.0
    if fixed_scale is not None:
        # read ONLY at promotion (end of burn-in). A resume into cruise restores the
        # checkpointed lr_ctrl.scale and never reads this; it states the intended rate
        # for a fresh stage entry. Changing the rate of a resumed run needs a ramp record
        # in the checkpoint's lr_ctrl -- see the _lrramp surgery in this battery.
        cfg.setdefault('lr_control', {})['fixed_scale'] = float(fixed_scale)

    # --- the terminal force, on the seat where the reshape can actually fire
    lc['fwd']['reward_grads'] = 1.0 if force else 0.0
    lc['fwd']['path_grad_last_k'] = 1 if force else 0
    lc['fwd']['path_grad_scale'] = 1
    # THE RESHAPE is a DIFFERENT construction from the plain path, and only these two keys
    # arm it. Plain reward_grads alone leaves rewardgrad/push_norm_mean and force_norm_* not
    # at zero but ABSENT -- they are the reshape's own readouts. An arm meaning to test the
    # gate/clip must set one of these or it tests the plain path under another name.
    #
    # THEY GO IN THE BASE BLOCK, NOT THE STAGE. protocol.py::coeffs refuses a stage override
    # of any key absent from the base, and the loader drops null-valued entries, so a base
    # `reward_grad_gate: null` never creates the key and a stage override of it is refused
    # however the base is written. Floats in the base are the only spelling that loads, and
    # no override is needed because one stage trains.
    if gate is not None or force_clip > 0:
        assert force, 'the reshape is skipped unless reward_grads is also nonzero'
        bf = cfg.setdefault('fwd_loss_coeffs', {})
        bf['reward_grad_gate'] = float(gate if gate is not None else 0.0)
        bf['reward_grad_force_clip'] = float(force_clip)

    # THE HARD-FAILURE BAR is bar = root_hi + k*(root_hi - root_lo), fitted to burn-in at
    # burn_in_scale. mk_dev's own comment calls that "wrong for the live tripwire, which then
    # holds for a whole stage at a hotter rate", and in FIXED mode the cold bars stay live
    # until the post-promotion refit. Measured on ctb25_tb_f 2026-09-25: a burn-in span of
    # 0.269 put the bar at 3.592, and one batch at 5.069 fired at step 778 mid-ramp while
    # fwd/loss (0.750) and fwd/tb_err (1.148) were both descending smoothly -- a spurious
    # rewind plus a PERMANENT x0.5 rate cut. Note the span scales the bar, so a CLEAN burn-in
    # yields a TIGHTER one; and this channel is heavy-tailed (condition_log_z.trim_frac's
    # comment records logw_std 8-46), a tail the 200-step root window cannot sample.
    # THE LOSS-EXCURSION TRIGGER IS TAKEN OUT OF SERVICE, deliberately. It thresholds a RAW
    # single-batch loss (lr_bracket_probe.py: bar = hi + k*span, fire on loss >= bar, no dwell
    # and no smoothing) on a channel the codebase documents as heavy-tailed, so a tail draw and
    # a divergence are indistinguishable to it. It fired twice on this seat (5.069 over 3.592
    # at step 778; 7.67 over 7.263 at step 942) while every smoothed channel descended
    # monotonically, each time costing a rewind AND a permanent x0.5 rate cut. Every other
    # sensor here thresholds a SMOOTHED channel (hot_lr_sensor) or a windowed RISE
    # (under_coverage_rise150); raw-then-threshold is the one combination that cannot tell a
    # bad batch from a bad trajectory.
    #
    # THE KEY LIVES UNDER lr_control. controller.py:97 reads getattr(lr_control, 'hard_failure'),
    # and until 2026-09-25 this generator wrote a TOP-LEVEL `hard_failure` that nothing reads:
    # every ctb25 arm before ctb25_extreme_l1_cont2 ran at mk_dev's k=10, config_snapshot --check
    # passed all of them, and both fires above were at k=10 -- the bar moved 3.592 -> 7.263 from
    # a wider burn-in span, not from the 10 -> 40 change that was believed applied. Proof of use
    # is the startup line `hard-failure bars ... -> bar X`: solve X = hi + k*(hi - lo) for k.
    cfg.pop('hard_failure', None)   # the dead top-level spelling, if a base ever carries it
    live_hf = cfg.setdefault('lr_control', {}).setdefault('hard_failure', {})
    live_hf['loss_excursion_k'] = float(exc_k)
    # the catastrophic backstops are NOT restated here: the live block already carries them
    # explicitly from mk_dev (nonfinite is unconditional), so they are asserted, not overwritten.
    for k in ('grad_excursion_x', 'loss_abs', 'grad_abs'):
        assert k in live_hf, f'lr_control.hard_failure.{k} absent -- it would read as a code default'

    # RUNS SOLO on the laptop card. The base arms carry cuda_memory_fraction 0.23 from the
    # cl21 co-tenancy battery, and inheriting it caps this at 3.66 GiB of a 15.89 GiB card:
    # ctb25_extreme_l1 OOMed twice before step 250 and shrank 1000 -> 625 -> 390 with 10.95 GiB
    # free, so its whole run was at batch 390 and not comparable to the 1000-row arms.
    cfg['cuda_memory_fraction'] = 0.8
    # a run the owner stops by hand wants cheap restore points; hardlinked, so disk not I/O
    cfg['archive_period'] = int(archive_period)
    cfg['archive_buffers'] = False

    # --- forward seat: fwd carries weight every step, so no rollout cadence and no sidecar
    st['fracs'] = {'fwd': 0.5, 'bwd': 0.5, 'replay': 0.0}
    st['min_fracs'] = {'fwd': 0.02}
    st['fwd_rollout_every'] = 0
    st['fwd_z_sidecar'] = False
    st['coeff_schedule'] = {}                # lambda fixed per arm; no ramp in this battery
    # xcond cadence. MEASURED IN-LOOP and NOT the 0.7% a standalone timing suggested: at
    # period 250 train_step_time read 0.742 against train_step_time_replay 0.692 and the
    # run went 0.53 -> 1.04 s/it. 1000 keeps it under a percent.
    cfg.setdefault('xcond_eval', {})
    cfg['xcond_eval'].update({'enabled': True, 'period': int(xcond_period),
                              'conditions': 20, 'per_condition': 4, 'k': 4,
                              'chunk_rows': 2560})
    st['flags']['z_calibration'] = False     # refused true on the conditional route
    # FEED ema_logw FROM THE FORWARD ROLLOUTS ONLY, so the target adopts the FORWARD level.
    # The forward call site passes no do_update and so always feeds; the backward one passes
    # do_update=update_log_z. With it true the target lands at the evidence-weighted midpoint
    # of the two branch levels, which puts BOTH branches at -+ Delta/2 and drags the policy
    # toward a compromise instead of leaving it correctly centred -- measured 2026-09-25 on
    # ctb25_tb_f_l0p1: fwd/tb_resid_clipped +4.58, bwd -4.80, zmatch/delta_mean 10.2, and
    # Delta closed almost entirely from the FORWARD side (fwd_level +1.08 vs bwd_level -0.28).
    # False leaves fwd correctly centred and puts the whole gap on the backward branch, which
    # is the side that should absorb it. update_and_lookup_condition_log_z's own docstring
    # calls backward importance weights untrustworthy for the persistent estimate outside
    # phase 1/2, and the gate covers ONLY ema_logw: update_mode_level runs regardless, so
    # zmatch/delta_mean, fwd_level and bwd_level are still measured.
    st['flags']['update_log_z'] = bool(blend)   # True = blended fwd/bwd level (anchored,
    # biased by Delta/2); False = forward-only (on-policy, unanchored). Measured 2026-09-25 at
    # lambda=0.1 step 500: blend logr_mean +1.225 / logw_std 7.18 / fwd_level -4.24, fwd-only
    # +0.287 / 8.38 / -5.05, with the fwd-only tracker glued to fwd_level (-5.065 vs -5.049).
    # The bwd term is a TARGET, not an estimate: it says mass exists at high log-weight.
    on_enter = [a for a in st.get('on_enter', []) if not a.startswith('bootstrap_z')]
    if freeze_pb and not any(a.startswith('freeze_pb') for a in on_enter):
        on_enter.append('freeze_pb')
    if not freeze_pb:
        on_enter = [a for a in on_enter if not a.startswith('freeze_pb')]
    st['on_enter'] = on_enter
    assert not any(a.startswith('bootstrap_z') for a in on_enter), on_enter
    assert ('freeze_pb' in ' '.join(on_enter)) == bool(freeze_pb), on_enter

    if src_ckpt is not None:
        # LADDER STEP: full resume (optimizers, buffers, step, stage AND the condition tracker)
        # off an existing arm rather than the phase-1 stub. The stage is RESTORED, not entered,
        # so on_enter does not re-fire -- pb_frozen rides in the checkpoint, which is what keeps
        # P_B on the same snapshot instead of re-snapshotting a drifted trunk.
        cfg['checkpoint_name'] = src_ckpt
        cfg['load_weights_only'] = False
        cfg['continue_from_checkpoint'] = False
        assert epochs is not None, 'a resume needs an absolute epochs past the restored step'
        cfg['epochs'] = int(epochs)
    out = HERE / f'{out_name}.yaml'
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    # --- verify after write: a key this generator thinks it set must be readable back
    chk = yaml.safe_load(out.open(encoding='utf-8'))
    c = [s for s in chk['protocols'][chk['protocol']]['stages']
         if s['name'] == 'var_conditioning'][0]
    for mode in ('fwd', 'bwd'):
        assert float(c['loss_coeffs'][mode]['tb']) == 1.0
        assert c['loss_coeffs'][mode]['tb_z_source'] == 'persistent'
        # emp_z_persistent is a Z-HEAD sidecar, valid under TB; it sits in VG_KEYS only so
        # the pop loop starts every arm clean, and build() re-adds it on request.
        banned = [k for k in VG_KEYS if k != 'emp_z_persistent']
        assert not any(k in c['loss_coeffs'][mode] for k in banned), \
            [k for k in banned if k in c['loss_coeffs'][mode]]
    print(f'wrote {out.name}: tb=1 beta={beta:g} persistent-Z | '
          f'force={"ON" if force else "off"} (reward_grads '
          f'{c["loss_coeffs"]["fwd"]["reward_grads"]:g}, k '
          f'{c["loss_coeffs"]["fwd"]["path_grad_last_k"]}) | '
          f'freeze_pb={freeze_pb} | fracs {c["fracs"]} | lambda '
          f'{chk["energy_config"]["lambda_mix"]:g}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='lam0_ctrl', help='verified lambda=0 arm to derive from')
    ap.add_argument('--beta', type=float, default=TB_BETA)
    ap.add_argument('--no-freeze-pb', action='store_true')
    ap.add_argument('--gate', type=float, default=0.0)
    ap.add_argument('--force-clip', type=float, default=50.0)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--exc-k', type=float, default=1.0e6,
                    help='lr_control.hard_failure.loss_excursion_k; 10 (mk_dev) fired spuriously '
                         'on this seat at steps 778 and 942; 1e6 = out of service')
    a = ap.parse_args()
    fz = not a.no_freeze_pb
    TAGLAM = '' if a.lam == 0 else '_l' + ('%g' % a.lam).replace('.', 'p')
    build(a.base, f'{TAG}_tb_ctrl{TAGLAM}', force=False, freeze_pb=fz, beta=a.beta, exc_k=a.exc_k, lam=a.lam)
    build(a.base, f'{TAG}_tb_f{TAGLAM}', force=True, freeze_pb=fz, beta=a.beta, exc_k=a.exc_k, lam=a.lam)
    build(a.base, f'{TAG}_tb_fg{TAGLAM}', force=True, freeze_pb=fz, beta=a.beta,
          gate=a.gate, force_clip=a.force_clip, exc_k=a.exc_k, lam=a.lam)
    print('')
    print('_ctrl vs _f isolates the PLAIN terminal force -- the path that already worked')
    print('under VarGrad (2-6%). rewardgrad/push_norm_mean does NOT exist on either: it is')
    print('a readout of the RESHAPE, which plain reward_grads does not arm.')
    print('_fg is the arm that exercises the reshape. push_norm_mean > 0 there is the first')
    print('time that construction has delivered anything on a conditional seat -- it read')
    print('exactly 0 on every VarGrad arm because the branch loss carried no live log R.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
