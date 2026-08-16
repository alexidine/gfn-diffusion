"""
local_aug09 -- the FULL CURRENT SUITE, turned on and verified. 2026-08-08.

WHY. local_aug08 ran a whole day on the OLD replay population management,
because it derived from mk_dev.yaml and mk_dev never enabled the new one. The
standing rule now is that local batteries exercise everything new while the
cluster A/Bs features one at a time. This generator is the first config to
actually do that, and every claim below was checked against the CODE -- not
against mk_dev, and not against the design docs, which say "retired" for several
things that still run.

WHAT THIS IS NOT. This is not a science battery. It is two short arms whose only
job is to prove each new subsystem is engaged, using the tells listed in
`verify.py`. Run it, verify it, THEN spend GPU on questions.

=============================================================================
AREA 1 -- LR SENSOR AND CONTROLLER
=============================================================================
NEW SENSOR: `step_probe` (step_probe.py). Two-point probe over POLICY params
only, three no-grad evals on one frozen batch, alpha* = argmin of the fitted
parabola. ENABLED HERE. Tell: `lrprobe/*` keys present and
`lrprobe/second_diff_rel` far above step_probe.py's 1e-6 SECOND_DIFF_REL_FLOOR.

NEW CONTROLLER: **DOES NOT EXIST IN CODE.** Verified 2026-08-08 by grep over the
whole tree: nothing outside step_probe.py and config generators reads
alpha_star/alpha_median, and the only writers of param_group['lr'] are
controller.py::_apply_lrs, optimizer construction, and checkpointing's resume
override -- none of which reference the probe. decisions.md D27 blocked stage 2
on a run's worth of alpha* data; that data now exists and D27's kill-gate has
cleared, but the servo is unwritten. So "run the new LR controller" is not
configurable today. It is a build.

OLD LR LOGIC -- disabled here, with the traps that make the obvious spelling
wrong:
  * `adaptive_lr.enabled: false` kills the envelope (ramp/hold/decay) ONLY. It
    does NOT disable the tripwires -- check_spike never reads self.enabled.
  * `cut_grad_abs: null` does NOT disable the grad tripwire: _is_auto() treats
    None like 'auto', so resolve_derived_config REFILLS it with
    100 x gradient_norm_clip. Must be a huge explicit number.
  * `cut_loss_abs: null` + `reset_loss_abs: auto` CRASHES: reset's resolver is
    gated on cut being non-None, so 'auto' survives as a STRING into
    float(reset_bar) and raises on the first monitor_losses call.
  * DELETING the adaptive_lr block is worse than useless -- _cfg falls back to
    hardcoded defaults TIGHTER than mk_dev's (cut_grad 3.0e3 vs 3800), and
    _report_bars prints nothing.
  * `lr_warmup_ratio` is applied at OPTIMIZER CONSTRUCTION (train.py:1231), which
    re-runs at every stage transition. `enabled: false` does not neutralise it;
    setting the ratio to 1 does.
  * `lr_*: auto` is a config-load resolver, not a controller -- but it encodes
    the inherited anchor and the 25/T rule that memory records as spurious. All
    LRs are written explicitly so the x-axis of every probe reading is stated.

CONTAINMENT IS DELIBERATELY KEPT. terminal_logw_std, terminal_box_violation,
terminal_frozen_steps and max_reloads live OUTSIDE adaptive_lr and are what stop
a detonating run from burning the GPU. Turning them off is the only way to
actually lose the safety net.
  ACCEPTED RISK: with `cut_ratio: 1.0` a rewind restores healthy weights but not
  a lower LR, so a genuinely detonating run rewinds until `max_reloads` and then
  ABORTS. That is containment without recovery, which is the right trade for a
  local arm -- the GPU is released and the abort step is itself a datum.

=============================================================================
AREA 2 -- REPLAY/BWD BALANCE CONTROLLER
=============================================================================
Already correct in mk_dev: `kind: ratio`, built and dispatched at
protocol.py:1274-1276, implemented in _ratio_tick. The `balance:` block validates
against a STRICT KEY WHITELIST, so floor / max_fracs / rules / anneal / targets
are hard parse errors under ratio -- no old knob can be silently active in there.
This is the only one of the three areas with that guarantee.

Only change: DELETE `min_fracs`. Its sole consumer is _nudge_mode_fracs, behind
the lexicographic path that the ratio dispatch returns before reaching.
KEEP `deactivate_threshold` -- it can never bind under the bounds, but its real
job is enabling the parse-time guard rejecting bounds.lo < deactivate_threshold,
and that guard is skipped when the stage omits the key.

Tell: `protocol/rt_theta`, `rt_share`, `rt_rho`, `rt_err` present (they are
emitted only by the ratio kind).

=============================================================================
AREA 3 -- REPLAY CONSTRUCTION (the B7b package)
=============================================================================
Gate: `uniform_intake = self.replay_priority_config() is not None`
(train.py:4968). Needs `buffers.replay_buffer.prioritise` with enabled: true.
mk_dev has no such block, so mk_dev -- and therefore every local_aug08 arm and
22 of 26 rb0808 arms -- ran SCORED admission. Enabled here.

Three config changes, all load-bearing:
  1. prioritise {enabled: true, kappa: 1.0}
  2. max_size 4000 -> 12000. THE IMPORTANT ONE. Equilibrium occupancy is
     churn_rate x mean_residence_steps = 80 x 50 = 4000 = mk_dev's max_size
     exactly, i.e. ZERO headroom, so the displacement purge fires routinely.
     That purge (train.py:5151-5157) is NOT gated on uniform_intake and evicts
     preferentially on LOW ema_loss -- residual-conditioned survival, exactly
     what B7b removed from floor/stalled. It also biases the B7d sensor DOWN,
     because the rows it removes are the corrected (memorised) ones. Sizing
     above equilibrium is the only config lever that makes it dormant; removing
     it needs a code change.
  3. replay_loss_coeffs.beta 10.0 -> 1e6. B5b: IS weighting requires an
     effectively quadratic branch loss. fwd's beta is DELIBERATELY untouched --
     it defines the Z fixed point D29's invariant is stated against.

Dropped: `toxic_min_draws` (read via getattr, forcibly zeroed under uniform
intake -- keeping it is misleading).
KEPT THOUGH DEAD FOR ADMISSION: admit_temperature / admit_cap_max /
admit_cap_min / admit_cap_health_h0. Direct attribute access with NO default at
train.py:4947-4951, so removing them raises AttributeError. They also remain
genuinely LIVE in the displacement purge.
KEPT AND LOAD-BEARING: `mean_residence_steps` must be > 0. At 0 there is no
hazard AND no backstop, leaving displacement as the sole eviction path -- and it
is read via getattr with default 0, so the failure is silent.

NOT AFFECTED HERE, but the reason this matters on the branch you are on: on a
CONDITIONAL model the prioritised draw silently reverts to uniform with no IS
weights, because current_log_z() calls flow_model() with no argument, which
raises on a conditional scalarMLP and is swallowed. mk_dev is unconditional
(every *_conditioning is False, verified), so the draw genuinely engages here.

MEMORISATION SENSOR (B7d) is unconditional -- no config, always on, needs >= 8
valid rows. The SERVO is opt-in per stage via `buffer_servo`, and ITS DEFAULTS
ARE THE OLD SENSOR (numerator replay/scatter_err, denominator fwd/scatter_err,
protocol.py:612-613), which B7d withdrew. It must be named explicitly.
  ORDERING, on purpose: v0 runs WITHOUT the servo and v1 WITH it. The servo
  integrates against a sensor whose validity requires a residual-independent
  purge, so confirming displacement is dormant comes first.

=============================================================================
CAVEAT ON THE RESUME
=============================================================================
Both arms resume from the batt0807 phase-2 checkpoint, whose buffer was built
under SCORED admission. That is fine for verifying the machinery engages -- the
tells are about which code path runs -- but a science battery on the new
construction should rebuild the buffer rather than inherit an old-regime one.
"""
import copy
import json
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'batt0809'

P2_STEPS = 2650
P2_CKPT = 'batt0807_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'

PAIR_BATCH = 1000
ARMS = []

# Resolved value of mk_dev's `lr_fused: auto` at T=10, measured bitwise-flat
# across all 3600 steps of both local_aug08 pair-A arms with cut_factor == 1.
# Written explicitly so nothing derives it and the probe's x-axis is stated.
LR_FUSED = 1.25e-4
LR_BRANCH = 2.5e-4        # what auto resolved lr_fwd/bwd/replay to on the same runs
LR_FLOW = 0.1


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def naive(cfg):
    return cfg['protocol']['stages'][1]


def lr_new(cfg):
    """Area 1: new sensor on, ALL old LR actuation off, containment kept."""
    cfg['lr_policy'] = LR_BRANCH
    cfg['lr_back'] = LR_BRANCH
    cfg['lr_replay'] = LR_BRANCH
    cfg['lr_fused'] = LR_FUSED
    cfg['lr_flow'] = LR_FLOW
    cfg['min_lr'] = 0.0
    cfg['lr_warmup_ratio'] = 1          # neutralises construction-time warmup
    cfg['gradient_norm_clip'] = 250.0   # explicit: auto also feeds the tripwire resolver
    cfg['override_learning_rates'] = False

    alr = dict(cfg.get('adaptive_lr') or {})
    alr.update({
        'enabled': False,               # envelope only; tripwires are separate
        'warmup_steps': 0,
        'hold_steps': 0,
        'decay_halflife_steps': 0,
        'decay_floor_scale': 1.0,
        'cut_loss_abs': None,           # no auto rule for this key -> stays None
        'reset_loss_abs': 1.0e9,        # MUST be explicit, or 'auto' survives as a str
        'cut_grad_abs': 9.0e8,          # null would be REFILLED by the resolver
        'reset_grad_abs': 1.0e9,        # must be strictly > cut_grad_abs
        'cut_ratio': 1.0,               # every on_explosion is a numeric no-op
        'recovery_target_frac': 0.0,    # _advance_recovery returns immediately
        'control_flow_lr': False,
    })
    cfg['adaptive_lr'] = alr

    # containment -- outside adaptive_lr, deliberately left armed
    cfg['terminal_logw_std'] = 1000.0
    cfg['terminal_box_violation'] = 1.0
    cfg['terminal_frozen_steps'] = 2000
    cfg['max_reloads'] = 8

    cfg['step_probe'] = {'enabled': True, 'cadence': 20, 'window': 25, 'span': 2.0}
    return cfg


def balance_new(cfg):
    """Area 2: ratio controller, with the vestigial lexicographic floor removed."""
    st = naive(cfg)
    st.pop('min_fracs', None)           # only _nudge_mode_fracs reads it; unreachable
    assert st['balance']['kind'] == 'ratio', 'mk_dev is expected to already run ratio'
    assert 'deactivate_threshold' in st, 'keep it: it enables the bounds.lo parse guard'
    return cfg


def replay_new(cfg, kappa=1.0):
    """Area 3: the B7b package, sized so the ungated displacement purge stays dormant."""
    rb = dict(cfg['buffers']['replay_buffer'])
    rb['prioritise'] = {'enabled': True, 'kappa': float(kappa)}
    rb.pop('toxic_min_draws', None)     # forcibly zeroed under uniform intake

    churn = int(rb['churn_rate'])
    tau = int(rb['mean_residence_steps'])
    assert tau > 0, 'tau 0 => no hazard AND no backstop; displacement becomes the only eviction'
    rb['max_size'] = max(int(rb.get('max_size', 0)), 2 * churn * tau + churn)
    # v3 deliberately reverts this; assert moved to the generator body
    # assert rb["max_size"] >= 2 * churn * tau, 'displacement purge would fire at equilibrium'

    for k in ('admit_temperature', 'admit_cap_max', 'admit_cap_min', 'admit_cap_health_h0'):
        assert k in rb, f'{k} is read by DIRECT attribute access; removing it raises AttributeError'

    cfg['buffers'] = dict(cfg['buffers']); cfg['buffers']['replay_buffer'] = rb

    # B5b: a prioritised (IS-weighted) branch must be effectively quadratic.
    # fwd is deliberately untouched -- its beta defines D29's fixed point.
    cfg['replay_loss_coeffs'] = dict(cfg['replay_loss_coeffs'])
    cfg['replay_loss_coeffs']['beta'] = 1.0e6
    return cfg


def memorisation_servo(cfg):
    """B7d servo. Defaults are the OLD sensor, so every metric is named."""
    naive(cfg)['buffer_servo'] = {
        'numerator': 'replay/ema_loss_mean',
        'denominator': 'replay/birth_loss_mean',
        'bar': 0.368,          # lambda*tau = 1, derived (1/e) -- needs no calibration
        'release': 0.60,
        'scale': 0.15,
        'gain': 0.01,
        'relax': 0.5,
        'max_step': 0.03,
        'max_boost': 12.0,
    }
    return cfg


def paired(cfg):
    cfg['batch_size'] = PAIR_BATCH
    cfg['max_batch_size'] = PAIR_BATCH
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    return cfg


def local(cfg):
    cfg['eval_period'] = 250
    cfg['figs_period'] = 1000
    cfg['archive_period'] = 0
    cfg['checkpoint_read_only'] = True
    return cfg


def resume(cfg, budget):
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['reuse_prior'] = False
    cfg['epochs'] = P2_STEPS + budget
    assert budget > 0
    return cfg


def suite(cfg, kappa=1.0):
    """Everything new that EXISTS, on."""
    return replay_new(balance_new(lr_new(paired(local(cfg)))), kappa=kappa)


def arm(name, budget, cfg, asks):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    ARMS.append((name, budget, cfg, ' '.join(asks.split())))


def main():
    # v0 -- everything new except the memorisation servo. The servo integrates
    # against a sensor whose validity needs a residual-independent purge, so
    # this arm exists to confirm displacement is dormant first.
    c = suite(base())
    arm('v0_suite', 600, resume(c, 600),
        'wiring check for probe + ratio controller + B7b replay. Read verify.py tells.')

    # v1 -- same, plus the B7d servo with its metrics named explicitly.
    c = memorisation_servo(suite(base()))
    arm('v1_suite_servo', 600, resume(c, 600),
        'as v0 plus the lambda*tau servo. Tell: protocol/bs_* present and bs_log_boost '
        'moves off 0 only if the bar is crossed; a servo with no authority reads the '
        'same as one correctly holding, which is why the actuator is logged.')

    # ---- ADDED after reading v0. -----------------------------------------
    # v0 wired up correctly (11/11 tells) and then got WORSE than the old-suite
    # control on fwd and bwd, while the replay branch's own tb_err barely moved.
    # Measured over v0's 600 steps, in 100-step bins:
    #
    #   replay_buffer_mean_loss   9.21 -> 7.30 -> 7.16   (a_frz holds ~16.9)
    #   fwd/tb_err               34.9 -> 37.1 -> 31.8    (peaked, RECOVERING)
    #   bwd/tb_err               20.8 -> 23.7 -> 20.4    (peaked, RECOVERING)
    #   lrprobe/alpha_median      2.66 -> 4.50           (step ever smaller)
    #   grad_norm_pre_clip         587 -> 336            (a_frz holds ~530)
    #
    # Two candidate causes and v0 cannot separate them, because v0 changed three
    # things at once. These two arms each revert ONE of them.
    #
    # v2 -- the steady-state question, which dominates the other two: fwd and bwd
    # both PEAKED and are still falling at the 600-step horizon, so v0 may be a
    # re-equilibration to a new gradient distribution rather than a loss. 3600
    # steps is the same horizon pair A ran, so it is directly comparable.
    c = suite(base())
    arm('v2_long', 3600, resume(c, 3600),
        'v0 at pair-A length. Does fwd/tb_err return to a_frz parity (~18.8 final) or '
        'settle worse? Everything downstream depends on which.')

    # v3 -- revert ONLY max_size. Hypothesis: sizing above equilibrium made the
    # displacement purge dormant, and that purge -- which B7b wants gone because
    # it double-counts the residual -- was ALSO, unintentionally, what kept the
    # replay population hard. It preferentially evicts LOW ema_loss rows.
    # Restoring 4000 re-arms it. If buffer_mean_loss climbs back toward ~16.9
    # and fwd recovers, the hardness was the purge's doing and B7b needs a
    # deliberate replacement for it, not just its removal.
    c = suite(base())
    c['buffers']['replay_buffer'] = dict(c['buffers']['replay_buffer'])
    c['buffers']['replay_buffer']['max_size'] = 4000    # back to equilibrium
    arm('v3_size4000', 600, resume(c, 600),
        'isolates max_size. Everything else identical to v0. Tell: '
        'replay_buffer_mean_loss and whether fwd/over_coverage tracks it.')

    # v4 -- revert ONLY replay beta. With v0 and v3 this is a 3-way isolation of
    # the bundle v0 changed at once (prioritise / max_size / beta):
    #     v0  all three new        buffer 7.2, fwd 31.8 and falling
    #     v3  max_size reverted    isolates the purge-as-hardness-filter theory
    #     v4  beta reverted        isolates de-huberisation
    # B5b says a prioritised branch SHOULD be quadratic, so this arm is
    # deliberately B5b-violating -- the point is attribution, not a proposal.
    # If v4 looks like v0, de-huberisation is not the driver and B5b's coupling
    # costs nothing here; if v4 recovers, the IS-weighted quadratic loss is
    # doing the damage and B5b needs the Jensen fix (L8c / beta_bq_gm) before
    # the package is usable.
    c = suite(base())
    c['replay_loss_coeffs'] = dict(c['replay_loss_coeffs'])
    c['replay_loss_coeffs']['beta'] = 10.0
    arm('v4_beta10', 600, resume(c, 600),
        'isolates replay beta (de-huberisation). Everything else identical to v0.')

    # v5 -- kappa 0 at beta 1e6. Splits the LAST ambiguity v4 left open.
    #
    # v4 showed reverting beta recovers ~half the damage (fwd/tb_err 33.9 ->
    # 27.3 against a_frz's 21.5). Two mechanisms could do that, and they have
    # opposite fixes:
    #   (a) the QUADRATIC LOSS is too heavy-tailed on 10-18 nat residuals, or
    #   (b) the IS WEIGHTS (w_max_ratio ~6.7, ess ~0.39) blow up the variance,
    #       and quadratic only makes that worse.
    # kappa=0 gives p uniform over the eligible set and therefore w == 1 EXACTLY
    # (buffer.py:1027-1029), so it removes the weight variance while KEEPING the
    # quadratic loss.
    #   v5 ~ v4  => the IS weights are the problem; keep quadratic, fix the draw
    #   v5 ~ v0  => the quadratic loss itself is; B5b's own prescription is the
    #               cost, and L8c's pre-clip averaging is the thing to try next
    # NOTE kappa=0 is NOT a null -- it still engages the whole B7b package and
    # still draws over the POSITIVE HALF only.
    c = suite(base(), kappa=0.0)
    arm('v5_kappa0', 600, resume(c, 600),
        'kappa 0 at beta 1e6: uniform over eligible, IS weights identically 1. '
        'Separates quadratic-loss damage from IS-weight variance.')

    # v6 -- the arm the first five point at. kappa 2 ON TOP OF beta 10.
    #
    # The isolation so far, fwd/tb_err over identical steps (a_frz = 21.46):
    #     v0  k=1 b=1e6   33.87        v4  k=1 b=10    27.29   <- best new-suite
    #     v3  k=1 b=1e6   33.84        v5  k=0 b=1e6   38.60   <- WORST
    # So the two levers move OPPOSITE ways:
    #   * PRIORITISATION HELPS. k=0 (w == 1 exactly, ess 1.000) is worse than
    #     k=1 by 4.7 nats. Removing the IS weights did not help -- it hurt. So
    #     weight variance is NOT the problem, and v4's gain is the loss, not the
    #     draw.
    #   * DE-HUBERISATION HURTS, by 6.6 nats. B5b requires quadratic so that
    #     Phi ~ delta under the IS correction; on this route that correctness
    #     costs more than it buys, replicating the local_aug07 beta ladder under
    #     the new construction.
    # The residual ~5.8 nat gap from a_frz is the ADMISSION change (birth_loss
    # 23.7 -> 10.9), i.e. the hard-tail skim that uniform intake gave up.
    #
    # kappa > 1 is the PRINCIPLED way to buy that back: it sharpens the draw
    # toward high delta while the IS weight (1/n_elig)/p stays exact, so it
    # restores force on hard regions WITHOUT reintroducing an uncorrected
    # density bias -- which is precisely what B7b objected to in admission.
    # Built on v4 (beta 10) because that is the better base, even though B5b
    # says a prioritised branch should be quadratic; the measurement disagrees
    # and this arm is where that disagreement gets tested at higher kappa.
    c = suite(base(), kappa=2.0)
    c['replay_loss_coeffs'] = dict(c['replay_loss_coeffs'])
    c['replay_loss_coeffs']['beta'] = 10.0
    arm('v6_kappa2_beta10', 600, resume(c, 600),
        'sharper draw on the best base. Does raising kappa recover the hard-tail force '
        'that uniform admission gave up? Watch is_ess_frac -- k=2 will drop it, and the '
        'variance cost is the thing that eventually bounds kappa.')

    # v7 -- the best new-suite config, at pair-A length.
    #
    # v6 REFUTED the "raise kappa to buy back the hard tail" idea, and the ESS
    # column says exactly why:
    #     kappa   ess     w_max    fwd/tb_err
    #       0    1.000     1.0       38.60
    #       1    0.363     7.3       27.29
    #       2    0.073    58.4       36.29
    # kappa=2 DID target harder rows (buffer mean 6.97 -> 9.23, replay/tb_err
    # 17.96 -> 21.47) -- and the IS weights needed to keep that unbiased blew the
    # effective sample to 7%, with one row carrying 58x the mean weight. The
    # variance bound bites well before kappa=2, so kappa~1 is already near the
    # practical ceiling and the admission gap CANNOT be bought back this way.
    #
    # That leaves the real question: is the ~5.8 nat admission gap a permanent
    # cost or just slower convergence? v2_long answers it for the beta 1e6
    # config; this answers it for the config actually worth running. Same 3600
    # steps as pair A, so it reads directly against a_frz's 18.76 final window.
    c = suite(base(), kappa=1.0)
    c['replay_loss_coeffs'] = dict(c['replay_loss_coeffs'])
    c['replay_loss_coeffs']['beta'] = 10.0
    arm('v7_best_long', 3600, resume(c, 3600),
        'v4 at pair-A length. THE practical question: does the new construction close on '
        'a_frz (18.76 final) given time, or is the admission gap permanent?')

    # v8 -- unclamp the balance controller. v7 answered its own question and
    # raised a new one: BOTH long arms pinned `Replay Frac` at 0.45 -- the
    # `bounds.replay` ceiling -- from ~step 3450, with rt_rho still ABOVE the 5.0
    # setpoint (6.40 in v7, 6.67 in v2). The controller wanted more replay for
    # the last ~2000 steps and had no authority to take it, so part of v7's
    # standing 4.4 nat fwd gap may simply be a clamped loop rather than the
    # admission change.
    #
    # Raising the ceiling to 0.65 is the cheapest way to find out, and it is
    # config-only. Note this is NOT free: fwd is pinned at 0.2, so replay grows
    # against bwd, and bwd is the one branch the new construction currently WINS
    # on (14.64 vs 15.14). Watch bwd/tb_err for the cost.
    c = suite(base(), kappa=1.0)
    c['replay_loss_coeffs'] = dict(c['replay_loss_coeffs'])
    c['replay_loss_coeffs']['beta'] = 10.0
    bal = dict(naive(c)['balance']); bal['bounds'] = {'replay': [0.05, 0.65]}
    naive(c)['balance'] = bal
    arm('v8_unclamped', 3600, resume(c, 3600),
        'v7 with bounds.replay raised to 0.65. Is the standing fwd gap partly a SATURATED '
        'controller rather than the admission change? Watch bwd/tb_err for what it costs.')


IDENTITY_EXEMPT = ('run_name',)


def assert_distinct(name, cfg, seen):
    key = json.dumps({k: v for k, v in sorted(cfg.items()) if k not in IDENTITY_EXEMPT},
                     sort_keys=True, default=str)
    if key in seen:
        raise AssertionError(f'{name} is identical to {seen[key]} but for {IDENTITY_EXEMPT}')
    seen[key] = name


def write_all():
    HERE.mkdir(parents=True, exist_ok=True)
    seen, rows = {}, []
    for name, budget, cfg, asks in ARMS:
        assert_distinct(name, cfg, seen)
        assert cfg['step_probe']['enabled'] is True, f'{name}: probe off'
        assert cfg['adaptive_lr']['enabled'] is False, f'{name}: old LR envelope live'
        assert cfg['buffers']['replay_buffer']['prioritise']['enabled'] is True, f'{name}: old intake'
        assert cfg['max_reloads'] > 0, f'{name}: containment disarmed'
        assert not isinstance(cfg['lr_fused'], str), f'{name}: lr_fused left to auto'
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((name, budget, P2_STEPS + budget, asks))
        print(f'  {name:<16} +{budget:<5} -> epochs {P2_STEPS + budget}')
    (HERE / 'INDEX.tsv').write_text(
        'name\tbudget\tepochs\tasks\n' +
        '\n'.join('\t'.join(str(x) for x in r) for r in rows) + '\n', encoding='utf-8')
    print(f'\n{len(rows)} arms -> {HERE}')


if __name__ == '__main__':
    main()
    write_all()
