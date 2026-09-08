"""rr_sep08 -- the 18-arm overnight calibration battery for rarer rollouts v2.

    python configs/rr_sep08/make.py

ONE WAVE, 12-HOUR WALL, SINGLE LEG, UNATTENDED. There is no smoke set: the owner
is not awake to read one, so the gates a smoke wave would have provided are
replaced by generation-time assertions here plus tests/protocol/test_cadence_anchor.py
(the one failure a local run structurally cannot reproduce). Nothing chains -- the
lj_coeff resume crash is still open, so a killed arm is a dead arm and --requeue
would only restart it into the same crash.

THE CONTRACT COMES FROM rr_sep07/make.py, NOT FROM HERE. This module imports
`deltas()` from it, so the shipped v2 settings have exactly one home: bar 0,
ratchet_tol 0, guard channel bwd/under_coverage_rise150, ratchet on that channel's
LEVEL, the three cadence triggers, val_cap = batch_size, absorber + eval fill,
anchor-only prior. Each arm then applies ONE override on top, and `check()` below
re-asserts the whole contract afterwards so an override cannot silently take a
second key with it.

WHAT IS BEING ASKED, one block at a time:

  block 1  the controller's GAINS AND RAILS (the channel and both bars are the
           shipped contract and are not varied). ratchet_tol 0 clamps the ramp on
           any plateau and bwd rails at 0.9; tol05 and cap75 relieve that from
           opposite sides, fast4 asks whether tracking beats railing.
  block 2  CADENCE AND BUFFER COMPOSITION -- N and tau/N, never varied
           independently of each other, and the biggest blind spot in the design.
           N=50 is where invariant 1 ("the unpinned interval is bounded") gets
           checked rather than assumed; the ess_min bar only becomes live there.
  block 3  the MLIP cost ladder. Rarer rollouts removes GPU work while the
           host-side idle time per step stays, so it lowers utilization -- into
           the ~54% two-hour kill line that took five of five mipu arms on
           prod_sep02. Batch is the lever that buys the margin back.
  block 4  P_B frozen at T=100 (the local A/B was T=10, 3000 steps, one seed).

/!\\ THE BOOTSTRAP DEFECT THIS GENERATOR FIXES. rr_sep07's deltas() appends
`bootstrap_z:rollout:4000` only `if not any(a.startswith('bootstrap_z'))`. The
ACTIVE protocol on every one of these arms is `prod_eq`, whose fused stage already
carries `bootstrap_z:train_conditioner` -- so the append was skipped and the
rollout bootstrap has never run on any rr07 arm. rr_sep07's check() passes because
it asserts only that SOME action starts with `bootstrap_z`, which the pre-existing
one satisfies. `_force_rollout_bootstrap` REPLACES instead of skipping, and
check() asserts the exact string on the ACTIVE protocol's fused stage. (The two
actions are different mechanisms, not variants: the bare one regresses the flow
head onto the tracker's ema_logw with no rollout and no reward call; `:rollout:n`
takes n forward samples and sets log Z to their winsorized-Huber root, which is
the estimator the TB loss actually optimises.)

/!\\ THE BUFFER CAP HAS TO TRACK THE BATCH. Occupancy is `churn_rate * tau/N`
(admissions happen on rollout steps only), and churn_rate = batch_size. At the
shipped tau/N = 5 that is 5000 rows on ELJ (batch 1000) and 8000 on UMA (1600),
both under `max_size` 12000 -- but the block-3 ladder raises batch to 3200 and
6400, which would need 16000 and 32000. The cap would bind, hazard-based eviction
would stop being the mechanism, and the two arms the ladder exists for would
silently measure something else. `_size_replay` recomputes max_size from
churn_rate and tau/N on EVERY arm, and check() asserts the headroom.
"""
import importlib.util
import math
import pathlib
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
WALL = '12:00:00'
TAG = 'rr08'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rr07make = _load('rr07make', ROOT / 'rr_sep07' / 'make.py')
p02make = rr07make.p02make

# Ship defaults for this battery. N=20 because the local cadence sweep put
# N=20 ~ N=7 in quality per step with ELJ speed saturating ~2.7x by then, so it
# is the value most likely to ship; tau/N = 5 is rr_sep07's.
N_SHIP = 20
# LOOSE BACKSTOP for the bar sweep: the deterministic maximum that bounds time
# without fresh forward samples, while the val_gap bar sets the real cadence.
N_LOOSE = 200
TAU_OVER_N_SHIP = 3
# HARD FLOOR ON tau/N (owner 2026-09-08). Occupancy O = churn * tau/N and the draw
# takes a full batch every step, so O/B = tau/N -- the "occupancy in batches" the
# trigger reads IS this ratio. Below 2 the draw starts repeating itself out of a
# pool smaller than two batches.
#
# WHY THE GUARANTEE IS WORST-CASE: fwd_rollout_every is a MAXIMUM, not a target --
# triggers only ever SHORTEN the interval, which raises the admission rate, which
# raises occupancy. So tau = TAU_OVER_N_MIN * N bounds O/B from BELOW for any
# trigger behaviour; the ratio can only come out higher than configured.
TAU_OVER_N_MIN = 3
# ...and the trigger bar sits BELOW the floor on purpose. If bar == floor, the
# boundary arm reads exactly the floor and fires on any transient -- a bar that
# merely restates the config.
#
# THE SAWTOOTH IS WHY THE MARGIN HAS TO BE ONE HALF-BURST. Admissions arrive as a
# burst of B every N steps and the hazard drains ~B between bursts, so occupancy
# cycles from O+B/2 down to O-B/2 and the TROUGH is what the bar sees:
#
#     O_trough / B  =  tau/N - 0.5
#
# At tau/N = 2 the trough is 1.5, so a 2.0 bar fires every cycle. At tau/N = 3 the
# trough is 2.5 and a 2.0 bar has 25% headroom, firing only on a genuine shortfall
# (initial fill, a batch growth the buffer has not caught up with, a servo churn
# boost). Corroborated: local rr_n7_v1 ran length 2024-2381 against B=400, i.e. an
# amplitude of ~B as predicted. Owner call 2026-09-08: the floor is "never below 2
# batches", so the floor moves to 3 and the bar to 2.
OCCUPANCY_BAR = 2.0
ROLLOUT_BOOTSTRAP = 'bootstrap_z:rollout:4000'
GUARD_METRIC = 'bwd/under_coverage_rise150'
GUARD_LEVEL = 'bwd/under_coverage'
# ess_min retired 2026-09-08 -- Kish on exp(d) is scale-invariant, so it reads
# 1.000 for a uniform collapse and cannot see the policy leaving the buffer.
TRIGGER_KEYS = ('val_gap_max', 'occupancy_min_batches')

# (arm, family, N, overrides). EXACTLY ONE override per arm, except where the
# override is only meaningful in combination (n50_ess25: the ess bar is inert at
# N=20 because nothing drifts far enough to fire it, so the bar has to be varied
# at the cadence where it is live; pbf_mipu: the freeze needs a batch that
# survives the night, and block 3 predicts 3200 is it).
# RETIRED 2026-09-08, AFTER the local harm-curve work. These three keep their
# INDEX ROW -- the sbatch resolves an arm from `INDEX.tsv` line
# SLURM_ARRAY_TASK_ID + 2 AT TASK START, so deleting a row would silently re-map
# every queued task below it onto a different arm. The row stays, the config is
# not written, and the sbatch's `missing config` guard fails those tasks fast
# (before any checkpoint glob). --array is unchanged at 0-17.
#
#  vg10   the bar (1.0) sits BELOW the gap it must clear. The block-2 comment
#         already makes this argument at N=20 ("a lower bar can never be
#         satisfied, drives N_eff to the 2-step trigger floor"); at N_LOOSE the
#         gap is LARGER, so 1.0 latches for the same reason. A latched trigger
#         runs at the maximal rollout rate for the rest of the arm -- the most
#         expensive failure mode there is on an energy-bound route.
#  tau6   tau has NO CHANNEL to what block 3 says it separates. A row's exposure
#  tau12  is tau/O = 1/admissions (branch losses are normalised MEANS), so tau
#         and occupancy CANCEL EXACTLY; and policy drift saturates at cluster LR
#         (drift_std flat to slightly inverted over a 5.2x span of mean age), so
#         the staleness half is inert too. tau12 additionally reaches 48000 rows
#         against max_size 50000 at full batch and would run cap-bound, which is
#         the degenerate state that voided the local re-run.
# Their three SLOTS are reused below by the dose arms, which is why RETIRED
# is now empty: the rows are occupied again, so nothing re-maps either way.
RETIRED = set()

ARMS = (
    # -- block 1: CONTROLLER (already submitted; rows 0-2 are running or crashed
    #    and must keep their index so the INDEX still documents what they ran) --
    ('base',        'mip',  N_SHIP, {}),
    ('tol05',       'mip',  N_SHIP, {'ratchet_tol': 0.5}),
    ('cap75',       'mip',  N_SHIP, {'bwd_hi': 0.75}),

    # -- block 2: THE val_gap BAR SWEEP -- the point of the battery -------------
    # The bar is the KNOB and the effective cadence is the READOUT. Loose
    # backstop (N_max 200), so each arm's trigger pulls the cadence in to
    # wherever its bar is satisfied: reuse grows the gap, the gap hits the bar,
    # the trigger fires, admissions rise, reuse falls. Equilibrium is gap ~ bar
    # at whatever N_eff achieves it, so sweeping the bar traces the
    # quality-vs-cost curve with each arm self-selecting one point on it.
    # Read rollout/rate as an OUTPUT.
    #
    # No bar below 1.0: the measured gap at N=20 is 1.25, so a lower bar can
    # never be satisfied, drives N_eff to the 2-step trigger floor, and asks for
    # ~300 batches of buffer. That arm would spend 12h pinned against max_size
    # measuring overflow eviction.
    # -- SLOT 3 (was vg10). THE DOSE TEST ---------------------------------------
    # Memorisation goes as D = N * lr * w_eff / B, fitted at lambda ~ 0.0195*D^0.75
    # (R2 0.935) over six local arms spanning 12.8x in D. `w` is the ONE factor in
    # that law never varied anywhere in the corpus -- its exponent is assumed, not
    # measured -- and it is also the only factor that is FREE: N costs energy
    # calls, w costs nothing.
    #
    # These three arms and the existing n20_det form a D-matched set reached by
    # three different (N, w) routes, plus a pure-w contrast:
    #
    #   n20_det    N= 20  tb=1.0   D ~  20   (already in the battery, position 9)
    #   dose_n50   N= 50  tb=0.4   D ~  20   matched, 2.5x fewer energy calls
    #   dose_n200  N=200  tb=0.1   D ~  20   matched, 10x fewer energy calls
    #   dose_w10   N= 20  tb=0.1   D ~   2   pure w at fixed N -- the exponent
    #   vg_off     N=200  tb=1.0   D ~ 200   the UNCOMPENSATED control (position 6)
    #
    # PREDICTION: the three D~20 arms land together on lambda_tau and
    # val_gap_nats despite a 10x span in N and in energy cost; dose_w10 sits
    # 10^0.75 = 5.6x below them; vg_off sits 5.6x above.
    # IF IT HOLDS, the cadence penalty is cancellable in config -- ship
    # replay_tb ~ 20/N and let N be set by the speedup curve alone. IF IT FAILS,
    # w is not interchangeable with N and per-row exposure separates from
    # support recency, which no run in the corpus currently distinguishes.
    #
    # All three are DETERMINISTIC (val_gap_max 0): a firing trigger would change
    # admissions and so change the dose being matched. check() enforces it.
    ('dose_n200',   'mip', N_LOOSE, {'replay_tb': 0.1, 'val_gap_max': 0.0}),
    ('vg20',        'mip', N_LOOSE, {'val_gap_max': 2.0}),
    ('vg40',        'mip', N_LOOSE, {'val_gap_max': 4.0}),
    # bar 0 disables it (_rollout_trigger_fires skips bars <= 0): the
    # uncontrolled endpoint of the same curve, i.e. deterministic N = N_LOOSE.
    ('vg_off',      'mip', N_LOOSE, {'val_gap_max': 0.0}),

    # -- block 3: the tau axis at a fixed bar -----------------------------------
    # tau has NEVER been varied independently of N -- every arm ever run set
    # tau = 5N. At a fixed bar this separates staleness and occupancy from reuse.
    # NB these carry the SHIPPED bar (4.0), not 2.0 -- one override per arm, and
    # the shipped bar is the right control for a tau sweep. Named for what they
    # actually vary.
    # -- SLOTS 7-8 (were tau6/tau12). The rest of the D-matched set; see slot 3.
    ('dose_n50',    'mip', 50,      {'replay_tb': 0.4, 'val_gap_max': 0.0}),
    ('dose_w10',    'mip', N_SHIP,  {'replay_tb': 0.1, 'val_gap_max': 0.0}),

    # -- block 4: deterministic reference points --------------------------------
    # Fixed cadence, no adaptive bar, so an adaptive arm that settles at N_eff=X
    # can be read against a deterministic arm actually run at N=X. n20_det is
    # also the ONLY clean comparison to base/tol05/cap75, which ran N=20 under
    # the retired ess_min bar -- it isolates what retiring it changed.
    ('n20_det',     'mip',      20, {'val_gap_max': 0.0}),
    ('n50_det',     'mip',      50, {'val_gap_max': 0.0}),

    # -- block 5: MLIP cost ladder ----------------------------------------------
    # Where the wall-clock payoff lives, and where utilisation binds: the running
    # ELJ arms sit at ~61% mean GPU util with 5.7% memory allocated, and rare
    # rollouts REMOVE the energy work that was filling the card. Batch is the
    # lever that refills it, and it is nearly free here because energy cost is
    # batch * rollout_rate and the rate is already cut ~15x.
    ('mipu_b1600',  'mipu', N_LOOSE, {}),
    ('mipu_b3200',  'mipu', N_LOOSE, {'batch': 3200}),
    ('mipu_b6400',  'mipu', N_LOOSE, {'batch': 6400}),

    # -- block 6: transfer to other systems --------------------------------------
    ('nehu_b1600',  'nehu', N_LOOSE, {}),
    ('acr_b1000',   'acr',  N_LOOSE, {}),

    # -- block 7: P_B frozen at T=100 (separate workstream) ----------------------
    ('pbf_mip',     'mip',  N_LOOSE, {'freeze_pb': True}),
    ('pbf_mipu',    'mipu', N_LOOSE, {'freeze_pb': True}),

    # -- block 7: LET THE BUFFER GET LARGE (appended -- tasks 18-19) -------------
    # O = B * tau/N depends ONLY on tau/N, not on N, so the occupancy sweep is
    # run at N_SHIP where tau is SHORT and the buffer actually equilibrates: tau
    # = 60 / 240 / 480 steps, i.e. 5*tau = 300 / 1200 / 2400. At N_LOOSE the same
    # occupancies need tau = 600 / 2400 / 4800 and 5*tau up to 24000 steps, which
    # a 12h wall may not reach -- that arm would report a transient as a level.
    #
    #   n20_det   tau/N= 3   O =  3B  (already in the battery, task 9 -- CONTROL)
    #   tau12     tau/N=12   O = 12B  ->  48000 rows at the grown batch
    #   tau24     tau/N=24   O = 24B  ->  96000 rows at the grown batch
    #
    # All three N=20, tb=1.0, deterministic, so the DOSE is identical and only
    # occupancy and age differ. PREDICTION: lambda_tau and val_gap_nats are FLAT
    # across all three, because a row's exposure is tau/O = 1/admissions and tau
    # cancels against occupancy exactly. This is the assumption the tau6/tau12
    # pull rested on, and it has never been tested at scale -- the only large
    # buffer ever run was cap-bound, which measures nothing. If absorption moves
    # with tau, the cancellation is wrong and the dose law loses its denominator.
    # Second-order channels that do NOT cancel and are the reason to look:
    # draw diversity (B drawn from O), priority staleness (redraw spacing O/B),
    # and displacement eviction breaking birth_loss as an intake baseline.
    ('tau12',       'mip',  N_SHIP,  {'tau_over_n': 12, 'val_gap_max': 0.0}),
    ('tau24',       'mip',  N_SHIP,  {'tau_over_n': 24, 'val_gap_max': 0.0}),
)


OVERRIDE_KEYS = {'ratchet_tol', 'bwd_hi', 'gain_mult', 'tau_over_n',
                 'val_gap_max', 'fill_process_var', 'batch', 'freeze_pb',
                 'replay_tb'}


def _fused(cfg, active_only=False):
    protos = ([cfg['protocols'][cfg['protocol']]] if active_only
              else list((cfg.get('protocols') or {}).values()))
    return [s for p in protos for s in (p.get('stages') or [])
            if s.get('train_mode') == 'fused']


def _force_rollout_bootstrap(cfg):
    """REPLACE every bootstrap_z action on every fused stage with the rollout
    form. See the module docstring: appending only when none is present is what
    made this a no-op on all 12 rr_sep07 arms."""
    for st in _fused(cfg):
        on_enter = [a for a in (st.get('on_enter') or [])
                    if not str(a).startswith('bootstrap_z')]
        on_enter.append(ROLLOUT_BOOTSTRAP)
        st['on_enter'] = on_enter


def _size_replay(cfg, tau_over_n):
    """max_size as an honest MEMORY BUDGET, sized for the range the controller
    actually operates in rather than for the idle case.

    tau is fixed in STEPS, so occupancy O = B * tau / N_eff grows as 1/N_eff --
    and N_eff is what the val_gap trigger pulls DOWN. Sizing max_size from N_max
    (i.e. from tau_over_n alone) caps the buffer at the one cadence where the
    trigger never fires, so it binds the moment it does.

    When it binds, admission uniformly purges live rows to make room
    (train.py, `headroom`/`extra_purge`) -- residual-independent, so composition
    is NOT distorted. What is lost is tau: effective residence becomes
    max_size * N_eff / B and the configured tau goes inert. Verified on
    rr08_tol05: max_size 12000, grown batch 4000, N_eff 14.7 -> tau_eff ~44
    against a configured 100, and mean_age 21 ~ tau_eff/2.

    50 * B covers N_eff down to N_max/50 at tau/N = 3. Past that the arm is
    running tau-disconnected, which is fine if deliberate and bad if silent --
    watch replay_buffer_length == max_size.
    """
    rb = cfg['buffers']['replay_buffer']
    # SIZE FROM THE BATCH THE ARM GROWS TO. churn_rate is the ENTRY batch; with
    # grow_batch_size the draw reaches max_batch_size, and O = B*tau/N grows with
    # it. Sizing from churn alone under-sizes by exactly that ratio -- which is
    # what put the old tau12 at 48000 rows against a 50000 cap (entry 1000, grown
    # 4000, tau/N 12) and would have run it cap-bound, i.e. tau-disconnected, the
    # one state that measures nothing. No tau/N = 3 arm changes: churn*50 still
    # dominates for all of them.
    grown = max(int(rb['churn_rate']), int(cfg.get('max_batch_size') or 0))
    occupancy = grown * tau_over_n
    rb['max_size'] = max(12000, int(rb['churn_rate']) * 50, int(math.ceil(occupancy * 1.25)))


def _apply(cfg, every, ov):
    bad = set(ov) - OVERRIDE_KEYS
    assert not bad, 'unknown override keys %s' % sorted(bad)
    tau_over_n = ov.get('tau_over_n', TAU_OVER_N_SHIP)

    if 'batch' in ov:
        # churn_rate and val_cap are DERIVED from batch_size by rr07's deltas(),
        # so all three move together or the contract's own assertions fail.
        b = int(ov['batch'])
        cfg['batch_size'] = b
        cfg['max_batch_size'] = max(int(cfg.get('max_batch_size') or 0), b)
        rb = cfg['buffers']['replay_buffer']
        rb['churn_rate'] = b
        rb['val_cap'] = b

    if tau_over_n < TAU_OVER_N_MIN:
        raise ValueError(f"tau_over_n={tau_over_n} is below the occupancy floor "
                         f"{TAU_OVER_N_MIN}: O/B = tau/N would leave the draw taking a "
                         f"full batch from a pool smaller than {TAU_OVER_N_MIN} batches.")
    cfg['buffers']['replay_buffer']['mean_residence_steps'] = tau_over_n * every
    _size_replay(cfg, tau_over_n)

    if 'fill_process_var' in ov:
        cfg['z_calibration']['fill_process_var'] = float(ov['fill_process_var'])

    if 'replay_tb' in ov:
        # THE DOSE KNOB, and the only one that reaches w without touching fracs.
        # Memorisation goes as D = N * lr * w_eff / B with w_eff = fracs.replay *
        # replay_loss_coeffs.tb. `fracs.replay` is railed at [0.1, ...] by
        # gated_ramp and also gates deactivate_threshold, so it CANNOT carry the
        # sweep; the coefficient multiplies the TB loss directly
        # (gflownet_losses.py: `losses.append(tb_loss * loss_coeffs.tb)`) and the
        # branch stays fully live. Both readouts are invariant to it:
        # val_gap_nats is built from raw residuals in nats, and lambda_tau is a
        # RATIO of ema to birth loss, so a common factor cancels.
        assert 'replay_loss_coeffs' in cfg, 'no replay_loss_coeffs to scale'
        cfg['replay_loss_coeffs']['tb'] = float(ov['replay_tb'])

    for st in _fused(cfg):
        bal = st['balance']
        if 'ratchet_tol' in ov:
            bal['ratchet_tol'] = float(ov['ratchet_tol'])
        if 'bwd_hi' in ov:
            # a fresh dict: rr07's BOUNDS_* constants are shared by reference
            # across arms, so mutating in place would edit every later arm too
            lo_b, _ = bal['bounds']['bwd']
            lo_r, _ = bal['bounds']['replay']
            hi = float(ov['bwd_hi'])
            bal['bounds'] = {'bwd': [lo_b, hi], 'replay': [lo_r, 1.0 - lo_b]}
        if 'gain_mult' in ov:
            k = float(ov['gain_mult'])
            bal['up'], bal['down'] = bal['up'] * k, bal['down'] * k
        if 'val_gap_max' in ov:
            st['fwd_rollout_triggers']['val_gap_max'] = float(ov['val_gap_max'])
        if ov.get('freeze_pb'):
            # AFTER the bootstrap: on_enter runs in order and the freeze must
            # not be holding a snapshot while log Z is still being seeded.
            st['on_enter'] = list(st['on_enter']) + ['freeze_pb']

    if ov.get('freeze_pb'):
        # compile_policy 'step' installs compiled callables as instance
        # attributes and freeze_backward_policy deepcopies the trunk; gfn.py
        # raises on the pair. Pin the older mode rather than inherit.
        cfg['compile_policy'] = 'auto'
    return cfg


def check(cfg, name, fam, every, ov):
    tau_over_n = ov.get('tau_over_n', TAU_OVER_N_SHIP)
    where = name + ': '
    if ov.get('tau_over_n', TAU_OVER_N_SHIP) != TAU_OVER_N_SHIP:
        assert ov.get('val_gap_max') == 0.0, where + (
            'a tau arm MUST be deterministic -- a firing trigger raises '
            'admissions, which moves the occupancy the arm exists to set')
        rb = cfg['buffers']['replay_buffer']
        grown = max(int(rb['churn_rate']), int(cfg.get('max_batch_size') or 0))
        assert rb['max_size'] >= grown * tau_over_n * 1.2, where + (
            'max_size %d leaves no headroom over grown occupancy %d -- it would '
            'run cap-bound' % (rb['max_size'], grown * tau_over_n))
    if 'replay_tb' in ov:
        assert cfg['replay_loss_coeffs']['tb'] == float(ov['replay_tb']), where + 'replay_tb'
        assert ov.get('val_gap_max') == 0.0, where + (
            'a dose arm MUST be deterministic -- a firing trigger changes '
            'admissions and so changes the very dose being matched')

    st = _fused(cfg)
    active = _fused(cfg, active_only=True)
    assert st and active, where + 'no fused stage on the active protocol'
    assert all(s['fwd_rollout_every'] == every for s in st), where + 'cadence'
    assert all(s['flags'].get('z_calibration') is False for s in st), where + 'z servo on'
    assert all(s['fracs'] == {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5} for s in st), where + 'entry fracs'
    assert not any('min_fracs' in s for s in st), where + 'min_fracs left'

    # THE BOOTSTRAP, asserted on the stage that will actually execute and by
    # EXACT STRING. `startswith('bootstrap_z')` is what let the no-op ship.
    for s in active:
        oe = [str(a) for a in (s.get('on_enter') or [])]
        boots = [a for a in oe if a.startswith('bootstrap_z')]
        assert boots == [ROLLOUT_BOOTSTRAP], (
            where + 'active-protocol fused stage must carry exactly '
            '[%s], got %r' % (ROLLOUT_BOOTSTRAP, boots))

    # the shipped controller contract -- unchanged by any override except the
    # one the arm declares
    for s in st:
        b = s['balance']
        assert b['kind'] == 'gated_ramp' and b['guard'] == 'bwd' and b['ramp'] == 'replay', where + 'controller'
        assert b['bar'] == 0.0, where + 'bar is not varied in this battery'
        assert b['metric'] == GUARD_METRIC, where + 'guard channel'
        assert b['ratchet_metric'] == GUARD_LEVEL, where + "ratchet must be the guard's level"
        assert b['ratchet_tol'] == float(ov.get('ratchet_tol', 0.0)), where + 'ratchet_tol'
        assert b['bounds']['bwd'][1] == float(ov.get('bwd_hi', 0.9)), where + 'bwd upper rail'
        k = float(ov.get('gain_mult', 1.0))
        assert abs(b['up'] - 0.000425 * k) < 1e-12 and abs(b['down'] - 0.01075 * k) < 1e-12, where + 'gains'
        assert b['up'] < b['down'], where + 'guard must act faster than the ramp'
        trig = s.get('fwd_rollout_triggers') or {}
        assert all(t in trig for t in TRIGGER_KEYS), where + 'cadence triggers'
        assert trig['val_gap_max'] == float(ov.get('val_gap_max', 4.0)), where + 'val_gap_max'
        assert 'ess_min' not in trig, where + 'ess_min is retired'
        # this constant documents the contract; rr07's deltas() OWNS it, so a
        # drift between the two would otherwise be silent
        assert trig['occupancy_min_batches'] == OCCUPANCY_BAR, where + 'occupancy bar'
        assert (cfg['buffers']['replay_buffer']['mean_residence_steps']
                >= TAU_OVER_N_MIN * s['fwd_rollout_every']), (
            where + 'tau/N below the occupancy floor')
        assert ('freeze_pb' in s['on_enter']) == bool(ov.get('freeze_pb')), where + 'freeze_pb'
        if ov.get('freeze_pb'):
            assert s['on_enter'].index(ROLLOUT_BOOTSTRAP) < s['on_enter'].index('freeze_pb'), \
                where + 'freeze_pb must run after the Z bootstrap'
    if ov.get('freeze_pb'):
        assert cfg.get('compile_policy') == 'auto', where + "compile_policy 'step' is incompatible with the freeze"

    # buffer composition, and the cap that has to track the batch
    rb = cfg['buffers']['replay_buffer']
    b = int(cfg['batch_size'])
    assert b == int(ov.get('batch', b)), where + 'batch'
    assert rb['churn_rate'] == b and rb['val_cap'] == b, where + 'churn_rate/val_cap follow batch_size'
    assert int(cfg['max_batch_size']) >= b, where + 'max_batch_size below batch_size'
    assert rb['val_frac'] == 0.1, where + 'val split'
    assert rb['mean_residence_steps'] == tau_over_n * every, where + 'residence'
    occupancy = rb['churn_rate'] * tau_over_n
    assert rb['max_size'] >= occupancy, (
        where + 'replay max_size %d binds below the hazard occupancy %d -- eviction '
        'would stop being hazard-driven and this arm would measure something else'
        % (rb['max_size'], occupancy))

    # Z fill: the absorber and the eval feed are contract; only q may move
    zc = cfg['z_calibration']
    assert zc['fill_mode'] == 'absorb' and zc['fill_from_eval'] == 'fill', where + 'absorber'
    assert zc['fill_threshold'] > 0, where + 'fill off'
    assert zc['fill_process_var'] == float(ov.get('fill_process_var', 0.01)), where + 'fill_process_var'

    # warm start, prior, and the step budget
    assert cfg['buffers']['prior_buffer'].get('source') == 'anchors', where + 'prior source'
    assert not any('snapshot_prior' in (s.get('on_exit') or [])
                   for p in cfg['protocols'].values() for s in p['stages']), where + 'snapshot_prior left'
    assert cfg['checkpoint_name'] == p02make.PLACEHOLDER, where + 'checkpoint placeholder'
    assert cfg['prior_model_name'] == p02make.PRIOR_PLACEHOLDER, where + 'prior placeholder'
    # `epochs` is an ABSOLUTE step bound and every arm resumes part-way through,
    # so it must sit far past anything a 12 h wall can reach: the wall stops the
    # run, not the counter.
    exit_step = p02make.FAM[fam]['exit']
    assert int(cfg['epochs']) - exit_step >= 30000, (
        where + 'epochs %d leaves only %d steps past the phase-1 exit'
        % (cfg['epochs'], int(cfg['epochs']) - exit_step))
    assert cfg['tag'] == TAG and cfg['run_name'] == name, where + 'identity'
    assert not any('fwd_rollout_drift_max' in s for s in st), where + 'retired drift key'
    p02make._scan_local_paths(cfg, name)


def build():
    out = {}
    for name, fam, every, ov in ARMS:
        base = ROOT / 'prod_sep02' / (rr07make.CENTRE[fam] + '.yaml')
        cfg = yaml.safe_load(base.read_text(encoding='utf-8'))
        cfg = rr07make.deltas(cfg, name, every, fam)   # the shipped v2 contract
        cfg['tag'] = TAG
        _force_rollout_bootstrap(cfg)
        cfg = _apply(cfg, every, ov)
        check(cfg, name, fam, every, ov)
        assert name not in out, 'duplicate arm ' + name
        out[name] = (cfg, fam, every, ov)
    return out


def emit(arms):
    rows = []
    for name, (cfg, fam, every, ov) in arms.items():
        if name in RETIRED:
            # position preserved, config removed -- see RETIRED above
            (HERE / (name + '.yaml')).unlink(missing_ok=True)
            rows.append(('RETIRED_' + name, rr07make.SRC[fam], every,
                         cfg['buffers']['replay_buffer']['mean_residence_steps'] // every,
                         cfg['batch_size'], 'retired 2026-09-08'))
            continue
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((name, rr07make.SRC[fam], every,
                     cfg['buffers']['replay_buffer']['mean_residence_steps'] // every,
                     cfg['batch_size'], ','.join('%s=%s' % kv for kv in sorted(ov.items())) or '-'))
    with (HERE / 'INDEX.tsv').open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tfwd_rollout_every\ttau_over_n\tbatch\toverride\n')
        for r in rows:
            f.write('%s\t%s\t%d\t%d\t%d\t%s\n' % r)
    for r in rows:
        if r[0].startswith('RETIRED_'):
            assert not (HERE / (r[0][len('RETIRED_'):] + '.yaml')).exists(),                 'retired arm still has a config: ' + r[0]
            continue
        assert (HERE / (r[0] + '.yaml')).exists(), 'index names a missing config: ' + r[0]
    assert len(rows) == len(ARMS), 'INDEX row count changed -- queued tasks would RE-MAP'

    text = p02make._HEAD.format(
        wall=WALL, last=len(rows) - 1, jobname='rr08', index='INDEX.tsv',
        label='18-arm overnight calibration battery, single leg',
        placeholder=p02make.PLACEHOLDER, prior_placeholder=p02make.PRIOR_PLACEHOLDER,
        sentinel='', epilogue='')
    text = (text.replace('configs/prod_sep02', 'configs/rr_sep08')
                .replace('# prod_sep02', '# rr_sep08')
                # the b6400 arm carries a 40000-row replay buffer; 48G is the
                # prod_sep02 ask and the profiler already OOM'd the host at it
                .replace('#SBATCH --mem=48G', '#SBATCH --mem=96G'))
    with (HERE / 'submit_rr_sep08.sbatch').open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)

    print('rr_sep08: %d arms -> submit_rr_sep08.sbatch  (%s wall, single leg)' % (len(rows), WALL))
    print('  %-13s %-22s %5s %6s %7s  %s' % ('arm', 'warm_src', 'N', 'tau/N', 'batch', 'override'))
    for r in rows:
        print('  %-13s %-22s %5d %6d %7d  %s' % r)


def main():
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    emit(build())


if __name__ == '__main__':
    main()
