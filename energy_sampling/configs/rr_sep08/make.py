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
TAU_OVER_N_SHIP = 5
ROLLOUT_BOOTSTRAP = 'bootstrap_z:rollout:4000'
GUARD_METRIC = 'bwd/under_coverage_rise150'
GUARD_LEVEL = 'bwd/under_coverage'
TRIGGER_KEYS = ('ess_min', 'val_gap_max', 'occupancy_min_batches')

# (arm, family, N, overrides). EXACTLY ONE override per arm, except where the
# override is only meaningful in combination (n50_ess25: the ess bar is inert at
# N=20 because nothing drifts far enough to fire it, so the bar has to be varied
# at the cadence where it is live; pbf_mipu: the freeze needs a batch that
# survives the night, and block 3 predicts 3200 is it).
ARMS = (
    # -- block 1: controller gains and rails ------------------------------------
    ('base',        'mip',  N_SHIP, {}),
    ('tol05',       'mip',  N_SHIP, {'ratchet_tol': 0.5}),
    ('cap75',       'mip',  N_SHIP, {'bwd_hi': 0.75}),
    ('fast4',       'mip',  N_SHIP, {'gain_mult': 4.0}),
    # -- block 2: cadence and buffer composition --------------------------------
    ('n5',          'mip',       5, {}),
    ('n50',         'mip',      50, {}),
    ('n50_ess25',   'mip',      50, {'ess_min': 0.25}),
    ('tau2',        'mip',  N_SHIP, {'tau_over_n': 2}),
    ('tau10',       'mip',  N_SHIP, {'tau_over_n': 10}),
    ('q05',         'mip',  N_SHIP, {'fill_process_var': 0.05}),
    # -- block 3: MLIP cost ladder ----------------------------------------------
    ('mipu_b1600',  'mipu', N_SHIP, {}),
    ('mipu_b3200',  'mipu', N_SHIP, {'batch': 3200}),
    ('mipu_b6400',  'mipu', N_SHIP, {'batch': 6400}),
    ('nehu_b1600',  'nehu', N_SHIP, {}),
    ('acr_b1000',   'acr',  N_SHIP, {}),
    ('acr_b2000',   'acr',  N_SHIP, {'batch': 2000}),
    # -- block 4: P_B frozen at T=100 -------------------------------------------
    ('pbf_mip',     'mip',  N_SHIP, {'freeze_pb': True}),
    ('pbf_mipu',    'mipu', N_SHIP, {'batch': 3200, 'freeze_pb': True}),
)

OVERRIDE_KEYS = {'ratchet_tol', 'bwd_hi', 'gain_mult', 'tau_over_n',
                 'ess_min', 'fill_process_var', 'batch', 'freeze_pb'}


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
    """max_size from the occupancy the hazard actually produces, so eviction
    stays hazard-driven rather than cap-driven at every batch on the ladder.
    1.25x headroom for the Poisson spread around the mean residence; never
    below rr_sep07's 12000, because a cap that never binds costs nothing."""
    rb = cfg['buffers']['replay_buffer']
    occupancy = int(rb['churn_rate']) * tau_over_n
    rb['max_size'] = max(12000, int(math.ceil(occupancy * 1.25)))


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

    cfg['buffers']['replay_buffer']['mean_residence_steps'] = tau_over_n * every
    _size_replay(cfg, tau_over_n)

    if 'fill_process_var' in ov:
        cfg['z_calibration']['fill_process_var'] = float(ov['fill_process_var'])

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
        if 'ess_min' in ov:
            st['fwd_rollout_triggers']['ess_min'] = float(ov['ess_min'])
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
        assert trig['ess_min'] == float(ov.get('ess_min', 0.10)), where + 'ess_min'
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
        assert (HERE / (r[0] + '.yaml')).exists(), 'index names a missing config: ' + r[0]

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
