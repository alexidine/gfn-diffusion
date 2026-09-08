"""rr_sep07 -- rarer rollouts v0 on the cluster (docs/design/rarer_rollouts.md).

Two sets, one generator, all SINGLE-LEG (a killed arm cannot resume until the
lj_coeff stamp fix lands, so nothing here chains):

  smoke   4 arms, one per system, N = 10, 20-minute wall. Pass = every arm reaches
          wandb and logs z_fill/gap on the cadence and nowhere else.
  prod    4 paper systems x N in {5, 20} at the p02 centre rates, 2-day wall.
          Compare to the running p02 arms at matched step.

EVERY ARM IS A prod_sep02 ARM WITH THE v0 DELTAS APPLIED. Those YAMLs are running
on this cluster right now, so paths, hashes and MLIP settings are already proven;
anything that breaks is attributable to the deltas:

  stage.fwd_rollout_every = N        forward rollout + energy call on 1 in N steps
  stage.flags.z_calibration = false  the servo would call the MLIP on every skipped
                                     step (refused at load otherwise)
  z_calibration.fill_threshold 0.5,  z_level_fill pins log Z at each rollout; the
      fill_se 3, fill_cooldown 0     stash is single-use so it cannot fire twice
  replay_buffer.churn_rate = batch   store the whole forward batch
  replay_buffer.mean_residence_steps = 5N   (steps, after the clock fix; each fill
                                            replaces <= 20% of the buffer)

The sbatch is prod_sep02's long template (single leg, no .dead sentinel, sidecar
kept, prior model resolved by glob -> null on a first launch).
"""
import importlib.util
import pathlib
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
P02 = ROOT / 'prod_sep02'

_spec = importlib.util.spec_from_file_location('p02make', P02 / 'make.py')
p02make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p02make)

CENTRE = {'mip': 'p02_mip_lr1', 'neh': 'p02_neh_lr0p5',
          'mipu': 'p02_mipu_lr0p0625', 'nehu': 'p02_nehu_lr0p125',
          'acr': 'p02_acr_lr0p025'}
SRC = {k: v['src'] for k, v in p02make.FAM.items()}

SMOKE = [('mip', 10), ('mipu', 10), ('nehu', 10), ('acr', 10)]
PROD = [(fam, n) for fam in ('mip', 'neh', 'mipu', 'nehu') for n in (5, 20)]

# Controller rails. The bwd FLOOR is the measured one: 0.5 held on the MLIP arms
# where lower starved coverage; ELJ tolerates 0.25 (replay cap 0.75).
MLIP = {'mipu', 'nehu', 'acr'}
BOUNDS_MLIP = {'bwd': [0.5, 0.9], 'replay': [0.1, 0.5]}
BOUNDS_ELJ = {'bwd': [0.25, 0.9], 'replay': [0.1, 0.75]}


def deltas(cfg, name, every, fam):
    cfg['run_name'] = name
    cfg['tag'] = 'rr07'
    zc = cfg.setdefault('z_calibration', {})
    zc['fill_threshold'] = 0.5
    zc['fill_se'] = 3.0
    zc['fill_cooldown_steps'] = 0
    # the absorber: every measurement moves Z by its precision share; the eval
    # rollout (2500-10000 samples) is fed to the same actuator
    zc['fill_mode'] = 'absorb'
    zc['fill_process_var'] = 0.01
    zc['fill_moment_reset'] = 0.5
    zc['fill_from_eval'] = 'fill'
    n = 0
    for proto in (cfg.get('protocols') or {}).values():
        for st in (proto.get('stages') or []):
            # ANCHOR-ONLY PRIOR: nothing writes a prior model any more
            if 'snapshot_prior' in (st.get('on_exit') or []):
                st['on_exit'] = [a for a in st['on_exit'] if a != 'snapshot_prior']
            if st.get('train_mode') != 'fused':
                continue
            st['fwd_rollout_every'] = int(every)
            st.setdefault('flags', {})['z_calibration'] = False
            st['fracs'] = {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
            st.pop('min_fracs', None)
            # THE REPLAY-SIDE BARS SET THE CADENCE; fwd_rollout_every above is the
            # backstop, and the only thing bounding how long log Z goes unpinned
            # (nothing measures the root between rollouts -- getting one IS the
            # rollout). On rr07_rr_n7_v2, fwd/tb_resid_clipped held within +-0.05
            # nats all run at N=7, so that cadence was far more often than the Z
            # pin needed.
            st['fwd_rollout_triggers'] = {
                # OVERFIT IS THE CONTROLLING BAR (owner 2026-09-08). It is the only
                # one of the three replay sensors that is both unconfounded and NOT
                # self-correcting: reuse is N identically, and nothing in the system
                # pushes back on memorisation the way the prioritised draw pushes
                # back on drift. Since reuse = N = 1/(energy calls per step), an
                # overfit-driven cadence is literally a quality-vs-compute dial.
                # The BAR ITSELF IS NOT CALIBRATED: 1.25 nats measured at reuse 20
                # is 8.6 sigma, but whether that level is harmful is unknown. 4.0 is
                # a guess with the resolution floor (~2 sigma) behind it.
                'val_gap_max': 4.0,
                # The draw needs something to draw from. NOTE this reading is
                # len(buffer)/batch_size, which in steady state IS tau/N -- it is the
                # configured ratio, not an independent measurement.
                # 2.0, one half-burst below the tau/N = 3 floor. Occupancy
                # sawtooths -- B admitted per burst, ~B drained between -- so the
                # TROUGH is what the bar sees: O_trough/B = tau/N - 0.5 = 2.5.
                # A bar AT the ratio would fire every cycle; at 2.0 there is 25%
                # headroom and it fires only on a genuine shortfall (initial fill,
                # a batch growth the buffer has not caught up with, a servo churn
                # boost). Owner rule 2026-09-08: never below 2 batches.
                'occupancy_min_batches': 2.0,
                # ess_min RETIRED. policy_drift_ess_frac is Kish on exp(d), so it is
                # scale-invariant and reads 1.000 when EVERY row's log p_F drops by
                # the same 5.7 nats -- it cannot see the policy walking away from the
                # buffer, only the spread. It was also the ONLY bar firing on the
                # rr08 arms (360 times, +36% rollouts, effective cadence 14.7 vs a
                # configured 20), so it was setting the cadence off a quantity we do
                # not want to steer on. drift_std_max stays off: same failure, and
                # policy_drift_nats is the natural reading, which is not wired as a
                # bar yet.
            }
            # GUARD: bar 0 on the Z-anchored channel, RATCHET on the same
            # quantity's level with tol 0. Both tolerances zero -- the guard's job
            # is to keep bwd coverage improving, so "did it get worse at all" is
            # the bar and "is it at its best" is the release (owner 2026-09-07).
            # Gains 4x slower than the original: v2 moved bwd 0.50 -> 0.79 in 2000
            # steps at 20x slower, so the loop closes well inside a cluster leg.
            st['balance'] = {
                'kind': 'gated_ramp', 'ramp': 'replay', 'guard': 'bwd',
                'pinned': {'fwd': 0.0},
                'metric': 'bwd/under_coverage_rise150', 'bar': 0.0,
                'ratchet_metric': 'bwd/under_coverage', 'ratchet_tol': 0.0,
                'up': 0.000425, 'down': 0.01075,
                'bounds': BOUNDS_MLIP if fam in MLIP else BOUNDS_ELJ,
            }
            # Z BOOTSTRAP AT PHASE-2 ENTRY. Phase 1 gives the flow scalar no
            # gradient -- the exit checkpoints carry log Z at exactly 0.0 -- so
            # without this the fill walks the level in over the first few hundred
            # steps and everything trained meanwhile is trained against a wrong
            # level. Measured on v2: 0.000 -> 16.557 in one shot, at se 0.73
            # against the 400-row entry fill's 2.60.
            on_enter = list(st.get('on_enter') or [])
            if not any(str(a).startswith('bootstrap_z') for a in on_enter):
                on_enter.append('bootstrap_z:rollout:4000')
            st['on_enter'] = on_enter
            n += 1
    assert n >= 1, name + ': no fused stage'
    cfg['buffers']['prior_buffer']['source'] = 'anchors'
    rb = cfg['buffers']['replay_buffer']
    rb['churn_rate'] = int(cfg['batch_size'])
    rb['mean_residence_steps'] = 5 * int(every)
    # hold 10% of every admission out of the training draw so the buffer's
    # generalisation gap (replay/val_gap) is measured rather than inferred.
    # Measurement only -- nothing actuates on it.
    rb['val_frac'] = 0.1
    # _replay_val_size returns min(val_cap, batch_size, n_val), so anything above
    # batch_size is inert. Set to batch_size as the honest ceiling.
    rb['val_cap'] = int(cfg['batch_size'])
    return cfg


def check(cfg, name, every):
    st = [s for p in cfg['protocols'].values() for s in p['stages'] if s.get('train_mode') == 'fused']
    assert st and all(s['fwd_rollout_every'] == every for s in st), name
    assert all(s['flags'].get('z_calibration') is False for s in st), name + ': servo on'
    assert cfg['z_calibration']['fill_threshold'] > 0, name + ': fill off'
    assert cfg['z_calibration']['fill_mode'] == 'absorb' and cfg['z_calibration']['fill_from_eval'] == 'fill', name + ': absorber'
    for s in st:
        b = s.get('balance') or {}
        assert b.get('kind') == 'gated_ramp' and b.get('guard') == 'bwd', name + ': controller'
        assert s['fracs'] == {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}, name + ': entry fracs'
        assert 'min_fracs' not in s, name
    assert cfg['buffers']['prior_buffer'].get('source') == 'anchors', name + ': prior source'
    assert not any('snapshot_prior' in (s.get('on_exit') or [])
                   for p in cfg['protocols'].values() for s in p['stages']), name + ': snapshot_prior left'
    rb = cfg['buffers']['replay_buffer']
    assert rb['churn_rate'] == cfg['batch_size'] and rb['mean_residence_steps'] == 5 * every, name
    assert rb['val_frac'] == 0.1, name + ': val split'
    assert rb['val_cap'] == cfg['batch_size'], name + ': val cap above batch_size is inert'
    assert not any('fwd_rollout_drift_max' in s for s in st), name + ': retired drift key'
    for s in st:
        b = s['balance']
        assert b['bar'] == 0.0 and b['ratchet_tol'] == 0.0, name + ': tolerances'
        assert b['metric'] == 'bwd/under_coverage_rise150', name + ': guard channel'
        assert b['ratchet_metric'] == 'bwd/under_coverage', name + ": ratchet must be the guard's LEVEL"
        assert s['fwd_rollout_triggers'], name + ': no cadence triggers'
        assert any(str(a).startswith('bootstrap_z') for a in (s.get('on_enter') or [])),             name + ': no phase-2 Z bootstrap'
    assert cfg['checkpoint_name'] == p02make.PLACEHOLDER, name
    assert cfg['prior_model_name'] == p02make.PRIOR_PLACEHOLDER, name
    p02make._scan_local_paths(cfg, name)


def build(arms, prefix):
    out = {}
    for fam, every in arms:
        base = P02 / (CENTRE[fam] + '.yaml')
        name = '%s_%s_n%d' % (prefix, fam, every)
        cfg = yaml.safe_load(base.read_text(encoding='utf-8'))
        cfg = deltas(cfg, name, every, fam)
        check(cfg, name, every)
        out[name] = (cfg, fam, every)
    return out


def emit(arms, index, sbatch, jobname, label, walltime):
    rows = []
    for name, (cfg, fam, every) in arms.items():
        with (HERE / (name + '.yaml')).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        rows.append((name, SRC[fam], every))
    with (HERE / index).open('w', encoding='utf-8', newline='\n') as f:
        f.write('arm\twarm_src\tfwd_rollout_every\n')
        for r in rows:
            f.write('%s\t%s\t%d\n' % r)
    for n, _s, _e in rows:
        assert (HERE / (n + '.yaml')).exists(), 'index names a missing config: ' + n
    text = p02make._HEAD.format(
        wall=walltime, last=len(rows) - 1, jobname=jobname, index=index, label=label,
        placeholder=p02make.PLACEHOLDER, prior_placeholder=p02make.PRIOR_PLACEHOLDER,
        sentinel='', epilogue='')
    # the template is prod_sep02's; retarget its directory and log path
    text = text.replace('configs/prod_sep02', 'configs/rr_sep07').replace('# prod_sep02', '# rr_sep07')
    with (HERE / sbatch).open('w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    print('%-6s %2d arms -> %s  (%s)' % (label.split()[0], len(rows), sbatch, walltime))
    for n, _s, e in rows:
        print('        %-20s every=%d' % (n, e))


def main():
    logs = HERE / 'joblogs'
    logs.mkdir(exist_ok=True)
    (logs / '.gitkeep').write_text(
        'ships this directory to the cluster; SLURM cannot create --output\n', encoding='utf-8')
    emit(build(SMOKE, 'rrs'), 'INDEX_smoke.tsv', 'submit_rr_sep07_smoke.sbatch',
         'rr07smk', 'smoke 20-minute single-leg arms', '00:20:00')
    emit(build(PROD, 'rrp'), 'INDEX_prod.tsv', 'submit_rr_sep07_prod.sbatch',
         'rr07prod', 'prod 2-day single-leg arms', '2-00:00:00')


if __name__ == '__main__':
    main()
