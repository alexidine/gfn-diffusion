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


def deltas(cfg, name, every):
    cfg['run_name'] = name
    cfg['tag'] = 'rr07'
    zc = cfg.setdefault('z_calibration', {})
    zc['fill_threshold'] = 0.5
    zc['fill_se'] = 3.0
    zc['fill_cooldown_steps'] = 0
    n = 0
    for proto in (cfg.get('protocols') or {}).values():
        for st in (proto.get('stages') or []):
            if st.get('train_mode') != 'fused':
                continue
            st['fwd_rollout_every'] = int(every)
            st.setdefault('flags', {})['z_calibration'] = False
            n += 1
    assert n >= 1, name + ': no fused stage'
    rb = cfg['buffers']['replay_buffer']
    rb['churn_rate'] = int(cfg['batch_size'])
    rb['mean_residence_steps'] = 5 * int(every)
    return cfg


def check(cfg, name, every):
    st = [s for p in cfg['protocols'].values() for s in p['stages'] if s.get('train_mode') == 'fused']
    assert st and all(s['fwd_rollout_every'] == every for s in st), name
    assert all(s['flags'].get('z_calibration') is False for s in st), name + ': servo on'
    assert cfg['z_calibration']['fill_threshold'] > 0, name + ': fill off'
    rb = cfg['buffers']['replay_buffer']
    assert rb['churn_rate'] == cfg['batch_size'] and rb['mean_residence_steps'] == 5 * every, name
    assert cfg['checkpoint_name'] == p02make.PLACEHOLDER, name
    assert cfg['prior_model_name'] == p02make.PRIOR_PLACEHOLDER, name
    p02make._scan_local_paths(cfg, name)


def build(arms, prefix):
    out = {}
    for fam, every in arms:
        base = P02 / (CENTRE[fam] + '.yaml')
        name = '%s_%s_n%d' % (prefix, fam, every)
        cfg = yaml.safe_load(base.read_text(encoding='utf-8'))
        cfg = deltas(cfg, name, every)
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
