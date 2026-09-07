"""Local acceptance rig for rarer rollouts v0 (docs/design/rarer_rollouts.md).

Two arms off the validated local ELJ shape (configs/local_prod_sep02/lp02.yaml):

  rr_n1    fwd_rollout_every: 1 with lp02's own buffer settings. Exercises the new
           gate on the path that must reproduce today's behaviour; compare to the
           localprod_lp02 run at matched step.
  rr_n10   fwd_rollout_every: 10, store-all replay (churn_rate = batch), residence
           5N steps. The design under test.

Pass conditions are the spec's local acceptance list: z_fill/gap logged exactly on
the cadence and nowhere else, lr_ctrl/scale flat, anchors static, replay occupancy
~ B*tau/N, Bwd Frac 0.5 between rollouts and 0.475 on them, zero divergences, and
energies / log Z tracking the N=1 arm within noise at matched step.
"""
import pathlib
import yaml

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent / 'local_prod_sep02' / 'lp02.yaml'

STEPS = 2000


def build(name, every, store_all):
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    cfg['run_name'] = name
    cfg['tag'] = 'rr07'
    cfg['epochs'] = STEPS

    zc = cfg.setdefault('z_calibration', {})
    # the fill is the ONLY Z pin once the servo is off; >0 keeps it armed, small
    # makes it fire at every rollout, se-gated so a batch must resolve the gap
    zc['fill_threshold'] = 0.5
    zc['fill_se'] = 3.0
    zc['fill_cooldown_steps'] = 0

    n_fused = 0
    for proto in (cfg.get('protocols') or {}).values():
        for st in (proto.get('stages') or []):
            if st.get('train_mode') != 'fused':
                continue
            st['fwd_rollout_every'] = int(every)
            st.setdefault('flags', {})['z_calibration'] = False
            n_fused += 1
    assert n_fused >= 1, f'{name}: no fused stage to cadence'

    if store_all:
        rb = cfg['buffers']['replay_buffer']
        rb['churn_rate'] = int(cfg['batch_size'])
        rb['mean_residence_steps'] = 5 * int(every)
    return cfg


def main():
    arms = {'rr_n1': build('rr_n1', 1, store_all=False),
            'rr_n10': build('rr_n10', 10, store_all=True)}
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        st = [s for p in cfg['protocols'].values() for s in p['stages'] if s.get('train_mode') == 'fused'][0]
        rb = cfg['buffers']['replay_buffer']
        print(f"{name:<8} every={st['fwd_rollout_every']} z_cal={st['flags']['z_calibration']} "
              f"fill_thr={cfg['z_calibration']['fill_threshold']} churn={rb['churn_rate']} "
              f"tau={rb['mean_residence_steps']} batch={cfg['batch_size']}")


if __name__ == '__main__':
    main()
