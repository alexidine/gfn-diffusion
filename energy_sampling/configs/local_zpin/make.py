"""local_zpin: does pinning log Z more often help, holding the buffer fixed?

  zpin_ctrl   fwd_rollout_every 20, z_pin_rollout_every 0   -- Z pinned every 20
  zpin_z10    fwd_rollout_every 20, z_pin_rollout_every 10  -- Z pinned every 10

THE ONE DIFFERENCE IS THE PIN CADENCE. `fwd_ran` gates the rollout, the z_fill
stash and replay admission together, so fanning fwd_rollout_every moves pin
frequency, fresh-data rate and buffer reuse at once and cannot attribute a
result to any of them. z_pin_rollout_every runs an extra rollout on the skipped
steps whose ONLY product is the z_fill stash: no gradient weight, and its rows
never reach manage_replay_buffer. So zpin_z10 pins log Z twice as often as
zpin_ctrl while both arms admit exactly the same rows at exactly the same rate,
into a buffer with exactly the same occupancy and mean age.

WHAT IT COSTS, AND WHY THAT IS THE POINT. zpin_z10 makes 2x the energy calls --
a z-pin is a real MLIP call. Read it as `rollout/n_z_pin` rather than trusting
the configured cadence. If the extra pins buy nothing, the cheaper arm wins on
cost alone; if they buy something, this is the first measurement that says so
without the buffer moving underneath it.

OCCUPANCY IS HELD BY tau_mult, NOT BY tau. O = B * tau/N, so tau must move WITH N
to hold occupancy; at N=20 and tau_mult 4 both arms sit at tau=80, occupancy
1600 against a 12000 cap -- cap-free, which is the state the first hc2 re-run
lost when a fixed cap pinned occupancy and killed the knob it was fanning.
"""
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent / 'local_prod_sep02' / 'lp02.yaml'

STEPS = 4000        # 200 ordinary rollouts; zpin_z10 adds 200 more pins
EVERY = 20          # fwd_rollout_every, both arms
TAU_MULT = 4        # tau = TAU_MULT * EVERY = 80, per the owner's "tau maybe 4N"

ARMS = {'zpin_ctrl': 0,      # z_pin_rollout_every
        'zpin_z10': 10}


def build(name, z_pin):
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    cfg['run_name'] = name
    cfg['tag'] = 'zpin'
    cfg['epochs'] = STEPS

    rb = cfg['buffers']['replay_buffer']
    # STORE-ALL: every rollout row is admitted, so admission rate is exactly the
    # ordinary cadence and nothing about intake differs between the arms.
    rb['churn_rate'] = int(cfg['batch_size'])
    rb['mean_residence_steps'] = TAU_MULT * EVERY
    # measured, not actuated on -- replay/val_gap is the overfitting readout
    rb['val_frac'] = 0.1

    for stage in [s for p in cfg['protocols'].values() for s in p['stages']
                  if s.get('train_mode') == 'fused']:
        stage['fwd_rollout_every'] = EVERY
        if z_pin:
            stage['z_pin_rollout_every'] = z_pin
        # the servo does its OWN rollout + energy call per Z step off a sensor
        # frozen between rollouts, and protocol.py refuses the pairing outright.
        # It also feeds replay intake, which would be a third thing on the
        # admission stream this battery exists to hold still.
        stage.setdefault('flags', {})['z_calibration'] = False

    check(cfg, name, z_pin)
    return cfg


def check(cfg, name, z_pin):
    fused = [s for p in cfg['protocols'].values() for s in p['stages']
             if s.get('train_mode') == 'fused']
    assert fused, name
    for s in fused:
        assert s['fwd_rollout_every'] == EVERY, name
        assert s['flags']['z_calibration'] is False, name + ': servo on'
        assert s.get('z_pin_rollout_every', 0) == z_pin, name
    rb = cfg['buffers']['replay_buffer']
    assert rb['mean_residence_steps'] == TAU_MULT * EVERY, name
    assert rb['churn_rate'] == cfg['batch_size'], name
    # A CAP-BOUND BUFFER IS tau-DISCONNECTED and measures nothing: occupancy
    # pins at the cap and both arms come out identical on every buffer metric
    # regardless of what else changed.
    occ = cfg['batch_size'] * TAU_MULT
    assert rb['max_size'] >= occ * 1.5, (
        f'{name}: max_size {rb["max_size"]} leaves no headroom over occupancy {occ}')


def main():
    for name, z_pin in ARMS.items():
        cfg = build(name, z_pin)
        (HERE / f'{name}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
            encoding='utf-8')
        rb = cfg['buffers']['replay_buffer']
        pins = STEPS // EVERY + (STEPS // z_pin - STEPS // EVERY if z_pin else 0)
        print(f"{name:<11} every={EVERY} z_pin={z_pin or '-':<3} "
              f"tau={rb['mean_residence_steps']} occ={cfg['batch_size'] * TAU_MULT} "
              f"steps={STEPS}  Z pins={pins}  energy calls={pins}")
    print(f'\nboth arms: same admission rate, same occupancy, same mean age.')
    print(f'read the cost as rollout/n_z_pin, the effect as z_fill/gap and log Z.')


if __name__ == '__main__':
    main()
