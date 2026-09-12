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

STEPS = 2000        # 100 ordinary rollouts; zpin_z10 adds 100 more pins
EVERY = 20          # fwd_rollout_every, both arms
TAU_MULT = 4        # tau = TAU_MULT * EVERY = 80, per the owner's "tau maybe 4N"
BATCH = 1000        # matches rr07_rr_hc2_n20; see SEED below for why it matters

#: THE SEED, and it is not interchangeable with lp02's. The eight e01bd1 phase-1
#: exits differ in quality: screened at 2000 forward samples, six sit at
#: frac(E>0) 0.42-0.46 while dev_race_L2_transition sits at 0.054 with mean
#: energy -97.5 against a prior buffer at -126.5 (configs/local_rr_sep07). lp02
#: ships dev_elj_p2_cruise, whose exit carries log Z ~ -0.1; this one carries
#: ~16.6. Starting cold means 2000 steps only ever measures the absorber
#: ACQUIRING the level, never TRACKING it -- which is the behaviour under test.
#: Both files must move together: the policy and the prior model it samples.
SEED = 'dev_race_L2_transition_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1'

ARMS = {'zpin_ctrl': 0,      # z_pin_rollout_every
        'zpin_z10': 10}


def build(name, z_pin):
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    cfg['run_name'] = name
    cfg['tag'] = 'zpin'
    cfg['epochs'] = STEPS
    cfg['checkpoint_name'] = f'{SEED}_phase1_exit.pt'
    cfg['prior_model_name'] = f'{SEED}_prior.pt'
    # BATCH SETS THE ABSORBER'S GAIN. K = P/(P + se^2) and se ~ 1/sqrt(B), so a
    # 400-row measurement is weighted down against a 1000-row one and the filter
    # tracks differently. Matching the baseline's batch is what makes this a fork
    # of it rather than a different experiment.
    cfg['batch_size'] = BATCH
    cfg['max_batch_size'] = BATCH

    # THE FILL MUST BE THE ABSORBER, AND IT MUST BE ALLOWED TO RUN.
    # lp02 inherits mk_dev's z_calibration block, which is the SNAP: fill_mode
    # absent (code default 'snap'), a 20-nat threshold, and a 200-step cooldown.
    # Against a 10-step pin cadence that cooldown alone blocks 19 of every 20
    # pins, so the two arms would collapse into each other whatever the bar did.
    # These are rr08_ctrl_tol10's settings -- the mechanism actually in use.
    #
    # absorb takes NO thresholds (train.py: "the snap gates existed to keep noise
    # out, and this weights noise down instead of discarding it"); fill_threshold
    # survives only as an on/off switch, since <= 0 returns before the mode split.
    zc = cfg['z_calibration']
    zc['fill_mode'] = 'absorb'
    zc['fill_threshold'] = 0.5      # > 0 = ON. Not a bar in absorb mode.
    zc['fill_se'] = 3
    zc['fill_cooldown_steps'] = 0   # the pin cadence IS the cadence under test
    zc['fill_process_var'] = 0.01
    zc['fill_moment_reset'] = 0.5
    # 'report', NOT rr08's 'fill': an eval-sourced fill lands on the eval cadence
    # in BOTH arms, which is a fill stream the pin cadence does not control and
    # would dilute the only difference between them. 'report' keeps the
    # fixed-cadence MEASUREMENT (z_fill/eval_gap) and drops the action.
    zc['fill_from_eval'] = 'report'

    rb = cfg['buffers']['replay_buffer']
    # STORE-ALL: every rollout row is admitted, so admission rate is exactly the
    # ordinary cadence and nothing about intake differs between the arms.
    rb['churn_rate'] = int(cfg['batch_size'])
    rb['val_cap'] = int(cfg['batch_size'])
    rb['mean_residence_steps'] = TAU_MULT * EVERY
    # measured, not actuated on -- replay/val_gap is the overfitting readout
    rb['val_frac'] = 0.1

    live = cfg['protocols'][cfg['protocol']]
    for stage in [s for s in live['stages'] if s.get('train_mode') == 'fused']:
        # fracs ARE LOSS WEIGHTS, and fracs.fwd 0 is how the forward TB gradient
        # was turned off in favour of the absorber (rr_sep08, 2026-09-08). lp02
        # predates that and still ships 0.05, which trains log Z by the OLD
        # mechanism and would compete with the absorber for the level. With 0 and
        # deactivate_threshold 0.01 the rollout still fires -- it is still the
        # z_fill measurement and the replay intake -- but carries no gradient.
        stage['fracs'] = {'fwd': 0.0, 'bwd': 0.5, 'replay': 0.5}
        # balance.pinned.fwd IS fracs.fwd -- the same quantity written twice, and
        # config_invariants refuses the pair when they disagree. Zeroing one and
        # not the other kills every arm at load, in a sibling protocol this
        # battery never runs.
        bal = stage.get('balance')
        if isinstance(bal, dict) and isinstance(bal.get('pinned'), dict)                 and 'fwd' in bal['pinned']:
            bal['pinned']['fwd'] = 0.0
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
    live = cfg['protocols'][cfg['protocol']]
    fused = [s for s in live['stages'] if s.get('train_mode') == 'fused']
    assert fused, name
    for s in fused:
        assert s['fwd_rollout_every'] == EVERY, name
        assert s['flags']['z_calibration'] is False, name + ': servo on'
        assert s.get('z_pin_rollout_every', 0) == z_pin, name
    for s_ in fused:
        _b = s_.get('balance')
        if isinstance(_b, dict) and isinstance(_b.get('pinned'), dict):
            assert float(_b['pinned'].get('fwd', 0.0)) == 0.0, (
                name + ': balance.pinned.fwd and fracs.fwd are the same quantity')
        assert float(s_['fracs']['fwd']) == 0.0, (
            name + ': a nonzero fwd frac trains log Z by TB gradient, which is '
                   'the mechanism the absorber replaced -- the two would compete')
    assert cfg['checkpoint_name'] == f'{SEED}_phase1_exit.pt', name
    assert cfg['prior_model_name'] == f'{SEED}_prior.pt', (
        name + ': the policy and its prior model must come from the SAME exit')
    assert cfg['load_weights_only'] is True, name
    assert cfg['batch_size'] == BATCH == cfg['max_batch_size'], name
    zc = cfg['z_calibration']
    assert zc['fill_mode'] == 'absorb', name + ': snap is not the mechanism in use'
    assert zc['fill_cooldown_steps'] == 0, (
        name + ': a cooldown above the pin cadence blocks the pins under test')
    assert zc['fill_threshold'] > 0, name + ': <= 0 turns the fill off entirely'
    assert zc['fill_from_eval'] == 'report', (
        name + ": 'fill' adds an eval-cadence fill stream to both arms")
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
