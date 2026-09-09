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
# ELJ rails: bwd floor 0.25, replay cap 0.75. (MLIP arms use a 0.5 bwd floor.)
BOUNDS = {'bwd': [0.25, 0.9], 'replay': [0.1, 0.75]}


def build(name, every, store_all, fwd_frac=0.0, boot=0, warm=None, steps=None,
          batch=None, lr_scale=None, tau_mult=5, val_gap_max=None):
    """fwd_frac > 0 is the level-blind forward policy step: on rollout steps the
    forward TB loss trains the policy with the batch's own root standing in for
    log Z (tb_z_source batch_root), at that loss weight; the head is still set
    only by the fill."""
    cfg = yaml.safe_load(BASE.read_text(encoding='utf-8'))
    cfg['run_name'] = name
    cfg['tag'] = 'rr07'
    cfg['epochs'] = int(steps or STEPS)
    if batch:
        # SET BEFORE val_cap AND churn_rate, both of which derive from batch_size
        # below. grow_batch_size is on, so max must move with it or the cap binds.
        cfg['batch_size'] = int(batch)
        cfg['max_batch_size'] = int(batch)
        # max_size MUST SCALE WITH BATCH or the cap binds and takes tau with it.
        # At batch 1000 against the hardcoded 12000, occupancy pinned at the cap
        # and rr_hc2_n20 / rr_hc2_n50 came out IDENTICAL on every buffer metric
        # (occupancy 12000, reuse 2.85/2.90, mean_age 21.0/21.1) despite a 2.5x
        # cadence difference -- the N knob was dead. Same rule the cluster
        # generator uses (configs/rr_sep08/make.py _size_replay).
        cfg['buffers']['replay_buffer']['max_size'] = max(12000, int(batch) * 50)
    if lr_scale is not None:
        # lr_control.mode is 'fixed', so the LIVE rate is seed_lr * fixed_scale.
        # The first harm curve ran 1.25e-4 * 0.125 = 1.5625e-5.
        cfg['lr_control']['fixed_scale'] = float(lr_scale)
    if warm:
        # A DIFFERENT PHASE-1 EXIT. The eight e01bd1 exits on disk are not
        # interchangeable in quality: screened at 2000 forward samples each,
        # six sit at frac(E>0) 0.42-0.46 and one -- dev_race_L2_transition --
        # at 0.054 with mean energy -97.5 kJ/mol (prior buffer is -126.5).
        # The W1 progress gate phase 1 exits on cannot see that: marginals
        # match (wass_debiased 0.0018 vs null 0.028) while the energy is
        # ~150 kJ/mol out. Both files must move together -- prior_model_name
        # is what skip_if: prior_loaded tests, and a mismatched pair silently
        # retrains phase 1 instead of warm-starting.
        stem = f'{warm}_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-e01bd1'
        cfg['checkpoint_name'] = f'{stem}_phase1_exit.pt'
        cfg['prior_model_name'] = f'{stem}_prior.pt'

    zc = cfg.setdefault('z_calibration', {})
    # the fill is the ONLY Z pin once the servo is off; >0 keeps it armed, small
    # makes it fire at every rollout, se-gated so a batch must resolve the gap
    zc['fill_threshold'] = 0.5
    zc['fill_se'] = 3.0
    zc['fill_cooldown_steps'] = 0
    # the absorber: every measurement moves Z by its precision share
    # (K = P/(P + se^2)); the eval rollout is the most precise one and is fed
    # to the same actuator
    zc['fill_mode'] = 'absorb'
    zc['fill_process_var'] = 0.01
    zc['fill_moment_reset'] = 0.5
    zc['fill_from_eval'] = 'fill'
    # q RE-MEASURED ON THE GOOD WARM START, and it reverts to the original.
    # The 0.09-0.50 estimated from the dev_elj_p2_cruise arms was that
    # checkpoint's policy THRASHING (root moving 0.5-0.7 nats/step), not a
    # property of the system. From dev_race_L2_transition the same estimator
    # gives 0.003-0.010 (0.06-0.10 nats/step) -- the shipped value was right.
    # At 0.25 with se now 0.55, K ran to 0.87 and log Z was near-snapping onto
    # measurement noise. See docs/design/z_fill_process_variance.md sec.4/5:
    # this is the "re-estimate from a good warm start" caveat firing.
    zc['fill_process_var'] = 0.01

    # _replay_val_size returns min(val_cap, batch_size, n_val), so anything above
    # batch_size is INERT -- v2 ran val_cap 2048 and measured val_n 234-400. The
    # se improvement there (2.0 -> 0.44 nats) came from the better checkpoint's
    # tighter residuals, not from this. Left at batch_size as the honest ceiling.
    cfg['buffers']['replay_buffer']['val_cap'] = int(cfg['batch_size'])

    n_fused = 0
    for proto in (cfg.get('protocols') or {}).values():
        for st in (proto.get('stages') or []):
            # ANCHOR-ONLY PRIOR: nothing writes a prior model any more
            if 'snapshot_prior' in (st.get('on_exit') or []):
                st['on_exit'] = [a for a in st['on_exit'] if a != 'snapshot_prior']
            if st.get('train_mode') != 'fused':
                continue
            # Z BOOTSTRAP AT PHASE-2 ENTRY. Phase 1 gives the flow scalar no
            # gradient -- the exit checkpoints carry log Z at exactly 0.0 -- so
            # without this the cadenced fill walks the level in from zero over
            # the first several hundred steps and everything trained meanwhile is
            # trained against a wrong level. 4000 samples: the entry fill is the
            # one that takes the WHOLE gap, so it is the one that must be well
            # resolved (se ~ rms/(sqrt(B)*frac_unclipped); the 400-row step-0 fill
            # ran at se 2.60, the worst of the run).
            if boot:
                on_enter = list(st.get('on_enter') or [])
                if not any(str(a).startswith('bootstrap_z') for a in on_enter):
                    on_enter.append(f'bootstrap_z:rollout:{int(boot)}')
                st['on_enter'] = on_enter
            st['fwd_rollout_every'] = int(every)
            # THE BACKSTOP IS DELIBERATELY LOOSE. On v2 at N=7,
            # fwd/tb_resid_clipped held within +-0.05 nats of zero for the whole
            # run -- log Z was sitting on its fixed point continuously, so that
            # cadence was far more often than the Z pin needed. The replay-side
            # bars below are meant to set the real cadence; `every` only bounds
            # how long Z can go unpinned when they are all quiet, which is the
            # one thing nothing else watches.
            # THE CADENCE IS EVENT-DRIVEN; `every` is only the backstop period.
            # Values picked by judgement, not derived -- the grid that would
            # derive them is in docs/design/replay_occupancy_and_cadence.md.
            # Each bar owns one failure and each fixes what it fires on:
            #
            #  ess_min 0.10        policy_drift_ess_frac. Below this the replay
            #                      gradient is >90% spent on trajectories the
            #                      policy has left. We measured 0.15 at N=7, so
            #                      this sits just under the known-tolerable point.
            #  val_gap_max 4.0     nats of held-out gap. ~2x the se at val_cap
            #                      2048 (~0.7), so it needs a real signal to fire.
            #                      A rollout RAISES the admission rate and so
            #                      LOWERS reuse -- the correct actuator here.
            #  occupancy_min 2.0   in batches. O >= B is hard (every step draws a
            #                      full batch); 2B leaves headroom to react. This
            #                      is the bar a loose period violates first.
            #
            # drift_std_max is left off: it is the same failure as ess_min in nats
            # rather than as a fraction, and two bars on one failure just double
            # the fire rate.
            #
            # THERE IS NO Z BAR, deliberately. Nothing measures log Z's fixed
            # point between rollouts -- the root is a property of a FRESH batch and
            # getting one IS the rollout. So the unpinned Z interval is bounded
            # OPEN-LOOP by fwd_rollout_every, and that period is the only thing
            # standing behind design invariant 1. Keep it tight enough to mean it.
            # ess_min IS RETIRED -- it LATCHES, and no rr_sep08 arm carries it.
            # policy_drift_ess_frac falls through 0.10 around step 2500 at a
            # realistic LR (never at 1.56e-5). The trigger then fires EVERY OTHER
            # STEP -- rollout/n per 10-step window 0.5 -> 5 -- floods the buffer
            # (occupancy 5.5k -> cap, mean_age 72 -> 21, reuse 21 -> 3), and the
            # sensor STILL does not clear (0.131 -> 0.088). The actuator cannot
            # satisfy its own sensor at that rate, so the arm spends the rest of
            # its life at a 10x rollout rate -- the premise of rare rollouts,
            # inverted. Measured on rr_hc2_n20 (step 2520) and rr_hc2_n50 (2460).
            st['fwd_rollout_triggers'] = {
                'val_gap_max': 4.0,
                'occupancy_min_batches': 2.0,
            }
            if val_gap_max is not None:
                # THE CADENCE BECOMES AN OUTPUT. `every` is only the backstop;
                # the bar pulls the effective cadence in to wherever the gap is
                # satisfied. Read rollout/n summed, not `every`.
                st['fwd_rollout_triggers']['val_gap_max'] = float(val_gap_max)
            st.setdefault('flags', {})['z_calibration'] = False
            # the gated-ramp controller: one sensor, two motions, hard rails
            f = float(fwd_frac)
            st['fracs'] = {'fwd': f, 'bwd': (1 - f) / 2, 'replay': (1 - f) / 2}
            st.pop('min_fracs', None)
            if f > 0:
                fwd_lc = st.setdefault('loss_coeffs', {}).setdefault('fwd', {})
                fwd_lc.update({'tb_z_source': 'batch_root', 'freeze_policy': 0.0, 'freeze_z': 1.0})
            st['balance'] = {
                'kind': 'gated_ramp', 'ramp': 'replay', 'guard': 'bwd',
                'pinned': {'fwd': f},
                # SETPOINT 0 ON THE Z-ANCHORED CHANNEL (owner call 2026-09-07).
                # The guard's purpose is to keep bwd coverage IMPROVING, so the
                # honest bar is "did it get worse at all", not a tolerated rate.
                # bar 1.0 on the level-blind channel fired 0 times in 2000 steps
                # while the metric it watched doubled (rr07_rr_n7_v1): the
                # deterioration that actually occurs is ~1.6 nats/1000 steps and
                # a 150-step-difference sensor needs 6.7 to clear 1.0.
                # under_coverage carries the Z level, which at bar 0 with a
                # symmetric hunt is noise the servo averages out rather than the
                # one-way starve it caused at bar 1.0.
                'metric': 'bwd/under_coverage_rise150', 'bar': 0.0,
                # RATCHET ON THE SAME QUANTITY THE SLOPE WATCHES -- its level.
                # A ratchet whose level and slope are different metrics does not
                # ratchet: the "best" being chased is not the thing the guard
                # guards, and the two can disagree indefinitely.
                #
                # And the channel is the Z-ANCHORED one on purpose (owner
                # 2026-09-07). under_coverage is the RMS of the negative tail of
                # the RAW residual log_pf + log_Z - log_r - log_pb, so it measures
                # the buffer's absorption w.r.t. the LIVE POLICY's normalisation.
                # relative_under re-centres on the batch's own Jensen centre
                # (utils.py:2019), which removes log Z entirely and leaves the
                # internal calibration -- the VarGrad-equivalent. That is a
                # different question and not the one this guard asks.
                # BOTH TOLERANCES ZERO (owner 2026-09-07). The slope bar is
                # already 0 ("did it get worse at all"); the level tol matches it,
                # so the ramp is released ONLY at or below the running best.
                # Consequence to watch: on rr07_rr_n7_v2 the level plateaued at
                # 14.8-15.8 against a best of 14.71, so at tol 0 the ramp is
                # clamped nearly always and bwd drifts to its 0.9 cap. There is
                # still no mechanism that decides a plateau counts as absorbed.
                # see configs/rr_sep07/make.py for the measurement behind these
                'ratchet_metric': 'bwd/under_coverage', 'ratchet_tol': 0.5,
                'ratchet_release_tol': 0.25,
                # 4x slower than the original, i.e. 5x FASTER than the 20x cut
                # (owner 2026-09-07, third pass). v2 moved bwd 0.50 -> 0.79 in
                # 2000 steps at the slow gains, so the loop does close in a run;
                # this makes it close inside a local arm too. Ticks are 10 steps,
                # so 0.25 of share is ~230 steps down and ~5900 up.
                'up': 0.004,
                'down': 0.006,
                'bounds': BOUNDS,
            }
            n_fused += 1
    assert n_fused >= 1, f'{name}: no fused stage to cadence'
    cfg['buffers']['prior_buffer']['source'] = 'anchors'

    rb = cfg['buffers']['replay_buffer']
    # OUTSIDE the store_all guard: rr_n1 is the control, and it can only be
    # compared on the gap if it carries the split too. 10% of every admission
    # is held out of the training draw so replay/val_gap is measured rather
    # than inferred. Measurement only -- nothing actuates on it.
    rb['val_frac'] = 0.1
    if store_all:
        rb['churn_rate'] = int(cfg['batch_size'])
        # tau_mult is the OCCUPANCY knob: O = B * tau/N = B * tau_mult, so it sets
        # the pool size and the mean age together and leaves reuse (= N) alone.
        rb['mean_residence_steps'] = int(tau_mult) * int(every)
    return cfg


def check(cfg, name, every, store_all, tau_mult=5):
    st = [s for p in cfg['protocols'].values() for s in p['stages']
          if s.get('train_mode') == 'fused']
    assert st and all(s['fwd_rollout_every'] == every for s in st), name
    assert all(s['flags'].get('z_calibration') is False for s in st), name + ': servo on'
    assert not any('fwd_rollout_drift_max' in s for s in st), name + ': drift trigger armed'
    rb = cfg['buffers']['replay_buffer']
    assert rb['val_frac'] == 0.1, name + ': val split'
    if store_all:
        assert rb['churn_rate'] == cfg['batch_size'], name
        assert rb['mean_residence_steps'] == tau_mult * every, name
        # a cap-bound buffer is tau-DISCONNECTED and measures nothing -- this is
        # the state that voided the first hc2 re-run (occupancy pinned at 12000
        # for both N=20 and N=50, mean_age 21 for both, the N knob dead).
        occ = cfg['batch_size'] * tau_mult
        assert rb['max_size'] >= occ * 1.5, (
            '%s: max_size %d leaves no headroom over occupancy %d'
            % (name, rb['max_size'], occ))


def main():
    # N = 7 is COPRIME with the 10-step metric cadence, so logged rows cover
    # both rollout and non-rollout steps; at N = 10 every logged row was a
    # rollout step and the 0.5/0.5 renormalisation was unobservable.
    # rr_n20 / rr_n50: the cadence sweep. Each rollout's z_fill/gap measures how far
    # log Z drifted over the N steps since the last one, and replay/resid_vs_intake
    # measures memorisation at reuse = N, so gap(N) and memo(N) on this system are a
    # real-data estimate of a reasonable N: where |gap| approaches the fill's se.
    # rr_n7_fwd: rr_n7 plus the level-blind forward policy step at weight 0.05
    # (the only arm on which the forward branch trains anything).
    # rr_n7_v1: the v1 ACCEPTANCE arm. Identical settings to rr_n7 -- a distinct
    # NAME, because rr_n7 is also the pre-v1 run it is compared against
    # (docs/design/handoff_rr_v1.md), and both the wandb run name and the local
    # checkpoint prefix are {tag}_{run_name}. Re-using the name would overwrite the
    # baseline's checkpoints and put two same-named runs in the comparison.
    # rr_n7_uc0: the SECOND acceptance run -- same arm as rr_n7_v1, but every
    # config here now carries the bar-0 guard on bwd/under_coverage_rise150, so
    # it needs its own name to stay comparable against rr07_rr_n7_v1 (which ran
    # the bar-1.0 guard on the level-blind channel and never fired).
    spec = {'rr_n1': (1, False, 0.0), 'rr_n7': (7, True, 0.0),
            'rr_n7_v1': (7, True, 0.0), 'rr_n7_uc0': (7, True, 0.0),
            # rr_n7_rat: the ratchet arm. Third acceptance run of the same arm --
            # bar 0 on the slope, high-water mark on the level, gains halved.
            'rr_n7_rat': (7, True, 0.0),
            'rr_n20': (20, True, 0.0), 'rr_n50': (50, True, 0.0),
            'rr_n7_fwd': (7, True, 0.05)}
    # rr_n7_boot: rr_n7_rat plus the phase-2 Z bootstrap, and NOTHING else --
    # a 4000-sample entry rollout whose winsorized-Huber root sets log Z. Its own
    # arm rather than a change to rr_n7_rat so the bootstrap's effect on the level
    # walk-in is readable as a one-variable difference.
    boots = {'rr_n7_boot': (7, True, 0.0, 4000)}
    # rr_n7_race: 300 steps from the ONE phase-1 exit that clears the owner's
    # bars, to confirm the screen's verdict in a live run before anything else
    # is re-based on it. Everything else matches rr_n7_boot.
    warms = {'rr_n7_race': (7, True, 0.0, 4000, 'dev_race_L2_transition', 300),
             # rr_n7_v2: the acceptance run for everything built 2026-09-07 --
             # good warm start, entry Z bootstrap, ratchet + bar-0 guard at 20x
             # slower gains, q 0.25, val_cap 2048, event-driven rollout triggers.
             # Full length, so it is comparable to rr07_rr_n7_v1 at matched step.
             'rr_n7_v2': (7, True, 0.0, 4000, 'dev_race_L2_transition', 2000),
             # rr_n50_v3: the loose-backstop arm. N=50 with tau=5N keeps occupancy
             # at 2000 rows and reuse at 50 -- 7x v2's reuse and 1/7 its energy
             # calls, with the replay-side bars expected to pull the effective
             # cadence back in wherever that is too far.
             'rr_n50_v3': (50, True, 0.0, 4000, 'dev_race_L2_transition', 2000),
             # THE HARM CURVE. val_gap_nats against reuse, from the good warm start,
             # all else fixed. Per-training-row reuse is ~N/(1-v), so these are ~22,
             # 56 and 111 draws per row against rr_n7_v2's ~7.8 -- the first look at
             # whether the gap grows with reuse and, if it does, whether run quality
             # follows it down. 4000 steps because tau = 5N and the buffer needs a few
             # tau to reach its age equilibrium (at N=100 that is tau=500).
             #
             # LOCAL RIG IS T=10 / batch 400 against the cluster's T=100 / batch 1000,
             # so read the SHAPE of gap-vs-reuse, not the absolute nats.
             'rr_hc_n20':  (20,  True, 0.0, 4000, 'dev_race_L2_transition', 4000),
             'rr_hc_n50':  (50,  True, 0.0, 4000, 'dev_race_L2_transition', 4000),
             'rr_hc_n100': (100, True, 0.0, 4000, 'dev_race_L2_transition', 4000)}
    # THE HARM CURVE, SECOND CUT -- same three cadences, two things changed.
    # BATCH 1000 matches the cluster rig (val_cap and churn_rate follow it), and
    # fixed_scale 0.8 puts the live LR at exactly 1.0e-4 against the first cut's
    # 1.5625e-5. The first cut measured val_gap 0.36/0.59/1.36 at N=20/50/100 with
    # the policy barely moving; reuse cannot hurt a policy that is not changing,
    # so a 6.4x LR is the condition under which the knee should move IN.
    hc2 = {'rr_hc2_n20':  (20,  True, 0.0, 4000, 'dev_race_L2_transition', 4000, 1000, 0.8, 5),
           'rr_hc2_n50':  (50,  True, 0.0, 4000, 'dev_race_L2_transition', 4000, 1000, 0.8, 5),
           'rr_hc2_n100': (100, True, 0.0, 4000, 'dev_race_L2_transition', 4000, 1000, 0.8, 5),
           # 5x THE RETENTION at the same cadence: tau 100 -> 500, so occupancy
           # goes 5000 -> 25000 rows and mean age 73 -> ~370 steps, while reuse
           # stays at N = 20. The direct test of whether tau cancels: a row's
           # exposure is tau/O = 1/admissions, so absorption should NOT move.
           # 6000 steps, not 4000 -- the buffer needs ~5*tau = 2500 steps to reach
           # age equilibrium, and a settled window has to come after that.
           'rr_hc2_n20_t25': (20, True, 0.0, 4000, 'dev_race_L2_transition', 6000, 1000, 0.8, 25)}

    # CAN A BIG BUFFER PAY FOR A LOOSE CADENCE? The two levers point opposite ways:
    # N sets reuse (total draws per row) and costs energy calls; tau/N sets DENSITY
    # (draws per row per STEP = N/tau) and is free. n20_t25 showed density alone
    # buys -34% on val_gap at zero cost. These ask whether it also pays off the
    # N=100 penalty.
    #
    #   hc2_n20      N= 20  O= 5544  density 0.20   val_gap 1.497
    #   hc2_n100     N=100  O= 6371  density 0.20   val_gap 2.730
    #   hc2_n20_t25  N= 20  O=26340  density 0.04   val_gap 0.942
    #   n100_t25     N=100  O=25000  density 0.04   val_gap ?      <- HIGH reuse,
    #                                                                 LOW density
    # Same buffer as n20_t25, so N is isolated. If density fully compensates it
    # lands near 0.94; if reuse dominates it stays near 2.73; the -34% seen at
    # N=20 would put it at ~1.80, i.e. BETTER than N=20 today at a FIFTH of the
    # energy calls.
    #
    # 18000 steps because tau = 2500 and the buffer needs 5*tau = 12500 to reach
    # age equilibrium -- reading this before then inverts the sign (measured: the
    # same n20 pair gave lambda_tau +18% at 2*tau and -4% settled).
    #
    # _vg is the PRODUCTION shape: the cadence is emergent, `every` is only the
    # backstop, and the bar is the knob. Few energy calls is then a RESULT, not
    # a setting -- read rollout/n summed over the settled window.
    hc3 = {'rr_hc2_n100_t25':
               (100, True, 0.0, 4000, 'dev_race_L2_transition', 18000, 1000, 0.8, 25, None),
           'rr_hc2_n100_t25_vg':
               (100, True, 0.0, 4000, 'dev_race_L2_transition', 18000, 1000, 0.8, 25, 1.5)}
    arms = {}
    for name, (every, store_all, fwd_frac) in spec.items():
        cfg = build(name, every, store_all, fwd_frac)
        check(cfg, name, every, store_all)
        arms[name] = cfg
    for name, (every, store_all, fwd_frac, boot) in boots.items():
        cfg = build(name, every, store_all, fwd_frac, boot=boot)
        check(cfg, name, every, store_all)
        arms[name] = cfg
    for name, (every, store_all, fwd_frac, boot, warm, steps) in warms.items():
        cfg = build(name, every, store_all, fwd_frac, boot=boot, warm=warm, steps=steps)
        check(cfg, name, every, store_all)
        assert cfg['checkpoint_name'].startswith(warm) and cfg['prior_model_name'].startswith(warm), name
        arms[name] = cfg
    for name, (every, store_all, fwd_frac, boot, warm, steps, batch, lrs, tm) in hc2.items():
        cfg = build(name, every, store_all, fwd_frac, boot=boot, warm=warm, steps=steps,
                    batch=batch, lr_scale=lrs, tau_mult=tm)
        check(cfg, name, every, store_all, tau_mult=tm)
        arms[name] = cfg
    for name, (every, store_all, fwd_frac, boot, warm, steps, batch, lrs, tm, vg) in hc3.items():
        cfg = build(name, every, store_all, fwd_frac, boot=boot, warm=warm, steps=steps,
                    batch=batch, lr_scale=lrs, tau_mult=tm, val_gap_max=vg)
        check(cfg, name, every, store_all, tau_mult=tm)
        st = [s for p in cfg['protocols'].values() for s in p['stages']
              if s.get('train_mode') == 'fused'][0]
        assert st['fwd_rollout_triggers']['val_gap_max'] == (4.0 if vg is None else vg), name
        rb = cfg['buffers']['replay_buffer']
        assert cfg['batch_size'] == batch == cfg['max_batch_size'], name
        assert rb['val_cap'] == batch and rb['churn_rate'] == batch, name
        assert cfg['lr_control']['fixed_scale'] == lrs, name
        arms[name] = cfg
    for name, cfg in arms.items():
        with (HERE / f'{name}.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        st = [s for p in cfg['protocols'].values() for s in p['stages'] if s.get('train_mode') == 'fused'][0]
        rb = cfg['buffers']['replay_buffer']
        print(f"{name:<8} every={st['fwd_rollout_every']} z_cal={st['flags']['z_calibration']} "
              f"fill_thr={cfg['z_calibration']['fill_threshold']} churn={rb['churn_rate']} "
              f"tau={rb['mean_residence_steps']} batch={cfg['batch_size']} "
              f"val_frac={rb['val_frac']}")


if __name__ == '__main__':
    main()
