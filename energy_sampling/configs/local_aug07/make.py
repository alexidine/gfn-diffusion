"""
Local shakedown battery, 2026-08-07. Validates everything that landed on
2026-08-06/07: the two-point step probe, prioritised replay + uniform intake +
hazard-only purge, and the lambda*tau memorisation servo.

See docs/to_do_rebuild.md 0b/0c for the battery's design and kill-gates.

mk_dev.yaml is USER-OWNED and only ever READ here. Every config gets its own
run_name, so the checkpoint prefix ({tag}_{run_name}_{problem}) is distinct and
no arm can clobber the dev_mk_dev_* set (or each other).

TWO TRAPS these configs are written around:
  1. On a RESUMED run the loop is trange(init_step, epochs+1), so `epochs` is an
     ABSOLUTE step ceiling, not a budget. An arm resuming at 8000 with
     epochs: 2000 runs ZERO steps and verifies nothing. Every resumed arm below
     sets epochs = P1_STEPS + its own budget.
  2. train.py ignores wandb_mode -- every local run creates a real run in the
     "GFN Energy" project. Expected here; the tag keeps them grouped.
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
MK_DEV = HERE.parent / 'mk_dev.yaml'
TAG = 'batt0807'

# Warm start for the p1 arm. WEIGHTS ONLY and deliberately so: phase 1 still
# RUNS -- exercising MLE, tbc, the exit gates and the stage handoff, which is
# what a shakedown battery is for -- but it starts from converged weights, so it
# should trip its exit gates in a few evals instead of converging from scratch.
#
# load_weights_only restores model weights alone: optimizers, schedulers,
# buffers, metrics and condition_log_z all initialise fresh at step 0. The
# prior buffer is therefore reseeded (~159k samples, seconds). The GFN is
# rebuilt from the CHECKPOINT's stored gfn_config bar RECONFIGURABLE_GFN_KEYS,
# so architecture follows the checkpoint, not this config.
#
# Read-only w.r.t. its source: this battery only ever writes batt0807_* prefixes.
WARM_CKPT = 'dev_mk_dev_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_phase1_exit.pt'

# Set from the p1 arm's ACTUAL phase-1 exit step once it lands. Every resumed
# arm's `epochs` is P1_STEPS + budget because of trap 1 above.
P1_STEPS = 580

# PHASE-2 STABLE resume point (user 2026-08-07: "checkpointing from our first
# main phase 2 run would be more informative -- we can skip the initial log Z
# calibration which is shifting everything very fast").
#
# Confirmed by measurement, and it invalidated a headline reading. Across p1's
# phase 2 (start -> 25% -> 50% -> 75% -> end):
#
#   log_Z_learned      0.91   20.7   20.5   20.45  20.58   settles by ~25%
#   tb_resid_clipped  -6.24   0.13   0.21   -0.06  -0.15   inside D29's +-0.5 after
#   alpha_median      10.19   5.14   2.42    1.73   1.68   STILL MOVING at 75%
#
# alpha* read at step ~790 gave ~0.5 and an "edge of stability" conclusion; the
# settled value is ~1.7, i.e. UNDERSHOOT. Anything measured before log Z lands is
# measuring the transient, and that goes double for the replay arms: is_elig_frac
# tracked log_Z 0.27 -> 0.54 over the same window, because delta_plus is not
# shift-invariant (B8).
P2_STEPS = 2650
P2_CKPT = f'{TAG}_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_running.pt'
P1_CKPT = f'{TAG}_p1_elj-mipcas_sg2_zp1_elj_prior_dataset-T2.5-573c92_phase1_exit.pt'


def base():
    with MK_DEV.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write(name, cfg):
    cfg['run_name'] = name
    cfg['tag'] = TAG
    p = HERE / f'{name}.yaml'
    with p.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'  wrote {p.name}')


def local(cfg):
    """Shared local-shakedown settings: short, frequent eval, cheap figures."""
    cfg['eval_period'] = 250
    cfg['figs_period'] = 1000        # must be a multiple of eval_period
    cfg['archive_period'] = 0        # no archives: these are throwaway arms
    return cfg


def probe(cfg, enabled=True, cadence=20):
    cfg['step_probe'] = {'enabled': bool(enabled), 'cadence': cadence, 'window': 25}
    return cfg


def prioritise(cfg, enabled=True, kappa=1.0):
    cfg['buffers']['replay_buffer']['prioritise'] = {
        'enabled': bool(enabled), 'kappa': float(kappa)}
    return cfg


def dehuberise(cfg, beta=1.0e6):
    """B5b: any branch running a prioritised draw must be effectively quadratic.
    fwd is DELIBERATELY untouched -- its beta defines the Z fixed point that
    D29's invariant is stated against."""
    cfg['bwd_loss_coeffs']['beta'] = beta
    cfg['replay_loss_coeffs']['beta'] = beta
    return cfg


PAIR_BATCH = 1000


def paired(cfg):
    """Size an arm so TWO can share the GPU.

    Measured 2026-08-07: one arm at batch 2831 holds 14.7 of 16.3 GB (the 0.9
    cuda_memory_fraction is a hard reservation, not a high-water mark) at 57%
    util -- so the GPU is under-fed but the VRAM is spoken for. Batch 1000 lands
    near 5 GB, and halving the fraction stops either process from claiming the
    lot.

    max_batch_size is pinned EQUAL to batch_size deliberately. Growth is already
    off, but this also makes the OOM-recovery cut the only thing that can move
    the batch -- and that path must not fire, because:

    BATCH SIZE IS NOT A NEUTRAL KNOB HERE. B7a derives lambda*tau ~ B / rate, so
    the batch enters the memorisation product directly. An OOM cut in one arm
    and not another would silently give the two different lambda*tau, which is
    exactly the quantity r4's servo arm exists to measure. Watch for 'OOM' in
    the logs; if it fires, the paired result is void, not merely noisy.
    """
    cfg['batch_size'] = PAIR_BATCH
    cfg['max_batch_size'] = PAIR_BATCH
    cfg['grow_batch_size'] = False
    cfg['auto_batch_throughput_opt'] = False
    cfg['cuda_memory_fraction'] = 0.45
    return cfg


def resume_p2(cfg, budget):
    """Resume from the POST-TRANSIENT phase-2 state. Same pinning discipline as
    resume(), different anchor -- use this for anything whose reading depends on
    the residual distribution or on log Z having settled."""
    cfg['checkpoint_name'] = P2_CKPT
    cfg['continue_from_checkpoint'] = False
    cfg['reuse_prior'] = False
    cfg['epochs'] = P2_STEPS + budget
    assert cfg['epochs'] > P2_STEPS, 'zero budget verifies nothing'
    return cfg


def resume(cfg, budget):
    """Pin the resume point EXPLICITLY. mk_dev defaults are
    continue_from_checkpoint: true + checkpoint_name: null, which resolve to
    {tag}_{run_name}_{problem}_running.pt. Since run_name is unique per arm that
    file never exists, so a generator that forgets this does not chain arms --
    it silently RETRAINS PHASE 1 in every arm, which is invisible in the results
    and costs the whole day. Asserted below, per the local-run recipe."""
    cfg['checkpoint_name'] = P1_CKPT
    cfg['continue_from_checkpoint'] = False   # checkpoint_name takes precedence
    cfg['reuse_prior'] = False
    cfg['epochs'] = P1_STEPS + budget         # ABSOLUTE ceiling -- see trap 1
    assert_pinned_resume(cfg, budget)
    return cfg


def assert_pinned_resume(cfg, budget):
    assert cfg['checkpoint_name'] == P1_CKPT, 'resume point not pinned'
    assert cfg['continue_from_checkpoint'] is False, 'would resolve to _running.pt'
    assert cfg['reuse_prior'] is False, 'would auto-reload a stale prior'
    assert cfg['epochs'] == P1_STEPS + budget, 'epochs must be ABSOLUTE, not a budget'
    assert budget > 0, 'zero budget runs no steps and verifies nothing'


def main():
    # --- run 0: null regression. Everything new OFF. Confirms the five call
    # sites touched on 08-06/07 are genuinely inert without their config blocks.
    c = local(base())
    c['epochs'] = 600
    c['continue_from_checkpoint'] = False
    write('r0_null', c)

    # --- run 1: fresh phase 1, probe ON. Produces the snapshot every arm below
    # branches from, and doubles as the probe's first read. The ONE number that
    # matters early is probe/second_diff_rel: at the 1e-6 floor the probe is
    # resolving float32 rounding, not curvature, and Part A stops.
    c = probe(local(base()))
    c['checkpoint_name'] = WARM_CKPT
    c['load_weights_only'] = True
    c['continue_from_checkpoint'] = False
    c['reuse_prior'] = False     # must stay false, or skip_if: prior_loaded
    c['epochs'] = 4000           # skips the stage we are trying to exercise
    write('p1', c)

    # --- run 2: replay wiring sanity. is_elig_frac ~0.5 is the check that the
    # newly-live update_logw_stats has the right sign and tiling.
    c = paired(dehuberise(prioritise(probe(local(base())))))
    write('r2_wiring', resume_p2(c, 800))

    # --- run 3: kappa ladder. Phi is invariant by construction, so any
    # difference IS estimator variance. Watch is_ess_frac.
    for k in (0.0, 1.0):
        c = paired(dehuberise(prioritise(probe(local(base())), kappa=k)))
        write(f'r3_kappa{str(k).replace(".", "")}', resume_p2(c, 1500))

    # --- run 4: deliberate overfit -> servo recovery. Starve intake so
    # lambda*tau climbs past 1 (ratio below 1/e), then confirm the servo pulls
    # it back. churn_rate is the actuator B7a proves is the only lever.
    c = paired(dehuberise(prioritise(probe(local(base())))))
    # lambda*tau ~ B/rate (B7a). Batch fell 2831 -> 1000 under paired(), which
    # cuts lambda*tau ~2.8x on its own -- so the intake starve has to be ~2.8x
    # deeper to induce the SAME memorisation this arm is built to provoke.
    c['buffers']['replay_buffer']['churn_rate'] = 3       # was 8 at batch 2831
    c['buffers']['replay_buffer']['mean_residence_steps'] = 400
    write('r4_overfit', resume_p2(copy.deepcopy(c), 2000))
    c['protocol']['stages'][1]['buffer_servo'] = {
        'numerator': 'replay/ema_loss_mean',
        'denominator': 'replay/birth_loss_mean',
        'bar': 0.368, 'release': 0.60, 'scale': 0.15,
        'gain': 0.05, 'relax': 0.5, 'max_step': 0.05, 'max_boost': 8.0,
    }
    write('r4_overfit_servo', resume_p2(c, 2000))

    # --- run 5: replay's structural case (P8 arm i). fwd carries policy grads,
    # replay off entirely -- the closest thing to standard on-policy TB, which
    # is the bar Part B has to beat.
    c = paired(dehuberise(probe(local(base()))))
    st = c['protocol']['stages'][1]
    st['fracs'] = {'fwd': 0.4, 'bwd': 0.6, 'replay': 0.0}
    st['loss_coeffs']['fwd'].pop('freeze_policy', None)
    # No balance controller on this arm. With replay at 0 the bwd/replay split
    # is degenerate, and `pinned.fwd` would contradict the fracs above (the
    # config regression catches exactly that). Dropping it also removes a
    # confound: r5 is the BASELINE the prioritised work has to beat, so it
    # should be a fixed mix, not a fixed mix plus a controller that landed the
    # same week.
    st.pop('balance', None)
    st.pop('min_fracs', None)
    st.pop('deactivate_threshold', None)
    write('r5_nopolicy_replayoff', resume_p2(c, 1500))

    # --- run 6: bwd beta ladder. B5b says de-huberizing is what restores
    # within-tail ordering; this is the arm that shows it.
    for beta, name in ((10.0, 'b10'), (60.0, 'b60')):
        c = paired(dehuberise(prioritise(probe(local(base()))), beta=beta))
        write(f'r6_bwdbeta_{name}', resume_p2(c, 800))

    # --- run 7: DIRECT FALSIFICATION OF THE PROBE (added 2026-08-07 off the
    # alpha* ~ 0.5 reading in p1). For a locally quadratic loss with step
    # d = lr * dir,
    #
    #     alpha* = (g.d)/(d'Hd) = (1/lr) * (g.dir)/(dir'H dir)
    #
    # so alpha* is EXACTLY inversely proportional to lr, everything else held.
    # p1 measured alpha_median 0.42-0.50 at the configured lr, so:
    #
    #     lr x 0.5  ->  alpha_median ~0.84-1.00
    #     lr x 2.0  ->  alpha_median ~0.21-0.25
    #
    # This is a sharp, quantitative, cheap prediction. If alpha* tracks 1/lr the
    # sensor measures what A3 claims and alpha_target becomes meaningful. If it
    # does not, alpha* is reading something else and stage 2 must not be built
    # on it -- which no amount of watching a single run would reveal, because
    # the fault would look like a plausible number the whole time.
    #
    # Cheap and short: the reading stabilises within a few hundred steps.
    for mult, name in ((0.5, 'half'), (2.0, 'double')):
        c = paired(probe(local(base())))
        for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
            c[k] = 'auto'          # resolver fills these; scale via peak below
        c['lr_scale_probe'] = mult  # recorded for provenance only
        c['adaptive_lr'] = dict(c['adaptive_lr'])
        c['adaptive_lr']['enabled'] = False   # flat LR: no ramp/hold confound
        c['lr_fused'] = round(1.25e-4 * mult, 10)   # p1's resolved lr_fused
        write(f'r7_lr_{name}', resume_p2(c, 800))


    # --- run 8: IS THE PAIRED BATTERY MEASURING BATCH SIZE? (2026-08-07)
    # Every paired arm degrades (fwd/tb_err rising) while p1 -- unpaired, batch
    # 2831 -- improved over its phase 2. The one thing every paired arm shares
    # and p1 does not is batch 1000, imposed by paired() so two fit in VRAM.
    # If batch is the driver then every paired result measures batch size rather
    # than its nominal variable, and the whole pairing strategy is void.
    # Identical to r6_bwdbeta_b10 except batch: runs ALONE at 2831.
    c = dehuberise(prioritise(probe(local(base()))), beta=10.0)
    c['batch_size'] = 2831
    c['max_batch_size'] = 2831
    c['grow_batch_size'] = False
    c['auto_batch_throughput_opt'] = False
    c['cuda_memory_fraction'] = 0.9
    write('r8_batch2831', resume_p2(c, 800))


    # --- run 9: IS IT REPLAY, OR IS IT freeze_policy? (2026-08-07)
    # r5 improves (tb_err 21.5 -> 20.1) while every replay-on arm degrades, and
    # r5 is at batch 1000 paired -- so batch is NOT the confound. But r5 changed
    # TWO things: replay off, AND fwd carrying policy gradients. This arm moves
    # only one: replay stays ON and prioritised, fwd gets its policy grads back.
    #   improves  -> the culprit was freeze_policy, replay is fine
    #   degrades  -> the culprit is the replay branch itself, which is P8 arm (i)
    #                answering in the direction that kills most of Part B
    c = paired(dehuberise(prioritise(probe(local(base()))), beta=10.0))
    c['protocol']['stages'][1]['loss_coeffs']['fwd'].pop('freeze_policy', None)
    write('r9_replay_on_fwdpolicy', resume_p2(c, 1500))


if __name__ == '__main__':
    print(f'writing {TAG} configs to {HERE}')
    main()
