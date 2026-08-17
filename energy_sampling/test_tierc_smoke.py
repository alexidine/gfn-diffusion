"""
Tests for `tierc_smoke` -- the tier-C instrument.

WHAT THESE ARE FOR. The harness's whole value is that a pass means something, so
the tests that matter are the ones that pin its ability to FAIL. Every check
here was written by breaking the thing it guards and requiring a failure; the
three that came from real defects during the build are marked, because a test
whose bug is hypothetical and one whose bug actually shipped are different
evidence.

The end-to-end behaviour -- null zero, negative control non-zero -- is not
tested here. It needs a GPU, the data drive and ~90 s per run, and it is run as
a command (`python -m tierc_smoke --null <cfg>`), which is the form the plan
asks for. What IS tested here is everything that decides whether that command's
verdict can be believed.
"""
import copy
import math

import pytest

import tierc_smoke as T


# --------------------------------------------------------------- the trap ---

def test_epochs_is_an_absolute_index_not_a_count():
    """`epochs` is where a smoke harness silently measures nothing.

    `trange(init_step, epochs + 1)` runs `epochs - init_step + 1` steps, so the
    budget has to be expressed against the resume step AND carry the -1. A
    warm-started config given a raw count runs zero steps and reports clean."""
    assert T.epochs_for_steps(0, 30) == 29          # trange(0, 30) -> 30 steps
    assert T.epochs_for_steps(6680, 30) == 6709     # not 30, and not 6710
    # The registry's own formula omits the -1; that is the off-by-one this
    # harness has to correct rather than inherit.
    from benchmarks import registry
    w = registry.benchmark(T.BENCHMARK_ID)['work']
    n = w['warmup_steps'] + w['measure_steps']
    assert registry.epochs_for(T.BENCHMARK_ID, 6680) == T.epochs_for_steps(6680, n) + 1


def _trace(n_steps=30, n_logged=3, loss=1.0):
    return {
        'meta': {'executed_steps': n_steps, 'init_step': 0,
                 'epochs': T.epochs_for_steps(0, n_steps),
                 'final_step_ind': T.epochs_for_steps(0, n_steps)},
        'steps': [{'step': i, 'step_type': 'bwd', 'loss': loss + i,
                   'batch_size': 1000, 'lr': {'bwd': 1e-5}}
                  for i in range(n_steps)],
        'logged': [{'step': 10 * i, 'values': {'bwd/loss': 1.0 * i,
                                               'train_step_time': 0.2 + i}}
                   for i in range(n_logged)],
    }


def test_zero_steps_is_reported_as_the_trap_it_is():
    tr = _trace(n_steps=0)
    tr['meta']['executed_steps'] = 0
    ok, msg = T.verify_step_count(tr, 30)
    assert not ok
    assert 'ZERO STEPS' in msg


def test_step_count_verifier_catches_a_short_run():
    tr = _trace(n_steps=30)
    tr['meta']['executed_steps'] = 29
    ok, msg = T.verify_step_count(tr, 30)
    assert not ok and 'executed 29' in msg


def test_step_count_verifier_catches_a_mis_set_epochs():
    """The three readings must AGREE. A budget that is right by accident while
    `epochs` is wrong is a budget that stops being right at the next resume."""
    tr = _trace(n_steps=30)
    tr['meta']['epochs'] = 30                       # the missing -1
    ok, msg = T.verify_step_count(tr, 30)
    assert not ok and 'epochs=30' in msg


def test_step_count_verifier_passes_a_good_run():
    ok, msg = T.verify_step_count(_trace(), 30)
    assert ok, msg


def test_a_record_the_train_step_wrapper_did_not_write_is_caught():
    """THE DEFECT THIS WAS WRITTEN FOR. The fused sub-loss hook fires from
    INSIDE `train_step`, so `recorded_steps[-1]` is still the previous step when
    it runs. Appending on that mismatch turned 1200 executed steps into 2019
    records -- one extra per fused step, and the error is UPWARD, so it reads as
    more work rather than as a broken instrument.

    It also shows why the null had to be re-established at the longer length:
    the 30-step window never leaves the bwd stage, so this hook never fires
    there and no amount of running the short null would have found it."""
    tr = _trace(n_steps=30)
    tr['steps'].append({'step': 29, 'sub_losses': {'bwd': 1.0}})
    tr['meta']['executed_steps'] = 31
    ok, msg = T.verify_step_count(tr, 30)
    assert not ok
    assert 'not written by the train_step wrapper' in msg


# ------------------------------------------------------ capture, not count ---

def test_empty_capture_fails_even_though_the_run_completed():
    """THE DEFECT THIS WAS WRITTEN FOR. The first build of this harness ran all
    30 steps, verified the count, and captured zero metrics: `wandb.init`
    rebinds `wandb.log` and replaced the wrapper. Nothing raised, and two empty
    captures compare equal -- so the null test would have passed and certified
    an instrument that was recording nothing."""
    tr = _trace(n_logged=0)
    ok, msg = T.verify_capture(tr, 30)
    assert not ok
    assert 'detached' in msg


def test_capture_passes_when_the_reporting_grid_is_covered():
    ok, msg = T.verify_capture(_trace(n_steps=30, n_logged=3), 30)
    assert ok, msg


def test_capture_rejects_a_non_finite_loss():
    tr = _trace()
    tr['steps'][7]['loss'] = float('nan')
    ok, msg = T.verify_capture(tr, 30)
    assert not ok and 'non-finite' in msg


def test_capture_rejects_a_step_record_with_no_loss():
    tr = _trace()
    del tr['steps'][3]['loss']
    ok, msg = T.verify_capture(tr, 30)
    assert not ok and 'no loss' in msg


# ------------------------------------------------------------ comparison ----

def test_identical_traces_compare_identical():
    assert T.compare_traces(_trace(), _trace()).identical


def test_one_changed_float_is_caught():
    """The bar the negative control is held to, in miniature: a single value
    moving by one float32 ulp must register."""
    a, b = _trace(), _trace()
    b['steps'][17]['loss'] = a['steps'][17]['loss'] * (1 + 1e-7)
    cmp = T.compare_traces(a, b)
    assert not cmp.identical
    assert any(p == 'steps[17].loss' for p, _, _ in cmp.diffs)


def test_nan_equals_nan_but_nan_does_not_equal_a_number():
    """The NaN fold must not become a general tolerance.

    `zmatch/*_level` is NaN in every short run, and `nan != nan` reported it as
    a difference on both sides -- a comparator bug, not a finding. Folding it is
    correct; folding it too far would hide a metric that WENT NaN, which is one
    of the loudest things a config change can do."""
    a, b = _trace(), _trace()
    a['logged'][1]['values']['zmatch/fwd_level'] = float('nan')
    b['logged'][1]['values']['zmatch/fwd_level'] = float('nan')
    assert T.compare_traces(a, b).identical

    b['logged'][1]['values']['zmatch/fwd_level'] = 0.0
    cmp = T.compare_traces(a, b)
    assert not cmp.identical, 'a metric that went from NaN to a number was missed'


def test_a_key_present_on_only_one_side_is_reported_not_ignored():
    a, b = _trace(), _trace()
    b['logged'][1]['values']['lr_ctrl/hyper_cos'] = 0.3
    cmp = T.compare_traces(a, b)
    assert not cmp.identical
    assert 'logged[1].values.lr_ctrl/hyper_cos' in cmp.only_b


def test_a_differing_step_count_is_never_identical():
    assert not T.compare_traces(_trace(30), _trace(29)).identical


def test_wallclock_values_do_not_make_traces_differ():
    """Two identical runs disagree on `train_step_time` every time. If that
    counted, the null could never be zero and tier C could not be an exact
    test."""
    a, b = _trace(), _trace()
    for e in b['logged']:
        e['values']['train_step_time'] += 1.5
    assert T.compare_traces(a, b).identical


# ------------------------------------------------------- classification -----

@pytest.mark.parametrize('name', [
    'train_step_time', 'eval_sampling_time', 'samples_per_sec',
    'gpu/util_recent', 'gpu/util_policy', 'vram/peak_reserved_mb',
    'energy/seconds', 'energy/ms_per_sample', 'energy/frac_of_step',
    'initialization_time', 'probe/step_time_max10',
])
def test_timing_and_occupancy_are_not_compared(name):
    assert T.is_wallclock(name, T.registry_wallclock_metrics())


@pytest.mark.parametrize('name', [
    'bwd/loss', 'bwd/mle', 'bwd/tbc', 'lr_bwd', 'lr_ctrl/peak_scale',
    'phase', 'gradnorm/nonfinite_steps', 'batch/oom_events', 'energy/calls',
    'Batch Size',
])
def test_deterministic_quantities_are_compared(name):
    assert not T.is_wallclock(name, T.registry_wallclock_metrics())


def test_millisecond_timers_are_excluded_and_rms_metrics_are_not():
    """THE DEFECT THIS WAS WRITTEN FOR, and the reason the rule is narrow.

    `probe/churn_add_ms_max` and `probe/churn_purge_ms_max` are millisecond
    timers that matched none of the original rules. They were the ENTIRE content
    of a 42-value non-zero null at 600 steps -- the first length that reaches the
    buffer-churn code they time, which is why the 30-step null passed without
    them.

    The `_ms` rule must not catch `_rms`. `tracker/logw_std_rms` and its three
    siblings are deterministic statistics, and excluding them would drop real
    signal while looking like a tightening."""
    reg = T.registry_wallclock_metrics()
    assert T.is_wallclock('probe/churn_add_ms_max', reg)
    assert T.is_wallclock('probe/churn_purge_ms_max', reg)
    for name in ('tracker/logw_std_rms', 'tracker/tb_err_rms',
                 'tracker/z_bias_rms', 'tracker/z_grad_rms'):
        assert not T.is_wallclock(name, reg), name


def test_batch_size_survives_the_registrys_cost_grouping():
    """`Batch Size` sits in the registry's `cost` group because it is a
    throughput DENOMINATOR, not because it is a timing. Classifying it as
    wallclock would blind the comparison to a config that changed the batch --
    which changes everything downstream of it."""
    assert 'Batch Size' in T.registry_wallclock_metrics()
    assert not T.is_wallclock('Batch Size', T.registry_wallclock_metrics())


def test_split_keeps_both_halves():
    det, wall = T.split_trace(_trace())
    assert det['logged'][1]['values'] == {'bwd/loss': 1.0}
    assert 'train_step_time' in wall['logged'][1]['values'], (
        'the wallclock half must be RECORDED, not dropped -- a key that '
        'vanishes from a trace reads as a key that never existed')


# ------------------------------------------------------------- overrides ----

def test_registry_neutralisers_survive_into_the_overrides():
    """The five settings that silently unfix a run's work quantity. Three of
    them are actuated by wall clock, which is why an exact comparison needs them
    off and not merely constant."""
    ov, _ = T.registry_overrides()
    assert ov['checkpoint_read_only'] is True
    assert ov['grow_batch_size'] is False
    assert ov['auto_batch_throughput_opt'] is False
    assert ov['max_step_seconds'] == 0
    assert ov['archive_period'] == 0


def test_the_shipping_registry_sets_no_retired_key():
    """THE DEFECT THIS WAS WRITTEN FOR, now fixed at the source.
    `benchmarks/registry.yaml` used to set `ray_calibration.enabled: false`, and
    both that key and its parent are retired under current code -- so applying
    the registry verbatim hard-failed at preflight, and `_validate_defaults`
    meanwhile REQUIRED the key, so the two halves could not both be satisfied.
    The registry no longer sets it (the probe arms from the stage declarations
    and an override cannot reach them). Nothing should be dropped now, because
    there is nothing retired left to drop."""
    ov, dropped = T.registry_overrides()
    assert 'ray_calibration' not in ov
    assert T._retired_paths_in(ov) == []
    assert dropped == [], f'registry has drifted again: {dropped}'


def test_a_retired_override_is_dropped_and_reported(monkeypatch):
    """The drop machinery itself, exercised on an injected retirement rather
    than on the registry's own drift.

    Pinned separately BECAUSE the test above now passes trivially: with a clean
    registry, `registry_overrides` returning the identity would satisfy it, and
    the day the registry drifts again the drop path would be running for the
    first time. A silent drop is how that drift stops being visible, so the
    report is asserted alongside the removal.

    The parent/child pair is the case that needs the ordering: both
    `ray_calibration` and `ray_calibration.enabled` are retired, and popping the
    parent first leaves the child dangling -- while popping only the child
    leaves `ray_calibration: {}`, which is still the retired top-level key,
    since preflight fires on PRESENCE and not on value."""
    from benchmarks import registry
    monkeypatch.setattr(registry, 'resolved_overrides',
                        lambda _bid: {'checkpoint_read_only': True,
                                      'ray_calibration': {'enabled': False}})
    ov, dropped = T.registry_overrides()
    assert 'ray_calibration' not in ov, 'the emptied parent block survived'
    assert ov['checkpoint_read_only'] is True, 'a live override was collateral'
    assert any('ray_calibration.enabled' in d for d in dropped)
    assert any('ray_calibration' in d for d in dropped)
    assert T._retired_paths_in(ov) == []


def test_retired_key_detection_reads_the_shipping_table():
    """Not a second copy of the list. A second list goes stale exactly when it
    matters."""
    import utils
    probe = {k: 1 for k in list(utils._RETIRED_KEYS)[:1] if '.' not in k}
    if not probe:
        pytest.skip('no top-level retired key to probe with')
    assert T._retired_paths_in(probe)
    assert T._retired_paths_in({'definitely_not_retired_xyz': 1}) == []


# ---------------------------------------------------------------- merge -----

def test_deep_merge_does_not_drop_the_other_half_of_a_block():
    base = {'energy_config': {'temperature': 1.0, 'lj_coeff': 1.0}}
    T._deep_merge(base, {'energy_config': {'temperature': 2.0}})
    assert base['energy_config'] == {'temperature': 2.0, 'lj_coeff': 1.0}


def test_deep_merge_replaces_lists_wholesale():
    """Load-bearing for the negative control: stage lists are merged as whole
    lists, so a sparse patch would delete every stage it omitted -- and a config
    with one stage missing still runs."""
    base = {'stages': [{'name': 'a'}, {'name': 'b'}]}
    T._deep_merge(base, {'stages': [{'name': 'a'}]})
    assert base['stages'] == [{'name': 'a'}]


def test_stage_patch_returns_every_stage():
    stages = [{'name': 'one', 'loss_coeffs': {'bwd': {'mle': 1.0}}},
              {'name': 'two', 'loss_coeffs': {'fwd': {'tb': 1.0}}}]
    out = T._stage_patch(stages, 'one', 'bwd', 'mle', 2.0)
    assert len(out) == 2, 'a stage was dropped by the perturbation patch'
    assert out[0]['loss_coeffs']['bwd']['mle'] == 2.0
    assert stages[0]['loss_coeffs']['bwd']['mle'] == 1.0, 'input was mutated'


# ------------------------------------------------------------ the LR pin ----

_SENSORLESS = {
    'lr_policy': 'auto', 'lr_back': 'auto', 'lr_replay': 'auto',
    'lr_fused': 'auto',
    'adaptive_lr': {'seed_lr': 1.25e-4},
    'protocol': 'p',
    'protocols': {'p': {'stages': [{'name': 's', 'train_mode': 'bwd'}]}},
}


def test_pin_translates_a_sensorless_auto_into_the_rate_it_trained_at():
    cfg = copy.deepcopy(_SENSORLESS)
    notes = T.pin_auto_lr_without_sensor(cfg)
    assert notes and notes[0].startswith('REPAIR:')
    assert cfg['lr_policy'] == pytest.approx(1.25e-4)
    assert cfg['lr_fused'] == pytest.approx(1.25e-4)


def test_pin_is_inert_when_a_stage_declares_a_sensor():
    """It must fire ONLY on the config that cannot load. Pinning a config whose
    stages declare sensors would silently disable the servo -- turning the
    repair into the largest behaviour change in the comparison."""
    cfg = copy.deepcopy(_SENSORLESS)
    cfg['protocols']['p']['stages'][0]['lr_sensor'] = {'kind': 'hyper', 'beta': 0.1}
    assert T.pin_auto_lr_without_sensor(cfg) == []
    assert cfg['lr_policy'] == 'auto'


def test_pin_is_inert_when_the_rates_are_already_explicit():
    cfg = copy.deepcopy(_SENSORLESS)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3e-4
    assert T.pin_auto_lr_without_sensor(cfg) == []


def test_pin_refuses_rather_than_inventing_a_rate():
    """No seed to pin to means what the config trained at cannot be
    reconstructed. Picking a plausible number produces a config that loads and
    trains on a value nobody chose -- the same reason `config_state.migrate`
    reports judgment cases instead of guessing them."""
    cfg = copy.deepcopy(_SENSORLESS)
    cfg['adaptive_lr']['seed_lr'] = None
    with pytest.raises(ValueError, match='no number to pin'):
        T.pin_auto_lr_without_sensor(cfg)


# -------------------------------------------------------------- assembly ----

def test_the_assembled_canonical_config_carries_no_retired_key():
    """`build_config` raises on one, so reaching the assertion is the test. The
    check exists because generating a config is not loading it: retired keys
    fire on PRESENCE, and every layer above the base file can introduce one."""
    cfg, prov = T.build_config('configs/mk_dev.yaml', steps=30)
    assert T._retired_paths_in(cfg) == []
    assert cfg['epochs'] == T.epochs_for_steps(0, 30)
    assert cfg['energy_function'] == 'latent_gaussian'
    assert cfg['continue_from_checkpoint'] is False
    assert cfg['checkpoint_read_only'] is True
    assert prov['notes'], 'the assembly departs from the file on disk and must say so'


def test_the_assembled_config_preflights():
    """The instrument's own smoke test. If this fails, every run fails ~15 s in
    with a stack trace instead of a config error."""
    import json

    import utils
    cfg, _ = T.build_config('configs/mk_dev.yaml', steps=30)
    args = utils.dict2namespace(json.loads(json.dumps(cfg)))
    utils.resolve_derived_config(utils.preflight_config(args))


def test_eval_is_pinned_to_integrator_T():
    """`eval_T != integrator.T` is refused at load. The harness overrides both
    from different layers, so the agreement is asserted rather than assumed."""
    cfg, _ = T.build_config('configs/mk_dev.yaml', steps=30)
    assert cfg['eval_T'] == cfg['integrator']['T']


def test_the_problem_overlay_is_read_from_problems_yaml():
    ov = T.problem_overlay('latent_gaussian')
    assert ov['energy_function'] == 'latent_gaussian'
    assert 'description' not in ov, 'prose would be injected as a config key'


def test_the_problems_yaml_gap_is_declared_rather_than_patched_quietly():
    """`problems.yaml` cannot run `latent_gaussian` as it stands: `prior_path`
    is null and `init_prior_dataset` torch.loads it unconditionally, and
    `analyze_kwargs` is empty so the analytic target has no centre or width. The
    harness fills both, and the fill is REPORTED -- a gap patched invisibly is a
    gap that never gets closed."""
    raw = T.problem_overlay('latent_gaussian')
    assert raw['prior_path'] is None
    assert raw['analyze_kwargs'] == {}
    fill, notes = T.problem_gap_fill('latent_gaussian')
    assert fill['prior_path'] and fill['energy_config']['analyze_kwargs']['c']
    assert len(notes) == 2
    assert math.isclose(fill['energy_config']['analyze_kwargs']['width'], 0.1)
