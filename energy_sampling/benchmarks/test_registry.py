"""
EVERY INVARIANT IS TESTED IN BOTH DIRECTIONS.

A validator test that only checks the shipped registry passes proves the registry
is currently well-formed and proves nothing about the validator. Each rule below
therefore comes in a pair: the registry satisfies it, and a deliberately mutated
copy is REQUIRED to raise. This project has shipped a `check()` that recorded
failures without raising and a battery whose tests all exercised a copy of the
code the product never called; the pairing is what stops that.

No GPU, no data drive, no torch. Marked `fast`.
"""
import copy

import pytest
import yaml

from benchmarks import registry as R

pytestmark = pytest.mark.fast


@pytest.fixture(scope='module')
def reg():
    return R.load()


@pytest.fixture
def mut(reg):
    """A deep copy to break. Never mutate the module-scoped registry."""
    return copy.deepcopy(reg)


def _must_raise(r, fragment):
    with pytest.raises(R.RegistryError) as e:
        R.validate(r)
    assert fragment in str(e.value), f'raised, but not for the expected reason: {e.value}'


# ------------------------------------------------------------ shipped state --

def test_registry_loads_and_validates(reg):
    assert reg['schema_version'] == 1
    assert reg['benchmarks'] and reg['suites']


def test_yaml_is_parseable_without_the_loader():
    """The file must be plain data, so a human or another tool can read it."""
    with open(R.REGISTRY_PATH, encoding='utf-8') as f:
        assert isinstance(yaml.safe_load(f), dict)


def test_every_declared_profile_is_covered(reg):
    """
    The four branch profiles, both conditioning modes, and all four energies. These
    differ materially and the plan forbids assuming any of them stands in for
    another, so absence is a spec defect rather than a gap to fill later.
    """
    modes = {b['training_mode']['train_mode'] for b in reg['benchmarks']}
    assert modes == {'bwd', 'fused'}, modes
    branches = {tuple(sorted(b['training_mode']['branches'])) for b in reg['benchmarks']}
    for want in (('bwd',), ('fwd',), ('replay',), ('bwd', 'fwd', 'replay')):
        assert want in branches, f'no benchmark isolates {want}'
    conditioning = {b['workload']['conditioning'] for b in reg['benchmarks']}
    assert 'unconditional' in conditioning
    assert any(c.startswith('conditional') for c in conditioning)
    energies = {b['workload']['energy_function'] for b in reg['benchmarks']}
    for want in ('elj', 'uma', 'mace'):
        assert want in energies, f'no benchmark on {want}'
    assert energies & {'latent_gaussian', 'latent_multiharmonic'}, 'no toy benchmark'


def test_bwd_dataset_declares_the_energy_absent(reg):
    """
    `train_prior` is bwd/dataset MLE and NEVER CALLS THE ENERGY. A benchmark that
    reported `energy/ms_per_sample` here would be reading a key that is absent,
    and absent is not zero.
    """
    b = R.benchmark('elj-bwd-dataset-uncond', reg)
    assert 'energy/ms_per_sample' in b['metrics']['unusable']
    assert any('ABSENT' in a for a in b['liveness'])


def test_mlip_benchmarks_refuse_an_exact_bar(reg):
    for bid in ('uma-fused-uncond', 'mace-fused-uncond'):
        c = R.benchmark(bid, reg)['correctness']
        assert c['exactness'] == 'floor'
        assert c['reference'] == 'control_comparison'


def test_only_the_analytic_toy_claims_exactness(reg):
    exact = [b['id'] for b in reg['benchmarks'] if b['correctness']['exactness'] == 'exact']
    assert exact == ['toy-latentgauss-fused-uncond'], exact


def test_a100_only_benchmarks_are_absent_from_local_dev(reg):
    local = set(reg['suites']['local-dev']['benchmarks'])
    for b in reg['benchmarks']:
        if b['hardware']['local_adequate'] is False:
            assert b['id'] not in local


def test_the_standard_a100_suite_exists_and_is_named(reg):
    """'Rerun the standard A100 throughput suite' has to resolve to a list."""
    s = R.suite('a100-throughput', reg)
    assert [b['id'] for b in s]
    assert all(b['hardware']['class'] in ('a100', 'both') for b in s)


def test_no_floor_is_measured_yet(reg):
    """
    Guards the honest state of the deliverable. This test is EXPECTED TO FAIL and be
    updated once floors are measured -- that failure is the reminder that the
    numbers arrived. It must never be deleted to make the suite green.
    """
    unmeasured = [b['id'] for b in reg['benchmarks'] if b['noise_floor']['measured'] is None]
    assert len(unmeasured) == len(reg['benchmarks']), (
        f'a floor has been measured for {set(b["id"] for b in reg["benchmarks"]) - set(unmeasured)}; '
        f'record it and update this test')


# ----------------------------------------------------------------- helpers --

def test_resolved_overrides_are_deep_merged(reg):
    ov = R.resolved_overrides('toy-latentgauss-fused-uncond', reg)
    assert ov['z_calibration']['enabled'] is False      # from defaults
    assert ov['controller']['refresh_every'] == 1000000  # from the benchmark
    assert ov['checkpoint_read_only'] is True


def test_epochs_is_absolute_not_a_count(reg):
    """The trap: `epochs` bounds `trange(init_step, epochs + 1)`."""
    assert R.epochs_for('elj-fused-uncond', 0, reg) == 500
    assert R.epochs_for('elj-fused-uncond', 6680, reg) == 7180


def test_relative_span_is_a_pure_function():
    assert R.relative_span([10.0, 10.0]) == 0.0
    assert R.relative_span([9.0, 10.0, 11.0]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        R.relative_span([1.0])


def test_exceeds_floor_is_symmetric_and_has_no_denominator():
    assert R.exceeds_floor(100.0, 110.0, 0.05)
    assert R.exceeds_floor(110.0, 100.0, 0.05)
    assert not R.exceeds_floor(100.0, 102.0, 0.05)
    assert not R.exceeds_floor(102.0, 100.0, 0.05)


def test_floor_for_refuses_an_unmeasured_floor(reg):
    with pytest.raises(R.RegistryError) as e:
        R.floor_for('elj-fused-uncond', 'samples_per_sec', reg)
    assert 'has NOT been measured' in str(e.value)


def test_score_repeats_drops_incomplete_runs_instead_of_averaging(reg):
    reps = [
        {'completed': True, 'metrics': {'train_step_time': 1.00, 'samples_per_sec': 1000.0,
                                        'energy/frac_of_step': 0.5}, 'catastrophes': {}},
        {'completed': True, 'metrics': {'train_step_time': 1.02, 'samples_per_sec': 980.0,
                                        'energy/frac_of_step': 0.5}, 'catastrophes': {}},
        {'completed': True, 'metrics': {'train_step_time': 1.04, 'samples_per_sec': 960.0,
                                        'energy/frac_of_step': 0.5}, 'catastrophes': {}},
        # OOMed at step 12: fast, and describing different work.
        {'completed': False, 'metrics': {'train_step_time': 0.20, 'samples_per_sec': 5000.0,
                                         'energy/frac_of_step': 0.5},
         'catastrophes': {'batch/oom_events': 1}},
    ]
    out = R.score_repeats('elj-fused-uncond', reps, reg)
    assert out['usable'] == 3 and out['dropped_incomplete'] == 1
    assert out['catastrophes'] == {'batch/oom_events': 1}
    assert out['per_metric']['train_step_time']['median'] == pytest.approx(1.02)
    # The dropped run would have moved the median to 1.01 and tripled the span.
    assert out['per_metric']['train_step_time']['relative_span'] == pytest.approx(0.0392, abs=1e-3)


def test_catastrophes_are_summed_across_repeats_never_averaged(reg):
    reps = [{'completed': True, 'metrics': {}, 'catastrophes': {'batch/oom_events': 0}}
            for _ in range(5)]
    reps[2]['catastrophes'] = {'batch/oom_events': 7}
    out = R.score_repeats('elj-fused-uncond', reps, reg)
    assert out['catastrophes']['batch/oom_events'] == 7, (
        'a count that is excellent on four repeats and detonates on the fifth must '
        'survive as 7, not become 1.4')


# ------------------------------------------- negative controls, one per rule --

def test_rejects_a_primary_metric_that_is_a_ratio_to_a_reference(mut):
    mut['metrics']['cost']['speedup_vs_baseline'] = {'source': 'nowhere'}
    mut['benchmarks'][0]['metrics']['primary'].append('speedup_vs_baseline')
    _must_raise(mut, 'no headline metric may depend on a reference rate')


def test_rejects_an_uncatalogued_metric(mut):
    mut['benchmarks'][0]['metrics']['primary'].append('made/up_metric')
    _must_raise(mut, 'not in the metrics catalogue')


def test_rejects_a_metric_that_is_both_reportable_and_unusable(mut):
    b = mut['benchmarks'][0]
    b['metrics']['unusable'][b['metrics']['primary'][0]] = 'contradictory'
    _must_raise(mut, 'both as reportable and as unusable')


def test_rejects_an_unusable_metric_with_no_reason(mut):
    mut['benchmarks'][0]['metrics']['unusable']['energy/calls'] = ''
    _must_raise(mut, 'carries no reason')


def test_rejects_empty_catastrophes(mut):
    mut['benchmarks'][0]['metrics']['catastrophes'] = []
    _must_raise(mut, 'Catastrophes are counted')


def test_rejects_empty_liveness(mut):
    mut['benchmarks'][0]['liveness'] = []
    _must_raise(mut, 'liveness is empty')


def test_rejects_too_few_floor_repeats(mut):
    mut['benchmarks'][0]['noise_floor']['repeats'] = 2
    _must_raise(mut, 'noise_floor.repeats must be >=')


def test_rejects_a_within_run_floor(mut):
    mut['benchmarks'][0]['noise_floor']['method'] = 'within_run_scatter'
    _must_raise(mut, 'must be repeat_launch')


def test_rejects_a_floor_that_misses_a_primary_metric(mut):
    b = mut['benchmarks'][0]
    b['noise_floor']['measured'] = {
        'date': '2026-08-20', 'host': 'local', 'repeats': 5,
        'per_metric': {b['metrics']['primary'][0]: 0.03}}
    _must_raise(mut, 'floor does not cover primary metrics')


def test_rejects_a_floor_built_from_fewer_repeats_than_declared(mut):
    b = mut['benchmarks'][0]
    b['noise_floor']['measured'] = {
        'date': '2026-08-20', 'host': 'local', 'repeats': 2,
        'per_metric': {k: 0.03 for k in b['metrics']['primary']}}
    _must_raise(mut, 'fewer than the')


def test_rejects_an_exact_bar_on_a_non_analytic_reference(mut):
    for entry in mut['benchmarks']:
        if entry['id'] == 'uma-fused-uncond':
            entry['correctness']['exactness'] = 'exact'
    _must_raise(mut, 'only defensible against a closed form')


def test_rejects_a_control_comparison_with_no_harness(mut):
    for entry in mut['benchmarks']:
        if entry['id'] == 'uma-fused-uncond':
            entry['correctness']['gate'] = None
    _must_raise(mut, 'must name the harness')


def test_rejects_util_policy_on_a_run_shorter_than_its_window(mut):
    b = mut['benchmarks'][0]
    b['metrics']['unusable'].pop('gpu/util_policy', None)
    b['metrics']['primary'].append('gpu/util_policy')
    _must_raise(mut, 'the window would never fill')


def test_rejects_a_fused_window_that_is_not_whole_refresh_periods(mut):
    for entry in mut['benchmarks']:
        if entry['training_mode']['train_mode'] == 'fused':
            entry['work']['measure_steps'] = 405
            break
    _must_raise(mut, 'force-refresh period')


def test_rejects_a_half_pinned_batch(mut):
    """max_batch_size and batch_size are independent hard stops."""
    for entry in mut['benchmarks']:
        if entry['work'].get('batch_size') is not None and entry['work']['pin_batch']:
            entry['overrides']['max_batch_size'] = entry['work']['batch_size'] * 4
            break
    _must_raise(mut, 'they are independent')


def test_rejects_a_rung_benchmark_that_pins_one_batch(mut):
    for entry in mut['benchmarks']:
        if entry['work']['kind'] == 'fixed_steps_per_rung':
            entry['overrides']['batch_size'] = 1000
            break
    _must_raise(mut, 'must not fix')


def test_rejects_a_rung_benchmark_with_no_rungs(mut):
    for entry in mut['benchmarks']:
        if entry['work']['kind'] == 'fixed_steps_per_rung':
            entry['work']['batch_rungs'] = []
            break
    _must_raise(mut, 'batch_rungs is empty')


def test_rejects_an_epochs_formula_that_is_a_count(mut):
    mut['benchmarks'][0]['work']['epochs_formula'] = 'warmup_steps + measure_steps'
    _must_raise(mut, 'ABSOLUTE step index')


def test_rejects_zero_warmup(mut):
    mut['benchmarks'][0]['work']['warmup_steps'] = 0
    _must_raise(mut, 'warmup_steps must be >= 1')


def test_rejects_defaults_that_leave_z_calibration_on(mut):
    mut['defaults']['overrides']['z_calibration']['enabled'] = True
    _must_raise(mut, 'must be False')


def test_rejects_defaults_that_leave_the_runaway_guard_armed(mut):
    mut['defaults']['overrides']['max_step_seconds'] = 60
    _must_raise(mut, 'max_step_seconds must be 0')


def test_rejects_defaults_that_permit_checkpoint_writes(mut):
    mut['defaults']['overrides']['checkpoint_read_only'] = False
    _must_raise(mut, 'checkpoint_read_only must be True')


def test_rejects_an_a100_only_benchmark_claiming_local_adequacy(mut):
    for entry in mut['benchmarks']:
        if entry['hardware']['class'] == 'a100':
            entry['hardware']['local_adequate'] = True
            break
    _must_raise(mut, 'local_adequate is not False')


def test_rejects_a_hardware_requirement_with_no_reason(mut):
    mut['benchmarks'][0]['hardware']['reason'] = '  '
    _must_raise(mut, 'hardware.reason is empty')


def test_rejects_a_duplicate_benchmark_id(mut):
    mut['benchmarks'].append(copy.deepcopy(mut['benchmarks'][0]))
    _must_raise(mut, 'duplicate benchmark id')


def test_rejects_a_suite_naming_an_unknown_benchmark(mut):
    mut['suites']['local-dev']['benchmarks'].append('does-not-exist')
    _must_raise(mut, 'names unknown benchmarks')


def test_rejects_a_benchmark_in_no_suite(mut):
    for s in mut['suites'].values():
        s['benchmarks'] = [i for i in s['benchmarks'] if i != 'elj-eval-cost']
    mut['suites'] = {k: v for k, v in mut['suites'].items() if v['benchmarks']}
    _must_raise(mut, 'benchmarks in no suite')


def test_rejects_an_a100_only_benchmark_added_to_local_dev(mut):
    mut['suites']['local-dev']['benchmarks'].append('a100-batch-scaling-elj')
    _must_raise(mut, 'declares local_adequate false')
