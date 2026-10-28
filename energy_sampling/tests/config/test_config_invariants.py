"""
Tests for config_invariants.

EVERY RULE IS MUTATION-TESTED. A rule is asserted to pass on the canonical config
AND to fire on a config carrying exactly the fault it describes. A rule that has
never fired has not been tested -- it may be reading a key that does not exist,
or abstaining on every input, and it would pass this suite either way.

Run: python -m pytest test_config_invariants.py -q
"""

import copy
from pathlib import Path

import pytest
import yaml

import config_invariants as ci
from config_invariants import BASELINE, ERROR, RULES, check, errors
from protocol import Stage

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def canonical():
    # the resolved form (config_invariants.canonical_model_section), which the dotted mutations below address
    return ci.canonical_model_section(yaml.safe_load(CANONICAL.read_text(encoding='utf-8')))


def broken(canonical, **dotted):
    """A copy of the canonical config with dotted paths overwritten."""
    cfg = copy.deepcopy(canonical)
    for path, value in dotted.items():
        node = cfg
        parts = path.replace('__', '.').split('.')
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return cfg


# ---------------------------------------------------------------------------
# The canonical config satisfies everything
# ---------------------------------------------------------------------------

def test_canonical_config_has_no_errors(canonical):
    assert errors(canonical) == [], '\n'.join(str(v) for v in errors(canonical))


def test_canonical_config_has_no_baseline_departures(canonical):
    """Not a hard requirement in general -- a run may depart deliberately -- but
    the CANONICAL config is the project's stated default and should meet its own
    baselines. A departure here means the baseline or the config is stale."""
    assert check(canonical) == [], '\n'.join(str(v) for v in check(canonical))


# ---------------------------------------------------------------------------
# ...and every rule fires on the fault it describes
# ---------------------------------------------------------------------------

def _fires(cfg, rule_name, severity=ERROR):
    vs = [v for v in check(cfg) if v.rule == rule_name and v.severity == severity]
    return len(vs) > 0


def test_util_target_must_be_actuable(canonical):
    # each control disarms the ladder EXPLICITLY, so the assertions hold whatever
    # grow/max values the canonical config ships (it now ships armed: 60/true/20000)
    assert _fires(broken(canonical, batch_util_target=0.6, grow_batch_size=False),
                  'util_target_actuable')
    # growth on but no headroom above the base: the ladder has one rung
    assert _fires(broken(canonical, batch_util_target=0.6, grow_batch_size=True,
                         max_batch_size=1000), 'util_target_actuable')
    # THE UNIT GATE (state 9). A leftover percent value must fail rather than
    # ask for 6000% occupancy and call every batch INFEASIBLE. This is the
    # clause the old (0, 100] range could not express: under it, 60 passed.
    assert _fires(broken(canonical, batch_util_target=60, grow_batch_size=True,
                         max_batch_size=20000), 'util_target_actuable')
    assert _fires(broken(canonical, batch_util_target=140, grow_batch_size=True,
                         max_batch_size=20000), 'util_target_actuable')
    # actuable: growth on, headroom above the base, a real fraction
    assert not _fires(broken(canonical, batch_util_target=0.6, grow_batch_size=True,
                             max_batch_size=20000), 'util_target_actuable')
    # off is the shipping default and clean
    assert not _fires(broken(canonical, batch_util_target=0), 'util_target_actuable')


def test_figs_period_not_a_multiple_never_fires_on_the_conformer_route(canonical):
    conformer = dict(energy_function='conformer_torsions')
    assert _fires(broken(canonical, figs_period=501, **conformer), 'figs_period_fires')
    assert not _fires(broken(canonical, figs_period=1000, **conformer), 'figs_period_fires')
    # off the conformer route figs_period is retired, not checked for coincidence
    assert not _fires(broken(canonical, figs_period=501), 'figs_period_fires')


def test_step_cadence_keys_are_retired_on_the_crystal_route(canonical):
    assert _fires(broken(canonical, figs_period=500), 'step_cadence_keys_are_retired')
    assert _fires(broken(canonical, xcond_eval__period=2500), 'step_cadence_keys_are_retired')
    # presence, not value: a null still names a key nothing reads
    assert _fires(broken(canonical, figs_period=None), 'step_cadence_keys_are_retired')
    assert ci.retired_step_cadence_keys(broken(canonical, figs_period=500,
                                               xcond_eval__period=2500)) == \
        ['figs_period', 'xcond_eval.period']
    # the conformer route still reads figs_period
    assert not _fires(broken(canonical, figs_period=500, xcond_eval__period=2500,
                             energy_function='conformer_torsions'),
                      'step_cadence_keys_are_retired')


def test_batch_ceiling_below_floor(canonical):
    assert _fires(broken(canonical, max_batch_size=500), 'batch_ceiling_above_floor')


def test_dplr_rho_at_one(canonical):
    assert _fires(broken(canonical, model__dplr_rho_max=1.0), 'dplr_is_well_formed')
    assert _fires(broken(canonical, model__dplr_rho_max=-0.1), 'dplr_is_well_formed')


def test_dplr_unmasked_with_periodic_dims(canonical):
    assert canonical['model']['periodic_centroids'] is True, 'fixture assumption'
    assert _fires(broken(canonical, model__dplr_mask_angular=False),
                  'dplr_is_well_formed')


def test_dplr_unmasked_on_a_crystal_without_periodic_centroids(canonical):
    """The second sufficient condition: a crystal energy function carries cell
    angles, so ang_dim > 0 even with periodic_centroids off. Without this branch
    the rule misses 20 configs in the corpus that die at model construction."""
    cfg = broken(canonical, model__periodic_centroids=False,
                 model__dplr_mask_angular=False, energy_function='elj')
    assert _fires(cfg, 'dplr_is_well_formed')


def test_dplr_angular_rule_abstains_on_a_toy(canonical):
    """A toy with no angular dims and no periodic centroids may legitimately run
    DPLR unmasked. Firing here would be a false positive on a valid config."""
    cfg = broken(canonical, model__periodic_centroids=False,
                 model__dplr_mask_angular=False, energy_function='latent_multiharmonic')
    assert not _fires(cfg, 'dplr_is_well_formed')


def test_dplr_angular_rule_abstains_on_an_unknown_energy_function(canonical):
    """An unrecognised energy function must abstain rather than assume. A rule
    that guesses is worse than one that admits it does not know."""
    cfg = broken(canonical, model__periodic_centroids=False,
                 model__dplr_mask_angular=False, energy_function='some_new_thing')
    assert not _fires(cfg, 'dplr_is_well_formed')


def test_dplr_rules_abstain_when_dplr_is_off(canonical):
    """rank 0 disables DPLR entirely, so its sub-rules must not fire on values
    that are irrelevant. A rule that fires on a disabled feature trains people to
    ignore it."""
    cfg = broken(canonical, model__dplr_rank=0, model__dplr_mask_angular=False,
                 model__dplr_rho_max=5.0)
    assert not _fires(cfg, 'dplr_is_well_formed')


def test_deactivate_threshold_at_a_third(canonical):
    assert _fires(broken(canonical, controller__deactivate_threshold=0.34),
                  'deactivate_threshold_is_sane')


def test_deactivate_threshold_checked_per_stage(canonical):
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocols']['unconditional_tb']['stages'] if 'deactivate_threshold' in s)
    stage['deactivate_threshold'] = 0.5
    assert _fires(cfg, 'deactivate_threshold_is_sane')


def test_pinned_frac_disagreeing_with_fracs(canonical):
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocols']['unconditional_tb']['stages']
                 if (s.get('balance') or {}).get('pinned'))
    mode = next(iter(stage['balance']['pinned']))
    stage['balance']['pinned'][mode] = stage['fracs'][mode] + 0.1
    assert _fires(cfg, 'pinned_frac_matches_fracs')


def test_a_selector_naming_a_missing_protocol_is_an_error(canonical):
    """THE hazard this restructure created, and the reason the gate exists.

    Every stage-scoped check reads the ACTIVE stage list, so a selector pointing
    at a protocol that is not defined resolves to zero stages -- and a rule with
    nothing to iterate reports nothing wrong. Measured while making the change:
    with the selector mistyped, the auto-LR gate stopped firing on a config it
    had rejected moments before. One word used to disarm every stage check."""
    assert _fires(broken(canonical, protocol='does_not_exist'),
                  'protocol_selector_resolves')


def test_an_absent_or_non_string_selector_is_an_error(canonical):
    cfg = copy.deepcopy(canonical); cfg.pop('protocol')
    assert _fires(cfg, 'protocol_selector_resolves')
    assert _fires(broken(canonical, protocol={'stages': []}),
                  'protocol_selector_resolves')


def test_a_selected_protocol_with_no_stages_is_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['unconditional_tb'] = {'stages': []}
    assert _fires(cfg, 'protocol_selector_resolves')


def test_the_selector_rule_abstains_on_a_fragment(canonical):
    """A config with no protocol at all is an overlay, not a broken run config.
    Firing there would flag every fragment in the tree."""
    assert not _fires({'batch_size': 100}, 'protocol_selector_resolves')


def test_a_disarmed_stage_check_is_caught_by_the_selector_rule(canonical):
    """The two halves together: with a bad selector the stage-scoped rule goes
    quiet, and the selector rule is what keeps the config from passing.

    The rule is first shown firing on the fault under a valid selector, because
    `_fires` matches by name and a name no rule emits is quiet on every config.
    The fault is one `protocol.Stage` parses, so `every_protocol_parses`, which
    reads every protocol whether selected or not, does not reject it either."""
    rule = 'exit_patience_is_reachable'         # iterates active_stages(cfg)
    # more writes of its metric than the run has steps: unreachable at any cadence
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 0, patience=canonical['epochs'] + 1)
    assert _fires(cfg, rule), 'fixture assumption: the stage rule sees the fault'
    cfg['protocol'] = 'does_not_exist'
    assert not _fires(cfg, rule), 'fixture assumption: the stage rule DOES go quiet'
    assert {v.rule for v in errors(cfg)} == {'protocol_selector_resolves'}, \
        'but the config must still be rejected, and by the selector rule alone'


def test_an_inactive_protocol_that_cannot_parse_is_caught(canonical):
    """The point of checking the whole library: an inactive protocol is otherwise
    unexamined until it is SELECTED, and then it fails at load -- on the switch,
    which is the worst moment. Switching route is meant to be one word."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['conditional_vargrad'] = {
        'stages': [{'name': 'train_prior', 'train_mode': 'bwd',
                    'lr_sensor': {'kind': 'ray', 'period': 7}}]}   # period must be a multiple of 10
    assert _fires(cfg, 'every_protocol_parses')


def test_an_inactive_protocol_with_an_unknown_key_is_caught(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['conditional_vargrad'] = {
        'stages': [{'name': 'x', 'train_mode': 'fused', 'not_a_stage_key': 1}]}
    assert _fires(cfg, 'every_protocol_parses')


def test_duplicate_stage_names_in_any_protocol_are_caught(canonical):
    """The trainer identifies the live stage BY NAME, so a duplicate makes the
    run's position ambiguous."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['conditional_vargrad'] = {
        'stages': [{'name': 'same', 'train_mode': 'bwd'},
                   {'name': 'same', 'train_mode': 'fused'}]}
    assert _fires(cfg, 'every_protocol_parses')


def test_a_well_formed_inactive_protocol_passes(canonical):
    """Mutation for the three above: a valid second protocol must NOT fire, or
    the rule would block the very thing it exists to make safe."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['conditional_vargrad'] = {
        'stages': [{'name': 'train_prior', 'train_mode': 'bwd',
                    'lr_sensor': {'kind': 'none'}},
                   {'name': 'var_conditioning', 'train_mode': 'fused',
                    'lr_sensor': {'kind': 'none'}}]}
    assert not _fires(cfg, 'every_protocol_parses')
    assert errors(cfg) == [], [str(e) for e in errors(cfg)]


def test_auto_lr_with_no_lr_control_block_is_an_error(canonical):
    """`auto` claims something owns the rate. With no `lr_control` block nothing
    sets the scale, so the run trains at the seed for its whole life while the
    config reads as adaptive.

    A RUN-LEVEL CHECK: `lr_control` is not declared per stage."""
    cfg = copy.deepcopy(canonical)
    cfg.pop('lr_control', None)
    assert _fires(cfg, 'auto_lr_requires_lr_control')


def test_auto_lr_with_an_unknown_mode_is_also_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['lr_control']['mode'] = 'adaptive'
    assert _fires(cfg, 'auto_lr_requires_lr_control')


def test_an_absent_mode_reads_as_fixed(canonical):
    """LRController defaults `mode` to 'fixed', so both rules judge absence as it."""
    cfg = copy.deepcopy(canonical)
    cfg['lr_control'].pop('mode', None)
    assert not _fires(cfg, 'auto_lr_requires_lr_control')
    assert not _fires(cfg, 'lr_control_is_well_formed')


@pytest.mark.parametrize('mode', ['bracket', 'adaptive'])
def test_a_mode_other_than_fixed_is_an_error(canonical, mode):
    """`bracket` was deleted 2026-10-28; LRController raises on it."""
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'lr_control_is_well_formed')
    cfg['lr_control']['mode'] = mode
    assert _fires(cfg, 'lr_control_is_well_formed')
    assert _fires(cfg, 'auto_lr_requires_lr_control')


def test_a_control_arm_is_expressible_as_fixed_mode(canonical):
    cfg = copy.deepcopy(canonical)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3.0e-4
    cfg['lr_control'] = {'mode': 'fixed',
                         'schedule': [{'to': 1.0, 'steps': 0, 'kind': 'hold'}]}
    assert not _fires(cfg, 'lr_control_is_well_formed')
    assert not _fires(cfg, 'auto_lr_requires_lr_control')


def test_fixed_mode_without_a_schedule_is_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['lr_control'] = {'mode': 'fixed'}
    assert _fires(cfg, 'lr_control_is_well_formed')
    # on the conformer route the keys `schedule` replaced stand in for it
    cfg['energy_function'] = 'conformer_torsions'
    assert _fires(cfg, 'lr_control_is_well_formed')
    cfg['lr_control'].update(burn_in_steps=3000, burn_in_scale=0.05, fixed_scale=0.2)
    assert not _fires(cfg, 'lr_control_is_well_formed')
    cfg['lr_control']['schedule'] = [{'to': 1.0, 'steps': 0, 'kind': 'hold'}]
    assert _fires(cfg, 'lr_control_is_well_formed'), 'schedule beside the replaced keys'


@pytest.mark.parametrize('segment,detail', [
    ({'to': 0.2, 'steps': 10, 'kind': 'ramp'}, "kind is 'ramp'"),
    ({'to': 0.0, 'steps': 10, 'kind': 'linear'}, 'to is 0.0'),
    ({'to': -0.2, 'steps': 10, 'kind': 'cosine'}, 'to is -0.2'),
    ({'to': 0.2, 'steps': 1.5, 'kind': 'exp'}, 'steps is 1.5'),
])
def test_a_malformed_schedule_segment_is_an_error(canonical, segment, detail):
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'lr_control_is_well_formed')
    cfg['lr_control']['schedule'] = [cfg['lr_control']['schedule'][0], segment]
    (v,) = [v for v in check(cfg) if v.rule == 'lr_control_is_well_formed']
    assert detail in v.detail


def test_a_first_segment_other_than_a_hold_is_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['lr_control']['schedule'][0]['kind'] = 'linear'
    assert _fires(cfg, 'lr_control_is_well_formed')


def test_an_open_ended_move_needs_every_stage_using_it_to_have_a_cap(canonical):
    """`steps: null` runs a segment to the stage's max_steps; a non-hold one on a
    stage with no cap and no lr_schedule of its own is refused."""
    cfg = copy.deepcopy(canonical)
    cfg['lr_control']['schedule'].append({'to': 0.01, 'steps': None, 'kind': 'cosine'})
    (v, *_) = [v for v in check(cfg) if v.rule == 'lr_control_is_well_formed']
    assert 'no max_steps' in v.detail
    for proto in cfg['protocols'].values():
        for st in proto['stages']:
            st.setdefault('max_steps', None)
            if st['max_steps'] is None:
                st['lr_schedule'] = [{'to': 0.05, 'steps': 0, 'kind': 'hold'}]
    assert not _fires(cfg, 'lr_control_is_well_formed')
    # an open-ended hold needs no cap
    cfg = copy.deepcopy(canonical)
    cfg['lr_control']['schedule'].append({'to': 0.2, 'steps': None, 'kind': 'hold'})
    assert not _fires(cfg, 'lr_control_is_well_formed')


def test_a_stage_lr_schedule_is_judged_against_its_own_cap(canonical):
    cfg = copy.deepcopy(canonical)
    stage = cfg['protocols'][cfg['protocol']]['stages'][-1]
    stage['lr_schedule'] = [{'to': 0.05, 'steps': 10, 'kind': 'hold'},
                            {'to': 0.01, 'steps': None, 'kind': 'linear'}]
    stage.pop('max_steps', None)
    assert [v for v in check(cfg) if v.rule == 'every_protocol_parses'
            and 'lr_schedule' in v.detail and 'no max_steps' in v.detail]
    stage['max_steps'] = 1000
    stage['exit'] = None
    assert not [v for v in check(cfg) if v.rule == 'every_protocol_parses'
                and 'lr_schedule' in v.detail]


@pytest.mark.parametrize('key,value', [
    ('burn_in_steps', 3000), ('burn_in_scale', 0.05),
    ('promotion_ramp_steps', 1000), ('fixed_scale', 0.2)])
def test_a_replaced_burn_in_key_is_an_error_off_the_conformer_route(canonical, key, value):
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'lr_schedule_replaces_burn_in_keys')
    cfg['lr_control'][key] = value
    (v,) = [v for v in check(cfg) if v.rule == 'lr_schedule_replaces_burn_in_keys']
    assert v.severity == ERROR and key in v.detail and 'schedule' in v.detail
    cfg['energy_function'] = 'conformer_torsions'
    assert not _fires(cfg, 'lr_schedule_replaces_burn_in_keys')


@pytest.mark.parametrize('key,value', [
    ('candidate_scales', [0.05, 0.2, 0.8]), ('trial_steps', 150),
    ('safety_rungs', 1), ('repeat_every', 0), ('trial_settle_steps', 10),
    ('logz_detour_nats', 2.0), ('boundary_confirm_repeats', 1),
    ('boundary_densify', True), ('min_root_bias_correction', 0.9)])
def test_a_retired_bracket_key_is_an_error_off_the_conformer_route(canonical, key, value):
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'lr_bracket_keys_are_retired')
    cfg['lr_control'][key] = value
    (v,) = [v for v in check(cfg) if v.rule == 'lr_bracket_keys_are_retired']
    assert v.severity == ERROR and key in v.detail
    cfg['energy_function'] = 'conformer_torsions'
    assert not _fires(cfg, 'lr_bracket_keys_are_retired')


def test_auto_keys_survives_resolution_destroying_the_evidence(canonical):
    """THE bug this parameter exists for.

    `resolve_derived_config` overwrites the string `auto` with the seed float in
    place. A caller running after it sees four ordinary numbers, so re-deriving
    the auto set from the config finds none and the rule passes on exactly the
    configs it exists to reject -- silently. Such a caller must pass the
    managed-key list it already computed."""
    resolved = copy.deepcopy(canonical)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        resolved[k] = 1.25e-4                       # as resolution leaves it
    resolved.pop('lr_control', None)

    # derived from the config: the evidence is gone, so nothing fires
    assert ci.auto_lr_requires_lr_control(resolved) == []
    # told what was managed: fires correctly
    told = ci.auto_lr_requires_lr_control(
        resolved, auto_keys=['lr_policy', 'lr_back', 'lr_replay', 'lr_fused'])
    assert told and all(v.severity == ERROR for v in told)


def test_explicit_float_lrs_need_no_lr_control(canonical):
    """The other half of the rule: a float is a fixed rate that takes divergence
    handling and the max_lr rail only, so it has nothing to yield to and must not
    be flagged."""
    cfg = copy.deepcopy(canonical)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3.0e-4
    cfg.pop('lr_control', None)
    assert not _fires(cfg, 'auto_lr_requires_lr_control')


def test_no_config_can_make_ray_incoherent_any_more(canonical):
    """`ray_sensor_needs_a_coherent_stage` IS DELETED, and its absence is the
    claim.

    It has been wrong twice. First it required a fused replay-TB stage, because
    the probe drew from replay -- which is exactly what left phase 1 unmeasured
    across 19 runs. Rewritten 2026-08-22 to fire on a forward bank carrying a
    term the replay evaluator has no counterpart for, it then had one live case:
    `var_conditioning`'s `emp_z: 1.0`. The owner decision to exclude the Z
    sidecar from ALL LR control removed that case too, by zeroing those terms
    before scoring (lr_larder.LarderScorer.Z_SIDECAR_TERMS).

    Nothing is left for it to judge, and a rule that can never fire is worse
    than none -- it reads as coverage. This test replaces it: the stages that
    used to be refused must now load clean.
    """
    for stage_ix, sensor in ((0, {'kind': 'ray'}), (1, {'kind': 'ray'})):
        cfg = copy.deepcopy(canonical)
        cfg['protocols']['unconditional_tb']['stages'][stage_ix]['lr_sensor'] = sensor
        assert not any(v.rule.startswith('ray_sensor')
                       for v in check(cfg)), f'stage {stage_ix} refused'

    # ...including the one case the rewritten rule existed for
    cfg = copy.deepcopy(canonical)
    st = cfg['protocols']['unconditional_tb']['stages'][1]
    st['lr_sensor'] = {'kind': 'ray'}
    st.setdefault('loss_coeffs', {}).setdefault('fwd', {})['emp_z'] = 1.0
    assert not any(v.rule.startswith('ray_sensor') for v in check(cfg))


def test_hyper_is_refused_at_load(canonical):
    """`hyper` was removed 2026-09-20. A config still declaring it must FAIL --
    ignored, a dead kind reads as an armed sensor."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['unconditional_tb']['stages'][0]['lr_sensor'] = {'kind': 'hyper', 'beta': 0.05}
    assert _fires(cfg, 'every_protocol_parses')
    with pytest.raises(ValueError, match='2026-09-20'):
        Stage({'name': 'x', 'train_mode': 'bwd',
               'lr_sensor': {'kind': 'hyper', 'beta': 0.05}}, 0)


def test_periodic_centroids_with_two_space_groups_is_an_error(canonical):
    """The feature bakes a per-SG axis set into the model's expanded_dim, so two
    space groups would be intersected into a weaker or empty wrap, silently."""
    assert _fires(broken(canonical, space_groups=[2, 14]),
                  'periodic_centroids_needs_one_crystal_space_group')


def test_periodic_centroids_on_a_toy_is_an_error(canonical):
    """A toy has no cell to wrap."""
    assert _fires(broken(canonical, energy_function='latent_multiharmonic'),
                  'periodic_centroids_needs_one_crystal_space_group')


def test_the_rule_abstains_when_periodic_centroids_is_off(canonical):
    """Mutation: with the feature off, neither condition is a fault -- a toy with
    several space groups is an ordinary config."""
    cfg = broken(canonical, space_groups=[2, 14],
                 energy_function='latent_multiharmonic')
    cfg['model']['periodic_centroids'] = False
    assert not _fires(cfg, 'periodic_centroids_needs_one_crystal_space_group')


# ---------------------------------------------------------------------------
# NULL DATA PATHS. `configs/problems.yaml`'s `latent_gaussian` entry carried
# `prior_path: null` and `molecules_path: null` while declaring an anchor
# seed_source that reads the prior dataset. Arms generated from it validated
# clean and died at startup on `torch.load(None)`.
# ---------------------------------------------------------------------------

RULE_NULL_PATHS = 'loaded_data_paths_are_not_null'


@pytest.mark.parametrize('key', ['prior_path', 'molecules_path'])
def test_a_null_data_path_is_an_error(canonical, key):
    """Both are torch.loaded with no null branch, so both are fatal alone."""
    assert _fires(broken(canonical, **{key: None}), RULE_NULL_PATHS)


def test_the_null_path_rule_does_not_depend_on_the_anchor_seed_source(canonical):
    """THE REASON THE RULE IS NOT WRITTEN ON `seed_source`. Both loads are
    unconditional, so `generated` with a null prior_path is equally fatal -- a
    rule anchored on the seed source would have passed this config."""
    cfg = broken(canonical, prior_path=None)
    cfg['buffers']['anchor_buffer']['seed_source'] = 'generated'
    assert _fires(cfg, RULE_NULL_PATHS)


def test_the_seed_source_contradiction_is_reported_as_its_own_statement(canonical):
    """...and when the seed source IS `prior_dataset`, that is a second,
    independent thing wrong with the same config: it names a dataset the run was
    never given a file to build."""
    cfg = broken(canonical, prior_path=None)
    cfg['buffers']['anchor_buffer']['seed_source'] = 'prior_dataset'
    details = [v.detail for v in check(cfg) if v.rule == RULE_NULL_PATHS]
    assert len(details) == 2, details
    assert any('seed_source' in d for d in details)


def test_the_conformer_route_may_declare_null_data_paths(canonical):
    """`ConformerModeller` overrides both data-init methods -- `init_mol_dataset`
    branches on `if path:` and `init_prior_dataset` samples the fitted prior -- so
    a null path there is a declared configuration, not a defect. 21 of the 26
    configs in the corpus carrying one are this energy function."""
    cfg = broken(canonical, prior_path=None, molecules_path=None,
                 energy_function='conformer_torsions')
    assert not _fires(cfg, RULE_NULL_PATHS)


def test_an_absent_data_path_is_not_a_null_one(canonical):
    """Mutation guard on this module's absence policy: generation and migration
    both produce partial dicts, and `null` is an assertion while a missing key is
    not. A rule that conflated them would fire on every partial config."""
    cfg = broken(canonical, prior_path=None)
    del cfg['prior_path']
    del cfg['molecules_path']
    assert not _fires(cfg, RULE_NULL_PATHS)


def test_the_null_path_rule_abstains_without_an_energy_function(canonical):
    """Which loader runs is not provable from a config that does not name its
    energy function, and a rule that cannot prove it has nothing to say."""
    cfg = broken(canonical, prior_path=None)
    del cfg['energy_function']
    assert not _fires(cfg, RULE_NULL_PATHS)


# The regression itself -- that an arm generated from the `latent_gaussian`
# registry entry now passes -- lives in test_generate.py. It needs `generate`,
# which transitively imports torch, and this file is a KNOWN_CHEAP tier floor
# (tests/config/test_tiering.py): a test that keeps the fast lane's source scan
# happy while pulling torch behind it is how the lane stops being fast.


# ---------------------------------------------------------------------------
# Exit triggers -- the two dead-gate shapes from the 2026-08-16 audit
# (docs/design/next_battery.md 1.1a and 1.3)
# ---------------------------------------------------------------------------

#: the term these tests edit when the stage they target ships no exit block.
#: `fwd/logw_std_within` is the one metric MEASURED_METRIC_RANGES covers, so it is
#: the only one that can exercise the rule at all.
_LOGW_TERM = {'metric': 'fwd/logw_std_within', 'below': 6.0}


def _conditional_route_globals(cfg):
    """The one GLOBAL a switch onto conditional_vargrad must hand-edit for the
    route to start at all: its var_conditioning (pooled_source 'replay',
    condition_draw) refuses a prioritised replay draw, which mk_dev keeps on for
    the unconditional route. The route's other globals are tuning, not
    refusals (docs/module_protocol.md §2, second table), so they stay at
    mk_dev's values."""
    cfg['buffers']['replay_buffer']['prioritise']['enabled'] = False
    return cfg


def _exit_term(canonical, protocol, stage_index, term_index, **keys):
    """The canonical config with one exit term of one protocol edited in place,
    and that protocol selected. Real stages, real metrics -- a synthetic exit
    block would not prove the rule fires on a config anyone would write.

    A STAGE WITH NO EXIT BLOCK GETS ONE. `var_conditioning` used to ship
    `fwd/logw_std_within < 6.0`, an unreachable bar these tests were written
    against; it has since been deleted from the canonical config, because
    next_battery.md 1.1 concluded the stage is terminal by design. Indexing into
    an absent block just raises KeyError, which reads as a broken suite rather
    than as the config improving. The term is installed instead, so these tests
    keep testing the RULE -- which still has to fire on any config that declares
    such a bar -- and stop depending on the canonical config carrying a fault."""
    cfg = copy.deepcopy(canonical)
    cfg['protocol'] = protocol
    if protocol == 'conditional_vargrad':
        _conditional_route_globals(cfg)
    stage = cfg['protocols'][protocol]['stages'][stage_index]
    if not stage.get('exit'):
        assert term_index == 0, 'only term 0 can be installed into an empty exit block'
        stage['exit'] = [copy.deepcopy(_LOGW_TERM)]
    stage['exit'][term_index].update(keys)
    return cfg


def test_the_unreachable_exit_bar_fires_wherever_it_is_declared(canonical):
    """THE MEASURED CASE. `var_conditioning` used to declare
    `fwd/logw_std_within < 6.0`. Measured minimum over 4,348 ticks and 6 arms of
    qm9anchor_aug14 is 17.1 -- no arm came within 2.9x of it. next_battery.md 1.1
    concluded the block was VESTIGIAL (the stage is terminal by design) rather
    than mis-set, and it has since been deleted from the canonical config.

    THE RULE STILL HAS TO FIRE, which is why this test outlived the bar it was
    written for: the fault it catches is a property of any config that declares
    such a term, not of mk_dev. Deleting the test along with the bar would have
    left the next config free to reintroduce it silently.

    REPORTED, NOT REFUSED. 17.1 was measured on runs with five railed controls,
    and the configs written to unrail them are exactly the ones that should be
    allowed to aim under it. A rule built on evidence blocks the next
    experiment; this one must never be the reason a run will not start."""
    cfg = _exit_term(canonical, 'conditional_vargrad', 1, 0, below=6.0)
    assert _fires(cfg, 'exit_bar_is_within_measured_range', severity=BASELINE)
    assert errors(cfg) == [], (
        'a measured floor is evidence, not a config contradiction -- it must '
        'never block a run from starting')


def test_the_canonical_config_no_longer_ships_that_bar(canonical):
    """The other half, and the one that would have caught the drift: selecting
    the conditional route must be CLEAN on this rule. If a future edit puts an
    unreachable bar back into either protocol, this fails."""
    cfg = copy.deepcopy(canonical)
    cfg['protocol'] = 'conditional_vargrad'
    assert not _fires(cfg, 'exit_bar_is_within_measured_range')


def test_a_bar_above_the_measured_floor_but_inside_sigma_is_a_baseline(canonical):
    """Measured min 17.1, sigma 9.9. A bar at 20 is reachable but sits inside
    the metric's own scatter -- R14's read-time condition, at load time. Worth
    stating, never worth failing on: a run may set a bar it expects to be tight."""
    cfg = _exit_term(canonical, 'conditional_vargrad', 1, 0, below=20.0)
    assert _fires(cfg, 'exit_bar_is_within_measured_range', severity=BASELINE)
    assert errors(cfg) == [], 'a tight bar is a departure, not a contradiction'


def test_a_bar_clear_of_the_measured_floor_passes(canonical):
    """MUTATION IN THE PASSING DIRECTION. Without this the rule could be firing
    on every bar on this metric and the tests above would not notice."""
    cfg = _exit_term(canonical, 'conditional_vargrad', 1, 0, below=40.0)
    assert not _fires(cfg, 'exit_bar_is_within_measured_range', severity=ERROR)
    assert not _fires(cfg, 'exit_bar_is_within_measured_range', severity=BASELINE)


def test_a_measured_range_can_never_produce_an_error(canonical):
    """THE SEVERITY IS THE RULE'S CONTRACT, so it is pinned rather than left to
    whoever edits the table next.

    This rule reasons from a measurement, and a measurement is not a property of
    the config being checked -- the floor was read off a battery with five
    railed controls, and the configs written to unrail them are precisely the
    ones that should be allowed to aim under it. Promoting any branch to ERROR
    turns the evidence into a law and blocks the next experiment. Asserted over
    an absurd bar so it holds for whatever the table grows to contain."""
    cfg = _exit_term(canonical, 'conditional_vargrad', 1, 0, below=-1e9)
    assert _fires(cfg, 'exit_bar_is_within_measured_range', severity=BASELINE)
    assert [v for v in check(cfg) if v.rule == 'exit_bar_is_within_measured_range'
            and v.severity == ERROR] == []


def test_an_unmeasured_metric_abstains(canonical):
    """The table covers one metric. Everything else must abstain -- a missing
    entry is not evidence that a bar is fine, and a rule that guessed would make
    every other rule here less trustworthy."""
    cfg = _exit_term(canonical, 'conditional_vargrad', 1, 0,
                     metric='fwd/nothing_ever_measured', below=1e-9)
    assert not _fires(cfg, 'exit_bar_is_within_measured_range')


def test_patience_on_a_coarse_metric_that_outruns_the_run_is_an_error(canonical):
    """`patience` counts WRITES of its metric. The coarse term here is now
    gates/progress_done -- published off the EVAL path, not the 10-step tick block,
    because the marginal and energy metrics it reads exist only there -- so it is
    written once per eval_period (250 here). patience 5 therefore needs 1,250 train
    steps; a 1,000-step run can never reach it and the stage exits on its other
    terms while the config reads as if this one gates."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 0, patience=5)
    cfg['epochs'] = 1000
    assert _fires(cfg, 'exit_patience_is_reachable', severity=ERROR)


def test_patience_on_a_coarse_metric_that_fits_is_a_baseline(canonical):
    """Same term in a run long enough to satisfy it. Reachable, so not an error --
    but the same integer on a tick-cadence term in the same block costs 50 steps
    rather than 1,250, and nothing at the point of writing it says so.

    This is why _EVAL_CADENCE_GATES exists: gates/* is tick-rate by default, and
    classifying gates/progress_done that way would understate its patience cost by
    eval_period/10. Publishing it every tick instead would be worse -- it would
    republish a stale verdict, and _advance_term counts fresh writes."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 0, patience=5)
    assert _fires(cfg, 'exit_patience_is_reachable', severity=BASELINE)
    assert errors(cfg) == []


def test_patience_on_a_tick_cadence_metric_is_not_flagged(canonical):
    """MUTATION IN THE PASSING DIRECTION. gates/mle_flat is published from the
    same 10-step block that runs the tick, so patience 5 is 50 steps and the
    rule must stay quiet -- the canonical config already carries exactly this
    term, and a rule that flagged it would fire on every protocol in the repo."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 0, patience=20)
    assert not _fires(cfg, 'exit_patience_is_reachable')


def test_patience_one_is_always_reachable(canonical):
    """One measurement is one measurement at any cadence. This is the shape
    prod0810 actually shipped (`eval/wass_debiased` with no patience key), and
    it is NOT a fault -- see the streak tests for what was wrong with it."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 0, patience=1)
    # 100 steps is shorter than ONE eval_period (250) and still fine: the term
    # needs a single write, and evaluation() forces one at step 50. Long enough,
    # though, for the sibling tick terms -- patience 5 at 10 steps is 50 -- so
    # this isolates the coarse term rather than tripping over its neighbours.
    cfg['epochs'] = 100
    assert not _fires(cfg, 'exit_patience_is_reachable')


def test_effective_batch_below_baseline_is_a_baseline_not_an_error(canonical):
    cfg = broken(canonical, batch_size=100, max_batch_size=100,
                 fused_grad_accum_min_samples=0, batch_util_target=0)
    assert _fires(cfg, 'effective_batch_meets_baseline', severity=BASELINE)
    assert errors(cfg) == [], 'a baseline departure must not be an ERROR'


def test_accumulation_lifts_a_small_batch_to_the_baseline(canonical):
    """batch 100 with accumulation to 1000 samples MEETS the baseline -- the
    effective optimization batch is what matters, not the physical one."""
    cfg = broken(canonical, batch_size=100, max_batch_size=100,
                 fused_grad_accum_min_samples=1000)
    assert not _fires(cfg, 'effective_batch_meets_baseline', severity=BASELINE)


# ---------------------------------------------------------------------------
# Rule hygiene
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The two rules from the 2026-08-17 conditional-detonation comparison (F-042).
# Both are keyed to the CONDITIONAL route, which the canonical config is not, so
# each fixture builds the conditional case rather than mutating mk_dev in place.
# ---------------------------------------------------------------------------

def _conditional(canonical, **dotted):
    """mk_dev switched onto the conditional route, with the Z trio already set
    correctly -- i.e. the state a correct conditional config is in, so a test
    that mutates one key is testing exactly that key.

    TWO OF THE THREE ARE NOW CARRIED BY THE PROTOCOL and are deliberately NOT set
    here. Since the mode-key migration, selecting `conditional_vargrad` brings
    `tb_z_source: persistent` (each stage's `loss_coeffs`) and z_calibration-off
    (the flag, omitted) with it. Setting them here again would test this helper
    rather than the config: the rule would pass even if mk_dev's stages lost them,
    which is the regression these tests exist to catch. `half_life_visits` is
    global, so it stays -- it is the one key a mode switch still hand-edits --
    and so is the replay draw's prioritise switch (_conditional_route_globals),
    which the route refuses to start without."""
    cfg = broken(canonical, **dotted)
    cfg['embedding_conditioning'] = True
    cfg['protocol'] = 'conditional_vargrad'
    cfg['condition_log_z']['half_life_visits'] = 28.0
    return _conditional_route_globals(cfg)


def _fired(cfg, rule_name):
    return [v for v in check(cfg) if v.rule == rule_name]


def test_conditional_z_settings_pass_when_set_for_the_route(canonical):
    assert _fired(_conditional(canonical), 'conditional_z_settings_are_conditional') == []


def test_inherited_half_life_fires(canonical):
    """The one key of the F-042 trio a mode switch still has to hand-edit."""
    cfg = _conditional(canonical)
    cfg['condition_log_z']['half_life_visits'] = 7.0
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


# `var_conditioning` ships fracs.fwd = 0.0 -- fwd trains only as the Z sidecar,
# outside the frac mix -- and the rule reads the frac, so it skips fwd there (see
# the `live` computation). Only the two LIVE branches, bwd and replay, fire.
@pytest.mark.parametrize('branch', ['bwd', 'replay'])
def test_inherited_tb_z_source_fires_at_its_state6_home(canonical, branch):
    """A stage that resolves to `learned` on the conditional route -- F-042's
    second key, checked where state 6 put it."""
    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs'][branch]['tb_z_source'] = 'learned'
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


def test_a_branch_the_stage_does_not_train_is_not_flagged(canonical):
    """The narrowness is load-bearing, not incidental. The rule judges only the
    branches a stage's fracs train, and `var_conditioning` zeroes the fwd frac
    (fwd enters as the Z sidecar, outside the mix); flagging a zero-frac branch
    would add a violation to every correct conditional config, and a BASELINE
    rule that cries wolf is one people learn to skip. Guarded in both
    directions -- wrong value AND absent -- because the absence branch is the
    newer one."""
    cfg = _conditional(canonical)
    assert _vg_stage(cfg)['fracs']['fwd'] == 0.0, 'precondition: fwd has no frac'
    _vg_stage(cfg)['loss_coeffs']['fwd']['tb_z_source'] = 'learned'
    assert _fired(cfg, 'conditional_z_settings_are_conditional') == []

    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs']['fwd'].pop('tb_z_source', None)
    cfg['fwd_loss_coeffs'].pop('tb_z_source', None)
    assert _fired(cfg, 'conditional_z_settings_are_conditional') == []


def test_z_calibration_flag_on_a_conditional_stage_fires(canonical):
    """F-042's third key, as the state-5 stage flag rather than a global."""
    cfg = _conditional(canonical)
    _vg_stage(cfg).setdefault('flags', {})['z_calibration'] = True
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


# --- absence, which is the way this actually goes wrong -------------------
# A generator that writes the PRE-MIGRATION spelling leaves the live home unset.
# Both keys then fall back to a code default that is the UNCONDITIONAL value, so
# the config trains in the detonation regime while reading as correct. Judging
# only present values made the rule blind to exactly this.

@pytest.mark.parametrize('branch', ['bwd', 'replay'])
def test_absent_tb_z_source_fires(canonical, branch):
    """Unset everywhere -> train.py:1105 falls back to `learned`."""
    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs'][branch].pop('tb_z_source', None)
    cfg[f'{branch}_loss_coeffs'].pop('tb_z_source', None)
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


def test_absent_tb_z_source_names_the_pre_migration_home_when_carried(canonical):
    """The diagnosis, not just the fault: a config carrying `persistent` at the
    dead `condition_log_z.*_tb_z_source` is the case this rule was blind to, and
    the message has to say where the value actually is."""
    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs']['bwd'].pop('tb_z_source', None)
    cfg['bwd_loss_coeffs'].pop('tb_z_source', None)
    cfg['condition_log_z']['bwd_tb_z_source'] = 'persistent'
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1, [str(v) for v in vs]
    assert 'condition_log_z.bwd_tb_z_source' in str(vs[0]), str(vs[0])


def test_absent_half_life_is_now_safe(canonical):
    """INVERTED 2026-08-17, and the inversion is the point.

    This used to assert the rule FIRES on an absent key, because the code default
    was 7.0 -- the unconditional value -- so omitting the key silently selected
    the setting that detonates var_conditioning. `buffer.DEFAULT_HALF_LIFE_VISITS`
    is now 200.0, so an absent key resolves to a safe value and a rule that still
    complained would be reporting a hazard that no longer exists.

    THE SECOND ASSERTION IS WHY THIS TEST STILL HAS TEETH. Checking only that the
    rule is quiet would pass just as happily if someone put the default back to
    7.0 -- quiet rule, quiet test, live hazard. So the test reads the ACTUAL
    default and requires it to be safe by the rule's own threshold.
    """
    from buffer import DEFAULT_HALF_LIFE_VISITS

    cfg = _conditional(canonical)
    cfg['condition_log_z'].pop('half_life_visits', None)
    assert _fired(cfg, 'conditional_z_settings_are_conditional') == []
    assert DEFAULT_HALF_LIFE_VISITS >= 28.0, (
        f'the code default is {DEFAULT_HALF_LIFE_VISITS}, below the 28 this rule '
        f'requires of an explicit value -- so an omitted key now selects a setting '
        f'the rule would reject if it were written down')


def test_explicit_short_half_life_still_fires(canonical):
    """The hazard that remains reachable: an explicitly written short value."""
    cfg = _conditional(canonical)
    cfg['condition_log_z']['half_life_visits'] = 7.0
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


def test_unconditional_route_is_left_alone(canonical):
    """The SAME values are correct on the unconditional route. A rule that fired
    there would flag the canonical config and every unconditional battery.

    z_calibration is enabled by the STAGE FLAG, not a global `enabled`, which is
    the state-5 spelling. The flag is SET here rather than read off mk_dev, whose
    equilibration stage may run without it (it is refused beside
    fwd_rollout_every > 0); the conditional control at the end shows the rule
    sees it."""
    assert not any(canonical.get(k) is True for k in
                   ('embedding_conditioning', 'molecule_conditioning', 'vector_conditioning')), \
        'precondition: mk_dev is unconditional'
    assert 'enabled' not in canonical['z_calibration'], \
        'precondition: z_calibration.enabled is a pre-state-5 spelling'
    cfg = copy.deepcopy(canonical)
    eq = [s for s in cfg['protocols']['unconditional_tb']['stages']
          if s['name'] == 'equilibration'][0]
    eq.setdefault('flags', {})['z_calibration'] = True
    assert _fired(cfg, 'conditional_z_settings_are_conditional') == []
    assert _fired(canonical, 'conditional_z_settings_are_conditional') == []
    cfg['embedding_conditioning'] = True
    assert any('z_calibration flag' in v.detail
               for v in _fired(cfg, 'conditional_z_settings_are_conditional'))


def _vg_stage(cfg):
    return [s for s in cfg['protocols']['conditional_vargrad']['stages']
            if s['name'] == 'var_conditioning'][0]


def test_vargrad_grouping_passes_as_shipped(canonical):
    assert _fired(_conditional(canonical), 'vargrad_needs_groups') == []


def test_fwd_vargrad_singleton_group_fires(canonical):
    """fwd repeats is the ONLY source of a forward group; at 1 every group is a
    singleton and vg_loss is identically zero -- silent, not a crash."""
    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs']['fwd']['repeats'] = 1.0
    vs = _fired(cfg, 'vargrad_needs_groups')
    assert len(vs) == 1 and vs[0].severity == ERROR


def test_bwd_vargrad_fires_only_when_BOTH_group_sources_are_absent(canonical):
    """The backward condition is a DISJUNCTION, and it must stay one: aug14 and
    aug11 satisfy it with repeats 2 at condition_block_m 1, aug13 with
    condition_block_m 2 at repeats 1. A conjunction rejects two configs that ran.

    `var_conditioning` now ships the THIRD source -- condition_draw's prior_rows
    2 at repeats 1 and condition_block_m 0 -- so every case below that tests the
    first two drops the block, and the shipped stage is the positive case for
    the third."""
    def _stage(cfg, **bwd):
        st = _vg_stage(cfg)
        st.pop('condition_draw', None)
        st['loss_coeffs']['bwd'].update(bwd)
        return cfg

    # SET IT ON THE STAGE, not just the base. Since state 6 condition_block_m is
    # a loss coefficient, and `var_conditioning` overrides it -- so a base value
    # alone is shadowed and the case would silently test the stage's value.
    both_absent = _stage(_conditional(canonical, bwd_loss_coeffs__condition_block_m=1),
                         condition_block_m=1.0, repeats=1.0)
    vs = _fired(both_absent, 'vargrad_needs_groups')
    assert len(vs) == 1 and vs[0].severity == ERROR

    # aug13's spelling: repeats 1 but blocked draws -- must NOT fire
    via_blocks = _stage(_conditional(canonical), condition_block_m=2.0, repeats=1.0)
    assert _fired(via_blocks, 'vargrad_needs_groups') == []

    # aug14's spelling: repeats 2, blocking off -- must NOT fire
    via_repeats = _stage(_conditional(canonical), condition_block_m=1.0, repeats=2.0)
    assert _fired(via_repeats, 'vargrad_needs_groups') == []

    # ...and the STAGE override is what the rule reads, not just the base block:
    # a base of 2 with the stage turning blocking off must fire.
    stage_off = _stage(_conditional(canonical, bwd_loss_coeffs__condition_block_m=2),
                       condition_block_m=0, repeats=1.0)
    vs = _fired(stage_off, 'vargrad_needs_groups')
    assert len(vs) == 1 and vs[0].severity == ERROR

    # the shipped spelling: the aligned draw's prior_rows at repeats 1 and
    # condition_block_m 0 -- must NOT fire
    shipped = _conditional(canonical)
    assert _vg_stage(shipped)['condition_draw']['prior_rows'] >= 2
    assert _vg_stage(shipped)['loss_coeffs']['bwd']['condition_block_m'] == 0
    assert _vg_stage(shipped)['loss_coeffs']['bwd']['repeats'] == 1.0
    assert _fired(shipped, 'vargrad_needs_groups') == []


def test_vargrad_rule_abstains_off_the_vargrad_route(canonical):
    """On a TB route `repeats` means something else and 1 is correct. mk_dev's
    unconditional protocol runs repeats 1 everywhere and must stay clean."""
    assert _fired(canonical, 'vargrad_needs_groups') == []


def _cadenced_fused_stage(canonical):
    """A copy of canonical with the active protocol's fused stage running forward
    rollouts on 1 in 10 steps (docs/design/rarer_rollouts.md). Returns (cfg, stage)
    with the stage dict being cfg's own object."""
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocols']['unconditional_tb']['stages']
                 if s.get('train_mode', 'fused') == 'fused' and 'fracs' in s)
    stage['fwd_rollout_every'] = 10
    stage.setdefault('flags', {})
    return cfg, stage


def test_fwd_rollout_cadence_refuses_the_z_calibration_servo(canonical):
    """The servo's rollout mode calls the energy function on every step it fires,
    off a sensor frozen between rollouts -- so left on, it silently restores the
    cost the cadence removes. Must fire."""
    cfg, stage = _cadenced_fused_stage(canonical)
    stage['flags']['z_calibration'] = True
    cfg['z_calibration']['fill_threshold'] = 0.5
    assert _fires(cfg, 'fwd_rollout_cadence_is_well_formed')


def test_fwd_rollout_cadence_refuses_a_disabled_fill(canonical):
    """With the servo off, z_level_fill is the ONLY thing pinning log Z at each
    rollout; fill_threshold 0 disables it, leaving Z unpinned. Must fire."""
    cfg, stage = _cadenced_fused_stage(canonical)
    stage['flags']['z_calibration'] = False
    cfg['z_calibration']['fill_threshold'] = 0
    assert _fires(cfg, 'fwd_rollout_cadence_is_well_formed')


def test_fwd_rollout_cadence_accepts_the_well_formed_shape(canonical):
    cfg, stage = _cadenced_fused_stage(canonical)
    stage['flags']['z_calibration'] = False
    cfg['z_calibration']['fill_threshold'] = 0.5
    assert not _fires(cfg, 'fwd_rollout_cadence_is_well_formed')
    # and with no cadence key at all the rule has nothing to say
    assert not _fires(canonical, 'fwd_rollout_cadence_is_well_formed')


def test_z_fill_mode_refuses_a_misspelt_mode_and_a_bad_eval_source(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['z_calibration']['fill_mode'] = 'ema'
    assert _fires(cfg, 'z_fill_mode_is_well_formed')
    cfg = copy.deepcopy(canonical)
    cfg['z_calibration']['fill_from_eval'] = 'always'
    assert _fires(cfg, 'z_fill_mode_is_well_formed')
    cfg = copy.deepcopy(canonical)
    cfg['z_calibration']['fill_process_var'] = -1.0
    assert _fires(cfg, 'z_fill_mode_is_well_formed')


def test_z_fill_mode_accepts_the_absorber(canonical):
    cfg = copy.deepcopy(canonical)
    cfg['z_calibration'].update({'fill_mode': 'absorb', 'fill_process_var': 0.01,
                                 'fill_moment_reset': 0.5, 'fill_from_eval': 'fill'})
    assert not _fires(cfg, 'z_fill_mode_is_well_formed')
    assert not _fires(canonical, 'z_fill_mode_is_well_formed')


def test_batch_root_is_forward_only_and_needs_a_cadence(canonical):
    # on the backward branch: refused outright
    cfg = copy.deepcopy(canonical)
    cfg['bwd_loss_coeffs']['tb_z_source'] = 'batch_root'
    assert _fires(cfg, 'batch_root_forward_is_well_formed')
    # on fwd without a cadence: refused. mk_dev's stage may carry one, so zero it
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocols']['unconditional_tb']['stages']
                 if s.get('train_mode', 'fused') == 'fused' and 'fracs' in s)
    stage['fwd_rollout_every'] = 0
    stage.setdefault('loss_coeffs', {}).setdefault('fwd', {}).update(
        {'tb_z_source': 'batch_root', 'freeze_policy': 0.0})
    assert _fires(cfg, 'batch_root_forward_is_well_formed')
    # with a cadence but the policy frozen: refused (the branch would train nothing)
    cfg, stage = _cadenced_fused_stage(canonical)
    stage.setdefault('loss_coeffs', {}).setdefault('fwd', {}).update(
        {'tb_z_source': 'batch_root', 'freeze_policy': 1.0})
    assert _fires(cfg, 'batch_root_forward_is_well_formed')
    # the well-formed shape
    cfg, stage = _cadenced_fused_stage(canonical)
    stage.setdefault('loss_coeffs', {}).setdefault('fwd', {}).update(
        {'tb_z_source': 'batch_root', 'freeze_policy': 0.0})
    assert not _fires(cfg, 'batch_root_forward_is_well_formed')
    assert not _fires(canonical, 'batch_root_forward_is_well_formed')


def test_replay_seat_keys_are_judged_on_the_resolved_config(canonical):
    """The full case table is tests/config/test_replay_seat_invariants.py; this is
    the rule's mutation test in the registry's own file."""
    # a base pooled_source nothing understands: refused on every stage
    assert _fires(broken(canonical, fwd_loss_coeffs__pooled_source='bwd'),
                  'replay_seat_is_well_formed')
    # a warm-up the buffer's cap can never satisfy: it would never release
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocols']['unconditional_tb']['stages']
                 if (s.get('fwd_rollout_every') or 0) > 0)
    stage['replay_warmup_rows'] = int(cfg['buffers']['replay_buffer']['max_size'])
    assert _fires(cfg, 'replay_seat_is_well_formed')
    stage['replay_warmup_rows'] = 1000
    assert not _fires(cfg, 'replay_seat_is_well_formed')
    assert not _fires(canonical, 'replay_seat_is_well_formed')


def test_condition_draws_are_judged_on_the_resolved_config(canonical):
    """The full case table is tests/config/test_condition_draw_invariants.py; this
    is the rule's mutation test in the registry's own file."""
    # a rollout draw nothing implements
    assert _fires(broken(canonical, condition_log_z__rollout_condition_draw='stratified'),
                  'condition_draw_is_well_formed')
    # a power only 'under_drawn' reads, moved under the iid draw: a dead key
    assert _fires(broken(canonical, condition_log_z__rollout_under_drawn_power=2.0),
                  'condition_draw_is_well_formed')
    assert not _fires(broken(canonical, condition_log_z__rollout_condition_draw='under_drawn',
                             condition_log_z__rollout_under_drawn_power=2.0),
                      'condition_draw_is_well_formed')
    assert not _fires(canonical, 'condition_draw_is_well_formed')


def test_a_global_untrusted_z_without_an_energy_reference_is_an_error(canonical):
    # one all-condition level only stands in for conditions whose depth was removed
    cfg = broken(canonical, condition_log_z__untrusted_z='global', energy_config__energy_reference=None)
    assert _fires(cfg, 'energy_reference_is_consistent')
    ok = broken(canonical, condition_log_z__untrusted_z='global', energy_config__energy_reference='seed_min')
    assert not _fires(ok, 'energy_reference_is_consistent')


def test_an_energy_reference_with_temperature_conditioning_is_an_error(canonical):
    cfg = broken(canonical, energy_config__energy_reference='seed_min', temperature_conditioning=True)
    assert _fires(cfg, 'energy_reference_is_consistent')


def test_an_nf_policy_refuses_sde_only_settings(canonical):
    """Under model.kind nf an SDE-only setting refuses at load; the canonical keys pass once the auto clip (fitted
    to SDE gradients) is given a number. Full coverage: tests/models/test_nf_policy.py."""
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'nf_policy_refuses_sde_only_settings')      # kind sde
    cfg['policy']['kind'] = 'nf'
    assert _fires(cfg, 'nf_policy_refuses_sde_only_settings')          # gradient_norm_clip auto
    cfg['gradient_norm_clip'] = 10.0
    assert not _fires(cfg, 'nf_policy_refuses_sde_only_settings')
    cfg['compile_policy'] = True
    assert _fires(cfg, 'nf_policy_refuses_sde_only_settings')


def test_a_malformed_heldout_gate_is_an_error(canonical):
    """Full coverage: tests/protocol/test_heldout_gate.py."""
    cfg = copy.deepcopy(canonical)
    assert not _fires(cfg, 'heldout_gate_is_well_formed')
    cfg['heldout_gate']['smooth'] = 0
    assert _fires(cfg, 'heldout_gate_is_well_formed', severity=ERROR)
    cfg['heldout_gate'] = {'source': 'auto', 'patience': 2}
    assert _fires(cfg, 'heldout_gate_is_well_formed', severity=ERROR)


def test_every_rule_is_mutation_tested():
    """Each rule in RULES must have at least one test above that makes it fire.
    Without this, adding a rule and forgetting its mutation test leaves a check
    that is asserted to pass and never shown capable of failing.

    The conformer-route rules are mutation-tested against their own minimal
    conformer config in test_conformer_route_rules.py, and the retired prior
    model's rule in test_crystal_prior_model_retired.py, so those files count too."""
    src = ''.join(Path(__file__).with_name(name).read_text(encoding='utf-8')
                  for name in (Path(__file__).name, 'test_conformer_route_rules.py',
                               'test_crystal_prior_model_retired.py'))
    for rule in RULES:
        assert f"'{rule.__name__}'" in src, (
            f'rule {rule.__name__} has no mutation test in this file or '
            f'test_conformer_route_rules.py')


def test_rules_abstain_on_an_empty_config():
    """No rule may fire on a config that simply omits the keys. Migration and
    generation both produce partial dicts, and a rule that fires on absence
    reports a fault that is not there."""
    assert check({}) == []


def test_rules_abstain_on_auto_values():
    """`auto` is resolved later from (W, T). A rule that treats the string as a
    number, or as zero, would judge a value that does not exist yet."""
    cfg = {'batch_util_target': 'auto', 'batch_growth_factor': 'auto',
           'gradient_norm_clip': 'auto', 'batch_size': 'auto'}
    assert check(cfg) == []


def test_severity_ordering_puts_errors_first(canonical):
    cfg = broken(canonical, figs_period=500, batch_size=10, max_batch_size=10,
                 fused_grad_accum_min_samples=0)
    vs = check(cfg)
    severities = [v.severity for v in vs]
    assert ERROR in severities and BASELINE in severities
    assert severities == sorted(severities, key=lambda s: s != ERROR)


def test_a_stray_lr_probe_block_is_an_error(canonical):
    """Replay Racing is deleted. Nothing reads `lr_probe` any more, so a config
    still carrying the block would configure NOTHING -- silently, which is the
    failure mode the whole retired-key apparatus exists to prevent.

    A load-time gate rather than a retired-key migration, following state 9's
    precedent: a migration moves `project_state_version` and every config in the
    tree then needs restamping, and there is nothing to migrate -- no tracked
    config outside the deleted `configs/race_*.yaml` ever carried it."""
    cfg = copy.deepcopy(canonical)
    cfg['lr_probe'] = {'enabled': True, 'window': 10}
    assert _fires(cfg, 'lr_probe_is_retired')


def test_the_lr_probe_rule_abstains_on_a_config_without_the_block(canonical):
    """Mutation guard: the canonical config has never carried it, so the rule
    must be silent there -- a rule that fires on every config is not a rule."""
    assert not _fires(copy.deepcopy(canonical), 'lr_probe_is_retired')


def test_a_stray_grad_geometry_block_is_an_error_on_the_crystal_route(canonical):
    """The fused-step gradient geometry probe is deleted from train.py, so a
    crystal config carrying `grad_geometry` configures nothing."""
    cfg = copy.deepcopy(canonical)
    cfg['grad_geometry'] = {'enabled': False, 'every': 50}
    assert _fires(cfg, 'grad_geometry_is_retired')


def test_the_grad_geometry_rule_abstains_without_the_block_and_on_the_conformer_route(canonical):
    """Silent on mk_dev, which no longer carries the block, and on
    conformer_torsions, whose configs still do."""
    assert not _fires(copy.deepcopy(canonical), 'grad_geometry_is_retired')
    cfg = copy.deepcopy(canonical)
    cfg['grad_geometry'] = {'enabled': True, 'every': 50}
    cfg['energy_function'] = 'conformer_torsions'
    assert not [v for v in check(cfg) if v.rule == 'grad_geometry_is_retired']


def test_a_crystal_config_carrying_hold_dead_latent_rows_is_an_error(canonical):
    """Dead rows are always held off the conformer route, so the key configures
    nothing there, true or false; conformer_torsions configs still carry it."""
    assert not _fires(copy.deepcopy(canonical), 'hold_dead_rows_is_retired')
    for value in (True, False):
        cfg = copy.deepcopy(canonical)
        cfg['model']['hold_dead_latent_rows'] = value
        assert _fires(cfg, 'hold_dead_rows_is_retired'), value
    cfg['energy_function'] = 'conformer_torsions'
    assert not [v for v in check(cfg) if v.rule == 'hold_dead_rows_is_retired']


def test_a_crystal_config_carrying_eval_T_or_integrator_is_an_error(canonical):
    """model.sde.T is the one trajectory length off the conformer route; an
    `eval_T` or `integrator` block there configures nothing."""
    assert not _fires(copy.deepcopy(canonical), 'trajectory_length_is_model_T')
    for key, value in (('eval_T', 10), ('integrator', {'T': 10})):
        cfg = copy.deepcopy(canonical)
        cfg[key] = value
        assert _fires(cfg, 'trajectory_length_is_model_T'), key


def test_an_sde_crystal_config_without_model_T_is_an_error_and_nf_needs_none(canonical):
    """The SDE policy needs model.T; the NF policy (one step) reads none."""
    cfg = copy.deepcopy(canonical)
    del cfg['model']['T']
    assert _fires(cfg, 'trajectory_length_is_model_T')
    cfg['model']['T'] = 0
    assert _fires(cfg, 'trajectory_length_is_model_T')
    del cfg['model']['T']
    cfg['policy']['kind'] = 'nf'
    assert not [v for v in check(cfg) if v.rule == 'trajectory_length_is_model_T']


def test_the_trajectory_length_rule_abstains_on_the_conformer_routes(canonical):
    """conformer_torsions configs and train_conformer.py's conformer_dev (no
    energy_function) keep `integrator.T` and `eval_T`."""
    for ef in ('conformer_torsions', None):
        cfg = copy.deepcopy(canonical)
        cfg['eval_T'], cfg['integrator'] = 32, {'T': 32}
        del cfg['model']['T']
        if ef is None:
            del cfg['energy_function']
        else:
            cfg['energy_function'] = ef
        assert not [v for v in check(cfg) if v.rule == 'trajectory_length_is_model_T'], ef


def test_train_py_no_longer_reads_grad_geometry():
    """The rule above is only true while nothing in train.py reads the key."""
    src = (HERE / 'train.py').read_text(encoding='utf-8')
    for name in ('grad_geometry', '_fused_grad_diag_armed', '_log_fused_gradient_geometry',
                 '_fused_grad_geom_report'):
        assert name not in src, name


def test_a_crystal_ema_decay_is_an_error(canonical):
    """There is no EMA model off the conformer route (ema_model IS gfn_model),
    so a decay configures nothing."""
    cfg = copy.deepcopy(canonical)
    cfg['ema_decay'] = 0.99
    assert _fires(cfg, 'ema_decay_is_conformer_only')


def test_the_ema_decay_rule_abstains_on_null_and_on_the_conformer_route(canonical):
    """Silent on mk_dev (no key), on an explicit null, and on conformer_torsions,
    where ConformerModeller keeps its EMA copy."""
    assert not _fires(copy.deepcopy(canonical), 'ema_decay_is_conformer_only')
    cfg = copy.deepcopy(canonical)
    cfg['ema_decay'] = None
    assert not _fires(cfg, 'ema_decay_is_conformer_only')
    cfg['ema_decay'] = 0.5
    cfg['energy_function'] = 'conformer_torsions'
    assert not [v for v in check(cfg) if v.rule == 'ema_decay_is_conformer_only']
