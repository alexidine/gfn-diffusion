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

HERE = Path(__file__).parent
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def canonical():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


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


def test_figs_period_not_a_multiple_never_fires(canonical):
    assert _fires(broken(canonical, figs_period=501), 'figs_period_fires')
    assert not _fires(broken(canonical, figs_period=1000), 'figs_period_fires')


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
    quiet, and the selector rule is what keeps the config from passing."""
    cfg = broken(canonical, protocol='does_not_exist')
    assert not _fires(cfg, 'auto_lr_requires_an_adaptive_sensor'),         'fixture assumption: the stage rule DOES go quiet'
    assert errors(cfg), 'but the config must still be rejected'


def test_an_inactive_protocol_that_cannot_parse_is_caught(canonical):
    """The point of checking the whole library: an inactive protocol is otherwise
    unexamined until it is SELECTED, and then it fails at load -- on the switch,
    which is the worst moment. Switching route is meant to be one word."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['conditional_vargrad'] = {
        'stages': [{'name': 'train_prior', 'train_mode': 'bwd',
                    'lr_sensor': {'kind': 'hyper'}}]}   # hyper REQUIRES beta
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
                    'lr_sensor': {'kind': 'hyper', 'beta': 0.1}},
                   {'name': 'var_conditioning', 'train_mode': 'fused',
                    'lr_sensor': {'kind': 'hyper', 'beta': 0.05}}]}
    assert not _fires(cfg, 'every_protocol_parses')
    assert errors(cfg) == [], [str(e) for e in errors(cfg)]


def test_auto_lr_with_no_sensor_anywhere_is_an_error(canonical):
    """`auto` claims a servo owns the rate. With no adaptive sensor nothing moves
    peak_scale off 1.0, so the run trains at the seed for its whole life while the
    config reads as adaptive."""
    cfg = copy.deepcopy(canonical)
    for st in cfg['protocols']['unconditional_tb']['stages']:
        st.pop('lr_sensor', None)
    assert _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


def test_auto_lr_with_sensor_kind_none_is_also_an_error(canonical):
    """`{kind: none}` and an omitted block mean the same thing. Only the second
    is silent, but neither owns an `auto` rate."""
    cfg = copy.deepcopy(canonical)
    for st in cfg['protocols']['unconditional_tb']['stages']:
        st['lr_sensor'] = {'kind': 'none'}
    assert _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


def test_one_stage_missing_a_sensor_still_fires(canonical):
    """Checked PER STAGE: a sensor on the terminal stage does nothing for a
    phase-1 learning rate."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['unconditional_tb']['stages'][0].pop('lr_sensor', None)
    v = [x for x in check(cfg) if x.rule == 'auto_lr_requires_an_adaptive_sensor']
    assert len(v) == 1 and repr(cfg['protocols']['unconditional_tb']['stages'][0]['name']) in v[0].detail


def test_explicit_float_lrs_need_no_sensor(canonical):
    """The other half of the rule: a float is a fixed peak that takes the warmup
    envelope and divergence handling only, so it has nothing to yield to and must
    not be flagged."""
    cfg = copy.deepcopy(canonical)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3.0e-4
    for st in cfg['protocols']['unconditional_tb']['stages']:
        st.pop('lr_sensor', None)
    assert not _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


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
    for st in resolved['protocols']['unconditional_tb']['stages']:
        st.pop('lr_sensor', None)

    # derived from the config: the evidence is gone, so nothing fires
    assert ci.auto_lr_requires_an_adaptive_sensor(resolved) == []
    # told what was managed: fires correctly
    told = ci.auto_lr_requires_an_adaptive_sensor(
        resolved, auto_keys=['lr_policy', 'lr_back', 'lr_replay', 'lr_fused'])
    assert told and all(v.severity == ERROR for v in told)


def test_ray_on_a_non_fused_stage_is_an_error(canonical):
    """The probe draws from replay and scores replay_loss_coeffs; a bwd stage
    trains neither, so it would rate a loss nobody is optimising -- silently."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['unconditional_tb']['stages'][0]['lr_sensor'] = {'kind': 'ray'}   # train_prior is bwd
    assert _fires(cfg, 'ray_sensor_needs_a_coherent_stage')


def test_ray_on_a_fused_stage_without_replay_tb_is_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    st = cfg['protocols']['unconditional_tb']['stages'][1]
    assert st['train_mode'] == 'fused', 'fixture assumption'
    st.setdefault('loss_coeffs', {}).setdefault('replay', {})['tb'] = 0.0
    assert _fires(cfg, 'ray_sensor_needs_a_coherent_stage')


def test_hyper_is_accepted_on_a_non_fused_stage(canonical):
    """`hyper` reads no loss, so unlike `ray` it is coherent whatever the stage
    trains. Flagging it would push people back to omitting the block."""
    cfg = copy.deepcopy(canonical)
    cfg['protocols']['unconditional_tb']['stages'][0]['lr_sensor'] = {'kind': 'hyper', 'beta': 0.05}
    assert not _fires(cfg, 'ray_sensor_needs_a_coherent_stage')
    assert not _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


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
# Exit triggers -- the two dead-gate shapes from the 2026-08-16 audit
# (docs/design/next_battery.md 1.1a and 1.3)
# ---------------------------------------------------------------------------

#: the term these tests edit when the stage they target ships no exit block.
#: `fwd/logw_std_within` is the one metric MEASURED_METRIC_RANGES covers, so it is
#: the only one that can exercise the rule at all.
_LOGW_TERM = {'metric': 'fwd/logw_std_within', 'below': 6.0}


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
    """`patience` counts WRITES of its metric. eval/wass_debiased is written
    once per eval_period (250 here), so patience 5 needs 1,250 train steps; a
    1,000-step run can never reach it and the stage exits on its other terms
    while the config reads as if this one gates."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 1, patience=5)
    cfg['epochs'] = 1000
    assert _fires(cfg, 'exit_patience_is_reachable', severity=ERROR)


def test_patience_on_a_coarse_metric_that_fits_is_a_baseline(canonical):
    """Same term in a run long enough to satisfy it. Reachable, so not an
    error -- but the same integer on a tick-cadence term in the same block
    costs 50 steps rather than 1,250, and nothing at the point of writing it
    says so."""
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 1, patience=5)
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
    cfg = _exit_term(canonical, 'unconditional_tb', 0, 1, patience=1)
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
    global, so it stays -- it is the one key a mode switch still hand-edits."""
    cfg = broken(canonical, **dotted)
    cfg['embedding_conditioning'] = True
    cfg['protocol'] = 'conditional_vargrad'
    cfg['condition_log_z']['half_life_visits'] = 28.0
    return cfg


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


# `var_conditioning` ships fracs.replay = 0.0, so the replay branch is not
# trained there and the rule deliberately skips it -- see the `live` computation.
# Only the two LIVE branches are expected to fire.
@pytest.mark.parametrize('branch', ['fwd', 'bwd'])
def test_inherited_tb_z_source_fires_at_its_state6_home(canonical, branch):
    """A stage that resolves to `learned` on the conditional route -- F-042's
    second key, checked where state 6 put it."""
    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs'][branch]['tb_z_source'] = 'learned'
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1 and vs[0].severity == BASELINE, [str(v) for v in vs]


def test_a_branch_the_stage_does_not_train_is_not_flagged(canonical):
    """The narrowness is load-bearing, not incidental. `var_conditioning` zeroes
    the replay frac, so its Z source is never read there; flagging it would add a
    violation to every correct conditional config, and a BASELINE rule that cries
    wolf is one people learn to skip. Guarded in both directions -- wrong value
    AND absent -- because the absence branch is the newer one."""
    cfg = _conditional(canonical)
    assert _vg_stage(cfg)['fracs']['replay'] == 0.0, 'precondition: replay is not trained'
    _vg_stage(cfg)['loss_coeffs']['replay']['tb_z_source'] = 'learned'
    assert _fired(cfg, 'conditional_z_settings_are_conditional') == []

    cfg = _conditional(canonical)
    _vg_stage(cfg)['loss_coeffs']['replay'].pop('tb_z_source', None)
    cfg['replay_loss_coeffs'].pop('tb_z_source', None)
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

@pytest.mark.parametrize('branch', ['fwd', 'bwd'])
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
    _vg_stage(cfg)['loss_coeffs']['fwd'].pop('tb_z_source', None)
    cfg['fwd_loss_coeffs'].pop('tb_z_source', None)
    cfg['condition_log_z']['fwd_tb_z_source'] = 'persistent'
    vs = _fired(cfg, 'conditional_z_settings_are_conditional')
    assert len(vs) == 1, [str(v) for v in vs]
    assert 'condition_log_z.fwd_tb_z_source' in str(vs[0]), str(vs[0])


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

    Note what the precondition now asserts: z_calibration is enabled by the STAGE
    FLAG, not a global `enabled`, which is the state-5 spelling."""
    eq = [s for s in canonical['protocols']['unconditional_tb']['stages']
          if s['name'] == 'equilibration'][0]
    assert eq['flags']['z_calibration'] is True, 'precondition: mk_dev is unconditional'
    assert 'enabled' not in canonical['z_calibration'], \
        'precondition: z_calibration.enabled is a pre-state-5 spelling'
    assert _fired(canonical, 'conditional_z_settings_are_conditional') == []


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
    condition_block_m 2 at repeats 1. A conjunction rejects two configs that ran."""
    # SET IT ON THE STAGE, not just the base. Since state 6 condition_block_m is
    # a loss coefficient, and `var_conditioning` overrides it to 2.0 -- so a base
    # of 1 is shadowed and this arm was silently testing the passing case.
    both_absent = _conditional(canonical, bwd_loss_coeffs__condition_block_m=1)
    _vg_stage(both_absent)['loss_coeffs']['bwd']['condition_block_m'] = 1.0
    _vg_stage(both_absent)['loss_coeffs']['bwd']['repeats'] = 1.0
    vs = _fired(both_absent, 'vargrad_needs_groups')
    assert len(vs) == 1 and vs[0].severity == ERROR

    # aug13's spelling: repeats 1 but blocked draws -- must NOT fire
    via_blocks = _conditional(canonical, bwd_loss_coeffs__condition_block_m=2)
    _vg_stage(via_blocks)['loss_coeffs']['bwd']['repeats'] = 1.0
    assert _fired(via_blocks, 'vargrad_needs_groups') == []

    # aug14's spelling: repeats 2, blocking off -- must NOT fire
    via_repeats = _conditional(canonical, bwd_loss_coeffs__condition_block_m=1)
    _vg_stage(via_repeats)['loss_coeffs']['bwd']['repeats'] = 2.0
    assert _fired(via_repeats, 'vargrad_needs_groups') == []

    # ...and the STAGE override is what the rule reads, not just the base block:
    # a base of 2 with the stage turning blocking off must fire.
    stage_off = _conditional(canonical, bwd_loss_coeffs__condition_block_m=2)
    _vg_stage(stage_off)['loss_coeffs']['bwd']['repeats'] = 1.0
    _vg_stage(stage_off)['loss_coeffs']['bwd']['condition_block_m'] = 0
    vs = _fired(stage_off, 'vargrad_needs_groups')
    assert len(vs) == 1 and vs[0].severity == ERROR


def test_vargrad_rule_abstains_off_the_vargrad_route(canonical):
    """On a TB route `repeats` means something else and 1 is correct. mk_dev's
    unconditional protocol runs repeats 1 everywhere and must stay clean."""
    assert _fired(canonical, 'vargrad_needs_groups') == []


def test_every_rule_is_mutation_tested():
    """Each rule in RULES must have at least one test above that makes it fire.
    Without this, adding a rule and forgetting its mutation test leaves a check
    that is asserted to pass and never shown capable of failing."""
    src = (HERE / 'test_config_invariants.py').read_text(encoding='utf-8')
    for rule in RULES:
        assert f"'{rule.__name__}'" in src, (
            f'rule {rule.__name__} has no mutation test in this file')


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
    cfg = broken(canonical, figs_period=501, batch_size=10, max_batch_size=10,
                 fused_grad_accum_min_samples=0)
    vs = check(cfg)
    severities = [v.severity for v in vs]
    assert ERROR in severities and BASELINE in severities
    assert severities == sorted(severities, key=lambda s: s != ERROR)
