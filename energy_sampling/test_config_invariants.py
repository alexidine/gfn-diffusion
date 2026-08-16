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


def test_growth_gain_at_the_factor_freezes_the_batch(canonical):
    # factor 1.65 -> anything >= 0.65 makes every jump unachievable
    assert _fires(broken(canonical, batch_growth_min_throughput_gain=0.65),
                  'growth_gain_below_growth_factor')
    assert _fires(broken(canonical, batch_growth_min_throughput_gain=0.9),
                  'growth_gain_below_growth_factor')
    # just below is fine
    assert not _fires(broken(canonical, batch_growth_min_throughput_gain=0.6),
                      'growth_gain_below_growth_factor')


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
    stage = next(s for s in cfg['protocol']['stages'] if 'deactivate_threshold' in s)
    stage['deactivate_threshold'] = 0.5
    assert _fires(cfg, 'deactivate_threshold_is_sane')


def test_pinned_frac_disagreeing_with_fracs(canonical):
    cfg = copy.deepcopy(canonical)
    stage = next(s for s in cfg['protocol']['stages']
                 if (s.get('balance') or {}).get('pinned'))
    mode = next(iter(stage['balance']['pinned']))
    stage['balance']['pinned'][mode] = stage['fracs'][mode] + 0.1
    assert _fires(cfg, 'pinned_frac_matches_fracs')


def test_auto_lr_with_no_sensor_anywhere_is_an_error(canonical):
    """`auto` claims a servo owns the rate. With no adaptive sensor nothing moves
    peak_scale off 1.0, so the run trains at the seed for its whole life while the
    config reads as adaptive."""
    cfg = copy.deepcopy(canonical)
    for st in cfg['protocol']['stages']:
        st.pop('lr_sensor', None)
    assert _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


def test_auto_lr_with_sensor_kind_none_is_also_an_error(canonical):
    """`{kind: none}` and an omitted block mean the same thing. Only the second
    is silent, but neither owns an `auto` rate."""
    cfg = copy.deepcopy(canonical)
    for st in cfg['protocol']['stages']:
        st['lr_sensor'] = {'kind': 'none'}
    assert _fires(cfg, 'auto_lr_requires_an_adaptive_sensor')


def test_one_stage_missing_a_sensor_still_fires(canonical):
    """Checked PER STAGE: a sensor on the terminal stage does nothing for a
    phase-1 learning rate."""
    cfg = copy.deepcopy(canonical)
    cfg['protocol']['stages'][0].pop('lr_sensor', None)
    v = [x for x in check(cfg) if x.rule == 'auto_lr_requires_an_adaptive_sensor']
    assert len(v) == 1 and repr(cfg['protocol']['stages'][0]['name']) in v[0].detail


def test_explicit_float_lrs_need_no_sensor(canonical):
    """The other half of the rule: a float is a fixed peak that takes the warmup
    envelope and divergence handling only, so it has nothing to yield to and must
    not be flagged."""
    cfg = copy.deepcopy(canonical)
    for k in ('lr_policy', 'lr_back', 'lr_replay', 'lr_fused'):
        cfg[k] = 3.0e-4
    for st in cfg['protocol']['stages']:
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
    for st in resolved['protocol']['stages']:
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
    cfg['protocol']['stages'][0]['lr_sensor'] = {'kind': 'ray'}   # train_prior is bwd
    assert _fires(cfg, 'ray_sensor_needs_a_coherent_stage')


def test_ray_on_a_fused_stage_without_replay_tb_is_an_error(canonical):
    cfg = copy.deepcopy(canonical)
    st = cfg['protocol']['stages'][1]
    assert st['train_mode'] == 'fused', 'fixture assumption'
    st.setdefault('loss_coeffs', {}).setdefault('replay', {})['tb'] = 0.0
    assert _fires(cfg, 'ray_sensor_needs_a_coherent_stage')


def test_hyper_is_accepted_on_a_non_fused_stage(canonical):
    """`hyper` reads no loss, so unlike `ray` it is coherent whatever the stage
    trains. Flagging it would push people back to omitting the block."""
    cfg = copy.deepcopy(canonical)
    cfg['protocol']['stages'][0]['lr_sensor'] = {'kind': 'hyper', 'beta': 0.05}
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


def test_effective_batch_below_baseline_is_a_baseline_not_an_error(canonical):
    cfg = broken(canonical, batch_size=100, max_batch_size=100,
                 fused_grad_accum_min_samples=0)
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
    cfg = {'batch_growth_min_throughput_gain': 'auto', 'batch_growth_factor': 'auto',
           'gradient_norm_clip': 'auto', 'batch_size': 'auto'}
    assert check(cfg) == []


def test_severity_ordering_puts_errors_first(canonical):
    cfg = broken(canonical, figs_period=501, batch_size=10, max_batch_size=10,
                 fused_grad_accum_min_samples=0)
    vs = check(cfg)
    severities = [v.severity for v in vs]
    assert ERROR in severities and BASELINE in severities
    assert severities == sorted(severities, key=lambda s: s != ERROR)
