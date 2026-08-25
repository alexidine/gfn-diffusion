"""
Tests for config_snapshot -- the Phase 1 consolidation comparator.

THE COMPARATOR IS THE SAFETY NET FOR THE CANONICAL-CONFIG REWRITE, so it is
tested the way a safety net should be: every "it passes cleanly" case is paired
with a change it MUST catch. A comparator that reports no differences is only
worth something if it has been shown capable of reporting one.

Run: python -m pytest test_config_snapshot.py -q
"""

import copy
from pathlib import Path

import pytest
import yaml

import config_snapshot as cs

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


@pytest.fixture(scope='module')
def raw():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def base_snap():
    return cs.snapshot(str(CANONICAL))


def write(tmp_path, name, cfg):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return str(p)


def snap_of(tmp_path, name, cfg):
    return cs.snapshot(write(tmp_path, name, cfg))


# ---------------------------------------------------------------------------
# What the snapshot captures
# ---------------------------------------------------------------------------

def test_auto_values_are_resolved_not_left_as_auto(base_snap):
    """The point of snapshotting the RESOLVED config: `auto` is a promise about
    what the run will do, and comparing the promise instead of the resolved value
    would miss a change in the rule that resolves it.

    NB for lr_* keys the resolved number is the SEED the run starts from, not an
    operating point -- the ray-calibration servo owns it from there. It is
    captured because a changed seed is a changed run, not because it is where the
    run trains."""
    cfg = base_snap['config']
    assert cfg['lr_policy'] != 'auto'
    assert isinstance(cfg['lr_policy'], float)
    assert isinstance(cfg['gradient_norm_clip'], float)


def test_servo_management_is_recorded_per_key(base_snap):
    managed = base_snap['config']['lr_servo_managed']
    assert isinstance(managed, dict)
    assert managed == {'lr_policy': True, 'lr_replay': True,
                       'lr_back': True, 'lr_fused': True}


def test_auto_swapped_for_its_own_resolved_value_is_caught(tmp_path, raw, base_snap):
    """THE case the resolved value alone cannot see.

    `lr_policy: auto` and `lr_policy: 1.25e-4` resolve to the SAME number and mean
    opposite things -- bracket-managed versus a fixed rate the bracket's scale
    never touches. A comparator that only checked resolved values would wave this
    through."""
    seed = raw['lr_control']['seed_lr']
    new = copy.deepcopy(raw)
    new['lr_policy'] = seed                      # identical number, explicit
    cand = snap_of(tmp_path, 'explicit_lr.yaml', new)

    assert cand['config']['lr_policy'] == base_snap['config']['lr_policy'], \
        'fixture assumption: the resolved value must be unchanged'

    c = cs.compare(base_snap, cand)
    assert not c.behaviour_preserved
    changed = {p: (o, n) for p, o, n in c.changed}
    assert changed == {'config.lr_servo_managed.lr_policy': (True, False)}, changed


def test_leaving_auto_alone_does_not_move_servo_management(tmp_path, raw, base_snap):
    """The mutation for the test above: an unrelated edit must not perturb the
    servo map, or the check above would pass for the wrong reason."""
    new = copy.deepcopy(raw)
    new['batch_size'] = raw['batch_size'] + 1
    c = cs.compare(base_snap, snap_of(tmp_path, 'unrelated.yaml', new))
    assert not any('lr_servo_managed' in p for p, _, _ in c.changed)


def test_stages_are_captured_with_effective_coefficients(base_snap):
    stages = base_snap['stages']
    assert [s['name'] for s in stages] == ['train_prior', 'equilibration']
    for s in stages:
        eff = s['effective_loss_coeffs']
        assert isinstance(eff, dict), eff
        assert set(eff) == {'fwd', 'bwd', 'replay'}


def test_effective_coefficients_are_base_overlaid_with_stage(base_snap):
    """train_prior overrides bwd.mle to 1.0 while leaving beta at the base 10;
    equilibration overrides bwd.beta to 80. Both must show through."""
    by_name = {s['name']: s['effective_loss_coeffs'] for s in base_snap['stages']}
    assert by_name['train_prior']['bwd']['mle'] == 1.0
    assert by_name['train_prior']['bwd']['beta'] == 10.0     # base, not overridden
    assert by_name['equilibration']['bwd']['beta'] == 80     # stage override


def test_lr_sensor_is_captured_per_stage(base_snap):
    """The LR controller is a PER-STAGE choice and the kinds behave completely
    differently (ray / plateau / hyper / none), so it has to be in the snapshot
    or two configs that train nothing alike compare equal."""
    for s in base_snap['stages']:
        assert 'lr_sensor' in s


def test_dropping_an_lr_sensor_block_is_caught(tmp_path, raw, base_snap):
    """The silent case: omitting the block means 'no sensor', with no error. A
    consolidation that lost one must not pass clean."""
    new = copy.deepcopy(raw)
    new['protocols']['unconditional_tb']['stages'][1]['lr_sensor'] = {'kind': 'hyper', 'beta': 0.05}
    withsensor = snap_of(tmp_path, 'sensor.yaml', new)

    gained = cs.compare(base_snap, withsensor)
    assert not gained.behaviour_preserved
    assert any('lr_sensor' in p for p, _, _ in gained.changed)

    # and the reverse -- losing it -- is equally visible
    lost = cs.compare(withsensor, base_snap)
    assert not lost.behaviour_preserved
    assert any('lr_sensor' in p for p, _, _ in lost.changed)


def test_changing_the_sensor_kind_is_caught(tmp_path, raw, base_snap):
    """ray vs hyper vs plateau are different controllers, not variants of one."""
    def with_kind(node, name):
        new = copy.deepcopy(raw)
        new['protocols']['unconditional_tb']['stages'][1]['lr_sensor'] = node
        return snap_of(tmp_path, name, new)

    ray = with_kind({'kind': 'ray'}, 'ray.yaml')
    hyper = with_kind({'kind': 'hyper', 'beta': 0.05}, 'hyper.yaml')
    c = cs.compare(ray, hyper)
    assert not c.behaviour_preserved
    assert any('lr_sensor' in p for p, _, _ in c.changed)


def test_a_leaf_becoming_a_block_is_a_change_not_an_add(tmp_path, raw, base_snap):
    """The general form of the lr_sensor case, and it applies to any key.

    Flattening turns a scalar into one path and a block into several, so no path
    exists in both and the naive comparison files removed+added with CHANGED
    empty -- which reads as 'behaviour preserved'. Half the interesting config
    edits have this shape."""
    new = copy.deepcopy(raw)
    new['ema_decay'] = {'kind': 'something', 'rate': 0.9}   # was a scalar (null)
    c = cs.compare(base_snap, snap_of(tmp_path, 'shape.yaml', new))
    assert not c.behaviour_preserved, 'a scalar -> block edit must not pass clean'
    assert any(p == 'config.ema_decay' for p, _, _ in c.changed)


def test_a_block_becoming_a_leaf_is_also_caught(tmp_path, raw, base_snap):
    new = copy.deepcopy(raw)
    new['integrator'] = 10                     # was {T: 10}
    c = cs.compare(base_snap, snap_of(tmp_path, 'shape2.yaml', new))
    assert not c.behaviour_preserved
    assert any(p == 'config.integrator' for p, _, _ in c.changed)


def test_periodic_centroid_axes_are_resolved(base_snap):
    """The derived quantity that sets the model's expanded_dim. Recorded as a
    membership map because it is a SET -- see the note in _periodic_centroid_axes."""
    assert base_snap['config']['periodic_centroid_axes'] == {
        'axis_0': False, 'axis_1': True, 'axis_2': True}          # SG 2 wraps 1,2


def test_changing_the_space_group_changes_the_wrapped_axes(tmp_path, raw, base_snap):
    """SG 2 wraps [1, 2]; SG 14 wraps [0, 2]. Same COUNT, different axes -- so a
    space-group change alters which dims the SDE wraps, and the raw
    `space_groups` diff alone does not show that consequence."""
    new = copy.deepcopy(raw)
    new['space_groups'] = [14]
    c = cs.compare(base_snap, snap_of(tmp_path, 'sg14.yaml', new))
    assert not c.behaviour_preserved
    moved = {p: (o, n) for p, o, n in c.changed if 'periodic_centroid_axes' in p}
    # SG 2 wraps {1,2}; SG 14 wraps {0,2}. Axis 2 is common, so exactly axes 0
    # and 1 flip -- and the diff NAMES them rather than reporting index shifts.
    assert moved == {
        'config.periodic_centroid_axes.axis_0': (False, True),
        'config.periodic_centroid_axes.axis_1': (True, False),
    }, c.changed


def test_turning_periodic_centroids_off_is_distinguishable_from_no_axes(tmp_path, raw,
                                                                        base_snap):
    """'feature off' and 'this SG has no full-width axes' both produce an empty
    wrap but are different states; collapsing them to None would compare equal."""
    new = copy.deepcopy(raw)
    new['model']['periodic_centroids'] = False
    new['model']['dplr_mask_angular'] = False   # required only while periodic dims exist
    cand = snap_of(tmp_path, 'noperiodic.yaml', new)
    assert cand['config']['periodic_centroid_axes'] == 'off'
    c = cs.compare(base_snap, cand)
    assert any(p == 'config.periodic_centroid_axes' for p, _, _ in c.changed)


def test_multiple_space_groups_are_recorded_not_raised(tmp_path, raw):
    """periodic_centroids makes the model space-group specific, so >1 is refused
    at construction. The snapshot records that rather than dying -- it exists to
    explain a broken config, not to fail alongside it."""
    new = copy.deepcopy(raw)
    new['space_groups'] = [2, 14]
    snap = snap_of(tmp_path, 'twosg.yaml', new)
    assert 'INVALID' in snap['config']['periodic_centroid_axes']


def test_an_unloadable_reference_is_reported_not_raised(tmp_path, raw, base_snap):
    """The comparator must survive an invalid REFERENCE.

    The reference is normally the committed config, and the usual reason to
    tighten a rule is that the committed config violates it -- so the tool would
    die in exactly the case it exists for. Here: drop `lr_control`, which leaves
    the `auto` learning rates with nothing to move them and the config
    unloadable."""
    stale = copy.deepcopy(raw)
    stale.pop('lr_control', None)
    old = snap_of(tmp_path, 'stale_ref.yaml', stale)

    assert 'load_error' in old
    assert 'nothing can move them' in old['load_error']

    c = cs.compare(old, base_snap)
    assert not c.behaviour_preserved, 'no comparison happened, so nothing is preserved'
    assert c.reference_error and not c.candidate_error
    assert 'DOES NOT LOAD' in c.render()


def test_an_unloadable_candidate_is_reported_separately(tmp_path, raw, base_snap):
    stale = copy.deepcopy(raw)
    stale.pop('lr_control', None)
    c = cs.compare(base_snap, snap_of(tmp_path, 'stale_cand.yaml', stale))
    assert c.candidate_error and not c.reference_error
    assert 'CANDIDATE DOES NOT LOAD' in c.render()


def test_a_loadable_pair_reports_no_load_error(base_snap):
    """Mutation: the two tests above would pass on a comparator that flagged
    everything as unloadable."""
    c = cs.compare(base_snap, base_snap)
    assert c.reference_error is None and c.candidate_error is None
    assert c.behaviour_preserved


def test_flatten_indexes_lists():
    flat = cs.flatten({'a': [{'b': 1}, {'b': 2}]})
    assert flat == {'a[0].b': 1, 'a[1].b': 2}


# ---------------------------------------------------------------------------
# Clean cases
# ---------------------------------------------------------------------------

def test_identical_configs_compare_clean(base_snap):
    c = cs.compare(base_snap, base_snap)
    assert c.behaviour_preserved
    assert not c.changed and not c.added and not c.removed


def test_reordering_keys_is_not_a_change(tmp_path, raw, base_snap):
    """The case that makes a text diff useless during consolidation: the file is
    rewritten top to bottom and nothing about the run changes."""
    reordered = {k: raw[k] for k in sorted(raw)}
    c = cs.compare(base_snap, snap_of(tmp_path, 'reordered.yaml', reordered))
    assert c.behaviour_preserved
    assert not c.changed


def test_adding_an_inert_block_is_added_not_changed(tmp_path, raw, base_snap):
    """Consolidation ADDS keys by design -- the conditional settings that
    coexist with the unconditional ones. Those must not read as behaviour
    changes, or the signal is buried under the intended work."""
    new = copy.deepcopy(raw)
    new['a_block_for_an_inactive_mode'] = {'x': 1, 'y': 2}
    c = cs.compare(base_snap, snap_of(tmp_path, 'added.yaml', new))
    assert c.behaviour_preserved
    assert not c.changed
    assert any('a_block_for_an_inactive_mode' in p for p in c.added)


# ---------------------------------------------------------------------------
# ...and the changes it must catch
# ---------------------------------------------------------------------------

def test_a_changed_scalar_is_caught(tmp_path, raw, base_snap):
    new = copy.deepcopy(raw)
    new['batch_size'] = raw['batch_size'] + 1
    c = cs.compare(base_snap, snap_of(tmp_path, 'scalar.yaml', new))
    assert not c.behaviour_preserved
    assert any(p == 'config.batch_size' for p, _, _ in c.changed)


def test_a_changed_stage_override_is_caught_twice(tmp_path, raw, base_snap):
    """Once in the raw config and once in the effective coefficients. Both
    matter: the raw hit localises the edit, the effective hit proves it reaches
    the trainer."""
    new = copy.deepcopy(raw)
    new['protocols']['unconditional_tb']['stages'][1]['loss_coeffs']['bwd']['beta'] = 40
    c = cs.compare(base_snap, snap_of(tmp_path, 'override.yaml', new))
    assert not c.behaviour_preserved
    paths = [p for p, _, _ in c.changed]
    assert any('stages[1].loss_coeffs.bwd.beta' in p for p in paths)
    assert any('effective_loss_coeffs.bwd.beta' in p for p in paths)


def test_a_changed_base_coefficient_shows_in_the_effective_view(tmp_path, raw, base_snap):
    """THE case a raw-config diff alone would under-report. Changing a base block
    value silently re-bases every stage that does not override it -- here
    train_prior's bwd beta moves while equilibration's, which overrides it,
    does not."""
    new = copy.deepcopy(raw)
    new['bwd_loss_coeffs']['beta'] = 20.0        # base was 10.0; equilibration pins 80
    c = cs.compare(base_snap, snap_of(tmp_path, 'basecoeff.yaml', new))
    assert not c.behaviour_preserved
    eff = {p: (o, n) for p, o, n in c.changed if 'effective_loss_coeffs' in p}
    assert any('stages[0]' in p and p.endswith('bwd.beta') for p in eff), eff
    assert not any('stages[1]' in p and p.endswith('bwd.beta') for p in eff), \
        'equilibration overrides beta, so its effective value must NOT move'


def test_a_removed_key_is_caught(tmp_path, raw, base_snap):
    new = copy.deepcopy(raw)
    del new['seed']
    c = cs.compare(base_snap, snap_of(tmp_path, 'removed.yaml', new))
    assert 'config.seed' in c.removed


def test_a_changed_derived_value_is_caught(tmp_path, raw, base_snap):
    """gradient_norm_clip is `auto`, derived from (W, T). Changing T must move it
    -- if the snapshot stored the string `auto` this would compare equal."""
    new = copy.deepcopy(raw)
    new['integrator']['T'] = 25
    new['eval_T'] = 25                      # preflight requires these to agree
    c = cs.compare(base_snap, snap_of(tmp_path, 'derived.yaml', new))
    assert any(p == 'config.gradient_norm_clip' for p, _, _ in c.changed)


# ---------------------------------------------------------------------------
# Direction and classification
# ---------------------------------------------------------------------------

def test_direction_is_not_inverted(tmp_path, raw, base_snap):
    """compare(reference, candidate): a key the candidate gained is ADDED, not
    REMOVED. Inverting this reads as 'the consolidation deleted the block it just
    added', which is wrong in exactly the direction that matters."""
    new = copy.deepcopy(raw)
    new['brand_new_key'] = 1
    cand = snap_of(tmp_path, 'dir.yaml', new)
    assert 'config.brand_new_key' in cs.compare(base_snap, cand).added
    assert 'config.brand_new_key' in cs.compare(cand, base_snap).removed


def test_environment_keys_are_separated_from_behaviour(tmp_path, raw, base_snap):
    """A checkpoints_dir difference must not sit in the same list as a changed
    learning rate, and must not fail the comparison."""
    new = copy.deepcopy(raw)
    new['checkpoints_dir'] = r'D:\somewhere\else'
    c = cs.compare(base_snap, snap_of(tmp_path, 'env.yaml', new))
    assert c.behaviour_preserved
    assert any(p == 'config.checkpoints_dir' for p, _, _ in c.environment)
    assert not any(p == 'config.checkpoints_dir' for p, _, _ in c.changed)


def test_added_keys_alone_do_not_fail_the_comparison(tmp_path, raw, base_snap):
    """behaviour_preserved is about CHANGED only. Proving an added key is inert
    is the mode-safety test's job; conflating the two would make a passing
    comparison mean less than it appears to."""
    new = copy.deepcopy(raw)
    new['another_new_block'] = {'k': 'v'}
    assert cs.compare(base_snap, snap_of(tmp_path, 'addonly.yaml', new)).behaviour_preserved


def test_render_names_the_changed_paths(tmp_path, raw, base_snap):
    new = copy.deepcopy(raw)
    new['batch_size'] = 7
    text = cs.compare(base_snap, snap_of(tmp_path, 'render.yaml', new)).render()
    assert 'config.batch_size' in text
    assert 'alter behaviour' in text
