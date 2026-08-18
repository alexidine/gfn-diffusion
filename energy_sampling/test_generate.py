"""
Tests for `configs/generate.py` -- production config generation (Phase 2.1).

WHAT THESE HAVE TO CATCH, in priority order:

  1. PROVENANCE MUST BE INERT. The generator adds a key to every config it
     writes. This codebase's signature defect is a flag that looks inert and is
     not, so inertness is PROVEN here by resolving the same config with and
     without the block and comparing what a run would read -- not asserted in a
     comment. If provenance ever changes a resolved value, every arm the
     generator has written is a different experiment from the one intended.
  2. AN UNLOADABLE ARM MUST BE FATAL. It was not, and the way it failed is the
     one this project keeps paying for: `config_snapshot.snapshot` reports a load
     error instead of raising, so an unloadable candidate has empty
     changed/added/removed and the deviation count printed "0 deviations from
     canonical" -- a config that could never train, reading as perfectly clean.
  3. THE PROBLEM REGISTRY'S TRANSLATION IS REAL. problems.yaml is written flat
     and the config groups by consumer, so `temperature` has to land at
     `energy_config.temperature`. Nothing loaded problems.yaml before this
     module, so this mapping has never been exercised.
"""

import copy
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'configs'))

import config_invariants                       # noqa: E402
import config_snapshot                         # noqa: E402
import generate                                # noqa: E402

pytestmark = pytest.mark.fast


@pytest.fixture(scope='module')
def canonical():
    return generate.canonical()


# ------------------------------------------------- provenance is inert, proven

def test_provenance_changes_nothing_a_run_reads(tmp_path):
    """THE ONE THAT MATTERS. Resolve the canonical config with and without the
    provenance block and require every value a run reads to be identical.

    Asserting "nothing reads it" would be the same class of claim as the config
    comments this project has spent a week correcting. This resolves both files
    through the real loader and compares."""
    plain = tmp_path / 'plain.yaml'
    stamped = tmp_path / 'stamped.yaml'
    cfg = generate.canonical()
    plain.write_text(yaml.safe_dump(cfg, sort_keys=False))
    stamped.write_text(yaml.safe_dump(generate.stamp(cfg), sort_keys=False))

    cmp = config_snapshot.compare(config_snapshot.snapshot(str(plain)),
                                  config_snapshot.snapshot(str(stamped)))
    assert not cmp.candidate_error and not cmp.reference_error
    assert cmp.changed == [], (
        f'the provenance block changed {len(cmp.changed)} resolved value(s): '
        f'{cmp.changed[:5]}')
    assert cmp.removed == []
    # It IS present -- otherwise this test passes by the block not existing.
    assert generate.PROVENANCE_KEY in yaml.safe_load(stamped.read_text())


def test_the_stamp_records_the_canonical_it_was_built_from(canonical):
    """`canonical_sha` is the load-bearing field: it answers "was this arm built
    from the file I am looking at now", which is what went unanswerable when a
    battery's base moved under its arms."""
    prov = generate.arm('x', problem='mipcas_elj')[generate.PROVENANCE_KEY]
    assert prov['canonical_sha'] == generate.canonical_hash()
    assert prov['project_state_version'] == canonical['project_state_version']
    assert prov['problem'] == 'mipcas_elj'


def test_deviation_summary_excludes_provenance(tmp_path):
    """Provenance differs from canonical on every arm by construction. A summary
    whose first entries are always noise is one people stop reading."""
    arms = {'a': generate.arm('a', problem='mipcas_elj')}
    generate.emit(arms, outdir=tmp_path, quiet=True, index=False)
    cmp = generate.deviations(tmp_path / 'a.yaml')
    touched = [k for k in cmp.added + cmp.removed] + [c[0] for c in cmp.changed]
    assert not any(generate.PROVENANCE_KEY in k for k in touched), touched


# ------------------------------------------------- an unloadable arm is fatal

def test_an_unloadable_arm_is_fatal_not_zero_deviations(tmp_path):
    """THE BUG. `integrator.T` without a matching `eval_T` is refused at load.
    Before the fix this wrote the file and reported '0 deviations'."""
    bad = generate.arm('bad', problem='latent_gaussian', **{'integrator.T': 25})
    with pytest.raises(SystemExit, match='DOES NOT LOAD'):
        generate.emit({'bad': bad}, outdir=tmp_path, quiet=True, index=False)


def test_the_same_arm_with_the_matching_key_is_accepted(tmp_path):
    """MUTATION IN THE PASSING DIRECTION. Without this the fatal rule could be
    rejecting every arm and the test above would not notice."""
    good = generate.arm('good', problem='latent_gaussian', eval_T=25,
                        **{'integrator.T': 25})
    written = generate.emit({'good': good}, outdir=tmp_path, quiet=True, index=False)
    assert len(written) == 1 and written[0].exists()


# --------------------------------------------------------------- the gate

def test_config_ERROR_refuses_generation(tmp_path):
    """An ERROR is a self-contradiction provable from the file. Generation is
    where that should cost a rerun rather than a queue slot."""
    cfg = generate.arm('e', problem='mipcas_elj', protocol='no_such_protocol')
    with pytest.raises(SystemExit):
        generate.emit({'e': cfg}, outdir=tmp_path, quiet=True, index=False)


def test_a_BASELINE_departure_is_reported_not_refused(tmp_path, capsys):
    """A config written to depart from a measured default is the normal shape of
    an experiment. `effective_batch_meets_baseline` is a BASELINE rule."""
    # Accumulation is what the rule actually measures, so a small batch alone
    # does not depart from the baseline -- the effective optimization batch is
    # what counts (test_config_invariants pins that separately).
    cfg = generate.arm('b', problem='mipcas_elj', batch_size=8, max_batch_size=8,
                       fused_grad_accum_min_samples=8)
    written = generate.emit({'b': cfg}, outdir=tmp_path, index=False)
    assert written, 'a baseline departure must still generate'
    assert 'baseline note' in capsys.readouterr().out


# --------------------------------------------------- the problem registry

def test_problem_metadata_never_reaches_the_config():
    """`description`/`domain`/`conditioning` document the problem; a config
    carrying them is carrying prose into a run."""
    cfg = generate.arm('m', problem='mipcas_elj')
    for key in generate._PROBLEM_METADATA:
        assert key not in cfg, f'{key} leaked into the config'


@pytest.mark.parametrize('problem', ['mipcas_elj', 'latent_gaussian'])
def test_flat_registry_keys_land_at_their_config_paths(problem):
    """THE TRANSLATION NOTHING HAD EXERCISED. problems.yaml says `temperature`;
    the config wants `energy_config.temperature`."""
    raw = generate.problems()[problem]
    cfg = generate.arm('t', problem=problem)
    if 'temperature' in raw:
        assert cfg['energy_config']['temperature'] == raw['temperature']
        assert 'temperature' not in cfg, 'left at the flat path as well'
    if 'analyze_kwargs' in raw:
        assert cfg['energy_config']['analyze_kwargs'] == raw['analyze_kwargs']


def test_problem_sets_the_keys_that_define_it():
    """A problem that does not change energy_function is not a problem choice."""
    elj = generate.arm('a', problem='mipcas_elj')
    gauss = generate.arm('b', problem='latent_gaussian')
    assert elj['energy_function'] != gauss['energy_function']
    assert gauss['model']['periodic_centroids'] is False


def test_an_unknown_problem_names_the_ones_that_exist():
    with pytest.raises(SystemExit, match='latent_gaussian'):
        generate.arm('x', problem='no_such_problem')


# ------------------------------------------------------------- merge semantics

def test_an_override_beats_the_problem_block():
    """ORDER IS THE CONTRACT: override > problem > canonical."""
    cfg = generate.arm('o', problem='latent_gaussian', energy_function='elj')
    assert cfg['energy_function'] == 'elj'


def test_a_problem_beats_canonical(canonical):
    cfg = generate.arm('p', problem='latent_gaussian')
    assert canonical['energy_function'] != 'latent_gaussian'
    assert cfg['energy_function'] == 'latent_gaussian'


def test_lists_replace_rather_than_extend():
    """`space_groups`, `alphas` and `bounds` are sequences whose meaning is the
    whole list; extending one produces a value nobody wrote."""
    out = generate.merge({'space_groups': [2, 14]}, {'space_groups': [1]})
    assert out['space_groups'] == [1]


def test_merge_does_not_mutate_its_input():
    base = {'a': {'b': 1}}
    generate.merge(base, {'a': {'b': 2}})
    assert base == {'a': {'b': 1}}


def test_dotted_override_refuses_to_tunnel_through_a_non_dict():
    """`lr_sensor.beta` on a stage whose lr_sensor is None would otherwise
    replace the None with a dict carrying only `beta` -- a sensor with no kind."""
    with pytest.raises(ValueError, match='not a section'):
        generate.set_dotted({'lr_sensor': 3.0}, 'lr_sensor.beta', 0.1)


def test_dotted_and_nested_overrides_reach_the_same_place():
    a = generate.arm('a', problem='mipcas_elj', **{'integrator.T': 7, 'eval_T': 7})
    b = generate.arm('b', problem='mipcas_elj', integrator={'T': 7}, eval_T=7)
    assert a['integrator']['T'] == b['integrator']['T'] == 7
    assert a['integrator'].keys() == b['integrator'].keys(), 'dotted set lost siblings'


# ------------------------------------------------------------------- the stamp

def test_canonical_refuses_a_state_mismatch(monkeypatch):
    """A canonical config at a different state than the code means one moved
    without the other, and every arm between would stamp a version that lies."""
    import config_state
    monkeypatch.setattr(config_state, 'CURRENT_STATE_VERSION',
                        config_state.CURRENT_STATE_VERSION + 1)
    with pytest.raises(SystemExit, match='state'):
        generate.canonical()


def test_index_records_what_identifies_an_arm(tmp_path):
    # Real epochs: a 10-step run trips exit_patience_is_reachable (an ERROR),
    # which is the gate working and not what this test is about.
    arms = {'a1': generate.arm('a1', problem='mipcas_elj', epochs=2000),
            'a2': generate.arm('a2', problem='latent_gaussian', epochs=3000)}
    generate.emit(arms, outdir=tmp_path, quiet=True)
    rows = (tmp_path / 'INDEX.tsv').read_text().strip().splitlines()
    assert rows[0].split('\t')[:2] == ['name', 'problem']
    assert len(rows) == 3
    assert generate.canonical_hash() in rows[1]
