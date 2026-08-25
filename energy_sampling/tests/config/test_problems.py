"""
Tests for configs/problems.yaml -- the problem registry.

THE POINT OF TESTING A REGISTRY. Its predecessor, mode_presets.yaml, declared
itself "Reference only -- never loaded by train.py". Nothing consumed it and
nothing checked it, so it drifted: by the time it was replaced it prescribed
seven RETIRED config keys and a learning-rate rule that had been deleted, and its
"current mk_dev state" column named a width and rollout length the config had not
used for weeks.

A registry nothing executes needs tests, or it becomes confident documentation of
a system that no longer exists.

Run: python -m pytest test_problems.py -q
"""

import ast
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
PROBLEMS = HERE / 'configs' / 'problems.yaml'
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'

# Keys a problem entry may carry. Anything else is a tuning knob that has leaked
# in, which is the failure mode this file exists to prevent.
ALLOWED = {
    'description', 'domain', 'conditioning', 'energy_function', 'space_groups',
    'z_primes', 'vector_conditioning', 'vector_conditioning_dim',
    'molecule_conditioning', 'embedding_conditioning',
    'embedding_conditioning_dim', 'prior_path', 'molecules_path',
    'test_molecules_path', 'temperature', 'analyze_kwargs', 'model', 'buffers',
    'protocol',
}

# The three conditioning mechanisms, and they are NOT interchangeable.
# `embedding_conditioning` was missing from this set until 2026-08-17, which is
# why the registry had no conditional MOLECULE problem: the live QM9 route
# conditions on a frozen Mo3ENet embedding and could not be described here at
# all, so the project's main experimental line was reachable only by hand-editing
# a config -- exactly what this file exists to end.
CONDITIONING_FLAGS = ('vector_conditioning', 'molecule_conditioning',
                      'embedding_conditioning')
# The only model/buffer sub-keys that follow from the DOMAIN rather than tuning.
ALLOWED_MODEL = {'periodic_centroids'}
ALLOWED_BUFFERS = {'anchor_buffer'}


@pytest.fixture(scope='module')
def registry():
    return yaml.safe_load(PROBLEMS.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def problems(registry):
    return registry['problems']


def test_registry_parses_and_is_versioned(registry):
    assert registry['schema'] == 1
    assert registry['problems']


@pytest.mark.parametrize('name', ['mipcas_elj', 'toy_big_unif',
                                  'toy_hard_uncond_multi', 'latent_gaussian'])
def test_expected_problems_are_present(problems, name):
    assert name in problems


def test_no_tuning_knobs_have_leaked_in(problems):
    """The rule that keeps this a registry. A key set differently per problem
    because it happened to be TUNED that way is a mode-safety defect in
    mk_dev.yaml, not a property of the problem."""
    for name, p in problems.items():
        extra = set(p) - ALLOWED
        assert not extra, f'{name} carries non-problem keys: {sorted(extra)}'
        extra_model = set(p.get('model') or {}) - ALLOWED_MODEL
        assert not extra_model, f'{name}.model carries tuning keys: {sorted(extra_model)}'
        extra_buf = set(p.get('buffers') or {}) - ALLOWED_BUFFERS
        assert not extra_buf, f'{name}.buffers carries tuning keys: {sorted(extra_buf)}'


def test_the_leak_check_would_catch_a_leak(problems):
    """Mutation: the check above passes trivially if ALLOWED is a superset of
    everything imaginable. A learning rate must be rejected."""
    leaked = dict(next(iter(problems.values())))
    leaked['lr_fused'] = 1e-4
    assert set(leaked) - ALLOWED == {'lr_fused'}


# ---------------------------------------------------------------------------
# Domain rules, stated once in the file's header and enforced here
# ---------------------------------------------------------------------------

def test_periodic_centroids_follows_the_domain(problems):
    """True for crystals (there is a cell to wrap), false for toys (there is
    not)."""
    for name, p in problems.items():
        want = p['domain'] == 'molecule'
        got = (p.get('model') or {}).get('periodic_centroids')
        assert got == want, f'{name}: periodic_centroids {got}, domain {p["domain"]}'


def test_toys_run_at_temperature_one(problems):
    for name, p in problems.items():
        if p['domain'] == 'toy':
            assert p['temperature'] == 1.0, name


def test_conditioning_flag_matches_the_declared_conditioning(problems):
    """`conditioning: conditional` must be backed by an actual flag, or the
    registry says one thing and the run does another."""
    for name, p in problems.items():
        declared = p['conditioning'] == 'conditional'
        flagged = any(p.get(f) for f in CONDITIONING_FLAGS)
        assert declared == flagged, f'{name}: declared {p["conditioning"]}, flags {flagged}'


def test_embedding_conditioning_dim_present_whenever_embedding_conditioning_is_on(problems):
    """The dim is not optional: the conditioner is built to it, and an absent
    value reads as 0 rather than as an error."""
    for name, p in problems.items():
        if p.get('embedding_conditioning'):
            assert p.get('embedding_conditioning_dim'), name


def test_exactly_one_conditioning_mechanism_per_problem(problems):
    """Two flags at once is not a richer problem, it is an ambiguous one -- the
    three mechanisms build different conditioners off different inputs."""
    for name, p in problems.items():
        on = [f for f in CONDITIONING_FLAGS if p.get(f)]
        assert len(on) <= 1, f'{name}: {on}'


def test_vector_conditioning_dim_present_whenever_vector_conditioning_is_on(problems):
    for name, p in problems.items():
        if p.get('vector_conditioning'):
            assert p.get('vector_conditioning_dim'), name


def test_conditional_problems_declare_a_test_set(problems):
    """R17: on conditional runs the held-out set is read FIRST, because train
    metrics can all improve on the same evaluation where held-out blows up. A
    conditional problem with no test set cannot be read that way."""
    for name, p in problems.items():
        if p['conditioning'] == 'conditional':
            assert p.get('test_molecules_path'), f'{name} has no held-out set'


def test_only_conditional_problems_name_an_alternate_protocol(problems):
    """The one structural exception. An unconditional problem naming a
    conditional protocol would schedule a stage whose loss is identically zero
    on it."""
    for name, p in problems.items():
        if p.get('protocol'):
            assert p['conditioning'] == 'conditional', name


# ---------------------------------------------------------------------------
# Agreement with the canonical config
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def canonical():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


def test_canonical_config_matches_its_problem_entry(canonical, problems):
    """mk_dev.yaml is written against `mipcas_elj`. If the two disagree, one of
    them is lying about what the canonical route is -- and the registry is the
    thing a generator will read."""
    p = problems['mipcas_elj']
    assert canonical['energy_function'] == p['energy_function']
    assert canonical['space_groups'] == p['space_groups']
    assert canonical['z_primes'] == p['z_primes']
    assert canonical['vector_conditioning'] == p['vector_conditioning']
    assert canonical['molecule_conditioning'] == p['molecule_conditioning']
    assert canonical['prior_path'] == p['prior_path']
    assert canonical['molecules_path'] == p['molecules_path']
    assert canonical['energy_config']['temperature'] == p['temperature']
    assert canonical['model']['periodic_centroids'] == p['model']['periodic_centroids']
    assert (canonical['buffers']['anchor_buffer']['seed_source']
            == p['buffers']['anchor_buffer']['seed_source'])


# ---------------------------------------------------------------------------
# The failure that killed the predecessor
# ---------------------------------------------------------------------------

def _retired_keys():
    tree = ast.parse((HERE / 'utils.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == '_RETIRED_KEYS' for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError('_RETIRED_KEYS not found')


def _all_keys(node):
    """Every mapping key in a nested structure."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(k)
            out |= _all_keys(v)
    elif isinstance(node, list):
        for v in node:
            out |= _all_keys(v)
    return out


def test_registry_sets_no_retired_key(registry):
    """mode_presets.yaml ended up prescribing SEVEN retired keys, each of which
    now hard-fails at load. A registry that recommends a key the schema rejects is
    worse than no registry.

    Checked against parsed KEYS, not raw text: a substring search over the file
    matches English prose too, and flagged this file for the word 'discovery' in
    a sentence about mode discovery. A check that fires on prose gets muted, and a
    muted check is not a check."""
    used = _all_keys(registry)
    retired_leaves = {k.split('.')[-1] for k in _retired_keys()}
    named = sorted(used & retired_leaves)
    assert not named, f'problems.yaml sets retired keys: {named}'


def test_the_retired_key_check_still_fires_on_a_real_one(registry):
    """Mutation: the parsed-key form must not have become vacuous. Injecting an
    actual retired key must be caught."""
    poisoned = {'problems': {'x': {'gpu_util_floor': 40}}}
    retired_leaves = {k.split('.')[-1] for k in _retired_keys()}
    assert _all_keys(poisoned) & retired_leaves == {'gpu_util_floor'}


def test_the_retired_key_check_ignores_prose(registry):
    """The other half: prose must NOT trip it, or the check gets muted."""
    prose_only = {'problems': {'x': {'description': 'obviates mode discovery'}}}
    retired_leaves = {k.split('.')[-1] for k in _retired_keys()}
    assert not (_all_keys(prose_only) & retired_leaves)


def test_registry_does_not_prescribe_the_deleted_lr_rule():
    """The anchor x 25/T rule is gone (utils.py). Its survival in a reference
    file is how it kept being applied after deletion."""
    text = PROBLEMS.read_text(encoding='utf-8')
    assert '25/T' not in text
