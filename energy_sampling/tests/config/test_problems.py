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
# Buffer LEAVES a selected PROTOCOL refuses to start without, carried beside
# `protocol` for the same reason the protocol is. Each is the only value its
# protocol accepts, so it is not a tuning knob: path under `buffers` -> value.
PROTOCOL_REQUIRED_BUFFERS = {
    # conditional_vargrad's var_conditioning refuses a prioritised replay draw
    ('replay_buffer', 'prioritise', 'enabled'): False,
}


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, node


def _buffer_leaks(p):
    """Buffer keys outside ALLOWED_BUFFERS that are not a protocol-required leaf
    at its required value."""
    out = []
    for block, node in (p.get('buffers') or {}).items():
        if block in ALLOWED_BUFFERS:
            continue
        for path, value in _leaves(node, (block,)):
            if PROTOCOL_REQUIRED_BUFFERS.get(path, object()) != value:
                out.append('.'.join(path))
    return out


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
        extra_buf = _buffer_leaks(p)
        assert not extra_buf, f'{name}.buffers carries tuning keys: {sorted(extra_buf)}'


def test_the_leak_check_would_catch_a_leak(problems):
    """Mutation: the check above passes trivially if ALLOWED is a superset of
    everything imaginable. A learning rate must be rejected."""
    leaked = dict(next(iter(problems.values())))
    leaked['lr_fused'] = 1e-4
    assert set(leaked) - ALLOWED == {'lr_fused'}


def test_the_buffer_leak_check_admits_only_the_required_leaf_at_its_value():
    """Mutation, for the protocol-required exemption: it admits ONE leaf at ONE
    value. A replay tuning key beside it, the leaf at the other value, or a
    block nothing requires must all still read as leaks."""
    required = {'buffers': {'anchor_buffer': {'seed_source': 'prior_dataset'},
                            'replay_buffer': {'prioritise': {'enabled': False}}}}
    assert _buffer_leaks(required) == []
    tuned = {'buffers': {'replay_buffer': {'prioritise': {'enabled': False},
                                           'max_size': 150000}}}
    assert _buffer_leaks(tuned) == ['replay_buffer.max_size']
    flipped = {'buffers': {'replay_buffer': {'prioritise': {'enabled': True}}}}
    assert _buffer_leaks(flipped) == ['replay_buffer.prioritise.enabled']
    assert _buffer_leaks({'buffers': {'prior_buffer': {'source': 'anchors'}}}) == \
        ['prior_buffer.source']


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


def test_no_problem_declares_a_data_path_the_trainer_cannot_load(problems):
    """`prior_path` and `molecules_path` are torch.loaded with no null branch --
    `Modeller._load_condition_file` from `init_mol_dataset`, and
    `init_prior_dataset` a few lines later. A null one is a registry entry that
    generates a config which validates clean and dies at startup with
    `'NoneType' object has no attribute 'seek'`, naming neither the key nor the
    file. `latent_gaussian` carried both until 2026-09-09.

    Stated here as well as in `config_invariants.loaded_data_paths_are_not_null`
    because the two catch it at different moments: the invariant catches an ARM,
    this catches the REGISTRY, which is where the entry has to be fixed. Every
    problem here is a crystal-route problem; the conformer route overrides both
    data-init methods and is not described in this file."""
    for name, p in problems.items():
        for key in ('prior_path', 'molecules_path'):
            assert p.get(key), f'{name}.{key} is {p.get(key)!r}'


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
# latent_gaussian must agree with the target's single source of truth
#
# `configs/gauss_aug12/spec.py` is imported by the prior generator, that
# battery's config generator and the closed-form check alike, precisely so the
# file on disk, the config that scores it and the number it is compared against
# cannot drift. The registry is now a fourth reader of the same target, and it is
# the one nothing else validates -- a prior drawn at one width and scored at
# another trains perfectly well and reports a wrong log Z, with nothing to see in
# either file on its own.
#
# READ BY `ast`, NOT IMPORTED, for the same reason `_retired_keys` above is:
# `spec.dead_rows` resolves through `mxtaltools`, which pulls torch, and this
# file is a fast-lane tier floor (tests/config/test_tiering.py KNOWN_CHEAP). The
# one assertion that genuinely needs the resolver -- that the zeros in `c` sit on
# THIS space group's dead rows -- lives in tests/crystal/test_latent_gaussian.py,
# which is already a torch suite about this exact target.
# ---------------------------------------------------------------------------

SPEC = HERE / 'configs' / 'gauss_aug12' / 'spec.py'


@pytest.fixture(scope='module')
def spec_constants():
    """Module-level literal assignments in gauss_aug12/spec.py."""
    tree = ast.parse(SPEC.read_text(encoding='utf-8'))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                out[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_the_spec_constants_are_actually_readable(spec_constants):
    """Mutation guard: an `ast` scan that silently found nothing would make every
    assertion below vacuous."""
    for key in ('WIDTH', 'MODE', 'DIM', 'PRIOR_DIR', 'PRIOR_STEM'):
        assert key in spec_constants, f'{key} not readable from {SPEC}'


def test_latent_gaussian_points_at_the_prior_its_own_space_group_names(
        problems, spec_constants):
    """The path is not free: the dead rows are baked into the prior at draw time,
    so `space_groups` and `prior_path` choose each other. Both path keys carry the
    same file deliberately -- for this toy the prior IS the target, and
    prep_prior.py saves one batch under both 'prior' and 'equalized_prior'."""
    p = problems['latent_gaussian']
    assert len(p['space_groups']) == 1
    sg = p['space_groups'][0]
    want = (spec_constants['PRIOR_DIR'].rstrip('\\') + '\\'
            + spec_constants['PRIOR_STEM'].format(sg=sg) + '.pt')
    assert p['prior_path'] == want
    assert p['molecules_path'] == p['prior_path']


def test_latent_gaussian_scores_at_the_width_its_prior_was_drawn_at(
        problems, spec_constants):
    kwargs = problems['latent_gaussian']['analyze_kwargs']
    assert kwargs['width'] == spec_constants['WIDTH']


def test_latent_gaussian_centres_on_the_spec_mode(problems, spec_constants):
    """`c` is MODE on live rows and the canonical 0.0 on dead ones, over the full
    latent width. MODE is 0.5 rather than 0 on purpose: at 0 the target sits on
    the SDE's own origin and a mis-indexed dead row is invisible, because correct
    and swapped both look 0-centred. So a `c` that has lost its mode is not a
    cosmetic difference -- it disarms the check the whole target is built around.
    An empty `analyze_kwargs`, which is what this entry carried until 2026-09-09,
    is exactly that case: `latent_harmonic_en` defaults `c` to zeros."""
    c = problems['latent_gaussian']['analyze_kwargs']['c']
    assert len(c) == spec_constants['DIM']
    assert set(c) == {spec_constants['MODE'], 0.0}, c


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


def _key_paths(node, prefix=''):
    """Every mapping key in a nested structure, as its dotted path."""
    out = set()
    if isinstance(node, dict):
        for k, v in node.items():
            path = f'{prefix}.{k}' if prefix else str(k)
            out.add(path)
            out |= _key_paths(v, path)
    elif isinstance(node, list):
        for v in node:
            out |= _key_paths(v, prefix)
    return out


def _retired_hits(node):
    """Retired keys set anywhere in `node`. A retired key is a dotted config
    path (`z_calibration.enabled`), so a key matches when its own path ENDS in
    the whole retired path -- not on the leaf alone, which would flag every
    `enabled` in the file for `z_calibration.enabled`. An undotted retired key
    still matches at any depth, as the leaf check did."""
    retired = _retired_keys()
    return sorted(k for k in retired for p in _key_paths(node)
                  if p == k or p.endswith('.' + k))


def test_registry_sets_no_retired_key(registry):
    """mode_presets.yaml ended up prescribing SEVEN retired keys, each of which
    now hard-fails at load. A registry that recommends a key the schema rejects is
    worse than no registry.

    Checked against parsed KEYS, not raw text: a substring search over the file
    matches English prose too, and flagged this file for the word 'discovery' in
    a sentence about mode discovery. A check that fires on prose gets muted, and a
    muted check is not a check."""
    named = _retired_hits(registry)
    assert not named, f'problems.yaml sets retired keys: {named}'


def test_the_retired_key_check_still_fires_on_a_real_one(registry):
    """Mutation: the parsed-key form must not have become vacuous. Injecting an
    actual retired key must be caught -- undotted at any depth, and dotted at
    its own path."""
    poisoned = {'problems': {'x': {'gpu_util_floor': 40}}}
    assert _retired_hits(poisoned) == ['gpu_util_floor']
    poisoned = {'problems': {'x': {'z_calibration': {'enabled': True}}}}
    assert _retired_hits(poisoned) == ['z_calibration.enabled']


def test_the_retired_key_check_ignores_prose(registry):
    """The other half: prose must NOT trip it, or the check gets muted."""
    prose_only = {'problems': {'x': {'description': 'obviates mode discovery'}}}
    assert not _retired_hits(prose_only)


def test_the_retired_key_check_reads_a_dotted_key_by_its_path(registry):
    """The same leaf under another block is a different key: qm9_conditional's
    buffers.replay_buffer.prioritise.enabled is live, while every retired
    `*.enabled` names another block."""
    live = {'problems': {'x': {'buffers': {'replay_buffer': {'prioritise': {'enabled': False}}}}}}
    assert any(k.split('.')[-1] == 'enabled' for k in _retired_keys()), \
        'precondition: a retired key ends in enabled'
    assert not _retired_hits(live)


def test_registry_does_not_prescribe_the_deleted_lr_rule():
    """The anchor x 25/T rule is gone (utils.py). Its survival in a reference
    file is how it kept being applied after deletion."""
    text = PROBLEMS.read_text(encoding='utf-8')
    assert '25/T' not in text
