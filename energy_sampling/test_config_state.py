"""
Tests for config_state: the state history and its migrations.

EVERY STRUCTURAL CHECK IS PAIRED WITH A MUTATION. A test that asserts a property
of the shipped records proves nothing on its own -- it passes equally well if the
check is vacuous. So each invariant is written as a predicate over an arbitrary
transition, run once over the real history and once over a deliberately broken
transition that it MUST reject.

Run: python -m pytest test_config_state.py -q
"""

import ast
from pathlib import Path

import pytest
import yaml

import config_state as cs
from config_state import (CURRENT_STATE_VERSION, STATE_HISTORY, UNSTAMPED_VERSION,
                          VERSION_KEY, StateTransition, config_version, migrate)

HERE = Path(__file__).parent
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'

# The problem identity: migration versions the config SCHEMA and must never touch
# these. They are the checkpoint schema_version's axis.
PROBLEM_IDENTITY = ('energy_function', 'prior_path', 'molecules_path', 'space_groups',
                    'vector_conditioning', 'molecule_conditioning', 'z_primes')


def _retired_keys():
    """utils._RETIRED_KEYS, read from source. Parsed rather than imported so the
    test does not drag in torch for a dict literal."""
    tree = ast.parse((HERE / 'utils.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', '') == '_RETIRED_KEYS' for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError('_RETIRED_KEYS not found in utils.py')


# ---------------------------------------------------------------------------
# Structural invariants, as predicates so they can be mutation-tested
# ---------------------------------------------------------------------------

def classes_are_disjoint(tr: StateTransition) -> bool:
    """A key may appear in at most one of added/renamed/removed/manual. Two
    classes claiming one key makes the transform order-dependent, and the order
    is an implementation detail nobody should have to know."""
    groups = [set(tr.added), set(tr.renamed), set(tr.removed), set(tr.manual)]
    total = sum(len(g) for g in groups)
    return len(set().union(*groups)) == total


def rename_targets_are_live(tr: StateTransition, retired: set) -> bool:
    """A rename must not point at a key that is itself retired -- that is a
    two-step transition written as one, and it produces a config that fails the
    load-time gate immediately after being 'migrated'."""
    return all(new not in retired for new in tr.renamed.values())


def versions_are_monotone(history) -> bool:
    return [t.version for t in history] == list(range(1, len(history) + 1))


# ---------------------------------------------------------------------------
# The shipped history satisfies them
# ---------------------------------------------------------------------------

def test_history_versions_are_monotone_from_one():
    assert versions_are_monotone(STATE_HISTORY)
    assert STATE_HISTORY[-1].version == CURRENT_STATE_VERSION


@pytest.mark.parametrize('tr', STATE_HISTORY, ids=lambda t: f'v{t.version}')
def test_key_classes_are_disjoint(tr):
    assert classes_are_disjoint(tr)


@pytest.mark.parametrize('tr', STATE_HISTORY, ids=lambda t: f'v{t.version}')
def test_rename_targets_are_not_themselves_retired(tr):
    assert rename_targets_are_live(tr, _retired_keys())


def test_v1_covers_the_retired_keys_exactly():
    """The load-time gate and the migration must describe the same set. A key the
    gate rejects but no transition handles is a config with no route forward;
    a key a transition handles but the gate does not know about is a migration
    for an event that never fires."""
    retired = _retired_keys()
    v1 = STATE_HISTORY[0]
    handled = set(v1.renamed) | set(v1.removed) | set(v1.manual)
    assert handled - retired == set(), f'handled but not retired: {sorted(handled - retired)}'
    assert retired - handled == set(), f'retired but unhandled: {sorted(retired - handled)}'


# ---------------------------------------------------------------------------
# ...and the checks would catch it if they did not
# ---------------------------------------------------------------------------

def test_disjointness_check_rejects_an_overlapping_transition():
    broken = StateTransition(
        version=99, summary='mutation',
        removed={'foo.bar': 'x'}, manual={'foo.bar': 'y'},
    )
    assert not classes_are_disjoint(broken)


def test_rename_target_check_rejects_a_chained_rename():
    broken = StateTransition(
        version=99, summary='mutation',
        renamed={'adaptive_lr.cut_ratio': 'adaptive_lr.trigger'},  # target is retired
    )
    assert not rename_targets_are_live(broken, _retired_keys())


def test_monotonicity_check_rejects_a_gap():
    a = StateTransition(version=1, summary='a')
    c = StateTransition(version=3, summary='c')
    assert not versions_are_monotone((a, c))


def test_coverage_check_would_notice_a_dropped_key():
    """Same comparison as test_v1_covers_the_retired_keys_exactly, run against a
    history missing one key -- it must come back unequal."""
    retired = _retired_keys()
    v1 = STATE_HISTORY[0]
    handled = set(v1.renamed) | set(v1.removed) | set(v1.manual)
    handled.discard('gpu_util_floor')
    assert retired - handled == {'gpu_util_floor'}


# ---------------------------------------------------------------------------
# Migration behavior
# ---------------------------------------------------------------------------

def test_unstamped_config_reads_as_state_zero():
    assert config_version({}) == UNSTAMPED_VERSION


def test_stamped_config_reads_its_version():
    assert config_version({VERSION_KEY: 7}) == 7


def test_current_version_roundtrip_is_a_noop():
    cfg = {VERSION_KEY: CURRENT_STATE_VERSION, 'batch_size': 1000, 'model': {'dropout': 0}}
    out, report = migrate(cfg)
    assert out == cfg
    assert report.unchanged
    assert report.applied == [] and report.needs_judgment == []


def test_migration_does_not_mutate_its_input():
    cfg = {'gpu_util_floor': 40, 'batch_size': 1000}
    before = yaml.safe_dump(cfg, sort_keys=True)
    migrate(cfg)
    assert yaml.safe_dump(cfg, sort_keys=True) == before


def test_rename_moves_the_key_and_keeps_the_value():
    cfg = {'adaptive_lr': {'cut_ratio': 0.25, 'warmup_steps': 1000}}
    out, report = migrate(cfg)
    assert 'cut_ratio' not in out['adaptive_lr']
    assert out['adaptive_lr']['divergence_cut'] == 0.25
    assert out['adaptive_lr']['warmup_steps'] == 1000  # untouched neighbour
    assert any('rename' in a and 'cut_ratio' in a for a in report.applied)


def test_nested_rename_across_blocks():
    cfg = {'buffers': {'anchor_buffer': {'health_gate_r2': 0.3, 'max_size': 200000}}}
    out, _ = migrate(cfg)
    ab = out['buffers']['anchor_buffer']
    assert 'health_gate_r2' not in ab
    assert ab['health_gate_floor'] == 0.3
    assert ab['max_size'] == 200000


def test_removal_drops_the_key():
    cfg = {'gpu_util_floor': 40, 'batch_size': 1000}
    out, report = migrate(cfg)
    assert 'gpu_util_floor' not in out
    assert out['batch_size'] == 1000
    assert any('drop' in a and 'gpu_util_floor' in a for a in report.applied)


def test_absent_keys_produce_no_report_lines():
    """A clean config must migrate silently. A migration that narrates work it did
    not do is noise, and noise is what stops people reading migration reports."""
    cfg = {'batch_size': 1000}
    out, report = migrate(cfg)
    assert out[VERSION_KEY] == CURRENT_STATE_VERSION
    assert report.needs_judgment == []
    assert all('add' in a for a in report.applied)  # only the version stamp


def test_manual_key_is_reported_and_left_alone():
    """The whole point of the manual class: the value stays exactly as found, so
    nobody can mistake a migration for a decision."""
    cfg = {'max_reloads': 5, 'batch_size': 1000}
    out, report = migrate(cfg)
    assert out['max_reloads'] == 5, 'manual key was rewritten'
    assert 'max_reloads_per_1k_steps' not in out, 'manual key was silently renamed'
    assert any('max_reloads' in m for m in report.needs_judgment)
    assert not report.unchanged


def test_ruler_swap_is_manual_not_a_rename():
    """health_gate_zerr -> health_gate_ceiling changes the metric the bar applies
    to. Carrying 18.0 across would gate on a threshold that means something else."""
    cfg = {'buffers': {'anchor_buffer': {'health_gate_zerr': 18.0}}}
    out, report = migrate(cfg)
    assert out['buffers']['anchor_buffer']['health_gate_zerr'] == 18.0
    assert 'health_gate_ceiling' not in out['buffers']['anchor_buffer']
    assert any('health_gate_zerr' in m for m in report.needs_judgment)


def test_migration_stamps_the_version():
    out, _ = migrate({'batch_size': 1000})
    assert out[VERSION_KEY] == CURRENT_STATE_VERSION


def test_migration_leaves_problem_identity_untouched():
    cfg = {
        'gpu_util_floor': 40,  # forces a real migration
        'energy_function': 'elj',
        'prior_path': 'D:/x.pt', 'molecules_path': 'D:/x.pt',
        'space_groups': [2], 'z_primes': [1],
        'vector_conditioning': False, 'molecule_conditioning': False,
    }
    out, report = migrate(cfg)
    assert not report.unchanged  # the migration did do something
    for k in PROBLEM_IDENTITY:
        assert out[k] == cfg[k], f'migration altered problem identity: {k}'


def test_downgrade_is_refused():
    with pytest.raises(ValueError, match='AHEAD'):
        migrate({VERSION_KEY: CURRENT_STATE_VERSION + 1})


def test_report_names_the_unresolved_count():
    _, report = migrate({'max_reloads': 5, 'batch_growth_max_step_regression': 0.15})
    text = report.render()
    assert 'JUDGMENT' in text
    assert '2 item(s) need a decision' in text


# ---------------------------------------------------------------------------
# The canonical config
# ---------------------------------------------------------------------------

def test_canonical_config_migrates_clean():
    """configs/mk_dev.yaml is the master: it must already be at the current state,
    carrying no retired key and needing no judgment. If this fails, the canonical
    config has drifted behind its own schema."""
    cfg = yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))
    out, report = migrate(cfg)
    assert report.needs_judgment == [], report.render()
    assert out[VERSION_KEY] == CURRENT_STATE_VERSION


def test_canonical_config_declares_its_state():
    cfg = yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))
    assert config_version(cfg) == CURRENT_STATE_VERSION, (
        'configs/mk_dev.yaml must carry project_state_version: '
        f'{CURRENT_STATE_VERSION}')


def test_history_renders():
    md = cs.render_history_markdown()
    assert '# Project state history' in md
    assert '## State 1' in md
    assert 'do not edit by hand' in md


def test_committed_history_doc_matches_the_records():
    """docs/state_history.md is generated, and the generator's output is the only
    thing that makes it true. Committing it keeps the chronology readable without
    running Python; this check is what stops the committed copy drifting behind the
    records after a transition is added.

    Regenerate with:  python -c "import config_state as cs; \
open('docs/state_history.md','w',encoding='utf-8',newline='\\n').write(cs.render_history_markdown())"
    """
    doc = HERE / 'docs' / 'state_history.md'
    assert doc.exists(), 'docs/state_history.md is missing -- regenerate it'
    assert doc.read_text(encoding='utf-8') == cs.render_history_markdown(), (
        'docs/state_history.md is stale relative to config_state.STATE_HISTORY -- '
        'regenerate it (command in this test\'s docstring)')
