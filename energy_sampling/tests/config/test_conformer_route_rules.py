"""
Tests for the conformer-route rules in config_invariants.

EACH RULE FIRES ON A ONE-KEY PERTURBATION of a minimal conditional conformer
config that every conformer rule passes, and fires ALONE, at its declared
severity -- so the test pins the rule rather than whichever neighbour happens to
share the key. Each rule is also silent on mk_dev, and that silence is shown to
come from the route gate rather than from rules that cannot fire: mk_dev carries
DPLR, snapshot_prior, a molecules_path and crystal energy keys, and relabelled as
a conformer config it trips the rules that read them.

The names of the rules appear here in quotes because
test_config_invariants.py::test_every_rule_is_mutation_tested reads this file too.

Run: python -m pytest tests/config/test_conformer_route_rules.py -q
"""

import copy
import inspect
from pathlib import Path

import pytest
import yaml

import config_invariants as ci
from config_invariants import BASELINE, ERROR, check

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'

CONFORMER_RULES = tuple(r for r in ci.RULES if r.__name__.startswith('conformer_'))
_DELETE = object()


@pytest.fixture(scope='module')
def canonical():
    return yaml.safe_load(CANONICAL.read_text(encoding='utf-8'))


def conditional_conformer():
    """The smallest conditional conformer config every conformer rule passes: a set
    policy over a condition set, conditioned on the molecule, with the energy named
    and clipped, and a two-stage protocol shaped like the conformer phase 1 / phase 2.
    Values are plausible, not calibrated -- energy_clip in particular is a
    placeholder, since the rule judges presence and never the number."""
    return {
        'energy_function': 'conformer_torsions',
        'molecules_path': 'sets/r1_conditions.pt',
        'embedding_conditioning': True,
        'embedding_conditioning_dim': 256,
        'prior_model_name': None,
        'compile_policy': False,
        'xcond_eval': {'enabled': False},
        'model': {'policy_kind': 'set', 'dplr_rank': 0},
        'energy_config': {
            'level': 'full', 'force_field': 'mmff', 'log_temperature': 0.0,
            'temperature': 1.0, 'energy_clip': 150.0,
            'internal_prior_path': 'prior.pt', 'prior_sample_size': 50000,
            'prior_relax_steps': 0,
        },
        'protocol': 'conformer_conditional_tb',
        'protocols': {'conformer_conditional_tb': {'stages': [
            {'name': 'train_prior', 'train_mode': 'bwd', 'bwd_sampling_mode': 'dataset',
             'flags': {'update_log_z': True, 'scramble_conditions': False},
             'on_exit': ['snapshot:phase1_exit']},
            {'name': 'tb_conditioning', 'train_mode': 'fused', 'bwd_sampling_mode': 'prior',
             'flags': {'update_log_z': False, 'buffers_active': True},
             'on_enter': ['rebuild_prior_by_churn', 'set_lr_flow:1.0e-4', 'freeze_pb']},
        ]}},
    }


def perturbed(cfg, **dotted):
    """A copy with `a__b` paths set, or removed when the value is _DELETE."""
    out = copy.deepcopy(cfg)
    for path, value in dotted.items():
        node = out
        parts = path.split('__')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if value is _DELETE:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    return out


def _stage(cfg, name):
    stages = cfg['protocols'][cfg['protocol']]['stages']
    return next(s for s in stages if s['name'] == name)


def _conformer_violations(cfg):
    names = {r.__name__ for r in CONFORMER_RULES}
    return [v for v in check(cfg) if v.rule in names]


def _fires_alone(cfg, rule, severity):
    vs = _conformer_violations(cfg)
    assert vs, f'{rule} did not fire'
    assert {v.rule for v in vs} == {rule}, '\n'.join(map(str, vs))
    assert {v.severity for v in vs} == {severity}, '\n'.join(map(str, vs))


def _silent(cfg):
    vs = _conformer_violations(cfg)
    assert vs == [], '\n'.join(map(str, vs))


# ---------------------------------------------------------------------------
# The clean baseline, and mk_dev
# ---------------------------------------------------------------------------

def test_the_minimal_conditional_conformer_config_is_clean():
    _silent(conditional_conformer())


def test_mk_dev_is_silent_on_every_conformer_rule(canonical):
    for rule in CONFORMER_RULES:
        assert rule(canonical) == [], rule.__name__


def test_the_route_gate_is_what_keeps_mk_dev_silent(canonical):
    """The same file, relabelled onto the conformer route. Its DPLR over a
    molecules_path, its crystal energy_config (no level, no force_field, crystal
    keys), its snapshot_prior and its unclipped energy all trip the rules that
    read them -- so the silence above is the energy_function gate."""
    cfg = perturbed(canonical, energy_function='conformer_torsions')
    fired = {v.rule for v in _conformer_violations(cfg)}
    assert {'conformer_dplr_is_off_on_a_set',
            'conformer_level_and_force_field_are_stated',
            'conformer_energy_config_keys_are_read',
            'conformer_protocol_omits_snapshot_prior',
            'conformer_set_clips_energy'} <= fired, fired


# ---------------------------------------------------------------------------
# One focused test per rule
# ---------------------------------------------------------------------------

def test_dplr_under_a_set_policy_is_an_error():
    _fires_alone(perturbed(conditional_conformer(), model__dplr_rank=4),
                 'conformer_dplr_is_off_on_a_set', ERROR)


def test_dplr_under_a_flat_policy_over_a_set_is_reported():
    """A flat policy on a CARRIER leaks pad columns into log P_F through the DPLR
    path; the config cannot tell a carrier from an equal-layout set, so BASELINE."""
    flat = perturbed(conditional_conformer(), model__policy_kind='flat')
    _silent(flat)
    _fires_alone(perturbed(flat, model__dplr_rank=4),
                 'conformer_dplr_is_off_on_a_set', BASELINE)
    # one molecule, no set: DPLR is the unconditional conformer route's own setting
    _silent(perturbed(flat, model__dplr_rank=4, molecules_path=None,
                      embedding_conditioning=False))


def test_an_unconditioned_set_policy_over_a_set_is_an_error():
    _fires_alone(perturbed(conditional_conformer(), embedding_conditioning=False),
                 'conformer_set_policy_is_conditioned', ERROR)
    # a single-molecule set policy needs no conditioning
    _silent(perturbed(conditional_conformer(), embedding_conditioning=False,
                      molecules_path=None))


@pytest.mark.parametrize('dim', [255, 0, None, _DELETE])
def test_a_set_policy_embedding_width_must_be_positive_and_even(dim):
    _fires_alone(perturbed(conditional_conformer(), embedding_conditioning_dim=dim),
                 'conformer_set_policy_is_conditioned', ERROR)


@pytest.mark.parametrize('level', [_DELETE, None, 'ful', 'internal'])
def test_the_level_must_be_named_and_known(level):
    _fires_alone(perturbed(conditional_conformer(), energy_config__level=level),
                 'conformer_level_and_force_field_are_stated', ERROR)


def test_an_absent_force_field_is_an_error():
    """The constructor default is 'reference': absence is a silent choice."""
    _fires_alone(perturbed(conditional_conformer(), energy_config__force_field=_DELETE),
                 'conformer_level_and_force_field_are_stated', ERROR)


@pytest.mark.parametrize('key', ['forcefield', 'lambda_mix', 'physical_energy_clip'])
def test_an_energy_config_key_the_route_drops_is_an_error(key):
    """`forcefield` is the misspelling that would silently train on 'reference';
    the other two are crystal-lineage keys a literal port carries."""
    _fires_alone(perturbed(conditional_conformer(), **{f'energy_config__{key}': 1.0}),
                 'conformer_energy_config_keys_are_read', ERROR)


def test_every_consumed_energy_config_key_is_accepted():
    """Each _NON_ENERGY_KEYS entry and `temperature` are consumed or tolerated,
    and every ConformerTorsions parameter is accepted -- the rule refuses only
    what the route drops."""
    params, non_energy, _ = ci._conformer_energy_contract()
    cfg = conditional_conformer()
    for key in sorted(params | non_energy | {'temperature'}):
        cfg['energy_config'].setdefault(key, 0)
    assert ci.conformer_energy_config_keys_are_read(cfg) == []


@pytest.mark.slow   # imports the energy (torch) and the modeller: the proof, not the dev loop
def test_the_parsed_contract_is_the_one_the_run_filters_with():
    """The AST read must equal what init_energy_function actually filters with:
    inspect.signature(ConformerTorsions.__init__) and conformer_modeller's
    _NON_ENERGY_KEYS, and the LEVELS the constructor checks."""
    from energies.conformer_torsions import ConformerTorsions
    import conformer_modeller
    params, non_energy, levels = ci._conformer_energy_contract()
    assert params == set(inspect.signature(ConformerTorsions.__init__).parameters) - {'self'}
    assert non_energy == set(conformer_modeller._NON_ENERGY_KEYS)
    assert levels == tuple(ConformerTorsions.LEVELS)


def test_an_unreadable_contract_is_reported_not_passed(monkeypatch, tmp_path):
    """If the source moved, the two contract-reading rules must say they did not
    run. Abstaining would read as a clean energy_config."""
    monkeypatch.setattr(ci, '_CONFORMER_TORSIONS_SRC', str(tmp_path / 'missing.py'))
    ci._conformer_energy_contract.cache_clear()
    try:
        vs = _conformer_violations(conditional_conformer())
        assert {v.rule for v in vs} == {'conformer_level_and_force_field_are_stated',
                                        'conformer_energy_config_keys_are_read'}, vs
        assert all(v.severity == ERROR and 'did not run' in v.detail for v in vs), vs
    finally:
        monkeypatch.undo()
        ci._conformer_energy_contract.cache_clear()


@pytest.mark.parametrize('where', ['on_exit', 'on_enter'])
def test_snapshot_prior_in_a_stage_action_is_an_error(where):
    cfg = conditional_conformer()
    _stage(cfg, 'train_prior').setdefault(where, []).append('snapshot_prior')
    _fires_alone(cfg, 'conformer_protocol_omits_snapshot_prior', ERROR)


def test_a_named_prior_model_is_an_error():
    _fires_alone(perturbed(conditional_conformer(), prior_model_name='mipcas_prior.pt'),
                 'conformer_prior_model_name_is_null', ERROR)


def test_scrambled_conditions_under_a_set_policy_are_an_error():
    cfg = conditional_conformer()
    _stage(cfg, 'train_prior')['flags']['scramble_conditions'] = True
    _fires_alone(cfg, 'conformer_set_policy_is_not_scrambled', ERROR)


@pytest.mark.parametrize('clip', [None, _DELETE])
def test_an_unclipped_set_is_reported(clip):
    """Absent is judged: the constructor default is None, i.e. no clip."""
    _fires_alone(perturbed(conditional_conformer(), energy_config__energy_clip=clip),
                 'conformer_set_clips_energy', BASELINE)
    # one molecule and no condition file: nothing starves the tracker's trim
    _silent(perturbed(conditional_conformer(), energy_config__energy_clip=clip,
                      molecules_path=None))


def test_xcond_on_a_conditional_set_is_reported():
    _fires_alone(perturbed(conditional_conformer(), xcond_eval__enabled=True),
                 'conformer_set_xcond_is_off', BASELINE)
    # the pass is inert off the conditional route, so there is nothing to report
    _silent(perturbed(conditional_conformer(), xcond_eval__enabled=True,
                      embedding_conditioning=False, model__policy_kind='flat'))


@pytest.mark.parametrize('setting', [True, 'auto', 'step', 'off'])
def test_a_compiled_set_policy_is_reported(setting):
    """`auto`/`step` compile on the cluster; a QUOTED 'off' is a truthy string,
    which maybe_compile_policy reads as on."""
    _fires_alone(perturbed(conditional_conformer(), compile_policy=setting),
                 'conformer_set_policy_is_not_compiled', BASELINE)


def test_a_compiled_flat_policy_is_not_this_rules_business():
    _silent(perturbed(conditional_conformer(), compile_policy='auto',
                      model__policy_kind='flat'))
