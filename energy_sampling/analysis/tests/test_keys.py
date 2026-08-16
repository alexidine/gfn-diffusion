"""
Tests for route detection and key resolution.

Network-free: the fixtures are the flattened config shapes real runs carry,
captured by inspecting local runs rather than invented. The shapes matter -- the
first version of this module read a nested `protocol` block that does not exist
in a wandb config (it is stored as a repr STRING), and every route came back
UNKNOWN.

Run: python -m pytest analysis/tests -q
"""

import pytest

from analysis import keys as K
from analysis.keys import KeyState, Route


# ---------------------------------------------------------------------------
# Fixtures: real config shapes, flattened as wandb stores them
# ---------------------------------------------------------------------------

def _cfg(**over):
    """A TB/unconditional config: train_prior -> naive, TB in the terminal stage.
    Modelled on run 44gt5whr."""
    base = {
        'vector_conditioning': {'value': False},
        'molecule_conditioning': {'value': False},
        'energy_function': {'value': 'elj'},
        'protocol_stages_0_name': {'value': 'train_prior'},
        'protocol_stages_0_train_mode': {'value': 'bwd'},
        'protocol_stages_1_name': {'value': 'naive'},
        'protocol_stages_1_train_mode': {'value': 'fused'},
        # base defaults: everything off, as the canonical config has it
        'fwd_loss_coeffs_tb': {'value': 0.0},
        'fwd_loss_coeffs_vg_lb': {'value': 0},
        'fwd_loss_coeffs_vg_lme': {'value': 0},
        'bwd_loss_coeffs_tb': {'value': 0.0},
        'bwd_loss_coeffs_vg_lb': {'value': 0},
        'bwd_loss_coeffs_vg_lme': {'value': 0},
        'bwd_loss_coeffs_mle': {'value': 0.0},
        'bwd_loss_coeffs_tbc': {'value': 0.0},
        'replay_loss_coeffs_tb': {'value': 1.0},
        # stage 0 turns MLE on; stage 1 turns TB on
        'protocol_stages_0_loss_coeffs_bwd_mle': {'value': 1.0},
        'protocol_stages_0_loss_coeffs_bwd_tbc': {'value': 1.0},
        'protocol_stages_1_loss_coeffs_fwd_tb': {'value': 1.0},
        'protocol_stages_1_loss_coeffs_bwd_tb': {'value': 1.0},
    }
    base.update({k: {'value': v} for k, v in over.items()})
    return base


def _vg_cfg():
    """Conditional VarGrad: train_prior -> var_conditioning. Modelled on
    run x4rbzv88."""
    c = _cfg(vector_conditioning=True)
    c.pop('protocol_stages_1_loss_coeffs_fwd_tb')
    c['protocol_stages_1_name'] = {'value': 'var_conditioning'}
    c['protocol_stages_1_loss_coeffs_bwd_vg_lb'] = {'value': 1.0}
    c['protocol_stages_1_loss_coeffs_fwd_vg_lb'] = {'value': 1.0}
    return c


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

def test_stage_names_come_from_flattened_keys():
    assert K.stage_names(_cfg()) == ['train_prior', 'naive']


def test_stage_names_sort_numerically_not_lexically():
    """stages_10 must not sort before stages_2. A lexical sort silently
    reorders the protocol once a run has more than ten stages."""
    cfg = {f'protocol_stages_{i}_name': {'value': f's{i}'} for i in range(12)}
    assert K.stage_names(cfg) == [f's{i}' for i in range(12)]


def test_effective_coeffs_overlay_stage_on_base():
    """The base blocks are structural; a stage turns things on. Reading the base
    alone classifies every run by the defaults, which here say 'no TB'."""
    eff = K.effective_loss_coeffs(_cfg(), 1)
    assert float(eff['fwd']['tb']) == 1.0     # from the stage override
    assert float(eff['replay']['tb']) == 1.0  # from the base
    eff0 = K.effective_loss_coeffs(_cfg(), 0)
    assert float(eff0['fwd']['tb']) == 0.0    # stage 0 does not override it
    assert float(eff0['bwd']['mle']) == 1.0


def test_config_reader_handles_both_wrapped_and_bare_values():
    """Local config.yaml wraps values as {'value': x}; the cloud API returns them
    bare. One reader must serve both or the cloud path silently sees nothing."""
    assert K._value({'k': {'value': 3}}, 'k') == 3
    assert K._value({'k': 3}, 'k') == 3


# ---------------------------------------------------------------------------
# Route detection
# ---------------------------------------------------------------------------

def test_tb_unconditional_route():
    assert K.detect_route(_cfg()) is Route.TB_UNCONDITIONAL


def test_vargrad_conditional_route():
    assert K.detect_route(_vg_cfg()) is Route.VARGRAD_CONDITIONAL


def test_train_prior_stage_is_the_mle_route():
    """Stage 0 of the SAME config is the prior route. A run that died in
    train_prior must not be read with the terminal stage's topline."""
    assert K.detect_route(_cfg(), stage_index=0) is Route.MLE_PRIOR
    assert K.detect_route(_vg_cfg(), stage_index=0) is Route.MLE_PRIOR


def test_route_defaults_to_the_last_stage():
    assert K.detect_route(_cfg()) is K.detect_route(_cfg(), stage_index=1)


def test_vargrad_without_a_conditioning_flag_is_unknown_not_tb():
    """A real configuration the toplines were not written for. Saying UNKNOWN is
    the honest answer; picking TB would hand back a topline whose log Z and TB
    terms may not track."""
    c = _vg_cfg()
    c['vector_conditioning'] = {'value': False}
    assert K.detect_route(c) is Route.UNKNOWN


def test_empty_config_is_unknown():
    assert K.detect_route({}) is Route.UNKNOWN


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------

def test_current_stage_uses_the_one_based_phase_metric():
    """`phase` is stage.index + 1 (train.py). Off-by-one here reports every run
    as one stage behind."""
    assert K.current_stage({'phase': 2}, _cfg()) == 'naive'
    assert K.current_stage({'phase': 1}, _cfg()) == 'train_prior'
    assert K.current_stage_index({'phase': 2}, _cfg()) == 1


def test_current_stage_is_none_when_unknown_never_the_last_stage():
    """A run that died in phase 1 must not be reported as having reached the
    terminal stage -- those are exactly the runs being read to find out why they
    stopped."""
    assert K.current_stage({}, _cfg()) is None
    assert K.current_stage({'phase': 99}, _cfg()) is None
    assert K.current_stage({'phase': 'nonsense'}, _cfg()) is None


# ---------------------------------------------------------------------------
# Key resolution -- the three states (spec H2)
# ---------------------------------------------------------------------------

def test_present_key_is_live():
    r, = K.resolve({'fwd/tb_err_worst'}, ['fwd/tb_err_worst'], Route.TB_UNCONDITIONAL)
    assert r.state is KeyState.LIVE and r.key == 'fwd/tb_err_worst'


def test_absent_key_is_absent():
    r, = K.resolve({'fwd/other'}, ['fwd/nope'], Route.TB_UNCONDITIONAL)
    assert r.state is KeyState.ABSENT and r.key is None


def test_suffix_rename_resolves_and_is_reported():
    """H2: `bwd/under_coverage_wcen` does not exist; the run logs
    `bwd/under_coverage`. Resolved, and reported AS a rename -- a silent
    substitution would hide that the `_wcen` bias correction is not applied."""
    r, = K.resolve({'bwd/under_coverage'}, ['bwd/under_coverage_wcen'],
                   Route.TB_UNCONDITIONAL)
    assert r.state is KeyState.LIVE
    assert r.resolved_to == 'bwd/under_coverage'
    assert r.key == 'bwd/under_coverage'


def test_unambiguous_namespace_rename_resolves():
    r, = K.resolve({'bwd/log_Z_learned'}, ['log_Z_learned'], Route.TB_UNCONDITIONAL)
    assert r.state is KeyState.LIVE and r.resolved_to == 'bwd/log_Z_learned'


def test_ambiguous_namespace_refuses_to_guess():
    """fwd and bwd log Z are DIFFERENT QUANTITIES. Two candidates means the tool
    names both and picks neither."""
    r, = K.resolve({'bwd/log_Z_learned', 'fwd/log_Z_learned'}, ['log_Z_learned'],
                   Route.TB_UNCONDITIONAL)
    assert r.state is KeyState.ABSENT
    assert 'ambiguous' in r.note
    assert 'bwd/log_Z_learned' in r.note and 'fwd/log_Z_learned' in r.note


# --- the acceptance test the spec calls most likely to be got wrong ---

def test_vargrad_route_marks_logz_and_tb_as_na_not_absent_not_zero():
    """Spec acceptance #4. On the VarGrad route these keys EXIST and carry
    numbers; reading them as one would on a TB run is wrong. They must resolve
    NA_ROUTE -- not ABSENT, and never a value."""
    available = {'bwd/log_Z_learned', 'fwd/tb_err_worst', 'fwd/tb_resid_clipped',
                 'fwd/vg_lb'}
    res = {r.wanted: r for r in K.resolve(
        available, ['bwd/log_Z_learned', 'fwd/tb_err_worst', 'fwd/tb_resid_clipped',
                    'fwd/vg_lb'], Route.VARGRAD_CONDITIONAL)}
    for k in ('bwd/log_Z_learned', 'fwd/tb_err_worst', 'fwd/tb_resid_clipped'):
        assert res[k].state is KeyState.NA_ROUTE, k
        assert res[k].state is not KeyState.ABSENT
        assert res[k].key is None, 'an NA_ROUTE key must never be requested'
    assert res['fwd/vg_lb'].state is KeyState.LIVE


def test_the_same_keys_are_live_on_the_tb_route():
    """The mutation for the test above: NA_ROUTE must be a property of the ROUTE,
    not of the key. If these came back NA on TB too, the check above would pass
    while meaning nothing."""
    available = {'bwd/log_Z_learned', 'fwd/tb_err_worst'}
    res = K.resolve(available, sorted(available), Route.TB_UNCONDITIONAL)
    assert all(r.state is KeyState.LIVE for r in res)


def test_na_route_is_checked_before_presence():
    """The defining property of NA_ROUTE is that the key IS present. Testing
    presence first would file it LIVE and hand back a misleading number."""
    r, = K.resolve({'bwd/log_Z_learned'}, ['bwd/log_Z_learned'],
                   Route.VARGRAD_CONDITIONAL)
    assert r.state is KeyState.NA_ROUTE


def test_live_keys_excludes_absent_and_na():
    res = K.resolve({'fwd/vg_lb', 'bwd/log_Z_learned'},
                    ['fwd/vg_lb', 'bwd/log_Z_learned', 'fwd/missing'],
                    Route.VARGRAD_CONDITIONAL)
    assert K.live_keys(res) == ['fwd/vg_lb']


def test_every_route_has_a_topline():
    for route in Route:
        assert K.TOPLINE[route], route


def test_vargrad_topline_carries_held_out_series():
    """R17: on conditional runs the held-out set is read FIRST -- train r2,
    tb_err and scatter_err can all improve on the same evaluation where held-out
    blows up."""
    assert any(k.startswith('eval_test/') for k in K.TOPLINE[Route.VARGRAD_CONDITIONAL])


def test_vargrad_topline_contains_no_na_keys():
    """A topline that resolves to NA_ROUTE for its own route would be a report
    with a hole in it by construction."""
    res = K.resolve(set(K.TOPLINE[Route.VARGRAD_CONDITIONAL]),
                    K.TOPLINE[Route.VARGRAD_CONDITIONAL],
                    Route.VARGRAD_CONDITIONAL)
    assert all(r.state is not KeyState.NA_ROUTE for r in res)


def test_ema_detection():
    assert K.is_ema('tracker/logw_std_rms')
    assert not K.is_ema('fwd/vg_lb')
