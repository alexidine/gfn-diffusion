"""The coefficient names that arm condition grouping live in ONE place.

This class of bug has shipped twice -- `vg_lme` (2026-08-26) and `pooled_vg`
(2026-08-28). Both times a caller open-coded "is VarGrad running?" as a
disjunction over coefficient names, a new flavour arrived, and the caller went
on reading 0. The failure is silent every time: condition-blocked draws switch
off, groups collapse to singletons, and a term that costs a full rollout
contributes nothing.
"""
import ast
import sys

sys.path.insert(0, '.')

import config_invariants as ci


def _get(d):
    return lambda k: d.get(k)


def test_predicate_covers_every_flavour():
    for name in ci.BRANCH_VARGRAD_COEFFS:
        assert ci.runs_grouped_vargrad(_get({name: 1.0}), _get({})), name
    assert ci.runs_grouped_vargrad(_get({}), _get({ci.CROSS_BRANCH_VARGRAD_COEFF: 1.0}))


def test_cross_branch_coeff_arms_a_branch_that_declares_nothing():
    """`pooled_vg` lives on fwd_loss_coeffs and pools BOTH branches into one
    grouping, so it must arm a backward draw whose own block is all zeros. A
    gate widened only on the branch's own coefficients still reads 0 here -- the
    exact 2026-08-28 defect."""
    bwd_declares_nothing = _get({'vg_lb': 0.0, 'vg_lme': 0.0})
    assert ci.runs_grouped_vargrad(bwd_declares_nothing,
                                   _get({ci.CROSS_BRANCH_VARGRAD_COEFF: 1.0}))
    assert not ci.runs_grouped_vargrad(bwd_declares_nothing, _get({}))


def test_no_caller_reopens_the_coefficient_list():
    """No module may name these coefficients in its own grouping gate."""
    names = set(ci.BRANCH_VARGRAD_COEFFS) | {ci.CROSS_BRANCH_VARGRAD_COEFF}
    for path, allowed in (('train.py', {'_runs_grouped_vargrad'}),
                          ('config_invariants.py', {'runs_grouped_vargrad'})):
        tree = ast.parse(open(path, encoding='utf-8').read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name in allowed:
                continue
            body = ast.dump(node)
            if 'block_m' not in body and 'any_vg' not in body:
                continue
            for n in names:
                assert repr(n) not in body and f"'{n}'" not in body, (
                    f'{path}:{node.name} names {n} in a grouping gate; call the '
                    'shared predicate instead')
