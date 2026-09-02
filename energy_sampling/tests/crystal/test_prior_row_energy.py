"""
Prior-buffer rows are judged at the CURRENT lambda, not at the one they were
admitted under, and not in the raw-lattice currency `y` carries.

THE BUG THIS PINS. `_expire_stale_prior_rows`, the reach trigger and
`top_up_prior_from_anchors`' purge ranking all compute `row_energy - Emin(c)`.
`prior_buffer.y` comes from `_buffer_y_fn`, which returns the energy_function's
name -- so on the crystal route it is `batch.elj`, the RAW lattice sum with no
lj_coeff, no /z_prime and none of the density, pressure, reduction or jacobian
terms. `_condition_energy_floor` returns the full composite total. Measured on
qm9c_lam003: -416.0 against -65.5, an offset of ~385 energy units against a
`ramp_floor` of 100 -- four times the entire window, so the expiry channel could
essentially never fire. That predates lambda entirely.

    pytest tests/crystal/test_prior_row_energy.py
"""
import os
import sys
import types

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.train import Modeller  # noqa: E402

PHYS = torch.tensor([100.0, 200.0, -50.0])
FLOW = torch.tensor([-10.0, -20.0, 30.0])
ELJ = torch.tensor([-400.0, -410.0, -420.0])       # the raw lattice term `y` holds


def stub(lam=1.0, flow=True, phys=True):
    """Minimal stand-in: the helper reads only the buffer batch and lambda_mix."""
    batch = types.SimpleNamespace()
    if phys:
        batch.physical_energy = PHYS.clone()
    if flow:
        batch.flow_energy = FLOW.clone()
    m = types.SimpleNamespace()
    m.prior_buffer = types.SimpleNamespace(batch=batch, y=ELJ.clone())
    m.energy_function = types.SimpleNamespace(lambda_mix=lam)
    return m


def energy(m):
    return Modeller._prior_row_energy(m)


def test_composes_the_two_legs_at_the_live_lambda():
    for lam in (0.0, 0.003, 0.5, 1.0):
        got = energy(stub(lam=lam))
        want = (1.0 - lam) * FLOW + lam * PHYS
        assert torch.allclose(got, want), f'lambda={lam}: {got.tolist()}'


def test_the_answer_moves_with_lambda_on_the_same_rows():
    """A row admitted at one lambda must be re-judged at today's -- the whole
    reason this is composed at read instead of read from a frozen scalar."""
    assert not torch.allclose(energy(stub(lam=0.0)), energy(stub(lam=1.0)))


def test_it_is_not_the_raw_lattice_term():
    """⚠ THE REGRESSION. Returning `prior_buffer.y` would reinstate the -385
    offset against Emin(c). A future edit that 'simplifies' this back to `.y`
    must fail here."""
    got = energy(stub(lam=1.0))
    assert not torch.allclose(got, ELJ), \
        'the helper returned the raw lattice term; the Emin(c) comparison is broken again'
    assert torch.allclose(got, PHYS)


def test_lambda_free_rows_need_no_flow_leg():
    """Every shipped config is lambda-free and its rows carry no flow_energy."""
    got = energy(stub(flow=False))
    assert torch.allclose(got, PHYS)


def test_a_stale_buffer_fails_loudly():
    """A buffer restored from before the leg split holds an older currency. That
    must raise, not silently fall back -- a quiet fallback is how the -385 offset
    survived unnoticed in the first place."""
    with pytest.raises(AttributeError) as e:
        energy(stub(phys=False, flow=False))
    assert 'physical_energy' in str(e.value)


def test_no_call_site_reads_prior_buffer_dot_y():
    """`prior_buffer.y` must not reappear as an energy. AST, not text: the first
    version of this grepped source lines and flagged the DOCSTRING that warns
    against the pattern -- a check that fires on its own warning label is a check
    that will be silenced rather than heeded."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(Modeller)))
    offenders = [n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == 'y'
                 and isinstance(n.value, ast.Attribute)
                 and n.value.attr == 'prior_buffer']
    assert not offenders, (
        f'{len(offenders)} site(s) still read prior_buffer.y as an energy '
        f'(Modeller-relative lines {offenders}); use _prior_row_energy()')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
