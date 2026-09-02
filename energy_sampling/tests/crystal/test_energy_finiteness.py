"""
`generator_energy` RAISES on a non-finite energy rather than passing it on.

WHY A RAISE AND NOT A FILTER. `log_reward` is `-energy/T`, so a +inf energy is a
-inf reward, and nothing downstream rejects one: `AnchorBuffer.admit` tests no
finiteness and sorts candidates by ASCENDING energy, so a -inf row is admitted as
the BEST anchor for its condition and then persists. One such row also NaNs the
whole fused loss (train.py adds `pooled_coeff * pooled_rows.mean()` unguarded).

WHY IT MATTERS NOW. `reward_range` ran `.clip(max=clip)`, which silently turns
+inf into a finite number. Lambda mixing REQUIRES `reward_range: null`, so every
annealing run has that accidental protection removed.

    pytest tests/crystal/test_energy_finiteness.py
"""
import os
import sys

import pytest
import torch

CPU = torch.device('cpu')
_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_here, os.path.dirname(_here),
          os.path.join(os.path.dirname(_here), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from energy_sampling.energies.molecular_crystal import MolecularCrystal  # noqa: E402

SG, DIM = 2, 12


def energy_fn():
    return MolecularCrystal(
        device=CPU, energy_function='latent_gaussian', space_groups=[SG], z_primes=(1,),
        temperature=1.0, bounding_coeff=1.0, reduction_coeff=1.0,
        reward_range=None, internal_oom_recovery=False, host_gas_phase_reference=False,
        analyze_kwargs={'c': [0.5] * DIM, 'width': 0.4})


class _Batch:
    """Minimal stand-in: the guard reads only `identifier` off the batch."""

    def __init__(self, ids=None):
        self.identifier = ids


def test_finite_energy_passes_silently():
    ef = energy_fn()
    total = torch.linspace(-5.0, 5.0, 8)
    ens = {'crystal_energy': total.clone(), 'bounding_energy': torch.zeros(8)}
    ef._assert_finite_energy(total, ens, _Batch())   # must not raise


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_non_finite_energy_raises(bad):
    """Every non-finite flavour, not just inf: -inf is the dangerous one for the
    buffer (it sorts FIRST as the best anchor) and nan is what a bad derivative
    produces, so a guard that caught only +inf would miss both."""
    ef = energy_fn()
    total = torch.zeros(6)
    total[3] = bad
    with pytest.raises(FloatingPointError) as e:
        ef._assert_finite_energy(total, {'crystal_energy': total.clone()}, _Batch())
    assert '1/6' in str(e.value)
    assert '[3]' in str(e.value)


def test_message_names_the_offending_component():
    """The report is the point -- it must localise WHICH term blew up, or the bug
    is inferred from a NaN loss thousands of steps later instead."""
    ef = energy_fn()
    total = torch.zeros(4)
    total[1] = float('inf')
    ens = {'crystal_energy': torch.zeros(4),
           'jacobian_energy': torch.tensor([0.0, float('inf'), 0.0, 0.0]),
           'bounding_energy': torch.zeros(4)}
    with pytest.raises(FloatingPointError) as e:
        ef._assert_finite_energy(total, ens, _Batch(['a', 'b', 'c', 'd']))
    msg = str(e.value)
    assert 'jacobian_energy=1' in msg
    assert 'crystal_energy' not in msg.split('Non-finite components:')[1].split('.')[0]
    assert "'b'" in msg, msg      # the identifier, not just the row index


def test_every_return_from_generator_energy_is_guarded():
    """The unit tests above call the guard directly; this one proves
    generator_energy actually invokes it on EVERY exit.

    ⚠ A COUNT OF CALL SITES IS NOT THIS TEST, and the first version of it was
    exactly that -- `src.count('_assert_finite_energy') == 2`. It passed while
    one of the two calls sat AFTER a `return` in the energy_clip branch and was
    dead code, so the clipped route was entirely unguarded. Counting text proves
    the string is present, not that control flow reaches it. This walks the AST
    and requires the guard to precede every `return` by line number.
    """
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(MolecularCrystal.generator_energy)))
    fn = tree.body[0]
    guards = [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == '_assert_finite_energy']
    returns = [n.lineno for n in ast.walk(fn)
               if isinstance(n, ast.Return) and n.value is not None]
    assert guards, 'generator_energy does not call the finiteness guard at all'
    assert returns, 'no value-returning exit found -- the walk is broken, not the code'
    for r in returns:
        assert any(g < r for g in guards), (
            f'return at line {r} of generator_energy is reachable with no preceding '
            f'_assert_finite_energy (guards at {guards})')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
