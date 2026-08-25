"""The soft reward clip must bound the tail, leave the bulk alone, and be reachable.

log Z's TB fixed point is a MEAN of log w, so an unbounded left tail on log_reward drags
it without limit -- measured -8,000..-15,000 on tetraglycine phase 2. Clipping the force
field above a cutoff bounds it. These gates exist because each half fails silently: an
"off" path that quietly clipped would change every existing run, a clip that never fired
would look configured and do nothing, and a clip applied to the WALL or the MEASURE would
break the domain guarantee or the change of variables without raising.
"""
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings('ignore')
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from energies.conformer_torsions import ConformerTorsions

SMI = 'NCC(=O)NCC(=O)NCC(=O)NCC(=O)O'          # tetraglycine, d=87
ONE = torch.tensor(1.0)


def _en(clip=None):
    torch.set_default_dtype(torch.float32)
    return ConformerTorsions(smiles=SMI, level='full', device='cpu',
                             force_field='mmff', energy_clip=clip)


def _states(en, n=256, seed=0):
    prior = torch.load('conformer_prior_v2.pt', weights_only=False)
    xs, _ = en.sample_prior_states(prior, n, np.random.default_rng(seed), report=False)
    return torch.as_tensor(xs, dtype=en.dtype)


def test_off_is_bitwise_unclipped():
    """Default must not perturb a single sample -- not 'close', bitwise."""
    a, b = _en(None), _en(None)
    x = _states(a)
    assert a.energy_clip is None
    assert torch.equal(a.potential_energy(x, ONE), b.potential_energy(x, ONE))


def test_clip_bounds_the_tail():
    """RE-INTRODUCING the problem must FAIL this: an inert clip has to be caught."""
    off, on = _en(None), _en(250.0)
    x = _states(off)
    e_off = off.potential_energy(x, ONE).double()
    e_on = on.potential_energy(x, ONE).double()
    assert e_off.max() > 1e4, 'test molecule no longer has a heavy tail to clip'
    assert e_on.max() < 400.0, f'clip did not bound the tail: max {e_on.max():.1f}'
    assert not torch.equal(e_off, e_on), 'clip is INERT'


def test_bulk_is_untouched_below_the_cutoff():
    """Identity below the cutoff: anything physical must be bit-for-bit unchanged."""
    off, on = _en(None), _en(250.0)
    x = _states(off)
    e_off = off.potential_energy(x, ONE)
    e_on = on.potential_energy(x, ONE)
    below = e_off < 250.0
    assert below.any(), 'no samples below the cutoff to compare'
    assert torch.equal(e_off[below], e_on[below]), 'clip altered energies BELOW the cutoff'


def test_clip_is_monotone():
    """A non-monotone rescale would reorder rewards -- worse than not clipping."""
    on = _en(100.0)
    x = _states(on)
    e_off = _en(None).potential_energy(x, ONE).double().numpy()
    e_on = on.potential_energy(x, ONE).double().numpy()
    order_off = np.argsort(e_off)
    assert np.all(np.diff(e_on[order_off]) >= -1e-4), 'clip reordered the energies'


def test_measure_terms_are_not_clipped():
    """energy() = potential + BAT + chart. Only the POTENTIAL may be compressed; a clipped
    change of measure is no longer a change of measure."""
    on = _en(250.0)
    x = _states(on)
    total = on.energy(x, None, torch.tensor(0.0)).double()
    pot = on.potential_energy(x, ONE).double()
    jac = on.jacobian_energy(x, ONE).double()
    # at T=1 the identity is total == pot + jac - log_chart_jacobian, exactly
    expect = pot + jac - float(on.log_chart_jacobian)
    assert torch.allclose(total, expect, atol=1e-3), 'measure terms did not survive intact'


def test_config_key_reaches_the_constructor():
    """ConformerTorsions has no **kwargs and the modeller announces dropped keys, so a
    live key cannot be silently swallowed -- but verify it is actually in the signature."""
    import inspect

    from conformer_modeller import _NON_ENERGY_KEYS

    assert 'energy_clip' in inspect.signature(ConformerTorsions.__init__).parameters
    assert 'energy_clip' not in _NON_ENERGY_KEYS, 'consumed by the modeller, never forwarded'


def test_non_finite_clip_raises():
    with pytest.raises(ValueError):
        _en(float('nan'))
