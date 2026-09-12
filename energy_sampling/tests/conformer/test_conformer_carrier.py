"""The carrier state holds a MIXED-k molecule set in one width, and reads each row through its
own chart.

Load-bearing claims, each asserted directly because a failure of any returns plausible numbers:
  * dispatch is EXACT -- a carrier row scores to the member's own energy of the member state;
  * the graph reconstruction through the carrier-padded condition equals the member chart;
  * a nonzero pad, or a state_mask from another layout, is REFUSED, not scored;
  * on a carrier, chart methods of the dispatcher itself are refused by name;
  * a same-block set is the IDENTITY layout, i.e. the pre-carrier route unchanged.
"""
import numpy as np
import pytest
import torch

from energies.conformer_carrier import (CarrierLayout, carrier_pad_condition,
                                        check_carrier_convention)
from energies.conformer_data import collate_conditions, condition_from_energy
from energies.dof_features import free_dof_atom_index
from energies.multi_conformer import MultiConformerTorsions

KW = dict(device='cpu', level='full', force_field='mmff')
SMIS = ['C', 'CO', 'N']          # k = 9, 12, 6


@pytest.fixture(scope='module')
def multi():
    torch.set_default_dtype(torch.float64)
    return MultiConformerTorsions(SMIS, identifiers=SMIS, **KW)


@pytest.fixture(scope='module')
def carrier_batch(multi):
    lay = multi.carrier
    rows, xs = [], []
    g = torch.Generator().manual_seed(0)
    for ident, m in multi._members.items():
        c = condition_from_energy(m, identifier=ident)
        a, msk = free_dof_atom_index(m)
        cc = carrier_pad_condition(c, lay, ident, m, atoms=a, mask=msk, R=1)
        x = torch.rand((3, lay.k(ident)), generator=g, dtype=torch.float64) * 0.4 - 0.2
        xs.append((ident, m, x))
        rows += [cc.__copy__() for _ in range(3)]
    batch = collate_conditions(rows)
    X = torch.cat([lay.to_carrier(i, x) for i, _, x in xs])
    return batch, X, xs


def test_layout_blocks_and_placement(multi):
    lay = multi.carrier
    assert multi.is_carrier and lay.K == 12
    assert lay.block_width == [5, 4, 3]
    # methane: r 4, theta 3, phi 2 -> first slots of each block
    assert lay.cols['C'].tolist() == [0, 1, 2, 3, 5, 6, 7, 9, 10]
    # the periodic mask is column-constant: exactly the phi block
    assert multi.periodic_dims == [False] * 9 + [True] * 3


def test_dispatch_is_exact(multi, carrier_batch):
    batch, X, xs = carrier_batch
    got = multi.energy(X, batch)
    want = torch.cat([m.energy(x) for _, m, x in xs])
    assert torch.equal(got, want)


def test_graph_reconstruction_through_the_carrier(multi):
    lay = multi.carrier
    for ident, m in multi._members.items():
        cc = carrier_pad_condition(condition_from_energy(m, identifier=ident), lay, ident, m)
        assert check_carrier_convention(cc, lay, ident, m) == 0.0


def test_a_nonzero_pad_is_refused(multi, carrier_batch):
    batch, X, _ = carrier_batch
    X2 = X.clone()
    X2[0, int(multi.carrier.pad_cols('C')[0])] = 1e-3
    with pytest.raises(RuntimeError, match='PAD'):
        multi.energy(X2, batch)


def test_a_foreign_state_mask_is_refused(multi, carrier_batch):
    batch, X, _ = carrier_batch
    b2 = batch.clone()
    b2.state_mask = torch.ones_like(b2.state_mask)
    with pytest.raises(RuntimeError, match='state_mask'):
        multi.energy(X, b2)


def test_a_carrier_energy_without_the_batch_is_refused(multi, carrier_batch):
    _, X, _ = carrier_batch
    with pytest.raises(RuntimeError, match='mol_batch'):
        multi.energy(X, None)


@pytest.mark.parametrize('name', ['potential_energy', 'sample_prior_states',
                                  'dof_from_state', 'build_positions'])
def test_chart_methods_of_the_dispatcher_are_refused(multi, name):
    with pytest.raises(NotImplementedError, match='CARRIER'):
        getattr(multi, name)(torch.zeros(2, multi.carrier.K))


def test_the_reference_member_is_its_own_chart(multi):
    """The dispatcher's data_ndim is K; the reference member must still be methane's 9."""
    assert multi.data_ndim == 12 and multi._members['C'].data_ndim == 9
    assert multi._members['C'] is not multi


def test_same_block_set_is_the_identity_layout():
    torch.set_default_dtype(torch.float64)
    en = MultiConformerTorsions(['CO', 'C=C'], identifiers=['CO', 'C=C'], **KW)
    assert not en.is_carrier and en.data_ndim == 12
    lay = CarrierLayout(en._members)
    assert lay.is_identity and all((c == np.arange(12)).all() for c in lay.cols.values())


def test_prebuilt_reward_slices_per_member(multi, carrier_batch):
    """State-dependent log J at `full`: each row's measure through its own member."""
    from energies.conformer_data import set_batch_states
    batch, X, xs = carrier_batch
    one = torch.tensor(1.0, dtype=torch.float64)
    baked = torch.cat([m.potential_energy(x, one) for _, m, x in xs])
    b = set_batch_states(batch.clone(), X, baked)
    got = multi.prebuilt_sample_to_reward(b, 1.0)
    want = -torch.cat([m.energy(x) for _, m, x in xs])
    assert torch.allclose(got, want, atol=1e-8)
