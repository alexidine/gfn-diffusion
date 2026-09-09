"""One state must produce one geometry, through either reconstruction, at every DoF tier.

The conformer stack has TWO builders: `ConformerTorsions.build_positions`, which reconstructs
through the energy's own cached tree, and `conformer_data.states_to_positions`, which
reconstructs from the parameterisation stored on the graph. They are supposed to be the same
function. Until 2026-09-08 they agreed only at `level='torsion'` and missed each other by 5-6
Angstrom above it, because `conformer_data` had no notion of `level`: the stored map came from
`energy.mask` (the rotatable-bond mask, identical at every tier) rather than from `energy._M`,
so at `flex` the columns it drove were theta latents and at `full` bond-length latents, scaled
by a hardcoded pi.

THE TARGET IS `full` -- every internal degree of freedom free -- so a test that only covers
`torsion` would certify exactly the tier that already worked. Every case here runs all four.

EQUALITY IS EXACT, not approximate. Both paths evaluate the same affine map and the same NeRF
chain in the same dtype; anything above float64 round-off means they are computing different
functions, which is the failure this file exists to catch. The tolerance is 1e-9 A, roughly
seven orders below the 5-6 A the bug produced and still far above genuine round-off.
"""
import numpy as np
import pytest
import torch

from energies.conformer_data import (attach_states, check_state_convention,
                                     condition_from_energy, state_to_dof, states_to_positions)
from energies.conformer_torsions import ConformerTorsions

LEVELS = ('torsion', 'dihedral', 'flex', 'full')
TOL = 1e-9

#: spans the structural cases that broke differently: plain acyclic, branched, a RING (closure
#: encoding), a CHIRAL centre (parity must survive the round trip), and a LINEAR centre, which
#: used to raise IndexError at `dihedral` rather than merely disagree.
#: 'CC#N' and 'CCC#N' were added when the transverse chart landed. They are the cases whose
#: linear bend is COVERED by it -- so their theta and phi slots hold (u, v), the two builders
#: must agree about that, and the graph path must reconstruct through place_nerf_transverse.
#: 'CC#CCO' is NOT covered (its linear angle sits at a frame seed with collinear reference
#: frames) and is kept for the held-row case; the two are different tests, not a duplicate.
MOLECULES = ['CCCCO', 'CCC(C)CO', 'C1CCCCC1CO', 'C[C@H](O)CC=O', 'CC#CCO',
             'CC#N', 'CCC#N']


def _energy(smiles, level):
    # allow_constrained: 'CC#CCO' is here precisely BECAUSE its chart is incomplete at
    # 'full' -- it is the held-row contrast case for the transverse molecules. Without the
    # opt-in the refusal would turn this file's most interesting case into a skip, which is
    # coverage lost to a guard rather than a guard doing its job.
    return ConformerTorsions(smiles=smiles, device='cpu', level=level,
                             allow_constrained=True)


def _states(energy, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, energy.data_ndim, generator=g, dtype=torch.float64) * 2 - 1


def _graph(energy, smiles, x):
    cond = condition_from_energy(energy, identifier=smiles)
    return attach_states(cond, x, torch.zeros(x.shape[0], dtype=torch.float64),
                         identifier=smiles, periodic=energy.periodic_dims)


@pytest.fixture(autouse=True)
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.mark.parametrize('smiles', MOLECULES)
@pytest.mark.parametrize('level', LEVELS)
def test_both_builders_agree_at_every_tier(smiles, level):
    """The headline invariant: same state, same geometry, either route, any tier."""
    try:
        energy = _energy(smiles, level)
    except ValueError as exc:
        # `CC#CCO` at `torsion` has no rotatable bond once linear frames are excluded. That is
        # a separate open question about linear centres, not a reconstruction failure.
        pytest.skip(f'{smiles} does not build at {level}: {exc}')
    x = _states(energy)
    pos_energy = energy.build_positions(x).reshape(x.shape[0], -1, 3)
    pos_graph = states_to_positions(_graph(energy, smiles, x), x).reshape(x.shape[0], -1, 3)
    err = float((pos_energy - pos_graph).norm(dim=-1).max())
    assert err < TOL, f'{smiles} @ {level}: builders disagree by {err:.3e} A'


@pytest.mark.parametrize('level', LEVELS)
def test_stored_map_reproduces_dof_from_state(level):
    """The map itself, before any geometry: r, theta and phi must match coordinate-wise.

    Checked separately from the positions because the NeRF chain can mask a small coordinate
    error in one block, and because `full` is the first tier where r and theta are driven at
    all -- the old map left 27 of butanol's 39 columns structurally unrepresentable.
    """
    energy = _energy('CCCCO', level)
    x = _states(energy)
    r_e, th_e, ph_e = energy.dof_from_state(x)
    r_g, th_g, ph_g = state_to_dof(_graph(energy, 'CCCCO', x), x)
    # the graph path returns the batch-FLATTENED layout aligned with `batch_tree`; the energy
    # returns [B, n_block]. Same numbers, different shape.
    for name, a, b in (('r', r_e, r_g), ('theta', th_e, th_g), ('phi', ph_e, ph_g)):
        assert float((a - b.reshape(a.shape)).abs().max()) < TOL, f'{name} differs at {level}'


@pytest.mark.parametrize('level', LEVELS)
def test_every_state_column_is_actually_driven(level):
    """No latent column may be silently ignored.

    The old map read 2 columns out of 39 at `full` and dropped the other 37 on the floor --
    which is invisible in any test that only checks the reference conformer, since every
    defect here vanishes at x = 0.
    """
    energy = _energy('CCCCO', level)
    k = int(energy.data_ndim)
    batch = _graph(energy, 'CCCCO', _states(energy, n=2))
    seen = set()
    for field in ('ctree_r_col', 'ctree_th_col', 'ctree_ph_col'):
        seen |= {int(c) for c in getattr(batch, field).tolist() if int(c) >= 0}
    assert seen == set(range(k)), f'{level}: columns {sorted(set(range(k)) - seen)} drive nothing'


@pytest.mark.parametrize('level', LEVELS)
def test_reference_state_is_the_reference_conformer(level):
    """x = 0 must return the stored reference exactly.

    This passed before the fix too -- every defect was a delta-on-the-reference defect, so a
    smoke test that only built x = 0 saw nothing. Kept precisely so that is on the record.
    """
    energy = _energy('CCCCO', level)
    x = torch.zeros(2, energy.data_ndim, dtype=torch.float64)
    r, th, ph = state_to_dof(_graph(energy, 'CCCCO', x), x)
    assert float((r.reshape(2, -1) - energy.r0).abs().max()) < TOL
    assert float((th.reshape(2, -1) - energy.th0).abs().max()) < TOL


@pytest.mark.parametrize('smiles', ['CC#N', 'CCC#N'])
def test_a_covered_linear_bend_is_driven_not_held(smiles):
    """The point of the transverse chart, asserted on the energy the graph is built from.

    Without it these molecules lose their linear bend to `constrained_rows` and report fewer
    than 3N-6 columns at `full`. The equivalence tests above then still pass -- both builders
    agree on a constrained chart just as happily as on a complete one -- so agreement alone
    would never have shown the coordinate back.
    """
    energy = _energy(smiles, 'full')
    assert int(energy.transverse_angles.sum()) > 0, 'no covered linear bend to test'
    assert energy.uncovered_linear_angles == 0
    assert energy.data_ndim == 3 * energy.spec.n_atoms - 6
    assert getattr(energy, 'constrained_rows', 0) == 0
    # the flag has to reach the graph, or states_to_positions reads (u, v) as (theta, phi)
    mol = condition_from_energy(energy, identifier=smiles)
    assert bool(mol.ctree_transverse.any())
    assert int(mol.ctree_transverse.sum()) == int(energy.transverse_angles.sum())


@pytest.mark.parametrize('level', LEVELS)
def test_check_state_convention_passes(level):
    """The build-time gate must now be a working gate rather than a standing failure."""
    energy = _energy('CCCCO', level)
    mol = condition_from_energy(energy, identifier='CCCCO')
    assert check_state_convention(mol, energy) < TOL


def test_a_batch_of_different_molecules_reconstructs_per_row():
    """Mixed batches must use each row's own map, not the first graph's."""
    from energies.conformer_data import collate_conditions
    # ISOMERS, deliberately. `CCCCO` and `CCCCN` share k at `torsion` (2 rotatable bonds)
    # but not at `full`, where k is 3N-6 and they have different atom counts -- 39 vs 42.
    # A conditions file is one k at whatever tier it is built for.
    level, smis = 'full', ['CCCCO', 'CCC(C)O']
    ens = [_energy(s, level) for s in smis]
    assert len({e.data_ndim for e in ens}) == 1, 'pick two molecules with equal k'
    x = _states(ens[0], n=2)
    mols = [condition_from_energy(e, identifier=s) for e, s in zip(ens, smis)]
    batch = collate_conditions(mols)
    batch.torsion_state = x.reshape(-1)
    pos_graph = states_to_positions(batch, x)
    offset = 0
    for i, e in enumerate(ens):
        n = int(mols[i].num_nodes)
        want = e.build_positions(x[i:i + 1]).reshape(-1, 3)
        got = pos_graph[offset:offset + n]
        offset += n
        assert float((want - got).norm(dim=-1).max()) < TOL, f'row {i} used the wrong map'


def test_a_file_without_the_map_is_refused_by_name():
    """An old file must fail with instructions, not with a wrong geometry."""
    energy = _energy('CCCCO', 'full')
    x = _states(energy, n=2)
    batch = _graph(energy, 'CCCCO', x)
    del batch._store['ctree_r_col']
    with pytest.raises(AttributeError, match='build_conformer_conditions'):
        state_to_dof(batch, x)
