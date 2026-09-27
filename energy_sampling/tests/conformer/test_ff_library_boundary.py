"""Boundary: MXtalTools' force field and builder, fed a batch of DIFFERENT molecules.

The one-pass multi-molecule energy uses MXtalTools in a way MXtalTools never exercises
itself. `ff_from_mmff` only ever builds tiled copies of ONE molecule; here a `ForceField` is
assembled from several molecules' single-copy tables (energies/ff_library.py) and handed to
`intramolecular_energy`, and `build` / `log_jacobian` run on a `BatchedTree` read off a
collated batch of different molecules (`conformer_data.batch_tree`). Nothing on the
MXtalTools side promises that works, so it is pinned here, on the consumer's side:

  * every energy COMPONENT of every row equals that molecule's own single-copy result;
  * the mix covers terms present on some members and absent on others -- torsion,
    out-of-plane, stretch-bend and nonbonded pairs -- because `intramolecular_energy`
    branches on None versus empty (energy.py: torsion, stretch-bend, out-of-plane,
    electrostatics), and a batch in which NO row carries a term takes a path a
    homogeneous test never reaches;
  * the mixed tree rebuilds each member's geometry and each member's log J.
"""
import dataclasses
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from mxtaltools.conformers.builder import build, collate, log_jacobian
from mxtaltools.conformers.energy import intramolecular_energy

from energies.conformer_carrier import CarrierLayout, carrier_pad_condition
from energies.conformer_data import (batch_tree, collate_conditions, condition_from_energy,
                                     state_to_dof)
from energies.conformer_torsions import ConformerTorsions
from energies.ff_library import TERMS, ForceFieldLibrary

KW = dict(device='cpu', level='full')
# C: no torsion, no oop, no pairs | N: oop, no torsion | CO: torsion, no oop |
# CC(=O)N: torsion + oop | c1ccccc1O: a ring closure
MIX = ['C', 'N', 'CO', 'CC(=O)N', 'c1ccccc1O']
COMPONENTS = ('bond', 'angle', 'lj', 'torsion', 'stretch_bend', 'oop', 'electrostatic')


@contextmanager
def _float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _no_stretch_bend(ff):
    """A single-copy table WITHOUT the stretch-bend term. Every MMFF molecule the chart
    admits carries one, so the per-member-absent path for it is reachable only this way."""
    return dataclasses.replace(ff, sb_index=None, sb_batch=None, sb_k_ijk=None, sb_k_kji=None,
                               sb_r0_ij=None, sb_r0_kj=None, sb_theta0=None)


class _Mix:
    """Members, their library, a carrier-padded mixed batch, and one state per row."""

    def __init__(self, smis, force_field='mmff', rows=None, seed=0):
        with _float64():
            self.members = {s: ConformerTorsions(smiles=s, force_field=force_field, **KW)
                            for s in smis}
            self.lay = CarrierLayout(self.members)
            conds = {s: carrier_pad_condition(condition_from_energy(m, identifier=s), self.lay,
                                              s, m)
                     for s, m in self.members.items()}
            self.idents = list(self.members)
            self.lib = ForceFieldLibrary.from_members(self.members)
            self.rows = (np.arange(len(smis)).repeat(2) if rows is None
                         else np.asarray(rows))
            self.batch = collate_conditions([conds[self.idents[j]].__copy__()
                                             for j in self.rows])
            g = torch.Generator().manual_seed(seed)
            self.xm = [torch.rand((1, self.members[self.idents[j]].data_ndim), generator=g,
                                  dtype=torch.float64) * 1.6 - 0.8 for j in self.rows]
            self.X = torch.cat([self.lay.to_carrier(self.idents[j], x)
                                for j, x in zip(self.rows, self.xm)])
            self.lib_ids = torch.as_tensor(self.rows, dtype=torch.long)
            self.tree = batch_tree(self.batch)
            self.r, self.th, self.ph = state_to_dof(self.batch, self.X)
            self.pos = build(self.tree, self.r, self.th, self.ph)

    def row_pos(self, b):
        ptr = self.batch.ptr
        return self.pos[int(ptr[b]):int(ptr[b + 1])]

    def member(self, b):
        return self.members[self.idents[self.rows[b]]]


@pytest.fixture(scope='module')
def mix():
    return _Mix(MIX)


def _components(tree, pos, ff):
    _, comp = intramolecular_energy(tree, pos, ff, components=True)
    return comp


def test_the_mix_covers_presence_and_absence(mix):
    has = lambda f, t: getattr(f, t) is not None and getattr(f, t).numel() > 0
    ffs = [m.ff_single for m in mix.members.values()]
    for field in ('torsion_index', 'oop_index', 'pair_index'):
        assert any(has(f, field) for f in ffs) and not all(has(f, field) for f in ffs), field
    assert any(has(f, 'closure_index') for f in ffs)
    assert not mix.lay.is_identity


def test_the_mixed_tree_rebuilds_each_members_geometry_and_log_j(mix):
    ljac = log_jacobian(mix.tree, mix.r, mix.th)
    for b in range(len(mix.rows)):
        m = mix.member(b)
        assert torch.allclose(mix.row_pos(b), m.build_positions(mix.xm[b]), rtol=0, atol=1e-10)
        tree1, _ = m._batch(1)
        r, th, ph = m.dof_from_state(mix.xm[b])
        want = m._log_jac(tree1, r, th, ph, 1)
        assert torch.allclose(ljac[b:b + 1], want, rtol=1e-12, atol=1e-12)


def test_every_component_of_every_row_is_the_members_own(mix):
    ff = mix.lib.gather(mix.lib_ids, mix.batch.ptr[:-1])
    got = _components(mix.tree, mix.pos, ff)
    for b in range(len(mix.rows)):
        m = mix.member(b)
        tree1 = collate([m.spec])
        want = _components(tree1, mix.row_pos(b), m.ff_single)
        for c in COMPONENTS:
            assert torch.allclose(got[c][b:b + 1], want[c], rtol=1e-12, atol=1e-12), (
                mix.idents[mix.rows[b]], c, float(got[c][b]), float(want[c]))


def test_a_term_absent_from_the_whole_batch_is_emitted_as_the_members_none(mix):
    """Rows of C and N only: no torsion and no nonbonded pair anywhere in the batch; rows of
    C only: no out-of-plane either. The gathered table must take the member's own branch."""
    for keep in (['C', 'N'], ['C']):
        sel = np.flatnonzero(np.isin(np.asarray(mix.idents)[mix.rows], keep))
        sub = mix.batch.subsample_new_batch(sel)
        lib_ids = mix.lib_ids[torch.as_tensor(sel)]
        ff = mix.lib.gather(lib_ids, sub.ptr[:-1])
        assert ff.torsion_index is None and ff.tors_v is None
        assert ff.pair_index.shape == (0, 2)
        if keep == ['C']:
            assert ff.oop_index is None
        tree = batch_tree(sub)
        pos = build(tree, *state_to_dof(sub, mix.X[torch.as_tensor(sel)]))
        got = _components(tree, pos, ff)
        for k, b in enumerate(sel):
            m = mix.member(b)
            want = _components(collate([m.spec]), mix.row_pos(b), m.ff_single)
            for c in COMPONENTS:
                assert torch.allclose(got[c][k:k + 1], want[c], rtol=1e-12, atol=1e-12), c


def test_a_member_without_stretch_bend_beside_members_with_it(mix):
    """The per-member-absent path for stretch-bend, which no admitted MMFF molecule reaches."""
    co = mix.members['CO']
    ffs = [m.ff_single for m in mix.members.values()] + [_no_stretch_bend(co.ff_single)]
    lib = ForceFieldLibrary(ffs, [m.spec.n_atoms for m in mix.members.values()] +
                            [co.spec.n_atoms],
                            [np.asarray(m.spec.z) for m in mix.members.values()] +
                            [np.asarray(co.spec.z)],
                            [m.log_chart_jacobian for m in mix.members.values()] +
                            [co.log_chart_jacobian], dtype=torch.float64)
    # every row of CO is re-pointed at the stretch-bend-free entry, the rest kept
    no_sb = len(ffs) - 1
    lib_ids = torch.where(mix.lib_ids == mix.idents.index('CO'), no_sb, mix.lib_ids)
    got = _components(mix.tree, mix.pos, lib.gather(lib_ids, mix.batch.ptr[:-1]))
    for b in range(len(mix.rows)):
        m = mix.member(b)
        ff1 = ffs[int(lib_ids[b])]
        want = _components(collate([m.spec]), mix.row_pos(b), ff1)
        for c in COMPONENTS:
            assert torch.allclose(got[c][b:b + 1], want[c], rtol=1e-12, atol=1e-12), c
    co_rows = (lib_ids == no_sb)
    assert co_rows.any() and bool((got['stretch_bend'][co_rows] == 0).all())
    assert bool((got['stretch_bend'][~co_rows] != 0).all())


def test_the_reference_force_field_packs_its_absent_terms_as_none():
    """`ff_from_reference` carries no torsion, stretch-bend, out-of-plane or electrostatic
    term and uses soft-core LJ: every optional field is None on every member."""
    m = _Mix(['CO', 'CCO', 'CCCO'], force_field='reference')
    ff = m.lib.gather(m.lib_ids, m.batch.ptr[:-1])
    assert ff.torsion_index is None and ff.sb_index is None and ff.oop_index is None
    assert ff.vdw_rstar is None and ff.ele_qq is None and ff.bond_cs is None
    got = _components(m.tree, m.pos, ff)
    for b in range(len(m.rows)):
        mem = m.member(b)
        want = _components(collate([mem.spec]), m.row_pos(b), mem.ff_single)
        for c in COMPONENTS:
            assert torch.allclose(got[c][b:b + 1], want[c], rtol=1e-12, atol=1e-12), c


def test_the_library_refuses_what_it_cannot_pack(mix):
    co = mix.members['CO']
    n, z, ch = [co.spec.n_atoms], [np.asarray(co.spec.z)], [co.log_chart_jacobian]
    tiled = co._batch(3)[1]
    with pytest.raises(ValueError, match='SINGLE-copy'):
        ForceFieldLibrary([tiled], n, z, ch)
    other = dataclasses.replace(co.ff_single, lj_k_factor=co.ff_single.lj_k_factor + 1.0)
    with pytest.raises(ValueError, match='differs across members'):
        ForceFieldLibrary([co.ff_single, other], n * 2, z * 2, ch * 2)
    torsion_without_values = dataclasses.replace(co.ff_single, tors_v=None)
    with pytest.raises(ValueError, match='present on some members'):
        ForceFieldLibrary([co.ff_single, torsion_without_values], n * 2, z * 2, ch * 2)
    with pytest.raises(ValueError, match='atomic numbers'):
        ForceFieldLibrary([co.ff_single], n, [np.asarray(co.spec.z)[:-1]], ch)


def test_every_forcefield_field_is_packed():
    """A ForceField field the library does not pack would be DROPPED from every gathered
    table. The library refuses that at construction; this pins the current field set."""
    packed = {'n_mols', 'lj_k_factor', 'bond_cs', 'angle_cb', 'ele_delta', 'ele_dielectric',
              'vdw_softcore_frac'}
    for ifield, _, vfields, bfield, _ in TERMS.values():
        packed.update((ifield, bfield, *vfields))
    from mxtaltools.conformers.energy import ForceField
    assert {f.name for f in dataclasses.fields(ForceField)} == packed
