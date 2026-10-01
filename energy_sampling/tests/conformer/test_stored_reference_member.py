"""A member built from a STORED reference conformer is the member that reference came from.

``ConformerTorsions(reference_positions=...)`` skips the ETKDG embedding and the MMFF94
relaxation and takes the given geometry as the reference. What each test would catch, at level
`full` (and `torsion` for one), CPU, float64:

  * a rebuilt member that differs from its original in ANY attribute -- the tree, the reference
    and its internals, the column map, the force field tables, the stereo table, the linearity
    and ring flags, the aromaticity MMFF typing leaves on the RDKit molecule -- or in its
    condition graph, its signature or its energy at a few states;
  * a placement-order stored reference (a conditions file's ``pos``, a database's ``ref_pos``)
    put back in the wrong RDKit order, with the order derived from the bond graph or given;
  * a stored reference that is not the member's passing ``stored_reference_mismatch``;
  * a stored-reference build that runs the embedding (``seed`` must not matter);
  * ConformerModeller's energy init building a member from anything but the conditions file's
    stored reference, reading that file more than once, or accepting a member that does not
    reproduce it;
  * the reference-table builder embedding its member where the conditions file stores one;
  * ``release_batch_cache`` changing an energy, a built member or a thermal-tiled set member
    keeping its batched force field (1 to 2 MB a member), and the conformer energy losing the
    two names train.py's energy-reference path reads off it.

The molecules cover an aromatic ring MMFF re-types (4-pyridone), a nitrile (transverse pair), an
alkyne (dummy frame), a ring, and tagged stereocentres and a double bond under the stereo lock.

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -p no:cacheprovider tests/conformer/test_stored_reference_member.py
"""
import dataclasses
import math

import numpy as np
import pytest
import torch

MOLECULES = ('O=c1cc[nH]cc1', 'N#CC[C@H](C)O', 'C#CCC(=O)C', 'OC1CCCC1', 'C[C@@H](O)[C@H](N)CC',
             r'C/C=C/CO')
KW = dict(level='full', force_field='mmff', stereo_coeff=300.0)


@pytest.fixture(autouse=True)
def _float64():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def _same(a, b, path, seen):
    """Assert ``a`` and ``b`` are equal all the way down, naming the first difference."""
    if id(a) in seen:
        return
    seen.add(id(a))
    assert type(a) is type(b), f'{path}: {type(a).__name__} against {type(b).__name__}'
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape, f'{path}: {a.dtype}{tuple(a.shape)}'
        assert torch.equal(a, b) or (a.is_floating_point() and torch.equal(
            torch.nan_to_num(a, nan=1.2345), torch.nan_to_num(b, nan=1.2345))), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'), path
    elif isinstance(a, float):
        assert a == b or (math.isnan(a) and math.isnan(b)), f'{path}: {a!r} against {b!r}'
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{i}]', seen)
    elif isinstance(a, dict):
        assert list(a) == list(b), f'{path}: keys differ'
        for k in a:
            _same(a[k], b[k], f'{path}[{k!r}]', seen)
    elif isinstance(a, (str, int, bool, type(None), np.generic, torch.dtype, torch.device)):
        assert a == b, f'{path}: {a!r} against {b!r}'
    elif type(a).__module__.startswith('rdkit'):
        from rdkit import Chem
        assert Chem.MolToMolBlock(a) == Chem.MolToMolBlock(b), path
        flags = lambda m: ([(x.GetIsAromatic(), x.GetChiralTag(), x.GetHybridization())
                            for x in m.GetAtoms()]
                           + [(x.GetIsAromatic(), x.GetBondType(), x.GetStereo())
                              for x in m.GetBonds()])
        assert flags(a) == flags(b), f'{path}: atom or bond flags differ'
    elif dataclasses.is_dataclass(a) or hasattr(a, '__dict__'):
        _same(vars(a), vars(b), path, seen)
    else:
        assert a == b, f'{path}: {a!r} against {b!r}'


def _pair(smiles, **kw):
    from energies.conformer_torsions import ConformerTorsions
    fresh = ConformerTorsions(smiles=smiles, device='cpu', **{**KW, **kw})
    rd = np.asarray(fresh.mol.GetConformer().GetPositions(), dtype=np.float64)
    # another seed: a stored reference must not be re-embedded
    stored = ConformerTorsions(smiles=smiles, device='cpu', reference_positions=rd,
                               **{**KW, **kw, 'seed': 12345})
    return fresh, stored


@pytest.mark.parametrize('smiles', MOLECULES)
def test_a_member_rebuilt_from_its_reference_is_the_member(smiles):
    from conformer_modeller import ConformerModeller
    from energies.conformer_data import bake_energies, condition_from_energy
    fresh, stored = _pair(smiles)
    assert (fresh.reference_source, stored.reference_source) == ('embedded', 'stored')
    a, b = dict(vars(fresh)), dict(vars(stored))
    a.pop('reference_source'), b.pop('reference_source')
    _same(a, b, smiles, set())
    assert ConformerModeller._member_signature(smiles, fresh) == \
        ConformerModeller._member_signature(smiles, stored)
    _same(dict(condition_from_energy(fresh, identifier=smiles)._store),
          dict(condition_from_energy(stored, identifier=smiles)._store), 'graph', set())
    x = torch.as_tensor(np.random.default_rng(0).uniform(-1, 1, (6, fresh.data_ndim)))
    x[0] = 0.0
    with torch.no_grad():
        assert torch.equal(bake_energies(fresh, x), bake_energies(stored, x))


def test_the_torsion_tier_and_the_reference_force_field_rebuild_too():
    for kw in (dict(level='torsion'), dict(force_field='reference', stereo_coeff=0.0)):
        fresh, stored = _pair('O=c1cc[nH]cc1CCO', **kw)
        a, b = dict(vars(fresh)), dict(vars(stored))
        a.pop('reference_source'), b.pop('reference_source')
        _same(a, b, str(kw), set())


@pytest.mark.parametrize('give_perm', (False, True))
def test_a_placement_order_reference_goes_back_to_rdkit_order(give_perm):
    from energies.conformer_torsions import (ConformerTorsions, rdkit_order_reference,
                                             stored_reference_mismatch)
    for smiles in MOLECULES:
        fresh = ConformerTorsions(smiles=smiles, device='cpu', **KW)
        pos = fresh.ref_pos.detach().numpy()                 # placement order, as stored
        rd, perm = rdkit_order_reference(smiles, pos, level='full', z=fresh.spec.z,
                                         perm=fresh.spec.perm if give_perm else None)
        assert np.array_equal(perm, np.asarray(fresh.spec.perm))
        assert np.array_equal(rd, fresh.mol.GetConformer().GetPositions())
        stored = ConformerTorsions(smiles=smiles, device='cpu', reference_positions=rd, **KW)
        assert stored_reference_mismatch(stored, pos, perm) == ''


def test_a_reference_that_is_not_the_members_is_named():
    from energies.conformer_torsions import (ConformerTorsions, rdkit_order_reference,
                                             stored_reference_mismatch)
    fresh = ConformerTorsions(smiles='CCCO', device='cpu', **KW)
    pos = fresh.ref_pos.detach().numpy()
    assert 'differs' in stored_reference_mismatch(fresh, pos + 1e-12)
    assert 'placement order' in stored_reference_mismatch(fresh, pos,
                                                          np.roll(fresh.spec.perm, 1))
    with pytest.raises(ValueError, match='atomic numbers'):
        rdkit_order_reference('CCCO', pos, level='full', z=np.roll(fresh.spec.z, 1))
    with pytest.raises(ValueError, match='shape'):
        ConformerTorsions(smiles='CCCO', device='cpu', reference_positions=pos[:-1], **KW)


def test_the_modeller_builds_each_member_from_the_conditions_file(tmp_path, monkeypatch):
    """ConformerModeller's energy init takes each member's reference from molecules_path.

    Every stored `pos` is rigidly rotated first: no chart, energy or stereo element changes,
    and no embedding would produce it, so a member reproducing it was built from it. The file
    is read once; a member that does not reproduce its stored reference is refused.
    """
    import types

    import build_conformer_conditions as bcc
    from conformer_modeller import ConformerModeller
    from energies.conformer_data import save_condition_file
    from energies.multi_conformer import MultiConformerTorsions

    path = tmp_path / 'cond.pt'
    bcc.main(['--smiles', *MOLECULES, '--carrier', '--out', str(path), '--level', 'full',
              '--force-field', 'mmff', '--stereo-coeff', '300', '--threads', '1'])
    b = torch.load(path, weights_only=False)['prior']
    c, s = math.cos(0.7), math.sin(0.7)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=b.pos.dtype)
    b.pos = b.pos @ rot.T
    save_condition_file(b, path)
    ptr = b.ptr.tolist()
    want = {i: b.pos[ptr[j]:ptr[j + 1]] for j, i in enumerate(b.identifier)}

    m = ConformerModeller.__new__(ConformerModeller)
    m.args = types.SimpleNamespace(molecules_path=str(path))
    smis, idents = m._condition_set_molecules()
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('read the file twice'))
    assert m._condition_set_molecules() == (smis, idents)
    refs = m._stored_references(smis, idents, 'full')
    assert set(refs) == set(idents) == set(MOLECULES)
    m.energy_function = MultiConformerTorsions(
        smis, identifiers=idents, reference_positions={i: r[0] for i, r in refs.items()},
        device='cpu', **KW)
    m._check_stored_references(refs, 0.0)
    for ident, mem in m.energy_function._members.items():
        assert mem.reference_source == 'stored'
        assert torch.equal(mem.ref_pos, want[ident]), ident
    ident = idents[1]
    rd, perm, pos = refs[ident]
    with pytest.raises(SystemExit, match='do not reproduce'):
        m._check_stored_references({ident: (rd, np.roll(perm, 1), pos)}, 0.0)


def test_the_reference_builder_builds_its_member_from_the_stored_reference():
    """build_conformer_references.build_member(pos=, z=): a rigidly rotated stored reference,
    which no embedding produces, comes back as the member's own."""
    import build_conformer_references as bcr
    from energies.conformer_torsions import ConformerTorsions
    kw = {k: v for k, v in KW.items()}
    for smiles in MOLECULES[:3]:
        fresh = ConformerTorsions(smiles=smiles, device='cpu', **kw)
        c, s = math.cos(0.7), math.sin(0.7)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pos = fresh.ref_pos.numpy() @ rot.T
        member = bcr.build_member(smiles, kw, pos=pos, z=np.asarray(fresh.spec.z))
        assert member.reference_source == 'stored'
        assert bcr.check_member_matches(member, np.asarray(fresh.spec.z), pos) == 0.0
        assert bcr.build_member(smiles, kw).reference_source == 'embedded'


# ------------------------------------------------------------------ the batch cache


def _cache_sizes(member):
    assert set(member._tree_cache) == set(member._ff_cache)
    return set(member._tree_cache)


def test_releasing_the_batch_cache_changes_no_energy():
    """release_batch_cache drops every cached batch but size 1, and the next evaluation is
    the one before it, bit for bit."""
    from energies.conformer_data import bake_energies
    from energies.conformer_torsions import ConformerTorsions
    en = ConformerTorsions(smiles='C[C@@H](O)[C@H](N)CC', device='cpu', **KW)
    x = torch.as_tensor(np.random.default_rng(1).uniform(-1, 1, (7, en.data_ndim)))
    with torch.no_grad():
        before = bake_energies(en, x)
    assert 7 in _cache_sizes(en)
    en.release_batch_cache()
    assert _cache_sizes(en) <= {1}
    with torch.no_grad():
        assert torch.equal(bake_energies(en, x), before)


def test_a_built_member_and_a_tiled_set_member_keep_no_batch_cache():
    """What grows host (and device) memory by 1 to 2 MB a member: the batch build_member's
    checks leave on the member, and the one the thermal tile's curvature evaluation leaves
    on every member of a set."""
    import build_conformer_conditions as bcc
    from energies.multi_conformer import MultiConformerTorsions
    from energies.thermal_tile import ThermalTile
    mb = bcc.build_member('CCCO', 'CCCO', KW, carrier=True)
    assert _cache_sizes(mb.energy) <= {1}
    smis = ['CCCO', 'CCCCO', 'CC(C)CO']
    en = MultiConformerTorsions(smis, identifiers=smis, device='cpu', **KW)
    tile = ThermalTile(en)
    for ident in smis:
        sigma, _ = tile.widths(ident)
        assert float(sigma.abs().sum()) > 0
        member = en._members[ident]
        if member is not en:
            assert _cache_sizes(member) <= {1}, ident


def test_the_conformer_energy_carries_the_trainers_energy_reference_surface():
    """train.py::Modeller.train wraps the prior intake in energy_function.allow_unreferenced(),
    init_energy_reference reads energy_function.energy_reference_mode, and the absolute reward
    floors call energy_function.unreferenced_log_r; without them on this energy the conformer
    route dies at startup or at its first floor. The mode is None: nothing is installed."""
    from energies.conformer_torsions import ConformerTorsions
    from energies.molecular_crystal import resolve_energy_reference
    en = ConformerTorsions(smiles='CCCO', device='cpu', **KW)
    with en.allow_unreferenced():
        pass
    log_r = torch.tensor([-3.0, 0.5])
    assert en.unreferenced_log_r(log_r, torch.tensor([0, 1])) is log_r

    def never():
        raise AssertionError('a table was computed with no reference mode')
    assert resolve_energy_reference(en.energy_reference_mode, None, never, 1.0) is None
    assert resolve_energy_reference(en.energy_reference_mode, {'mode': None, 'table': None},
                                    never, 1.0) is None
