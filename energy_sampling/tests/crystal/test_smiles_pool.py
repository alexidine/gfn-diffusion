"""CPU tests for data_processing/smiles_pool.py, the small-molecule SMILES pool.

WHAT THIS FILE IS FOR. The pool cuts larger molecules into every H-capped fragment of 4-9 heavy atoms. Failures that
would run without an error: a non-H/C/N/O/F, charged or radical molecule let in; a ring bond cut (a fragment with an
opened ring); a terminal F (or other H-free terminal atom) cut off; a cut aromatic N left without its H (an
unparseable fragment silently dropped); fragments outside the window; and a per-parent cap that is not reproducible.

    python -m pytest -q tests/crystal/test_smiles_pool.py
"""
import os
import sys

from rdkit import Chem

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_processing import smiles_pool as sp  # noqa: E402


def _can(s):
    m = Chem.MolFromSmiles(s)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, isomericSmiles=False)


def test_prefilter():
    for s in ('CCCl', 'CS', 'C[Na]', 'CBr', 'CP', 'C[Si](C)C', 'CI'):
        assert not sp.prefilter(s), s
    for s in ('CC(=O)O', 'c1cc[nH]c1', 'C[C@H](O)F', 'C/C=C/C', 'CC[N+](=O)[O-]', 'CCO.CC'):
        assert sp.prefilter(s), s


def test_components_keep_neutral_closed_shell_hcnof():
    assert [sp.canon(m) for m in sp.components('CCO.c1ccccc1')] == [_can('CCO'), _can('c1ccccc1')]
    assert sp.components('CC(=O)[O-].[NH4+]') == []          # each ion is charged
    assert sp.components('[CH3]') == []                       # radical
    assert sp.components('C[N+](=O)[O-]') != []               # charge-separated but neutral: kept
    assert sp.components('not a smiles') == []


def test_cuttable_bonds():
    mol = Chem.MolFromSmiles('CCc1ccccc1F')
    cut = set(sp.cuttable_bonds(mol))
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        syms = {a1.GetSymbol(), a2.GetSymbol()}
        if b.IsInRing():
            assert b.GetIdx() not in cut                      # ring bonds never
        elif 'F' in syms:
            assert b.GetIdx() not in cut                      # a terminal F has no H: never cut off
        else:
            assert b.GetIdx() in cut                          # CH3-CH2 and CH2-ring are cut
    carbonyl = Chem.MolFromSmiles('CC(C)=O')
    assert all(carbonyl.GetBondWithIdx(i).GetBondType() == Chem.BondType.SINGLE
               for i in sp.cuttable_bonds(carbonyl))           # the C=O double bond is never cut


def test_fragments_of_paracetamol():
    frs = set(sp.fragments(Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1'), max_per_parent=1000))
    for want in ('Oc1ccccc1', 'Nc1ccccc1', 'Nc1ccc(O)cc1', 'CC(N)=O', 'O=CNc1ccccc1', 'c1ccccc1'):
        assert _can(want) in frs, want
    assert _can('CC(=O)Nc1ccc(O)cc1') not in frs              # the 11-heavy-atom parent itself is outside the window
    for f in frs:
        m = Chem.MolFromSmiles(f)
        assert m is not None and 4 <= m.GetNumHeavyAtoms() <= 9, f
        assert sum(a.GetFormalCharge() for a in m.GetAtoms()) == 0, f


def test_cut_aromatic_nitrogen_gets_its_hydrogen():
    frs = set(sp.fragments(Chem.MolFromSmiles('CCCCCn1cccc1'), max_per_parent=1000))
    assert _can('c1cc[nH]c1') in frs                           # pyrrole, not a dropped unkekulisable n


def test_rings_are_never_cut():
    parent = Chem.MolFromSmiles('c1ccc(CCc2ccncc2)cc1')
    frs = sp.fragments(parent, max_per_parent=1000)
    assert frs
    for f in frs:
        m = Chem.MolFromSmiles(f)                              # kept: RingInfo dies with its molecule
        ri = m.GetRingInfo()
        assert ri.NumRings() >= 1, f                           # every piece of 4+ heavy atoms here holds a whole ring
        assert all(len(r) == 6 for r in ri.AtomRings()), f
    assert _can('Cc1ccncc1') in set(frs) and _can('CCc1ccccc1') in set(frs)


def test_per_parent_cap_is_reproducible():
    mol = Chem.MolFromSmiles('CCOC(=O)c1ccc(NC(=O)CCN)cc1OC')
    full = sp.fragments(mol, max_per_parent=1000)
    assert len(full) > 4
    a = sp.fragments(mol, max_per_parent=4, seed_key='k')
    b = sp.fragments(mol, max_per_parent=4, seed_key='k')
    assert a == b and len(a) == 4 and set(a) <= set(full)


def test_process_lines_end_to_end():
    lines = ['CCO 1', 'CC(=O)Nc1ccc(O)cc1 2', 'CCCl 3', 'C' * 50 + ' 4', 'c1ccccc1 5', '']
    agg, stats = sp.process_lines(lines, 4, 9, 40, 1000)
    assert stats['lines'] == 6 and stats['prefilter_kept'] == 4
    assert stats['too_large'] == 1 and stats['parents'] == 1
    assert _can('CCO') not in agg                              # 3 heavy atoms: below the window
    assert agg[_can('c1ccccc1')][0] == 1                        # benzene: once directly
    assert agg[_can('c1ccccc1')][1] == 1                        # and once as a fragment of paracetamol
    assert agg[_can('Oc1ccccc1')] == [0, 1, 11]
