"""A conditions or graph-form prior file whose stored reference conformer is not the member's
the run rebuilt is REFUSED at load (ConformerModeller._refuse_reference_mismatch).

WHAT WAS WRONG. A stored conformer state is a delta from its member's reference conformer, and
the run re-embeds every member from its SMILES. Another RDKit re-embeds many members
differently (45 of 88 small-rung members, by up to 3.6 A, 2025.03.5 against 2025.09.4), and
nothing at run time compared the file's `pos` with the rebuilt `ref_pos`: `_resolve_rows`
checks `z` and the chart flags, which a re-embedding leaves unchanged.

WHAT IS PINNED, on a small carrier set (MultiConformerTorsions, `full`, MMFF) and on a single
chart, CPU, float64:
  * a file built by the same members passes, on the carrier and on one chart;
  * a reference moved by 1e-5 A passes, and by 0.01 A is refused, naming the identifier and
    the deviation;
  * a member re-embedded under another ETKDG seed -- what another RDKit does -- is refused;
  * a graph with another atom count under the identifier is refused as such;
  * both loaders call the guard (the conditions file and the graph-form prior).

    CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/conformer/test_reference_guard.py
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

from conformer_modeller import REFERENCE_POS_TOL, ConformerModeller
from energies.conformer_torsions import ConformerTorsions

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

KW = dict(level='full', force_field='mmff', device='cpu', dtype=torch.float64)
SET = ['CCO', 'CCCO', 'CC(C)CO']


def _set(smis, **kw):
    from energies.multi_conformer import MultiConformerTorsions
    return MultiConformerTorsions(smis, identifiers=smis, **{**KW, **kw})


def _file(maker, rows_per=2):
    """A collated conditions batch as the set builder writes it: each member's condition
    graph (carrier-padded on a carrier), `rows_per` copies each."""
    from energies.conformer_carrier import carrier_pad_condition
    from energies.conformer_data import collate_conditions, condition_from_energy
    lay = maker.carrier
    rows = []
    for ident, m in maker._members.items():
        c = condition_from_energy(m, identifier=ident)
        c = carrier_pad_condition(c, lay, ident, m) if lay is not None else c
        rows += [c.__copy__() for _ in range(rows_per)]
    return collate_conditions(rows)


def _guard(energy, batch):
    fake = SimpleNamespace(energy_function=energy)
    ConformerModeller._refuse_reference_mismatch(fake, batch, 'x.pt', 'conditions')


@pytest.fixture(scope='module')
def carrier_set():
    maker = _set(SET)
    assert maker.carrier is not None, 'the set must exercise the carrier layout'
    return maker, _file(maker)


def test_the_same_members_pass(carrier_set, capsys):
    maker, batch = carrier_set
    _guard(maker, batch)
    assert 'reference conformers match' in capsys.readouterr().out


def test_a_single_chart_passes_and_a_moved_reference_is_refused():
    from energies.conformer_data import collate_conditions, condition_from_energy
    en = ConformerTorsions(smiles='CCCO', **KW)
    c = condition_from_energy(en, identifier='CCCO')
    batch = collate_conditions([c, c.__copy__()])
    _guard(en, batch)
    batch.pos[0, 0] += 0.01
    with pytest.raises(SystemExit, match="'CCCO' \\(0.01 A\\)"):
        _guard(en, batch)


@pytest.mark.parametrize('shift,refused', [(1e-5, False), (0.01, True)])
def test_the_tolerance_separates_rounding_from_a_moved_reference(carrier_set, shift, refused):
    maker, batch = carrier_set
    moved = batch.clone()
    g = list(moved.identifier).index('CCCO')
    moved.pos[int(moved.ptr[g]) + 1] += shift
    assert (shift > REFERENCE_POS_TOL) == refused
    if not refused:
        _guard(maker, moved)
        return
    with pytest.raises(SystemExit) as e:
        _guard(maker, moved)
    msg = str(e.value)
    assert "1 of 3 condition(s)" in msg and "'CCCO'" in msg and 'rebuild the rung' in msg
    assert 'max deviation 0.01 A' in msg


def test_a_member_re_embedded_under_another_seed_is_refused(carrier_set):
    """What another RDKit does: the same SMILES, z, chart flags and carrier layout, another
    reference geometry."""
    maker, batch = carrier_set
    other = _set(SET, seed=7)
    moved = [i for i in SET if float((other._members[i].ref_pos
                                      - maker._members[i].ref_pos).abs().max()) > REFERENCE_POS_TOL]
    assert moved, 'seed 7 re-embeds no member differently; pick another seed'
    with pytest.raises(SystemExit, match='store a reference conformer that is not the one'):
        _guard(other, batch)


def test_another_atom_count_under_an_identifier_is_refused(carrier_set):
    from energies.conformer_data import collate_conditions, condition_from_energy
    maker, _ = carrier_set
    c = condition_from_energy(maker._members['CCO'], identifier='CCCO')   # CCO's graph as CCCO
    batch = collate_conditions([c, c.__copy__()])
    with pytest.raises(SystemExit, match="'CCCO' \\(atom count differs\\)"):
        _guard(maker, batch)


def test_an_identifier_without_a_member_is_refused(carrier_set):
    from energies.conformer_data import collate_conditions, condition_from_energy
    maker, _ = carrier_set
    c = condition_from_energy(ConformerTorsions(smiles='CCN', **KW), identifier='CCN')
    batch = collate_conditions([c, c.__copy__()])
    with pytest.raises(SystemExit, match='have no member'):
        _guard(maker, batch)


def test_both_loaders_call_the_guard():
    for fn in (ConformerModeller.init_mol_dataset, ConformerModeller.init_prior_dataset):
        assert 'self._refuse_reference_mismatch(batch, ' in inspect.getsource(fn), fn.__name__
