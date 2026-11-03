"""Day-landscape benchmark pieces whose arithmetic the coverage numbers rest on (``eval/day_bench``).

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_day_bench_protocol.py

What each test would catch:

  * a chiral landscape read from one CSV only (``_all.csv`` can hold the Sohncke rows alone), an id whose block sits in
    another CIF, or a corrected energy joined to the wrong id or made relative to the wrong lowest;
  * a centred cell left conventional (twice the molecules to relax) or reduced to a cell that is not a sublattice;
  * a van der Waals overlap that counts bonded atoms of one molecule, or misses a molecule touching its own image;
  * single linkage chaining three minima into one mode whose ends lie beyond the tolerance;
  * a mode the correction does not cover dropped from the window, or kept when FIT+DMA puts it outside;
  * a converged draw charged the full depth, or cumulative calls read off the wrong prefix of an order.
"""
import numpy as np
import pytest
from ase import Atoms

from eval.day_bench.day_landscape import in_window, landscape, primitive
from eval.day_bench.protocol_refine import overlap
from eval.day_bench.protocol_report import at_calls, draw_costs, mode_table, modes, representable


def _block(name):
    return f'data_{name}\n_cell_length_a 5\n'


def test_landscape_unions_every_csv_and_joins_corrections(tmp_path):
    d = tmp_path / 'abc'
    d.mkdir()
    head = 'id,spacegroup,density,energy,minimization_step,trial_number,minimization_time,metadata\n'
    (d / 'abc_all.csv').write_text(head + 'abc-1,19,1.2,-80.0,3,1,1.0,\n')  # Sohncke rows only
    (d / 'abc_racsg.csv').write_text(head + 'abc-2,14,1.2,-82.0,3,2,1.0,\nabc-3,2,1.2,-70.0,3,3,1.0,\n')
    (d / 'abc_sohncke.cif').write_text(_block('abc-1'))
    (d / 'abc_racsg.cif').write_text(_block('abc-2') + _block('abc-3'))
    corr = tmp_path / 'corrected.csv'
    corr.write_text('structure,zprime,energy,uncertainty,raw_energy,corrected_energy_kjmol,corrected_energy_eV\n'
                    'abc-1,4,0,0,0,-90.0,0\nabc-2,4,0,0,0,-85.0,0\n')
    rows = landscape(str(d), str(corr), 'abc')
    assert sorted(rows) == ['abc-1', 'abc-2', 'abc-3']
    assert rows['abc-1']['cif'].endswith('abc_sohncke.cif') and rows['abc-3']['cif'].endswith('abc_racsg.cif')
    assert rows['abc-2']['fitdma_rel'] == 0.0 and rows['abc-1']['fitdma_rel'] == pytest.approx(2.0)
    assert rows['abc-1']['corrected_rel'] == 0.0 and rows['abc-2']['corrected_rel'] == pytest.approx(5.0)
    assert rows['abc-3']['corrected'] is None and rows['abc-3']['corrected_rel'] is None
    assert in_window(rows['abc-1'], 3) and not in_window(rows['abc-3'], 10)
    assert in_window(rows['abc-3'], 12)  # FIT+DMA alone where there is no correction


def test_landscape_refuses_an_id_without_a_block(tmp_path):
    d = tmp_path / 'abc'
    d.mkdir()
    (d / 'abc.csv').write_text('id,spacegroup,density,energy\nabc-1,14,1.2,-80.0\n')
    (d / 'abc.cif').write_text(_block('abc-9'))
    with pytest.raises(AssertionError, match='no CIF block'):
        landscape(str(d))


def test_primitive_halves_a_c_centred_cell():
    rng = np.random.default_rng(0)
    frac = rng.random((5, 3)) * 0.4
    sym = ['C', 'N', 'O', 'C', 'H']
    conv = Atoms(sym + sym, scaled_positions=np.concatenate([frac, (frac + [0.5, 0.5, 0.0]) % 1]),
                 cell=[7.0, 9.0, 11.0], pbc=True)
    prim = primitive(conv)
    assert len(prim) == 5
    assert prim.get_volume() == pytest.approx(conv.get_volume() / 2, rel=1e-6)
    assert sorted(prim.get_chemical_symbols()) == sorted(sym)


def test_primitive_keeps_a_primitive_cell():
    rng = np.random.default_rng(1)
    atoms = Atoms('CNOCH', scaled_positions=rng.random((5, 3)), cell=[7.0, 8.0, 9.0], pbc=True)
    assert len(primitive(atoms)) == 5


def _labelled(symbols, pos, labels, cell):
    a = Atoms(symbols, positions=pos, cell=cell, pbc=True)
    a.arrays['bb_indices'] = np.array(labels)
    return a


def test_overlap_reads_intermolecular_heavy_pairs_only():
    # two C2 molecules (bond 1.5 A, ignored as intramolecular), 2.55 A apart between molecules: 2.55 / 3.4 = 0.75
    pos = [[0, 0, 0], [1.5, 0, 0], [0, 2.55, 0], [1.5, 2.55, 0]]
    a = _labelled('CCCC', pos, [0, 0, 1, 1], [20, 20, 20])
    assert overlap(a) == pytest.approx(0.75, abs=1e-6)
    # hydrogens never count: an H 1.0 A from the other molecule's C leaves the C-C ratio in charge
    b = _labelled('CCCCH', pos + [[0, 3.55, 0]], [0, 0, 1, 1, 0], [20, 20, 20])
    assert overlap(b) == pytest.approx(0.75, abs=1e-6)
    # far apart: capped at 1
    c = _labelled('CCCC', [[0, 0, 0], [1.5, 0, 0], [0, 6, 0], [1.5, 6, 0]], [0, 0, 1, 1], [20, 20, 20])
    assert overlap(c) == 1.0


def test_overlap_counts_a_molecule_touching_its_own_image():
    # one C2 molecule in a cell 2.0 A long along x: atom 1 sits 0.5 A from atom 0's image
    a = _labelled('CC', [[0, 0, 0], [1.5, 0, 0]], [0, 0], [2.0, 20, 20])
    assert overlap(a) == pytest.approx(0.5 / 3.4, abs=1e-6)


def test_modes_complete_linkage_does_not_chain():
    tol = 0.1
    D = np.array([[0.0, 0.08, 0.16], [0.08, 0.0, 0.08], [0.16, 0.08, 0.0]])
    lab = modes(D, tol)
    assert len(set(lab)) == 2  # single linkage would give one mode spanning 0.16
    for i in range(3):
        for j in range(3):
            if lab[i] == lab[j]:
                assert D[i, j] <= tol


def test_mode_table_window_uses_both_rankings():
    lab_of = np.array([0, 0, 1, 2, 3])
    z = dict(ref_fitdma_rel=np.array([0.0, 1.0, 5.0, 9.0, 3.0]),
             ref_corrected_rel=np.array([2.0, np.nan, 9.5, 1.0, np.nan]),
             ref_zprim=np.array([4, 4, 2, 8, 4]))
    mt = mode_table(z, lab_of, 8.0)
    assert mt['fitdma'].tolist() == [0.0, 5.0, 9.0, 3.0]
    assert mt['corrected'][0] == 2.0 and np.isnan(mt['corrected'][3])
    assert mt['in_window'].tolist() == [True, False, False, True]  # mode 1: corrected out; 2: FIT+DMA out
    assert mt['fitdma_only'].tolist() == [False, False, False, True]


def test_representable_by_cell_size():
    assert representable(np.array([2, 4, 8, 3]), {4}).tolist() == [True, True, False, False]
    assert representable(np.array([2, 4, 8, 3]), {6, 8}).tolist() == [True, True, True, True]


def test_draw_costs_and_calls_axis():
    rows = [dict(rebuilt='True', steps='5'), dict(rebuilt='True', steps='300'), dict(rebuilt='False', steps='')]
    cost = draw_costs(rows, [0, 10, 100])
    assert cost[0].tolist() == [0, 0, 0]
    assert cost[10].tolist() == [5, 10, 0]
    assert cost[100].tolist() == [5, 100, 0]
    # one order (1, 0, 2): calls 100, 105, 105; coverage after each draw 0.5, 1.0, 1.0
    cov = np.array([[0.5, 1.0, 1.0]])
    got = at_calls(cov, cost[100], [np.array([1, 0, 2])], np.array([50, 100, 104, 105, 1000]))
    assert got.tolist() == [[0.0, 0.5, 0.5, 1.0, 1.0]]


def test_allocate_by_largest_remainder():
    from eval.day_bench.protocol_random import allocate
    got = allocate({14: 0.5, 2: 0.3, 19: 0.2}, 7)  # exact 3.5, 2.1, 1.4
    assert sum(got.values()) == 7 and got == {14: 4, 2: 2, 19: 1}
    assert allocate({1: 1.0, 2: 1.0}, 1) == {1: 1}  # tie to the lower group
    assert sum(allocate({g: 1.0 / (g + 1) for g in range(26)}, 1000).values()) == 1000


def test_csd_plan_reads_the_committed_statistics():
    from eval.day_bench.protocol_random import DAY_GROUPS, parse_plan
    plan = dict(parse_plan('csd:1000', 'CHNOF_zp1'))
    assert sum(plan.values()) == 1000 and set(plan) <= set(DAY_GROUPS)
    assert max(plan, key=plan.get) == 14  # P2_1/c dominates Z' = 1 CSD structures
