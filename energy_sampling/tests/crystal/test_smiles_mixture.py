"""CPU tests for data_processing/smiles_mixture.py (the SMILES list a random-conformer crystal search runs on).

WHAT THIS FILE IS FOR. A mixture that let a QM9 molecule through would put held-out evaluation molecules into training,
and one that drew a molecule twice or let a large pool crowd out a small one would silently change the distribution the
owner chose. Covered: QM9 exclusion, draw-once across pools in the given order with every pool named in `sources`,
exact quotas and 'all', seeded reproducibility, and the written list (columns, shuffle, refusal to overwrite).

    python -m pytest -q tests/crystal/test_smiles_mixture.py
"""
import json
import os
import sys

import pytest

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_processing import smiles_mixture as sm  # noqa: E402

HEADER = ['smiles', 'heavy_atoms', 'n_direct', 'n_parents', 'min_parent_heavy']


def _pool(tmp_path, source, smiles):
    d = tmp_path / 'pool' / source
    d.mkdir(parents=True)
    with open(d / 'pool.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(HEADER) + '\n')
        for s in smiles:
            f.write(f'{s}\t5\t1\t0\t0\n')


def test_mix_excludes_draws_once_and_respects_order_and_quotas():
    a = [(f'A{i}', 'x') for i in range(10)] + [('S0', 'x'), ('Q0', 'x')]
    b = [(f'B{i}', 'x') for i in range(100)] + [('S0', 'x'), ('Q1', 'x')]
    rows, counts = sm.mix({'a': a, 'b': b}, [('a', None), ('b', 30)], exclude={'Q0', 'Q1'}, seed=0)
    smiles = [r[0] for r in rows]
    assert len(smiles) == len(set(smiles)) == 11 + 30
    assert not {'Q0', 'Q1'} & set(smiles)
    s0 = [r for r in rows if r[0] == 'S0']
    assert len(s0) == 1 and s0[0][-2:] == ('a', 'a,b')                 # the earlier pool takes it; both are named
    assert counts == {'a': dict(pool=12, eligible=11, drawn=11), 'b': dict(pool=102, eligible=100, drawn=30)}
    again, _ = sm.mix({'a': a, 'b': b}, [('a', None), ('b', 30)], exclude={'Q0', 'Q1'}, seed=0)
    other, _ = sm.mix({'a': a, 'b': b}, [('a', None), ('b', 30)], exclude={'Q0', 'Q1'}, seed=1)
    assert again == rows and other != rows
    over, c2 = sm.mix({'a': a}, [('a', 1000)], exclude=set(), seed=0)
    assert len(over) == 12 and c2['a']['drawn'] == 12                 # a quota above the pool takes all of it


def test_split_is_fixed_per_smiles_and_about_the_set_fraction():
    smiles = [f'C{"C" * (i % 7)}O{i}' for i in range(20000)]
    sides = [sm.split_of(x, 'rc_oct22', 50) for x in smiles]
    assert 0.04 < sides.count('held_out') / len(sides) < 0.06
    assert [sm.split_of(x, 'rc_oct22', 50) for x in smiles[:500]] == sides[:500]   # a molecule's side ignores the set
    assert [sm.split_of(x, 'other', 50) for x in smiles] != sides                  # and follows the salt


def test_main_writes_a_shuffled_list_and_refuses_to_overwrite(tmp_path, monkeypatch):
    _pool(tmp_path, 'small', ['CCO', 'CCN', 'CCC'])
    _pool(tmp_path, 'big', [f'C{"C" * i}O' for i in range(2, 40)] + ['CCO'])
    monkeypatch.setattr(sm, 'qm9_reference', lambda *a: {'CCC'})
    out = tmp_path / 'sel'
    argv = ['--pool', str(tmp_path / 'pool'), '--quota', 'small=all', 'big=10', '--out', str(out)]
    sm.main(argv)
    lines = (out / 'selected.tsv').read_text(encoding='utf-8').splitlines()
    assert lines[0].split('\t') == HEADER + ['source', 'sources', 'split']
    rows = [line.split('\t') for line in lines[1:]]
    assert len(rows) == 12 and all(len(r) == len(HEADER) + 3 for r in rows)
    assert 'CCC' not in {r[0] for r in rows}
    assert [r[-3] for r in rows][:2] != ['small', 'small']             # shuffled: not in draw order
    assert all(r[-1] == sm.split_of(r[0], 'rc_oct22', 50) for r in rows)
    meta = json.loads((out / 'selection.json').read_text())
    assert meta['total'] == 12 and meta['counts']['small']['drawn'] == 2
    for src, n in (('small', 2), ('big', 10)):                      # each source's rows in a file of its own
        part = (out / 'by_source' / f'{src}.tsv').read_text(encoding='utf-8').splitlines()
        assert part[0] == lines[0] and sorted(part[1:]) == sorted(ln for ln in lines[1:] if ln.split('\t')[-3] == src)
        assert len(part) == n + 1
    with pytest.raises(SystemExit):
        sm.main(argv)
    out2 = tmp_path / 'sel2'                                           # a later source's quota changes; small.tsv not
    sm.main(['--pool', str(tmp_path / 'pool'), '--quota', 'small=all', 'big=20', '--out', str(out2)])
    assert (out2 / 'by_source' / 'small.tsv').read_bytes() == (out / 'by_source' / 'small.tsv').read_bytes()
    assert (out2 / 'selected.tsv').read_bytes() != (out / 'selected.tsv').read_bytes()
    out3 = tmp_path / 'sel3'                                           # a source added after it: same SMILES, same order
    sm.main(['--pool', str(tmp_path / 'pool'), '--quota', 'small=all', '--out', str(out3)])
    col = [ln.split('\t')[0] for ln in (out3 / 'by_source' / 'small.tsv').read_text(encoding='utf-8').splitlines()]
    assert col == [ln.split('\t')[0] for ln in (out / 'by_source' / 'small.tsv').read_text(encoding='utf-8').splitlines()]
