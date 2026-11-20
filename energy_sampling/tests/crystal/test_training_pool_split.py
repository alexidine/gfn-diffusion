"""data_processing/training_pool_split.py: the size-step scaffold split on a synthetic molecule table."""
import numpy as np
import pandas as pd
import pytest

from data_processing.training_pool_split import size_step_split

pytestmark = pytest.mark.fast


def table(seed=0):
    """600 scaffolds, sizes 4-14 heavy atoms; one scaffold of small molecules large enough to dominate a whole-pool
    draw at <= 8 (the defect the split replaces), plus some acyclic molecules."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(600):
        lo = int(rng.integers(4, 13))
        for j in range(int(rng.integers(1, 60))):
            rows.append((f'k{s}_{j}', int(min(14, lo + rng.integers(0, 3))), f'S{s}'))
    rows += [(f'big_{j}', 6, 'BIG') for j in range(400)]
    rows += [(f'acyc_{j}', int(rng.integers(4, 15)), '') for j in range(300)]
    return pd.DataFrame(rows, columns=['key', 'n_heavy', 'generic_scaffold'])


def test_arms_are_scaffold_disjoint_capped_and_filled():
    df, cap = table(), 0.02
    split, report, _ = size_step_split(df, held_frac=0.05, cap=cap, first_rung=8)
    held = set(df.generic_scaffold[split == 'heldout_scaffold'])
    rest = set(df.generic_scaffold[split != 'heldout_scaffold'])
    assert held and not held & rest
    assert '' not in held and 'BIG' not in held          # acyclic and over-cap scaffolds are never held
    for r in report:
        t = r['rung']
        m = df.n_heavy <= t
        per = df[m & (split == 'heldout_scaffold')].generic_scaffold.value_counts()
        assert (per <= cap * m.sum()).all()
        assert r['shortfall'] or r['arm'] >= 0.05 * m.sum()
    assert not any(r['shortfall'] for r in report)
    assert set(split.unique()) == {'train', 'heldout_random', 'heldout_scaffold'}


def test_deterministic_and_frozen_split_is_kept():
    df = table()
    a, _, _ = size_step_split(df, held_frac=0.05, cap=0.02)
    b, _, _ = size_step_split(df, held_frac=0.05, cap=0.02)
    assert a.equals(b)
    small = df.n_heavy <= 8
    frozen = dict(zip(df.key[small], a[small]))
    grown = pd.concat([df, table(seed=1).assign(key=lambda d: 'new_' + d.key)], ignore_index=True)
    c, _, _ = size_step_split(grown, held_frac=0.05, cap=0.02, frozen=frozen)
    assert (c[:len(df)][small.values] == a[small]).all()


def test_frozen_split_that_is_not_disjoint_raises():
    df = table()
    keys = df.key[df.generic_scaffold == 'S1'].tolist()
    if len(keys) < 2:
        pytest.skip('scaffold S1 has one molecule in this draw')
    frozen = {keys[0]: 'heldout_scaffold', keys[1]: 'train'}
    with pytest.raises(ValueError, match='not scaffold-disjoint'):
        size_step_split(df, frozen=frozen)
