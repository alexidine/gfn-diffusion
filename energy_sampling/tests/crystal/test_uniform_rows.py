"""buffer.py::uniform_rows (the row draw that does not permute a large pool) and CrystalBuffer._bump_counts."""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from energy_sampling.buffer import CrystalBuffer, UNIFORM_ROWS_LARGE_POOL, uniform_rows

pytestmark = pytest.mark.fast


def test_a_large_pool_draw_has_k_distinct_rows_in_range():
    np.random.seed(0)
    n = UNIFORM_ROWS_LARGE_POOL * 3
    for k in (1, 17, 1000, n // 16):
        rows = uniform_rows(n, k)
        assert rows.size == k and np.unique(rows).size == k and rows.min() >= 0 and rows.max() < n


def test_an_index_array_pool_returns_only_its_rows():
    np.random.seed(1)
    pool = np.arange(UNIFORM_ROWS_LARGE_POOL) * 2 + 5
    rows = uniform_rows(pool, 500)
    assert np.unique(rows).size == 500 and np.isin(rows, pool).all()


def test_the_draw_is_uniform_over_the_pool_and_in_random_order():
    np.random.seed(2)
    n, k, draws = UNIFORM_ROWS_LARGE_POOL, 2000, 2000
    bins = 20
    hits, first = np.zeros(bins), np.zeros(bins)
    for _ in range(draws):
        rows = uniform_rows(n, k)
        hits += np.bincount(rows * bins // n, minlength=bins)
        first[rows[0] * bins // n] += 1
    expected = k * draws / bins
    assert np.abs(hits - expected).max() < 5 * np.sqrt(expected)        # a 5-sigma band per bin
    assert np.abs(first - draws / bins).max() < 5 * np.sqrt(draws / bins)   # no bias in the order either


@pytest.mark.parametrize('n,k,replace', [(1000, 10, False), (UNIFORM_ROWS_LARGE_POOL, UNIFORM_ROWS_LARGE_POOL // 2, False),
                                         (UNIFORM_ROWS_LARGE_POOL * 2, 100, True)])
def test_small_pools_dense_draws_and_replacement_keep_numpys_stream(n, k, replace):
    np.random.seed(3)
    ours = uniform_rows(n, k, replace=replace)
    np.random.seed(3)
    assert np.array_equal(ours, np.random.choice(n, size=k, replace=replace))


def test_a_large_sparse_draw_does_not_scale_with_the_pool():
    n = UNIFORM_ROWS_LARGE_POOL * 30
    start = time.perf_counter()
    for _ in range(20):
        uniform_rows(n, 1000)
    assert (time.perf_counter() - start) / 20 < 0.05    # a full permutation of 30M rows takes over a second


def test_bump_counts_counts_every_occurrence_and_touches_nothing_else():
    holder = SimpleNamespace(select_counts=torch.tensor([5, 0, 2, 0, 1], dtype=torch.long))
    CrystalBuffer._bump_counts(holder, np.array([1, 1, 4, 2, 1]))
    assert holder.select_counts.tolist() == [5, 3, 3, 0, 2]
    CrystalBuffer._bump_counts(holder, np.empty(0, dtype=np.int64))
    assert holder.select_counts.tolist() == [5, 3, 3, 0, 2]
