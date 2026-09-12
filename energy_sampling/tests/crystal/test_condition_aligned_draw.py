"""The aligned per-condition draw on CrystalBuffer (train.py's stage condition_draw):
_sample_condition_aligned_indices and condition_row_counts, their route through
_sample_indices / sample_graphs / loader, and sample_graphs_at -- the draw of
caller-chosen rows the non-iid rollout condition draw uses.

The buffers are real CrystalBuffer instances made with __new__ over a stub batch
carrying only condition_id, so every method under test is the shipped one and
nothing builds a crystal. The held-out tests are the load-bearing ones: a draw
that stops masking `is_val` puts the held-out set back into training with no
error, and here that shows up as row 3 being drawn.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from energy_sampling.buffer import CrystalBuffer

# row:        0  1  2  3  4  5  6  7  8  9
CID = np.array([7, 3, 7, 9, 7, 5, 9, 3, 7, 9])
VAL = np.array([i == 3 for i in range(10)])     # one of condition 9's three rows held out


class _Batch:
    """What the draw paths read: condition_id, num_graphs, a key store, and a
    subsample_new_batch that hands back the indices it was given."""

    def __init__(self, cid):
        self.condition_id = torch.as_tensor(cid, dtype=torch.long)
        self._store = {}

    @property
    def num_graphs(self):
        return int(self.condition_id.numel())

    def subsample_new_batch(self, inds):
        return SimpleNamespace(rows=np.asarray(inds), _store={})


def _buf(cid=CID, is_val=None, traj=False):
    b = CrystalBuffer.__new__(CrystalBuffer)
    b.device = torch.device('cpu')
    b.batch = _Batch(cid)
    n = len(cid)
    b.is_val = (torch.zeros(n, dtype=torch.bool) if is_val is None
                else torch.as_tensor(is_val, dtype=torch.bool))
    b.select_counts = torch.zeros(n, dtype=torch.long)
    b.traj = torch.arange(n, dtype=torch.float32)[:, None] if traj else None
    return b


def _aligned(buf, cids, rows, repeats=1):
    return buf._sample_indices(len(cids) * rows, repeats=repeats,
                               draw_cids=np.asarray(cids), rows_per_condition=rows)


# ------------------------------------------------------------------ the draw

def test_exactly_rows_per_distinct_rows_for_each_condition_in_order():
    buf = _buf()
    for _ in range(50):
        inds = _aligned(buf, [7, 3, 9], 2)
        assert list(CID[inds]) == [7, 7, 3, 3, 9, 9]
        for g in range(3):
            assert inds[2 * g] != inds[2 * g + 1], 'rows within a condition are distinct'


def test_held_out_rows_are_never_drawn():
    """Condition 9 has three rows and one is held out: with rows_per 2 the draw
    must return exactly the two trainable ones, every time."""
    buf = _buf(is_val=VAL)
    for _ in range(100):
        inds = _aligned(buf, [9, 3], 2)
        assert 3 not in set(inds.tolist())
        assert set(inds[:2].tolist()) == {6, 9}


def test_a_condition_short_only_by_its_held_out_rows_raises():
    _aligned(_buf(), [9], 3)                             # three rows, none held out
    with pytest.raises(ValueError, match='fewer than 3'):
        _aligned(_buf(is_val=VAL), [9], 3)


def test_a_short_or_absent_condition_raises():
    """No top-up: a condition that cannot fill its group is the caller's
    eligibility bug, not something to paper over with unaligned rows."""
    with pytest.raises(ValueError, match='fewer than 2'):
        _aligned(_buf(), [7, 5], 2)
    with pytest.raises(ValueError, match='fewer than 2'):
        _aligned(_buf(), [42], 2)


def test_repeats_tile_each_row_terminal_major():
    buf = _buf()
    np.random.seed(0)
    base = _aligned(buf, [7, 3], 2)
    np.random.seed(0)
    tiled = _aligned(buf, [7, 3], 2, repeats=3)
    assert np.array_equal(tiled, np.repeat(base, 3))


def test_row_choice_within_a_condition_is_uniform():
    buf = _buf()
    hits = np.zeros(10)
    n = 4000
    for _ in range(n):
        hits[_aligned(buf, [7], 2)] += 1
    freq = hits[[0, 2, 4, 8]] / n                          # condition 7's four rows
    assert np.all(np.abs(freq - 0.5) < 0.05), freq


def test_repeated_conditions_are_refused():
    with pytest.raises(ValueError, match='repeat'):
        _buf()._sample_indices(4, draw_cids=np.array([7, 7]), rows_per_condition=2)


@pytest.mark.parametrize('kw', [
    {'p': np.full(10, 0.1)}, {'condition_block_m': 2}, {'target_cids': np.array([7])}])
def test_the_aligned_draw_takes_no_other_measure(kw):
    with pytest.raises(ValueError, match='takes no'):
        _buf()._sample_indices(4, draw_cids=np.array([7, 3]), rows_per_condition=2, **kw)


def test_a_size_that_is_not_the_product_is_refused():
    with pytest.raises(ValueError, match='batch_size'):
        _buf()._sample_indices(5, draw_cids=np.array([7, 3]), rows_per_condition=2)
    with pytest.raises(ValueError, match='rows_per_condition'):
        _buf()._sample_indices(2, draw_cids=np.array([7, 3]), rows_per_condition=0)


def test_a_batch_without_condition_id_is_refused():
    buf = _buf()
    del buf.batch.condition_id
    with pytest.raises(ValueError, match='condition_id'):
        buf.condition_row_counts()


# ------------------------------------------------------------------ counts

def test_condition_row_counts_counts_trainable_rows_only():
    counts = _buf(is_val=VAL).condition_row_counts(12)
    assert counts.size == 12
    assert counts[9] == 2 and counts[7] == 4 and counts[3] == 2 and counts[5] == 1
    assert _buf().condition_row_counts(12)[9] == 3


# ------------------------------------------------------------ loader / graphs

def test_the_loader_routes_the_draw_and_bumps_counts():
    buf = _buf(traj=True)
    graphs, traj, inds = next(buf.loader(4, mode='graphs', repeats=2, return_inds=True,
                                         return_traj=True, draw_cids=np.array([7, 3]),
                                         rows_per_condition=2))
    assert inds.size == 8 and np.array_equal(graphs.rows, inds)
    assert list(CID[inds]) == [7, 7, 7, 7, 3, 3, 3, 3]
    assert torch.equal(traj[:, 0], torch.as_tensor(inds, dtype=torch.float32))
    assert int(buf.select_counts.sum()) == 8
    assert set(buf.select_counts[np.unique(inds)].tolist()) == {2}


def test_the_tensors_mode_refuses_the_aligned_draw():
    with pytest.raises(ValueError, match='graphs-mode only'):
        next(_buf().loader(4, mode='tensors', draw_cids=np.array([7, 3]),
                           rows_per_condition=2))


def test_default_arguments_draw_exactly_as_before():
    """The new keywords default off, and then the uniform path is the draw it
    always was: without replacement over the trainable pool."""
    buf = _buf(is_val=VAL)
    np.random.seed(1)
    got = buf._sample_indices(5)
    np.random.seed(1)
    want = np.random.choice(np.flatnonzero(~VAL), size=5, replace=False)
    assert np.array_equal(got, want)


def test_sample_graphs_at_tiles_the_given_rows_and_bumps_counts():
    buf = _buf()
    g = buf.sample_graphs_at(np.array([2, 5]), repeats=3)
    assert list(g.rows) == [2, 2, 2, 5, 5, 5]
    assert buf.select_counts[2] == 3 and buf.select_counts[5] == 3
    assert int(buf.select_counts.sum()) == 6
