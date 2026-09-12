"""The forward rollout's condition draw (condition_log_z.rollout_condition_draw):
Modeller._rollout_mol_batch and the two non-iid row draws behind it, `cycle`
(_cycle_rows) and `under_drawn` (_under_drawn_probs).

A rollout draws batch_size mol_dataset rows and tiles each `repeats` times, so a
rollout consumes batch_size conditions (one row = one condition on a single
SG/Z' route). Stub Modellers with the real methods bound; the mol_dataset stub
records which draw it was asked for.
"""
import ast
import inspect
import textwrap
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from energy_sampling.buffer import CrystalBuffer
from energy_sampling.train import Modeller


class _Mol:
    def __init__(self, n):
        self.n, self.calls = n, []
        self.batch = SimpleNamespace(mol_id=torch.arange(n))

    def __len__(self):
        return self.n

    def loader(self, *args, **kw):
        self.calls.append(('loader', args, kw))
        return iter(['iid_batch'])

    def sample_graphs_at(self, rows, repeats=1):
        self.calls.append(('at', np.asarray(rows), repeats))
        return 'at_batch'


class _Counts:
    def __init__(self, counts):
        self.counts = np.asarray(counts, dtype=np.int64)

    def __len__(self):
        return int(self.counts.sum()) or 1

    def condition_row_counts(self, minlength=0):
        out = np.zeros(max(int(minlength), self.counts.size), dtype=np.int64)
        out[:self.counts.size] = self.counts
        return out


P_WEIGHTED = np.linspace(1.0, 2.0, 10) / np.linspace(1.0, 2.0, 10).sum()


def _roller(n_rows=10, mode='cycle', power=1.0, rep=None, n_sg=1, batch_size=4,
            flag=False, global_on=True, drop_mode=False):
    cfg = SimpleNamespace(rollout_condition_draw=mode, rollout_under_drawn_power=power,
                          weighted_condition_sampling=global_on,
                          weighted_condition_sampling_temperature=0.3,
                          weighted_condition_sampling_clip_quantile=0.99,
                          weighted_condition_sampling_uniform_beta=0.5)
    if drop_mode:
        del cfg.rollout_condition_draw
    m = SimpleNamespace(args=SimpleNamespace(condition_log_z=cfg), mol_dataset=_Mol(n_rows),
                        batch_size=batch_size,
                        energy_function=SimpleNamespace(n_sg=n_sg, n_zp=1),
                        protocol=SimpleNamespace(flag=lambda name: flag,
                                                 stage=SimpleNamespace(name='s')),
                        weighted_condition_sampling=lambda **kw: P_WEIGHTED)
    if rep is not None:
        m.replay_buffer = rep
    for name in ('_rollout_condition_mode', '_rollout_mol_batch', '_rollout_condition_rows',
                 '_cycle_rows', '_under_drawn_probs'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


# ------------------------------------------------------------------ iid

def test_iid_is_today_s_loader_call():
    m = _roller(mode='iid')
    assert m._rollout_mol_batch(2) == 'iid_batch'
    (call,) = m.mol_dataset.calls
    assert call == ('loader', (4,), {'mode': 'graphs', 'repeats': 2, 'p': None, 'beta': None})


def test_iid_on_a_weighted_stage_hands_the_loader_the_measure():
    m = _roller(mode='iid', flag=True)
    m._rollout_mol_batch(2)
    (_, _, kw) = m.mol_dataset.calls[0]
    assert kw['p'] is P_WEIGHTED and kw['beta'] == 0.5


def test_an_absent_key_reads_as_iid():
    m = _roller(drop_mode=True)
    m._rollout_mol_batch(1)
    assert m.mol_dataset.calls[0][0] == 'loader'


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match='rollout_condition_draw'):
        _roller(mode='stratified')._rollout_mol_batch(1)


def test_a_non_iid_draw_on_a_weighted_stage_is_refused():
    with pytest.raises(ValueError, match='weighted_condition_sampling'):
        _roller(mode='cycle', flag=True)._rollout_mol_batch(1)
    # the flag is inert while the global key is off, and then nothing conflicts
    _roller(mode='cycle', flag=True, global_on=False)._rollout_mol_batch(1)


def test_fwd_train_step_draws_through_the_helper():
    src = textwrap.dedent(inspect.getsource(Modeller.fwd_train_step))
    calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)}
    assert 'self._rollout_mol_batch' in calls
    assert not any('mol_dataset.loader' in c for c in calls)


# ------------------------------------------------------------------ cycle

def test_cycle_hands_batch_size_rows_to_sample_graphs_at():
    m = _roller(mode='cycle', batch_size=4)
    assert m._rollout_mol_batch(2) == 'at_batch'
    (kind, rows, repeats) = m.mol_dataset.calls[0]
    assert kind == 'at' and rows.size == 4 and repeats == 2


def test_cycle_draws_every_row_once_per_pass_and_never_twice_in_a_batch():
    np.random.seed(0)
    m = _roller(mode='cycle', n_rows=10, batch_size=7)
    stream = []
    for _ in range(40):                                     # 28 passes, 20+ boundaries
        rows = m._rollout_condition_rows(7, 'cycle')
        assert len(set(rows.tolist())) == 7, rows
        stream.extend(rows.tolist())
    for k in range(len(stream) // 10):
        assert sorted(stream[10 * k:10 * k + 10]) == list(range(10)), k


def test_cycle_serves_a_batch_larger_than_the_dataset():
    m = _roller(mode='cycle', n_rows=10)
    rows = m._rollout_condition_rows(25, 'cycle')
    assert rows.size == 25
    assert sorted(rows[:10].tolist()) == list(range(10))
    assert sorted(rows[10:20].tolist()) == list(range(10))


def test_cycle_restarts_on_a_resized_dataset():
    m = _roller(mode='cycle', n_rows=10)
    m._rollout_condition_rows(3, 'cycle')
    rows = m._cycle_rows(6, 6)
    assert sorted(rows.tolist()) == list(range(6))


# ------------------------------------------------------------------ under_drawn

def test_under_drawn_weights_fall_with_the_row_count():
    m = _roller(mode='under_drawn', n_rows=5, rep=_Counts([0, 1, 3, 7, 0]))
    p = m._under_drawn_probs(1.0)
    w = 1.0 / (1.0 + np.array([0, 1, 3, 7, 0]))
    assert np.allclose(p, w / w.sum())
    assert p[0] > p[1] > p[2] > p[3] and p[0] == p[4]


def test_the_power_is_the_exponent():
    m = _roller(mode='under_drawn', n_rows=2, rep=_Counts([0, 1]))
    p = m._under_drawn_probs(2.0)
    assert p[0] / p[1] == pytest.approx(4.0)


def test_under_drawn_sums_a_molecule_s_whole_space_group_block():
    """condition_id = mol_id * n_sg*n_zp + local: a row's count is its block's."""
    m = _roller(mode='under_drawn', n_rows=3, n_sg=2, rep=_Counts([1, 2, 0, 0, 3, 3]))
    p = m._under_drawn_probs(1.0)
    w = 1.0 / (1.0 + np.array([3, 0, 6]))
    assert np.allclose(p, w / w.sum())


def test_under_drawn_without_a_replay_buffer_is_uniform():
    p = _roller(mode='under_drawn', n_rows=4)._under_drawn_probs(1.0)
    assert np.allclose(p, 0.25)


def test_under_drawn_counts_trainable_replay_rows_only():
    buf = CrystalBuffer.__new__(CrystalBuffer)
    buf.batch = SimpleNamespace(condition_id=torch.tensor([0, 0, 1, 1]), num_graphs=4)
    buf.is_val = torch.tensor([True, True, False, False])
    m = _roller(mode='under_drawn', n_rows=2, rep=buf)
    p = m._under_drawn_probs(1.0)
    assert p[0] / p[1] == pytest.approx(3.0), 'row 0 holds 0 trainable rows, row 1 holds 2'


def test_under_drawn_draws_without_replacement():
    m = _roller(mode='under_drawn', n_rows=10, rep=_Counts([5] * 5 + [0] * 5))
    for _ in range(100):
        rows = m._rollout_condition_rows(8, 'under_drawn')
        assert len(set(rows.tolist())) == 8
    assert m._rollout_condition_rows(15, 'under_drawn').size == 15   # batch > dataset
