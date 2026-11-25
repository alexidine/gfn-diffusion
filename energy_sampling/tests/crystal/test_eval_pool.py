"""The eval's trailing pool of on-policy rollout rows (crystal_store.RolloutPool; train.py Modeller._eval_pool_add,
._eval_forward_rows, ._pool_eval_stats, ._fwd_eval_summary), on a Modeller built without __init__. CPU, no energy computed.

    python -m pytest -q tests/crystal/test_eval_pool.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling import crystal_store as CS
from energy_sampling.energies.molecular_crystal import merge_batches
from energy_sampling.models.nf_policy import NFPolicy
from energy_sampling.tests.crystal import store_stub as S
from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast


def crystals(n, first=0):
    t = S.templates()
    b = t.subsample_new_batch(torch.arange(n) % t.num_graphs)
    b.add_graph_attr(torch.arange(first, first + n, dtype=torch.long), 'row_id')
    b.add_graph_attr(torch.full((n,), 5.0), 'mol_energy')
    return b


def rollout_rows(n, first=0):
    v = torch.arange(first, first + n, dtype=torch.float32)
    x = v[:, None].repeat(1, S.DIM) / 1000
    return dict(log_r=-v, log_pf=-2 * v, log_pb=torch.zeros(n), log_Z=v / 4,
                condition_id=torch.arange(n) % 4, log_T_tensor=torch.full((n,), math.log10(S.T)),
                flow_states=torch.stack([torch.zeros_like(x), x], 1))


def pool_of(n, chunks, size=5):
    pool = CS.RolloutPool(n)
    for c in range(chunks):
        pool.add(rollout_rows(size, c * size), crystals(size, c * size), step=10 + c)
    return pool


def modeller(n=8, active=True):
    m = S.build()
    m.args.eval_num_samples = n
    m.args.fwd_loss_coeffs = SimpleNamespace(exploration_std=0.0)
    m.protocol.flag = lambda name: active if name == 'buffers_active' else False
    m.gfn_model = SimpleNamespace(GAUSS_PARAM_KEYS=NFPolicy.GAUSS_PARAM_KEYS)
    m.times = {}
    m._eval_pool = CS.RolloutPool(n)
    m.step_ind = 40
    m.fresh_calls = []

    def fresh(model, disc, override_num_samples=None, **kw):
        k = n if override_num_samples is None else override_num_samples
        m.fresh_calls.append(k)
        r = rollout_rows(k, 1000)
        lf = torch.stack([r['log_Z'], torch.zeros(k)], 1)
        stats = dict(flow_states=r['flow_states'], log_r=r['log_r'], log_pf=r['log_pf'], log_pb=r['log_pb'],
                     log_pfs=r['log_pf'][:, None], log_pbs=r['log_pb'][:, None], log_flow=lf,
                     log_T_tensor=r['log_T_tensor'], condition_id=r['condition_id'],
                     temperature=torch.full((k,), S.T))
        stats.update({g: torch.zeros(k, 1) for g in NFPolicy.GAUSS_PARAM_KEYS})
        m._fwd_eval_summary(stats)
        return stats, crystals(k, 1000)

    m.fwd_eval_sampling = fresh
    return m


def test_the_pool_reads_exactly_the_newest_n_rows_row_for_row():
    pool = pool_of(12, 4)                                # 20 rows in chunks of 5
    assert len(pool._chunks) == 3 and pool.rows == 15    # dropped once the chunks after it hold 12
    rows, batch, oldest = pool.read(merge_batches)
    assert rows['log_r'].tolist() == [-float(v) for v in range(8, 20)]
    assert batch.row_id.tolist() == list(range(8, 20)) and batch.num_graphs == 12
    assert rows['flow_states'].shape == (12, 2, S.DIM) and oldest == 11
    assert pool.rows == 12


def test_the_pool_holds_under_n_plus_one_chunk():
    pool = CS.RolloutPool(7)
    for c in range(50):
        pool.add(rollout_rows(3, 3 * c), crystals(3, 3 * c), step=c)
        assert pool.rows < 7 + 3
    rows, batch, _ = pool.read(merge_batches)
    assert batch.row_id.tolist() == list(range(143, 150))


def test_a_chunk_larger_than_n_is_trimmed_to_its_newest_rows():
    pool = CS.RolloutPool(4)
    pool.add(rollout_rows(10), crystals(10), step=3)
    rows, batch, oldest = pool.read(merge_batches)
    assert batch.row_id.tolist() == [6, 7, 8, 9] and rows['log_r'].tolist() == [-6.0, -7.0, -8.0, -9.0] and oldest == 3


def test_the_batch_read_is_a_copy():
    pool = pool_of(5, 1)
    _, batch, _ = pool.read(merge_batches)
    batch.row_id[:] = -1
    _, again, _ = pool.read(merge_batches)
    assert again.row_id.tolist() == [0, 1, 2, 3, 4]


def test_a_short_or_cleared_pool_reads_what_it_holds():
    pool = pool_of(50, 2)
    rows, batch, oldest = pool.read(merge_batches)
    assert rows['log_r'].numel() == 10 and batch.num_graphs == 10 and oldest == 10
    pool.clear()
    assert pool.rows == 0 and pool.read(merge_batches) == (None, None, None)


def test_a_chunk_whose_columns_disagree_with_its_batch_raises():
    pool = CS.RolloutPool(4)
    with pytest.raises(CS.CrystalStoreError, match='crystals'):
        pool.add(rollout_rows(3), crystals(2), step=0)


def test_only_a_training_rollout_enters_the_pool():
    m = modeller(n=8)
    fwd = rollout_rows(4)
    m._store_admit(fwd, crystals(4), CS.ORIGIN_EVAL, eval_row=True, drawable=True)
    m._store_admit(fwd, crystals(4), CS.ORIGIN_BOOTSTRAP, eval_row=True, drawable=False)
    assert m._eval_pool.rows == 0
    m._store_admit(fwd, crystals(4), CS.ORIGIN_ROLLOUT)
    assert m._eval_pool.rows == 4


def test_a_rollout_outside_a_buffers_active_stage_stays_out():
    m = modeller(n=8, active=False)
    m._store_admit(rollout_rows(4), crystals(4), CS.ORIGIN_ROLLOUT)
    assert m._eval_pool.rows == 0


def test_a_full_pool_serves_the_eval_with_no_draw_and_no_admission():
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(5), crystals(5))
    m.step_ind = 45
    m._eval_pool_add(rollout_rows(5, 5), crystals(5, 5))
    m.step_ind = 50
    before = m.row_store.n_rows
    stats, batch, fresh, read = m._eval_forward_rows(None)
    assert m.fresh_calls == [] and fresh is None
    assert read == {'eval/pool_rows': 8, 'eval/pool_fresh_rows': 0, 'eval/pool_oldest_age_steps': 10}
    assert batch.row_id.tolist() == list(range(2, 10)) and stats['log_r'].tolist() == [-float(v) for v in range(2, 10)]
    assert m.row_store.n_rows == before


def test_pool_stats_carry_every_key_of_a_fresh_eval_at_its_shape():
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(8), crystals(8))
    pooled, _, _, _ = m._eval_forward_rows(None)
    drawn, _ = m.fwd_eval_sampling(None, None)
    assert set(pooled) == set(drawn)
    for k, v in drawn.items():
        assert pooled[k].shape == v.shape and pooled[k].dtype == v.dtype, k
    assert torch.equal(pooled['log_Z_learned'], rollout_rows(8)['log_Z'])
    assert bool((pooled['log_flow'][:, 1] == 0).all()) and bool((pooled['means_f'] == 0).all())


def test_the_summary_is_the_same_function_of_the_rows_on_both_routes():
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(8), crystals(8))
    pooled, _, _, _ = m._eval_forward_rows(None)
    w = pooled['log_r'] - pooled['log_pfs'].sum(-1)
    assert float(pooled['log_Z_lb']) == pytest.approx(float(w.mean()))
    assert float(pooled['log_Z']) == pytest.approx(float(torch.logsumexp(w, 0) - math.log(8)))
    assert float(pooled['nonfinite_rows']) == 0


def test_a_short_pool_tops_up_only_its_shortfall_and_hands_back_only_those_rows():
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(5), crystals(5))
    stats, batch, fresh, read = m._eval_forward_rows(None)
    assert m.fresh_calls == [3]
    assert read['eval/pool_rows'] == 5 and read['eval/pool_fresh_rows'] == 3
    assert batch.row_id.tolist() == [0, 1, 2, 3, 4, 1000, 1001, 1002]
    assert stats['log_r'].numel() == 8 and stats['flow_states'].shape == (8, 2, S.DIM)
    assert fresh[0]['log_r'].numel() == 3 and fresh[1].row_id.tolist() == [1000, 1001, 1002]


def test_an_empty_pool_draws_the_whole_eval():
    m = modeller(n=8)
    stats, batch, fresh, read = m._eval_forward_rows(None)
    assert m.fresh_calls == [8] and fresh[0] is stats
    assert read['eval/pool_rows'] == 0 and read['eval/pool_fresh_rows'] == 8
    assert math.isnan(read['eval/pool_oldest_age_steps'])


def test_an_mle_stage_draws_fresh():
    m = modeller(n=8, active=False)
    m._eval_pool.add(rollout_rows(8), crystals(8), step=1)   # rows that must not be read in this stage
    stats, _, fresh, read = m._eval_forward_rows(None)
    assert m.fresh_calls == [8] and fresh[0] is stats and read['eval/pool_rows'] == 0


@pytest.mark.parametrize('arm, word', [
    (lambda m: setattr(m.args.fwd_loss_coeffs, 'exploration_std', 0.1), 'exploration_std'),
    (lambda m: setattr(m.gfn_model, '_framed', True), 'frames'),
    (lambda m: setattr(m, 'drift_force', object()), 'drift_force'),
    (lambda m: (setattr(m.args, 'z_calibration', SimpleNamespace(fill_from_eval='fill')),
                setattr(m, '_z_fill_head_is_fillable', lambda: True)), 'fill_from_eval'),
    (lambda m: (setattr(m.args, 'condition_log_z', SimpleNamespace(weighted_condition_sampling=True)),
                setattr(m.protocol, 'flag', lambda name: name in ('buffers_active', 'weighted_condition_sampling'))),
     'weighted_condition_sampling'),
])
def test_a_rollout_that_is_not_a_sample_of_the_eval_target_falls_back_and_says_so_once(arm, word, capsys):
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(8), crystals(8))
    arm(m)
    for _ in range(2):
        stats, _, fresh, read = m._eval_forward_rows(None)
        assert fresh[0] is stats and read['eval/pool_rows'] == 0
    assert m.fresh_calls == [8, 8]
    out = capsys.readouterr().out
    assert out.count('drawn fresh') == 1 and word in out


@pytest.mark.parametrize('stage_flag, config_key', [(True, False), (False, True)])
def test_a_weighted_condition_draw_refuses_the_pool_only_with_both_switches_on(stage_flag, config_key):
    m = modeller(n=8)
    m.args.condition_log_z = SimpleNamespace(weighted_condition_sampling=config_key)
    on = {'buffers_active'} | ({'weighted_condition_sampling'} if stage_flag else set())
    m.protocol.flag = lambda name: name in on
    m._eval_pool_add(rollout_rows(8), crystals(8))
    assert m._eval_pool_refusal() is None and m._eval_forward_rows(None)[3]['eval/pool_rows'] == 8


def test_an_eval_fill_on_a_head_it_cannot_fill_keeps_the_pool():
    m = modeller(n=8)
    m.args.z_calibration = SimpleNamespace(fill_from_eval='fill')
    m._z_fill_head_is_fillable = lambda: False
    m._eval_pool_add(rollout_rows(8), crystals(8))
    assert m._eval_forward_rows(None)[3]['eval/pool_rows'] == 8


def test_a_rewind_clears_the_pool():
    m = modeller(n=8)
    m._eval_pool_add(rollout_rows(8), crystals(8))
    Modeller._store_rewind(m, 3)
    assert m._eval_pool.rows == 0


def test_store_off_has_no_pool_and_draws_as_before():
    m = modeller(n=8)
    del m._eval_pool
    stats, batch, fresh, read = m._eval_forward_rows(None)
    assert m.fresh_calls == [8] and read == {} and fresh == (stats, batch)
