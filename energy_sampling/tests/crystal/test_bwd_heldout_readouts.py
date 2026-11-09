"""train.py Modeller.bwd_heldout_readouts: backward TB residuals on held-out stored crystals against training ones,
through the shared backward eval draw (Modeller._bwd_eval_draw), on a Modeller built without __init__ around stub
rows, a stub energy function and a stub policy. What each test would catch:

  * the gap's sign or arithmetic wrong, or a key missing;
  * the two sets read on different row counts, or n not capped at the eval draw;
  * the rows redrawn between evals (the curve would carry row-sampling noise), including when the eval draw shrinks;
  * the readout present on a run with no held-out rows;
  * an OOM that never shrinks the eval draw (step 0) spinning the readout forever;
  * a non-OOM error swallowed;
  * the test molecules' crystals (eval_test/tb_rows) missing, read on the held-out/train row count, or their gap's
    sign wrong; the test set dropped on a run with no held-out rows;
  * the within-condition std mixing conditions (it must not see a per-condition log Z offset);
  * on a run over several space groups, a group's sg<g>/ mean taken over another group's rows, or the split shown on a
    single-group run (train.group_split_means).

CPU, synthetic rows, no energy computed.

    python -m pytest -q tests/crystal/test_bwd_heldout_readouts.py
"""
import math
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.train import Modeller

pytestmark = pytest.mark.fast

DIM, LOG_Z = 3, 5.0


class _Rows:
    """Stored rows: latent x and prebuilt log R."""
    device = torch.device('cpu')

    def __init__(self, x, log_r):
        self.x, self.log_r = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(log_r, dtype=torch.float32)
        self.num_graphs = len(self.log_r)
        self.ptr = torch.arange(self.num_graphs + 1)

    def subsample_new_batch(self, idx):
        out = _Rows(self.x[idx], self.log_r[idx])
        if hasattr(self, 'sg_ind'):
            out.sg_ind = self.sg_ind[idx]
        return out

    def to(self, device):
        return self


class _Ef:
    """Condition id = x[:, 2] (0 unless a test sets it)."""
    def condition_samples(self, mol_batch, temperature, **kw):
        n = int(mol_batch.num_graphs)
        return mol_batch, torch.log10(temperature), torch.zeros(n, 1), mol_batch.x[:, 2].long()

    def prebuilt_sample_to_reward(self, mol_batch, temperature):
        return mol_batch.log_r


class _Policy:
    """Backward rollout of two steps: sum log P_F = x[:, 0], sum log P_B = 0, log Z = LOG_Z on every row."""
    device = torch.device('cpu')

    def __init__(self, fail=None):
        self.fail, self.seen = fail, []

    def get_traj_bwd(self, x, discretizer, condition, mol_batch, return_gauss_params=False):
        if self.fail is not None:
            raise self.fail
        self.seen.append(x.clone())
        n = x.shape[0]
        log_pfs = torch.stack([0.5 * x[:, 0], 0.5 * x[:, 0]], 1)
        log_flow = torch.full((n, 3), LOG_Z)
        return torch.zeros(n, 3, DIM), log_pfs, torch.zeros(n, 2), log_flow, {}


def _rows(n, delta, offset):
    """Rows whose TB residual is exactly `delta`: x[:, 0] = log R - LOG_Z + delta. offset tags the rows by x[:, 1]."""
    log_r = torch.linspace(-3.0, 3.0, n)
    x = torch.zeros(n, DIM)
    x[:, 0] = log_r - LOG_Z + delta
    x[:, 1] = offset + torch.arange(n, dtype=torch.float32)
    return _Rows(x, log_r)


def _modeller(n_held=7, n_train=20, batch_size=16, fail=None, step_ind=5, n_test=0):
    m = Modeller.__new__(Modeller)
    m.args = SimpleNamespace(seed=0, energy_config=SimpleNamespace(temperature=1.0), oom_batch_shrink_factor=0.5)
    m.gfn_model = _Policy(fail)
    m.energy_function = _Ef()
    m.device = torch.device('cpu')
    m.batch_size, m.step_ind = batch_size, step_ind
    m.optimizers = {}
    m._batch_latents = lambda b: b.x
    m.prior_heldout_batch = _rows(n_held, -2.0, 1000.0) if n_held else None
    m.prior_dataset = SimpleNamespace(batch=_rows(n_train, 0.0, 0.0))
    if n_test:
        m.test_mol_dataset = SimpleNamespace(batch=_rows(n_test, -3.0, 2000.0))
    return m


def test_gap_is_train_minus_heldout_on_equal_row_counts():
    m = _modeller()
    out = m.bwd_heldout_readouts(None)
    assert set(out) == {'eval_bwd/tb_heldout_rows', 'eval_bwd/tb_heldout_rows_std',
                        'eval_bwd/tb_heldout_rows_std_within', 'eval_bwd/tb_train_rows', 'eval_bwd/tb_train_rows_std',
                        'eval_bwd/tb_train_rows_std_within', 'eval_bwd/tb_heldout_gap'}
    assert math.isclose(out['eval_bwd/tb_heldout_rows'], -2.0, abs_tol=1e-5)
    assert math.isclose(out['eval_bwd/tb_train_rows'], 0.0, abs_tol=1e-5)
    assert math.isclose(out['eval_bwd/tb_heldout_gap'], 2.0, abs_tol=1e-5)      # > 0: held-out rows fit worse
    assert out['eval_bwd/tb_heldout_rows_std'] < 1e-5
    held, train = m.gfn_model.seen
    assert held.shape[0] == train.shape[0] == 7                                  # n = min(draw 16, 7, 20)
    assert bool((held[:, 1] >= 1000).all()) and bool((train[:, 1] < 1000).all())


def test_n_is_capped_at_the_eval_draw_and_rows_stay_fixed_when_the_draw_shrinks():
    m = _modeller(batch_size=5)
    m.bwd_heldout_readouts(None)
    first = [s[:, 1] for s in m.gfn_model.seen]
    assert [len(s) for s in first] == [5, 5]
    m.gfn_model.seen.clear()
    m.eval_batch_cap = 2                                                         # an eval OOM cut the draw
    m.bwd_heldout_readouts(None)
    assert [len(s) for s in m.gfn_model.seen] == [2, 2, 1, 2, 2, 1]               # chunked at the new draw
    again = [torch.cat(m.gfn_model.seen[:3])[:, 1], torch.cat(m.gfn_model.seen[3:])[:, 1]]
    assert all(torch.equal(a, b) for a, b in zip(first, again))                  # the same rows, in the same order


def test_absent_without_heldout_rows():
    m = _modeller(n_held=0)
    assert m.bwd_heldout_readouts(None) == {}
    assert m.gfn_model.seen == []


def test_an_oom_that_never_shrinks_the_draw_is_bounded():
    m = _modeller(fail=RuntimeError('CUDA out of memory. Tried to allocate'), step_ind=0)
    assert m.bwd_heldout_readouts(None) == {}
    assert m.batch_oom_events == 2 * (Modeller.BWD_READOUT_MAX_RETRIES + 1)     # each set gave up after its bound


def test_a_non_oom_error_is_raised():
    m = _modeller(fail=ValueError('bad condition'))
    with pytest.raises(ValueError, match='bad condition'):
        m.bwd_heldout_readouts(None)


def test_test_molecules_read_on_their_own_rows_with_their_gap():
    m = _modeller(n_test=11)
    out = m.bwd_heldout_readouts(None)
    assert math.isclose(out['eval_test/tb_rows'], -3.0, abs_tol=1e-5)
    assert math.isclose(out['eval_test/tb_rows_gap'], 3.0, abs_tol=1e-5)       # > 0: held-out molecules worse
    assert math.isclose(out['eval_bwd/tb_heldout_gap'], 2.0, abs_tol=1e-5)      # the pair is unchanged by the third set
    held, train, test = m.gfn_model.seen
    assert held.shape[0] == train.shape[0] == 7                                  # the pair's n ignores the test set
    assert test.shape[0] == 11 and bool((test[:, 1] >= 2000).all())              # n = min(draw 16, 11)


def test_test_molecules_read_without_heldout_rows():
    m = _modeller(n_held=0, n_test=5)
    out = m.bwd_heldout_readouts(None)
    assert set(out) == {'eval_test/tb_rows', 'eval_test/tb_rows_std', 'eval_test/tb_rows_std_within'}
    assert len(m.gfn_model.seen) == 1


def test_std_within_ignores_a_per_condition_offset():
    """Two conditions whose residuals sit 10 nats apart (a per-condition log Z error), each with an internal spread of
    exactly 1: the pooled std sees the offset, std_within does not."""
    m = _modeller(n_held=0, n_test=8)
    b = m.test_mol_dataset.batch
    b.x[:, 2] = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    spread = torch.tensor([-1.0, 1.0, -1.0, 1.0, 9.0, 11.0, 9.0, 11.0])        # condition 1 sits 10 nats higher
    b.x[:, 0] = b.log_r - LOG_Z + spread
    out = m.bwd_heldout_readouts(None)
    within = float(torch.tensor([-1.0, 1.0, -1.0, 1.0]).std())
    assert math.isclose(out['eval_test/tb_rows_std_within'], within, abs_tol=1e-5)
    assert out['eval_test/tb_rows_std'] > 4.0


def test_group_split_means_is_per_space_group_and_only_on_multi_group_runs():
    from types import SimpleNamespace
    from energy_sampling.train import group_split_means
    v = torch.tensor([1.0, 3.0, 10.0, 20.0, float('nan')])
    sg = torch.tensor([2, 2, 14, 14, 14])
    two = SimpleNamespace(space_groups=[2, 14])
    assert group_split_means('k', v, sg, None, two) == {'sg2/k': 2.0, 'sg14/k': 15.0}
    assert group_split_means('k', v, sg, torch.tensor([True, False, True, True, True]), two) == {'sg2/k': 1.0,
                                                                                                'sg14/k': 15.0}
    assert group_split_means('k', v, sg, None, two, stat='median')['sg14/k'] == 15.0
    assert group_split_means('k', v, sg, None, SimpleNamespace(space_groups=[2])) == {}


def test_tb_readouts_split_by_space_group_on_a_multi_group_run():
    m = _modeller(n_test=6)
    m.args.space_groups = [2, 14]
    for b, sgs in ((m.prior_heldout_batch, [2] * 7), (m.prior_dataset.batch, [2] * 10 + [14] * 10),
                   (m.test_mol_dataset.batch, [14] * 6)):
        b.sg_ind = torch.tensor(sgs)
    out = m.bwd_heldout_readouts(None)
    assert math.isclose(out['sg2/eval_bwd/tb_heldout_rows'], -2.0, abs_tol=1e-5)
    assert math.isclose(out['sg14/eval_test/tb_rows'], -3.0, abs_tol=1e-5)
    assert {k for k in out if k.startswith('sg')} >= {'sg2/eval_bwd/tb_train_rows', 'sg14/eval_bwd/tb_train_rows'}
