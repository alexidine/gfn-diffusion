"""The aligned per-condition draw in the trainer (stage.condition_draw):
Modeller._choose_draw_conditions, its wiring through fused_train_step, and the two
branch draws it hands the decision to (draw_bwd_sample, draw_replay_sample).

Stub Modellers with the real methods bound. The buffers are stubs that either
report per-condition row counts or record the loader call they receive; the
buffer-side draw itself is tests/crystal/test_condition_aligned_draw.py.
"""
import ast
import inspect
import textwrap
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from energy_sampling.train import Modeller

SPEC = {'conditions': 0, 'replay_rows': 2, 'prior_rows': 2, 'pick': 'uniform'}


class _Counts:
    """A buffer answering condition_row_counts from a fixed per-condition array."""

    def __init__(self, counts):
        self.counts = np.asarray(counts, dtype=np.int64)

    def __len__(self):
        return int(self.counts.sum())

    def condition_row_counts(self, minlength=0):
        out = np.zeros(max(int(minlength), self.counts.size), dtype=np.int64)
        out[:self.counts.size] = self.counts
        return out


def _chooser(rep, pri, batch_size=100, n_sg=1, p_row=None, beta=0.0):
    m = SimpleNamespace(
        energy_function=SimpleNamespace(n_sg=n_sg, n_zp=1, condition_library_size=len(rep)),
        replay_buffer=_Counts(rep), prior_buffer=_Counts(pri), batch_size=batch_size,
        args=SimpleNamespace(condition_log_z=SimpleNamespace(
            weighted_condition_sampling_temperature=0.3,
            weighted_condition_sampling_clip_quantile=0.99,
            weighted_condition_sampling_uniform_beta=beta)),
        mol_dataset=SimpleNamespace(batch=SimpleNamespace(mol_id=torch.arange(len(rep)))),
        weighted_condition_sampling=lambda **kw: p_row)
    m._choose_draw_conditions = MethodType(Modeller._choose_draw_conditions, m)
    m._weighted_draw_conditions = MethodType(Modeller._weighted_draw_conditions, m)
    return m


# ------------------------------------------------------------ the choice

def test_only_conditions_with_enough_rows_in_both_buffers_are_eligible():
    #        cond  0  1  2  3  4  5
    m = _chooser([2, 1, 3, 0, 2, 5], [2, 2, 2, 2, 1, 2])
    chosen = m._choose_draw_conditions(SPEC)
    assert sorted(chosen.tolist()) == [0, 2, 5]
    s = m._cond_draw_stats
    assert s['cond_draw/n_conditions'] == 3.0
    assert s['cond_draw/eligible_replay'] == 4.0          # 0, 2, 4, 5
    assert s['cond_draw/eligible_both'] == 3.0
    assert s['cond_draw/target_conditions'] == 50.0      # 100 // 2
    assert s['cond_draw/short_frac'] == pytest.approx(1 - 3 / 50)


def test_conditions_zero_derives_c_from_the_live_batch():
    m = _chooser([4] * 100, [4] * 100, batch_size=20)
    assert m._choose_draw_conditions(SPEC).size == 10
    spec = dict(SPEC, prior_rows=4)                        # the larger of X, Y sets C
    chosen = m._choose_draw_conditions(spec)
    assert chosen.size == 5 and len(set(chosen.tolist())) == 5
    assert m._cond_draw_stats['cond_draw/short_frac'] == 0.0


def test_a_fixed_c_is_capped_by_the_live_batch():
    """The OOM handler only cuts batch_size; a C it could not reach would repeat
    the OOM until the run dies."""
    m = _chooser([2] * 100, [2] * 100, batch_size=20)
    assert m._choose_draw_conditions(dict(SPEC, conditions=30)).size == 10
    assert m._choose_draw_conditions(dict(SPEC, conditions=5)).size == 5


def test_no_eligible_condition_is_an_empty_choice_and_a_counted_skip():
    m = _chooser([0] * 6, [2] * 6)
    chosen = m._choose_draw_conditions(SPEC)
    assert chosen.size == 0 and m._cond_draw_skips == 1
    assert m._cond_draw_stats['cond_draw/short_frac'] == 1.0
    del m.replay_buffer                                     # replay not built yet
    assert m._choose_draw_conditions(SPEC).size == 0 and m._cond_draw_skips == 2


def test_more_than_one_space_group_combination_is_refused():
    with pytest.raises(ValueError, match='space-group'):
        _chooser([2] * 6, [2] * 6, n_sg=2)._choose_draw_conditions(SPEC)


def test_the_uniform_pick_is_without_replacement_and_even():
    m = _chooser([2] * 10, [2] * 10, batch_size=10)        # C = 5 of 10
    hits = np.zeros(10)
    for _ in range(2000):
        c = m._choose_draw_conditions(SPEC)
        assert len(set(c.tolist())) == 5
        hits[c] += 1
    assert np.all(np.abs(hits / 2000 - 0.5) < 0.06), hits / 2000


def test_the_weighted_pick_follows_the_measure():
    p = np.full(10, 0.01)
    p[3] = 0.91
    m = _chooser([2] * 10, [2] * 10, batch_size=4, p_row=p)     # C = 2
    spec = dict(SPEC, pick='weighted')
    n3 = sum(3 in m._choose_draw_conditions(spec).tolist() for _ in range(500))
    assert n3 > 450                                         # uniform would give ~100
    assert m._cond_draw_stats['cond_draw/weighted_fallback'] == 0.0


def test_the_weighted_pick_is_uniform_without_a_measure():
    m = _chooser([2] * 10, [2] * 10, batch_size=4, p_row=None)
    c = m._choose_draw_conditions(dict(SPEC, pick='weighted'))
    assert c.size == 2 and len(set(c.tolist())) == 2
    assert m._cond_draw_stats['cond_draw/weighted_fallback'] == 1.0


def test_the_uniform_beta_takes_its_share_of_the_pick_uniformly():
    p = np.full(10, 1e-6)
    p[3] = 1.0
    m = _chooser([2] * 10, [2] * 10, batch_size=4, p_row=p, beta=1.0)
    n3 = sum(3 in m._choose_draw_conditions(dict(SPEC, pick='weighted')).tolist()
             for _ in range(1000))
    assert n3 < 350                                         # all uniform: ~200


# ------------------------------------------------------------ the fused step

@pytest.fixture
def pooled(monkeypatch):
    def fake(first, bwd, **kw):
        return None, {}
    monkeypatch.setitem(Modeller.fused_train_step.__globals__,
                        'pooled_condition_vargrad', fake)


def _fused(cond_spec=SPEC, cids=(4, 7), pooled_vg=1.0):
    gfn = SimpleNamespace(_live_fwd=None, _live_bwd=None, _live_replay=None)
    stage = SimpleNamespace(name='var_conditioning', deactivate_threshold=0.01,
                            fwd_z_sidecar=False, condition_draw=cond_spec)
    m = SimpleNamespace(
        args=SimpleNamespace(controller=SimpleNamespace(deactivate_threshold=0.01,
                                                        refresh_every=10),
                             fwd_loss_coeffs=SimpleNamespace(
                                 pooled_vg=pooled_vg, pooled_source='fwd', pooled_beta=40.0,
                                 pooled_ratio=0.5, pooled_bridge_only=0.0)),
        protocol=SimpleNamespace(stage=stage, mode_boostable=lambda mode: True,
                                 mode_dormant=lambda mode: False),
        replay_buffer=[0] * 5, gfn_model=gfn, fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5,
        fused_step_count=0, seen={})

    def fwd_step(discretizer, return_exp, repeats, report_losses):
        gfn._live_fwd = {'condition_id': torch.tensor([3, 4])}
        return torch.tensor(5.0), 'crystal_batch', {}

    def bwd_step(discretizer, repeats, report_losses, target_cids, **kw):
        m.seen['bwd'] = (target_cids, kw)
        return torch.tensor(1.0), {}

    def replay_step(discretizer, repeats, report_losses, **kw):
        m.seen['replay'] = kw
        return torch.tensor(3.0), {}

    def choose(spec):
        m.seen['choose'] = spec
        return np.asarray(cids, dtype=np.int64)

    m._fwd_gates = lambda deact, force_refresh: (True, False, False)
    m.fwd_train_step, m.bwd_train_step, m.replay_train_step = fwd_step, bwd_step, replay_step
    m._choose_draw_conditions = choose
    m.mode_repeats = lambda mode: 1
    m._stash_z_fill_logw = lambda d: None
    m._fused_grad_diag_armed = lambda: False
    m.manage_replay_buffer = lambda d, b: None
    m.fused_train_step = MethodType(Modeller.fused_train_step, m)
    return m


def test_both_branches_draw_on_the_one_condition_set(pooled):
    m = _fused()
    m.fused_train_step(None)
    assert m.seen['choose'] is SPEC
    tgt, bkw = m.seen['bwd']
    decision = bkw['condition_draw']
    assert m.seen['replay']['condition_draw'] is decision
    assert decision['cids'].tolist() == [4, 7]
    assert decision['replay_rows'] == 2 and decision['prior_rows'] == 2


def test_an_aligned_step_does_not_also_align_bwd_to_the_forward_batch(pooled):
    """With the pooled term on and its source fwd, today's step hands the forward
    batch's conditions to the bwd draw. The aligned condition set replaces that."""
    m = _fused(pooled_vg=1.0)
    m.fused_train_step(None)
    assert m.seen['bwd'][0] is None


def test_no_eligible_condition_skips_replay_and_folds_its_share_into_bwd(pooled):
    """The fold only. bwd's loss is a stub constant with no graph, so the step is
    also marked gradless; what train_step does with that on the real stage's
    (term-free) coefficients is tests/protocol/test_gradless_fused_step.py."""
    m = _fused(cids=())
    loss, subs = m.fused_train_step(None)
    assert 'replay' not in m.seen and 'replay' not in subs
    assert float(loss) == pytest.approx(1.0), 'bwd alone, at the whole weight'
    assert m._fused_step_gradless is True
    tgt, bkw = m.seen['bwd']
    assert bkw == {}, 'bwd draws as without the block'
    assert list(tgt) == [3, 4], "...including today's forward alignment"


def test_a_stage_without_the_block_is_untouched(pooled):
    m = _fused(cond_spec=None)
    m._choose_draw_conditions = lambda spec: pytest.fail('chose conditions with no block')
    m.fused_train_step(None)
    assert m.seen['bwd'][1] == {} and m.seen['replay'] == {}


# ------------------------------------------------------------ the two draws

class _MB:
    def to(self, device):
        return self


class _Recorder:
    """A buffer whose loader records its keyword arguments."""

    def __init__(self, out):
        self.out, self.calls = out, []

    def loader(self, **kw):
        self.calls.append(kw)
        while True:
            yield self.out


def _drawer(mode='prior'):
    m = SimpleNamespace(
        bwd_sampling_mode=mode, device='cpu',
        prior_buffer=_Recorder((_MB(), np.arange(4))),
        replay_buffer=_Recorder((_MB(), torch.zeros(4, 3), np.arange(4))),
        _batch_latents=lambda mb: torch.zeros(4, 12),
        energy_function=SimpleNamespace(
            condition_samples=lambda mb, repeats: (mb, torch.zeros(4), 'cond', 'cid'),
            prebuilt_sample_to_reward=lambda mb, t: torch.zeros(4)),
        protocol=SimpleNamespace(flag=lambda name: pytest.fail(f'read flag {name}')),
        replay_priority_config=lambda: pytest.fail('read the prioritised draw'),
        _replay_is_w=torch.ones(4), _replay_is_stats={'replay/is_ess_frac': 0.5})
    m.draw_bwd_sample = MethodType(Modeller.draw_bwd_sample, m)
    m.draw_replay_sample = MethodType(Modeller.draw_replay_sample, m)
    m._finish_replay_draw = MethodType(Modeller._finish_replay_draw, m)
    return m


DECISION = dict(SPEC, replay_rows=3, prior_rows=2, cids=np.array([4, 7]))


def test_the_bwd_draw_takes_exactly_the_decision():
    m = _drawer()
    m.draw_bwd_sample(2, condition_draw=DECISION)
    (kw,) = m.prior_buffer.calls
    assert kw['batch_size'] == 4 and kw['rows_per_condition'] == 2 and kw['repeats'] == 2
    assert kw['draw_cids'].tolist() == [4, 7]
    assert not ({'weighted', 'condition_block_m', 'target_cids', 'p'} & set(kw))


def test_the_bwd_draw_refuses_the_decision_off_the_prior_buffer():
    with pytest.raises(ValueError, match="bwd_sampling_mode 'prior'"):
        _drawer(mode='dataset').draw_bwd_sample(1, condition_draw=DECISION)


def test_the_replay_draw_takes_exactly_the_decision_and_no_is_weights():
    m = _drawer()
    m.draw_replay_sample(1, condition_draw=DECISION)
    (kw,) = m.replay_buffer.calls
    assert kw['batch_size'] == 6 and kw['rows_per_condition'] == 3
    assert kw['return_traj'] is True and kw['draw_cids'].tolist() == [4, 7]
    assert 'p' not in kw and 'condition_block_m' not in kw
    assert m._replay_is_w is None and m._replay_is_stats == {}


# ------------------------------------------------------------ the report

def test_the_report_publishes_the_decision_only_on_a_stage_with_the_block():
    """ten_step_reporting is too large to drive on a stub; pin the one gate."""
    src = textwrap.dedent(inspect.getsource(Modeller.ten_step_reporting))
    tree = ast.parse(src)
    gates = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and "'condition_draw'" in ast.unparse(n.test)]
    assert len(gates) == 1
    body = ast.unparse(gates[0])
    assert '_cond_draw_stats' in body and 'cond_draw/skipped_steps' in body
    assert '_cond_draw_skips = 0' in body, 'the skip count drains on read'
    assert 'cond_draw/no_grad_steps' in body and '_gradless_fused_steps = 0' in body
