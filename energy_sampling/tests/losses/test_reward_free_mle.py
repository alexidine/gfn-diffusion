"""A reward-free backward call (get_gfn_backward_loss log_r None): the MLE stage's draw is never scored.

MLE reads log P_F and log P_B only, so train.py::Modeller.draw_bwd_sample skips prebuilt_sample_to_reward when the
bwd coefficients weight no term that reads log R (gflownet_losses.bwd_loss_reads_reward). The loss must then refuse
any reward-reading term, keep the reward out of the condition log Z tracker and the level stream, and hand back a
loss_dict without the reward's keys.

    python -m pytest -q tests/losses/test_reward_free_mle.py
"""
from types import SimpleNamespace

import pytest
import torch

from energy_sampling.gflownet_losses import BWD_REWARD_TERMS, bwd_loss_reads_reward, get_gfn_backward_loss

pytestmark = pytest.mark.fast

B, T, D = 6, 1, 3


def _coeffs(**kw):
    c = dict(beta=10.0, freeze_policy=0.0, freeze_z=1.0, vg_by_condition=0.0, vg_lb=0.0, vg_lme=0.0,
             level_gap=0.0, pf_boost=0.0, db=0.0, subtb=0.0, emp_z=0.0, tb=0.0, emp_z_persistent=0.0, mle=1.0,
             tbc=0.0, loss_clip=1.0e9, traj_grads=0.0)
    c.update(kw)
    return SimpleNamespace(**c)


class _Gfn:
    device = 'cpu'
    _stash_live_branches = False

    def __init__(self):
        self.theta = torch.zeros(B, T, requires_grad=True)

    def get_traj_bwd(self, samples, discretizer, condition, mol_batch, **kw):
        log_pfs = -0.5 * (samples[:, None, :1] - self.theta[..., None]).pow(2).sum(-1)
        return torch.zeros(B, T + 1, D), log_pfs, torch.zeros(B, T), torch.zeros(B, T + 1)


class _Tracker:
    def __init__(self):
        self.calls = []

    def update_mode_level(self, *a, **k):
        self.calls.append('level')

    def update(self, *a, **k):
        self.calls.append('update')

    def lookup(self, cid):
        return torch.zeros(len(cid)), torch.zeros(len(cid), dtype=torch.bool)


def _samples(seed=0):
    return torch.randn(B, D, generator=torch.Generator().manual_seed(seed))


def _call(coeffs, log_r=None, tracker=None, samples=None, **kw):
    gfn = _Gfn()
    samples = _samples() if samples is None else samples
    loss, d = get_gfn_backward_loss(coeffs, samples, gfn, log_r, None, None, repeats=1,
                                    report_losses=True, condition_log_z=tracker, condition_id=torch.arange(B),
                                    update_log_z=True, mode_level_stream='bwd', **kw)
    return gfn, loss, d


def test_mle_alone_reads_no_reward():
    assert not bwd_loss_reads_reward(_coeffs())
    for k in BWD_REWARD_TERMS:
        assert bwd_loss_reads_reward(_coeffs(**{k: 1.0})), k


def test_a_reward_free_mle_call_trains_and_omits_the_reward_keys():
    gfn, loss, d = _call(_coeffs())
    assert torch.isfinite(loss)
    loss.backward()
    assert gfn.theta.grad is not None and torch.isfinite(gfn.theta.grad).all()
    assert 'mle' in d and not {'log_r', 'resid', 'logr_mean'} & set(d)


def test_the_same_mle_loss_with_or_without_the_reward():
    log_r = torch.randn(B, generator=torch.Generator().manual_seed(1))
    _, with_r, _ = _call(_coeffs(), log_r=log_r, samples=_samples(3))
    _, without, _ = _call(_coeffs(), samples=_samples(3))
    assert torch.equal(with_r, without)


def test_the_tracker_and_level_stream_are_not_fed():
    tr = _Tracker()
    _call(_coeffs(), tracker=tr)
    assert tr.calls == []
    tr = _Tracker()
    _call(_coeffs(), log_r=torch.zeros(B), tracker=tr)
    assert tr.calls == ['level', 'update']   # the scored call still feeds both


@pytest.mark.parametrize('term', BWD_REWARD_TERMS)
def test_a_reward_reading_term_refuses_a_reward_free_call(term):
    with pytest.raises(ValueError, match='reward-free'):
        _call(_coeffs(**{term: 1.0}))


def test_an_armed_pooled_stash_refuses_a_reward_free_call():
    gfn = _Gfn()
    gfn._stash_live_branches = True
    with pytest.raises(ValueError, match='reward-free'):
        get_gfn_backward_loss(_coeffs(), torch.randn(B, D), gfn, None, None, None, repeats=1,
                              condition_id=torch.arange(B))
