"""The replay seat's pooled VarGrad, through the REAL estimator.

On var_conditioning the policy trains ONLY through pooled_condition_vargrad over
(gfn._live_replay, gfn._live_bwd): bwd and replay carry no term of their own.
Every fused-step test patches the estimator, so this is where the two real
writers -- get_gfn_backward_loss with live_stash 'replay', and with its default
bwd slot -- meet the real estimator, on the condition ids the aligned draw hands
them: the same C conditions, 2 rows each, on both branches. Coefficients are the
arm's resolved ones (configs/qm9c_anneal.yaml: base block, then the stage's
override), so a stage that detached the policy on either branch fails here.
"""
import pathlib
from types import SimpleNamespace

import torch
import yaml

from energy_sampling.gflownet_losses import get_gfn_backward_loss, pooled_condition_vargrad

ARM = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'qm9c_anneal.yaml'
T, D, C = 20, 12, 8


def _resolved(mode):
    cfg = yaml.safe_load(ARM.read_text(encoding='utf-8'))
    vc = cfg['protocols']['conditional_vargrad']['stages'][1]
    assert vc['name'] == 'var_conditioning'
    block = dict(cfg[f'{mode}_loss_coeffs'])
    block.update(vc['loss_coeffs'].get(mode, {}))
    return SimpleNamespace(**block)


class _Gfn:
    """bwd's trajectories are fed by P_bwd, replay's by P_replay; stash ARMED.
    log_pfs SCALES with the parameter: a shift common to every row of log_pf
    cancels against the per-group zero-sum coefficients, so an additive one
    would read zero gradient whatever the wiring."""
    device = 'cpu'
    _stash_live_branches = True

    def __init__(self):
        self.P_bwd = torch.nn.Parameter(torch.zeros(()))
        self.P_replay = torch.nn.Parameter(torch.zeros(()))

    @staticmethod
    def _traj(n, p):
        return (torch.randn(n, T + 1, D) + p, torch.randn(n, T) * (1 + p), torch.randn(n, T) + p,
                torch.randn(n, T + 1) + p)

    def get_traj_bwd(self, samples, *a, **kw):
        return self._traj(samples.shape[0], self.P_bwd)

    def get_traj_replay(self, trajectories, *a, **kw):
        return self._traj(trajectories.shape[0], self.P_replay)


def _branches(gfn, replay_offset=0):
    """One replay call and one bwd call, as the fused step makes them."""
    bwd_c, rep_c = _resolved('bwd'), _resolved('replay')
    cids = torch.arange(C).repeat_interleave(2)
    n = cids.numel()
    rep_loss, _ = get_gfn_backward_loss(
        rep_c, torch.zeros(n, D), gfn, -10.0 * torch.rand(n), None, None,
        condition=torch.zeros(n, 1), repeats=1, trajectories=torch.zeros(n, T + 1, D),
        condition_id=cids + replay_offset, tb_z_source=rep_c.tb_z_source, live_stash='replay')
    bwd_loss, _ = get_gfn_backward_loss(
        bwd_c, torch.zeros(n, D), gfn, -10.0 * torch.rand(n), None, None,
        condition=torch.zeros(n, 1), repeats=1, condition_id=cids,
        tb_z_source=bwd_c.tb_z_source)
    return rep_loss, bwd_loss


def _pooled(gfn):
    fwd_c = _resolved('fwd')
    assert fwd_c.pooled_source == 'replay' and fwd_c.pooled_vg > 0
    return pooled_condition_vargrad(gfn._live_replay, gfn._live_bwd,
                                    beta=float(fwd_c.pooled_beta),
                                    ratio=float(fwd_c.pooled_ratio),
                                    # absent from the arm; fused_train_step reads 0 then
                                    bridge_only=float(getattr(fwd_c, 'pooled_bridge_only',
                                                              0.0) or 0.0) > 0.5)


def test_the_two_slots_form_mixed_groups_and_carry_the_policy_gradient():
    gfn = _Gfn()
    rep_loss, bwd_loss = _branches(gfn)
    assert not rep_loss.requires_grad and not bwd_loss.requires_grad, \
        'the branch losses are term-free on this stage; the pooled term is the gradient'
    rows, stats = _pooled(gfn)
    assert rows is not None
    assert float(stats['pooled_mixed_frac']) == 1.0, 'every group spans both branches'
    assert float(stats['pooled_lambda_b']) == 0.5
    assert float(stats['pooled_group_size_mean']) == 4.0     # 2 replay + 2 bwd rows
    assert float(stats['pooled_live_frac']) == 1.0
    rows.mean().backward()
    assert float(gfn.P_replay.grad.abs()) > 0, "through replay's log_pf"
    assert float(gfn.P_bwd.grad.abs()) > 0, "through bwd's log_pf"


def test_writers_that_disagree_on_condition_ids_read_as_single_source_groups():
    """The failure this seat can suffer silently: every group single-source, the
    cross-branch transfer gone. pooled_mixed_frac is the readout that shows it."""
    gfn = _Gfn()
    _branches(gfn, replay_offset=C)
    rows, stats = _pooled(gfn)
    assert rows is not None and float(stats['pooled_mixed_frac']) == 0.0
