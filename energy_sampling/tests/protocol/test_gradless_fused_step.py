"""A fused step with nothing to descend takes no optimizer step, rather than die.

var_conditioning's bwd and replay blocks carry no term of their own: the policy
trains only through the pooled replay/bwd term, and the Z(c) head through the fwd
sidecar. When the aligned draw finds no eligible condition, replay is skipped and
the pooled term is INERT; off the rollout cadence there is no sidecar either, so
the fused loss is get_gfn_backward_loss's zeros fallback -- no graph -- and
backward() raised a non-OOM RuntimeError that the host loop re-raises, ending the
run.

Real Modeller.fused_train_step, _choose_draw_conditions, train_step and step_loss
on a stub, at the arm's RESOLVED coefficients (configs/qm9c_anneal.yaml: base
block, then the stage's override), with the live stash ARMED as a run arms it and a
parameter feeding every trajectory tensor, so any term that exists carries a
gradient. Stubbed: the rollout gates, the three branch draws (each calls the real
loss on fixed condition ids), the optimizer, admission, the EMA and the ray probe.
"""
import pathlib
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from energy_sampling.gflownet_losses import get_gfn_backward_loss, get_gfn_forward_loss
from energy_sampling.train import Modeller

ARM = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'qm9c_anneal.yaml'
T, D, N_COND = 20, 12, 8


def _cfg():
    return yaml.safe_load(ARM.read_text(encoding='utf-8'))


def _vc(cfg):
    vc = cfg['protocols']['conditional_vargrad']['stages'][1]
    assert vc['name'] == 'var_conditioning'
    return vc


def _resolved(cfg, mode):
    """A branch's effective coefficients: the base block, then the stage's override."""
    block = dict(cfg[f'{mode}_loss_coeffs'])
    block.update(_vc(cfg)['loss_coeffs'].get(mode, {}))
    return SimpleNamespace(**block)


class _Gfn:
    """Trajectories only, every tensor fed by one parameter; the stash is ARMED.
    log_pfs SCALES with it: a shift common to log_pf and log_pb cancels in log w,
    and the pooled term's per-group coefficients sum to zero, so an additive
    parameter could never carry its gradient."""
    device = 'cpu'
    _stash_live_branches = True
    live_idx = None
    _live_fwd = _live_bwd = _live_replay = None

    def __init__(self):
        self.P = torch.nn.Parameter(torch.zeros(()))

    def _traj(self, n):
        return (torch.randn(n, T + 1, D) + self.P, torch.randn(n, T) * (1 + self.P),
                torch.randn(n, T) + self.P, torch.randn(n, T + 1) + self.P)

    def get_traj_fwd(self, s, *a, **kw):
        return self._traj(s.shape[0])

    def get_traj_bwd(self, s, *a, **kw):
        return self._traj(s.shape[0])

    def get_traj_replay(self, trajectories, *a, **kw):
        return self._traj(trajectories.shape[0])


class _Counts:
    """A buffer answering condition_row_counts: `rows` trainable rows per condition."""

    def __init__(self, rows):
        self.counts = np.full(N_COND, rows, dtype=np.int64)

    def __len__(self):
        return int(self.counts.sum())

    def condition_row_counts(self, minlength=0):
        return self.counts.copy()


class _Opt:
    def __init__(self):
        self.zeroed = 0

    def zero_grad(self, set_to_none=True):
        self.zeroed += 1


def _pairs():
    return torch.arange(N_COND).repeat_interleave(2)


def _decision_ids(condition_draw, key):
    if condition_draw is None:
        return _pairs()                       # the draw without the block
    return torch.as_tensor(np.repeat(condition_draw['cids'], int(condition_draw[key])))


def _modeller(cfg, rollout, replay_rows=None, condition_draw=True, accum_target=0):
    fwd_c, bwd_c, rep_c = (_resolved(cfg, mode) for mode in ('fwd', 'bwd', 'replay'))
    vc = _vc(cfg)
    gfn = _Gfn()
    stage = SimpleNamespace(name=vc['name'], deactivate_threshold=vc['deactivate_threshold'],
                            fwd_z_sidecar=vc['fwd_z_sidecar'],
                            condition_draw=vc['condition_draw'] if condition_draw else None)
    m = SimpleNamespace(
        args=SimpleNamespace(
            controller=SimpleNamespace(deactivate_threshold=cfg['controller']['deactivate_threshold'],
                                       refresh_every=cfg['controller']['refresh_every']),
            fwd_loss_coeffs=fwd_c, bwd_loss_coeffs=bwd_c, replay_loss_coeffs=rep_c,
            integrator=SimpleNamespace(T=T), fused_grad_accum_min_samples=accum_target),
        protocol=SimpleNamespace(stage=stage, mode_boostable=lambda mode: True,
                                 mode_dormant=lambda mode: False),
        gfn_model=gfn, fwd_frac=0.0, bwd_frac=0.5, replay_frac=0.5, fused_step_count=0,
        step_ind=1, batch_size=2 * N_COND, fused_accum_count=0,
        energy_function=SimpleNamespace(n_sg=1, n_zp=1, condition_library_size=N_COND),
        prior_buffer=_Counts(2), optimizers={'fused': _Opt()}, seen={})
    if replay_rows is not None:
        m.replay_buffer = _Counts(replay_rows)

    def fwd_step(discretizer, return_exp, repeats, report_losses):
        cids = _pairs()
        n = cids.numel()
        loss, d = get_gfn_forward_loss(
            fwd_c, torch.zeros(n, D), gfn, lambda x, *a, **kw: -x.pow(2).sum(-1), None, None,
            torch.zeros(n), condition=torch.zeros(n, 1), repeats=int(fwd_c.repeats),
            report_losses=True, condition_id=cids, tb_z_source=fwd_c.tb_z_source)
        return loss, 'crystal_batch', d

    def bwd_step(discretizer, repeats, report_losses, target_cids=None, condition_draw=None):
        cids = _decision_ids(condition_draw, 'prior_rows')
        n = cids.numel()
        return get_gfn_backward_loss(
            bwd_c, torch.zeros(n, D), gfn, -10.0 * torch.rand(n), None, None,
            condition=torch.zeros(n, 1), repeats=1, report_losses=True, condition_id=cids,
            tb_z_source=bwd_c.tb_z_source)

    def replay_step(discretizer, repeats, report_losses, condition_draw=None):
        cids = _decision_ids(condition_draw, 'replay_rows')
        n = cids.numel()
        m.seen['replay'] = cids
        return get_gfn_backward_loss(
            rep_c, torch.zeros(n, D), gfn, -10.0 * torch.rand(n), None, None,
            condition=torch.zeros(n, 1), repeats=1, report_losses=True,
            trajectories=torch.zeros(n, T + 1, D), condition_id=cids,
            tb_z_source=rep_c.tb_z_source, live_stash='replay')

    m.fwd_train_step, m.bwd_train_step, m.replay_train_step = fwd_step, bwd_step, replay_step
    m._fwd_gates = lambda deact, force_refresh: (rollout, False, False)
    m.mode_repeats = lambda mode: 1
    m._stash_z_fill_logw = lambda d: None
    m._fused_grad_diag_armed = lambda: False
    m.manage_replay_buffer = lambda d, b: None
    m._ray_probe_armed = lambda: False
    m.record_fused_substep_losses = lambda subs: None
    m.update_ema_model = lambda: None
    for name in ('fused_train_step', '_choose_draw_conditions', 'train_step', 'step_loss'):
        setattr(m, name, MethodType(getattr(Modeller, name), m))
    return m


def _record_steps(m):
    """Replace step_loss with a recorder that backpropagates what it is handed."""
    m.stepped = []

    def rec(step_type, loss, do_step=True):
        m.stepped.append(loss)
        loss.backward()
    m.step_loss = rec


# ------------------------------------------------------------ the fallback

def test_an_off_cadence_step_with_nothing_eligible_takes_no_optimizer_step():
    """THE CRASH. No replay buffer yet, no rollout: the real step_loss is bound,
    so reaching backward() on the grad-free loss raises here as it did in a run."""
    m = _modeller(_cfg(), rollout=False)
    assert not hasattr(m, 'replay_buffer')
    reported = m.train_step('fused')
    assert reported == 0.0
    assert m._cond_draw_skips == 1 and m._fused_step_gradless is True
    assert m._gradless_fused_steps == 1
    assert m.gfn_model.P.grad is None
    assert m.optimizers['fused'].zeroed == 1, 'the cycle still opens; nothing lands in it'


def test_an_ineligible_buffer_is_the_same_fallback():
    """A replay buffer that exists but holds fewer than replay_rows of every
    condition: the same skip, the same no-step."""
    m = _modeller(_cfg(), rollout=False, replay_rows=1)
    m.train_step('fused')
    assert 'replay' not in m.seen and m._gradless_fused_steps == 1


def test_the_skip_leaves_a_partial_accumulation_cycle_alone():
    m = _modeller(_cfg(), rollout=False, accum_target=3 * 2 * N_COND)
    m.fused_accum_count = 2 * N_COND                        # one batch into the cycle
    m.train_step('fused')
    assert m.fused_accum_count == 2 * N_COND
    assert m.optimizers['fused'].zeroed == 0, 'mid-cycle: the gradients piled up so far stay'


# ------------------------------------------------------------ steps that do descend

def test_a_rollout_step_descends_the_sidecar():
    m = _modeller(_cfg(), rollout=True)
    _record_steps(m)
    m.train_step('fused')
    (loss,) = m.stepped
    assert loss.requires_grad and m._fused_step_gradless is False
    assert m._sidecar_count == 1 and getattr(m, '_gradless_fused_steps', 0) == 0


def test_an_eligible_step_descends_the_pooled_term():
    """The ordinary off-cadence step: replay drawn, both slots stashed, the real
    pooled estimator ACTIVE and the only gradient."""
    m = _modeller(_cfg(), rollout=False, replay_rows=2)
    _record_steps(m)
    m.train_step('fused')
    (loss,) = m.stepped
    assert m._fused_step_gradless is False
    assert m._pooled_stats['pooled_mixed_frac'] == 1.0
    assert float(m.gfn_model.P.grad.abs()) > 0


def test_a_grad_free_loss_off_the_fallback_still_raises():
    """The skip is the zero-eligible fallback ONLY. Without condition_draw the
    same grad-free loss is a wiring fault, and step_loss still raises."""
    m = _modeller(_cfg(), rollout=False, condition_draw=False)
    with pytest.raises(RuntimeError, match='does not require grad'):
        m.train_step('fused')
    assert m._fused_step_gradless is False


# ------------------------------------------------------------ the announce

def test_the_pooled_announce_prints_each_state_once(capsys):
    """Step 0 of a stage entry is INERT (nothing eligible yet), and the term is
    ACTIVE from step 1. A once-per-process line read INERT for the whole run."""
    m = _modeller(_cfg(), rollout=True)
    _record_steps(m)
    m.train_step('fused')                                   # entry rollout, no buffer
    m.replay_buffer = _Counts(2)
    m._fwd_gates = lambda deact, force_refresh: (False, False, False)
    for step in (1, 2, 3):
        m.step_ind = step
        m.train_step('fused')
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith('pooled VarGrad: coeff')]
    assert len(lines) == 2, lines
    assert 'term INERT (cond_draw: no eligible condition this step)' in lines[0]
    assert 'term ACTIVE' in lines[1] and 'at step 1' in lines[1]
