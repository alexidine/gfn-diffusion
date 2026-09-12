"""get_gfn_backward_loss's live-tensor stash: one slot per caller.

The bwd and replay branches both train through get_gfn_backward_loss, and in
the fused step replay runs AFTER bwd. With one `_live_bwd` slot the replay call
silently overwrote the bwd rows the pooled VarGrad term reads later in the same
step. `live_stash` names the slot; the default keeps bwd_train_step's writes
where they were.
"""
import ast
import inspect
import textwrap
from types import MethodType, SimpleNamespace

import pytest
import torch

from energy_sampling.gflownet_losses import get_gfn_backward_loss
from energy_sampling.train import Modeller

B, T, D = 4, 3, 2
COEFFS = SimpleNamespace(beta=10.0, freeze_policy=0.0, freeze_z=1.0, vg_by_condition=0.0,
                         vg_lb=0.0, vg_lme=0.0, level_gap=0.0, pf_boost=0.0, db=0.0,
                         subtb=0.0, emp_z=0.0, tb=0.0, emp_z_persistent=0.0, mle=0.0,
                         tbc=0.0, loss_clip=1.0e9, traj_grads=0.0)


class _Gfn:
    device = 'cpu'

    def __init__(self, armed=True):
        self._stash_live_branches = armed

    def get_traj_replay(self, trajectories, discretizer, condition, mol_batch, **kw):
        return (torch.zeros(B, T + 1, D), torch.zeros(B, T), torch.zeros(B, T),
                torch.zeros(B, T + 1))


def _call(gfn, **kw):
    return get_gfn_backward_loss(COEFFS, torch.zeros(B, D), gfn, torch.zeros(B), None, None,
                                 repeats=1, trajectories=torch.zeros(B, T + 1, D),
                                 condition_log_z=None, condition_id=torch.arange(B), **kw)


def test_the_default_slot_is_bwd():
    gfn = _Gfn()
    _call(gfn)
    assert gfn._live_bwd is not None and not hasattr(gfn, '_live_replay')


def test_the_replay_slot_leaves_the_bwd_rows_alone():
    """THE BUG. A replay call after bwd must not replace what bwd stashed."""
    gfn = _Gfn()
    bwd_rows = {'tag': 'bwd'}
    gfn._live_bwd = bwd_rows
    _call(gfn, live_stash='replay')
    assert gfn._live_bwd is bwd_rows
    assert set(gfn._live_replay) == {'log_r', 'log_pb', 'log_pf', 'condition_id'}


def test_none_stashes_nothing():
    gfn = _Gfn()
    _call(gfn, live_stash=None)
    assert not hasattr(gfn, '_live_bwd') and not hasattr(gfn, '_live_replay')


def test_an_unarmed_stash_writes_no_slot():
    gfn = _Gfn(armed=False)
    _call(gfn, live_stash='replay')
    assert not hasattr(gfn, '_live_replay')


def test_an_unknown_slot_is_refused():
    """A typo would otherwise park the rows under a name nothing reads."""
    with pytest.raises(ValueError, match='live_stash'):
        _call(_Gfn(), live_stash='fwd')


# ---------------------------------------------------------------------------
# The callers: replay_train_step stashes into its own slot only when it is the
# training call; the probe (val_rows) and z_calibration's tick (side_effects
# False) stash nothing.
# ---------------------------------------------------------------------------

def _replay_stub(monkeypatch):
    seen = {}

    def fake(coeffs, latents, gfn, log_r, disc, mol, **kw):
        seen.update(kw)
        z = torch.zeros(B)
        return torch.tensor(0.0), {'resid': z, 'log_r': z, 'log_pb': z, 'log_pf': z}

    monkeypatch.setitem(Modeller.replay_train_step.__globals__, 'get_gfn_backward_loss', fake)
    m = SimpleNamespace(
        args=SimpleNamespace(replay_loss_coeffs=COEFFS), gfn_model=_Gfn(), device='cpu',
        condition_log_z=None, step_ind=1, _replay_is_w=None,
        protocol=SimpleNamespace(flag=lambda name: False),
        replay_buffer=SimpleNamespace(update_losses=lambda *a: None,
                                      update_logw_stats=lambda *a: None))
    m.tb_z_source = lambda mode: 'learned'
    m.draw_replay_sample = lambda repeats, val_rows=0: (
        None, torch.arange(B), torch.arange(B), torch.zeros(B, D), torch.zeros(B), None,
        torch.zeros(B, T + 1, D))
    m._larder_harvest = lambda *a, **k: None
    m.replay_train_step = MethodType(Modeller.replay_train_step, m)
    return m, seen


@pytest.mark.parametrize('side_effects,val_rows,expected', [
    (True, 0, 'replay'),       # the training call
    (False, 0, None),          # z_calibration's replay tick
    (False, 64, None),         # the held-out probe
])
def test_replay_train_step_names_its_own_slot(monkeypatch, side_effects, val_rows, expected):
    m, seen = _replay_stub(monkeypatch)
    m.replay_train_step(None, repeats=1, side_effects=side_effects, val_rows=val_rows)
    assert seen['live_stash'] == expected


def _calls_in(func, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, 'id', getattr(n.func, 'attr', None)) == name]


def test_bwd_train_step_keeps_the_bwd_slot():
    """It passes no live_stash, so it takes the default -- the only writer of
    _live_bwd."""
    (call,) = _calls_in(Modeller.bwd_train_step, 'get_gfn_backward_loss')
    assert 'live_stash' not in {k.arg for k in call.keywords}


def test_the_ray_probe_re_score_stashes_nothing():
    """LarderScorer.score runs after fused_train_step clears the slots; its rows
    are from another parameter point and carry no graph."""
    from energy_sampling.lr_larder import LarderScorer
    (call,) = _calls_in(LarderScorer.score, 'get_gfn_backward_loss')
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw['live_stash'], ast.Constant) and kw['live_stash'].value is None
