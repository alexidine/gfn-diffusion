"""Continuing a checkpoint written WITHOUT force gates in a model that has them.

An arm that adds a force term (model.force_drift_fwd / force_drift_bwd) to a shared
checkpoint differs from its control by the gates alone, and three things in the checkpoint
loader have to hold for that to be true:

  * the force keys are read from THIS run's config, not the checkpoint's stored gfn_config
    (else the arm silently runs as the control);
  * the state dict loads with the gates left at their starting values, and with nothing
    else forgiven;
  * the optimizer keeps every old parameter's moments and step count, the gates starting
    without any (else the arm also differs from its control by a reset optimizer).

The last test puts them together: with a gate that starts at zero, the arm's first update of
every shared parameter equals the control's.
"""
import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(os.path.dirname(_here)),
          os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_here))), 'mxtaltools')):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from models.gfn import GFN  # noqa: E402
from checkpointing import Checkpointer  # noqa: E402
from utils import uniform_discretizer  # noqa: E402

pytestmark = pytest.mark.fast

DIM, T, B = 4, 5, 12


def make_gfn(seed=0, **force):
    torch.manual_seed(seed)
    return GFN(dim=DIM, s_emb_dim=16, harmonics_dim=8, t_dim=8, t_hidden_dim=16, s_hidden_dim=16, s_layers=2,
               policy_hidden_dim=16, policy_layers=2, flow_hidden_dim=16, flow_layers=2, cond_hidden_dim=16,
               cond_layers=2, conditions_dim=0, condition_embedding_dim=8, conditions_type='vector',
               conditional=False, learn_pb=True, learned_variance=True, device=torch.device('cpu'),
               do_periodic_angles=False, hold_dead_latent_rows=False, **force)


def optimizer(g):
    """The trainer's policy group layout (train.py init_schedulers_optimizers)."""
    return torch.optim.Adam([{'params': g.t_model.parameters()}, {'params': g.s_model.parameters()},
                             {'params': g.forward_policy.parameters()},
                             {'params': g.backward_policy.parameters()}], lr=1e-3)


def quadratic_force(state, ctx, create_graph):
    return -(state - 0.2) / 0.3


def loss_on(g, traj):
    _, pf, pb, _ = g.get_traj_replay(traj, lambda b: uniform_discretizer(b, T), condition=None, mol_batch=None)
    return (pf.sum(1) - pb.sum(1)).pow(2).mean()


def trained_control(steps=3):
    """A gate-less model and its optimizer after a few updates, with the trajectories used."""
    g = make_gfn()
    opt = optimizer(g)
    torch.manual_seed(1)
    traj = torch.randn(B, T + 1, DIM) * 0.3
    traj[:, 0] = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss_on(g, traj).backward()
        opt.step()
    return g, opt, traj


def checkpointer(model_args=None):
    ck = Checkpointer.__new__(Checkpointer)
    ck.modeller = SimpleNamespace(args=SimpleNamespace(model=SimpleNamespace(**(model_args or {}))))
    ck._assert_dead_rows_match = lambda config: None
    return ck


def test_force_keys_come_from_this_runs_config():
    stored = {'dim': DIM, 't_scale': 0.05, 'force_drift_fwd': None}
    ck = checkpointer({'t_scale': 9.9, 'force_drift_fwd': 0.0, 'force_drift_t_min': 0.8,
                       'force_drift_learned': True})
    config = ck._gfn_config_from({'gfn_config': stored})
    assert config['force_drift_fwd'] == 0.0 and config['force_drift_t_min'] == 0.8
    assert config['force_drift_learned'] is True
    assert config['t_scale'] == 0.05, 'architecture keys still come from the checkpoint'
    assert 'force_drift_bwd' not in config, 'a key this config does not carry is not invented'
    # and a config without the keys leaves an old checkpoint exactly as it was
    assert checkpointer({'t_scale': 9.9})._gfn_config_from({'gfn_config': stored}) == stored


def test_state_dict_loads_with_gates_at_their_start_and_nothing_else_forgiven(capsys):
    old, _, _ = trained_control()
    ck = checkpointer()
    new = make_gfn(seed=5, force_drift_fwd=0.3, force_drift_bwd=-0.1)
    ck._load_state(new, old.state_dict(), 'seed.pt (model_train)')
    assert 'written without this model' in capsys.readouterr().out
    shared = old.state_dict()
    assert all(torch.equal(v, shared[k]) for k, v in new.state_dict().items() if '.force_gate.' not in k)
    a_f, a_b = new.force_gate_values(torch.linspace(0, 1, 4))
    assert torch.allclose(a_f, torch.full_like(a_f, 0.3)) and torch.allclose(a_b, torch.full_like(a_b, -0.1))

    # a rewind onto a model whose gates have trained puts them back
    with torch.no_grad():
        for p in new.forward_policy.force_gate.parameters():
            p.add_(1.0)
    assert not torch.allclose(new.force_gate_values(torch.linspace(0, 1, 4))[0], a_f)
    ck._load_state(new, old.state_dict(), 'seed.pt (model_train)', in_place=True)
    assert torch.allclose(new.force_gate_values(torch.linspace(0, 1, 4))[0], a_f)

    broken = {k: v for k, v in old.state_dict().items() if not k.startswith('t_model')}
    with pytest.raises(RuntimeError, match='Missing keys'):
        ck._load_state(make_gfn(force_drift_fwd=0.3), broken, 'broken')
    with pytest.raises(RuntimeError, match='unexpected keys'):
        ck._load_state(make_gfn(), new.state_dict(), 'gated checkpoint into a gate-less model')


def test_optimizer_keeps_old_parameters_state_and_starts_the_gates_fresh(capsys):
    old, opt_old, _ = trained_control()
    saved = opt_old.state_dict()
    new = make_gfn(force_drift_fwd=0.0, force_drift_bwd=0.0)
    opt_new = optimizer(new)
    with pytest.raises(ValueError):
        optimizer(new).load_state_dict(saved)          # what the loader used to fall back from

    ck = checkpointer()
    ck.modeller.optimizers = {'fwd': opt_new}
    ck.load_optimizer_state({'optimizers': {'fwd': saved}}, strict=True)
    n_gate = sum(1 for pol in (new.forward_policy, new.backward_policy) for _ in pol.force_gate.parameters())
    assert f"optimizer 'fwd': {n_gate} parameters are new" in capsys.readouterr().out

    gate_ids = {id(p) for pol in (new.forward_policy, new.backward_policy) for p in pol.force_gate.parameters()}
    for g_old, g_new in zip(opt_old.param_groups, opt_new.param_groups):
        carried = [p for p in g_new['params'] if id(p) not in gate_ids]
        assert len(carried) == len(g_old['params'])
        for p_old, p_new in zip(g_old['params'], carried):
            s_old, s_new = opt_old.state[p_old], opt_new.state[p_new]
            assert torch.equal(s_old['exp_avg'], s_new['exp_avg']) and torch.equal(s_old['exp_avg_sq'], s_new['exp_avg_sq'])
            assert float(s_old['step']) == float(s_new['step']) == 3.0
        assert g_new['lr'] == g_old['lr']
    assert all(len(opt_new.state.get(p, {})) == 0 for g in opt_new.param_groups for p in g['params']
               if id(p) in gate_ids)

    # identical layouts take the ordinary path
    assert Checkpointer._grown_optimizer_state(optimizer(make_gfn()), saved) is None
    # and a layout that is not "the same groups, longer" is not guessed at
    reordered = torch.optim.Adam([{'params': new.s_model.parameters()}, {'params': new.t_model.parameters()},
                                  {'params': new.forward_policy.parameters()},
                                  {'params': new.backward_policy.parameters()}], lr=1e-3)
    assert Checkpointer._grown_optimizer_state(reordered, saved) is None


def test_an_arm_with_a_zero_gate_starts_as_its_control():
    old, opt_old, traj = trained_control()
    ck = checkpointer()
    arm = make_gfn(seed=9, force_drift_fwd=0.0, force_drift_bwd=0.0)
    arm.install_drift_force(quadratic_force)
    ck._load_state(arm, old.state_dict(), 'seed (model_train)')
    opt_arm = optimizer(arm)
    ck.modeller.optimizers = {'fwd': opt_arm}
    # deep-copied, as a checkpoint read from disk is: the live state dict shares its tensors with opt_old
    ck.load_optimizer_state({'optimizers': {'fwd': copy.deepcopy(opt_old.state_dict())}}, strict=True)

    for g, opt in ((old, opt_old), (arm, opt_arm)):
        opt.zero_grad()
        loss_on(g, traj).backward()
        opt.step()
    after = old.state_dict()
    for k, v in arm.state_dict().items():
        if '.force_gate.' not in k:
            assert torch.allclose(v, after[k], atol=1e-7, rtol=1e-6), k
    moved = [p for p in arm.forward_policy.force_gate.parameters()][-1]
    assert moved.abs().max() > 0, 'the gate itself has started to train'
