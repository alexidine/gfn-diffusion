"""models/flow_policy.py::FlowPolicy, the exact-density flow as a train.py policy (docs/design/flow_policy_port.md):
the policy interface the trainer reads (models/policy_interface.py, re-derived here from the trainer's source), its
one-step trajectories (states [N, 2, D], log P_F = log q exact, log P_B = 0, log_flow[:, 0] the level head), the
SDE policy's geometry (wrapped and dead rows), Z-only training, condition scrambling, and parameter groups."""
import ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.flow_policy import FlowPolicy  # noqa: E402
from models.gfn import GFN  # noqa: E402
from models.policy_interface import (GETATTR_ONLY, POLICY_ATTRIBUTES, RECEIVERS, TRAINER_FILES,  # noqa: E402
                                     WRITTEN_STATE, missing_policy_attributes)

pytestmark = pytest.mark.fast
DEV = 'cpu'


def gfn_config(conditional=False, cond_dim=4, dead=None):
    return dict(dim=12, s_emb_dim=32, conditions_dim=cond_dim if conditional else 0, harmonics_dim=16, t_dim=8,
                t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32, policy_layers=2,
                flow_hidden_dim=32, flow_layers=2, cond_hidden_dim=32, cond_layers=2, log_var_range=6.0, t_scale=0.05,
                learned_variance=True, condition_embedding_dim=16 if conditional else 0, conditions_type='vector',
                clipping=True, gfn_clip=200.0, pb_drift_range=0.4, pb_var_range=6.0, conditional=conditional,
                learn_pb=True, dropout=0, norm='layer', zero_init=False, device=DEV, max_z_prime=1, full_flow=False,
                do_periodic_angles=True, periodic_centroids=True, periodic_centroid_axes=(1, 2),
                dead_latent_rows=dead, dead_latent_values=None, dplr_rank=0, pb_exact_reversal=True)


SMALL = dict(conditioner='made', hidden=32, hidden_layers=1, n_bins=8, context_dim=16, context_hidden=32,
             tied_derivatives=False, tf_fp32=False)


def flow_policy(conditional=False, dead=None, **flow):
    torch.manual_seed(0)
    return FlowPolicy(gfn_config(conditional, dead=dead), {**SMALL, **flow}, seed=0)


# -- the interface -------------------------------------------------------------------------------------------------

def _trainer_reads():
    torch_names = set(dir(torch.nn.Module))
    reads = set()
    for f in TRAINER_FILES:
        tree = ast.parse((ROOT / f).read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and ast.unparse(n.value) in RECEIVERS and n.attr not in torch_names:
                reads.add(n.attr)
    return reads


def test_the_interface_is_every_policy_read_in_the_trainer():
    reads = _trainer_reads()
    known = set(POLICY_ATTRIBUTES) | set(WRITTEN_STATE)
    assert reads <= known, f"the trainer reads policy attributes the interface lacks: {sorted(reads - known)}"
    stale = known - reads - set(GETATTR_ONLY)
    assert not stale, f"interface names the trainer no longer reads: {sorted(stale)}"


@pytest.mark.parametrize('conditional', [False, True])
def test_both_policies_answer_the_whole_interface(conditional):
    torch.manual_seed(0)
    sde = GFN(**gfn_config(conditional))
    assert missing_policy_attributes(sde) == []
    assert missing_policy_attributes(flow_policy(conditional)) == []


# -- geometry ----------------------------------------------------------------------------------------------------

def test_geometry_is_the_sde_policys():
    torch.manual_seed(0)
    sde = GFN(**gfn_config(dead=(9,)))
    pol = flow_policy(dead=(9,))
    assert torch.equal(pol.ang_mask, sde.ang_mask) and pol.dead_rows == sde.dead_rows
    assert torch.equal(pol._dead_values, sde._dead_values) and pol.live_dim == sde.live_dim
    x = torch.rand(5, 12) * 4 - 2
    assert torch.allclose(pol._wrap_ang(x), sde._wrap_ang(x)) and torch.allclose(pol._pin_dead(x), sde._pin_dead(x))


# -- one-step trajectories --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('conditional', [False, True])
def test_a_forward_trajectory_is_one_step_with_exact_log_q(conditional):
    pol = flow_policy(conditional, dead=(9,))
    n = 7
    c = torch.randn(n, 4) if conditional else None
    s0 = torch.zeros(n, 12)
    states, log_pfs, log_pbs, log_flow = pol.get_traj_fwd(s0, None, 0, c, None)
    assert states.shape == (n, 2, 12) and log_pfs.shape == (n, 1) and log_pbs.shape == (n, 1)
    assert log_flow.shape == (n, 2) and bool((log_pbs == 0).all()) and bool((log_flow[:, 1] == 0).all())
    x = states[:, 1]
    assert torch.equal(states[:, 0], s0)
    assert bool((x[:, pol.ang_idx].abs() <= 1).all())
    assert torch.allclose(x[:, 9], pol._dead_values[9].expand(n))                  # dead row pinned
    ref = pol.sampler.log_prob(x, c) if conditional else pol.sampler.log_prob(x)
    assert torch.allclose(log_pfs[:, 0], ref.detach(), atol=1e-5)
    assert log_pfs.requires_grad and not x.requires_grad                           # draw detached, log q live
    if conditional:
        assert pol._z_cal_embedding is not None and not pol._z_cal_embedding.requires_grad
    # the same row scored backward and by replay: the same log P_F, log P_B 0
    _, lpf_b, lpb_b, _ = pol.get_traj_bwd(x, None, c, None)
    _, lpf_r, lpb_r, _ = pol.get_traj_replay(states, None, c, None)
    assert torch.allclose(lpf_b, log_pfs, atol=1e-5) and torch.allclose(lpf_r, log_pfs, atol=1e-5)
    assert bool((lpb_b == 0).all()) and bool((lpb_r == 0).all())


def test_the_level_head_reads_detached_condition_features():
    pol = flow_policy(True)
    c = torch.randn(6, 4)
    _, log_pfs, _, log_flow = pol.get_traj_fwd(torch.zeros(6, 12), None, 0, c, None)
    log_flow[:, 0].sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in pol.conditions_embedding_model.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pol.flow_model.parameters()) or \
        all(float(p.abs().sum()) == 0 for p in pol.flow_model.parameters())   # a zero-initialised head may have no grad


def test_freeze_policy_trains_the_level_alone():
    pol = flow_policy(True)
    c = torch.randn(6, 4)
    _, log_pfs, _, log_flow = pol.get_traj_fwd(torch.zeros(6, 12), None, 0, c, None, freeze_policy=True)
    assert not log_pfs.requires_grad and log_flow.requires_grad


def test_scrambled_conditions_permute_whole_tiles():
    pol = flow_policy(True)
    c = torch.arange(6.0).reshape(6, 1).expand(6, 4).contiguous()
    torch.manual_seed(3)
    out = pol._scramble(c, 6, 2)
    tiles = out[:, 0].reshape(3, 2)
    assert all(int(t[0]) % 2 == 0 and t[1] == t[0] + 1 for t in tiles)   # each tile of 2 rows moved whole
    assert sorted(out[:, 0].tolist()) == sorted(c[:, 0].tolist())


def test_unsupported_sde_paths_refuse():
    pol = flow_policy(True)
    c = torch.randn(3, 4)
    with pytest.raises(NotImplementedError):
        pol.get_traj_fwd(torch.zeros(3, 12), None, 0, c, None, detach_traj=False)
    with pytest.raises(NotImplementedError):
        pol.get_traj_replay(torch.zeros(3, 2, 12), None, c, None, resample_last_k=1)
    with pytest.raises(NotImplementedError):
        pol.install_drift_force(object())
    pol.install_drift_force(None)


# -- parameter groups --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('conditional', [False, True])
def test_parameter_groups_cover_every_parameter_once(conditional):
    pol = flow_policy(conditional)
    groups = [pol.t_model, pol.s_model, pol.forward_policy, pol.backward_policy, pol.conditions_embedding_model,
              pol.flow_model]
    ids = [id(p) for g in groups for p in g.parameters()]
    assert len(ids) == len(set(ids)), "a parameter in two groups"
    assert set(ids) == {id(p) for p in pol.parameters()}, "a parameter in no group"
    torch.optim.Adam([{'params': g.parameters()} for g in groups], lr=1e-3)   # empty groups are accepted


def test_molecule_mode_reads_atom_tokens_from_the_batch():
    from types import SimpleNamespace
    cfg = gfn_config(True)
    cfg['conditions_type'] = 'molecule'
    torch.manual_seed(0)
    pol = FlowPolicy(cfg, {**SMALL, 'conditioner': 'transformer', 'tf_width': 32, 'tf_layers': 1, 'tf_heads': 2,
                           'atom_tokens': True, 'atom_node_dim': 16, 'atom_convs': 1}, seed=0)
    counts = torch.tensor([3, 5, 4])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    mb = SimpleNamespace(z=torch.randint(1, 9, (12,)), pos=torch.randn(12, 3),
                         batch=torch.repeat_interleave(torch.arange(3), counts), ptr=ptr, num_graphs=3)
    states, log_pfs, log_pbs, log_flow = pol.get_traj_fwd(torch.zeros(3, 12), None, 0, None, mb)
    assert states.shape == (3, 2, 12) and bool((log_pbs == 0).all())
    _, lpf_b, _, _ = pol.get_traj_bwd(states[:, 1], None, None, mb)
    assert torch.allclose(log_pfs, lpf_b, atol=1e-5)
    assert pol.flow_model(pol.get_condition_embedding(None, mb)).shape == (3,)
    with pytest.raises(ValueError, match='one graph per row'):
        pol.get_traj_bwd(torch.zeros(4, 12), None, None, mb)
