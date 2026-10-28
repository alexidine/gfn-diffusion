"""models/nf_policy.py::NFPolicy, a normalizing flow as a train.py policy (docs/design/flow_policy_port.md):
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

from models.nf_policy import NFPolicy  # noqa: E402
from models.gfn import GFN  # noqa: E402
from models.policy_interface import (GETATTR_ONLY, OPTIONAL_HOOKS, POLICY_ATTRIBUTES, RECEIVERS,  # noqa: E402
                                     TRAINER_FILES, WRITTEN_STATE, missing_policy_attributes)

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
    return NFPolicy(gfn_config(conditional, dead=dead), {**SMALL, **flow}, seed=0)


# -- the interface -------------------------------------------------------------------------------------------------

def _trainer_reads():
    torch_names = set(dir(torch.nn.Module()))   # an instance: its own state (training) is set in __init__
    reads = set()
    for f in TRAINER_FILES:
        tree = ast.parse((ROOT / f).read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and ast.unparse(n.value) in RECEIVERS and n.attr not in torch_names:
                reads.add(n.attr)
    return reads


def test_the_interface_is_every_policy_read_in_the_trainer():
    reads = _trainer_reads()
    known = set(POLICY_ATTRIBUTES) | set(WRITTEN_STATE) | set(OPTIONAL_HOOKS)
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
    pol = NFPolicy(cfg, {**SMALL, 'conditioner': 'transformer', 'tf_width': 32, 'tf_layers': 1, 'tf_heads': 2,
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


# -- the trainer's selection -----------------------------------------------------------------------------------------

def test_mk_dev_model_section_is_the_sde_policy_with_the_nf_defaults():
    import yaml
    from models.nf_policy import NF_DEFAULTS
    raw = yaml.safe_load((ROOT / 'configs' / 'mk_dev.yaml').read_text(encoding='utf-8'))
    assert raw['model']['kind'] == 'sde' and 'policy' not in raw
    assert raw['model']['nf'] == NF_DEFAULTS   # one set of defaults: the YAML block and the code agree


def test_build_policy_dispatches_and_leaves_an_sde_config_alone():
    from types import SimpleNamespace
    from models.nf_policy import build_policy
    from train import Modeller
    host = object.__new__(Modeller)
    host.args = SimpleNamespace(policy=SimpleNamespace(kind='sde', nf=SimpleNamespace(**SMALL)), seed=3)
    assert host._policy_entry() == {}                        # the SDE gfn_config gains no key
    host.args = SimpleNamespace(seed=3)
    assert host._policy_entry() == {}                        # nor without a policy block
    host.args = SimpleNamespace(policy=SimpleNamespace(kind='nf', nf=SimpleNamespace(**SMALL)), seed=3)
    entry = host._policy_entry()
    assert entry == {'policy': {'kind': 'nf', 'nf': SMALL, 'seed': 3}}
    torch.manual_seed(0)
    assert isinstance(build_policy(gfn_config()), GFN)
    assert isinstance(build_policy(gfn_config() | entry), NFPolicy)
    with pytest.raises(ValueError):
        build_policy(gfn_config() | {'policy': {'kind': 'other'}})


@pytest.mark.parametrize('conditional', [False, True])
def test_bounded_rows_linear_bounds_every_live_unwrapped_row(conditional):
    """Both flows draw inside [-1, 1] on every row (the FD chart's box: a conditional FD draw past a face built a
    degenerate crystal), and a draw maps back to its own base point."""
    pol = flow_policy(conditional, dead=(9,), bounded_rows='linear')
    wrap = pol.ang_mask.tolist()
    assert pol.sampler.bounded_rows == [i for i in range(12) if i != 9 and not wrap[i]]
    states, *_ = pol.get_traj_fwd(torch.zeros(64, 12), None, 0, torch.randn(64, 4) if conditional else None, None)
    assert bool((states[:, 1].abs() <= 1 + 1e-6).all())
    flow = pol.sampler.double()   # in double, so the round trip is exact (float32 inverses miss by ~1e-3 here)
    with torch.no_grad():   # a non-trivial map: every block's parameters away from the identity
        for p in flow.parameters():
            p.add_(0.3 * torch.randn_like(p))
    c = torch.randn(256, 4, dtype=torch.float64) if conditional else None
    x, z0 = flow.sample(256, c, return_base=True)
    assert bool((x.abs() <= 1).all())
    z, _ = flow.to_base(x, c)
    assert torch.allclose(z, z0, atol=1e-9)


# -- the config audit's fixes (2026-10-27) -------------------------------------------------------------------------

def test_unset_settings_resolve_per_route_as_the_inn_does():
    from models.nf_policy import NF_DEFAULTS, Z1_ORDER, resolve_flow_settings
    u = resolve_flow_settings(NF_DEFAULTS, False, list(range(12)))
    assert (u['bounded_rows'], u['min_bin_height'], u['tied_derivatives'], u['tied_factor'], u['block_orders']) == \
        ('linear', 1e-7, True, 2.0, Z1_ORDER)
    assert resolve_flow_settings(NF_DEFAULTS, False, [0, 1, 2])['block_orders'] is None   # not the Z' = 1 latent
    c = resolve_flow_settings(NF_DEFAULTS, True, list(range(12)))
    assert (c['bounded_rows'], c['min_bin_height'], c['tied_derivatives'], c['block_orders']) == ('linear', None, False, None)
    n = resolve_flow_settings({**NF_DEFAULTS, 'bounded_rows': 'none', 'block_orders': 'none'}, False, list(range(12)))
    assert n['bounded_rows'] is None and n['block_orders'] is None


def test_scrambled_conditions_train_the_blocks_and_not_the_condition_encoder():
    pol = flow_policy(True)
    x = torch.rand(6, 12) * 2 - 1
    _, log_pfs, _, _ = pol.get_traj_bwd(x, None, torch.randn(6, 4), None, scramble_condition_tiles=2)
    log_pfs.sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in pol.conditions_embedding_model.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in pol.forward_policy.parameters())


def test_freezing_p_b_on_a_policy_without_one_is_a_no_op(capsys):
    from copy import deepcopy
    from train import Modeller
    host = object.__new__(Modeller)
    host.gfn_model = flow_policy(True)
    host.ema_model = deepcopy(host.gfn_model)
    for mode in ('full', 'head', None):
        host.set_pb_freeze(mode)
    assert 'no learned P_B' in capsys.readouterr().out


def test_a_resume_refuses_a_checkpoint_of_another_policy():
    from types import SimpleNamespace
    from checkpointing import Checkpointer
    ck = object.__new__(Checkpointer)
    flow_entry = {'policy': {'kind': 'nf', 'nf': SMALL, 'seed': 1}}
    ck.modeller = SimpleNamespace(_policy_entry=lambda: flow_entry, args=SimpleNamespace(model=SimpleNamespace()))
    with pytest.raises(ValueError, match="policy in this run's config"):
        ck._gfn_config_from({'gfn_config': gfn_config()})                       # an SDE checkpoint, a flow run
    ck.modeller = SimpleNamespace(_policy_entry=lambda: {}, args=SimpleNamespace(model=SimpleNamespace()))
    with pytest.raises(ValueError, match="policy in this run's config"):
        ck._gfn_config_from({'gfn_config': gfn_config() | flow_entry})          # a flow checkpoint, an SDE run


def test_flow_runs_refuse_sde_only_settings_at_load():
    import copy
    import yaml
    import config_invariants as ci
    cfg = ci.canonical_model_section(yaml.safe_load((ROOT / 'configs' / 'mk_dev.yaml').read_text(encoding='utf-8')))
    rule = ci.nf_policy_refuses_sde_only_settings
    assert rule(cfg) == []                                                       # kind sde: nothing judged
    f = copy.deepcopy(cfg)
    f['policy']['kind'] = 'nf'
    assert [v.detail for v in rule(f)] and all('gradient_norm_clip' in v.detail for v in rule(f))
    f['gradient_norm_clip'] = 10.0
    for proto in ('unconditional_tb', 'conditional_vargrad'):                   # both canonical protocols are clean
        f['protocol'] = proto
        assert rule(f) == []
    g = copy.deepcopy(f)
    g['compile_policy'] = True
    g['protocols'][g['protocol']]['stages'][0].setdefault('loss_coeffs', {})['bwd'] = {'tbc': 1.0, 'repeats': 4}
    g['protocols'][g['protocol']]['stages'][0].setdefault('on_enter', []).append('set_traj_checkpoint:true')
    found = ' | '.join(v.detail for v in rule(g))
    for needle in ('compile_policy', 'bwd tbc', 'bwd repeats', 'set_traj_checkpoint'):
        assert needle in found, needle



def test_the_loader_resolves_a_sectioned_model_block():
    import config_invariants as ci
    raw = {'seed': 1, 'model': {'kind': 'nf', 'shared': {'periodic_centroids': True},
                                'sde': {'dplr_rank': 6}, 'nf': {'n_bins': 8}}}
    out = ci.canonical_model_section(raw)
    assert out['model'] == {'periodic_centroids': True, 'dplr_rank': 6}
    assert out['policy'] == {'kind': 'nf', 'nf': {'n_bins': 8}}
    assert ci.canonical_model_section(out) == out                       # the resolved form passes through
    flat = {'model': {'dplr_rank': 6}}
    assert ci.canonical_model_section(flat) is flat                     # an unsectioned config untouched
    for bad in ({'model': {'kind': 'other'}},
                {'model': {'kind': 'sde', 'extra': {}}},
                {'model': {'kind': 'sde', 'shared': {'a': 1}, 'sde': {'a': 2}}},
                {'model': {'kind': 'sde'}, 'policy': {'kind': 'nf'}}):
        with pytest.raises(ValueError):
            ci.canonical_model_section(bad)


def test_mk_dev_resolves_to_the_unsectioned_keys_the_sde_code_reads():
    from utils import load_yaml
    cfg = load_yaml(str(ROOT / 'configs' / 'mk_dev.yaml'))
    assert cfg['policy']['kind'] == 'sde'
    for key in ('hold_dead_latent_rows', 'periodic_centroids', 'dplr_rank', 's_emb_dim', 'full_flow'):
        assert key in cfg['model'], key


# -- the trainer's losses on a one-step trajectory -------------------------------------------------------------------

def _bwd_coeffs(**on):
    from types import SimpleNamespace
    base = dict(freeze_policy=0, freeze_z=0, traj_grads=1, beta=10.0, vg_lb=0, vg_lme=0, pf_boost=0, db=0, subtb=0,
                emp_z=0, tb=0, mle=0, loss_clip=-1)
    return SimpleNamespace(**{**base, **on})


@pytest.mark.parametrize('repeats', [1, 3])
def test_the_backward_losses_are_their_hand_values_on_one_step(repeats):
    """gflownet_losses.get_gfn_backward_loss on the NF policy: log P_F = log q(x | c), log P_B = 0, Z = the level head,
    so terminal MLE is -log q (the Jensen bound at K = 1; the DReG 'exact' estimator over K identical deterministic
    backward paths at K > 1), TB is beta * smooth_l1(log q + Z - log r), and the live stash the pooled VarGrad term
    reads carries log q, 0 and log r."""
    import torch.nn.functional as F
    from gflownet_losses import get_gfn_backward_loss
    pol = flow_policy(True, dead=(9,))
    n = 5
    c = torch.randn(n, 4).repeat_interleave(repeats, 0)
    with torch.no_grad():
        x = pol.get_traj_fwd(torch.zeros(n, 12), None, 0, c[::repeats], None)[0][:, 1]
    x = x.repeat_interleave(repeats, 0)
    log_r = torch.randn(n * repeats) * 5
    log_q = pol.sampler.log_prob(x, c).detach()
    z = pol.get_traj_bwd(x, None, c, None)[3][:, 0].detach()

    loss, d = get_gfn_backward_loss(_bwd_coeffs(mle=1), x, pol, log_r, None, None, condition=c, repeats=repeats,
                                    report_losses=True)
    assert torch.allclose(d['log_pf'], log_q, atol=1e-5) and bool((d['log_pb'] == 0).all())
    assert torch.allclose(d['mle'], -log_q.mean(), atol=1e-5) and torch.allclose(loss, -log_q.mean(), atol=1e-5)

    beta = 10.0
    tb_hand = beta * F.smooth_l1_loss(log_q + z - log_r, torch.zeros_like(log_r), reduction='none', beta=beta)
    pol._stash_live_branches = True
    loss, d = get_gfn_backward_loss(_bwd_coeffs(tb=1, beta=beta), x, pol, log_r, None, None, condition=c,
                                    repeats=repeats, report_losses=True, condition_id=torch.arange(n * repeats))
    assert torch.allclose(loss, tb_hand.mean(), atol=1e-4)
    live = pol._live_bwd
    assert torch.allclose(live['log_pf'].detach(), log_q, atol=1e-5) and bool((live['log_pb'] == 0).all())
    assert torch.equal(live['log_r'], log_r) and live['log_pf'].requires_grad


def test_the_forward_tb_is_its_hand_value_on_one_step():
    """gflownet_losses.get_gfn_forward_loss on the NF policy: one draw x ~ q(. | c), log r scored on it, and TB =
    beta * smooth_l1(log q(x | c) + Z - log r), with log q the flow's own density of the row it drew."""
    import torch.nn.functional as F
    from gflownet_losses import get_gfn_forward_loss
    pol = flow_policy(True, dead=(9,))
    n, beta = 6, 10.0
    c = torch.randn(n, 4)
    seen = {}

    def log_reward_fn(x, mol_batch, log_T, return_exp, keep_grads=False):
        seen['x'] = x.detach().clone()
        return -(x ** 2).sum(-1) * 3

    coeffs = _bwd_coeffs(tb=1, beta=beta, traj_grads=0, reward_grads=0, pooled_source=None, coeff_matrix=None)
    loss, d = get_gfn_forward_loss(coeffs, torch.zeros(n, 12), pol, log_reward_fn, None, None, torch.zeros(n),
                                   condition=c, report_losses=True)
    x = seen['x']
    log_q = pol.sampler.log_prob(x, c).detach()
    z = pol.get_traj_bwd(x, None, c, None)[3][:, 0].detach()
    log_r = -(x ** 2).sum(-1) * 3
    tb_hand = beta * F.smooth_l1_loss(log_q + z - log_r, torch.zeros_like(log_r), reduction='none', beta=beta)
    assert torch.allclose(loss, tb_hand.mean(), atol=1e-4)


def test_a_state_dict_round_trip_restores_the_flow_and_its_scalers(tmp_path):
    """A resume rebuilds the policy from its gfn_config and loads the state dict (checkpointing.py): the fitted
    standardisers ride in it (scalers_fitted stays True, so train.py's fit_policy_data_scalers does not refit them on
    the resumed run's data), and the restored flow scores every row as the saved one did."""
    pol = flow_policy(True, dead=(9,))
    torch.manual_seed(1)
    pol.fit_data_scalers(torch.randn(200, 12) * 3 + 1, torch.randn(50, 4) * 2)
    c = torch.randn(8, 4)
    with torch.no_grad():
        x = pol.get_traj_fwd(torch.zeros(8, 12), None, 0, c, None)[0][:, 1]
        ref = pol.sampler.log_prob(x, c)
    torch.save(pol.state_dict(), tmp_path / 'p.pt')

    fresh = NFPolicy(gfn_config(True, dead=(9,)), SMALL, seed=7)
    assert not bool(fresh.scalers_fitted)
    fresh.load_state_dict(torch.load(tmp_path / 'p.pt'))
    assert bool(fresh.scalers_fitted)
    with torch.no_grad():
        assert torch.allclose(fresh.sampler.log_prob(x, c), ref, atol=1e-5)


@pytest.mark.parametrize('conditional', [False, True])
def test_log_density_is_the_backward_steps_log_pf(conditional):
    """train.py::Modeller.density_readouts reads policy.log_density: the same exact log q the backward step trains,
    with no graph built."""
    pol = flow_policy(conditional)
    x = torch.rand(8, 12) * 2 - 1
    c = torch.randn(8, 4) if conditional else None
    _, lpf, _, _ = pol.get_traj_bwd(x, None, c, None)
    lq = pol.log_density(x, c, None)
    assert torch.allclose(lq, lpf[:, 0].detach(), atol=1e-5) and not lq.requires_grad
