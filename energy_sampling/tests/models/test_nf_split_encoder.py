"""The split molecule encoder of the conditional flow policy (models/spline_flow.py::SplitAtomEncoder, NFPolicy
atom_tokens 'split' / 'split_pooled') and the condition columns that ride beside the molecule in molecule mode.

Contracts:
- the invariant pool s, and with it the level head's input, does not move when the molecule is rotated or reflected;
  the point pool p does (it carries the frame the orientation rows rotate from);
- the space-group, temperature and Z' columns reach log q and the level head's input in molecule mode (they were
  dropped before: models/nf_policy.py::_context returned the row ids alone);
- both new modes draw and score one density: log q of a draw equals log q re-scored, and the per-row terms of
  eval/nf_fit.py sum to it."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval.nf_fit import rows_given_ctx  # noqa: E402
from models.nf_policy import NFPolicy  # noqa: E402
from models.spline_flow import SplitAtomEncoder  # noqa: E402
from test_nf_policy import SMALL, gfn_config  # noqa: E402

pytestmark = pytest.mark.fast
TF = dict(conditioner='transformer', tf_width=32, tf_layers=1, tf_heads=2, tf_fp32=True, atom_node_dim=16,
          atom_convs=2)


def _rotation(seed, proper=True):
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(seed)))
    if (torch.det(q) > 0) != proper:
        q[:, 0] = -q[:, 0]
    return q


def _mols(n=3, seed=0, rot=None):
    g = torch.Generator().manual_seed(seed)
    counts = torch.tensor([3, 5, 4][:n])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    pos = torch.randn(int(counts.sum()), 3, generator=g) * 1.5
    if rot is not None:
        pos = pos @ rot.T
    return SimpleNamespace(z=torch.randint(1, 9, (int(counts.sum()),), generator=g), pos=pos,
                           batch=torch.repeat_interleave(torch.arange(n), counts), ptr=ptr, num_graphs=n)


def _policy(mode, cond_dim=4):
    cfg = gfn_config(True, cond_dim=cond_dim)
    cfg['conditions_type'] = 'molecule'
    torch.manual_seed(0)
    pol = NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': mode}, seed=0)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():   # off the zero-initialised heads, so the condition can move log q and the level
        for p in pol.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return pol.eval()


@pytest.mark.parametrize('proper', [True, False])
def test_the_invariant_pool_ignores_the_frame_and_the_point_pool_does_not(proper):
    torch.manual_seed(0)
    enc = SplitAtomEncoder(32, node_dim=16, convs=2)
    z = torch.randint(1, 9, (2, 6))
    z[1, 4:] = 0
    n = torch.tensor([6, 4])
    pos = torch.randn(2, 6, 3) * 1.5
    rot = _rotation(1, proper)
    _, _, s0, p0 = enc(z, pos, n)
    _, _, s1, p1 = enc(z, pos @ rot.T, n)
    assert torch.allclose(s0, s1, atol=1e-5), float((s0 - s1).abs().max())
    assert not torch.allclose(p0, p1, atol=1e-3)


@pytest.mark.parametrize('mode', ['split', 'split_pooled'])
def test_the_level_head_reads_an_invariant_and_log_q_reads_the_frame(mode):
    pol = _policy(mode)
    cond = torch.randn(3, 4)
    rot = _rotation(2)
    mb, mb_rot = _mols(), _mols(rot=rot)
    with torch.no_grad():
        e0, e1 = pol.get_condition_embedding(cond, mb), pol.get_condition_embedding(cond, mb_rot)
        assert torch.allclose(e0, e1, atol=1e-5), float((e0 - e1).abs().max())
        assert torch.allclose(pol.flow_model(e0), pol.flow_model(e1), atol=1e-5)
        x = pol.get_traj_fwd(torch.zeros(3, 12), None, 0, cond, mb)[0][:, 1]
        assert not torch.allclose(pol.log_density(x, cond, mb), pol.log_density(x, cond, mb_rot), atol=1e-4)


@pytest.mark.parametrize('mode', [True, 'split', 'split_pooled'])
def test_condition_columns_reach_log_q_and_the_level_input_in_molecule_mode(mode):
    pol = _policy(mode)
    mb = _mols()
    a, b = torch.randn(3, 4), torch.randn(3, 4)
    with torch.no_grad():
        x = pol.get_traj_fwd(torch.zeros(3, 12), None, 0, a, mb)[0][:, 1]
        assert not torch.allclose(pol.log_density(x, a, mb), pol.log_density(x, b, mb), atol=1e-4)
        assert not torch.allclose(pol.get_condition_embedding(a, mb), pol.get_condition_embedding(b, mb))


@pytest.mark.parametrize('mode', ['split', 'split_pooled'])
def test_a_draw_and_its_rescore_are_one_density(mode):
    pol = _policy(mode)
    assert (pol.sampler.flow.blocks[0].made.cross is (mode == 'split'))
    mb, cond = _mols(), torch.randn(3, 4)
    with torch.no_grad():
        states, log_pfs, _, _ = pol.get_traj_fwd(torch.zeros(3, 12), None, 0, cond, mb)
        _, lpf_b, _, _ = pol.get_traj_bwd(states[:, 1], None, cond, mb)
        assert torch.allclose(log_pfs, lpf_b, atol=1e-5)
        c = pol._context(cond, mb, 3)
        x = pol._pin_dead(pol._wrap_ang(states[:, 1]))
        terms = rows_given_ctx(pol.sampler, x, pol.sampler.flow.context_input(c)[0])
        assert torch.allclose(terms.sum(-1), log_pfs.flatten(), atol=1e-4)


def test_molecule_mode_standardises_the_passed_columns_from_the_conditions_it_was_fitted_on():
    pol = _policy('split', cond_dim=2)
    conds = torch.randn(256, 2) * torch.tensor([3.0, 0.1]) + torch.tensor([5.0, -2.0])
    pol.fit_data_scalers(torch.rand(64, 12) * 1.6 - 0.8, conds)
    f = pol.sampler.flow
    assert float(f.c_mu[0]) == 0.0   # the id column is never standardised
    c = pol._context(conds[:3], _mols(), 3)
    assert torch.allclose(c[:, 1:], (conds[:3] - conds.mean(0)) / conds.std(0), atol=1e-5)
    assert torch.equal(c[:, 0], torch.arange(3, dtype=torch.float32))


def test_an_unknown_atom_token_mode_is_refused():
    cfg = gfn_config(True)
    cfg['conditions_type'] = 'molecule'
    with pytest.raises(ValueError, match='atom_tokens'):
        NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': 'spilt'}, seed=0)


# -- nf mol_scalars and atom_point_convs (pending settings) ----------------------------------------------------------

def _with_fields(mb, seed=0):
    g = torch.Generator().manual_seed(seed)
    mb.mol_volume = torch.rand(mb.num_graphs, generator=g) * 100 + 80
    mb.radius = torch.rand(mb.num_graphs, generator=g) * 2 + 2.5
    return mb


@pytest.mark.parametrize('proper', [True, False])
def test_molecule_scalars_are_invariant_and_read_by_the_level_and_log_q(proper):
    cfg = gfn_config(True, cond_dim=4)
    cfg['conditions_type'] = 'molecule'
    torch.manual_seed(0)
    pol = NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': 'split', 'mol_scalars': True}, seed=0)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in pol.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    pol.eval()
    cond = torch.randn(3, 4)
    a = pol.get_condition_embedding(cond, _with_fields(_mols()))
    sc = pol.sampler.flow.atom_scalars.clone()
    b = pol.get_condition_embedding(cond, _with_fields(_mols(rot=_rotation(2, proper))))
    assert torch.allclose(sc, pol.sampler.flow.atom_scalars, atol=1e-5)
    assert torch.allclose(a, b, atol=1e-5)
    x = torch.rand(3, 12) * 1.6 - 0.8
    with torch.no_grad():
        lq = pol.log_density(x, cond, _with_fields(_mols()))
        mb = _with_fields(_mols())
        mb.mol_volume = mb.mol_volume * 2
        assert not torch.allclose(pol.get_condition_embedding(cond, mb), a)
        assert not torch.allclose(pol.log_density(x, cond, mb), lq)
    with pytest.raises(ValueError, match='mol_volume'):
        pol.get_condition_embedding(cond, _mols())


def test_point_convs_zero_is_a_per_atom_stream_and_needs_the_split_encoder():
    cfg = gfn_config(True, cond_dim=4)
    cfg['conditions_type'] = 'molecule'
    pol = NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': 'split', 'atom_point_convs': 0}, seed=0)
    assert len(pol.sampler.flow.atom_encoder.point.interaction_blocks) == 0
    assert len(pol.sampler.flow.atom_encoder.gnn.interaction_blocks) == TF['atom_convs']
    cond = torch.randn(3, 4)
    with torch.no_grad():
        assert torch.isfinite(pol.log_density(torch.rand(3, 12) * 1.6 - 0.8, cond, _mols())).all()
    with pytest.raises(ValueError, match='atom_point_convs'):
        NFPolicy(cfg, {**SMALL, **TF, 'atom_tokens': True, 'atom_point_convs': 1}, seed=0)
