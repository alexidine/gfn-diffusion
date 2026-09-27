"""ConformerGFN's per-row carrier masking is the parent's arithmetic with the sum masked.

The identity used throughout: for any row mask m, ``masked(m) + masked(~m)`` equals the
parent's unmasked value. That pins both halves at once -- nothing dropped, nothing counted
twice -- without re-deriving the kernels. The P_B mixture is a REPLICA of the shared-file
method minus its final sum, so it is checked against the parent directly.

Two further contracts live here. DPLR on a carrier is REFUSED, because its Woodbury density
never reaches the mask. And the log Z(c) head can be swapped for one over the pooled
molecular embedding (`install_flow_head('mol_emb')`); those tests drive it through real
`get_traj_fwd` / `_bwd` / `_replay` calls, since `log_flow[:, 0]` there is what the losses
actually read.
"""
import numpy as np
import pytest
import torch

from energy_sampling.models.gfn import GFN
from energy_sampling.utils import get_gfn_optimizer, uniform_discretizer
from models.conformer_gfn import ConformerGFN

DIM = 9
MASK = [False] * 6 + [True] * 3          # r, theta linear; phi block wraps
BLOCKS = [0] * 3 + [1] * 3 + [2] * 3     # the energy's per-column _free_block code
B = 16
MOL = 12                                 # pooled molecular embedding width
COND = 3                                 # raw condition vector width
T = 3


def _gfn(pb_exact_reversal=True, seed=0, dplr_rank=0):
    torch.manual_seed(seed)
    g = GFN(dim=DIM, angular_mask=MASK, s_emb_dim=32, conditions_dim=1, harmonics_dim=8,
            t_dim=8, t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32,
            policy_layers=2, flow_hidden_dim=16, flow_layers=2, t_scale=1.0,
            learned_variance=True, learn_pb=True, conditional=False, device='cpu',
            zero_init=False, dplr_rank=dplr_rank, pb_exact_reversal=pb_exact_reversal).eval()
    g.__class__ = ConformerGFN
    g._mol_cond, g._state_mask, g._carrier = None, None, True
    return g


def _cond_gfn(seed=0, carrier=False, **kw):
    """A conditional GFN re-classed exactly as the modeller does it (so no ConformerGFN
    __init__ runs, and every new attribute must survive being absent)."""
    torch.manual_seed(seed)
    g = GFN(dim=DIM, angular_mask=MASK, s_emb_dim=32, conditions_dim=COND,
            condition_embedding_dim=8, conditions_type='vector', harmonics_dim=8, t_dim=8,
            t_hidden_dim=32, s_hidden_dim=32, s_layers=2, policy_hidden_dim=32,
            policy_layers=2, flow_hidden_dim=16, flow_layers=2, cond_hidden_dim=16,
            cond_layers=2, t_scale=1.0, learned_variance=True, learn_pb=True,
            conditional=True, device='cpu', zero_init=False, dplr_rank=0, **kw)
    g.__class__ = ConformerGFN
    g._mol_cond, g._state_mask, g._carrier = None, None, carrier
    return g


class _Batch:
    """Only the attributes the trajectory binding reads."""

    def __init__(self, n=B, **attrs):
        self.num_graphs = n
        for k, v in attrs.items():
            setattr(self, k, v)


def _log_z(g, condition, batch, seed=7):
    """log_flow[:, 0] off a real forward rollout: the value the TB losses read."""
    torch.manual_seed(seed)
    out = g.get_traj_fwd(torch.zeros(B, DIM), lambda n: uniform_discretizer(n, T), None,
                         condition, batch)
    return out[3][:, 0]


def _mask(seed=1):
    torch.manual_seed(seed)
    m = torch.rand(B, DIM) > 0.4
    m[:, 0] = True
    m[:, -1] = False                     # every row has a real and a pad column
    return m


def _pb_args(g):
    torch.manual_seed(2)
    prev, nxt = torch.rand(B, DIM) * 2 - 1, torch.rand(B, DIM) * 2 - 1
    dc = torch.rand(B, 1) * 0.1
    bmc = torch.rand(B, DIM) + 0.5
    bv = torch.rand(B, DIM) * 0.05 + 0.01
    t = torch.rand(B) * 0.8 + 0.1
    return prev, nxt, dc, bmc, bv, t


def test_gauss_logprob_splits_exactly_over_the_mask():
    g = _gfn()
    torch.manual_seed(3)
    dx, dr, v = torch.randn(B, DIM), torch.randn(B, DIM), torch.rand(B, DIM) + 0.1
    full = GFN.gauss_logprob(g, dx, dr, v)
    m = _mask()
    g._state_mask = m
    a = g.gauss_logprob(dx, dr, v)
    g._state_mask = ~m
    b = g.gauss_logprob(dx, dr, v)
    assert torch.allclose(a + b, full, atol=1e-5)
    g._state_mask = torch.ones_like(m)
    assert torch.allclose(g.gauss_logprob(dx, dr, v), full, atol=1e-6)


@pytest.mark.parametrize('exact', [True, False])
def test_pb_logprob_splits_exactly_over_the_mask(exact):
    g = _gfn(pb_exact_reversal=exact)
    args = _pb_args(g)
    full = GFN._pb_logprob(g, *args)
    m = _mask()
    g._state_mask = m
    a = g._pb_logprob(*args)
    g._state_mask = ~m
    b = g._pb_logprob(*args)
    assert torch.allclose(a + b, full, atol=1e-5)


def test_mixture_replica_equals_the_shared_method():
    g = _gfn()
    prev, nxt, dc, bmc, bv, t = _pb_args(g)
    ang = g.ang_idx
    x, y = g._wrap_ang(prev).index_select(1, ang), g._wrap_ang(nxt).index_select(1, ang)
    k, bs = bmc.index_select(1, ang), bv.index_select(1, ang)
    want = GFN._pb_mixture_ang_logprob(g, x, y, dc, k, bs, t)
    got = g._pb_mixture_ang_terms(x, y, dc, k, bs, t).sum(1)
    assert torch.allclose(got, want, atol=1e-6)


def test_pads_are_pinned_to_exact_zero_even_from_nan():
    g = _gfn()
    m = _mask()
    g._state_mask = m
    s = torch.randn(B, DIM)
    s[~m] = float('nan')
    out = g._pin_dead(s)
    assert torch.equal(out[~m], torch.zeros_like(out[~m]))
    assert torch.equal(out[m], s[m])


def test_no_mask_is_the_parent_unchanged():
    g = _gfn()
    torch.manual_seed(4)
    dx, dr, v = torch.randn(B, DIM), torch.randn(B, DIM), torch.rand(B, DIM) + 0.1
    assert torch.equal(g.gauss_logprob(dx, dr, v), GFN.gauss_logprob(g, dx, dr, v))
    s = torch.randn(B, DIM)
    assert g._pin_dead(s) is s


def test_a_mask_for_a_different_batch_is_refused():
    g = _gfn()
    g._state_mask = _mask()[:4]
    with pytest.raises(RuntimeError, match='does not match'):
        g._pin_dead(torch.randn(B, DIM))


def test_a_carrier_gfn_refuses_a_batch_without_state_mask():
    g = _gfn()

    class _Batch:
        num_graphs = B
    with pytest.raises(RuntimeError, match='state_mask'):
        g._bind_state_mask(_Batch())


# ------------------------------------------------------------------ DPLR on a carrier

def _dplr_args():
    torch.manual_seed(5)
    dx, dr = torch.randn(B, DIM), torch.randn(B, DIM)
    d, dt = torch.rand(B, DIM) + 0.1, torch.rand(B) * 0.1 + 0.01
    return dx, dr, d, dt, torch.randn(B, DIM, 2) * 0.1


def test_dplr_on_a_carrier_is_refused_in_fwd_gauss_logprob():
    g = _gfn(dplr_rank=2)
    dx, dr, d, dt, V = _dplr_args()
    g._state_mask = _mask()
    with pytest.raises(NotImplementedError, match='dplr_rank 2 on a CARRIER'):
        g.fwd_gauss_logprob(dx, dr, d, dt, V)
    # keyed on the carrier, not on this batch having a mask bound
    g._state_mask = None
    with pytest.raises(NotImplementedError, match='CARRIER'):
        g.fwd_gauss_logprob(dx, dr, d, dt, V)
    # the diagonal path is untouched: still the masked gauss_logprob
    g._state_mask = _mask()
    assert torch.equal(g.fwd_gauss_logprob(dx, dr, d, dt, None),
                       g.gauss_logprob(dx, dr, dt.unsqueeze(1) * d))


def test_dplr_refusal_is_reached_by_a_real_carrier_rollout():
    """The refusal must sit on the live path, not only on a direct call."""
    g = _gfn(dplr_rank=2)
    with pytest.raises(NotImplementedError, match='CARRIER'):
        g.get_traj_fwd(torch.zeros(B, DIM), lambda n: uniform_discretizer(n, T), None,
                       None, _Batch(state_mask=_mask()))


def test_dplr_off_a_carrier_is_the_parent_unchanged():
    g = _gfn(dplr_rank=2)
    g._carrier = False
    args = _dplr_args()
    assert torch.equal(g.fwd_gauss_logprob(*args), GFN.fwd_gauss_logprob(g, *args))


# ------------------------------------------------------------------ the log Z(c) head

def test_condition_head_is_the_default_and_installing_it_changes_nothing():
    g = _cond_gfn()
    assert g.flow_head_kind == 'condition'
    cond = torch.randn(B, COND)
    batch = _Batch(embedding=torch.randn(B, MOL))
    ref = _log_z(g, cond, batch)
    # the default read IS the parent's
    parent = GFN._condition_flow(g, g.get_condition_embedding(cond, None))
    assert torch.equal(ref, parent)

    head, rng = g.flow_model, torch.get_rng_state()
    g.install_flow_head('condition', mol_dim=MOL)
    assert g.flow_model is head
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(_log_z(g, cond, batch), ref)


def test_mol_emb_head_gives_one_log_z_per_row_and_trains_only_itself():
    g = _cond_gfn()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    assert g.flow_head_kind == 'mol_emb'
    # requires_grad on both inputs stands in for a LIVE encoder and conditioner: the head
    # must still send its gradient nowhere but its own parameters
    emb = torch.randn(B, MOL, requires_grad=True)
    cond = torch.randn(B, COND, requires_grad=True)
    log_z = _log_z(g, cond, _Batch(embedding=emb))
    assert log_z.shape == (B,)
    assert torch.isfinite(log_z).all()

    g.zero_grad(set_to_none=True)
    log_z.sum().backward()
    assert emb.grad is None, 'Z gradient reached the molecular embedding'
    assert cond.grad is None, 'Z gradient reached the raw condition'
    reached = {n for n, p in g.named_parameters() if p.grad is not None}
    assert reached, 'the new head did not train at all'
    assert all(n.startswith('flow_model.') for n in reached), sorted(reached)
    assert all(p.grad is not None for p in g.flow_model.parameters())


def test_mol_emb_head_is_named_under_flow_model_so_lr_flow_routing_applies():
    g = _cond_gfn()
    old = g.flow_model
    old_ids = {id(p) for p in old.parameters()}
    rng = torch.get_rng_state()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    assert torch.equal(torch.get_rng_state(), rng), 'install reseeded the caller RNG'

    new_ids = {id(p) for p in g.flow_model.parameters()}
    under = {id(p) for n, p in g.named_parameters() if n.startswith('flow_model.')}
    assert new_ids == under
    assert not old_ids & {id(p) for p in g.parameters()}, 'the old head is still registered'
    # an A/B differs in the input only
    assert g.flow_model.input_dim == MOL
    assert g.flow_model.n_layers == old.n_layers
    assert g.flow_model.fc_layers[0].out_features == old.fc_layers[0].out_features

    lr_flow = 0.0371
    opt = get_gfn_optimizer(g, lr_policy=1e-3, lr_flow=lr_flow)
    routed = {id(p) for grp in opt.param_groups if grp['lr'] == lr_flow for p in grp['params']}
    assert routed == new_ids


def test_mol_emb_head_counts_valid_columns_per_block_on_a_carrier():
    g = _cond_gfn(carrier=True)
    # a numpy int64 array, which is what the energies' `_free_block` is
    g.install_flow_head('mol_emb', mol_dim=MOL, column_blocks=np.asarray(BLOCKS))
    m, emb = _mask(), torch.randn(B, MOL)
    log_z = _log_z(g, torch.randn(B, COND), _Batch(embedding=emb, state_mask=m))
    assert log_z.shape == (B,)
    # z_calibration's regression mode re-feeds _z_cal_embedding to flow_model directly, so
    # it must be the head's OWN input for this rollout -- the one log_flow[:, 0] was read on
    x = g._z_cal_embedding
    want = torch.stack([m[:, :3].sum(1), m[:, 3:6].sum(1), m[:, 6:].sum(1)], 1)
    assert torch.equal(x[:, :MOL], emb)
    assert torch.equal(x[:, MOL:], want.to(x.dtype))
    assert torch.equal(g.flow_model(x).flatten(), log_z)


def _carrier_batch(seed):
    """B rows with their own pooled embedding and their own valid columns."""
    torch.manual_seed(100 + seed)
    return _Batch(embedding=torch.randn(B, MOL), state_mask=_mask(seed))


def _head_on(g, batch):
    """What the mol_emb head should read for `batch`, built from the batch alone."""
    m = batch.state_mask
    counts = torch.stack([m[:, :3].sum(1), m[:, 3:6].sum(1), m[:, 6:].sum(1)], 1)
    with torch.no_grad():
        return g.flow_model(torch.cat([batch.embedding, counts.to(batch.embedding.dtype)],
                                      1)).flatten()


def _run(g, entry, batch, seed=7):
    """log_flow[:, 0] off one real trajectory of the named kind: that branch's TB centre."""
    torch.manual_seed(seed)
    disc, cond, live = lambda n: uniform_discretizer(n, T), torch.randn(B, COND), batch.state_mask
    if entry == 'fwd':
        out = g.get_traj_fwd(torch.zeros(B, DIM), disc, None, cond, batch)
    elif entry == 'bwd':
        term = torch.where(live, torch.randn(B, DIM) * 0.3, torch.zeros(()))
        out = g.get_traj_bwd(term, disc, cond, batch)
    else:
        traj = torch.where(live.unsqueeze(1), torch.randn(B, T + 1, DIM) * 0.3, torch.zeros(()))
        traj[:, 0] = 0.0
        out = g.get_traj_replay(traj, disc, cond, batch)
    return out[3][:, 0]


@pytest.mark.parametrize('entry', ['fwd', 'bwd', 'replay'])
def test_mol_emb_head_reads_each_trajectorys_own_batch(entry):
    """Every entry point rebinds. Batch B has batch A's row count on purpose: that is what the
    branch batches of one training step look like, and it is the case the row check in
    `_condition_flow` cannot catch -- a stale binding would score B's rows as A's molecules."""
    g = _cond_gfn(carrier=True)
    g.install_flow_head('mol_emb', mol_dim=MOL, column_blocks=BLOCKS)
    a, b = _carrier_batch(1), _carrier_batch(2)
    _run(g, 'fwd', a)
    got = _run(g, entry, b)
    assert torch.allclose(got, _head_on(g, b), atol=1e-6)
    assert not torch.allclose(got, _head_on(g, a), atol=1e-3)


def test_a_trajectory_that_skips_the_rebind_is_refused_not_scored_on_the_last_batch():
    """The parent's entry point, reached without this subclass's binding, must not find the
    previous rollout's input still bound."""
    g = _cond_gfn(carrier=True)
    g.install_flow_head('mol_emb', mol_dim=MOL, column_blocks=BLOCKS)
    _run(g, 'fwd', _carrier_batch(1))
    b = _carrier_batch(2)
    g.bind_molecular_conditioning(b)     # the policy side is bound; the log Z input is not
    term = torch.where(b.state_mask, torch.randn(B, DIM) * 0.3, torch.zeros(()))
    with pytest.raises(RuntimeError, match='no batch bound'):
        GFN.get_traj_bwd(g, term, lambda n: uniform_discretizer(n, T), torch.randn(B, COND), b)


def test_mol_emb_head_refuses_a_bound_batch_of_another_size():
    g = _cond_gfn()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    g._bind_flow_input(_Batch(embedding=torch.randn(B, MOL)), torch.randn(B, COND))
    with pytest.raises(RuntimeError, match='not the one being scored'):
        g._condition_flow(torch.zeros(B - 4, g.condition_embedding_dim))


def test_mol_emb_head_refuses_the_conditioner_output():
    """train.py bootstrap_log_z calls flow_model(get_condition_embedding(...)) directly."""
    g = _cond_gfn()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    with pytest.raises(RuntimeError, match='bootstrap_log_z'):
        g.flow_model(g.get_condition_embedding(torch.randn(B, COND), None))


def test_mol_emb_head_refuses_a_batch_it_cannot_read():
    g = _cond_gfn()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    with pytest.raises(RuntimeError, match='embedding'):
        _log_z(g, torch.randn(B, COND), _Batch())
    with pytest.raises(RuntimeError, match='condition=False'):
        _log_z(g, False, _Batch(embedding=torch.randn(B, MOL)))
    with pytest.raises(RuntimeError, match='installed for'):
        _log_z(g, torch.randn(B, COND), _Batch(embedding=torch.randn(B, MOL + 1)))


@pytest.mark.parametrize('make, kw, err, match', [
    (lambda: _gfn(), dict(mol_dim=MOL), NotImplementedError, 'unconditional'),
    (lambda: _cond_gfn(full_flow=True), dict(mol_dim=MOL), NotImplementedError, 'full_flow'),
    (lambda: _cond_gfn(scalar_flow=True), dict(mol_dim=MOL), NotImplementedError,
     'LearnableScalar'),
    (lambda: _cond_gfn(), dict(mol_dim=None), ValueError, 'mol_dim'),
    (lambda: _cond_gfn(), dict(mol_dim=8), ValueError, 'condition_embedding_dim'),
    (lambda: _cond_gfn(), dict(mol_dim=MOL, column_blocks=[0, 1]), ValueError, 'column_blocks'),
])
def test_mol_emb_install_refusals(make, kw, err, match):
    with pytest.raises(err, match=match):
        make().install_flow_head('mol_emb', **kw)


def test_a_second_install_is_refused():
    g = _cond_gfn()
    g.install_flow_head('mol_emb', mol_dim=MOL)
    for kind in ('mol_emb', 'condition'):
        with pytest.raises(RuntimeError, match='already installed'):
            g.install_flow_head(kind, mol_dim=MOL)
    with pytest.raises(ValueError, match='one of'):
        _cond_gfn().install_flow_head('state', mol_dim=MOL)
