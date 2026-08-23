"""Gates for the set-structured policy head (models/set_policy.py).

Four properties, each written so that breaking it FAILS rather than degrades quietly:

  1. The output block layout matches what GFN.split_params actually slices. A per-token
     model produces the right numbers in the wrong order by default, and the symptom would
     be one coordinate's mean landing in another's log-variance -- finite, plausible, and
     wrong.
  2. Permutation equivariance in the coordinate table's storage order. That order is a
     spanning-tree artifact; a policy sensitive to it has learned an accident.
  3. Parameter count independent of dimension, which is the whole point -- a flat MLP's
     width is tied to one molecule.
  4. The pooling carries SIZE. A mean-only pool is size-blind and would still look fine on
     a fixed molecule.

    python -m pytest -q test_set_policy.py
"""
import numpy as np
import pytest
import torch

from energies.conformer_torsions import ConformerTorsions
from energies.dof_features import state_features, state_feature_names
from models.set_policy import SetPolicy, set_policy_for

T_DIM = 8


def _en(smi='CCC1CCCCC1', level='full'):
    return ConformerTorsions(smiles=smi, level=level, force_field='mmff',
                             log_temperature=0.0, device='cpu')


def _policy(en, out_per_token=2, **kw):
    torch.manual_seed(0)
    return set_policy_for(en, T_DIM, hidden_dim=16, layers=2,
                          out_per_token=out_per_token, **kw)


def test_output_layout_matches_split_params():
    """The blocks must be CONTIGUOUS per channel, which is what split_params assumes.

    Driven against the real ``GFN.split_params`` rather than a restatement of it, so a
    change to the layout there fails here instead of silently disagreeing.

    Tests ``_to_blocks`` on a SYNTHETIC per-token tensor rather than re-deriving the forward
    pass. An earlier version recomputed phi/pool/rho by hand and broke the moment the
    aggregation changed -- a test that mirrors the implementation is a copy of it, and
    passes or fails for reasons that have nothing to do with the contract it names.
    """
    from models.gfn import GFN
    en = _en()
    pol = _policy(en)
    b, d = 5, en.data_ndim

    # channel k is filled with the constant k, so a mis-ordered flatten is unmissable
    per_token = torch.stack([torch.full((b, d), float(k)) for k in range(2)], dim=-1)
    blocks = pol._to_blocks(per_token)
    assert blocks.shape == (b, 2 * d)

    gfn = GFN(dim=d, angular_mask=en.periodic_dims, s_emb_dim=8, conditions_dim=0,
              harmonics_dim=8, t_dim=T_DIM, device=torch.device('cpu'),
              learned_variance=False, conditional=False, learn_pb=False)
    mean, logvar, _, _ = gfn.split_params(blocks, torch.zeros(b, d))
    assert (mean == 0).all(), 'the mean block is not channel 0 of the per-token output'
    assert mean.shape == (b, d) and logvar.shape == (b, d)

    # and the live forward produces that same layout. Only the MEAN is a pass-through --
    # split_params scales, offsets and clips the log-variance block -- so the boundary is
    # pinned by perturbing one block and requiring the other not to move.
    out = pol(torch.rand(b, d) * 2 - 1, torch.zeros(b, T_DIM))
    assert out.shape == (b, 2 * d)
    m2, lv2, _, _ = gfn.split_params(out, torch.zeros(b, d))
    assert torch.equal(m2, out[:, :d]), 'the mean block is not the first d columns'

    bumped = out.clone()
    bumped[:, d:] += 1.0
    m3, lv3, _, _ = gfn.split_params(bumped, torch.zeros(b, d))
    assert torch.equal(m3, m2), 'touching the log-variance block moved the mean'
    gfn.learned_variance = True
    _, lv_a, _, _ = gfn.split_params(out, torch.zeros(b, d))
    _, lv_b, _, _ = gfn.split_params(bumped, torch.zeros(b, d))
    assert not torch.equal(lv_a, lv_b), 'the log-variance block is not read from columns d:'


def test_permutation_equivariance():
    """Relabel the coordinate table and the answer must follow, not change.

    Identity travels in the static features, so a permuted table is the same problem. This
    is the property the flat MLP cannot have and the reason for the whole module.
    """
    en = _en()
    d = en.data_ndim
    f = state_features(en)
    ang = list(en.periodic_dims)
    torch.manual_seed(1)
    base = SetPolicy(f, ang, T_DIM, hidden_dim=16, layers=2)

    rng = np.random.default_rng(0)
    p = rng.permutation(d)
    torch.manual_seed(1)
    perm = SetPolicy(f[p], [ang[i] for i in p], T_DIM, hidden_dim=16, layers=2)

    b = 4
    state = torch.rand(b, d) * 2 - 1
    t = torch.rand(b, T_DIM)
    o_base = base(state, t).reshape(b, 2, d)
    o_perm = perm(state[:, p], t).reshape(b, 2, d)
    assert torch.allclose(o_base[:, :, p], o_perm, atol=1e-5), \
        'permuting the coordinate table changed the answer; something reads the index'


def test_parameter_count_is_independent_of_dimension():
    """One model shape for molecules of different size -- the point of the exercise."""
    counts = {}
    for smi in ('CCCO', 'CCC1CCCCC1', 'CC(=O)NC(C)C(=O)NC'):
        en = _en(smi)
        pol = _policy(en)
        counts[smi] = (en.data_ndim, sum(q.numel() for q in pol.parameters()))
    dims = {v[0] for v in counts.values()}
    pars = {v[1] for v in counts.values()}
    assert len(dims) == 3, counts
    assert len(pars) == 1, (
        'parameter count varies with dimension {} -- the model is still tied to one '
        'molecule'.format(counts))


def test_pooling_carries_size():
    """A mean-only pool is size-blind. The sum branch must actually reach the output.

    Constructed as a counterfactual: duplicate every coordinate of a molecule into a
    doubled table. The mean pool is unchanged by that; the sum doubles. If the output is
    identical, the size information is not reaching rho.
    """
    en = _en()
    f = state_features(en)
    ang = list(en.periodic_dims)
    torch.manual_seed(2)
    small = SetPolicy(f, ang, T_DIM, hidden_dim=16, layers=2)
    torch.manual_seed(2)
    big = SetPolicy(np.concatenate([f, f]), ang + ang, T_DIM, hidden_dim=16, layers=2)

    b, d = 3, en.data_ndim
    state = torch.rand(b, d) * 2 - 1
    t = torch.rand(b, T_DIM)
    o_small = small(state, t).reshape(b, 2, d)[:, :, 0]
    o_big = big(torch.cat([state, state], -1), t).reshape(b, 2, 2 * d)[:, :, 0]
    assert not torch.allclose(o_small, o_big, atol=1e-4), \
        ('doubling the coordinate table did not change the output, so only the mean pool '
         'reaches rho and the model is blind to molecular size')


def test_angular_lift_is_masked_on_linear_coordinates():
    """sin/cos of a bond length is meaningless and must be zeroed, not merely present."""
    en = _en(level='full')
    pol = _policy(en)
    state = torch.rand(2, en.data_ndim) * 2 - 1
    tok = pol.tokens(state)
    nstat = len(state_feature_names())
    lift = tok[..., nstat + 1:nstat + 3]
    ang = torch.as_tensor(list(en.periodic_dims))
    assert (lift[:, ~ang] == 0).all(), 'the sin/cos lift leaked onto a linear coordinate'
    assert (lift[:, ang].abs().sum() > 0), 'the lift is zero everywhere; it is not wired'
    # and the flag that says which is which must be present and correct
    assert torch.allclose(tok[..., -1], ang.to(tok.dtype).expand(2, -1))


@pytest.mark.parametrize('level', ['torsion', 'dihedral', 'flex', 'full'])
def test_builds_at_every_tier(level):
    """Including 'torsion', where a state column is COLLECTIVE and features are aggregated."""
    en = _en(level=level)
    pol = _policy(en)
    b = 3
    out = pol(torch.rand(b, en.data_ndim) * 2 - 1, torch.rand(b, T_DIM))
    assert out.shape == (b, 2 * en.data_ndim)
    assert torch.isfinite(out).all()


def test_refuses_a_mismatched_state():
    en = _en()
    pol = _policy(en)
    with pytest.raises(ValueError, match='coordinates'):
        pol(torch.zeros(2, en.data_ndim + 1), torch.zeros(2, T_DIM))
    with pytest.raises(ValueError, match='angular_mask'):
        SetPolicy(state_features(en), list(en.periodic_dims)[:-1], T_DIM)


def test_aggregation_is_selective_as_well_as_extensive():
    """Augmented softmax: one channel must be able to EMPHASISE a coordinate.

    The extensive channel is covered by test_pooling_carries_size. This covers the other
    half, which a mean pool cannot supply: moving a single coordinate while holding the
    rest fixed must change the pooled context by more than a mean would allow. Driven by
    comparing the softmax weights themselves, since a mean is the degenerate case where
    every weight is 1/dim.
    """
    en = _en()
    pol = _policy(en)
    state = torch.zeros(1, en.data_ndim)
    h = pol.phi(pol.tokens(state))
    a = torch.softmax(pol.score(h), dim=1).squeeze(-1)
    assert torch.allclose(a.sum(-1), torch.ones(1), atol=1e-5), 'weights are not a softmax'
    spread = float(a.max() - a.min())
    assert spread > 1e-4, (
        'the softmax weights are uniform to {:.2e}, so the selective channel has collapsed '
        'to a mean and only the extensive channel is doing work'.format(spread))
