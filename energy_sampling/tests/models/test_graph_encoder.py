"""Gates for models/graph_encoder.py, with and without the attention half.

The dangerous surface here is the dense batching that attention needs. Message passing is
batch-composition-independent for free -- it only ever touches real edges -- but attention
over a PADDED dense tensor is not, and a mask bug produces embeddings that silently depend on
which other molecules happened to share the batch. That failure is invisible in a loss curve
and would poison every cached ``{f_j}``, so it gets the strongest test here.

    python -m pytest -q tests/models/test_graph_encoder.py
"""
import os
import sys

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import numpy as np
import pytest
import torch

from energy_sampling.models.graph_encoder import (
    MPNNEncoder, SPDAttention, dense_spd_batch, from_dense_batch, to_dense_batch,
)
from energy_sampling.models.graph_encodings import (
    graph_from_smiles, shortest_paths,
)

NODE_DIM, EDGE_DIM, HIDDEN = 9, 5, 32


def _graph(smiles, seed=0):
    """A minimal (x, edge_index, edge_attr, spd) bundle -- features are random but fixed."""
    z, e1, _ = graph_from_smiles(smiles)
    n = len(z)
    g = torch.Generator().manual_seed(seed)
    ei = torch.as_tensor(np.concatenate([e1, e1[::-1]], axis=1), dtype=torch.long)
    return dict(n=n,
                x=torch.rand(n, NODE_DIM, generator=g),
                edge_index=ei,
                edge_attr=torch.rand(ei.shape[1], EDGE_DIM, generator=g),
                spd=shortest_paths(e1, n))


def _path_edges(n):
    """P_n as a [2, n-1] edge list."""
    return np.array([list(range(n - 1)), list(range(1, n))])


def _batch(graphs):
    off, xs, eis, eas, batch = 0, [], [], [], []
    for gi, g in enumerate(graphs):
        xs.append(g['x'])
        eis.append(g['edge_index'] + off)
        eas.append(g['edge_attr'])
        batch.append(torch.full((g['n'],), gi, dtype=torch.long))
        off += g['n']
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas), torch.cat(batch),
            len(graphs), max(g['n'] for g in graphs))


def _encoder(attention, seed=0, layers=2):
    torch.manual_seed(seed)
    return MPNNEncoder(NODE_DIM, EDGE_DIM, hidden=HIDDEN, layers=layers,
                       attention=attention, n_heads=4, max_spd=6).eval()


# ------------------------------------------------------------------- dense batching

def test_dense_batch_round_trips():
    h = torch.arange(12, dtype=torch.float32).view(6, 2)
    batch = torch.tensor([0, 0, 0, 1, 2, 2])
    dense, mask = to_dense_batch(h, batch, 3)
    assert dense.shape == (3, 3, 2) and mask.shape == (3, 3)
    assert mask.sum(1).tolist() == [3, 1, 2]
    assert torch.equal(from_dense_batch(dense, batch, 3), h)


def test_padding_slots_are_masked_not_merely_zero():
    """A zero row is a legal embedding; the mask is what says 'no atom here'."""
    h = torch.ones(3, 2)
    batch = torch.tensor([0, 1, 1])
    _, mask = to_dense_batch(h, batch, 2)
    assert mask.tolist() == [[True, False], [True, True]]


# -------------------------------------------------------------------------- SPD bias

def test_unreachable_gets_its_own_bucket_not_the_far_bucket():
    """'Disconnected' is a different statement from 'a long way away'."""
    att = SPDAttention(HIDDEN, n_heads=2, max_spd=6)
    spd = torch.tensor([[[0, 3, 9, -1]]]).view(1, 1, 4).expand(1, 4, 4).clone()
    b = att.bucket(spd)
    assert b[0, 0, 0].item() == 0
    assert b[0, 0, 1].item() == 3
    assert b[0, 0, 2].item() == 6, 'a distance past max_spd should saturate, not wrap'
    assert b[0, 0, 3].item() == 7, 'unreachable must not share the saturated bucket'


def test_dense_spd_batch_pads_with_unreachable():
    a = np.array([[0, 1], [1, 0]])
    out = dense_spd_batch([a], 1, 4)
    assert out[0, :2, :2].tolist() == [[0, 1], [1, 0]]
    assert (out[0, 2:, :] == -1).all() and (out[0, :, 2:] == -1).all()


def test_oversized_graph_is_refused():
    with pytest.raises(ValueError, match='dense length'):
        dense_spd_batch([np.zeros((5, 5), dtype=int)], 1, 3)


# ---------------------------------------------------------- batch-composition leakage

@pytest.mark.parametrize('attention', [False, True])
def test_embedding_does_not_depend_on_batch_composition(attention):
    """THE test on this file.

    One molecule's per-atom embedding must be identical whether it is encoded alone or
    alongside others. Message passing gets this for free; dense attention does not, and a
    mask bug here would make every cached {f_j} a function of the batch it was built in.
    """
    enc = _encoder(attention)
    a, b, c = _graph('CCO', 1), _graph('c1ccccc1', 2), _graph('NCC(=O)NCC(=O)O', 3)

    def run(graphs):
        x, ei, ea, batch, ng, length = _batch(graphs)
        spd = (dense_spd_batch([g['spd'] for g in graphs], ng, length) if attention
               else None)
        with torch.no_grad():
            h, g = enc(x, ei, ea, batch, ng, spd=spd)
        return h, g

    solo_h, solo_g = run([a])
    for company in ([a, b], [b, a], [c, a, b], [b, c, a]):
        h, g = run(company)
        i = company.index(a)
        start = sum(x['n'] for x in company[:i])
        got_h = h[start:start + a['n']]
        assert torch.allclose(solo_h, got_h, atol=1e-5), \
            f'per-atom embedding moved when batched with {len(company)} molecules'
        assert torch.allclose(solo_g[0], g[i], atol=1e-5), \
            'pooled embedding moved with batch composition'


# ---------------------------------------------------------------------- equivariance

@pytest.mark.parametrize('attention', [False, True])
def test_atom_order_permutes_the_output_and_changes_nothing_else(attention):
    """Storage order is an accident of the SMILES parse; an encoder sensitive to it has
    learned the accident."""
    enc = _encoder(attention)
    g = _graph('NCC(=O)NCC(=O)O', 4)
    n = g['n']
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    inv = torch.argsort(perm)

    def run(x, ei, ea, spd):
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        dspd = dense_spd_batch([spd], 1, x.shape[0]) if attention else None
        with torch.no_grad():
            return enc(x, ei, ea, batch, 1, spd=dspd)

    h0, g0 = run(g['x'], g['edge_index'], g['edge_attr'], g['spd'])
    h1, g1 = run(g['x'][perm], inv[g['edge_index']], g['edge_attr'],
                 g['spd'][np.ix_(perm.numpy(), perm.numpy())])
    assert torch.allclose(h0[perm], h1, atol=1e-5), 'not permutation equivariant'
    assert torch.allclose(g0, g1, atol=1e-5), 'pooled vector is not permutation invariant'


# ------------------------------------------------------------------------- contracts

def test_spd_on_a_broadcast_encoder_is_refused_not_ignored():
    """Silently dropping the bias would mislabel an arm and report a null for the wrong
    reason."""
    enc = _encoder(attention=False)
    g = _graph('CCO', 5)
    x, ei, ea, batch, ng, length = _batch([g])
    with pytest.raises(ValueError, match='attention=False'):
        enc(x, ei, ea, batch, ng, spd=dense_spd_batch([g['spd']], ng, length))


def test_wrong_spd_shape_is_refused():
    enc = _encoder(attention=True)
    g = _graph('CCO', 6)
    x, ei, ea, batch, ng, length = _batch([g])
    with pytest.raises(ValueError, match=r'spd must be'):
        enc(x, ei, ea, batch, ng, spd=torch.zeros(1, length + 2, length + 2,
                                                  dtype=torch.long))


def test_attention_changes_the_answer():
    """A separator for the battery: if the attention arm were silently inert, every Block C
    comparison would come back null and read as 'the broadcast was sufficient'."""
    g = _graph('NCC(=O)NCC(=O)NCC(=O)O', 7)
    x, ei, ea, batch, ng, length = _batch([g])
    spd = dense_spd_batch([g['spd']], ng, length)
    with torch.no_grad():
        h_mp, _ = _encoder(False)(x, ei, ea, batch, ng)
        h_at, _ = _encoder(True)(x, ei, ea, batch, ng, spd=spd)
    assert not torch.allclose(h_mp, h_at, atol=1e-4)


def test_pooled_vector_is_extensive():
    """A mean pool is size-blind, and the log Z head needs exactly the size dependence a
    mean would discard."""
    enc = _encoder(attention=False)
    small, large = _graph('CCO', 8), _graph('CCCCCCCCCC', 9)
    outs = []
    for gph in (small, large):
        x, ei, ea, batch, ng, _ = _batch([gph])
        with torch.no_grad():
            outs.append(enc(x, ei, ea, batch, ng)[1])
    sum_half = slice(HIDDEN, 2 * HIDDEN)
    assert outs[1][0, sum_half].abs().sum() > outs[0][0, sum_half].abs().sum(), \
        'the unnormalised half of the pool did not grow with molecule size'


# ------------------------------------------------------------------ global channel

@pytest.mark.parametrize('attention', [False, True])
def test_global_context_reaches_atoms_beyond_the_message_passing_radius(attention):
    """BOTH arms must have a global channel, and this is how you tell.

    Regression guard for a real defect: the pooled vector was originally computed at the END
    of forward and returned alongside h, so it never reached the per-atom embeddings at all.
    The non-attention arm then had NO global context, and the battery's block A/C comparison
    silently became "global vs none" instead of the intended "rank-one broadcast vs pairwise
    routing".

    THE SEPARATOR IS THE DISTANCE. Message passing reaches exactly `layers` hops, so on a
    path far longer than that, any influence of one endpoint on the other MUST have come
    through the global step. No control arm is needed -- the graph does the isolating.
    """
    layers = 2
    n = 3 * layers + 6                       # endpoints are 7 hops apart, MP reaches 2
    enc = _encoder(attention, layers=layers)
    ei = torch.as_tensor(np.concatenate([_path_edges(n), _path_edges(n)[::-1]], axis=1),
                         dtype=torch.long)
    ea = torch.zeros(ei.shape[1], EDGE_DIM)
    batch = torch.zeros(n, dtype=torch.long)
    spd_np = shortest_paths(_path_edges(n), n)
    dspd = dense_spd_batch([spd_np], 1, n) if attention else None
    assert spd_np[0, n - 1] > layers, 'the endpoints are inside the MP radius; test is blind'

    g = torch.Generator().manual_seed(11)
    x = torch.rand(n, NODE_DIM, generator=g)
    x2 = x.clone()
    x2[0] = x2[0] + 5.0                       # perturb one endpoint only

    with torch.no_grad():
        h0, _ = enc(x, ei, ea, batch, 1, spd=dspd)
        h1, _ = enc(x2, ei, ea, batch, 1, spd=dspd)

    far = (h0[n - 1] - h1[n - 1]).abs().max().item()
    assert far > 1e-6, (
        f'perturbing atom 0 left atom {n - 1} unchanged ({far:.2e}) at a separation of '
        f'{spd_np[0, n - 1]} hops with only {layers} message-passing layers -- this arm has '
        f'no global channel')


def test_attention_block_starts_as_pure_attention():
    """Zero-init of the extensive branch is deliberate, so pin it.

    The unnormalised sum is L times the magnitude of the convex combination (measured 9x at
    L=9, 31x at Gly4's 31 atoms, 63x at L=63). Projected jointly, it swamps the routed signal
    at init and the routed path gets a vanishing share of the gradient -- which is what
    collapsed every attention advantage in the 2026-08-27 run. Starting the extensive branch
    at zero means the block begins as pure attention and LEARNS the extensive channel in.
    """
    att = SPDAttention(HIDDEN, n_heads=4, max_spd=6).eval()
    assert att.proj_sum.weight.abs().max().item() == 0.0
    assert att.proj_sum.bias.abs().max().item() == 0.0


def test_attention_block_can_represent_an_extensive_quantity():
    """The CAPABILITY must be wired even though it is off at init.

    A pure softmax is a convex combination: with identical keys and values, doubling the
    atom count leaves the attended output bit-identical, so it cannot represent any extensive
    quantity. Give the extensive branch weights and the output must move -- that is the path
    the optimiser can travel down, and this test fails if the branch is not connected.
    """
    torch.manual_seed(0)
    att = SPDAttention(HIDDEN, n_heads=4, max_spd=6).eval()
    with torch.no_grad():                       # the state the optimiser can reach
        att.proj_sum.weight.normal_(0, 0.02)
    row = torch.rand(1, 1, HIDDEN, generator=torch.Generator().manual_seed(2))
    outs = []
    for n in (4, 8):
        dense = row.expand(1, n, HIDDEN).contiguous()      # n IDENTICAL atoms
        mask = torch.ones(1, n, dtype=torch.bool)
        spd = torch.ones(1, n, n, dtype=torch.long)        # all pairs equidistant
        with torch.no_grad():
            outs.append(att(dense, mask, spd)[0, 0])
    moved = (outs[0] - outs[1]).abs().max().item()
    assert moved > 1e-5, (
        f'doubling the atom count left the output unchanged ({moved:.2e}) even with the '
        f'extensive branch weighted; it is not connected')
