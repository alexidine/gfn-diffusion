"""
CPU tests for the atom-token condition of the exact-density flow (inn_modeller.AtomEncoder, the cross-attention of
energies/ar_transformer.py, ConditionalLatentFlow with atoms).

WHAT THIS FILE IS FOR. With --atom_tokens the condition is the molecule itself: one token per atom, from a ScalarGNN
over atomic numbers and standard-frame coordinates, which every causal layer of the conditioner cross-attends to. Four
failures would all run without an error: (1) a molecule's tokens depending on the other molecules encoded with it (the
batch dependence of the stored Mo3ENet embeddings), (2) padding atoms leaking into the tokens or the pool, (3) a row's
spline parameters seeing its own or a later row through the new attention, breaking the triangular Jacobian, and (4)
the memory being ignored. Each is checked, with NEGATIVE CONTROLS where equality could come from a model that ignores
its input.

    python test_inn_atom_tokens.py
"""
import os
import sys
import types

import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))   # tests/<area>/x.py -> energy_sampling/
for p in (_here, os.path.dirname(_here)):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import inn_modeller as im  # noqa: E402
from energies.ar_transformer import CausalTransformerConditioner  # noqa: E402

WRAP = [False] * 7 + [True, True, False, True, True]


def _molecules(n_mol=5, a_max=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = torch.randint(3, a_max + 1, (n_mol,), generator=g)
    z = torch.zeros(n_mol, a_max, dtype=torch.long)
    pos = torch.zeros(n_mol, a_max, 3)
    for i in range(n_mol):
        k = int(n[i])
        z[i, :k] = torch.tensor([1, 6, 7, 8])[torch.randint(0, 4, (k,), generator=g)]
        pos[i, :k] = 1.5 * torch.randn(k, 3, generator=g)
    return z, pos, n


def _encoder():
    torch.manual_seed(1)
    return im.AtomEncoder(width=64, node_dim=32, convs=2, cutoff=10.0, radial=8, pooled_dim=16).eval()


def test_encoder_is_batch_independent():
    enc = _encoder()
    z, pos, n = _molecules()
    for train in (True, False):   # train mode too: a norm that mixes molecules through batch statistics does so there
        enc.train(train)
        with torch.no_grad():
            tok_all, mask_all, pool_all = enc(z, pos, n)
            for i in range(z.shape[0]):
                tok_i, mask_i, pool_i = enc(z[i:i + 1], pos[i:i + 1], n[i:i + 1])
                assert torch.allclose(tok_i[0][mask_i[0]], tok_all[i][mask_all[i]], atol=1e-5), (train, i)
                assert torch.allclose(pool_i[0], pool_all[i], atol=1e-5), (train, i)


def test_encoder_ignores_padding():
    enc = _encoder()
    z, pos, n = _molecules()
    zp = torch.cat([z, torch.zeros(z.shape[0], 4, dtype=torch.long)], 1)
    pp = torch.cat([pos, 50.0 * torch.ones(pos.shape[0], 4, 3)], 1)   # padding far away, with junk coordinates
    with torch.no_grad():
        t1, m1, p1 = enc(z, pos, n)
        t2, m2, p2 = enc(zp, pp, n)
    assert torch.allclose(t1[m1], t2[m2], atol=1e-5) and torch.allclose(p1, p2, atol=1e-5)


def test_encoder_reads_coordinates_negative_control():
    enc = _encoder()
    z, pos, n = _molecules()
    with torch.no_grad():
        _, _, p1 = enc(z, pos, n)
        _, _, p2 = enc(z, pos @ torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]]), n)   # rotate every molecule
    assert float((p1 - p2).abs().max()) > 1e-3   # not rotation-invariant: the frame is part of the condition


def _conditioner():
    torch.manual_seed(2)
    order = [8, 11, 10, 9, 7, 6, 5, 4, 3, 2, 1, 0]
    feat_dim = []
    for i in range(12):
        feat_dim += [i, i] if WRAP[i] else [i]
    m = CausalTransformerConditioner(order, feat_dim, 8, width=64, layers=2, heads=4, context_dim=16, cross=True)
    with torch.no_grad():   # leave the zero initialisation, so outputs and the cross blocks are live
        for prm in m.parameters():
            if prm.dim() > 1 and float(prm.abs().max()) == 0.0:
                prm.normal_(0, 0.1)
            elif prm.dim() == 1 and float(prm.abs().max()) == 0.0:
                prm.normal_(0, 0.1)
    return m.double(), order, feat_dim


def _feats(x, feat_dim):
    cols = []
    for c, i in enumerate(feat_dim):
        if WRAP[i]:
            first = feat_dim.index(i) == c
            cols.append(torch.sin(torch.pi * x[:, i]) if first else torch.cos(torch.pi * x[:, i]))
        else:
            cols.append(x[:, i])
    return torch.stack(cols, -1)


def test_cross_attention_keeps_the_order_causal():
    m, order, feat_dim = _conditioner()
    torch.manual_seed(3)
    b = 4
    x = torch.rand(b, 12, dtype=torch.float64) * 2 - 1
    ctx = (torch.randn(b, 16, dtype=torch.float64), torch.randn(b, 6, 64, dtype=torch.float64),
           torch.ones(b, 6, dtype=torch.bool))
    with torch.no_grad():
        base = m(_feats(x, feat_dim), ctx).view(b, 12, -1)
        for j, row in enumerate(order):
            x2 = x.clone()
            x2[:, row] += 0.3
            out = m(_feats(x2, feat_dim), ctx).view(b, 12, -1)
            for k, other in enumerate(order):
                same = torch.allclose(out[:, other], base[:, other], atol=1e-12)
                if k <= j:
                    assert same, (row, other)   # rows at or before row in the order never see it
            later = [order[k] for k in range(j + 1, 12)]
            if later:   # negative control: some later row does see it
                assert any(not torch.allclose(out[:, r], base[:, r], atol=1e-9) for r in later), row


def test_memory_is_read_and_padding_masked():
    m, order, feat_dim = _conditioner()
    torch.manual_seed(4)
    b = 3
    x = torch.rand(b, 12, dtype=torch.float64) * 2 - 1
    vec = torch.randn(b, 16, dtype=torch.float64)
    mem = torch.randn(b, 6, 64, dtype=torch.float64)
    mask = torch.tensor([[True] * 4 + [False] * 2] * b)
    with torch.no_grad():
        a = m(_feats(x, feat_dim), (vec, mem, mask))
        mem2 = mem.clone()
        mem2[:, 4:] = 100.0   # change only the padding
        b2 = m(_feats(x, feat_dim), (vec, mem2, mask))
        mem3 = mem.clone()
        mem3[:, 0] += 1.0     # change a real token
        c3 = m(_feats(x, feat_dim), (vec, mem3, mask))
    assert torch.allclose(a, b2, atol=1e-12)
    assert float((a - c3).abs().max()) > 1e-6


def _flow():
    torch.manual_seed(5)
    f = im.ConditionalLatentFlow(12, WRAP, [], 1, 32, 64, 1, 8, (64,), 5.0, 0, transformer=(64, 2, 4),
                                 atoms=dict(node_dim=32, convs=2, cutoff=10.0, radial=8))
    f.set_bf16(False) if hasattr(f, 'set_bf16') else None
    z, pos, n = _molecules(n_mol=6)
    f.flow.set_atom_table(z, pos, n)
    for blk in f.flow.blocks:
        blk.min_height = torch.full_like(blk.bound, 1e-5)
        blk.tied = 2.0
        if hasattr(blk.made, 'bf16'):
            blk.made.bf16 = False
    with torch.no_grad():   # leave the identity start so the round trip is a real test
        for blk in f.flow.blocks:
            blk.made.head.weight.normal_(0, 0.05)
    return f


def test_flow_inverts_and_trains_the_encoder():
    f = _flow().double()
    c = torch.tensor([0, 1, 2, 3, 4, 5, 0, 3], dtype=torch.float64).reshape(-1, 1)
    torch.manual_seed(6)
    with torch.no_grad():
        x, z0 = f.sample(c.shape[0], c, return_base=True)
        zb, _ = f.to_base(x, c)
    assert float((zb - z0).abs().max()) < 1e-8
    lq = f.log_prob(x, c)
    assert bool(torch.isfinite(lq).all())
    lq.sum().backward()
    g = sum(float(prm.grad.abs().sum()) for prm in f.flow.atom_encoder.parameters() if prm.grad is not None)
    assert g > 0.0


def test_missing_atoms_raise():
    f = _flow()
    c = torch.tensor([[7.0]])   # no molecule at condition id 7 (out of range or empty)
    z, pos, n = _molecules(n_mol=8)
    n[7] = 0
    f.flow.set_atom_table(z, pos, n)
    try:
        f.log_prob(torch.zeros(1, 12), c)
    except RuntimeError:
        return
    raise AssertionError("a condition with no atoms did not raise")


def _live_flow(atoms=True):
    """A double-precision conditional flow with every parameter moved off its initialisation, so the cross-attention
    (zero-initialised output) and the head are both live."""
    torch.manual_seed(8)
    kw = dict(atoms=dict(node_dim=32, convs=2, cutoff=10.0, radial=8)) if atoms else {}
    f = im.ConditionalLatentFlow(12, WRAP, [], 1 if atoms else 5, 32, 64, 1, 8, (64,), 5.0, 0,
                                 transformer=(64, 2, 4), **kw)
    for blk in f.flow.blocks:
        blk.min_height = torch.full_like(blk.bound, 1e-5)
        blk.tied = 2.0
        blk.made.bf16 = False
    with torch.no_grad():
        for prm in f.parameters():
            prm.add_(0.1 * torch.randn_like(prm))
    f = f.double()
    if atoms:
        z, pos, n = _molecules(n_mol=6)
        f.flow.set_atom_table(z, pos.double(), n)
    return f


def _cond(atoms, b, seed=9):
    g = torch.Generator().manual_seed(seed)
    if atoms:
        return torch.randint(0, 6, (b, 1), generator=g).double()
    return torch.randn(b, 5, generator=g, dtype=torch.float64)


def _base_points(f, b, seed=10):
    g = torch.Generator().manual_seed(seed)
    u = 2 * torch.rand(b, 12, generator=g, dtype=torch.float64) - 1
    return torch.where(f.flow.is_wrap, u, torch.randn(b, 12, generator=g, dtype=torch.float64))


def test_cached_decode_with_context_equals_dpass():
    # the sampler's cached decode (one conditioner step per row, the memory's keys and values projected once) against
    # the d-pass inverse, on the same base points, with atom tokens and with a plain context vector
    for atoms in (True, False):
        f = _live_flow(atoms)
        b = 400
        c, z0 = _cond(atoms, b), _base_points(f, b)
        blk = f.flow.blocks[0]
        with torch.no_grad():
            ctx = f.flow.context_input(c)[0]
            x_d = blk.invert(z0, ctx)
            x_c = im.invert_block_cached(blk, z0, ctx)
        err = (x_c - x_d).abs().max(-1).values
        assert float(torch.quantile(err, 0.99)) < 1e-9 and float(err.max()) < 1e-5, (atoms, float(err.max()))
        if atoms:   # negative control: moving one molecule's atoms moves its rows' inverse
            f.flow.atom_pos[c[0, 0].long(), 0] += 0.5
            with torch.no_grad():
                x_m = blk.invert(z0, f.flow.context_input(c)[0])
            rows = c[:, 0] == c[0, 0]
            assert float((x_m - x_d)[rows].abs().max()) > 1e-4


def test_shared_context_encodes_once_per_tensor_and_changes_nothing():
    # an energy step's reads -- draws on the fresh conditions, log q and the level on them and on the stored rows'
    # conditions -- give the same loss and gradients with the context shared, from 3 encoder calls instead of 5
    f = _live_flow(True)
    calls = []
    f.flow.atom_encoder.register_forward_hook(lambda *a: calls.append(1))
    c_fresh, c_stored = _cond(True, 64, 11), _cond(True, 64, 12)
    x_stored = f.sample(64, c_stored)

    def step():
        torch.manual_seed(13)
        x = f.sample(64, c_fresh)
        loss = (f.log_prob(x, c_fresh) - f.level(c_fresh)).sum() + (f.log_prob(x_stored, c_stored)
                                                                    - f.level(c_stored)).sum()
        f.zero_grad(set_to_none=True)
        loss.backward()
        return float(loss.detach()), {k: p.grad.clone() for k, p in f.named_parameters() if p.grad is not None}

    calls.clear()
    l0, g0 = step()
    n0 = len(calls)
    calls.clear()
    with f.shared_context():
        l1, g1 = step()
        f.log_prob(x_stored[:4], c_stored[:4].clone())   # a different tensor is encoded again
    n1 = len(calls)
    assert f.flow._shared is None
    assert l0 == l1 and g0.keys() == g1.keys()
    # the same gradients up to summation order (one encoder backward with summed upstream gradients against two)
    worst = max(float((g0[k] - g1[k]).abs().max() / g0[k].abs().max().clamp_min(1e-300)) for k in g0)
    assert worst < 1e-12, worst
    assert (n0, n1) == (5, 4), (n0, n1)


def test_grouped_rows_keep_the_uniform_marginal():
    # --rows_per_molecule: k rows of one molecule per group, and every stored row as likely as in a uniform draw
    torch.manual_seed(7)
    counts = torch.tensor([1, 3, 10, 2, 7])
    cid = torch.repeat_interleave(torch.tensor([40, 3, 17, 9, 25]), counts)
    cid = cid[torch.randperm(cid.numel())]                     # rows of a molecule are not contiguous
    host = types.SimpleNamespace(cid_train=cid)
    k, n = 4, 200000
    idx = im.InnModeller.molecule_grouped_rows(host, n, k)
    assert idx.numel() == n
    groups = cid[idx].view(-1, k)
    assert bool((groups == groups[:, :1]).all())               # each group is one molecule
    freq = torch.bincount(idx, minlength=cid.numel()).double() / n
    # uniform over rows within sampling noise; molecules drawn uniformly instead would give the one-row molecule's
    # row 23 / 5 = 4.6 times its share
    assert float((freq * cid.numel() - 1).abs().max()) < 0.05


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)
