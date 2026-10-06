"""The state encoding that reads atoms (models/graph_state.py). CPU, random inputs, no checkpoint.

What is pinned:
  * a row's encoding is its own: the same alone and in a batch, under any padding width, and
    whatever the padding holds (NaN included), in training and in evaluation mode;
  * atoms are a set: reordering a row's atoms does not change its encoding;
  * every input is read: the state, each real atom's features and element, and the extra vector;
  * a loss on the encoding reaches every parameter and every real atom's features, and no
    padded atom's (the gradient a trunk would be trained with through the stored embeddings);
  * a row with no atoms, and a missing or unexpected context or extra vector, raise.
"""
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _root in (os.path.dirname(_here),
              os.path.join(os.path.dirname(os.path.dirname(_here)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from energy_sampling.models.graph_state import AtomSetPool, AtomStateEncoding  # noqa: E402

pytestmark = pytest.mark.fast

S_DIM, ATOM_DIM, EXTRA_DIM = 18, 11, 12
SIZES = (5, 9, 3, 12, 7, 1)


def _encoder(extra_dim=0, norm='layer'):
    torch.manual_seed(0)
    return AtomStateEncoding(S_DIM, ATOM_DIM, layers=2, hidden_dim=64, s_emb_dim=32, extra_dim=extra_dim,
                             atom_hidden_dim=48, blocks=2, heads=4, norm=norm).double()


def _rows(width=None, fill=0.0, seed=1):
    """Rows of SIZES atoms padded to `width`, the padding holding `fill` and a wrong element."""
    g = torch.Generator().manual_seed(seed)
    width = max(SIZES) if width is None else width
    b = len(SIZES)
    atoms = torch.full((b, width, ATOM_DIM), fill, dtype=torch.float64)
    z = torch.full((b, width), 99, dtype=torch.long)
    mask = torch.zeros(b, width, dtype=torch.bool)
    for i, n in enumerate(SIZES):
        atoms[i, :n] = torch.randn(n, ATOM_DIM, generator=g, dtype=torch.float64)
        z[i, :n] = torch.randint(1, 10, (n,), generator=g)
        mask[i, :n] = True
    s = torch.randn(b, S_DIM, generator=g, dtype=torch.float64)
    return s, atoms, z, mask


@pytest.mark.parametrize('train', [True, False])
def test_a_rows_encoding_does_not_depend_on_its_batch_or_its_padding(train):
    enc = _encoder().train(train)
    s, atoms, z, mask = _rows()
    whole = enc(s, atoms, z, mask)
    assert whole.shape == (len(SIZES), 32) and bool(torch.isfinite(whole).all())
    for i, n in enumerate(SIZES):
        alone = enc(s[i:i + 1], atoms[i:i + 1, :n], z[i:i + 1, :n], mask[i:i + 1, :n])
        assert torch.allclose(alone, whole[i:i + 1], atol=1e-10), i
    for fill in (0.0, 1e6, float('nan')):
        s2, atoms2, z2, mask2 = _rows(width=max(SIZES) + 6, fill=fill)
        assert torch.allclose(enc(s2, atoms2, z2, mask2), whole, atol=1e-10), fill
    with torch.no_grad():
        assert torch.allclose(enc(s, atoms, z, mask), whole, atol=1e-10)


def test_reordering_a_rows_atoms_does_not_change_its_encoding():
    enc = _encoder()
    s, atoms, z, mask = _rows()
    whole = enc(s, atoms, z, mask)
    g = torch.Generator().manual_seed(5)
    atoms2, z2 = atoms.clone(), z.clone()
    for i, n in enumerate(SIZES):
        perm = torch.randperm(n, generator=g)
        atoms2[i, :n], z2[i, :n] = atoms[i, perm], z[i, perm]
    assert not torch.equal(atoms2, atoms)
    assert torch.allclose(enc(s, atoms2, z2, mask), whole, atol=1e-10)


def test_every_input_is_read():
    enc = _encoder(extra_dim=EXTRA_DIM)
    s, atoms, z, mask = _rows()
    extra = torch.randn(len(SIZES), EXTRA_DIM, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    base = enc(s, atoms, z, mask, extra)

    def moved(out, row):
        """Only `row` changed."""
        diff = (out - base).abs().amax(dim=1)
        others = torch.ones(len(SIZES), dtype=torch.bool)
        others[row] = False
        return bool(diff[row] > 1e-6) and bool((diff[others] < 1e-10).all())

    s2 = s.clone()
    s2[1, 3] += 0.5
    assert moved(enc(s2, atoms, z, mask, extra), 1)
    atoms2 = atoms.clone()
    atoms2[3, SIZES[3] - 1, 4] += 0.5          # the last real atom of the widest row
    assert moved(enc(s, atoms2, z, mask, extra), 3)
    z2 = z.clone()
    z2[4, 0] = 16
    assert moved(enc(s, atoms, z2, mask, extra), 4)
    extra2 = extra.clone()
    extra2[0, 7] += 0.5
    assert moved(enc(s, atoms, z, mask, extra2), 0)


def test_a_loss_reaches_every_parameter_and_every_real_atom_only():
    enc = _encoder()
    s, atoms, z, mask = _rows(fill=3.0)
    atoms.requires_grad_(True)
    enc(s, atoms, z, mask).square().sum().backward()
    dead = [name for name, p in enc.named_parameters() if p.grad is None or not bool(p.grad.abs().sum() > 0)]
    # the element table counts as reached when any of its rows is: only elements present are touched
    assert dead == [], dead
    per_atom = atoms.grad.abs().sum(-1)
    assert bool(torch.isfinite(atoms.grad).all())
    assert bool((per_atom[mask] > 0).all()) and bool((per_atom[~mask] == 0).all())


def test_bad_calls_raise():
    enc = _encoder()
    s, atoms, z, mask = _rows()
    empty = mask.clone()
    empty[2] = False
    with pytest.raises(ValueError, match='no atoms'):
        enc(s, atoms, z, empty)
    with pytest.raises(ValueError, match='extra'):
        enc(s, atoms, z, mask, torch.zeros(len(SIZES), EXTRA_DIM, dtype=torch.float64))
    with pytest.raises(ValueError, match='extra'):
        _encoder(extra_dim=EXTRA_DIM)(s, atoms, z, mask)
    with pytest.raises(ValueError, match='features'):
        enc(s, atoms[..., :-1], z, mask)
    pool = AtomSetPool(ATOM_DIM, context_dim=0, hidden_dim=16, blocks=1, heads=2).double()
    with pytest.raises(ValueError, match='context'):
        pool(atoms, z, mask, context=s)
    assert pool(atoms, z, mask).shape == (len(SIZES), pool.out_dim)
