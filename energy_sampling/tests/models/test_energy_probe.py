"""`models/energy_probe.py`: the pieces a wrong answer would hide in.

The encoder runs INSIDE the training step there, from a padded table of every condition's
graph, and its per-atom output is reordered into each member's placement order. A wrong
gather conditions every atom on another atom with no shape error, so the table's batch is
held against `encoder_cache.embed`, one molecule at a time, on a shuffled batch with repeats.
Also pinned: the two prior file forms read as the same rows, and the centred error removes
exactly a per-molecule constant.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models import encoder_cache
from models import energy_probe as ep
from models.graph_encoder import MPNNEncoder

pytestmark = pytest.mark.fast

BS = chr(92)
SMIS = ['CCCCO', 'C[C@H](O)CC', 'C#C/C(=N/C)N(C)C=O', 'C#C/C(=N' + BS + 'C)N(C)C=O', 'c1ccccc1O', 'CC']


def _members():
    from energies.conformer_torsions import ConformerTorsions
    return [ConformerTorsions(smiles=s, device='cpu', level='full', force_field='mmff') for s in SMIS]


@pytest.mark.parametrize('stereo', [1, 2])
def test_graph_table_batch_is_embed_in_placement_order(stereo):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        members = _members()
        table = ep.GraphTable(SMIS, [m.spec.perm for m in members], [m.spec.z for m in members],
                              'rwse', 16, stereo, True, 'cpu')
        torch.manual_seed(0)
        enc = MPNNEncoder(table.node_dim + 16, table.edge_dim, hidden=32, layers=3,
                          attention=True, n_heads=4, max_spd=8).eval()
        bundle = {'encoder': enc, 'dtype': torch.float32, 'cfg': {'encoding': 'rwse', 'spd': True},
                  'k': 16, 'device': 'cpu', 'stereo_features': stereo}
        slots = torch.tensor([3, 0, 5, 3, 1, 4, 2])
        with torch.no_grad():
            h, g = ep.EnergyModel(None, enc).embeddings(table, slots)
        at = 0
        for b, c in enumerate(slots.tolist()):
            want_h, want_g, _ = encoder_cache.embed(bundle, SMIS[c], perm=members[c].spec.perm,
                                                    z_tree=members[c].spec.z)
            n = want_h.shape[0]
            assert torch.allclose(h[at:at + n], want_h, atol=1e-5), SMIS[c]
            assert torch.allclose(g[b], want_g, atol=1e-5), SMIS[c]
            at += n
        assert at == h.shape[0]
    finally:
        torch.set_default_dtype(old)


def test_graph_table_refuses_a_permutation_that_is_not_the_members():
    members = _members()
    perms = [m.spec.perm for m in members]
    perms[0] = np.asarray(perms[0])[::-1].copy()          # butanol read back to front
    with pytest.raises(SystemExit, match='does not map'):
        ep.GraphTable(SMIS, perms, [m.spec.z for m in members], 'rwse', 16, 2, True, 'cpu')


def test_both_prior_forms_read_as_the_same_rows(tmp_path):
    from energies.conformer_data import compact_prior

    idents = ['a', 'b', 'c', 'd']
    K = 5
    g = torch.Generator().manual_seed(0)
    row_ident = ['c', 'a', 'c', 'd', 'a', 'c']            # 'b' holds no rows
    states = torch.rand(len(row_ident), K, generator=g)
    cond = SimpleNamespace(identifier=idents, smiles=['C', 'CC', 'CCC', 'CCCC'])
    out = {}
    for form in ('graph', 'compact'):
        d = tmp_path / form
        d.mkdir()
        torch.save({'prior': cond}, d / 'conditions_train.pt')
        if form == 'graph':
            torch.save({'prior': SimpleNamespace(identifier=row_ident, torsion_state=states)},
                       d / 'prior_train.pt')
        else:
            holders = ['a', 'c', 'd']
            order = [r for i in holders for r, j in enumerate(row_ident) if j == i]
            torch.save(compact_prior(holders, [2, 3, 1], states[order], torch.zeros(6),
                                     torch.ones(3, K, dtype=torch.bool), torch.tensor([5, 8, 11])),
                       d / 'prior_train.pt')
        _, got_ids, _, slot, x = ep.load_rung(str(d))
        assert got_ids == idents
        out[form] = sorted((int(s), tuple(np.round(r.tolist(), 6))) for s, r in zip(slot, x))
    assert out['graph'] == out['compact']
    assert {s for s, _ in out['graph']} == {0, 2, 3}


def test_centred_error_removes_a_constant_per_molecule():
    g = torch.Generator().manual_seed(0)
    slots = torch.arange(40).repeat_interleave(6)
    y = 25 * torch.rand(240, generator=g)
    offset = 50 * torch.randn(40, generator=g)
    m = ep.metrics(y + offset[slots], y, slots)
    assert m['low_n'] == 240 and m['low_rmse'] > 10
    assert m['low_centred_rmse'] < 1e-4
    # and it does not hide an error that varies inside a molecule
    noisy = ep.metrics(y + offset[slots] + torch.randn(240, generator=g), y, slots)
    assert 0.5 < noisy['low_centred_rmse'] < 1.5
    # the reference beside it is the spread of the target about each molecule's own mean
    want = (y - torch.stack([y[slots == c].mean() for c in range(40)])[slots]).pow(2).mean().sqrt()
    assert abs(m['low_centred_sd'] - float(want)) < 1e-5
