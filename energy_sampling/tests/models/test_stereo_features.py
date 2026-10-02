"""The encoder's stereo feature levels (`models/graph_encodings.py::STEREO_FEATURES`).

Level 1 is what every checkpoint written before the key existed was trained on: R/S at CIP
tetrahedral centres and nothing else, so a pseudo-asymmetric centre (CIP r/s) and a double
bond's E/Z are invisible and such isomers are one input. Level 2 adds both. Each claim is
checked on isomer PAIRS, because a feature that is merely present proves nothing.
"""
import numpy as np
import pytest
import torch

from models import encoder_cache
from models import encoder_probe as ep
from models.graph_encodings import (STEREO_FEATURES, bond_features_from_smiles,
                                    bond_stereo_codes, cip_codes, graph_from_smiles,
                                    mol_for_labels)

BS = chr(92)
EZ = ('C#C/C(=N/C)N(C)C=O', 'C#C/C(=N' + BS + 'C)N(C)C=O')                # Z, E imine
RING_RS = ('C#CC#C[C@@H]1[C@@H]2C[C@H]1O2', 'C#CC#C[C@H]1[C@@H]2C[C@H]1O2')  # differ at an r/s centre
CIS_TRANS = ('C#CC#C[C@H]1C[C@@H](C)C1', 'C#CC#C[C@H]1C[C@H](C)C1')     # r/s centres only
ENANT = ('C[C@H](O)CC', 'C[C@@H](O)CC')


def _inputs(smi, level):
    z, e, parity = graph_from_smiles(smi, stereo=level)
    return z, e, parity, bond_features_from_smiles(smi, stereo=level)


def _same(a, b, level):
    return all(np.array_equal(x, y) for x, y in zip(_inputs(a, level), _inputs(b, level)))


def test_level_1_is_the_default_and_carries_no_new_column():
    for smi in (*EZ, *RING_RS, *ENANT):
        z, e, p = graph_from_smiles(smi)
        z1, e1, p1 = graph_from_smiles(smi, stereo=1)
        assert np.array_equal(p, p1) and np.array_equal(e, e1)
        assert np.array_equal(p, cip_codes(mol_for_labels(smi)))
        bf = bond_features_from_smiles(smi)
        assert bf.shape[1] == 4 and np.array_equal(bf, bond_features_from_smiles(smi, stereo=1))
    assert STEREO_FEATURES == (1, 2)
    with pytest.raises(ValueError, match='stereo features'):
        graph_from_smiles(EZ[0], stereo=3)


def test_level_1_cannot_tell_these_isomers_apart_and_level_2_can():
    for pair in (EZ, RING_RS, CIS_TRANS):
        assert _same(*pair, level=1), pair
        assert not _same(*pair, level=2), pair
    assert not _same(*ENANT, level=1) and not _same(*ENANT, level=2)


def test_pseudo_asymmetric_centres_enter_the_parity_column():
    for a, b in (RING_RS, CIS_TRANS):
        pa, pb = graph_from_smiles(a, stereo=2)[2], graph_from_smiles(b, stereo=2)[2]
        differ = np.flatnonzero(pa != pb)
        assert differ.size >= 1 and np.array_equal(pa[differ], -pb[differ])
        # exactly the centres level 1 reads as 0
        assert not graph_from_smiles(a, stereo=1)[2][differ].any()
    # R/S centres read the same at both levels
    p1, p2 = graph_from_smiles(ENANT[0], stereo=1)[2], graph_from_smiles(ENANT[0], stereo=2)[2]
    assert np.array_equal(p1, p2) and np.count_nonzero(p1) == 1


def test_bond_stereo_column_is_e_plus_z_minus_on_the_labelled_bond_only():
    codes = [bond_stereo_codes(mol_for_labels(s)) for s in EZ]
    assert [int(c.sum()) for c in codes] == [-1, 1]          # Z then E
    assert all(np.count_nonzero(c) == 1 for c in codes)
    assert np.flatnonzero(codes[0]).tolist() == np.flatnonzero(codes[1]).tolist()
    bf = bond_features_from_smiles(EZ[0], stereo=2)
    assert bf.shape[1] == 5 and np.array_equal(bf[:, 4], codes[0])
    assert np.array_equal(bf[:, :4], bond_features_from_smiles(EZ[0]))
    assert not bond_stereo_codes(mol_for_labels('CC=C(C)C')).any()     # no stereo to label


def test_sample_labels_follow_the_inputs():
    held = ep.PROBES
    ep.PROBES = held + [p for p in ep.EZ_PROBES]
    try:
        z_s, e_s = (ep.build_sample(s, 'rwse', 8, stereo=2) for s in EZ)
        assert z_s.edge_attr.shape[1] == 5
        # the tripwire label is the bond code at both ends of the labelled bond
        assert np.count_nonzero(z_s.labels['ez_code']) == 2
        assert np.array_equal(z_s.labels['ez_code'], -e_s.labels['ez_code'])
        assert float(z_s.labels['ez_moment']) == -float(e_s.labels['ez_moment']) != 0.0
        # cip_code is the parity INPUT column, r/s included
        r = ep.build_sample(CIS_TRANS[0], 'rwse', 8, stereo=2)
        assert np.array_equal(r.labels['cip_code'][:, 0], r.x[:, -2])
        assert np.count_nonzero(r.labels['cip_code']) == 2
        # at level 1 the bond-stereo labels are identically zero
        flat = ep.build_sample(EZ[0], 'rwse', 8, stereo=1)
        assert flat.edge_attr.shape[1] == 4 and not flat.labels['ez_code'].any()
    finally:
        ep.PROBES = held
    assert [p.name for p in ep.PROBES][-2:] == ['cip_code', 'chiral_moment'], \
        'the default probe list must not carry the E/Z probes'


def _tiny_checkpoint(tmp_path, level):
    """An untrained encoder saved the way `encoder_probe.run` saves one."""
    s = ep.build_sample('CCO', 'rwse', 8, stereo=level)
    torch.manual_seed(0)
    model = ep.ProbeModel(s.x.shape[1], 8, s.edge_attr.shape[1], hidden=16, layers=2,
                          attention=True)
    with torch.no_grad():                       # the bond-stereo column must reach h
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ck = {'state_dict': model.state_dict(), 'arm': 'mp+attn+spd', 'seed': 0, 'n_train': 0,
          'hidden': 16, 'layers': 2, 'k': 8, 'attention': True}
    if level is not None and level >= 2:
        ck['stereo_features'] = level
    path = tmp_path / f'enc_{level}.pt'
    torch.save(ck, path)
    return str(path)


def test_cache_builds_inputs_at_the_checkpoints_level(tmp_path):
    old = encoder_cache.load_encoder(_tiny_checkpoint(tmp_path, 1), device='cpu')
    new = encoder_cache.load_encoder(_tiny_checkpoint(tmp_path, 2), device='cpu')
    assert (old['stereo_features'], old['edge_dim']) == (1, 4)      # no key: level 1
    assert (new['stereo_features'], new['edge_dim']) == (2, 5)
    for pair in (EZ, CIS_TRANS):
        g_old = [encoder_cache.embed(old, s)[1] for s in pair]
        g_new = [encoder_cache.embed(new, s)[1] for s in pair]
        assert torch.equal(g_old[0], g_old[1]), 'level 1 sees one input for both isomers'
        assert not torch.allclose(g_new[0], g_new[1]), 'level 2 must separate them'


def test_shipped_checkpoint_is_level_1():
    b = encoder_cache.load_encoder(device='cpu')
    assert (b['stereo_features'], b['edge_dim']) == (1, 4)
