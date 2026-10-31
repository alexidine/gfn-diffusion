"""data_processing/prior_to_seeds.py: a train.py prior cut into shards of molecules and seeds for a seeded streaming
search.

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_prior_to_seeds.py

What each test would catch:
  * identifiers sharded in another order than a stream run takes its conditions (sorted as strings), a molecule split
    across two shards, or a row lost or repeated;
  * a shard's molecule not the molecule of its rows (z, pos at the stored frame) or not carrying its identifier;
  * a seed row changed on the way (its cell parameters or handedness);
  * an existing shard file overwritten, or a manifest from another cut extended.

CPU, synthetic rows of one 5-atom molecule under three identifiers, no data drive.
"""
import json

import pytest
import torch

from mxtaltools.dataset_utils.data_classes import MolCrystalData, MolData
from mxtaltools.dataset_utils.utils import collate_data_list

from data_processing.prior_to_seeds import main, shard_plan

pytestmark = pytest.mark.fast   # torch imported; CPU, synthetic rows, no data drive

POS = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [2.1, 1.3, 0.2], [-0.7, 1.2, -0.3], [0.3, -1.3, 0.5]])
Z = torch.tensor([6, 6, 7, 8, 6])


def _row(ident, a):
    mol = MolData(z=Z, pos=POS.clone(), x=torch.zeros(5), identifier=ident, do_mol_analysis=True)
    return MolCrystalData(molecule=mol, sg_ind=2, z_prime=torch.tensor(1, dtype=torch.long),
                          cell_lengths=torch.tensor([a, 7.0, 8.0]), cell_angles=torch.tensor([1.6, 1.7, 1.8]),
                          aunit_centroid=torch.tensor([0.2, 0.3, 0.4]), aunit_orientation=torch.tensor([0.1, 0.2, 0.3]),
                          aunit_handedness=torch.tensor([1.0]), identifier=ident, do_box_analysis=True)


def test_shard_plan_keeps_molecules_whole_in_sorted_order():
    plan = shard_plan(['b', 'a', 'b', 'c', 'a'], 2)
    assert plan == [(['a', 'b'], [0, 1, 2, 4]), (['c'], [3])]
    assert shard_plan([['x'], ['x']], 5) == [(['x'], [0, 1])]  # identifiers stored as one-element lists
    with pytest.raises(ValueError):
        shard_plan(['a'], 0)


def test_shards_hold_each_molecule_and_its_rows(tmp_path):
    rows = [_row(i, a) for i, a in (('m2', 6.0), ('m0', 6.5), ('m1', 7.0), ('m0', 7.5), ('m2', 8.0))]
    src = tmp_path / 'prior.pt'
    torch.save({'equalized_prior': collate_data_list(rows), 'frame': 'orient_molecule(mode=std) fixed point'}, src)
    out = tmp_path / 'seeds'
    main([str(src), '--out', str(out), '--per_shard', '2'])
    m0 = torch.load(out / 'shard_0000_mols.pt', weights_only=False)
    s0 = torch.load(out / 'shard_0000_seeds.pt', weights_only=False)
    s1 = torch.load(out / 'shard_0001_seeds.pt', weights_only=False)
    assert [str(m.identifier) for m in m0] == ['m0', 'm1']
    assert sorted(str(r.identifier) for r in s0) == ['m0', 'm0', 'm1'] and [str(r.identifier) for r in s1] == ['m2', 'm2']
    assert all(torch.equal(m.z, Z) and torch.allclose(m.pos, POS) for m in m0)
    lengths = sorted(float(r.cell_lengths.reshape(-1)[0]) for r in s0 + s1)
    assert lengths == [6.0, 6.5, 7.0, 7.5, 8.0]
    assert all(float(r.aunit_handedness.reshape(-1)[0]) == 1.0 for r in s0 + s1)
    man = json.loads((out / 'manifest.json').read_text())
    assert man['n_shards'] == 2 and man['n_molecules'] == 3 and man['n_rows'] == 5
    assert man['shards']['0']['seeds_count'] == 3 and man['shards']['1']['molecules'] == 1
    with pytest.raises(SystemExit):  # never overwrites a shard
        main([str(src), '--out', str(out), '--per_shard', '2', '--shards', '0'])
    with pytest.raises(SystemExit):  # nor extends a manifest of another cut
        main([str(src), '--out', str(out), '--per_shard', '1', '--shards', '2'])
