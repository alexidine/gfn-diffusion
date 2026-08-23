"""CPU/synthetic proof for the live GFN -> MXtalTools crystal boundary.

Run from ``energy_sampling/``:

    python -m pytest -q test_mxtaltools_crystal_boundary.py

This pins only the canonical ``mk_dev.yaml`` ELJ route:

    mk_dev.yaml -> MolecularCrystal.analyze_crystal_batch
      -> MolCrystalData.latent_to_cell_params
      -> MolCrystalData.analyze(['reduction_en', 'elj'])
      -> mol2cluster -> construct_radial_graph -> eLJ

It deliberately does not contract the MACE-only on-device PBC neighbour-list
adapter, historical dataset construction, conformer machinery, or MXtalTools
internals not exercised by this route.
"""

from pathlib import Path

import torch

import utils
from energies.molecular_crystal import MolecularCrystal
from mxtaltools.dataset_utils.data_class_methods.crystal_analysis import (
    MolCrystalAnalysis,
)
from mxtaltools.dataset_utils.data_class_methods.crystal_building import (
    MolCrystalBuilding,
)
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list


HERE = Path(__file__).resolve().parent
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'


def _synthetic_molecule(identifier: str, scale: float) -> MolData:
    """A tiny, non-collinear rigid molecule with the graph attrs GFN consumes."""
    pos = torch.tensor(
        [[-0.8, -0.2, 0.0],
         [0.7, -0.1, 0.1],
         [0.1, 0.8, -0.1]],
        dtype=torch.float32,
    ) * scale
    return MolData(
        z=torch.tensor([6, 7, 8], dtype=torch.long),
        pos=pos,
        x=torch.zeros((3, 1), dtype=torch.float32),
        identifier=identifier,
        mol_volume=torch.tensor(55.0 * scale ** 3),
        mass=torch.tensor(42.0),
        radius=torch.tensor(1.2 * scale),
        z_prime=torch.tensor(1, dtype=torch.long),
    )


def test_canonical_elj_route_crosses_the_expected_mxtaltools_boundary(
        monkeypatch, capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()  # derived-value summary is run provenance, not test output

    # This is the small argument mapping used by train.py's init_energy_function.
    # Importing the entire trainer here makes collection itself expensive and adds
    # unrelated dependencies; the operational contract separately pins the launch.
    energy_config = {
        'device': 'cpu',
        'energy_function': args.energy_function,
        'mlip_path': args.mlip_path,
        'space_groups': args.space_groups,
        'z_primes': args.z_primes,
        'sg_conditioning': args.sg_conditioning,
        'temperature_conditioning': args.temperature_conditioning,
        'zp_conditioning': args.zp_conditioning,
        'vector_conditioning': getattr(args, 'vector_conditioning', False),
        'vector_conditioning_dim': getattr(args, 'vector_conditioning_dim', None),
        'embedding_conditioning': getattr(args, 'embedding_conditioning', False),
        'embedding_conditioning_dim': getattr(
            args, 'embedding_conditioning_dim', None),
    }
    energy_config.update(vars(args.energy_config))
    energy_function = MolecularCrystal(**energy_config)

    assert args.energy_function == 'elj'
    assert args.space_groups == [2]
    assert args.z_primes == [1]
    assert energy_function.computes == ['reduction_en', 'elj']
    assert energy_function.computes_require_cluster is True
    assert energy_function.predictor is None

    calls = []
    captured = {}
    real_mol2cluster = MolCrystalBuilding.mol2cluster
    real_construct_radial_graph = MolCrystalAnalysis.construct_radial_graph

    def traced_mol2cluster(self, *call_args, **call_kwargs):
        calls.append('mol2cluster')
        return real_mol2cluster(self, *call_args, **call_kwargs)

    def traced_construct_radial_graph(self, *call_args, **call_kwargs):
        calls.append('construct_radial_graph')
        out = real_construct_radial_graph(self, *call_args, **call_kwargs)
        captured['cluster'] = self
        return out

    monkeypatch.setattr(MolCrystalBuilding, 'mol2cluster', traced_mol2cluster)
    monkeypatch.setattr(
        MolCrystalAnalysis, 'construct_radial_graph',
        traced_construct_radial_graph)

    molecules = collate_data_list([
        _synthetic_molecule('synthetic-a', 1.0),
        _synthetic_molecule('synthetic-b', 1.2),
    ])
    states = torch.tensor(
        [[0.0, 0.1, -0.1, 0.0, 0.0, 0.0,
          -0.2, 0.1, 0.3, 0.2, -0.1, 0.4],
         [0.2, -0.2, 0.1, 0.0, 0.0, 0.0,
          0.25, -0.1, 0.15, -0.3, 0.2, 0.35]],
        dtype=torch.float32,
    )
    temperature = torch.full((2,), 2.5)

    energy, scored = energy_function.analyze_crystal_batch(
        states, molecules, temperature, return_batch=True)

    assert calls == ['mol2cluster', 'construct_radial_graph']
    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(scored.elj).all()
    assert torch.isfinite(scored.reduction_en).all()
    torch.testing.assert_close(scored.gfn_energy, energy)

    # Pin the indexing semantics consumed by eLJ: source atoms come from periodic
    # image molecules, targets from the canonical asymmetric unit, and no edge may
    # cross graph boundaries or exceed the requested cutoff.
    cluster = captured['cluster']
    source, target = cluster.edges_dict['edge_index_inter']
    assert source.numel() > 0
    assert torch.all(cluster.aux_ind[source] == 1)
    assert torch.all(cluster.aux_ind[target] == 0)
    assert torch.all(cluster.batch[source] == cluster.batch[target])
    assert torch.all(cluster.mol_ind[source] != cluster.mol_ind[target])
    distances = (cluster.pos[source] - cluster.pos[target]).norm(dim=-1)
    assert torch.all(distances > 0)
    assert torch.all(distances <= 10.0 + 1e-5)
    assert torch.all(
        torch.bincount(cluster.batch[target], minlength=2) > 0)
