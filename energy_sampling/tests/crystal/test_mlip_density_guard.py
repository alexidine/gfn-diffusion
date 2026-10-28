"""cfg:energy_config.mlip_max_packing_coeff -- a cell far denser than any crystal is kept from the MLIP and scored as a
clash.

MACE's neighbour list grows with the density; an untrained exact-density flow on acridine (inn_acr_oct23, 2026-10-23)
drew cells up to ~600x denser than a crystal and its first evaluation ran the card out of memory at one crystal per
call. The constructor accepts the key on the mace route only, so the split itself is exercised here on the canonical
eLJ route on CPU (the route tests/crystal/test_mxtaltools_crystal_boundary.py pins), by setting the guard on a built
energy function: the backend does not enter the split, and the energy key is the route's own.
"""
from pathlib import Path

import pytest
import torch

import utils
from energies.molecular_crystal import MolecularCrystal
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parents[2]
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'
# three rows; the middle one has much shorter cell lengths, so it is the densest
STATES = torch.tensor([[0.6, 0.6, 0.6, 0.0, 0.0, 0.0, -0.2, 0.1, 0.3, 0.2, -0.1, 0.4],
                       [-0.6, -0.6, -0.6, 0.0, 0.0, 0.0, 0.1, 0.2, -0.1, 0.3, 0.1, -0.2],
                       [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.25, -0.1, 0.15, -0.3, 0.2, 0.35]], dtype=torch.float32)


def _molecule(identifier: str) -> MolData:
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32)
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=identifier, mol_volume=torch.tensor(55.0), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2), z_prime=torch.tensor(1, dtype=torch.long))


def _cfg(capsys, **extra):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning,
           'vector_conditioning': getattr(args, 'vector_conditioning', False),
           'vector_conditioning_dim': getattr(args, 'vector_conditioning_dim', None),
           'embedding_conditioning': getattr(args, 'embedding_conditioning', False),
           'embedding_conditioning_dim': getattr(args, 'embedding_conditioning_dim', None)}
    cfg.update(vars(args.energy_config))
    assert cfg['mlip_max_packing_coeff'] is None, 'mk_dev carries the key, off'
    cfg.update(extra)
    return cfg


def _score(ef):
    mols = collate_data_list([_molecule(f'synthetic-{k}') for k in range(3)])
    energy, scored = ef.analyze_crystal_batch(STATES.clone(), mols, torch.full((3,), 2.5), return_batch=True)
    return energy, scored


def test_the_guard_keeps_dense_rows_from_the_backend_and_leaves_the_others_and_their_order(capsys):
    plain = MolecularCrystal(**_cfg(capsys))
    key = plain.energy_function
    e0, b0 = _score(plain)
    pc = b0.packing_coeff.flatten()
    dense_row = int(torch.argmax(pc))
    assert dense_row == 1, f'the test needs the middle row densest: {pc.tolist()}'
    others = [0, 2]
    cap = (float(pc[dense_row]) + float(pc[others].max())) / 2   # between the densest row and the rest
    assert 'energy/mlip_guarded_rows' not in plain.drain_energy_timing(), 'no guard, no counter'

    guarded = MolecularCrystal(**_cfg(capsys, implausible_energy_below=-1000.0))
    guarded.mlip_max_packing_coeff = cap                          # the constructor admits it on the mace route only
    e1, b1 = _score(guarded)
    raw0, raw1 = getattr(b0, key).flatten(), getattr(b1, key).flatten()
    assert float(raw1[dense_row]) == pytest.approx(1000.0), 'the dense row carries the clash value'
    assert torch.allclose(raw1[others], raw0[others]), 'the other rows keep their own energies, in their own places'
    assert torch.allclose(b1.reduction_en.flatten(), b0.reduction_en.flatten()), 'the reduction penalty is still scored'
    assert torch.allclose(e1[others], e0[others], rtol=1e-5)
    assert guarded.mlip_guarded_rows == 1
    guarded.energy_calls = 1  # drain_energy_timing reports only after a timed call, which log_reward records
    assert guarded.drain_energy_timing()['energy/mlip_guarded_rows'] == 1.0 and guarded.mlip_guarded_rows == 0


def test_a_batch_of_only_dense_rows(capsys):
    guarded = MolecularCrystal(**_cfg(capsys, implausible_energy_below=-1000.0))
    guarded.mlip_max_packing_coeff = 1e-9
    _, b = _score(guarded)
    assert getattr(b, guarded.energy_function).flatten().tolist() == [1000.0] * 3
    assert guarded.mlip_guarded_rows == 3


def test_a_guarded_chunk_appends_to_an_unguarded_one(capsys):
    """batched_analyze_crystal_batch appends its chunks' batches; analyze binds `computes` on the batch it runs on, so
    a chunk whose analysis ran only on sub-batches must carry it too (the first local acridine run died on this)."""
    ef = MolecularCrystal(**_cfg(capsys, implausible_energy_below=-1000.0))
    ef.mlip_max_packing_coeff = 1e-9
    mols = collate_data_list([_molecule(f'synthetic-{k}') for k in range(3)])
    plain = ef.instantiate_crystals(STATES[:1].clone(), mols.subsample_new_batch(torch.tensor([0])))
    plain.analyze(ef.computes, cutoff=10, supercell_size=10, std_orientation=False)
    guarded = ef.instantiate_crystals(STATES[1:].clone(), mols.subsample_new_batch(torch.tensor([1, 2])))
    ef.analyze_density_guarded(guarded, dict(cutoff=10, supercell_size=10, std_orientation=False))
    assert plain.append_batch(guarded).num_graphs == 3


def test_the_constructor_refuses_the_guard_off_the_mace_route_or_without_its_clash_value(capsys):
    with pytest.raises(ValueError, match='mace route'):
        MolecularCrystal(**_cfg(capsys, implausible_energy_below=-1000.0, mlip_max_packing_coeff=2.0))
    with pytest.raises(ValueError, match='set that too'):
        MolecularCrystal(**_cfg(capsys, energy_function='mace', mlip_max_packing_coeff=2.0))
    with pytest.raises(ValueError, match='positive'):
        MolecularCrystal(**_cfg(capsys, energy_function='mace', implausible_energy_below=-1000.0,
                                mlip_max_packing_coeff=0.0))


def test_the_key_is_not_part_of_the_problem_identity():
    assert 'mlip_max_packing_coeff' in utils._NON_IDENTITY_ENERGY_CONFIG_KEYS
