"""
The reward side of energy_config.energy_reference, on the canonical ELJ route with two
synthetic molecules (the CPU fixture of test_mxtaltools_crystal_boundary.py).

A per-condition reference must shift each row's energy by exactly its condition's
constant and touch nothing else: the raw molecular energy stays raw, the absolute-floor
add-back restores the unreferenced log R, scoring without an installed table refuses
unless explicitly allowed, and a resume must match the checkpoint's reference.

Run from ``energy_sampling/``:
    python -m pytest -q tests/crystal/test_energy_reference_reward.py
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import utils
from energies.molecular_crystal import MolecularCrystal, resolve_energy_reference
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'
TABLE = torch.tensor([-400.0, -250.0])     # E_ref for condition 0 and condition 1
STATES = torch.tensor(
    [[0.0, 0.1, -0.1, 0.0, 0.0, 0.0, -0.2, 0.1, 0.3, 0.2, -0.1, 0.4],
     [0.2, -0.2, 0.1, 0.0, 0.0, 0.0, 0.25, -0.1, 0.15, -0.3, 0.2, 0.35]],
    dtype=torch.float32)


def _energy_function(capsys, **overrides):
    """train.py's init_energy_function mapping, as the boundary test spells it."""
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    cfg = {
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
        'embedding_conditioning_dim': getattr(args, 'embedding_conditioning_dim', None),
    }
    cfg.update(vars(args.energy_config))
    cfg.update(overrides)
    ef = MolecularCrystal(**cfg)
    ef.set_n_molecules(2)
    return ef


def _molecule(identifier: str, scale: float) -> MolData:
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]],
                       dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos,
                   x=torch.zeros((3, 1), dtype=torch.float32), identifier=identifier,
                   mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _molecules(condition_ids):
    b = collate_data_list([_molecule('synthetic-a', 1.0), _molecule('synthetic-b', 1.2)])
    b.add_graph_attr(torch.tensor(condition_ids, dtype=torch.long), 'condition_id')
    return b


def _temperature(ef):
    return torch.full((2,), float(ef.temperature))


def test_reference_shifts_each_row_by_its_own_conditions_constant(capsys):
    raw = _energy_function(capsys)
    ref = _energy_function(capsys, energy_reference='seed_min')
    ref.set_energy_reference(TABLE)
    cids = [1, 0]   # row 0 is condition 1: the lookup must follow the row, not the position
    e_raw, scored = raw.analyze_crystal_batch(STATES, _molecules(cids), _temperature(raw), return_batch=True)
    e_ref, _ = ref.analyze_crystal_batch(STATES, _molecules(cids), _temperature(ref), return_batch=True)
    torch.testing.assert_close(e_ref, e_raw - TABLE[cids])

    # the stored-row path: only the physical leg moves, the molecular energy stays raw
    lr_raw, ens_raw = raw.prebuilt_sample_to_reward(scored, _temperature(raw), return_ens_dict=True)
    lr_ref, ens_ref = ref.prebuilt_sample_to_reward(scored, _temperature(ref), return_ens_dict=True)
    torch.testing.assert_close(ens_ref['mol_energy'], ens_raw['mol_energy'])
    torch.testing.assert_close(ens_ref['physical_energy'], ens_raw['physical_energy'] - TABLE[cids])
    torch.testing.assert_close(lr_ref, lr_raw + TABLE[cids] / float(ref.temperature))
    torch.testing.assert_close(ens_ref['energy_reference'], TABLE[cids])
    assert 'energy_reference' not in ens_raw

    # the absolute floors' add-back gives back exactly the unreferenced log R
    torch.testing.assert_close(ref.unreferenced_log_r(lr_ref, scored.condition_id), lr_raw)
    assert raw.unreferenced_log_r(lr_raw, None) is lr_raw
    with pytest.raises(ValueError, match='condition_id'):
        ref.unreferenced_log_r(lr_ref, None)


def test_scoring_without_the_table_refuses_unless_allowed(capsys):
    raw = _energy_function(capsys)
    ref = _energy_function(capsys, energy_reference='seed_min')
    with pytest.raises(RuntimeError, match='no reference table is installed'):
        ref.analyze_crystal_batch(STATES, _molecules([0, 1]), _temperature(ref))
    with ref.allow_unreferenced():
        e = ref.analyze_crystal_batch(STATES, _molecules([0, 1]), _temperature(ref))
    torch.testing.assert_close(e, raw.analyze_crystal_batch(STATES, _molecules([0, 1]), _temperature(raw)))
    assert ref._unreferenced_scoring_ok is False     # the allowance ends with the block


def test_seed_energy_is_the_crystal_energy_without_jacobian_or_penalties(capsys):
    ef = _energy_function(capsys)
    ens = {'mol_energy': torch.tensor([-100.0]), 'density_energy': torch.tensor([0.5]),
           'pressure_energy': torch.tensor([0.25]), 'jacobian_energy': torch.tensor([-30.0]),
           'reduction_energy': torch.tensor([7.0])}
    want = -100.0 + ef.density_coeff * 0.5 + 0.25
    assert float(ef.seed_energy_from(ens)) == pytest.approx(want)


def test_condition_ids_follow_condition_samples(capsys):
    ef = _energy_function(capsys, energy_reference='seed_min')
    assert ef.condition_ids_of(SimpleNamespace(condition_id=torch.tensor([1, 0]))).tolist() == [1, 0]
    by_mol = SimpleNamespace(mol_id=torch.tensor([0, 1]), sg_ind=torch.tensor([2, 2]),
                             z_prime=torch.tensor([1, 1]))
    assert ef.condition_ids_of(by_mol).tolist() == [0, 1]      # one space group, one Z'
    with pytest.raises(ValueError, match='neither condition_id nor mol_id'):
        ef.condition_ids_of(SimpleNamespace(sg_ind=torch.tensor([2]), z_prime=torch.tensor([1])))
    with pytest.raises(ValueError, match='outside'):
        ef.condition_ids_of(SimpleNamespace(mol_id=torch.tensor([0]), sg_ind=torch.tensor([14]),
                                            z_prime=torch.tensor([1])))


def test_table_and_constructor_refusals(capsys):
    ef = _energy_function(capsys, energy_reference='seed_min')
    with pytest.raises(ValueError, match='entries'):
        ef.set_energy_reference(torch.zeros(3))
    with pytest.raises(ValueError, match='no finite reference'):
        ef.set_energy_reference(torch.tensor([0.0, float('nan')]))
    with pytest.raises(ValueError, match='energy_reference: null'):
        _energy_function(capsys).set_energy_reference(TABLE)
    with pytest.raises(ValueError, match='must be one of'):
        _energy_function(capsys, energy_reference='seed_mean')
    with pytest.raises(ValueError, match='temperature_conditioning'):
        _energy_function(capsys, energy_reference='seed_min', temperature_conditioning=True)


def test_resume_must_match_the_checkpoints_reference():
    fresh = TABLE.clone()
    T = 6.9
    calls = []

    def compute():
        calls.append(1)
        return fresh

    assert resolve_energy_reference(None, None, compute, T) is None and not calls
    assert resolve_energy_reference(None, {'mode': None, 'table': None}, compute, T) is None
    torch.testing.assert_close(resolve_energy_reference('seed_min', None, compute, T), fresh)

    for live, stored_mode in (('seed_min', None), (None, 'seed_min')):
        with pytest.raises(ValueError, match='trained with'):
            resolve_energy_reference(live, {'mode': stored_mode, 'table': fresh}, compute, T)

    # the recompute's float noise, as measured on a real resume (0.018 units at T 6.9):
    # accepted, and the STORED table is what gets installed
    stored = fresh + 0.018
    out = resolve_energy_reference('seed_min', {'mode': 'seed_min', 'table': stored}, compute, T)
    torch.testing.assert_close(out, stored)
    with pytest.raises(ValueError, match='moved'):          # 0.5 units = 0.07 nats at T 6.9
        resolve_energy_reference('seed_min', {'mode': 'seed_min', 'table': fresh + 0.5}, compute, T)
    with pytest.raises(ValueError, match='missing or sized'):
        resolve_energy_reference('seed_min', {'mode': 'seed_min', 'table': None}, compute, T)
    with pytest.raises(ValueError, match='missing or sized'):
        resolve_energy_reference('seed_min', {'mode': 'seed_min', 'table': torch.zeros(3)}, compute, T)
