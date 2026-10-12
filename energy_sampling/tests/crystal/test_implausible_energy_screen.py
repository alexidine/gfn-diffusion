"""cfg:energy_config.implausible_energy_below -- an evaluation under the floor is scored as a clash, not kept.

The energy function is trusted on sensible structures; an MLIP on a senseless one can return anything. The acridine
MLE run (2026-10-05) scored one draw of its untrained model at -1,327,550 against a prior whose best row is -45; the
tracker kept that as the condition's best energy and every gate measured from it failed afterwards.

The live test runs the canonical eLJ route on CPU (the route tests/crystal/test_mxtaltools_crystal_boundary.py pins)
with the floor placed BETWEEN two real rows, so one is screened and the other is not, through the real
MolecularCrystal.analyze_crystal_batch.
"""
from pathlib import Path

import pytest
import torch

import utils
from energies.molecular_crystal import MolecularCrystal, screen_implausible_energy
from energy_sampling import prior_scan_cache as psc
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parents[2]


def _synthetic_molecule(identifier: str, scale: float) -> MolData:
    """The boundary test's molecule: tiny, non-collinear, rigid, with the graph attrs GFN consumes."""
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=identifier, mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'
# cell lengths chosen so neither synthetic crystal clashes: the eLJ energies are about -13.3 and -2.5
STATES = torch.tensor([[0.6, 0.6, 0.6, 0.0, 0.0, 0.0, -0.2, 0.1, 0.3, 0.2, -0.1, 0.4],
                       [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.25, -0.1, 0.15, -0.3, 0.2, 0.35]], dtype=torch.float32)


def test_the_rule():
    e = torch.tensor([-45.0, -1.3e6, 2.0e9, float('nan'), -1000.0, -1000.5])
    out, flagged = screen_implausible_energy(e, -1000.0)
    assert flagged.tolist() == [False, True, False, False, False, True], 'strictly under the floor, NaN untouched'
    assert out[[0, 2, 4]].tolist() == [-45.0, 2.0e9, -1000.0] and torch.isnan(out[3])
    assert out[[1, 5]].tolist() == [1000.0, 1000.0], 'finite, and far above any real structure'


def _energy_function(capsys, **extra):
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
    assert cfg['implausible_energy_below'] is None, 'mk_dev carries the key, off'
    cfg.update(extra)
    return MolecularCrystal(**cfg)


def _score(ef):
    mols = collate_data_list([_synthetic_molecule('synthetic-a', 1.0), _synthetic_molecule('synthetic-b', 1.2)])
    energy, scored = ef.analyze_crystal_batch(STATES.clone(), mols, torch.full((2,), 2.5), return_batch=True)
    return energy, scored.mol_energy.flatten()


def test_the_live_route_screens_the_row_under_the_floor_and_only_that_row(capsys):
    plain = _energy_function(capsys)
    e0, m0 = _score(plain)
    assert 'energy/implausible_rows' not in plain.drain_energy_timing(), 'no screen, no counter'
    lo, hi = sorted(m0.tolist())
    assert lo < hi < 0, f'the test needs two distinct negative energies: {m0.tolist()}'
    floor = (lo + hi) / 2                                  # between the rows
    screened = _energy_function(capsys, implausible_energy_below=floor)
    e1, m1 = _score(screened)
    low_row = int(torch.argmin(m0))
    keep_row = 1 - low_row
    assert float(m1[low_row]) == pytest.approx(abs(floor)), 'the row under the floor carries the magnitude'
    assert float(m1[keep_row]) == pytest.approx(float(m0[keep_row])), 'the other row is untouched'
    assert float(e1[keep_row]) == pytest.approx(float(e0[keep_row]), rel=1e-5)
    assert float(e1[low_row]) > float(e0[low_row]), 'so the total energy of the screened row went UP'
    assert screened.implausible_rows == 1 and float(m0[low_row]) < floor


def test_a_floor_that_is_not_negative_is_refused(capsys):
    with pytest.raises(ValueError, match='must be negative'):
        _energy_function(capsys, implausible_energy_below=1000.0)


def test_the_key_moves_neither_the_problem_identity_nor_the_scan_cache_identity(tmp_path):
    assert 'implausible_energy_below' in utils._NON_IDENTITY_ENERGY_CONFIG_KEYS
    import types
    prior = tmp_path / 'p.pt'
    prior.write_bytes(b'x')

    def ident(**ec):
        a = types.SimpleNamespace(prior_path=str(prior), energy_function='mace', mlip_path=None, space_groups=[14],
                                  z_primes=[1], energy_config=types.SimpleNamespace(temperature=2.5, **ec))
        return psc.scan_identity(a, 10, None)
    assert ident() == ident(implausible_energy_below=None) == ident(implausible_energy_below=-1000.0)
