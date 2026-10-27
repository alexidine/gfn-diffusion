"""MolecularCrystal(crystal_coordinates='fd'): the energy function scores crystal_fd chart rows s in [-1, 1]^d.

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_fd_energy_option.py

What each test would catch:

  * the default no longer 'latent', or an unknown value accepted;
  * an FD row scored as another crystal than the one it charts: the latent route's crystals, charted to s and scored
    through the FD route, must give the same eLJ;
  * a reduction penalty on FD rows (the chart is the reduced set) or a bounding term inside [-1, 1];
  * the change of measure not being -T * fd_coordinates.log_reference;
  * a non-finite gradient of the reward with respect to s;
  * Z' > 1 or several space groups accepted; the state width not the chart's.
"""
import pytest
import torch

import utils
from energies.molecular_crystal import CRYSTAL_COORDINATES, MolecularCrystal
from mxtaltools.dataset_utils.utils import collate_data_list

from pathlib import Path

from mxtaltools.dataset_utils.data_classes import MolData

CANONICAL = Path(__file__).resolve().parents[2] / 'configs' / 'mk_dev.yaml'


def _synthetic_molecule(identifier: str, scale: float) -> MolData:
    """As tests/crystal/test_mxtaltools_crystal_boundary.py: a tiny non-collinear rigid molecule."""
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=identifier, mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _energy(capsys, **over):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    cfg = {'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
           'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
           'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning}
    cfg.update(vars(args.energy_config))
    cfg.update(over)
    return MolecularCrystal(**cfg)


def _molecules(n):
    return collate_data_list([_synthetic_molecule(f'synthetic-{i}', 1.6) for i in range(n)])


def test_default_and_validation(capsys):
    import inspect
    assert inspect.signature(MolecularCrystal).parameters['crystal_coordinates'].default == 'latent'
    assert CRYSTAL_COORDINATES == ('latent', 'fd')
    with pytest.raises(ValueError, match='crystal_coordinates'):
        _energy(capsys, crystal_coordinates='cartesian')


def test_fd_rows_score_the_same_crystals_as_the_latent_route(capsys):
    lat = _energy(capsys)
    fd = _energy(capsys, crystal_coordinates='fd')
    n = 16
    g = torch.Generator().manual_seed(3)
    states = (torch.rand(n, 12, generator=g) * 1.6 - 0.8)
    states[:, :3] = 0.75 + 0.2 * torch.rand(n, 3, generator=g)  # long cells: physical packing (an overlapping
    # crystal's eLJ depends on its description through the neighbour cap, so it is no test of the route)
    T = torch.full((n,), 2.5)
    mols = _molecules(n)
    _, scored = lat.analyze_crystal_batch(states, mols, T, return_batch=True)
    assert float(scored.packing_coeff.max()) < 1.5, scored.packing_coeff
    fc = fd.fd_coordinates_for(2, float(scored.radius[0]))
    rows = fc.from_params(scored.full_cell_parameters(), scored.aunit_handedness.reshape(-1))
    ok = rows.ok
    assert int(ok.sum()) >= 4, rows.status  # the rest: canonical cell outside the per-axis length box (D7)
    s = rows.s[ok].float()
    energy, scored_fd = fd.analyze_crystal_batch(s, _molecules(int(ok.sum())), T[ok], return_batch=True)
    assert torch.isfinite(energy).all()
    torch.testing.assert_close(scored_fd.elj, scored.elj[ok], rtol=2e-3, atol=2e-2)
    assert float(scored_fd.reduction_energy.abs().max()) < 1e-3
    assert float(scored_fd.bounding_energy.abs().max()) == 0.0
    _, _, log_ref = fc.to_params(s.double())
    torch.testing.assert_close(scored_fd.jacobian_energy, (-T[ok] * log_ref.float()), rtol=1e-5, atol=1e-4)


def test_reward_gradient_in_s_is_finite(capsys):
    fd = _energy(capsys, crystal_coordinates='fd')
    n = 8
    g = torch.Generator().manual_seed(4)
    s = (torch.rand(n, 12, generator=g) * 1.8 - 0.9).requires_grad_(True)
    energy, _ = fd.analyze_crystal_batch(s, _molecules(n), torch.full((n,), 2.5), keep_grads=True)
    energy.sum().backward()
    assert s.grad is not None and bool(torch.isfinite(s.grad).all())
    assert float(s.grad.abs().sum()) > 0


def test_fd_needs_one_space_group_and_z_prime_1(capsys):
    with pytest.raises(NotImplementedError, match="Z' = 1"):
        _energy(capsys, crystal_coordinates='fd', z_primes=(1, 2))
    with pytest.raises(NotImplementedError, match='one space group'):
        _energy(capsys, crystal_coordinates='fd', space_groups=[2, 14])


def test_fd_layout_sets_the_state_width(capsys):
    fd = _energy(capsys, crystal_coordinates='fd', space_groups=[14])
    assert fd.data_ndim == 10 and fd.fd_layout.dim == 10
    assert _energy(capsys).data_ndim == 12
