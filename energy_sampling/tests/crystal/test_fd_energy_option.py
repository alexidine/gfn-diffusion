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
  * Z' > 1 accepted; the state width not the chart's;
  * several space groups in one batch: a row scored through another group's chart, a pad slot's value read or not
    refused, the slot layout of one group not its chart, a circle slot that is an interval in some group, a stored
    row not round-tripping into its slots.
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


def test_fd_needs_z_prime_1(capsys):
    with pytest.raises(NotImplementedError, match="Z' = 1"):
        _energy(capsys, crystal_coordinates='fd', z_primes=(1, 2))


def test_fd_slot_layout():
    from energies.fd_coordinates import PAD, FDCoordinates, FDSlotLayout
    one, fc = FDSlotLayout([14]), FDCoordinates(14, 4.0)   # one group: the state is its chart
    assert (one.dim, one.sg, one.slots[14].tolist()) == (fc.dim, 14, list(range(fc.dim)))
    assert one.wrap_mask == fc.wrap_mask and one.names == fc.names
    two = FDSlotLayout([14, 2, 14])
    assert (two.groups, two.dim, two.sg, two.block_width) == ((2, 14), 12, None, [6, 3, 3])
    assert two.group_names[14] == ['log_c', 'log_a', 'log_b', 's', PAD, PAD, 't_x', 't_y', 't_z', 'theta', 'phi', 'r']
    # a slot is a circle iff it is one in every group whose row it carries: t_z is an interval in sg 14
    assert [n for n, w in zip(two.names, two.wrap_mask) if w] == ['t_x', 't_y', 'phi']
    assert FDSlotLayout([2, 19]).wrap_mask[6:9] == [False, False, True]   # sg 19: t_x, t_y intervals, t_z a circle
    live = two.live_mask(torch.tensor([2, 14]))
    assert live[0].all() and live[1].tolist() == [True] * 4 + [False] * 2 + [True] * 6
    s = torch.rand(3, 10)
    torch.testing.assert_close(two.unpack(14, two.pack(14, s)), s)
    assert bool((two.pack(14, s)[:, 4:6] == 0).all())
    with pytest.raises(ValueError, match='not in this FD layout'):
        two.live_mask(torch.tensor([19]))


def test_mixed_groups_score_each_row_through_its_own_chart(capsys):
    """One batch of sg 2 and sg 14 rows scores each row as the one-group route scores it, refuses a value in a pad
    slot, and a stored row goes back into its own slots (fd_rows_of, FDChartState)."""
    from buffer import FDChartState
    mixed = _energy(capsys, crystal_coordinates='fd', space_groups=[2, 14])
    lay = mixed.fd_layout
    assert mixed.data_ndim == 12 and lay.groups == (2, 14)
    sgs = torch.tensor([2, 14, 2, 14, 14, 2])
    n = len(sgs)
    g = torch.Generator().manual_seed(5)
    states = torch.zeros(n, 12)
    for i, sg in enumerate(sgs.tolist()):
        s = torch.rand(1, len(lay.slots[sg]), generator=g) * 1.6 - 0.8
        s[:, 0] = 0.85   # the longest axis long: physical packing
        states[i] = lay.pack(sg, s)[0]
    T = torch.full((n,), 2.5)
    mols = _molecules(n)
    mols.sg_ind = sgs
    energy, scored = mixed.analyze_crystal_batch(states, mols, T, return_batch=True)
    assert scored.sg_ind.reshape(-1).tolist() == sgs.tolist() and torch.isfinite(energy).all()
    for sg in (2, 14):
        rows = sgs == sg
        single = _energy(capsys, crystal_coordinates='fd', space_groups=[sg])
        m1 = _molecules(int(rows.sum()))
        m1.sg_ind = sgs[rows]
        e1, b1 = single.analyze_crystal_batch(lay.unpack(sg, states[rows]), m1, T[rows], return_batch=True)
        torch.testing.assert_close(scored.full_cell_parameters()[rows], b1.full_cell_parameters())
        torch.testing.assert_close(energy[rows], e1, rtol=1e-4, atol=1e-3)
    bad = states.clone()
    bad[1, 4] = 0.3   # a pad slot of the sg 14 row
    with pytest.raises(ValueError, match='pad slots'):
        mixed.analyze_crystal_batch(bad, mols, T)
    delattr(scored, 'fd_state')
    torch.testing.assert_close(mixed.fd_rows_of(scored), states, atol=2e-4, rtol=0)
    torch.testing.assert_close(FDChartState(12, lay.groups)(scored), states, atol=2e-4, rtol=0)


def test_fd_layout_sets_the_state_width(capsys):
    fd = _energy(capsys, crystal_coordinates='fd', space_groups=[14])
    assert fd.data_ndim == 10 and fd.fd_layout.dim == 10
    assert _energy(capsys).data_ndim == 12


def _gfn_config_stub(fd_layout):
    """A Modeller stand-in carrying what _build_gfn_config reads; the latent periodic-centroid resolution raises."""
    import types
    from types import SimpleNamespace
    from energy_sampling.train import Modeller

    def no_latent_axes():
        raise AssertionError('latent periodic-centroid axes resolved')

    m = SimpleNamespace(
        energy_function=SimpleNamespace(fd_layout=fd_layout, data_ndim=12, is_crystal=True),
        args=SimpleNamespace(molecule_conditioning=False, temperature_conditioning=False, sg_conditioning=False,
                             zp_conditioning=False, z_primes=[1], model=SimpleNamespace(periodic_centroids=True)),
        device='cpu', _state_features_dim=lambda: 0, get_conditioning_dim=lambda: 0, _policy_entry=lambda: {},
        _resolve_dead_latent_rows=lambda: None, _resolve_periodic_centroid_axes=no_latent_axes)
    for name in ('_build_gfn_config', '_build_latent_gfn_config'):
        setattr(m, name, types.MethodType(getattr(Modeller, name), m))
    return m


def test_fd_gfn_config_uses_the_chart_wrap_rows_not_latent_centroid_axes(capsys):
    from energies.fd_coordinates import FDCoordinates
    layout = FDCoordinates(2, 4.0)
    cfg = _gfn_config_stub(layout)._build_gfn_config()
    assert cfg['periodic_centroid_axes'] is None and cfg['dead_latent_rows'] is None
    assert cfg['dim'] == layout.dim and cfg['angular_mask'] == list(layout.wrap_mask)
    out = capsys.readouterr().out
    assert "['t_x', 't_y', 't_z', 'phi']" in out and 'periodic_centroids is not read' in out
    with pytest.raises(AssertionError, match='latent periodic-centroid'):
        _gfn_config_stub(None)._build_gfn_config()   # the latent route still resolves them
