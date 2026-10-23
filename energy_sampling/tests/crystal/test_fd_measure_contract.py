"""GFN boundary of the MXtalTools fundamental-domain package (fd_tooling.md gates 3, 4 and 8; MXtalTools crystal_fd).

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_fd_measure_contract.py

Pins four facts about crystals built by the canonical ``mk_dev.yaml`` route
(``MolecularCrystal.instantiate_crystals`` -> ``MolCrystalData.latent_to_cell_params``):

1. ``crystal_fd.charts.LatentV1Chart`` describes the crystal that route builds, bit for bit, and its clamp mask is
   exactly the rows whose latents the route clamps.
2. The description is the built crystal, atom by atom: the route's posed asymmetric unit (``pose_aunit`` with
   ``std_orientation=False``, as ``analyze`` builds it) is T t + R X within 1e-3 Angstrom, X the molecule centred as
   ``crystal_fd.description.MoleculeRef`` centres it; with R transposed it is more than 0.1 Angstrom off.
3. ``crystal_fd.measure.log_reference_density(desc, LatentV1Chart, 'latent_v1_lebesgue')`` equals
   ``-jacobian_energy / T`` from ``MolecularCrystal.compute_jacobian`` up to one constant, including rows at the
   theta and r floors where the log terms are largest. The constant is predicted from the latent ranges documented
   in ``MolCrystalOps.latent_transform`` and ``lat2sph_rotvec``, the Haar normalisation log(2 pi^2) and log
   sym_mult. Rows clamped to latent r = +1 (r = 2 pi, the identity rotation, Haar density 0, where
   compute_jacobian floors sin(r/2) at 1e-8) are set aside and counted.
4. Gate 4 through MolecularCrystal: every Gamma+ image of a dense sg 2 crystal inside the trainer box, and its fold,
   score the same eLJ through ``analyze_crystal_batch`` (relative 1e-3 or absolute 5e-3 raw, whichever is larger);
   an improper element read at handedness +1 moves the eLJ by 0.1 raw or more (all rows but one; median 0.5).

CPU, synthetic molecules, no data drive.
"""
import math
from pathlib import Path

import torch

import utils
from energies.molecular_crystal import MolecularCrystal
from mxtaltools.constants.asymmetric_units import ASYM_UNITS
from mxtaltools.crystal_fd import charts, domains, measure, ops, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_fd.tables import group_table
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list

HERE = Path(__file__).resolve().parents[2]   # tests/<area>/x.py -> energy_sampling/
CANONICAL = HERE / 'configs' / 'mk_dev.yaml'
#: the clamp as documented in MolCrystalOps.latent_to_cell_params (Z' = 1 layout)
DOC_LO = torch.tensor([-0.99] * 3 + [-1.0] * 6 + [-0.99, -1.0, -0.99])
DOC_HI = torch.tensor([1 - 1e-4] * 2 + [1.0] * 10)
F64 = torch.float64


def _molecule(identifier: str, scale: float) -> MolData:
    pos = torch.tensor([[-0.8, -0.2, 0.0], [0.7, -0.1, 0.1], [0.1, 0.8, -0.1]], dtype=torch.float32) * scale
    return MolData(z=torch.tensor([6, 7, 8], dtype=torch.long), pos=pos, x=torch.zeros((3, 1), dtype=torch.float32),
                   identifier=identifier, mol_volume=torch.tensor(55.0 * scale ** 3), mass=torch.tensor(42.0),
                   radius=torch.tensor(1.2 * scale), z_prime=torch.tensor(1, dtype=torch.long))


def _energy_function(capsys):
    args = utils.get_train_args(['--config', str(CANONICAL)])
    capsys.readouterr()
    energy_config = {
        'device': 'cpu', 'energy_function': args.energy_function, 'mlip_path': args.mlip_path,
        'space_groups': args.space_groups, 'z_primes': args.z_primes, 'sg_conditioning': args.sg_conditioning,
        'temperature_conditioning': args.temperature_conditioning, 'zp_conditioning': args.zp_conditioning,
        'vector_conditioning': getattr(args, 'vector_conditioning', False),
        'vector_conditioning_dim': getattr(args, 'vector_conditioning_dim', None),
        'embedding_conditioning': getattr(args, 'embedding_conditioning', False),
        'embedding_conditioning_dim': getattr(args, 'embedding_conditioning_dim', None),
    }
    energy_config.update(vars(args.energy_config))
    return MolecularCrystal(**energy_config), args


def _latents(B, seed):
    """B latents, uniform in the clamp box, with rows at the theta and r floors and rows outside the clamp box."""
    g = torch.Generator().manual_seed(seed)
    x = (-0.99 + 1.98 * torch.rand(B, 12, generator=g)).float()
    x[:60, 9] = -0.99                                  # theta at its floor, pi / 400
    x[60:120, 11] = -0.99                              # r at its floor, pi / 100
    x[120:180] = (-1.3 + 2.6 * torch.rand(60, 12, generator=g)).float()   # outside the clamp box
    return x


def documented_log_slopes(sg: int) -> float:
    """Sum of log |d(natural coordinate) / d(latent)| over the live latent rows, from the documented ranges: each
    log-length spans log 40 over 2 latent units, each angle 0.6 pi, each non-free centroid ASYM_UNITS[sg] (P-1 has no
    free axis), theta pi / 2, phi 2 pi, r 2 pi."""
    assert sg == 2
    box = [float(v) for v in ASYM_UNITS[str(sg)]]
    return (3 * math.log(math.log(40) / 2) + 3 * math.log(0.3 * math.pi) + sum(math.log(b / 2) for b in box)
            + math.log(math.pi / 4) + 2 * math.log(math.pi))


def test_reference_density_is_the_trainer_jacobian(capsys):
    energy_function, args = _energy_function(capsys)
    assert args.space_groups == [2] and args.z_primes == [1]
    sg = 2
    B = 600
    mols = collate_data_list([_molecule(f'm{i}', 1.0 + 0.2 * (i % 3)) for i in range(B)])
    x = _latents(B, 0)
    crystals = energy_function.instantiate_crystals(x, mols)

    chart = charts.LatentV1Chart(sg, mols.radius)
    params, clamped, _ = chart.cell_params(x)
    for name, sl in (('cell_lengths', slice(0, 3)), ('cell_angles', slice(3, 6)), ('aunit_centroid', slice(6, 9)),
                     ('aunit_orientation', slice(9, 12))):
        assert torch.equal(params[:, sl], getattr(crystals, name)), name
    assert torch.equal(clamped, ((x < DOC_LO) | (x > DOC_HI)).any(1))
    assert not bool(clamped[180:].any()) and bool(clamped[120:180].any())

    T = torch.full((B,), 2.5)
    jac, _ = energy_function.compute_jacobian(crystals, T)
    gfn = (-jac / T).double()
    desc, _ = chart.describe(x)
    ours = measure.log_reference_density(desc, chart, 'latent_v1_lebesgue')
    real = torch.isfinite(measure.cell_volume(desc.cell))   # angle triples with a real cell
    # latent r = +1 (reachable: the clamp's upper bound) is r = 2 pi, the identity, where the Haar density is 0
    # and compute_jacobian floors sin(r/2) at 1e-8; the two log-densities part there by construction
    identity = crystals.clone().latent_params()[:, 11] >= 1 - 1e-6
    assert int(identity.sum()) > 0
    keep = real & ~identity
    assert int(keep.sum()) > B // 2
    diff = (ours - gfn)[keep]
    assert float(diff.max() - diff.min()) <= 1e-4, float(diff.max() - diff.min())
    const = -documented_log_slopes(sg) + math.log(2 * math.pi ** 2) - math.log(int(crystals.sym_mult[0]))
    assert abs(float(diff.mean()) + const) <= 1e-4, (float(diff.mean()), const)


def test_described_crystals_are_the_built_crystals(capsys):
    energy_function, _ = _energy_function(capsys)
    B = 600
    mols = collate_data_list([_molecule(f'm{i}', 1.0 + 0.2 * (i % 3)) for i in range(B)])
    x = _latents(B, 1)
    crystals = energy_function.instantiate_crystals(x, mols)
    desc, _ = charts.LatentV1Chart(2, mols.radius).describe(x)
    real = torch.isfinite(measure.cell_volume(desc.cell))
    assert int(real.sum()) > B // 2
    posed = crystals.clone()
    posed.pose_aunit(std_orientation=False)
    built = posed.pos.double().reshape(B, 3, 3)
    X = torch.stack([MoleculeRef.from_coords(mols.pos[mols.batch == i], mols.z[mols.batch == i]).pos
                     for i in range(B)])
    T = desc.cell_matrix()

    def deviation(R):
        pred = (T @ desc.t[:, 0, :, None])[:, None, :, 0] + X @ R.transpose(1, 2)
        return (built - pred).norm(dim=-1).amax(1)

    dev = deviation(desc.R[:, 0])[real]
    assert float(dev.max()) <= 1e-3, float(dev.max())
    moved = real & ((desc.R[:, 0] - desc.R[:, 0].transpose(1, 2)).abs().amax((1, 2)) > 0.2)
    assert int(moved.sum()) > B // 4
    assert float(deviation(desc.R[:, 0].transpose(1, 2))[moved].min()) > 0.1


def _dense_crystals(n, scales, g):
    """n sg 2 crystals of the synthetic molecules at roughly packing 0.6 (cell edges 0.9-1.2 x (2 V_mol / 0.6)^(1/3)),
    angles 85-100 deg, uniform centroids and Haar rotations."""
    edge = (2 * 55.0 * scales ** 3 / 0.6) ** (1 / 3)
    L = edge[:, None] * (0.9 + 0.3 * torch.rand(n, 3, generator=g, dtype=F64))
    A = torch.deg2rad(85 + 15 * torch.rand(n, 3, generator=g, dtype=F64))
    return CrystalDescription.from_parts(2, torch.cat([L, A], 1), torch.rand(n, 3, generator=g, dtype=F64),
                                         so3.random_rotations(n, g))


def test_gamma_plus_images_and_folds_score_alike(capsys):
    energy_function, _ = _energy_function(capsys)
    sg, n = 2, 24
    tab = group_table(sg)
    g = torch.Generator().manual_seed(2)
    scales = 1.0 + 0.2 * (torch.arange(n) % 3).to(F64)
    d = _dense_crystals(n, scales, g)
    imgs, src, _ = domains.images(d, within='latent_v1_box')
    assert torch.equal(torch.bincount(src, minlength=n), torch.full((n,), tab.k_box))
    order = torch.argsort(src, stable=True)
    imgs, src = imgs[order], src[order]
    canon = domains.fold(d).desc
    k = int(torch.nonzero(tab.gamma_det < 0)[0, 0])                 # an improper element, read at +1
    wrong = ops.apply(canon, tab.gamma_W[k], tab.gamma_w[k], wrap=True)
    wrong = domains.fold(wrong.replace(h=torch.ones_like(wrong.h))).desc
    rows = CrystalDescription.cat([imgs, canon, wrong])
    which = torch.cat([src, torch.arange(n), torch.arange(n)])
    mols = collate_data_list([_molecule(f'm{int(i)}', float(scales[i])) for i in which])
    chart = charts.LatentV1Chart(sg, mols.radius)
    lat = chart.latents(rows)
    assert not bool(chart.cell_params(lat)[1].any()), 'a description left the clamp box'
    _, scored = energy_function.analyze_crystal_batch(lat, mols, torch.full((len(which),), 2.5), return_batch=True)
    e = scored.elj.double()
    e_img, e_canon, e_wrong = e[:len(src)], e[len(src):len(src) + n], e[len(src) + n:]
    ref = e_img[torch.searchsorted(src, torch.arange(n))]              # the first box image of each crystal
    tol = torch.clamp(1e-3 * ref.abs(), min=5e-3)
    assert bool(((e_img - ref[src]).abs() <= tol[src]).all()), float((e_img - ref[src]).abs().max())
    assert bool(((e_canon - ref).abs() <= tol).all()), float((e_canon - ref).abs().max())
    assert float(ref.abs().median()) >= 1.0                           # dense enough that a pose matters
    dE = (e_wrong - ref).abs()
    assert int((dE >= 0.1).sum()) >= n - 1 and float(dE.median()) >= 0.5, (float(dE.min()), float(dE.median()))
