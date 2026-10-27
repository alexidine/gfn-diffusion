"""energies/fd_coordinates.py: s in [-1, 1]^d <-> Z' = 1 crystals of one space group (the AR option's coordinates).

Run from ``energy_sampling/``:

    python -m pytest -q tests/crystal/test_fd_coordinates.py

What each test would catch:

  * an s whose crystal does not come back to the same s (the chart, its unit map or the centre wrap broken), in a
    triclinic, monoclinic, sg 6-9 (K cut), tetragonal and trigonal group;
  * a built row that is not the canonical form of its crystal (s would then not be unique);
  * a stored row given in another description of the same crystal (a symmetry image, a lattice translation) charted
    to a different s;
  * log_reference not equal to the reference density on the chart plus log |dy/ds| (checked against autograd);
  * a circle coordinate whose two ends are not the same crystal (same canonical form);
  * cell parameters the MXtalTools builder rejects or builds as another crystal.
"""
import math

import pytest
import torch

# crystal_fd is in MXtalTools commits not yet on every checkout (the cluster's included): skip there, rather than
# halting the whole lane at collection
pytest.importorskip('mxtaltools.crystal_fd')

from energies.fd_coordinates import FDCoordinates  # noqa: E402
from mxtaltools.crystal_fd import canon, charts, measure  # noqa: E402

F64 = torch.float64
GROUPS = [1, 2, 7, 14, 92, 148]
RADIUS = 4.0


def _s(fc, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, fc.dim, generator=g, dtype=F64) * 2 - 1


def _circle_err(fc, a, b):
    e = (a - b).abs()
    wm = torch.tensor(fc.wrap_mask)
    e[:, wm] = torch.minimum(e[:, wm], 2 - e[:, wm])
    return e


@pytest.mark.parametrize('sg', GROUPS)
def test_round_trip_through_stored_parameters(sg):
    fc = FDCoordinates(sg, RADIUS)
    s = _s(fc, 256, sg)
    desc, _ = fc.to_description(s)
    p, h, _ = fc.to_params(s)
    L, A, t, rv, h64 = desc.crystal_params()
    back64 = fc.from_params(torch.cat([L, A, t, rv], 1), h64.reshape(-1))
    assert int(back64.ok.sum()) >= 250  # special-metric rows are measure zero; random s hits none in practice
    assert float(_circle_err(fc, back64.s[back64.ok], s[back64.ok]).max()) < 1e-8
    back32 = fc.from_params(p, h)  # float32 storage, as a prior file holds rows
    assert float(_circle_err(fc, back32.s[back32.ok], s[back32.ok]).max()) < 1e-3


@pytest.mark.parametrize('sg', GROUPS)
def test_built_rows_are_canonical(sg):
    fc = FDCoordinates(sg, RADIUS)
    desc, _ = fc.to_description(_s(fc, 128, 100 + sg))
    c = canon.canonicalize(desc)
    names = c.names()
    assert all(n in ('canonical', 'special_metric') for n in names)
    keep = torch.tensor([n == 'canonical' for n in names])
    d0, d1 = desc[torch.nonzero(keep).flatten()], c.desc[torch.nonzero(keep).flatten()]
    assert torch.allclose(d0.cell, d1.cell, atol=1e-9)
    assert torch.allclose(d0.R, d1.R, atol=1e-8)
    dt = d0.t - d1.t
    assert float((dt - torch.round(dt)).abs().max()) < 1e-9  # the same centre up to a lattice translation


@pytest.mark.parametrize('sg', [2, 14, 92])
def test_other_descriptions_chart_to_the_same_s(sg):
    from mxtaltools.crystal_fd import tables
    from mxtaltools.crystal_fd.ops import apply
    fc = FDCoordinates(sg, RADIUS)
    s = _s(fc, 64, 200 + sg)
    desc, _ = fc.to_description(s)
    tab = tables.group_table(sg)
    k = len(tab.gamma_W) - 1  # a nontrivial element of the proper normaliser part
    img = apply(desc, tab.gamma_W[k], tab.gamma_w[k])
    img = img.replace(t=img.t + 1.0)  # and a lattice translation
    L, A, t, rv, h = img.crystal_params()
    back = fc.from_params(torch.cat([L, A, t, rv], 1), h.reshape(-1))
    assert bool(back.ok.all())
    assert float(_circle_err(fc, back.s, s).max()) < 1e-7


@pytest.mark.parametrize('sg', [2, 7, 14])
def test_log_reference_is_density_plus_jacobian(sg):
    fc = FDCoordinates(sg, RADIUS)
    s = _s(fc, 6, 300 + sg)
    desc, log_ref = fc.to_description(s)
    for i in range(len(s)):
        def y_of(si):
            y, _ = charts.from_unit(fc.chart, ((si[None] + 1) / 2))
            return y[0]
        J = torch.autograd.functional.jacobian(y_of, s[i])
        logdet = torch.linalg.slogdet(J)[1]
        y, _ = charts.from_unit(fc.chart, ((s[i:i + 1] + 1) / 2))
        d_i, _ = fc.chart.forward(y)
        ref_y = measure.log_reference_density(d_i, fc.chart, 'latent_v1_lebesgue')
        assert abs(float(ref_y[0] + logdet - log_ref[i])) < 1e-9


@pytest.mark.parametrize('sg', [2, 14])
def test_circle_ends_are_one_crystal(sg):
    fc = FDCoordinates(sg, RADIUS)
    s = _s(fc, 32, 400 + sg)
    for j in [j for j, w in enumerate(fc.wrap_mask) if w]:
        a, b = s.clone(), s.clone()
        a[:, j], b[:, j] = -1.0, 1.0
        da, _ = fc.to_description(a)
        db, _ = fc.to_description(b)
        # the ends are glued by a normaliser element (e.g. a half-cell origin shift in P-1), not necessarily a lattice
        # translation: compare canonical forms
        ca, cb = canon.canonicalize(da).desc, canon.canonicalize(db).desc
        assert torch.allclose(ca.cell, cb.cell, atol=1e-10)
        assert torch.allclose(ca.R, cb.R, atol=1e-8)
        dt = ca.t - cb.t
        assert float((dt - torch.round(dt)).abs().max()) < 1e-8


@pytest.mark.parametrize('sg', [2, 14, 148])
def test_builder_builds_the_described_crystal(sg):
    from eval.fd_search import fd_starts
    from eval.fd_stress import molecule as fd_mol
    try:
        mol = fd_mol.load_mipcas()
    except Exception as e:  # the MIPCAS conditions file lives on the data drive
        pytest.skip(f'MIPCAS molecule not available: {e}')
    fc = FDCoordinates(sg, RADIUS)
    s = _s(fc, 8, 500 + sg)
    desc, _ = fc.to_description(s)
    p, h, _ = fc.to_params(s)
    assert fd_starts.builder_deviation(sg, fd_starts.rows_from_params(mol, sg, p, h), desc) < 1e-3
