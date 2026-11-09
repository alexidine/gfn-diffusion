"""noised_prior.metric_kick: the iso kick measured in the FD identity metric (Angstrom) instead of the trainer latent.

Real Z' = 1 CSD structures (mxtaltools mini_new_csd.pt), CPU. The rows are first passed once through the latent round
trip the kick ends with, so that a row starts as a crystal the trainer latent holds; the checks then read the step
back in metric coordinates (noised_prior.metric_kick's docstring) from the parameters before and after.
"""
import os
import sys

import pytest
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _here not in sys.path:
    sys.path.insert(0, _here)

import noised_prior as NP  # noqa: E402

pytestmark = pytest.mark.fast

DATA = os.path.abspath(os.path.join(_here, '..', '..', 'mxtaltools', 'mini_datasets', 'mini_new_csd.pt'))


def _rows(sgs, per=4):
    from mxtaltools.dataset_utils.utils import collate_data_list
    if not os.path.exists(DATA):
        raise AssertionError(f"mini_new_csd.pt not found at {DATA}: the dataset is part of the MXtalTools checkout")
    data = torch.load(DATA, weights_only=False)
    picked = []
    for sg in sgs:
        rows = [e for e in data if int(e.sg_ind) == sg and int(e.z_prime) == 1 and bool(e.is_well_defined)][:per]
        assert rows, f"no Z' = 1 row of space group {sg} in the mini dataset"
        picked += rows
    b = collate_data_list(picked)
    b.latent_to_cell_params(b.latent_params(gauge_fix_free_axes=False).clip(-1 + 1e-6, 1 - 1e-6))
    b.clean_cell_parameters(mode='hard')
    return b


def _metric_delta(p0, p1, T0, rg):
    """[n, 12] metric coordinates of the displacement p0 -> p1 (lengths, scaled angles, Cartesian centroid shift,
    r_g times the rotation vector of R1 R0^T)."""
    from mxtaltools.common.geometry_utils import rotmat2rotvec_stable
    p0, p1 = p0.double(), p1.double()
    a, b, c = p0[:, 0], p0[:, 1], p0[:, 2]
    d = torch.zeros(len(p0), 12, dtype=torch.float64)
    d[:, :3] = p1[:, :3] - p0[:, :3]
    d[:, 3:6] = (p1[:, 3:6] - p0[:, 3:6]) * torch.stack([torch.sqrt(b * c), torch.sqrt(a * c), torch.sqrt(a * b)], 1)
    d[:, 6:9] = torch.einsum('nij,nj->ni', T0.double(), p1[:, 6:9] - p0[:, 6:9])
    R0, R1 = NP._rodrigues(p0[:, 9:12]), NP._rodrigues(p1[:, 9:12])
    d[:, 9:12] = rotmat2rotvec_stable(R1 @ R0.transpose(1, 2)) * rg[:, None]
    return d


def test_zero_step_is_the_identity():
    b = _rows((2, 14, 19, 145))
    p0 = b.full_cell_parameters().clone()
    T0, rg = b.T_fc.clone(), NP.radius_of_gyration(b)
    NP.metric_kick(b, -12.0, -12.0)
    d = _metric_delta(p0, b.full_cell_parameters(), T0, rg)
    assert float(d.abs().max()) < 1e-3, float(d.abs().max())


def test_step_length_is_rho_in_the_metric():
    b = _rows((1, 2, 4, 7, 9, 14, 15, 19, 33, 60, 61, 114, 145, 148), per=8)
    p0 = b.full_cell_parameters().clone()
    T0, rg = b.T_fc.clone(), NP.radius_of_gyration(b)
    torch.manual_seed(3)
    rho = NP.metric_kick(b, -2.0, -2.0)
    assert torch.allclose(rho, torch.full_like(rho, 0.01))
    got = _metric_delta(p0, b.full_cell_parameters(), T0, rg).norm(dim=1)
    rel = (got / 0.01 - 1).abs()
    # a row clipped at a bound by the latent round trip may differ (one of 55 at 0.1 A, none at 0.01 A, 2026-11-04)
    assert float((rel < 0.02).double().mean()) >= 0.95, rel.tolist()


def test_fixed_and_tied_cell_parameters_hold():
    b = _rows((14, 19, 145, 148))
    codes = b.sg_ind.reshape(-1)
    torch.manual_seed(5)
    NP.metric_kick(b, -1.0, -1.0)
    L, A = b.cell_lengths.double(), b.cell_angles.double()
    half = torch.pi / 2
    mono = codes == 14
    assert torch.allclose(A[mono][:, [0, 2]], torch.full_like(A[mono][:, [0, 2]], half), atol=1e-6)
    ortho = codes == 19
    assert torch.allclose(A[ortho], torch.full_like(A[ortho], half), atol=1e-6)
    hexa = (codes == 145) | (codes == 148)
    assert torch.allclose(L[hexa][:, 0], L[hexa][:, 1], rtol=1e-6)
    assert torch.allclose(A[hexa][:, 2], torch.full_like(A[hexa][:, 2], 2 * torch.pi / 3), atol=1e-6)


def test_direction_is_isotropic_on_the_live_coordinates():
    """One triclinic crystal kicked 3000 times at a fixed small step: the metric displacement's second moment is
    rho^2 / 12 on every one of the 12 coordinates and ~0 off the diagonal."""
    from mxtaltools.dataset_utils.utils import collate_data_list
    one = _rows((2,), per=1)
    n = 3000
    data = torch.load(DATA, weights_only=False)
    src = [e for e in data if int(e.sg_ind) == 2 and int(e.z_prime) == 1 and bool(e.is_well_defined)][0]
    b = collate_data_list([src.clone() for _ in range(n)])
    b.set_cell_parameters(one.full_cell_parameters().expand(n, 12).clone())
    p0 = b.full_cell_parameters().clone()
    T0, rg = b.T_fc.clone(), NP.radius_of_gyration(b)
    torch.manual_seed(7)
    rho = 1e-3
    NP.metric_kick(b, -3.0, -3.0)
    d = _metric_delta(p0, b.full_cell_parameters(), T0, rg) / rho
    m2 = d.T @ d / n
    diag = torch.diagonal(m2) * 12
    assert float((diag - 1).abs().max()) < 0.2, diag.tolist()
    off = m2 - torch.diag(torch.diagonal(m2))
    assert float(off.abs().max()) * 12 < 0.15, float(off.abs().max())


def test_refuses_zprime_above_one():
    from mxtaltools.dataset_utils.utils import collate_data_list
    data = torch.load(DATA, weights_only=False)
    src = next(e for e in data if int(e.z_prime) == 1 and bool(e.is_well_defined))
    b = collate_data_list([src.clone()], max_z_prime=2)
    with pytest.raises(ValueError, match="Z' = 1"):
        NP.metric_kick(b, -2.0, -2.0)


def test_the_builder_reads_metric_radii_in_angstrom():
    from data_processing.build_noised_prior import KICK_UNITS, build_parser, check_child_kernel_args
    base = ['--config', 'c.yaml', '--out', 'o.pt', '--source', 's']
    a = build_parser().parse_args(base + ['--child_kernel', 'metric'])
    check_child_kernel_args(a)
    assert a.log_lo == -2.0 and KICK_UNITS['metric'] == 'angstrom'
    with pytest.raises(SystemExit, match='basin only'):
        check_child_kernel_args(build_parser().parse_args(base + ['--child_kernel', 'metric', '--tau', '1']))


def _seam(tile, metric_range=(-2.0, -2.0)):
    """train.py::Modeller._noise_and_condition and _metric_noise_log_range on a stand-in Modeller: the anchor buffer
    config and an energy function whose conditioning passes the batch through."""
    import types
    from energy_sampling.train import Modeller

    class EF:
        @staticmethod
        def condition_samples(batch, sg_inds=None, z_primes=None):
            return batch, None, None, None

    m = types.SimpleNamespace(energy_function=EF())
    m.args = types.SimpleNamespace(buffers=types.SimpleNamespace(anchor_buffer=types.SimpleNamespace(
        tile=tile, metric_noise_log_range=None if metric_range is None else list(metric_range))))
    m._metric_noise_log_range = types.MethodType(Modeller._metric_noise_log_range, m)
    return m, types.MethodType(Modeller._noise_and_condition, m)


def test_the_trainer_seam_runs_the_metric_tile_and_refuses_the_rest():
    b = _rows((2, 14))
    p0 = b.full_cell_parameters().clone()
    T0, rg = b.T_fc.clone(), NP.radius_of_gyration(b)
    _, seam = _seam('metric')
    torch.manual_seed(1)
    out = seam(b.clone(), (-12.0, -12.0))[0]          # the latent range is unread under 'metric'
    got = _metric_delta(p0, out.full_cell_parameters(), T0, rg).norm(dim=1)
    assert float(((got / 0.01 - 1).abs() < 0.02).double().mean()) >= 0.95, got.tolist()
    for bad in (None, (-1.0, -2.0), (True, 0.0)):
        _, seam = _seam('metric', bad)
        with pytest.raises(ValueError, match='metric_noise_log_range'):
            seam(b.clone(), (-2.5, -1.5))
    _, seam = _seam('isotropic')
    with pytest.raises(ValueError, match="'iso', 'metric' or 'shaped'"):
        seam(b.clone(), (-2.5, -1.5))


def test_the_fd_chart_writes_the_step_without_folding_into_the_latent_box():
    """chart 'fd': the stepped parameters are written as they are. A P2_1/c molecule placed past the latent's
    asymmetric-unit box (y > 1/4) keeps its centroid through a zero step on 'fd'; on 'latent' the round trip and the
    hard clean move it back inside the box, which is another crystal."""
    b = _rows((14,), per=3)
    b.aunit_centroid[:, 1] = 0.4
    p0 = b.full_cell_parameters().clone()
    torch.manual_seed(3)
    NP.metric_kick(b, -12.0, -12.0, chart='fd')
    assert torch.allclose(b.full_cell_parameters()[:, :9], p0[:, :9], atol=1e-6)
    c = _rows((14,), per=3)
    c.aunit_centroid[:, 1] = 0.4
    NP.metric_kick(c, -12.0, -12.0, chart='latent')
    assert bool((c.aunit_centroid[:, 1] <= 0.25 + 1e-4).all())
    with pytest.raises(ValueError, match='chart must be one of'):
        NP.metric_kick(_rows((14,), per=1), -2.0, -2.0, chart='asu')
