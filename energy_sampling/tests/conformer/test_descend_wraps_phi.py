"""`energies/prior_baselines.py::descend` wraps a periodic state column and clamps the rest.

A phi column of the conformer state is an angle (``dof_from_state``: ph0 + pi * x), so [-1, 1)
is one full turn. Before 2026-09-29 `descend` clamped it, and a torsion that descended across
ph0 +/- pi stopped on the seam: a false minimum that the floor search, the run's prior relax,
the anchor-seed relax and the prior-dataset builder all inherited.
"""
import math

import pytest
import torch

from energies.prior_baselines import descend


class _SeamAndWall:
    """Two state columns. Column 0 is periodic, with its only minimum at x = -0.9: from a start
    at x = 0.9 the downhill way is UP, through x = 1 (the seam), to -0.9. Column 1 is a box
    column whose unconstrained minimum, x = 1.5, lies outside the box."""
    dtype = torch.float64
    temperature = 1.0
    periodic_dims = [True, False]

    def potential_energy(self, x, temperature, keep_grads=True):
        return (1.0 - torch.cos(math.pi * (x[:, 0] + 0.9))) + (x[:, 1] - 1.5) ** 2


@pytest.mark.fast
def test_a_periodic_column_descends_through_the_seam_and_a_box_column_stays_clamped():
    x0 = torch.tensor([[0.9, 0.0]], dtype=torch.float64)
    best_x, best_u = descend(_SeamAndWall(), x0, steps=300)
    # through the seam to the true minimum; a clamp would stop at x = 1, energy 0.049 above it
    assert best_x[0, 0].item() == pytest.approx(-0.9, abs=1e-3)
    # the box column ends on its wall, not at its unconstrained minimum
    assert best_x[0, 1].item() == pytest.approx(1.0, abs=1e-12)
    assert best_u[0].item() == pytest.approx(0.25, abs=1e-5)


@pytest.mark.fast
def test_an_energy_without_periodic_dims_is_refused():
    class _Undeclared:
        dtype = torch.float64
        temperature = 1.0

        def potential_energy(self, x, temperature, keep_grads=True):
            return (x ** 2).sum(-1)

    with pytest.raises(AttributeError):
        descend(_Undeclared(), torch.zeros(1, 2, dtype=torch.float64), steps=1)
