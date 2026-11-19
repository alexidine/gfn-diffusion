"""eval.exp6_recall.panel.normalise_h puts each H at its element's normalised X-H length (2026-11-19: it had put
them at 2r - L)."""
import numpy as np
import pytest

pytest.importorskip('ase')
from ase import Atoms  # noqa: E402

from eval.exp6_recall.panel import XH, normalise_h  # noqa: E402


def test_hydrogens_land_at_the_normalised_length_across_the_cell_boundary():
    # C-H 0.93 (X-ray-like), N-H 0.86, O-H across the periodic boundary
    a = Atoms('CHNHOH', positions=[[0, 0, 0], [0.93, 0.1, 0], [5, 5, 5], [5, 5.86, 5], [1, 8, 9.6], [1, 8, 0.4]],
              cell=[10, 10, 10], pbc=True)
    out = normalise_h(a.copy())
    for heavy, h in ((0, 1), (2, 3), (4, 5)):
        d = out.get_distance(heavy, h, mic=True)
        assert d == pytest.approx(XH[int(out.numbers[heavy])], abs=1e-9)
        # the H stays on its original side of the heavy atom
        v0, v1 = a.get_distance(heavy, h, mic=True, vector=True), out.get_distance(heavy, h, mic=True, vector=True)
        assert np.dot(v0, v1) > 0
