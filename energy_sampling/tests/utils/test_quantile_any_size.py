"""utils.quantile_any_size: torch.quantile's linear interpolation at any length (torch.quantile raises past 2^24
elements; a four-group pre-noised prior trains on ~3e7 rows, pool8sg4_nov20 smokes 2026-11-20)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TORCH_QUANTILE_MAX, quantile_any_size  # noqa: E402

pytestmark = pytest.mark.fast


@pytest.mark.parametrize('q', [0.0, 0.001, 0.5, 0.9, 0.999, 1.0])
def test_matches_torch_quantile_where_torch_works(q):
    x = torch.randn(10007, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(quantile_any_size(x, q), torch.quantile(x, q))


def test_sort_branch_equals_torch_on_a_tensor_torch_can_still_take(monkeypatch):
    import utils
    x = torch.randn(5001, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    monkeypatch.setattr(utils, 'TORCH_QUANTILE_MAX', 10)   # force the sort branch
    for q in (0.0, 0.25, 0.999, 1.0):
        assert torch.allclose(utils.quantile_any_size(x, q), torch.quantile(x, q))


def test_past_the_torch_limit():
    n = TORCH_QUANTILE_MAX + 1001
    x = torch.arange(n, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.quantile(x, 0.5)
    assert float(quantile_any_size(x, 0.5)) == pytest.approx((n - 1) / 2)
