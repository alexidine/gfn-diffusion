"""train.py::frame_readouts: the frame statistics density_readouts logs under model.nf rotate_frames / mirror_frames.

<set>_frame_mean is the random-frame -log q and <set>_frame_p90 the random-frame tail: the 90th percentile over rows of each row's mean -log q across the K frames.
These pin it and its two siblings on hand-made inputs, and the dropping of non-finite rows.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train import frame_readouts  # noqa: E402

pytestmark = pytest.mark.fast


def test_a_frame_independent_density_has_zero_spread_and_shift_and_the_standard_tail():
    f_std = torch.arange(10, dtype=torch.float32)
    f_all = f_std[:, None].expand(10, 4).clone()
    out = frame_readouts('eval_test/mle', f_std, f_all)
    assert out['eval_test/mle_frame_spread'] == pytest.approx(0.0)
    assert out['eval_test/mle_frame_shift'] == pytest.approx(0.0)
    assert out['eval_test/mle_frame_p90'] == pytest.approx(float(torch.quantile(f_std.double(), 0.9)))


def test_the_tail_reads_the_row_mean_over_frames_not_the_standard_frame():
    """Row i is i nats in its standard frame and i + 2 on average over its frames: the tail moves by the 2 nats."""
    f_std = torch.arange(100, dtype=torch.float32)
    f_all = torch.stack([f_std + 1.0, f_std + 3.0, f_std + 1.0, f_std + 3.0], 1)
    out = frame_readouts('k', f_std, f_all)
    assert out['k_frame_shift'] == pytest.approx(2.0)
    assert out['k_frame_mean'] == pytest.approx(float(f_std.mean()) + 2.0)
    assert out['k_frame_p90'] == pytest.approx(float(torch.quantile(f_std.double(), 0.9)) + 2.0, abs=1e-6)
    assert out['k_frame_spread'] == pytest.approx(float(torch.tensor([1.0, 3.0, 1.0, 3.0]).std()))


def test_one_bad_frame_drops_only_its_row():
    f_std = torch.tensor([0.0, 1.0, 2.0])
    f_all = torch.tensor([[0.0, 0.0], [float('inf'), 1.0], [4.0, 4.0]])
    out = frame_readouts('k', f_std, f_all)
    assert out['k_frame_shift'] == pytest.approx(1.0)   # rows 0 and 2: shifts 0 and 2
    assert out['k_frame_p90'] == pytest.approx(float(torch.quantile(torch.tensor([0.0, 4.0], dtype=torch.float64), 0.9)))


def test_no_finite_row_logs_nothing():
    assert frame_readouts('k', torch.tensor([float('nan')]), torch.tensor([[1.0, 2.0]])) == {}
