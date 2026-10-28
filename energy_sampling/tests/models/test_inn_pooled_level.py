"""inn_modeller.py --level_mode ema (pooled TB): InnModeller.ema_levels reads each row's level off the trainer's
tracker -- the forward stream (fwd_level_ema), or --level_mix x forward + (1 - --level_mix) x backward
(bwd_level_ema) -- and gives a row whose condition is not yet trusted on a stream (fewer than min_visits updates)
that stream's fallback."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import inn_modeller as im  # noqa: E402
from buffer import ConditionLogZTracker  # noqa: E402

pytestmark = pytest.mark.fast


def _host(source, mix=0.5, min_visits=3):
    tr = ConditionLogZTracker(library_size=4, min_visits=min_visits, half_life_visits=50)
    for step in range(min_visits):
        tr.update_mode_level('fwd', torch.tensor([0, 1]), torch.tensor([-40.0, -30.0]), step=step)
        tr.update_mode_level('bwd', torch.tensor([0, 2]), torch.tensor([-10.0, -5.0]), step=step)
    tr.update_mode_level('fwd', torch.tensor([3]), torch.tensor([-20.0]), step=99)   # condition 3: one visit only
    host = object.__new__(im.InnModeller)
    host.condition_log_z = tr
    host.inn = SimpleNamespace(level_source=source, level_mix=mix)
    return host


def test_forward_source_reads_the_forward_stream_and_falls_back_where_untrusted():
    host = _host('fwd')
    cid = torch.tensor([0, 1, 2, 3])
    fb = torch.tensor([7.0, 7.0, 7.0, 7.0])
    level, ok_f, ok_b = host.ema_levels(cid, fb, fb - 100)
    assert ok_b is None
    assert ok_f.tolist() == [True, True, False, False]          # 2: never fed forward; 3: below min_visits
    assert torch.allclose(level, torch.tensor([-40.0, -30.0, 7.0, 7.0]))


def test_pooled_source_mixes_the_two_streams_each_with_its_own_fallback():
    host = _host('pooled', mix=0.5)
    cid = torch.tensor([0, 1, 2])
    fb_f, fb_b = torch.tensor([1.0, 1.0, 1.0]), torch.tensor([3.0, 3.0, 3.0])
    level, ok_f, ok_b = host.ema_levels(cid, fb_f, fb_b)
    assert ok_f.tolist() == [True, True, False] and ok_b.tolist() == [True, False, True]
    expect = torch.tensor([0.5 * -40.0 + 0.5 * -10.0,    # both streams trusted
                           0.5 * -30.0 + 0.5 * 3.0,      # backward untrusted: its fallback
                           0.5 * 1.0 + 0.5 * -5.0])      # forward untrusted: its fallback
    assert torch.allclose(level, expect)


def test_mix_weights_the_forward_stream():
    host = _host('pooled', mix=0.75)
    level, _, _ = host.ema_levels(torch.tensor([0]), torch.zeros(1), torch.zeros(1))
    assert torch.allclose(level, torch.tensor([0.75 * -40.0 + 0.25 * -10.0]))
