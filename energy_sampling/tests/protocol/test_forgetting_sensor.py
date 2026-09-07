"""The gated-ramp forgetting sensor, bwd/under_coverage_rise150 (train.py
Modeller._forgetting_sensor), measured in STEPS on whatever call cadence it gets.

Regression: the first build counted SAMPLES, and _update_rolling only runs every
10th bwd step, so the window was 3000 steps and never filled in rr07_rr_n7 -- the
controller held at 0.5 for the whole run with gr_sensor unwritten.
"""
import math
from types import MethodType, SimpleNamespace

import pytest

from energy_sampling.train import Modeller


def _bind(step=0):
    m = SimpleNamespace(step_ind=step, _UC_WINDOW_STEPS=Modeller._UC_WINDOW_STEPS)
    m._forgetting_sensor = MethodType(Modeller._forgetting_sensor, m)
    return m


def _feed(m, series, stride):
    """series[i] arrives at step (i+1)*stride; returns the stats dict of the last call."""
    out = None
    for i, v in enumerate(series):
        m.step_ind = (i + 1) * stride
        out = {'under_coverage': v}
        m._forgetting_sensor(out)
    return out


@pytest.mark.parametrize('stride', [1, 10])
def test_written_once_300_steps_exist_regardless_of_cadence(stride):
    m = _bind()
    n_short = 300 // stride - 1
    assert 'under_coverage_rise150' not in _feed(m, [1.0] * n_short, stride)
    m2 = _bind()
    assert _feed(m2, [1.0] * (300 // stride), stride)['under_coverage_rise150'] == pytest.approx(0.0)


def test_a_three_nat_step_reads_as_three_nats_on_the_ten_step_cadence():
    m = _bind()
    series = [40.0] * 15 + [43.0] * 15          # 150 steps flat, 150 steps up
    st = _feed(m, series, 10)
    assert st['under_coverage_rise150'] == pytest.approx(3.0)


def test_period_matched_oscillation_cancels():
    m = _bind()
    series = [40.0 + 0.5 * math.sin(2 * math.pi * i / 15) for i in range(60)]  # 150-step period
    st = _feed(m, series, 10)
    assert abs(st['under_coverage_rise150']) < 0.05


def test_non_finite_samples_are_dropped_not_propagated():
    m = _bind()
    series = [40.0] * 10 + [float('nan')] + [40.0] * 25
    st = _feed(m, series, 10)
    assert st['under_coverage_rise150'] == pytest.approx(0.0)


def test_history_is_bounded_to_the_window():
    m = _bind()
    _feed(m, [1.0] * 200, 10)
    assert len(m._uc_hist) <= 30
