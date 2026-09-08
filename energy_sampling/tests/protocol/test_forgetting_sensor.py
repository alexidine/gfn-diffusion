"""The gated-ramp forgetting sensor (train.py Modeller._forgetting_sensor),
measured in STEPS on whatever call cadence it gets.

BOTH channels are computed -- bwd/under_coverage_rise150 (Z-anchored) and
bwd/relative_under_rise150 (level-blind) -- and the stage's balance.metric picks
which one steers. Computing only one made the choice a code edit; it is a design
decision, and having both in every run is also how they get compared.

Regression: the first build counted SAMPLES, and _update_rolling only runs every
10th bwd step, so the window was 3000 steps and never filled in rr07_rr_n7 -- the
controller held at 0.5 for the whole run with gr_sensor unwritten.
"""
import math
from types import MethodType, SimpleNamespace

import pytest

from energy_sampling.train import Modeller


CHANNELS = Modeller._FORGETTING_CHANNELS


def _bind(step=0):
    m = SimpleNamespace(step_ind=step, _UC_WINDOW_STEPS=Modeller._UC_WINDOW_STEPS,
                        _FORGETTING_CHANNELS=CHANNELS)
    m._forgetting_sensor = MethodType(Modeller._forgetting_sensor, m)
    return m


def _feed(m, series, stride, channel='relative_under'):
    """series[i] arrives at step (i+1)*stride; returns the stats dict of the last call."""
    out = None
    for i, v in enumerate(series):
        m.step_ind = (i + 1) * stride
        out = {channel: v}
        m._forgetting_sensor(out)
    return out


@pytest.mark.parametrize('channel', CHANNELS)
@pytest.mark.parametrize('stride', [1, 10])
def test_written_once_300_steps_exist_regardless_of_cadence(stride, channel):
    key = f'{channel}_rise150'
    m = _bind()
    n_short = 300 // stride - 1
    assert key not in _feed(m, [1.0] * n_short, stride, channel)
    m2 = _bind()
    assert _feed(m2, [1.0] * (300 // stride), stride, channel)[key] == pytest.approx(0.0)


@pytest.mark.parametrize('channel', CHANNELS)
def test_a_three_nat_step_reads_as_three_nats_on_the_ten_step_cadence(channel):
    m = _bind()
    series = [40.0] * 15 + [43.0] * 15          # 150 steps flat, 150 steps up
    st = _feed(m, series, 10, channel)
    assert st[f'{channel}_rise150'] == pytest.approx(3.0)


@pytest.mark.parametrize('channel', CHANNELS)
def test_period_matched_oscillation_cancels(channel):
    m = _bind()
    series = [40.0 + 0.5 * math.sin(2 * math.pi * i / 15) for i in range(60)]  # 150-step period
    st = _feed(m, series, 10, channel)
    assert abs(st[f'{channel}_rise150']) < 0.05


@pytest.mark.parametrize('channel', CHANNELS)
def test_non_finite_samples_are_dropped_not_propagated(channel):
    m = _bind()
    series = [40.0] * 10 + [float('nan')] + [40.0] * 25
    st = _feed(m, series, 10, channel)
    assert st[f'{channel}_rise150'] == pytest.approx(0.0)


@pytest.mark.parametrize('channel', CHANNELS)
def test_history_is_bounded_to_the_window(channel):
    m = _bind()
    _feed(m, [1.0] * 200, 10, channel)
    assert len(m._uc_hist[channel]) <= 30


def test_both_channels_are_written_from_one_stats_dict():
    """The stage's balance.metric selects; the run logs both, so switching the
    controller's channel is a config edit and the two are always comparable on
    the same run."""
    m = _bind()
    for i in range(30):
        m.step_ind = (i + 1) * 10
        st = {'under_coverage': 40.0 + 0.02 * i, 'relative_under': 3.0}
        m._forgetting_sensor(st)
    assert st['under_coverage_rise150'] == pytest.approx(0.3)   # 15 samples x 0.02
    assert st['relative_under_rise150'] == pytest.approx(0.0)


def test_a_channel_absent_from_stats_does_not_block_the_other():
    m = _bind()
    for i in range(30):
        m.step_ind = (i + 1) * 10
        st = {'under_coverage': 40.0}
        m._forgetting_sensor(st)
    assert st['under_coverage_rise150'] == pytest.approx(0.0)
    assert 'relative_under_rise150' not in st
