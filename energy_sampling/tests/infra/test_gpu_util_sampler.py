"""
The occupancy sampler now runs on its own thread, which buys two hazards worth
holding down: the deque is written from one thread and read from others, and the
thread is the ONLY thing populating the series -- the train loop no longer calls
the sampler at all, so a thread that fails to start is a silent loss of the one
number the scheduler cancels jobs on.

The phase argument itself (why off-thread at all) is in
`train.Modeller._start_gpu_util_thread`; it is a claim about bias that only a
side-by-side against wandb's `system.gpu.0.gpu` can settle, and these tests do
not pretend to settle it.
"""

import threading
import time
import types
from collections import deque

import pytest

pytestmark = pytest.mark.slow  # imports train.py (~11 s, pulls torch/PyG)


def _modeller(period_s=0.01, reading=lambda: 42.0):
    """A bare object carrying the sampler's real methods and nothing else."""
    import train

    m = types.SimpleNamespace()
    m.args = types.SimpleNamespace(gpu_util_sample_period_s=period_s,
                                   gpu_util_window_s=900,
                                   gpu_util_policy_window_s=7200)
    m._now = types.MethodType(train.Modeller._now, m)
    m._sample_gpu_util = types.MethodType(train.Modeller._sample_gpu_util, m)
    m._gpu_util_samples = types.MethodType(train.Modeller._gpu_util_samples, m)
    m._gpu_util_mean = types.MethodType(train.Modeller._gpu_util_mean, m)
    m._gpu_util_capacity = types.MethodType(train.Modeller._gpu_util_capacity, m)
    m._start_gpu_util_thread = types.MethodType(train.Modeller._start_gpu_util_thread, m)
    # the thread's first act is to name its sensor in the joblog; bound real so a
    # change to that announcement cannot break the shipping startup path unnoticed
    m._announce_gpu_util_source = types.MethodType(
        train.Modeller._announce_gpu_util_source, m)
    m._read_gpu_util = reading
    return m


def _wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_the_thread_populates_the_series_with_nobody_calling_the_sampler():
    """No train loop in this test -- if the thread is not doing it, nothing is."""
    m = _modeller()
    m._start_gpu_util_thread()
    assert _wait_for(lambda: len(m._gpu_util_samples()) >= 5), \
        'the sampler thread produced fewer than 5 readings in 5 s'
    # the readings are there; the MEAN still refuses until they span real time.
    # That is _UTIL_MIN_SPAN_S, asserted below rather than waited out.
    assert m._gpu_util_mean(3600) is None
    import train
    m._gpu_util.clear()
    now = m._now()
    for k in range(6):
        m._gpu_util.append((now - train._UTIL_MIN_SPAN_S - 10 + k * 20, 42.0))
    assert m._gpu_util_mean(3600) == pytest.approx(42.0)


def test_a_count_of_readings_is_not_enough_without_a_span():
    """
    THE REGRESSION THIS EXISTS FOR. Every occupancy guard used to be a sample
    COUNT, which silently encodes the sampling period: five readings span 300 s
    at a 60 s period and 10 s at 2 s. Lowering the period to buy resolution would
    otherwise have LOOSENED the gates that exist to refuse a coin flip -- so the
    count and the span are checked together, and a burst of readings inside a few
    seconds must not produce a mean.
    """
    m = _modeller(period_s=0)          # no thread; the deque is driven by hand
    m._gpu_util = deque(maxlen=4096)
    now = m._now()
    for k in range(50):                # fifty readings, all within 5 s
        m._gpu_util.append((now - 5 + k * 0.1, 42.0))
    assert m._gpu_util_mean(900) is None,         'fifty readings spanning 5 s were accepted as a 900 s mean'


def test_the_deque_holds_the_widest_window_at_the_configured_period():
    """
    A FIXED CAPACITY SILENTLY ENCODES A PERIOD TOO. 4096 readings is 68 hours at
    60 s and 68 MINUTES at 1 s -- shorter than the 7200 s policy window, at which
    point the windowed mean averages whatever survived eviction and reports it as
    a two-hour number. Capacity has to follow the period.
    """
    for period, window in ((60, 7200), (2, 7200), (1, 7200), (0.5, 900)):
        m = _modeller(period_s=period)
        m.args.gpu_util_window_s = 900
        m.args.gpu_util_policy_window_s = window
        cap = m._gpu_util_capacity()
        assert cap * period >= window, (
            f'at period {period}s the deque holds {cap * period:.0f}s, '
            f'short of the {window}s window it will be asked for')


def test_a_reader_survives_a_full_deque_being_appended_under_it():
    """
    THE CRASH THIS GUARDS. `_gpu_util` is bounded (maxlen), so once it is full an
    append also POPS -- and iterating it while that happens raises 'deque mutated
    during iteration'. The victim would be select_batch_size's per-rung sample
    filter, mid-calibration, intermittently. Every reader goes through
    `_gpu_util_samples` for exactly this reason.
    """
    from collections import deque

    m = _modeller(period_s=0.001)
    m._gpu_util = deque(maxlen=16)      # small, so it is full almost immediately
    m._start_gpu_util_thread()
    assert _wait_for(lambda: len(m._gpu_util_samples()) == 16), 'deque never filled'

    errors = []

    def read():
        end = time.time() + 1.0
        while time.time() < end:
            try:
                [u for ts, u in m._gpu_util_samples()]
            except Exception as e:
                errors.append(e)
                return

    readers = [threading.Thread(target=read) for _ in range(4)]
    for t in readers:
        t.start()
    for t in readers:
        t.join()
    assert not errors, f'reader raced the sampler: {errors[0]!r}'


def test_a_dead_sensor_goes_inert_instead_of_killing_the_thread_silently():
    """
    A sensor that raises must set `_gpu_util_off` -- the same state a missing
    sensor produces -- because that is what select_batch_size's S3 branch reads
    to hold the base batch. A thread that just died would leave the sizer
    believing the sensor was live and merely quiet.
    """
    def boom():
        raise RuntimeError('no such device')

    m = _modeller(reading=boom)
    m._start_gpu_util_thread()
    assert _wait_for(lambda: getattr(m, '_gpu_util_off', False)), \
        'the sampler thread did not mark the sensor off after it raised'
    assert m._gpu_util_mean(3600) is None


def test_a_zero_period_starts_no_thread():
    """`gpu_util_sample_period_s: 0` is 'do not sample'. It must not spin."""
    m = _modeller(period_s=0)
    m._start_gpu_util_thread()
    t = getattr(m, '_gpu_util_thread', None)
    assert t is None or not t.is_alive()
    time.sleep(0.1)
    assert m._gpu_util_samples() == ()
