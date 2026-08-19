"""
Region profiler: free when off, correct when on, and never a sync.

THE FIRST TEST IS THE ONE THAT MATTERS. This code sits in the hottest path in
the repository, so "disabled costs nothing" is not a nicety -- it is the
condition for the feature existing at all. It is checked by identity on the
returned context (one shared no-op object, so nothing is allocated per region
per step), not by timing, which would be flaky.

The rest pin the properties a GPU timer has to have and a wall clock does not:
totals accumulate across steps, a report resets, counts travel with totals, and
a pair still in flight is LEFT rather than waited on.
"""

import pytest

from profiling import _NULL, RegionProfiler

pytestmark = pytest.mark.fast


class _FakeEvent:
    """A torch.cuda.Event stand-in with controllable completion, so the drain
    logic is testable without a GPU."""

    def __init__(self, done=True, ms=1.0):
        self.done, self.ms = done, ms

    def record(self):
        pass

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.ms


# ------------------------------------------------------------------ free when off

def test_disabled_returns_the_shared_noop_context():
    """THE LOAD-BEARING ONE. Not merely 'cheap' -- the SAME object every time,
    so a disabled run allocates nothing per region per step."""
    prof = RegionProfiler(enabled=False, cuda=True)
    assert prof.region('energy') is _NULL
    assert prof.region('energy') is prof.region('rollout') is _NULL


def test_disabled_records_nothing():
    prof = RegionProfiler(enabled=False, cuda=False)
    with prof.region('energy'):
        pass
    assert prof.report() == {} and prof.pending() == 0


def test_an_unselected_region_is_also_free():
    """`regions` restricts what is paid for. A run may want the energy call
    timed without paying for five others."""
    prof = RegionProfiler(enabled=True, cuda=True, regions=('energy',))
    assert prof.region('rollout') is _NULL
    assert prof.region('energy') is not _NULL


def test_enabled_does_NOT_return_the_noop():
    """MUTATION IN THE PASSING DIRECTION. Without this, `region()` returning
    _NULL unconditionally would satisfy every no-op test above."""
    prof = RegionProfiler(enabled=True, cuda=False)
    assert prof.region('energy') is not _NULL


# ------------------------------------------------------------------ accumulation

def test_wall_path_accumulates_and_counts():
    prof = RegionProfiler(enabled=True, cuda=False)
    for _ in range(3):
        with prof.region('energy'):
            pass
    rep = prof.report()
    assert rep['perf/energy_n'] == 3.0
    assert rep['perf/energy_ms'] >= 0.0
    assert 'perf/energy_ms_mean' in rep


def test_report_resets_so_periods_do_not_accumulate():
    """A total that never resets turns a per-period metric into a run-to-date
    one, and every trend read off it is then wrong in the same direction."""
    prof = RegionProfiler(enabled=True, cuda=False)
    with prof.region('energy'):
        pass
    assert prof.report()['perf/energy_n'] == 1.0
    with prof.region('energy'):
        pass
    assert prof.report()['perf/energy_n'] == 1.0, 'second period saw the first'


def test_counts_travel_with_totals():
    """A region timed 3x and one timed 300x can share a total. Without the
    count a reader cannot tell a slow region from a frequent one."""
    prof = RegionProfiler(enabled=True, cuda=False)
    for _ in range(5):
        with prof.region('rollout'):
            pass
    rep = prof.report()
    assert rep['perf/rollout_n'] == 5.0
    assert rep['perf/rollout_ms_mean'] == pytest.approx(
        rep['perf/rollout_ms'] / 5.0)


# ---------------------------------------------------------------------- drain

def test_drain_reads_only_completed_pairs():
    prof = RegionProfiler(enabled=True, cuda=True)
    prof._submit('a', _FakeEvent(), _FakeEvent(done=True, ms=4.0))
    prof._submit('b', _FakeEvent(), _FakeEvent(done=False, ms=9.0))
    assert prof.drain() == 1
    assert prof.pending() == 1, 'the unfinished pair must be kept, not dropped'
    rep = prof.report()
    assert rep['perf/a_ms'] == pytest.approx(4.0)
    assert 'perf/b_ms' not in rep, 'reported a pair that had not finished'


def test_a_pair_finishing_later_is_picked_up_on_a_later_drain():
    """THE POINT OF POLLING. The alternative -- synchronising -- would serialise
    the overlap the profiler exists to measure."""
    prof = RegionProfiler(enabled=True, cuda=True)
    end = _FakeEvent(done=False, ms=7.0)
    prof._submit('slow', _FakeEvent(), end)
    assert prof.drain() == 0
    end.done = True
    assert prof.drain() == 1
    assert prof.report()['perf/slow_ms'] == pytest.approx(7.0)


def test_a_raising_body_still_records_the_region_WALL():
    """The launches happened, so the time is real. Dropping it would
    under-report exactly the steps that went wrong."""
    prof = RegionProfiler(enabled=True, cuda=False)
    with pytest.raises(ValueError):
        with prof.region('energy'):
            raise ValueError('boom')
    assert prof.report()['perf/energy_n'] == 1.0


def test_a_raising_body_still_records_the_region_CUDA(monkeypatch):
    """THE SAME PROPERTY ON THE OTHER PATH, and it needs its own test.

    The wall test above passes with the CUDA path dropping raising bodies
    entirely -- found by mutating `_CudaRegion.__exit__` and watching the suite
    stay green. Two implementations means two tests; a shared docstring is not
    shared coverage."""
    import torch
    monkeypatch.setattr(torch.cuda, 'Event', lambda **kw: _FakeEvent(ms=3.0))
    prof = RegionProfiler(enabled=True, cuda=True)
    with pytest.raises(ValueError):
        with prof.region('energy'):
            raise ValueError('boom')
    assert prof.pending() == 1, 'the raising region was never submitted'
    assert prof.report()['perf/energy_ms'] == pytest.approx(3.0)


def test_pending_pairs_are_bounded():
    """A profiler that holds every unfinished pair forever is a leak on a long
    run. The bound is reported, not silent."""
    prof = RegionProfiler(enabled=True, cuda=True)
    for _ in range(600):
        prof._submit('x', _FakeEvent(), _FakeEvent(done=False))
    assert prof.pending() <= 512
    assert prof.overflowed > 0
    assert prof.report()['perf/lost_pairs'] > 0


# --------------------------------------------------------------- from_config

class _Bag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_absent_config_block_means_disabled():
    """The one default where absence is unambiguous: a profiler nobody asked
    for should not be running."""
    assert not profiling_from(_Bag()).enabled


def test_enabled_block_turns_it_on():
    prof = profiling_from(_Bag(profiling=_Bag(enabled=True, regions=('energy',))))
    assert prof.enabled and prof.regions == ('energy',)


def profiling_from(args):
    from profiling import from_config
    return from_config(args, cuda=False)


# =========================================================================
# Layer 2 -- the bounded trace window
# =========================================================================

class _FakeProf:
    """Stands in for torch.profiler.profile so the state machine is testable
    without actually profiling anything."""

    def __init__(self, fail_write=False):
        self.entered = self.exited = False
        self.fail_write = fail_write

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def key_averages(self):
        if self.fail_write:
            raise RuntimeError('trace export blew up')
        return self

    def table(self, **kw):
        return 'TABLE'

    def export_chrome_trace(self, path):
        open(path, 'w').close()


def _window(tmp_path, **kw):
    from profiling import TraceWindow
    kw.setdefault('enabled', True)
    kw.setdefault('start_step', 5)
    kw.setdefault('active_steps', 3)
    fail = kw.pop('fail_write', False)
    w = TraceWindow(outdir=str(tmp_path), tag='t', **kw)
    w._fake = _FakeProf(fail_write=fail)
    w._make_profiler = lambda: w._fake
    return w


def test_disabled_window_never_opens(tmp_path):
    """The hot-loop call must be a boolean check and nothing else."""
    w = _window(tmp_path, enabled=False)
    for s in range(50):
        w.step(s)
    assert w._prof is None and not w._fake.entered and w.written == []


def test_it_does_not_open_before_start_step(tmp_path):
    """Opening early profiles CUDA context creation, autotuning and cold
    caches -- i.e. startup, not training."""
    w = _window(tmp_path, start_step=5)
    for s in range(5):
        w.step(s)
    assert not w._fake.entered


def test_it_opens_at_start_and_closes_after_active_steps(tmp_path):
    w = _window(tmp_path, start_step=5, active_steps=3)
    for s in range(20):
        w.step(s)
    assert w._fake.entered and w._fake.exited
    # THE TABLE ONLY, by default. The chrome trace is opt-in since it measured
    # ~93 MB/step and is what bounds active_steps; see write_trace.
    assert w.done and len(w.written) == 1
    assert w.written[0].endswith('_table.txt')


def test_the_chrome_trace_is_written_only_when_asked(tmp_path):
    """Both directions of the switch, because a default that silently stopped
    writing the timeline would look identical to a window that never fired."""
    off = _window(tmp_path / 'off', start_step=0, active_steps=1)
    for s in range(5):
        off.step(s)
    assert [f.rsplit('_', 1)[-1] for f in off.written] == ['table.txt']

    on = _window(tmp_path / 'on', start_step=0, active_steps=1, write_trace=True)
    for s in range(5):
        on.step(s)
    assert [f.rsplit('_', 1)[-1] for f in on.written] == ['table.txt', 'trace.json']


def test_once_done_it_stays_done(tmp_path):
    """The window fires ONCE. A second opening would double the cost and
    silently overwrite the first trace."""
    w = _window(tmp_path, start_step=2, active_steps=2)
    for s in range(30):
        w.step(s)
    first = list(w.written)
    w._fake = _FakeProf()
    for s in range(30, 60):
        w.step(s)
    assert not w._fake.entered, 'reopened after completing'
    assert w.written == first


def test_close_is_safe_when_the_window_never_opened(tmp_path):
    """A run that ends before start_step must leave no half-written file."""
    w = _window(tmp_path, start_step=10_000)
    w.step(0)
    w.close()
    assert w.done and w.written == [] and not w._fake.entered


def test_close_twice_is_safe(tmp_path):
    w = _window(tmp_path, start_step=0, active_steps=1)
    w.step(0)
    w.step(1)
    w.close()
    assert w.done and len(w.written) == 1


def test_a_write_failure_does_not_kill_the_run(tmp_path):
    """An artifact is not worth a training run. The failure is printed and
    swallowed, and `done` still latches so it is not retried every step."""
    w = _window(tmp_path, start_step=0, active_steps=1, fail_write=True)
    w._fake = _FakeProf(fail_write=True)
    w._make_profiler = lambda: w._fake
    w.step(0)
    w.step(1)
    assert w.done and w.written == []


def test_trace_config_defaults_to_off():
    from profiling import trace_from_config
    assert not trace_from_config(_Bag(), cuda=False).enabled
    assert not trace_from_config(_Bag(profiling=_Bag()), cuda=False).enabled
