"""
Region timing that is correct on a GPU, and free when off.

WHY WALL CLOCK IS NOT ENOUGH, and why this is not a duplicate of the timing the
trainer already does. `time()` around CUDA work measures when the LAUNCHES
returned, not when the device finished. On an async stream that attributes a
region's GPU time to whichever later region happens to hit a synchronisation
point -- so a wall-clock subdivision of the step can be confidently wrong while
the step TOTAL it sums to is exactly right.

That distinction is why `_recent_step_times` and `_throughput['seconds']` stay as
they are: the whole step is bounded by real work and a wall-clock total is the
honest measure of what it costs. It is the SUBDIVISION that needs events. The
existing `energy/seconds_in_step` is a subdivision, and its own module records
that the per-step hot path deliberately does not synchronise -- so validating it
against this is the first thing worth doing, before any optimisation decision
leans on it further.

    with prof.region('energy'):
        ...
    prof.drain()                 # non-blocking; harvests finished pairs only
    prof.report()                # {'perf/energy_ms': ..., ...}, then resets

THREE PROPERTIES IT HAS TO HAVE, because it sits in the hottest code:

  1. FREE WHEN DISABLED. `region()` returns one shared no-op context; nothing is
     allocated, no clock is read, and no CUDA call is made. A disabled run must
     be bit-identical, which `tierc_smoke` is used to prove rather than assert.
  2. NEVER SYNCHRONISES THE HOST. Events are recorded on the stream and read
     back only once `query()` says the pair has completed. A profiler that
     forces a sync changes the thing it measures -- and on this codebase would
     also change the wall-clock step time everything else is calibrated on.
  3. NO RNG CONTACT. It draws nothing and consumes no randomness, so it cannot
     shift the stream the way the ray probe's replay draws do (findings F-039).

CPU FALLBACK IS WALL CLOCK, and that is correct rather than a compromise: with
no device there is nothing asynchronous to mis-attribute.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict, deque
from time import perf_counter
from typing import Optional

#: One shared no-op context, reused. A fresh `nullcontext()` per call would
#: allocate on every region of every step for a feature that is off.
_NULL = contextlib.nullcontext()

#: How many steps a pair is given to finish before `drain` stops waiting for it.
#: Events are read only when `query()` reports completion, so this bounds memory
#: rather than correctness -- a pair still running after this many drains is
#: dropped and counted, because holding it forever is how a "profiler" becomes a
#: leak on a long run.
_MAX_PENDING = 512


class RegionProfiler:
    """Named-region GPU timing, accumulated between reports.

    Holds no reference to the trainer. It is constructed with what it needs and
    is a pure function of the calls made into it, so a test can drive it without
    a model, a device, or a config."""

    def __init__(self, enabled: bool = False, cuda: bool = False,
                 regions: Optional[tuple[str, ...]] = None):
        self.enabled = bool(enabled)
        self.cuda = bool(cuda)
        #: None means "time every region asked for". A tuple restricts it, so a
        #: run can pay for one region without paying for all of them.
        self.regions = tuple(regions) if regions else None
        self._pending = deque(maxlen=_MAX_PENDING)
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self.dropped = 0            # pairs that never completed in time
        self.overflowed = 0         # pairs evicted by the deque bound

    # ---------------------------------------------------------------- capture

    def _wanted(self, name: str) -> bool:
        return self.enabled and (self.regions is None or name in self.regions)

    def region(self, name: str):
        """Time `name`. A no-op context when disabled or not selected."""
        if not self._wanted(name):
            return _NULL
        return _CudaRegion(self, name) if self.cuda else _WallRegion(self, name)

    def _submit(self, name: str, start, end) -> None:
        if len(self._pending) == self._pending.maxlen:
            self.overflowed += 1
        self._pending.append((name, start, end))

    def _record_wall(self, name: str, ms: float) -> None:
        self._totals[name] += ms
        self._counts[name] += 1

    # ----------------------------------------------------------------- harvest

    def drain(self) -> int:
        """Harvest completed pairs. Non-blocking; returns how many were read.

        WAITS FOR NOTHING. `Event.query()` is a poll -- a pair still in flight is
        left for the next drain. This is what keeps the profiler out of the
        critical path: the alternative, `synchronize()`, would serialise the very
        overlap it is measuring."""
        if not self.enabled or not self.cuda or not self._pending:
            return 0
        read, keep = 0, deque(maxlen=_MAX_PENDING)
        while self._pending:
            name, start, end = self._pending.popleft()
            if end.query():
                self._totals[name] += start.elapsed_time(end)
                self._counts[name] += 1
                read += 1
            else:
                keep.append((name, start, end))
        self._pending = keep
        return read

    def report(self, prefix: str = 'perf') -> dict:
        """Totals since the last report, in ms, then reset.

        Emits a COUNT beside every total. A region timed 3 times and a region
        timed 300 times can produce the same total, and a reader with only the
        total cannot tell a slow region from a frequent one."""
        self.drain()
        if not self.enabled:
            return {}
        out = {}
        for name, total in sorted(self._totals.items()):
            n = self._counts[name]
            out[f'{prefix}/{name}_ms'] = total
            out[f'{prefix}/{name}_n'] = float(n)
            if n:
                out[f'{prefix}/{name}_ms_mean'] = total / n
        if self.dropped or self.overflowed:
            out[f'{prefix}/lost_pairs'] = float(self.dropped + self.overflowed)
        self._totals.clear()
        self._counts.clear()
        return out

    def pending(self) -> int:
        return len(self._pending)


class _WallRegion:
    """CPU path. Correct here: with no device there is nothing async to
    mis-attribute, so `perf_counter` measures the region and not its launches."""

    __slots__ = ('prof', 'name', 't0')

    def __init__(self, prof: RegionProfiler, name: str):
        self.prof, self.name = prof, name

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        self.prof._record_wall(self.name, (perf_counter() - self.t0) * 1e3)
        return False


class _CudaRegion:
    """GPU path: a recorded event pair, read back later by `drain`."""

    __slots__ = ('prof', 'name', 'start', 'end')

    def __init__(self, prof: RegionProfiler, name: str):
        self.prof, self.name = prof, name

    def __enter__(self):
        import torch
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        import torch
        self.end = torch.cuda.Event(enable_timing=True)
        self.end.record()
        # Submitted even when the body raised: the launches happened, so the
        # elapsed time is real, and dropping it would silently under-report
        # exactly the steps that went wrong.
        self.prof._submit(self.name, self.start, self.end)
        return False


def from_config(args, cuda: bool) -> RegionProfiler:
    """Build from `args.profiling`, defaulting to OFF.

    Absent block = disabled, and that is the one default where absence is
    unambiguous: a profiler nobody asked for should not be running."""
    node = getattr(args, 'profiling', None)
    if node is None:
        return RegionProfiler(enabled=False, cuda=cuda)
    return RegionProfiler(
        enabled=bool(getattr(node, 'enabled', False)),
        cuda=cuda,
        regions=getattr(node, 'regions', None))

# ---------------------------------------------------------------------------
# Layer 2: a bounded torch.profiler window
# ---------------------------------------------------------------------------

class TraceWindow:
    """Profile a FEW steps once, write a trace, then cost nothing for the rest
    of the run.

    WHY A WINDOW RATHER THAN A FLAG. `torch.profiler` records every op on both
    devices; left on it dominates the thing it measures and produces a trace too
    large to open. Bounded to a handful of steps it is affordable at any run
    length, because the cost does not scale with the run -- and after the window
    closes this object disables itself permanently, so the check that remains in
    the loop is one boolean.

    WHY IT STARTS LATE. `start_step` defaults past the first stage transition and
    well past warmup. The opening steps of a run are unrepresentative -- lazy CUDA
    context creation, autotuning, the first compile, cold caches -- and a profile
    of them measures startup rather than training.

    OUTPUT IS FILES, NEVER METRICS. A chrome trace and a text table land beside
    the producer; nothing reaches wandb. That is what keeps this usable on the
    cluster, where the UI is not available and an artifact is the only channel.
    """

    def __init__(self, enabled: bool = False, start_step: int = 1500,
                 active_steps: int = 8, outdir: str = 'profiling_results',
                 write_trace: bool = False,
                 cuda: bool = False, record_shapes: bool = False,
                 with_stack: bool = False, tag: str = 'run'):
        self.enabled = bool(enabled)
        self.start_step = int(start_step)
        self.active_steps = max(1, int(active_steps))
        self.write_trace = bool(write_trace)
        self.outdir, self.cuda, self.tag = outdir, bool(cuda), tag
        self.record_shapes, self.with_stack = bool(record_shapes), bool(with_stack)
        self._prof = None
        self._opened_at: Optional[int] = None
        self.done = False
        self.written: list[str] = []

    def step(self, step_ind: int) -> None:
        """Call once per training step. Returns immediately unless armed."""
        if self.done or not self.enabled:
            return
        if self._prof is None:
            if step_ind < self.start_step:
                return
            self._prof = self._make_profiler()
            self._prof.__enter__()
            self._opened_at = step_ind
            print(f'profiling: trace window OPEN at step {step_ind} '
                  f'({self.active_steps} steps)')
            return
        if step_ind - self._opened_at >= self.active_steps:
            self.close()

    def close(self) -> None:
        """Shut the window and write. Safe to call twice, and safe to call when
        the window never opened -- a run that ends early leaves no half file."""
        if self.done or self._prof is None:
            self.done = True
            return
        prof, self._prof = self._prof, None
        prof.__exit__(None, None, None)
        self.done = True
        try:
            self.written = self._write(prof)
            for path in self.written:
                print(f'profiling: wrote {path}')
        except Exception as e:            # never let an artifact write kill a run
            print(f'profiling: trace write FAILED ({type(e).__name__}: {e}); '
                  f'training is unaffected')

    # ------------------------------------------------------------- internals

    def _make_profiler(self):
        import torch
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda:
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        return torch.profiler.profile(
            activities=acts, record_shapes=self.record_shapes,
            with_stack=self.with_stack, profile_memory=False)

    def _write(self, prof) -> list[str]:
        os.makedirs(self.outdir, exist_ok=True)
        stem = os.path.join(self.outdir, f'{self.tag}_step{self._opened_at}')
        out = []
        # THE TABLE FIRST. The chrome trace is the complete record but needs a
        # viewer; the table is what a headless reader actually reads, and it is
        # the artifact that survives being pasted into a terminal.
        sort = 'cuda_time_total' if self.cuda else 'cpu_time_total'
        table = prof.key_averages().table(sort_by=sort, row_limit=40)
        with open(f'{stem}_table.txt', 'w', encoding='utf-8') as f:
            header = (f'# {self.active_steps} steps from step {self._opened_at}, sorted by {sort}')
            f.write(header + chr(10) * 2 + table + chr(10))
        out.append(f'{stem}_table.txt')
        # THE CHROME TRACE IS OPTIONAL, and off by default, because its SIZE is
        # what bounds how many steps may be profiled. Measured 2026-08-19 on the
        # ELJ route: 748 MB for 8 steps -- ~93 MB/step, with record_shapes and
        # with_stack both off -- which is effectively unopenable and pins
        # active_steps at single digits. The table above is 9 KB and is where
        # every result so far came from. Turn the trace on deliberately, for a
        # short window, when a timeline view is the actual question.
        if self.write_trace:
            prof.export_chrome_trace(f'{stem}_trace.json')
            out.append(f'{stem}_trace.json')
        return out


def trace_from_config(args, cuda: bool, tag: str = 'run') -> TraceWindow:
    """Build from `args.profiling.trace`, defaulting to OFF."""
    node = getattr(getattr(args, 'profiling', None), 'trace', None)
    if node is None:
        return TraceWindow(enabled=False, cuda=cuda, tag=tag)
    return TraceWindow(
        enabled=bool(getattr(node, 'enabled', False)),
        start_step=int(getattr(node, 'start_step', 1500) or 1500),
        active_steps=int(getattr(node, 'active_steps', 8) or 8),
        write_trace=bool(getattr(node, 'write_trace', False)),
        outdir=str(getattr(node, 'outdir', 'profiling_results')),
        record_shapes=bool(getattr(node, 'record_shapes', False)),
        with_stack=bool(getattr(node, 'with_stack', False)),
        cuda=cuda, tag=tag)
