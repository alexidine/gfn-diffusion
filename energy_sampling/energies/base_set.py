import abc
import time

from profiling import _NULL as _NULL_REGION
import torch
import numpy as np
from torch.utils.data import Dataset


def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / (sigma ** 2)
    return torch.sum(torch.flatten(loss, start_dim=1), -1)


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None
        self._gt_ksd = None

    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x, mol_batch, log_temperature, return_exp: bool=False):
        return

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def log_reward(self, x, mol_batch, log_temperature, return_exp: bool = False, keep_grads: bool = False,
                   internal_oom_recovery=None):
        """
        The single funnel every training energy evaluation goes through, so the
        timing lives here rather than in any one energy subclass.

        WHY IT IS TIMED. prod0810's uma arm was cancelled for low GPU usage with a
        host-bound signature (45% median utilization at 190-215 W on a 400 W A100)
        that nothing in the metrics could confirm -- this cluster logs no CPU
        columns. `energy/frac_of_step` settles it: if the MLIP call is most of the
        step and the GPU is still idle, the cost is host-side (graph construction,
        collation, transfers) and more CPUs is the lever; if the call is a small
        share, the rollout is the problem instead.

        WALL-CLOCK IS THE RIGHT PRIMARY, and that is not a limitation. The
        question `energy/frac_of_step` answers is where the STEP's seconds go, so
        an un-synchronised GPU section that returns immediately SHOULD read as
        cheap -- that return is the idle being exposed.

        IT ANSWERS A DIFFERENT QUESTION FROM "how much GPU time does the energy
        consume", which is what an optimisation decision needs, and which wall
        clock cannot see through an async launch. When a region profiler is
        attached (`profiling.py`, off by default) the same call is also timed
        with CUDA events, and BOTH are reported. The gap is the diagnostic:

            wall ~ events   the call genuinely occupies the device
            wall << events  work is deferred past the return; a later region is
                            being charged for this one
            wall >> events  host-bound -- graph construction, collation,
                            transfers. prod0810's uma signature.

        Counters are drained by ten_step_reporting.
        """
        # Forwarded ONLY when explicitly set: this base log_reward is inherited by
        # every energy, and the toy subclasses' energy() take no such argument. A
        # None default therefore leaves all existing call sites byte-identical.
        recovery_kw = {} if internal_oom_recovery is None else {
            'internal_oom_recovery': internal_oom_recovery}
        # Attached by the trainer only when profiling is enabled; absent is the
        # normal case and costs one getattr against a call measured in ms.
        _prof = getattr(self, '_region_profiler', None)
        _region = _prof.region('energy') if _prof is not None else _NULL_REGION
        t0 = time.time()
        # Entered/exited by hand rather than with a `with`, so the body below is
        # byte-identical to the version that had no profiler in it. Reindenting a
        # try/finally that returns from two branches is a real chance to change
        # behaviour for a feature that is off by default.
        _region.__enter__()
        try:
            if return_exp:
                energy, sample = self.energy(x, mol_batch, log_temperature, return_exp,
                                             keep_grads=keep_grads, **recovery_kw)
                return -energy, sample
            else:
                return -self.energy(x, mol_batch, log_temperature, return_exp,
                                    keep_grads=keep_grads, **recovery_kw)
        finally:
            # Closed FIRST, so the event pair brackets the energy call and not
            # the bookkeeping after it.
            _region.__exit__(None, None, None)
            self.energy_seconds = getattr(self, 'energy_seconds', 0.0) + (time.time() - t0)
            self.energy_calls = getattr(self, 'energy_calls', 0) + 1
            self.energy_samples = getattr(self, 'energy_samples', 0) + int(len(x))

    def drain_energy_timing(self) -> dict:
        """Pop the accumulated counters. Returns {} when nothing was timed, so a
        stage that never calls the energy (bwd/dataset MLE) logs nothing at all
        rather than a misleading zero."""
        calls = getattr(self, 'energy_calls', 0)
        if not calls:
            return {}
        secs = getattr(self, 'energy_seconds', 0.0)
        samples = getattr(self, 'energy_samples', 0)
        self.energy_seconds, self.energy_calls, self.energy_samples = 0.0, 0, 0
        out = {}
        # THE GPU-SIDE COMPANION, present only while a region profiler is
        # attached. Reported beside the wall number rather than instead of it --
        # see log_reward for what their GAP means. Absent is the normal case and
        # reads as "not measured", never as zero.
        prof = getattr(self, '_region_profiler', None)
        if prof is not None:
            gpu = prof.report(prefix='energy_gpu')
            gpu_ms = gpu.get('energy_gpu/energy_ms')
            if gpu_ms is not None:
                out['energy/seconds_gpu'] = gpu_ms / 1e3
                out['energy/gpu_over_wall'] = (gpu_ms / 1e3) / secs if secs > 0 else 0.0
        out.update({'energy/seconds': secs,
                'energy/calls': calls,
                # directly comparable across energies: uma measured ~5.5 ms/sample
                # against elj's ~0.3 at eval, and that 18x is the whole question
                'energy/ms_per_sample': 1e3 * secs / max(samples, 1)})
        return out

