import abc
import time
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

        Wall-clock, not CUDA events: the question is where the STEP's seconds go,
        and an un-synchronised GPU section that returns immediately is exactly the
        idle this is meant to expose. Counters are drained by ten_step_reporting.
        """
        # Forwarded ONLY when explicitly set: this base log_reward is inherited by
        # every energy, and the toy subclasses' energy() take no such argument. A
        # None default therefore leaves all existing call sites byte-identical.
        recovery_kw = {} if internal_oom_recovery is None else {
            'internal_oom_recovery': internal_oom_recovery}
        t0 = time.time()
        try:
            if return_exp:
                energy, sample = self.energy(x, mol_batch, log_temperature, return_exp,
                                             keep_grads=keep_grads, **recovery_kw)
                return -energy, sample
            else:
                return -self.energy(x, mol_batch, log_temperature, return_exp,
                                    keep_grads=keep_grads, **recovery_kw)
        finally:
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
        return {'energy/seconds': secs,
                'energy/calls': calls,
                # directly comparable across energies: uma measured ~5.5 ms/sample
                # against elj's ~0.3 at eval, and that 18x is the whole question
                'energy/ms_per_sample': 1e3 * secs / max(samples, 1)}

