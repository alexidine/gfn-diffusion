"""
Building the REAL `train.Modeller` on CPU.

Possible since the CUDA guard at `train.py:130`: `__init__` used to call
`torch.cuda.set_per_process_memory_fraction` and `torch.cuda.init()`
unconditionally, so on any machine with no visible GPU the object could not be
constructed at all.

WHAT THIS IS FOR. Not for running the bench -- `FakeModeller` is still what the
tests drive, because it is controllable and fast. This exists so the fake can be
CHECKED against the real thing (`test_fidelity.py`). A stand-in is only worth
anything while it still stands in, and the way that claim dies is silently:
someone adds an attribute to `Modeller` that a controller reads, the fake does
not grow it, and the bench keeps reporting green about a surface that no longer
matches.

`Modeller.__init__` does not touch the model, the datasets or the energy
function -- those are `init_gfn` / `init_energy_function`, called later. So this
is cheap after the ~11 s `train.py` import, and it needs no GPU and no data.

Run under a CPU-only environment (`CUDA_VISIBLE_DEVICES=-1`) or on a GPU box;
both work, and the guard is what makes the first case possible.
"""

import os
import sys

DEFAULT_CONFIG = 'configs/mk_dev.yaml'


def build_real_modeller(config=DEFAULT_CONFIG):
    """
    Construct `train.Modeller` from a real YAML config.

    `get_train_args` parses the explicit ``--config`` launch contract, so argv
    has to be shaped like a real invocation; it is restored afterwards because
    pytest's own argv is still live.
    """
    import train  # noqa: E402 -- deferred, ~11 s

    if not os.path.exists(config):
        raise FileNotFoundError(
            f'{config!r} not found. build_real_modeller resolves configs relative to '
            f'the process CWD, which must be energy_sampling/.')

    saved = list(sys.argv)
    sys.argv = ['train.py', '--config', str(config)]
    try:
        return train.Modeller()
    finally:
        sys.argv = saved


def real_args(config=DEFAULT_CONFIG):
    """The resolved args namespace for a config, without keeping the Modeller."""
    return build_real_modeller(config).args
