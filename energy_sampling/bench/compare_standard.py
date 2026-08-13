"""
The published single-loss methods against the ones invented here.

WHY MLE ONLY. Every arm added for this comparison assumes ONE loss and ONE set
of parameters descending it. That is what `mle` is; `equilibration` is three
players and `var_cond` has a per-condition level, and neither satisfies the
assumption any of these papers make. Running them there would be measuring the
assumption's failure, not the method.

Three passes:

  solo    each climber with braker='none', so the divergence tripwire is the
          only other thing acting. These methods are whole controllers and
          pairing them would confound them with F-021's pairing effect.
  paired  the ones that survived solo, against every braker.
  floor   the same board on `mle_floor`. Everything reading only differences or
          gradients MUST come back bit-identical; whatever moves was reading the
          loss level, which the real problem does not offer.
"""
import sys
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from bench.harness import BenchRun          # noqa: E402
from bench.scenarios import toolkit         # noqa: E402

#: the incumbent and the current best, carried in every pass as the reference
REFERENCE = [('ray', 'ray'), ('ramp', 'plateau')]

IMPORTED = ('armijo', 'bb', 'hyper', 'dog', 'sps')
LOCAL = ('none', 'ramp', 'ray', 'slope_seek')

SOLO = [(c, 'none') for c in LOCAL + IMPORTED] + REFERENCE
PAIRED = [(c, b) for c in ('armijo', 'bb', 'hyper')
          for b in BenchRun.BRAKERS] + REFERENCE
FLOOR = [(c, 'none') for c in LOCAL + IMPORTED] + REFERENCE


def main(which='solo'):
    if which == 'solo':
        toolkit(names=['mle'], pairs=SOLO, seeds=(0, 1, 2))
    elif which == 'paired':
        toolkit(names=['mle'], pairs=PAIRED, seeds=(0, 1, 2))
    elif which == 'floor':
        toolkit(names=['mle_floor'], pairs=FLOOR, seeds=(0, 1, 2))
    else:
        raise SystemExit(f'unknown pass {which!r}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'solo')
