"""Run MXtalTools crystal searches in turn with this process capped at a fraction of GPU memory (run_search sets 0.9).

usage: python -m eval.exp6_recall.run_capped FRACTION CONFIG.yaml [CONFIG.yaml ...]
Each config's wall time is printed; a config that raises is reported and the next one runs.
"""
import sys
import time
import traceback

import torch

FRACTION = float(sys.argv[1])
_orig = torch.cuda.set_per_process_memory_fraction
torch.cuda.set_per_process_memory_fraction = lambda fraction, device=None: _orig(FRACTION, device=0)

from mxtaltools.crystal_search import run_search  # noqa: E402

for path in sys.argv[2:]:
    t0 = time.time()
    try:
        run_search.crystal_search(run_search.dict2namespace(run_search.load_yaml(path)))
        print(f'DONE {path} {time.time() - t0:.1f} s', flush=True)
    except Exception:
        traceback.print_exc()
        print(f'FAILED {path} {time.time() - t0:.1f} s', flush=True)
