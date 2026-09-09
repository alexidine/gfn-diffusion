"""
Run every arm for a FIXED WALL CLOCK, one at a time.

WHY TIME AND NOT STEPS. The arms differ by up to 30x in step rate on purpose --
that is the axis. Bounding them by `epochs` would give the fast arms forty
paired windows and the slow ones three, so the comparison would be tightest
exactly where the duty cycle is least interesting. Every arm gets the same
number of MINUTES, and therefore roughly the same number of sampler readings and
the same number of wandb system-metric points to pair against.

ONE AT A TIME, because two arms sharing the card would each see the other's
occupancy: `system.gpu.0.gpu` is per-DEVICE, not per-process, and so is
`nvidia-smi`'s. A concurrent sweep would measure co-tenancy.

The arms are killed, not finished. Nothing here writes checkpoints
(`checkpoint_read_only`), and wandb streams as it goes, so a killed run's history
is complete up to the kill.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
# .../energy_sampling/configs/utilphase_sep09 -> parents[2] is gfn_diffusion,
# parents[3] is the repo root that holds both mxtaltools/ and gfn_diffusion/
REPO = HERE.resolve().parents[3]
ENERGY = REPO / 'gfn_diffusion' / 'energy_sampling'
PY = r'C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe'
ENV_PATH = f'{REPO}\mxtaltools;{REPO}\gfn_diffusion'

#: seconds of TRAINING per arm, past the startup it does not control. Startup is
#: minutes on the elj route (prior dataset, MLIP scan), so the wall clock cannot
#: start at launch; the runner waits for the first tqdm line instead.
SECONDS = 360
#: an arm that never reaches its first step is a failure, not a slow start.
STARTUP_LIMIT = 900

ARMS = ['b500', 'b500r', 'b125', 'b2000', 'T100', 'w128', 'w1024', 'lgauss', 'uma']


def run(arm, logdir):
    cfg = f'configs/utilphase_sep09/{arm}.yaml'
    log = Path(logdir) / f'{arm}.log'
    env = {**__import__('os').environ, 'PYTHONPATH': ENV_PATH}
    with open(log, 'w') as fh:
        p = subprocess.Popen([PY, 'train.py', '--config', cfg], cwd=str(ENERGY),
                             stdout=fh, stderr=subprocess.STDOUT, env=env)
        started, t0 = False, time.time()
        while p.poll() is None:
            time.sleep(5)
            text = log.read_text(errors='ignore')
            if not started and ('it/s' in text or 's/it' in text):
                # t0 is reassigned here, so the setup duration must be read off the
                # OLD one -- printing it after the reassignment reports 0 every time.
                started, setup = True, time.time() - t0
                t0 = time.time()
                print(f'  {arm}: training started after {setup:.0f}s of setup')
            if not started and time.time() - t0 > STARTUP_LIMIT:
                p.kill()
                return f'FAILED to start within {STARTUP_LIMIT}s'
            if started and time.time() - t0 > SECONDS:
                p.kill()
                return f'ran {SECONDS}s'
        return f'EXITED on its own (code {p.returncode}) -- check {log.name}'


def main():
    logdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        print(f'{arm}: launching', flush=True)
        print(f'  {arm}: {run(arm, logdir)}', flush=True)


if __name__ == '__main__':
    main()
