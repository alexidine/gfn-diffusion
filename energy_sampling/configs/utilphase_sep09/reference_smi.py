"""
The CLUSTER'S OWN INSTRUMENT, run locally beside the sweep.

Shenglong (NYU HPC) states the occupancy the scheduler judges is collected as

    nvidia-smi --query-gpu=index,timestamp,utilization.gpu \
               --format=csv,nounits,noheader --loop=10

-- a 10 s POINT sample of `utilization.gpu`, per GPU index, out of process. That
is the same instrument as the `joblogs/*_smi.csv` sidecar, and handoff §2 already
cross-checked that sidecar against wandb's `system.gpu.0.gpu` and found them in
agreement. So all three out-of-process sources are one measurement, and the
question "does our number match wandb" is really "does our number match the thing
that gets jobs cancelled".

Running it here gives that thing on THIS box, at 10 s, spanning the whole sweep,
so every arm can be compared against it without a network round trip and without
wandb's own aggregation in the way.

WINDOWS ARE 1 s WIDE, NOT 10. `utilization.gpu` reports the fraction of the last
~1 s in which a kernel was resident, so a 10 s loop observes about a tenth of the
timeline. Over the hours the scheduler averages, that is a fine estimator of the
duty cycle. Over the two-minute windows this battery compares on, it is not: a
1 Hz trace taken here during an arm measured sd 19.2 with a 2..83 range, and
10 s point sampling of it lands anywhere in 21.6..30.2 for a window whose true
mean is 25.7. Hence this file logs the cluster's cadence and the sweep's arms
sample at 2 s: same instrument, different resolution, and the difference between
them is a variance, not a disagreement.

Windows note: this box's nvidia-smi rejects `timestamp` in --query-gpu, so the
timestamp is taken here instead. It is the read time, which is what the pairing
needs anyway.
"""

import subprocess
import sys
import time

PERIOD = 10.0


def main(path):
    with open(path, 'a', buffering=1) as out:
        while True:
            t = time.time()
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu',
                 '--format=csv,nounits,noheader'], capture_output=True, text=True)
            for line in r.stdout.strip().splitlines():
                idx, util = (v.strip() for v in line.split(','))
                out.write(f'{t:.3f},{idx},{util}\n')
            time.sleep(max(0.0, PERIOD - (time.time() - t)))


if __name__ == '__main__':
    main(sys.argv[1])
