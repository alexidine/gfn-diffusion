"""Run the six paired-benchmark arms, STRICTLY ONE AT A TIME.

Two training runs on one card BSOD'd this machine -- the driver does not politely OOM and
there is nothing to catch after the fact -- so this never overlaps two runs, and it re-checks
that the card is free before each one rather than only at the start. A run that finds the GPU
busy waits; it does not start alongside.

Order is INTERLEAVED (uncond s0, cond s0, uncond s1, ...) rather than all of one arm then all
of the other. If the machine is interrupted halfway, an interleaved order leaves a usable
paired comparison at every prefix; a grouped order leaves three seeds of one arm and nothing
to compare them against. It also spreads any slow drift in machine state evenly across arms
instead of loading it onto whichever arm ran second.

    python configs/bench_propanol/run_all.py --dry-run      # show the plan
    python configs/bench_propanol/run_all.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # energy_sampling/
SEEDS = (0, 1, 2)
#: interleaved on purpose -- see the module docstring
PLAN = [f'bench_propanol_{arm}_s{s}' for s in SEEDS for arm in ('uncond', 'cond')]

PYTHON = r'C:\Users\mikem\venvs\csd_mxt_gfn\Scripts\python.exe'
PYPATH = ';'.join([r'C:\Users\mikem\Projects\mxt_gfn\mxtaltools',
                   r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion',
                   r'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling'])


def gpu_free_mib():
    """Free VRAM, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
        used, total = (int(v) for v in out.stdout.strip().splitlines()[0].split(','))
        return total - used
    except Exception:
        return None


def wait_for_gpu(need_mib, poll=60, limit=7200):
    """Block until the card has `need_mib` free. Returns False on timeout."""
    t0 = time.time()
    while time.time() - t0 < limit:
        free = gpu_free_mib()
        if free is None:
            print('   nvidia-smi unavailable -- proceeding without a VRAM check')
            return True
        if free >= need_mib:
            return True
        print(f'   waiting for the GPU: {free} MiB free, need {need_mib} '
              f'({int(time.time() - t0)}s)')
        time.sleep(poll)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--need-mib', type=int, default=6000,
                    help='free VRAM required before a run starts')
    ap.add_argument('--log-dir', type=Path, default=None)
    ap.add_argument('--only', nargs='*', default=None, help='run only these config stems')
    args = ap.parse_args()

    plan = [p for p in PLAN if not args.only or p in args.only]
    log_dir = args.log_dir or (HERE / 'logs')
    print(f'{len(plan)} runs, one at a time, interleaved by seed:')
    for i, name in enumerate(plan, 1):
        print(f'   {i}. {name}')
    if args.dry_run:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=PYPATH, GFN_GPU_GUARD='0')
    results = []
    for i, name in enumerate(plan, 1):
        cfg = HERE / f'{name}.yaml'
        log = log_dir / f'{name}.log'
        print(f'\n=== [{i}/{len(plan)}] {name} -> {log.name} ===', flush=True)
        if not wait_for_gpu(args.need_mib):
            print(f'   GPU never freed up; STOPPING at {name} rather than sharing the card')
            break
        t0 = time.time()
        with open(log, 'w', encoding='utf-8') as fh:
            rc = subprocess.call([PYTHON, '-u', 'conformer_modeller.py', '--config', str(cfg)],
                                 cwd=str(ROOT), env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
        dt = time.time() - t0
        results.append((name, rc, dt))
        print(f'   exit {rc} in {dt / 60:.1f} min')
        # a failed arm is reported and the rest still run: three seeds of one arm plus two of
        # the other is still a comparison, and stopping would waste the runs already done
        if rc != 0:
            print(f'   *** {name} FAILED -- see {log} ***')

    print('\n' + '=' * 60)
    for name, rc, dt in results:
        print(f'   {"ok " if rc == 0 else "FAIL"}  {name:32s} {dt / 60:6.1f} min')
    bad = [n for n, rc, _ in results if rc != 0]
    if bad:
        print(f'\n{len(bad)} run(s) failed: {", ".join(bad)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
