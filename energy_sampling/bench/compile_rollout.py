"""Does compiling the rollout actually reduce compiled-region entries, and does it pay?

WHY THIS EXISTS. prod_sep02's low-occupancy cancellations are a host-side problem:
utilization x step_time is constant across every fast and slow regime, so the added
time under a noisy neighbour is GPU idle, and 2.5-3.8 s of every step is time when no
kernel runs at all (configs/prod_sep02/analysis/low_util_cancellation/). The cluster
profiler named the cost: `CompiledFunction` fires ~2,994 times per training step at
~0.73 ms of self CPU each, which is ~2.2 s and accounts for essentially all of it.
One `_fwd_step` touches SEVEN separately-compiled submodules (three in
`_forward_kernel`, three in `_pb_net`, one for flow), and at T=100 across three
branches that is where the 3,000 comes from.

THE POINT OF A MICROBENCHMARK. Verifying this in a training arm costs a 3 h wall and
a GPU, and the first attempt at that OOM'd the host inside the profiler window. The
rollout needs no MLIP, no data and no optimizer, so the same question answers in
minutes. AND IT CAN ONLY BE ASKED ON THE CLUSTER: inductor has no CUDA backend on
native Windows, so `compile_policy: auto` resolves OFF on the dev box and every
compile-dependent result is unreachable there
([[project_compile_only_failures_invisible_locally]]).

READ THE LAUNCH COUNT, NOT THE TIME. `suppress_errors` is on (as in training), so a
compile that specialises badly blows `cache_size_limit` and falls back to eager
SILENTLY, which looks exactly like "compiled fine, didn't help". Counts separate
those; wall time cannot.

AND LAUNCHES, NOT REGION ENTRIES. Measured 2026-09-07: compiling the fused MLP kernels
cut entries 1797 -> 300 and bought nothing, so entries were never the cost. Launches
barely moved (116,510 -> 104,510). Note launches are IDENTICAL across routes and batch
sizes -- 116,510 for ELJ at 4000 and UMA at 1600 -- because they are a property of the
graph, not the data. That is why the host cost is a fixed number of seconds per step
regardless of batch, and it also means a SMALL batch answers the launch question when
a large one will not fit.

    python -m bench.compile_rollout --checkpoint <a phase1_exit .pt> --batch 1600
    python -m bench.compile_rollout --checkpoint ... --modes eager,auto,step --reps 5

Modes mirror `Modeller.maybe_compile_policy` exactly (same dynamo config, same trunk
tuple); they are restated here rather than imported because that method is bound to a
Modeller with a full training context, and this script deliberately builds nothing but
the policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# PATH BOOTSTRAP, same shape as migrate_buffer_sidecar.py. models/gfn.py imports
# `energy_sampling.utils`, so the package's PARENT has to be importable, while
# train.py's own imports are bare (`models.gfn`) and need energy_sampling itself.
# Both routes coexist in the real program; mirror it rather than picking one, or
# this benchmark builds a GFN that is a different class object from the trained one
# (see the dual-import-identity trap).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

# the five submodules `compile_policy: auto` compiles individually
TRUNK = ('t_model', 's_model', 'forward_policy', 'backward_policy', 'flow_model')


def _dynamo_config():
    """Exactly train.py's settings -- a benchmark under different dynamo config is
    measuring a different program. donated_buffer in particular changes what
    AOTAutograd traces, and it is baked in at trace time."""
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
    _dynamo.config.cache_size_limit = 24
    import torch._functorch.config as _fc
    _fc.donated_buffer = False


def build(ckpt_path, device, traj_checkpoint: bool):
    """A GFN with the checkpoint's own architecture. Weights are loaded when they fit
    so the benchmark runs the real shapes; a mismatch is not fatal here because timing
    depends on shape, not on the values."""
    from models.gfn import GFN
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = blob['gfn_config']
    model = GFN(**cfg).to(device)
    try:
        model.load_state_dict(blob['model_train'])
    except Exception as e:
        print(f'  (weights not loaded: {type(e).__name__}; architecture is what matters here)')
    model.traj_checkpoint = bool(traj_checkpoint)
    return model, cfg


def apply_compile(model, mode: str):
    if mode == 'eager':
        return
    _dynamo_config()
    if mode == 'auto':
        for name in TRUNK:
            mod = getattr(model, name, None)
            if isinstance(mod, torch.nn.Module):
                mod.compile()
    elif mode == 'step':
        mod = getattr(model, 'flow_model', None)
        if isinstance(mod, torch.nn.Module):
            mod.compile()
        model.compile_step_kernels()
    else:
        raise SystemExit(f'unknown mode {mode!r}')


def one_rollout(model, batch, T, backward: bool):
    from utils import uniform_discretizer
    x0 = torch.zeros(batch, model.dim, device=model.device)
    disc = lambda bsz: uniform_discretizer(bsz, T)
    out = model.get_traj_fwd(x0, disc, None, None, None, detach_traj=not backward)
    if backward:
        # mirror training: a scalar off the trajectory, then a real backward, so
        # CompiledFunctionBackward entries are counted too
        states, logpf, logpb = out[0], out[1], out[2]
        (logpf.sum() + logpb.sum()).backward()
        model.zero_grad(set_to_none=True)
    return out


def run_mode(ckpt, mode, batch, T, reps, backward, device, traj_checkpoint):
    print(f'\n=== mode {mode} ===', flush=True)
    model, _ = build(ckpt, device, traj_checkpoint)
    apply_compile(model, mode)

    # WARMUP IS NOT OPTIONAL. Compilation is lazy at first forward, so an unwarmed
    # timing measures the compiler, not the program. Two reps: the first traces, the
    # second confirms the traced graph is reused rather than recompiled.
    for _ in range(2):
        one_rollout(model, batch, T, backward)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        one_rollout(model, batch, T, backward)
    torch.cuda.synchronize()
    per_rollout = (time.perf_counter() - t0) / reps

    # TIME THE PROFILED ROLLOUT TOO. Without this the table carries self CUDA from
    # one rollout and wall from a different one, and dividing them is meaningless --
    # on ELJ that produced a "GPU busy" figure of 145%, which is what exposed the gap.
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    t1 = time.perf_counter()
    with torch.profiler.profile(activities=acts) as prof:
        one_rollout(model, batch, T, backward)
        torch.cuda.synchronize()
    prof_wall = time.perf_counter() - t1
    rows = {e.key: e for e in prof.key_averages()}

    def count(k):
        return int(rows[k].count) if k in rows else 0

    def self_cpu_ms(k):
        return (rows[k].self_cpu_time_total / 1e3) if k in rows else 0.0

    entries = count('CompiledFunction') + count('CompiledFunctionBackward')
    launches = count('cudaLaunchKernel') + count('cuLaunchKernel')
    total_self_cpu_ms = sum(e.self_cpu_time_total for e in prof.key_averages()) / 1e3
    total_self_cuda_ms = sum(e.self_device_time_total if hasattr(e, 'self_device_time_total')
                             else e.self_cuda_time_total for e in prof.key_averages()) / 1e3

    # ⚠ NOT AN OCCUPANCY FRACTION. This is summed kernel time over wall, and kernels
    # OVERLAP across streams (autograd uses more than one), so it exceeded 1.0 on ELJ
    # -- 1.421 on 2026-09-07. Read it as a kernel-time-to-wall RATIO and as a relative
    # comparison between modes on one card; it cannot be read as "the GPU was busy
    # X% of the time". A real occupancy number has to come from the NVML sampler.
    busy_prof = round((total_self_cuda_ms / 1e3) / prof_wall, 3) if prof_wall else None
    busy_clean = round((total_self_cuda_ms / 1e3) / per_rollout, 3) if per_rollout else None
    res = dict(mode=mode, s_per_rollout=round(per_rollout, 4),
               prof_wall_s=round(prof_wall, 4),
               gpu_busy_vs_prof_wall=busy_prof, gpu_busy_vs_clean_wall=busy_clean,
               compiled_region_entries=entries,
               compiled_fn=count('CompiledFunction'),
               compiled_fn_backward=count('CompiledFunctionBackward'),
               compiled_fn_self_cpu_ms=round(self_cpu_ms('CompiledFunction'), 1),
               kernel_launches=launches,
               self_cpu_ms=round(total_self_cpu_ms, 1),
               self_cuda_ms=round(total_self_cuda_ms, 1))
    print('  ' + json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='any .pt carrying gfn_config')
    ap.add_argument('--modes', default='eager,auto,step')
    ap.add_argument('--batch', type=int, default=1600)
    ap.add_argument('--T', type=int, default=100)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--no-backward', action='store_true',
                    help='forward rollout only (halves the entry count; the training '
                         'step does both, so the default is both)')
    ap.add_argument('--traj-checkpoint', default='true',
                    help='match the config under test; p02 arms run true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    import platform
    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    if platform.system() != 'Linux' and any(m != 'eager' for m in modes):
        raise SystemExit(
            "compile modes need inductor + triton, which has no CUDA backend on "
            "native Windows -- torch raises TritonMissing rather than degrading, and "
            "train.py's `compile_policy: auto` resolves OFF here for the same reason. "
            "Run the compile modes on the cluster. `--modes eager` works anywhere and "
            "is still worth running locally: it shows how CPU-dominant the rollout is "
            "before any compiler touches it.")
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA here. This benchmark only means anything where '
                         'inductor has a CUDA backend, i.e. the cluster -- on native '
                         'Windows compile_policy resolves OFF and every mode would '
                         'measure the same eager program.')

    device = 'cuda'
    tc = str(a.traj_checkpoint).lower() in ('1', 'true', 'yes')
    print(f'checkpoint {os.path.basename(a.checkpoint)}  batch {a.batch}  T {a.T}  '
          f'reps {a.reps}  backward {not a.no_backward}  traj_checkpoint {tc}')
    print(f'torch {torch.__version__}  device {torch.cuda.get_device_name(0)}')

    # A MODE THAT OOMs MUST NOT TAKE THE OTHERS WITH IT. compile_policy 'step'
    # OOM'd both routes on 2026-09-07 (64-75 GiB): compiling the step body defeats
    # gradient checkpointing, because AOTAutograd saves the compiled region's own
    # activations and the outer torch.utils.checkpoint cannot discard them. That is
    # a RESULT, not a crash, and the modes after it still need to report.
    out = []
    for m in modes:
        try:
            out.append(run_mode(a.checkpoint, m, a.batch, a.T, a.reps,
                                not a.no_backward, device, tc))
        except torch.OutOfMemoryError as e:
            print(f'  MODE {m} OUT OF MEMORY: {str(e).splitlines()[0]}', flush=True)
            print(f'  (this is a finding: {m} needs more activation memory than the card '
                  f'has at batch {a.batch}. Launch count is INDEPENDENT of batch, so rerun '
                  f'at a small batch to get it anyway.)', flush=True)
            out.append(dict(mode=m, oom=True, s_per_rollout=float('nan'),
                            kernel_launches=0, compiled_region_entries=0,
                            self_cpu_ms=float('nan'), self_cuda_ms=float('nan')))
            torch.cuda.empty_cache()

    print('\n' + 'mode'.ljust(8) + 'launches'.rjust(11) + 's/rollout'.rjust(11)
          + 'kern/wall'.rjust(10) + 'entries'.rjust(9) + 'self CPU ms'.rjust(13)
          + 'self CUDA ms'.rjust(14))
    print('-' * 77)
    base = out[0] if out else None
    for r in out:
        speed = ('' if r is base or not base['s_per_rollout']
                 else f"   ({base['s_per_rollout'] / r['s_per_rollout']:.2f}x vs {base['mode']})")
        # the clean-wall ratio: kernel durations barely move under profiling but the
        # profiled wall inflates badly, so this is the comparable one. Above 1.0 means
        # overlapping streams, NOT an occupancy over 100%.
        busy = r.get('gpu_busy_vs_clean_wall')
        print(r['mode'].ljust(8) + f"{r['kernel_launches']:11d}"
              + f"{r['s_per_rollout']:11.4f}" + (f"{busy:10.2f}" if busy else ' ' * 10)
              + f"{r['compiled_region_entries']:9d}"
              + f"{r['self_cpu_ms']:13.1f}" + f"{r['self_cuda_ms']:14.1f}" + speed)

    print('\nREAD THE LAUNCH COUNT FIRST. Region entries are NOT the cost -- cutting '
          'them 6x bought nothing on 2026-09-07. If launches do not fall well below '
          'eager, either the compile did not take (suppress_errors hides that) or it '
          'fused nothing, and the timing beside it is measuring the same program.')

    os.makedirs(RESULTS, exist_ok=True)
    path = a.out or os.path.join(RESULTS, 'compile_rollout.json')
    with open(path, 'w') as f:
        json.dump(dict(batch=a.batch, T=a.T, reps=a.reps, backward=not a.no_backward,
                       traj_checkpoint=tc, torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(0), rows=out), f, indent=2)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
