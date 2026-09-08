"""PROFILE, not science. Two prod_sep02 arms, byte-identical to the battery except
for the profiler, run once in steady state so the ~3 s/step of host-only time can be
attributed.

WHY. prod_sep02's kills are node co-tenancy inflating a host-bound loop
(configs/prod_sep02/analysis/low_util_cancellation/REPORT.md): utilization x
step_time is constant on every run, so the added time is GPU idle, and 2.5-3.8 s of
every step is time when no kernel runs at all, on every route and batch. Nothing in
the tree has ever attributed that chunk. The torch.profiler window (profiling.py
TraceWindow) is wired but has never been armed on the cluster.

WHY THESE TWO. mipu is the arm with the thinnest margin and the one the battery lost
five of. mip (ELJ) is the fastest to steady state and shows the same chunk with no
MLIP in the way, so if the two tables name the same operations the chunk is the
rollout/bookkeeping and not the energy route.

WHY THESE STEPS. `start_step` is an ABSOLUTE step index (TraceWindow.step compares
it to step_ind) and both arms resume from a phase-1 exit: mipu at 5010, mip at 19010.
The window sits a few hundred steps in -- past the transition transient, near enough
that a 3 h wall reaches it with hours to spare. The first attempt used 5700 / 19600,
which took mip 83 minutes to reach; there is no reason to spend that much of the wall
travelling to the measurement.

/!\\ ACTIVE_STEPS IS 2, NOT 8, AND --mem IS 200G. THE FIRST ATTEMPT OOM'd THE HOST.
p07_mip_prof, 2026-09-07: the window opened at step 19600 and process RSS went
2.7 GB -> 44.8 -> 48.2 GB in eight minutes against `--mem=48G`; the cgroup killed it
before 8 steps completed and nothing was written. torch.profiler buffers every CPU and
CUDA event in host RAM until the window closes, and this step launches enough of them
to outgrow a 48 GB allocation in roughly three steps -- which is itself evidence for
the dispatch-bound reading this run exists to test. Two steps is ample for a
key_averages table, and the nodes have ~940 GB free, so the memory ask is cheap
insurance rather than a tuning choice.

THE CHROME TRACE IS OFF. It was on for the first attempt to see the gaps between
kernels, but it is ~93 MB/step to write and the export doubles the peak. The table is
what a headless reader actually reads (profiling.py says so) and it is what names the
operation. Turn the trace back on, for a single step, once the table says where to
look.

Everything else -- batch, fracs, anchors, clip, LR -- is the committed p02 yaml.
"""
import copy, pathlib, yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# THE A/B. `_prof` arms are the CONTROL and inherit the battery's compile_policy
# ('auto' = five trunk submodules compiled separately). `_step` arms are the
# TREATMENT: compile_policy 'step' compiles the fused `_forward_kernel` and
# `_pb_net` as units instead, collapsing seven compiled-region entries per
# timestep to three. Same warm start, same trace step, everything else identical,
# so the two tables are directly comparable.
#
# PRE-NAMED FALSIFIER, and it is the region COUNT, not the step time.
#   control  : CompiledFunction ~5,988 calls / 2 steps (measured 2026-09-07)
#   treatment: expect ~2,600 if it took (3 of the 7 entries survive)
# If the count does NOT fall, the compile silently fell back to eager --
# `suppress_errors` guarantees that failure is otherwise invisible, and it is the
# most likely way this change does nothing while appearing live. Step time is the
# OUTCOME (expect ~1 s/step off ELJ) but it cannot distinguish "compiled and did
# not help" from "never compiled at all".
ARMS = {
    # name: (committed base, warm-src token, trace start step, compile_policy)
    'p07_mipu_prof': ('prod_sep02/p02_mipu_lr0p0625.yaml', 'pt100_mipu_lr4p0', 5400,  None),
    'p07_mip_prof':  ('prod_sep02/p02_mip_lr1.yaml',       'pt100_mip_lr4p0',  19300, None),
    'p07_mipu_step': ('prod_sep02/p02_mipu_lr0p0625.yaml', 'pt100_mipu_lr4p0', 5400,  'step'),
    'p07_mip_step':  ('prod_sep02/p02_mip_lr1.yaml',       'pt100_mip_lr4p0',  19300, 'step'),
}


def deltas(cfg, name, start_step, compile_policy):
    cfg['run_name'] = name
    cfg['tag'] = 'prof07'
    cfg['epochs'] = 500000          # the wall ends the job; the trace closes itself
    if compile_policy is not None:
        cfg['compile_policy'] = compile_policy
    cfg['profiling'] = {
        'enabled': True, 'regions': None,
        'trace': {'enabled': True, 'start_step': int(start_step), 'active_steps': 2,
                  'outdir': 'profiling_results', 'write_trace': False,
                  'record_shapes': False, 'with_stack': False},
    }
    return cfg


def main():
    rows = ['arm\twarm_src\tbase\ttrace_start\tcompile_policy']
    for name, (base, src, start, comp) in ARMS.items():
        cfg = yaml.safe_load((ROOT / base).read_text())
        cfg = deltas(copy.deepcopy(cfg), name, start, comp)
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        rows.append(f'{name}\t{src}\t{base}\t{start}\t' + str(cfg['compile_policy']))
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
