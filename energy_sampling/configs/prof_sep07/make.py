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
Steady state on the battery arrived ~2 h after start on mipu (regimes.csv) and within
1 h on mip; 5700 / 19600 sit inside it. The chrome trace is ON, deliberately: the
question is the gaps between kernels, which only a timeline shows. ~93 MB/step, 8
steps, so ~0.8 GB per arm under profiling_results/.

Everything else -- batch, fracs, anchors, clip, LR -- is the committed p02 yaml.
"""
import copy, pathlib, yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

ARMS = {
    # name: (committed base, warm-src token, absolute trace start step)
    'p07_mipu_prof': ('prod_sep02/p02_mipu_lr0p0625.yaml', 'pt100_mipu_lr4p0', 5700),
    'p07_mip_prof':  ('prod_sep02/p02_mip_lr1.yaml',       'pt100_mip_lr4p0',  19600),
}


def deltas(cfg, name, start_step):
    cfg['run_name'] = name
    cfg['tag'] = 'prof07'
    cfg['epochs'] = 500000          # the wall ends the job; the trace closes itself
    cfg['profiling'] = {
        'enabled': True, 'regions': None,
        'trace': {'enabled': True, 'start_step': int(start_step), 'active_steps': 8,
                  'outdir': 'profiling_results', 'write_trace': True,
                  'record_shapes': False, 'with_stack': False},
    }
    return cfg


def main():
    rows = ['arm\twarm_src\tbase\ttrace_start']
    for name, (base, src, start) in ARMS.items():
        cfg = yaml.safe_load((ROOT / base).read_text())
        cfg = deltas(copy.deepcopy(cfg), name, start)
        (HERE / f'{name}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        rows.append(f'{name}\t{src}\t{base}\t{start}')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
