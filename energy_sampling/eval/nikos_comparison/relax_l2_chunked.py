"""
L2 rigid-body relaxation for the whole Nikos pool, in small resumable chunks.

`levels.py --skip-l2` was how the artifact was built, so `nikos_levels.pt` carries
`l2: None` and every COMPACK number in the comparison is UNRELAXED-his against
RELAXED-ours. That is not like-for-like: our landscape members are optimiser
outputs, so his structures are compared from a systematically different place.

Why this exists instead of just dropping `--skip-l2`: `levels.build_l2` collates
every structure into ONE batch and runs 500 rprop steps at a 10 A cutoff. On this
box that batch took the machine down. Here the pool is relaxed a couple of
structures at a time, the CUDA cache is dropped between chunks, and each chunk is
checkpointed to disk the moment it finishes -- so a crash costs one chunk, not the
run, and `--resume` picks up where it stopped.

    python relax_l2_chunked.py --chunk 2            # relax, checkpointing as it goes
    python relax_l2_chunked.py --resume             # continue after an interruption
    python relax_l2_chunked.py --merge              # write l2 into nikos_levels.pt

Then the existing comparison can be run at the relaxed level:

    python compare.py --level l2
"""
import argparse
import gc
import os

import torch

from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import analyze, rdf_gap
from energy_sampling.utils import load_yaml

HERE = os.path.dirname(os.path.abspath(__file__))

#: The settings that ALREADY produced a completed L2 for the sg14-Z'1 subset
#: (atlas/relax_states.py), not levels.OPT_CONFIG. Two reasons: these are known to
#: run to completion on this box, and the three sg14-Z'1 structures relaxed with
#: them are already published in the atlas -- relaxing the other ten under a
#: different optimiser would put two definitions of "L2" in one table.
OPT = dict(optimizer_func='rprop', init_lr=0.01, max_num_steps=120,
           convergence_eps=1e-5, grad_norm_clip=0.1,
           cutoff=10, enforce_reduced=True, anneal_lr=True,
           compression_factor=0.0, target_packing_coeff=None, show_tqdm=False)

#: A SECOND, gentler arm. `stage2` moves structures a median RDF 0.116 and a max
#: 0.254 -- past the 0.10 cut that defines a basin -- so a match that appears only
#: after relaxing is ambiguous between "the basin was always there" and "the
#: optimiser walked it somewhere else". `gentle` is levels.OPT_CONFIG: a 100x
#: smaller initial step, no annealing, and a tighter convergence test, so it
#: polishes in place rather than travelling. Reporting BOTH is what separates the
#: two readings -- if a match survives the gentle arm it was not manufactured.
PRESETS = dict(
    stage2=OPT,
    gentle=dict(optimizer_func='rprop', init_lr=1e-4, max_num_steps=500,
                convergence_eps=1e-8, grad_norm_clip=0.1,
                cutoff=10, enforce_reduced=False, anneal_lr=False,
                compression_factor=0.0, target_packing_coeff=None,
                show_tqdm=False),
)

#: MACE at a 10 A cutoff took this machine down once, and filled 16 GB of VRAM on a
#: SINGLE structure. A fraction cap turns an overrun into a catchable Python OOM
#: instead of exhausting VRAM underneath the OS.
VRAM_FRACTION = 0.55


def relax(sub, ef, predictor, device, opt=None):
    """Rigid-body relax one small batch. No trajectory record is kept.

    `levels.build_l2` asks for `return_record=True`, which retains a snapshot per
    optimiser step; at a 10 A cutoff that is most of the footprint. Nothing
    downstream reads the record, so it is not requested here.
    """
    cfg = dict(opt or OPT, optim_target=ef, predictor=predictor)
    batch = sub.clone().to(device)
    out = batch.optimize_crystal_parameters(**cfg)
    res = collate_data_list(out)
    res.identifier = list(sub.identifier)
    return res


def free(device):
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak(device):
    if not str(device).startswith('cuda'):
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(HERE, 'config.yaml'))
    ap.add_argument('--device', default='cpu',
                    help="default cpu -- this relaxation filled 16 GB of VRAM on "
                         "a SINGLE structure and took the machine down; use "
                         "--device cuda only with the fraction cap in place")
    ap.add_argument('--chunk', type=int, default=2,
                    help='structures relaxed per batch (default 2, keep it small)')
    ap.add_argument('--max-steps', type=int, default=None,
                    help='override OPT_CONFIG max_num_steps')
    ap.add_argument('--resume', action='store_true',
                    help='keep chunks already in the checkpoint and do the rest')
    ap.add_argument('--merge', action='store_true',
                    help='write the checkpointed l2 into nikos_levels.pt and stop')
    ap.add_argument('--preset', default='stage2', choices=sorted(PRESETS),
                    help="'stage2' (default) is the optimiser the published atlas "
                         "relaxations used; 'gentle' takes 100x smaller steps so a "
                         "structure polishes in place. Run BOTH -- see PRESETS.")
    ap.add_argument('--polymorphs', action='store_true',
                    help='relax the 7 experimental polymorphs instead of his pool, '
                         'to polymorphs_l2.pt. Needed for the like-for-like '
                         'comparison: matching his RELAXED structure against an '
                         'UNRELAXED experimental reference still compares two '
                         'different points on the surface.')
    ap.add_argument('--threads', type=int, default=6,
                    help='torch CPU threads (default 6). MACE scales POORLY here: '
                         'measured on a 24-core box, 24 threads is only 1.3x '
                         'faster than 4 while taking 9.5 cores instead of 2.9. '
                         'Capping costs ~11%% of wall-clock and leaves the machine '
                         'usable. 0 = leave the torch default alone.')
    ap.add_argument('--ckpt', default=None)
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    opt = dict(PRESETS[cli.preset])
    suffix = '' if cli.preset == 'stage2' else '_' + cli.preset
    if cli.threads:
        torch.set_num_threads(cli.threads)
        #: set_num_interop_threads raises once any parallel work has happened, and
        #: it is the less important of the two -- don't let it kill the run
        try:
            torch.set_num_interop_threads(cli.threads)
        except RuntimeError:
            pass
    device = cli.device
    ef = cfg['energy_function']
    lev_path = os.path.join(cfg['out_dir'], 'nikos_levels.pt')
    ckpt = cli.ckpt or os.path.join(cfg['out_dir'],
                                    f'nikos_l2{suffix}_chunks.pt')

    if cli.polymorphs:
        poly = torch.load(cfg['polymorphs'], weights_only=False,
                          map_location='cpu').cpu()
        ids = list(poly.identifier)
        src = {k: c for k, c in zip(ids, poly.batch_to_list())}
        ckpt = cli.ckpt or os.path.join(
            cfg['out_dir'], f'polymorphs_l2{suffix}_chunks.pt')
        out_path = os.path.join(cfg['out_dir'], f'polymorphs_l2{suffix}.pt')
        print(f"polymorphs: {len(ids)} experimental structures -> {out_path}")
    else:
        lev = torch.load(lev_path, weights_only=False, map_location='cpu')
        l1 = lev['l1']
        ids = list(l1.identifier)
        src = {k: c for k, c in zip(ids, l1.batch_to_list())}
        out_path = None
        print(f"pool: {len(ids)} structures at L1")

    done = {}
    if (cli.resume or cli.merge) and os.path.exists(ckpt):
        done = torch.load(ckpt, weights_only=False, map_location='cpu')
        print(f"checkpoint: {len(done)} already relaxed "
              f"({', '.join(sorted(done)) if len(done) < 8 else '...'})")

    if cli.merge:
        missing = [i for i in ids if i not in done]
        if missing:
            raise SystemExit(f"cannot merge: {len(missing)} not relaxed yet "
                             f"({', '.join(missing)}). Run without --merge first.")
        if ef == 'mace':
            from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
            predictor = load_mace_model(cfg['mace_model'], device, torch.float32)
        else:
            raise NotImplementedError(ef)
        l2 = collate_data_list([done[i] for i in ids])
        l2.identifier = list(ids)
        #: score and RDF the relaxed set under the SAME definitions as L0/L1, or
        #: the levels are not comparable and the whole point is lost
        l2 = analyze(l2, ef, predictor, device, chunk=cli.chunk)
        l2.identifier = list(ids)
        moved = rdf_gap(l1, l2)
        drop = l1[ef] - l2[ef]
        print(f"  relaxation lowered {ef} by median {drop.median():.2f} kJ/mol "
              f"(max {drop.max():.2f})")
        print(f"  L1->L2 RDF distance: median {moved.median():.4f}, "
              f"max {moved.max():.4f}")
        lev['l2' + suffix] = l2
        lev['l1_l2_rdf_gap' + suffix] = moved
        lev['opt_config' + suffix] = dict(opt)
        torch.save(lev, lev_path)
        print(f"merged l2 into {lev_path}")
        for i, k in enumerate(ids):
            print(f"  {k:10s} {float(l1[ef][i]):9.2f} -> {float(l2[ef][i]):9.2f}  "
                  f"moved {float(moved[i]):.4f} RDF")
        return

    if ef == 'mace':
        from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
        predictor = load_mace_model(cfg['mace_model'], device, torch.float32)
    else:
        raise NotImplementedError(ef)

    if cli.max_steps:
        opt['max_num_steps'] = cli.max_steps
    if str(device).startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
        print(f"VRAM capped at {VRAM_FRACTION:.0%} of "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    todo = [i for i in ids if i not in done]
    print(f"preset {cli.preset}: init_lr {opt['init_lr']}, "
          f"{opt['max_num_steps']} steps")
    print(f"relaxing {len(todo)} structures, {cli.chunk} at a time, on {device}"
          f"{f' with {cli.threads} threads' if cli.threads else ''}")
    by_id = src

    for s in range(0, len(todo), cli.chunk):
        names = todo[s:s + cli.chunk]
        sub = collate_data_list([by_id[k].clone() for k in names])
        sub.identifier = list(names)
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        out = relax(sub, ef, predictor, device, opt)
        got = out.batch_to_list()
        for n, k in enumerate(names):
            done[k] = got[n].cpu()
        #: checkpoint EVERY chunk -- the whole reason this is chunked is that the
        #: previous attempt did not survive to the end
        torch.save(done, ckpt)
        del sub, out, got
        free(device)
        #: flush -- python block-buffers stdout through a pipe, so without this
        #: a long run looks identical to a hung one from outside
        print(f"  [{len(done):2d}/{len(ids)}] {', '.join(names):24s} "
              f"peak {peak(device):.2f} GB", flush=True)

    print(f"\nall {len(done)} relaxed; checkpoint at {ckpt}")
    if cli.polymorphs:
        rel = collate_data_list([done[i] for i in ids])
        rel.identifier = list(ids)
        rel.aunit_handedness = rel.aunit_handedness.abs()
        torch.save(rel, out_path)
        print(f"wrote {out_path}")
    else:
        print(f"now run:  python relax_l2_chunked.py --merge")


if __name__ == '__main__':
    main()
