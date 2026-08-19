"""
Re-score every stored UMA energy through the corrected neighbour graph (F-047).

WHY. Until 2026-08-19 the UMA route let fairchem build its own neighbour graph,
and that builder assumes atoms lie inside the unit cell. `unit_cell_pos` is not
wrapped, so contacts were silently dropped -- median crystal -0.06 kJ/mol, tail
crystals up to -85 kJ/mol, systematically TOO LOW (docs/findings.md F-047). Every
`uma` attribute written to disk before that date is wrong by that amount, and
nothing downstream re-derives it: the prior THINNING selects on these energies
(`e_min + 6 kT`), and `calibrate_energy_function_vs_uma` fits the ELJ thermal
scaling factor against them.

WHAT IT TOUCHES. Search-output chunks carrying a `uma` attribute. The acridine
chunks are MACE-only and the mace/elj routes were never affected, so they are not
in scope -- verified by inventory, not assumed.

WHAT IT DOES NOT DO. Nothing is overwritten. Corrected chunks are written to
`<dir>/f047_rescored/<same filename>`, so the originals remain as the pre-fix
record and any consumer opts in by pointing `search_output_dir` at the new
directory. Resumable: a chunk whose output exists is skipped.

RUN (venv + PYTHONPATH per docs; ~82k crystals total, tens of minutes):
    python -m data_processing.rescore_uma_f047
    python -m data_processing.rescore_uma_f047 --dry-run
"""
import argparse
import glob
import json
import os
import time

import torch

from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

#: Bumped when a change makes stored energies from an earlier state incomparable.
#: 2 = the F-047 external-graph fix. Written into every output chunk so a consumer
#: can refuse energies it did not compute (a file with no stamp is state 1).
UMA_ENERGY_STATE = 2

UMA_MODEL = r'D:/crystal_datasets/esen_s.pt'
OUT_SUBDIR = 'f047_rescored'

#: (run_name, search_output_dir). Only groups whose chunks carry a `uma` attribute.
GROUPS = [
    ('nehzor_4_uma', r'D:/crystal_datasets/nehzor/p6'),
    ('mipcas_uma', r'D:/crystal_datasets/mipcas'),
]

#: Start small and let adaptive_batched_analysis grow it. The MLIP allocates in
#: proportion to supercell atoms, not rows, so a large opening batch is how this
#: job would take the box down rather than report an OOM.
INITIAL_BATCH = 50


def chunk_files(run_name, directory):
    """The numbered chunks for one run, in numeric order. `<name>_<int>.pt` only --
    the same-prefix `_prior_dataset.pt` and `_traj*.pt` files are not chunks."""
    out = []
    for path in glob.glob(os.path.join(directory, f'{run_name}_*.pt')):
        tail = os.path.basename(path)[len(run_name) + 1:-3]
        if tail.isdigit():
            out.append((int(tail), path))
    return [p for _, p in sorted(out)]


def preflight():
    """Refuse to start if anything else holds the card. This script loads an MLIP
    and runs large batches; a second CUDA consumer BSODs this box."""
    try:
        import gpu_guard
    except ImportError:
        return                      # not on the GFN path; the memory cap still applies
    others = gpu_guard.training_processes()
    if others:
        raise SystemExit(f'another process holds the GPU: {others[0][1][:80]}')


def rescore_group(run_name, directory, predictor, dry_run=False):
    files = chunk_files(run_name, directory)
    out_dir = os.path.join(directory, OUT_SUBDIR)
    if not files:
        print(f'{run_name}: no chunks found in {directory}')
        return []
    print(f'\n{run_name}: {len(files)} chunks in {directory}')
    if dry_run:
        return []
    os.makedirs(out_dir, exist_ok=True)

    state = {'batch_size': INITIAL_BATCH}
    deltas = []
    for i, path in enumerate(files):
        out_path = os.path.join(out_dir, os.path.basename(path))
        if os.path.exists(out_path):
            print(f'  [{i + 1}/{len(files)}] {os.path.basename(path)} -- done, skipping')
            continue
        t0 = time.perf_counter()
        samples = torch.load(path, weights_only=False)
        batch = collate_data_list(samples) if isinstance(samples, list) else samples
        before = batch.uma.clone().float().cpu()

        batch = batch.to('cuda')
        # NO_GRAD, EXPLICITLY -- the same trap train.py's init pass documents.
        # fairchem's _run_inference uses nullcontext (not no_grad) whenever
        # direct_forces is False, which this predictor sets, so the scored energies
        # come back carrying grad_fn and pin their activations. Measured on this
        # chunk: ~100-250 MB retained PER CRYSTAL without it, zero with it -- which
        # is a cascading OOM a few hundred rows in, at batch size 1, on a scan that
        # differentiates nothing.
        with torch.no_grad():
            batch = adaptive_batched_analysis(
                batch, analyses=['uma'], state=state,
                initial_batch_size=state['batch_size'],
                predictor=predictor, device='cuda', show_tqdm=False)
        batch = batch.to('cpu')
        after = batch.uma.float().cpu()

        d = (before - after)                       # old minus corrected
        deltas.append(d)
        torch.save({'samples': batch.batch_to_list(),
                    'uma_energy_state': UMA_ENERGY_STATE,
                    'uma_graph': 'external',
                    'source_chunk': os.path.basename(path)}, out_path)
        del batch
        torch.cuda.empty_cache()
        print(f'  [{i + 1}/{len(files)}] {os.path.basename(path)}: {len(after)} rows, '
              f'median d {d.median():+.4f} kJ/mol, max |d| {d.abs().max():.4f} kJ/mol, '
              f'{time.perf_counter() - t0:.1f}s', flush=True)
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='list the work without touching the GPU')
    ap.add_argument('--memory-fraction', type=float, default=0.70)
    args = ap.parse_args()

    if args.dry_run:
        for run_name, directory in GROUPS:
            rescore_group(run_name, directory, None, dry_run=True)
        return

    preflight()
    # a hard cap so exhaustion arrives as a catchable OOM the adaptive batcher can
    # shrink against, instead of as a driver-level failure
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    predictor = init_uma_crystal_predictor(UMA_MODEL, device='cuda')
    from mxtaltools.mlip_interfaces.uma_utils import _predictor_wants_external_graph
    if not _predictor_wants_external_graph(predictor):
        raise SystemExit('predictor is NOT using the external graph -- this job would '
                         'reproduce the very energies it exists to replace')

    summary = {}
    for run_name, directory in GROUPS:
        deltas = rescore_group(run_name, directory, predictor)
        if deltas:
            d = torch.cat(deltas)
            summary[run_name] = {
                'rows': int(d.numel()),
                # `uma` is compute_lattice_uma: LATTICE ENERGY PER MOLECULE IN
                # kJ/mol, not a raw eV potential. kT = 2.5 kJ/mol, so the kT
                # column is what decides whether a delta matters thermally.
                'median_delta_kJmol': float(d.median()),
                'mean_abs_delta_kJmol': float(d.abs().mean()),
                'mean_abs_delta_kT': float(d.abs().mean()) / 2.5,
                'max_abs_delta_kJmol': float(d.abs().max()),
                'frac_above_1kT': float((d.abs() > 2.5).float().mean()),
                'frac_above_0p1kJmol': float((d.abs() > 0.1).float().mean()),
            }
    print('\n== summary (old minus corrected, kJ/mol lattice energy per molecule) ==')
    print(json.dumps(summary, indent=2))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results', 'f047_rescore_summary.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'uma_energy_state': UMA_ENERGY_STATE, 'groups': summary}, f, indent=2)
    print(f'written: {out}')


if __name__ == '__main__':
    main()
