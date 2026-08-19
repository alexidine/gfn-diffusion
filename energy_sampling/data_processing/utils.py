import glob
import os
import re

import torch

from mxtaltools.dataset_utils.utils import collate_data_list


def param_dist(ref, sample, scale):
    """
    :param ref: [n, k]
    :param sample: [k]
    :param scale: [k]
    :return: [k]
    """
    return (ref - sample[None, :]).abs() / scale


def thin_points_indices(traj, cutoff, eps=1e-8):
    """
    traj: [n, l, k]

    Returns:
        indices: list of 1D LongTensors, one per trajectory
    """

    L, N, D = traj.shape

    # per-sample, per-dim scale
    ptp = traj.quantile(0.95, dim=0) - traj.quantile(0.05, dim=0)  # [N, D]
    scale = ptp.clamp_min(eps)  # avoid div-by-zero

    all_indices = []

    for i in range(N):
        t = traj[:, i]  # [L, D]
        s = scale[i]  # [D]

        keep = [L - 1]
        last = t[L - 1]

        for j in range(L - 2, -1, -1):
            diff = (t[j] - last).abs() / s  # per-dim scaled diff

            # fire if any dimension exceeds cutoff
            if (diff >= cutoff).any():
                keep.append(j)
                last = t[j]

        keep.reverse()  # restore chronological order
        all_indices.append(torch.tensor(keep, device=traj.device))

    return all_indices


#: Current UMA energy state. 2 = the F-047 external-graph fix (docs/findings.md):
#: before it, fairchem built the neighbour graph itself on the assumption that atoms
#: sit inside the cell, and silently dropped contacts on our unwrapped unit cells.
#: Measured over 122k search rows, ~7% of them moved by more than a full kT.
UMA_ENERGY_STATE = 2

#: Where rescore_uma_f047.py writes corrected chunks, beside the originals.
RESCORED_SUBDIR = 'f047_rescored'


def load_search_chunks(search_output_dir, run_name, require_uma_state=False):
    """
    Every numbered search chunk for one run, as a flat list of crystals.

    PREFERS CORRECTED CHUNKS. If `<dir>/f047_rescored/` holds a chunk of the same
    name it is loaded instead of the original, so a rebuild picks up the F-047 fix
    without every call site being repointed by hand.

    REFUSES STALE UMA ENERGIES. With require_uma_state, a chunk whose stamp is
    missing or older than UMA_ENERGY_STATE raises rather than loading: an unstamped
    file is pre-fix by definition, and its `uma` values disagree with the current
    energy function by ~1 kT on average. Absence has to fail here, not abstain --
    every contaminated file on disk is one with no stamp at all.

    Handles both on-disk shapes: a plain list (original chunks) and the stamped
    dict rescore_uma_f047.py writes.
    """
    originals = {}
    for path in glob.glob(os.path.join(search_output_dir, f'{run_name}_*.pt')):
        tail = os.path.basename(path)[len(run_name) + 1:-3]
        if tail.isdigit():
            originals[int(tail)] = path

    samples, used_rescored, stale = [], 0, []
    for key in sorted(originals):
        path = originals[key]
        rescored = os.path.join(search_output_dir, RESCORED_SUBDIR,
                                os.path.basename(path))
        if os.path.exists(rescored):
            path, is_rescored = rescored, True
        else:
            is_rescored = False
        obj = torch.load(path, weights_only=False)
        if isinstance(obj, dict):
            state = obj.get('uma_energy_state', 1)
            if require_uma_state and state < UMA_ENERGY_STATE:
                stale.append((os.path.basename(path), state))
            samples.extend(obj['samples'])
        else:
            if require_uma_state:
                stale.append((os.path.basename(path), 1))
            samples.extend(obj)
        used_rescored += int(is_rescored)

    if stale:
        raise RuntimeError(
            f'{len(stale)} of {len(originals)} chunks for {run_name!r} carry UMA '
            f'energies from state {stale[0][1]}, but the current state is '
            f'{UMA_ENERGY_STATE} (F-047). Those energies are wrong by ~1 kT on '
            f'average and up to hundreds of kJ/mol. Re-score them first:\n'
            f'    python -m data_processing.rescore_uma_f047\n'
            f'  e.g. {stale[0][0]}')
    print(f'  {run_name}: {len(samples)} rows from {len(originals)} chunks '
          f'({used_rescored} F-047-corrected)')
    return samples


def calibrate_energy_function_vs_uma(search_output_dir,
                                     energy_function,
                                     run_name,
                                     test_energy,
                                     quantile: float = 0.1):
    """
    The factor putting `energy_function` into UMA-like units, from the two runs'
    own low-energy tails.

    WHY A TAIL RATIO AND NOT A PAIRED FIT. The two functions do not share a global
    minimum -- the MIPCAS ELJ minimum and the MIPCAS UMA minimum are different
    structures, each essentially zero-probability under the other's distribution.
    So there is no shared set to regress on: the object being matched is the DEPTH
    OF EACH FUNCTION'S OWN low-energy region, which is a comparison of two
    distributions, not of two columns.

    OVER EVERY CHUNK, not the first one. It used to read `traj_records[0]`, which on
    nehzor is 130 rows -- a bottom decile of THIRTEEN structures setting the
    effective temperature for a whole training run.

    The spread across quantile cuts is printed rather than returned: a single factor
    assumes the two tails have the same shape, and cuts that disagree are the signal
    that they do not. It costs nothing to look.
    """
    uma_run = run_name.replace(energy_function, 'uma')
    # UMA energies feed the factor directly, so a pre-F-047 chunk here silently
    # rescales every energy in the run that consumes it
    uma_samples = load_search_chunks(search_output_dir, uma_run, require_uma_state=True)
    uma_en = collate_data_list(uma_samples).uma

    def tail_ratio(q):
        return (uma_en[uma_en < uma_en.quantile(q)].mean()
                / test_energy[test_energy < test_energy.quantile(q)].mean()).item()

    factor = tail_ratio(quantile)
    spread = {q: tail_ratio(q) for q in (0.01, 0.05, 0.1, 0.2)}
    print(f'  scaling factor {factor:.4f} at q={quantile}; across cuts: '
          + ', '.join(f'q{q}={v:.4f}' for q, v in spread.items()))
    return factor
