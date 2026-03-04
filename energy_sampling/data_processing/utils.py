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


def calibrate_energy_function_vs_uma(search_output_dir,
                                     energy_function,
                                     run_name,
                                     test_energy):
    pattern = os.path.join(search_output_dir, run_name.replace(energy_function, 'uma') + '_*.pt')
    files = glob.glob(pattern)

    traj_records = [
        f for f in files
        if
        re.search(rf'{re.escape(run_name.replace(energy_function, "uma"))}_\d{{1,2}}\.pt$', os.path.basename(f))
    ]
    uma_opt_samples = torch.load(traj_records[0], weights_only=False)
    uma_batch = collate_data_list(uma_opt_samples)
    uma_en = uma_batch.uma
    en_scaling_factor = uma_en[uma_en < uma_en.quantile(0.1)].mean() / test_energy[
        test_energy < test_energy.quantile(0.1)].mean()
    return en_scaling_factor.item()
