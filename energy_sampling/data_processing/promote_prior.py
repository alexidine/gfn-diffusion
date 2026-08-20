"""
Promote a collate_prior output to the file layout train.py actually loads.

THE TWO LAYOUTS. collate_prior writes {prior_batch, noised_batch,
thermal_scaling_factor, log_noise_range} beside the search chunks. train.py reads a
different shape: `prior_path` loads data['equalized_prior'] and `molecules_path`
loads data['prior'] (see configs/deadrow_aug12/prep_priors.py, which documents the
contract). This is the rename between them:

    prior_batch   -> prior             (the thinned anchors)
    noised_batch  -> equalized_prior   (the noised expansion the sampler draws from)

VERIFIED, NOT ASSUMED: on the pre-F-047 files the counts line up exactly --
nehzor prior=5726 / equalized=200368 against its own prior_batch/noised_batch, and
mipcas 9237 / 176350.

`thermal_scaling_factor` is carried through because train.py OVERWRITES the config's
lj_coeff with it for the whole run. `uma_energy_state` is carried so a consumer can
tell which energy state selected these structures; the thinning cut is what the stamp
is really about.

Nothing is overwritten: the output name is explicit and the script refuses to clobber.
"""
import argparse
import os

import torch

from energy_sampling.data_processing.utils import UMA_ENERGY_STATE

PRIORS_DIR = r'D:\crystal_datasets\conditional\priors'


def promote(src, out_name, energy_function='uma', priors_dir=PRIORS_DIR):
    d = torch.load(src, weights_only=False)
    for key in ('prior_batch', 'noised_batch'):
        if key not in d:
            raise KeyError(f'{src} has no {key!r} -- is it a collate_prior output? '
                           f'keys: {sorted(d)}')
    state = d.get('uma_energy_state')
    if energy_function == 'uma' and state != UMA_ENERGY_STATE:
        raise RuntimeError(
            f'{src} carries uma_energy_state={state!r}, current is {UMA_ENERGY_STATE} '
            f'(F-047). Promoting it would put pre-fix structure SELECTION into a file '
            f'train.py loads. Rebuild with collate_prior first.')

    out = {
        'prior': d['prior_batch'],
        'equalized_prior': d['noised_batch'],
        'thermal_scaling_factor': d['thermal_scaling_factor'],
        'uma_energy_state': state,
    }
    dest = os.path.join(priors_dir, out_name)
    if os.path.exists(dest):
        raise FileExistsError(f'{dest} exists -- choose another name rather than '
                              f'overwriting a prior a config may already point at')
    torch.save(out, dest)
    print(f'{os.path.basename(src)} -> {dest}\n'
          f'  prior {out["prior"].num_graphs} graphs, '
          f'equalized_prior {out["equalized_prior"].num_graphs} graphs, '
          f'tsf {out["thermal_scaling_factor"]}, uma_energy_state {state}')
    return dest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out-name', required=True)
    ap.add_argument('--energy-function', default='uma')
    a = ap.parse_args()
    promote(a.src, a.out_name, a.energy_function)
