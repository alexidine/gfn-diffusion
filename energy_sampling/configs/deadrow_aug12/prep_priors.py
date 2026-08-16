"""
deadrow_aug12 -- prior prep. Run ONCE before the battery.

Builds small ELJ-target priors for the non-sg-2 arms:

  * SUBSAMPLE to 10k. The stock priors carry 50k-207k equalised rows and
    init_prior_dataset re-scores every one of them at startup, so a full-size
    prior costs minutes per arm before step 0. This battery is a wiring check,
    not a convergence run -- 10k is ample.

  * DROP `thermal_scaling_factor`. This is the load-bearing one. train.py:1344
    reads that key and OVERWRITES the config's `lj_coeff` for the whole run
    (a silent unit change -- see project_lj_coeff_silent_override). The acridine
    priors were calibrated against MACE, so carrying it into an ELJ arm would
    apply a MACE-derived scale to ELJ energies and confound every cross-arm
    comparison. With the key absent, the config's own lj_coeff is used.
    ELJ re-scoring itself needs no work: init_prior_dataset re-analyses the
    prior with the CONFIGURED energy unconditionally (train.py:1334, `if True`).

  * KEEP BOTH keys. `prior_path` loads data['equalized_prior'];
    `molecules_path` loads data['prior'] (train.py:1430-1433). A file missing
    either breaks one of the two loaders, not both, so the failure is partial.

Outputs land next to the originals with a `deadrow10k_` prefix. Nothing is
overwritten, and the source files are opened read-only.

    python configs/deadrow_aug12/prep_priors.py
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_es = os.path.dirname(os.path.dirname(_here))                 # energy_sampling
for _root in (os.path.dirname(_es),                           # gfn_diffusion
              os.path.join(os.path.dirname(os.path.dirname(_es)), 'mxtaltools')):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import torch

PRIOR_DIR = r'D:\crystal_datasets\conditional\priors'
N_KEEP = 10_000
SEED = 0

# source -> output basename. Only the arms that need a re-scored / shrunk prior.
SOURCES = {
    'nehzor_sg14_zp1_elj_prior_dataset': 'deadrow10k_sg14_zp1_elj',
    'acridine_sg9_zp2_mace_prior_dataset': 'deadrow10k_sg9_zp2_elj',
    'acridine_sg14_zp2_mace_prior_dataset': 'deadrow10k_sg14_zp2_elj',
}


def take(batch, n, seed):
    """Random n-row subsample of a collated batch, or the batch itself if smaller."""
    total = int(batch.num_graphs)
    if total <= n:
        return batch, total
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n].sort().values
    return batch.subsample_new_batch(idx.to(batch.sg_ind.device)), n


def main():
    made, skipped = [], []
    for src, dst in SOURCES.items():
        src_path = os.path.join(PRIOR_DIR, src + '.pt')
        dst_path = os.path.join(PRIOR_DIR, dst + '.pt')
        if not os.path.exists(src_path):
            print(f"SKIP {src}: not found")
            skipped.append(src)
            continue
        if os.path.exists(dst_path):
            print(f"SKIP {dst}: already exists (delete to rebuild)")
            continue

        data = torch.load(src_path, weights_only=False)
        if not isinstance(data, dict) or 'equalized_prior' not in data:
            print(f"SKIP {src}: unexpected structure {type(data).__name__}")
            skipped.append(src)
            continue

        out = {}
        for key in ('prior', 'equalized_prior'):
            if key not in data:
                continue
            sub, n = take(data[key], N_KEEP, SEED)
            out[key] = sub
            print(f"  {src}[{key}]: {int(data[key].num_graphs)} -> {n}")

        dropped = [k for k in data if k not in out]
        sg = int(out['equalized_prior'].sg_ind[0])
        zp = int(out['equalized_prior'].z_prime[0])
        torch.save(out, dst_path)
        print(f"  wrote {dst}.pt  sg={sg} Z'={zp}  dropped keys: {dropped or 'none'}")
        made.append(dst)

    print()
    print(f"{len(made)} prior(s) written, {len(skipped)} skipped.")
    if 'thermal_scaling_factor' not in ''.join(str(s) for s in made):
        print("thermal_scaling_factor is intentionally absent -- each arm's own "
              "lj_coeff governs, so ELJ arms are not scaled by a MACE calibration.")
    if skipped:
        print(f"WARNING: {skipped} unavailable -- the arms depending on them cannot run. "
              f"See INDEX.tsv for which.")


if __name__ == '__main__':
    main()
