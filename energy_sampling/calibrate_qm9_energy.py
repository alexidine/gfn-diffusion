"""Calibrate a condition/prior file's ELJ energies against UMA (eSEN-small), producing the
``thermal_scaling_factor`` that ``train.py`` uses in place of ``lj_coeff``.

    # report only
    python calibrate_qm9_energy.py --prior D:\crystal_datasets\conditional\priors\qm9_v8_prior.pt

    # report and write the factor into the file
    python calibrate_qm9_energy.py --prior ...\qm9_v8_prior.pt --write

WHAT THE FACTOR MEANS
---------------------
``train.py:init_prior_dataset`` reads ``thermal_scaling_factor`` off the prior file and
assigns it to ``energy_function.lj_coeff`` for the whole run, so every energy becomes
``factor * elj``. Choosing ``factor = uma / elj`` puts ELJ energies on UMA's kJ/mol scale,
after which ``energy_config.temperature`` means what it says (room temperature ~2.5) instead
of being a raw-ELJ number. mipcas's factor is 0.3636 -- ELJ overestimates by ~2.75x there --
which is why its nominal kT 2.5 reads as ~6.9 in raw ELJ units.

So calibrating is also a TEMPERATURE change: after writing a factor, drop
``energy_config.temperature`` from the hot raw-ELJ value back to the physical one, or the
run gets hotter by the same ratio twice over.

WHY THE LOW-ENERGY TAIL, NOT THE MEAN
-------------------------------------
The factor is the ratio of the two potentials' bottom-decile means, matching
``data_processing/utils.calibrate_energy_function_vs_uma``. LJ tails are heavy and
one-sided: a batch mean is dominated by the worst-packed structures, which are exactly the
ones the sampler should never visit, so a mean-matched factor is set by the garbage rather
than by the basins. The bottom decile matches the two potentials where the density
actually lives.

Unlike that helper, this compares the SAME structures under both potentials -- it analyses
one batch twice rather than pairing separate ELJ-search and UMA-search sample sets. That
makes it a paired comparison, so structure-set differences cannot leak into the ratio.

PER-MOLECULE SPREAD IS THE THING TO READ
----------------------------------------
``lj_coeff`` is ONE number for the whole run, but on a conditional molecule route it is
applied across every molecule in the condition set. If the per-molecule factors disagree
badly, a single global scaling is a compromise that silently runs some molecules hotter
than others -- which on a conditional run is a per-condition temperature gradient, not just
a units question. The spread is reported for that reason; sampling is stratified so no
molecule dominates it.
"""

import argparse
import collections
from pathlib import Path

import torch

from tqdm import tqdm

from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

DEFAULT_MLIP = r'D:\crystal_datasets\esen_s.pt'


def load_structures(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(data, dict):
        return data, None
    for key in ('equalized_prior', 'prior'):
        if key in data:
            return data[key], data
    raise ValueError(f"{path} has no 'equalized_prior' or 'prior' entry")


def stratified_sample(batch, n_total, seed):
    """Equal structures per molecule, so one replica-rich molecule cannot set the factor."""
    items = batch.batch_to_list()
    by_id = collections.defaultdict(list)
    for s in items:
        by_id[s.identifier].append(s)
    per = max(1, n_total // len(by_id))
    generator = torch.Generator().manual_seed(seed)
    out = []
    for ident in sorted(by_id):
        group = by_id[ident]
        take = min(per, len(group))
        idx = torch.randperm(len(group), generator=generator)[:take].tolist()
        out.extend(group[i] for i in idx)
    return out, len(by_id), per


def tail_ratio(uma, elj, quantile):
    """Ratio of bottom-quantile means. Returns None if either side has no qualifying
    samples or an ELJ tail that straddles zero (the ratio would be meaningless)."""
    if uma.numel() == 0:
        return None
    u = uma[uma < uma.quantile(quantile)]
    e = elj[elj < elj.quantile(quantile)]
    if u.numel() == 0 or e.numel() == 0:
        return None
    denominator = e.mean()
    if denominator.abs() < 1e-6:
        return None
    return float(u.mean() / denominator)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prior', required=True, type=Path,
                   help='a *_prior.pt or *_conditions.pt built by build_qm9_conditions.py')
    p.add_argument('--mlip', type=Path, default=Path(DEFAULT_MLIP),
                   help='UMA checkpoint; esen_s is the cheap one')
    p.add_argument('--n-samples', type=int, default=2000,
                   help='structures to analyse, spread evenly across molecules')
    p.add_argument('--quantile', type=float, default=0.1,
                   help='low-energy tail the factor is matched on')
    p.add_argument('--batch-size', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--write', action='store_true',
                   help='write thermal_scaling_factor into the file (in place)')
    args = p.parse_args()

    print(f"loading {args.prior}")
    batch, container = load_structures(args.prior)
    print(f"  {batch.num_graphs} structures, {len(set(batch.identifier))} molecules")

    samples, n_mols, per_mol = stratified_sample(batch, args.n_samples, args.seed)
    print(f"  sampling {len(samples)} structures ({per_mol} per molecule x {n_mols})")

    print(f"loading UMA predictor {args.mlip}")
    predictor = init_uma_crystal_predictor(str(args.mlip), device=args.device)

    print("analysing under both potentials (same structures, paired)")
    elj_parts, uma_parts, identifiers = [], [], []
    for start in tqdm(range(0, len(samples), args.batch_size), desc='  analysing'):
        chunk = collate_data_list(samples[start:start + args.batch_size]).to(args.device)
        # every sub-batch rebuilds its own fractional transform: box_analysis' output does
        # not survive the batch_to_list/collate round trip, which is also why
        # adaptive_batched_analysis cannot be used here
        chunk.box_analysis()
        # std_orientation=False matches energies/molecular_crystal.py:205, i.e. the frame
        # convention training actually scores in
        chunk.analyze(['elj', 'uma'], assign_outputs=True, cutoff=10, supercell_size=10,
                      std_orientation=False, predictor=predictor)
        chunk = chunk.cpu()
        # generator_energy divides the analytic potentials by z_prime but not the MLIP
        # ones; mirror that so the ratio is between the quantities training compares
        elj_parts.append(chunk.elj.flatten().float() / chunk.z_prime.flatten().float())
        uma_parts.append(chunk.uma.flatten().float())
        identifiers.extend(chunk.identifier)
        del chunk

    elj = torch.cat(elj_parts)
    uma = torch.cat(uma_parts)

    finite = torch.isfinite(elj) & torch.isfinite(uma)
    if int((~finite).sum()):
        print(f"  dropping {int((~finite).sum())} non-finite energies")
    elj, uma = elj[finite], uma[finite]
    identifiers = [i for i, keep in zip(identifiers, finite.tolist()) if keep]

    factor = tail_ratio(uma, elj, args.quantile)
    if factor is None:
        raise RuntimeError("could not form a tail ratio -- the ELJ bottom decile straddles "
                           "zero or no samples qualified; inspect the energies by hand")

    def q(t):
        return [round(float(t.quantile(v)), 2) for v in (0.0, 0.1, 0.5, 0.9)]

    print()
    print(f"  elj (per Z')  q(0,10,50,90) = {q(elj)}")
    print(f"  uma           q(0,10,50,90) = {q(uma)}")
    print()
    print(f"  thermal_scaling_factor = {factor:.4f}   (uma / elj on the bottom "
          f"{args.quantile:.0%})")
    print(f"  => elj overestimates depth by {1 / factor:.2f}x; at factor {factor:.4f} a "
          f"physical kT 2.5 corresponds to {2.5 / factor:.2f} in raw elj units")

    per_molecule = {}
    for ident in sorted(set(identifiers)):
        mask = torch.tensor([i == ident for i in identifiers])
        f = tail_ratio(uma[mask], elj[mask], args.quantile)
        if f is not None:
            per_molecule[ident] = f
    if len(per_molecule) > 1:
        values = torch.tensor(list(per_molecule.values()))
        lo, hi = min(per_molecule.items(), key=lambda kv: kv[1]), max(per_molecule.items(), key=lambda kv: kv[1])
        print()
        print(f"  per-molecule factors: median {values.median():.4f}, "
              f"range {values.min():.4f}-{values.max():.4f} "
              f"({values.max() / values.min():.2f}x spread over {len(per_molecule)} molecules)")
        print(f"    lowest  {lo[1]:.4f}  {lo[0]}")
        print(f"    highest {hi[1]:.4f}  {hi[0]}")
        print("  ONE global lj_coeff covers all of them, so this spread is a per-condition "
              "temperature gradient: molecules above the global factor train colder than "
              "those below it.")

    if args.write:
        if container is None:
            raise RuntimeError(f"{args.prior} is a bare batch, not a dict -- nothing to "
                               f"write thermal_scaling_factor into")
        container['thermal_scaling_factor'] = float(factor)
        container['thermal_scaling_provenance'] = {
            'mlip': str(args.mlip), 'n_samples': int(len(elj)),
            'quantile': float(args.quantile), 'seed': int(args.seed)}
        torch.save(container, args.prior)
        print()
        print(f"wrote thermal_scaling_factor={factor:.4f} into {args.prior}")
        print(f"NOW LOWER THE TEMPERATURE: energy_config.temperature should go from the hot "
              f"raw-elj value back to the physical one (~2.5). train.py applies the factor "
              f"to lj_coeff, so leaving temperature hot scales the run twice.")
    else:
        print()
        print("  (report only -- pass --write to store this in the file)")


if __name__ == '__main__':
    main()
