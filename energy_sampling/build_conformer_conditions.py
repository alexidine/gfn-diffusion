"""Write the conformer condition file, and optionally the graph-form prior beside it.

Two artifacts, the same split train.py's data layer already makes (see
``energies/conformer_data.py`` for the formats and why):

  ``--out``        the condition set -> ``molecules_path``. One graph per molecule,
                   carrying the internal-coordinate tree, the reference conformer and the
                   rotatable-torsion selection. No state.
  ``--prior-out``  the prior -> ``prior_path``. Every molecule replicated once per drawn
                   state, with the state and its raw energy baked in, under
                   ``equalized_prior``.

The prior draw reuses ``build_prior_states``' handcrafted ``InternalPrior`` path when a
fitted prior is available, and falls back to uniform-on-the-torus otherwise -- loudly,
because a silent fallback is how a run ends up training against a prior nobody chose.
Uniform is the correct dumb prior for a torsion (maximum entropy on the space), it is just
a much weaker one.

    # one molecule, conditions + prior, checked against the energy
    python build_conformer_conditions.py --smiles CCCCO --prior-out conformer_prior_graphs.pt

    # a small condition set, prior drawn from a fitted InternalPrior
    python build_conformer_conditions.py --smiles CCCCO CCCCCO CCCC=O \\
        --out conformer_conditions.pt --prior-out conformer_prior_graphs.pt \\
        --internal-prior conformer_prior.pt --n-prior 4000

Every molecule in one file must have the same number of rotatable torsions -- one file is
one k, because the GFN's state dimension is fixed at construction. ``collate_conditions``
says so by name if they don't.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from energies.conformer_data import (attach_states, bake_energies, check_state_convention,
                                     collate_conditions, condition_from_energy,
                                     save_condition_file, save_prior_file)
from energies.conformer_torsions import ConformerTorsions


def draw_prior_states(energy, n: int, internal_prior: Path, fatten: float, seed: int):
    """``[n, k]`` states from the fitted InternalPrior, or uniform if there isn't one."""
    rng = np.random.default_rng(seed)
    if internal_prior is not None and Path(internal_prior).exists():
        from build_prior_states import draw_states, fit_or_load

        prior = fit_or_load(Path(internal_prior), [], fatten)
        states, n_uniform = draw_states(energy, prior, n, rng)
        if n_uniform:
            print(f"  {energy.smiles}: {n_uniform}/{energy.data_ndim} dimensions had no "
                  f"fitted torsion type and fell through to uniform")
        return states.to(energy.dtype)

    print(f"  {energy.smiles}: no fitted InternalPrior at {internal_prior}; drawing "
          f"UNIFORM on the torus ({n} states). This is the max-entropy dumb prior, not a "
          f"failure -- but it is much weaker than the fitted one")
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (n, energy.data_ndim)), dtype=energy.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", nargs="+", default=["CCCCO"])
    ap.add_argument("--identifiers", nargs="*", default=None,
                    help="one per SMILES; defaults to the SMILES themselves. train.py "
                         "resolves condition identity through this string alone")
    ap.add_argument("--out", type=Path, default=Path("conformer_conditions.pt"))
    ap.add_argument("--prior-out", type=Path, default=None)
    ap.add_argument("--n-prior", type=int, default=4000,
                    help="states per molecule in the prior file")
    ap.add_argument("--internal-prior", type=Path, default=None,
                    help="a fitted InternalPrior .pt (see build_prior_states.py)")
    ap.add_argument("--fatten", type=float, default=0.15)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--min-separation", type=int, default=3)
    ap.add_argument("--scale-14", type=float, default=0.5)
    ap.add_argument("--lj-k-factor", type=float, default=2.5)
    ap.add_argument("--include-trivial-rotations", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-check", action="store_true",
                    help="skip the graph-vs-energy geometry check (don't)")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    identifiers = args.identifiers or args.smiles
    if len(identifiers) != len(args.smiles):
        raise SystemExit(f"{len(args.smiles)} SMILES against {len(identifiers)} identifiers")

    ff = dict(epsilon=args.epsilon, min_separation=args.min_separation,
              scale_14=args.scale_14, lj_k_factor=args.lj_k_factor,
              include_trivial_rotations=args.include_trivial_rotations, seed=args.seed)

    conditions, energies = [], []
    for smiles, ident in zip(args.smiles, identifiers):
        # see build_conformer_buffer.py: `torsion` is explicit, not a default. The
        # condition/prior file format stores per-graph `torsion_state` and n_torsions,
        # both of which mean something else at a wider level.
        energy = ConformerTorsions(smiles=smiles, device="cpu", level="torsion", **ff)
        print(energy.describe())
        mol = condition_from_energy(energy, identifier=ident)
        if not args.no_check:
            err = check_state_convention(mol, energy)
            print(f"   state convention: graph and energy agree to {err:.2e} A")
        conditions.append(mol)
        energies.append(energy)

    batch = collate_conditions(conditions)
    save_condition_file(batch, args.out)
    print(f"\nwrote conditions -> {args.out}  ({batch.num_graphs} graphs, "
          f"k = {int(batch.n_torsions[0])})")

    if args.prior_out is None:
        return

    print(f"\nprior: {args.n_prior} states per molecule")
    parts = []
    for mol, energy, ident in zip(conditions, energies, identifiers):
        states = draw_prior_states(energy, args.n_prior, args.internal_prior,
                                  args.fatten, args.seed)
        # RAW energy, T = 1: prebuilt_sample_to_reward divides by the sampling
        # temperature itself (see conformer_data.bake_energies)
        e = bake_energies(energy, states)
        print(f"  {ident}: E median {e.median():+8.3f}  p10 {torch.quantile(e, 0.1):+8.3f}"
              f"  p90 {torch.quantile(e, 0.9):+8.3f}")
        # the mask is NOT optional: at `flex` and above the state carries linear r/theta
        # columns, and wrapping one folds a bond length to the opposite corner of the box
        parts.append(attach_states(mol, states, e, identifier=ident,
                                   periodic=energy.periodic_dims))

    prior = parts[0]
    for part in parts[1:]:
        prior = prior.append_batch(part)
    save_prior_file(prior, args.prior_out,
                    source="InternalPrior" if args.internal_prior else "uniform",
                    n_per_molecule=args.n_prior)
    print(f"\nwrote prior -> {args.prior_out}  ({prior.num_graphs} rows)")


if __name__ == "__main__":
    main()
