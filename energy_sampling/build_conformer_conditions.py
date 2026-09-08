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
from energies.dof_features import free_dof_atom_index
from models import encoder_cache


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
    ap.add_argument("--k", type=int, default=None,
                    help="keep only molecules with this state dimension. Default: take k from "
                         "the first molecule that builds. A conditions file is ONE k, and k is "
                         "not a stable property of the SMILES -- see known gap 2")
    ap.add_argument("--encoder-ckpt", type=Path, default=None,
                    help="bake a frozen molecular embedding onto every entry, so the policy "
                         "can be conditioned on molecular identity. Writes per-graph "
                         "`embedding` (pooled, 2*hidden) and per-atom `atom_embedding` "
                         "(hidden, in TREE order). Enable with model.embedding_conditioning "
                         "and set embedding_conditioning_dim to the printed width")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)

    identifiers = args.identifiers or args.smiles
    if len(identifiers) != len(args.smiles):
        raise SystemExit(f"{len(args.smiles)} SMILES against {len(identifiers)} identifiers")

    ff = dict(epsilon=args.epsilon, min_separation=args.min_separation,
              scale_14=args.scale_14, lj_k_factor=args.lj_k_factor,
              include_trivial_rotations=args.include_trivial_rotations, seed=args.seed)

    bundle = None
    if args.encoder_ckpt is not None:
        bundle = encoder_cache.load_encoder(str(args.encoder_ckpt), device="cpu")
        print(f"encoder {bundle['arm']} @ {bundle['sha256'][:12]}  "
              f"hidden {bundle['hidden']}  ->  embedding_conditioning_dim: "
              f"{2 * bundle['hidden']}")

    conditions, energies, dof_rows, skipped = [], [], [], []
    want_k = int(args.k) if args.k else None
    for smiles, ident in zip(args.smiles, identifiers):
        # see build_conformer_buffer.py: `torsion` is explicit, not a default. The
        # condition/prior file format stores per-graph `torsion_state` and n_torsions,
        # both of which mean something else at a wider level.
        try:
            energy = ConformerTorsions(smiles=smiles, device="cpu", level="torsion", **ff)
        except Exception as exc:                              # noqa: BLE001 - reported below
            skipped.append((smiles, f"{type(exc).__name__}: {exc}"))
            continue
        # K IS DECIDED BY THE FIRST MOLECULE (or --k) AND THE REST MUST MATCH.
        # `collate_conditions` refuses a mixed-k file, but it refuses at the END, after every
        # molecule has been built and priors possibly drawn -- and `check_state_convention`
        # gets there first with a bare IndexError naming neither the molecule nor k. Worse,
        # k is NOT a stable property of the SMILES: section 6's second known gap is that the
        # chart is measured off a reference conformer, so the same molecule can present a
        # different d from a different embedding. Filtering here means a molecule set can be
        # handed in whole and the file comes out coherent, with the rejects named.
        # THE CHART AND THE COLLECTIVE MAP MUST AGREE ON HOW MANY COLUMNS THERE ARE.
        # `energy.mask` carries one column per ROTATABLE AXIS while `_M` and `data_ndim`
        # carry one per SURVIVING axis, and they disagree whenever an axis is dropped as
        # degenerate. `_state_columns` reads `mask`, so it then emits a column index the
        # state cannot hold: `condition_from_energy` succeeds and `check_state_convention`
        # dies deep in `state_to_phi` with a bare IndexError naming neither the molecule nor
        # the mismatch -- and had the surplus column not been LAST, rows would have been
        # attributed to the wrong state dimension with no error at all.
        #
        # Measured on 400 QM9 molecules: 11 (2.8%) disagree, and EVERY ONE CONTAINS AN
        # ALKYNE -- rotation about a bond adjacent to a linear C#C. This also violates the
        # invariant tests/conformer/test_conformer_levels.py asserts, so it is an upstream
        # defect in the chart, not a property a dataset should absorb. Refusing here keeps
        # the file coherent and names the molecules instead of writing quiet nonsense.
        if int(energy.mask.shape[1]) != int(energy._M.shape[1]):
            skipped.append((smiles, f"chart defect: mask has {int(energy.mask.shape[1])} "
                                    f"columns, _M has {int(energy._M.shape[1])} (alkyne?)"))
            continue
        k_here = int(energy.data_ndim)
        if want_k is None:
            want_k = k_here
        if k_here != want_k:
            skipped.append((smiles, f"k={k_here}, file is k={want_k}"))
            continue
        print(energy.describe())
        mol = condition_from_energy(energy, identifier=ident)
        if not args.no_check:
            err = check_state_convention(mol, energy)
            print(f"   state convention: graph and energy agree to {err:.2e} A")
        if bundle is not None:
            # TREE ORDER, via spec.perm, and asserted against spec.z inside `embed`. The
            # encoder and the conformer path order atoms differently (heavy-then-hydrogen
            # against tree placement), and attaching encoder-order rows to a conformer batch
            # would condition every atom on another atom with no shape error to catch it.
            h, g, _ = encoder_cache.embed(bundle, smiles,
                                          perm=energy.spec.perm, z_tree=energy.spec.z)
            mol.embedding = g[None, :].to(torch.get_default_dtype())
            mol.atom_embedding = h.to(torch.get_default_dtype())
            a, msk = free_dof_atom_index(energy)
            dof_rows.append((mol, a, msk))
        conditions.append(mol)
        energies.append(energy)

    if dof_rows:
        # R IS GLOBAL ACROSS THE FILE, not per molecule. A state column at level='torsion' is
        # collective and owns however many dihedral rows its bond drives -- 3 for one
        # molecule, 2 for the next -- and PyG concatenates these along dim 0, which requires
        # dim 1 to agree. Padding per molecule would fail at collation; padding to the file's
        # maximum makes the ragged dimension a property of the FILE, which is what the
        # consumer can reason about.
        R = max(a.shape[1] for _, a, _ in dof_rows)
        for mol, a, msk in dof_rows:
            pa = np.zeros((a.shape[0], R, a.shape[2]), dtype=np.int64)
            pm = np.zeros((a.shape[0], R), dtype=bool)
            pa[:, :a.shape[1]] = a
            pm[:, :msk.shape[1]] = msk
            # FLATTENED TO ONE ROW PER GRAPH, not left as [k, R, frame]. MolData's
            # `append_batch` classifies a tensor by matching dim 0 to the node or the graph
            # count; a [k, R, frame] field matches neither, so it is taken for SHARED
            # metadata and validated for equality across molecules -- which fails the moment
            # two molecules differ, i.e. always. As [1, k*R*frame] it is an ordinary
            # graph-level field that concatenates, replicates across prior states, and
            # survives the collate. `ConformerGFN.bind_molecular_conditioning` restores the
            # shape from k and MAX_FRAME.
            mol.dof_atoms = torch.as_tensor(pa).reshape(1, -1)
            mol.dof_mask = torch.as_tensor(pm).reshape(1, -1)
        print(f"   DoF atom frames: R = {R} (widest collective column in this file)")

    if skipped:
        print("")
        print(f"{len(skipped)} of {len(args.smiles)} molecules skipped:")
        why = {}
        for smi, reason in skipped:
            why.setdefault(reason.split(',')[0], []).append(smi)
        for reason, smis in sorted(why.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"   {len(smis):4d}  {reason}   e.g. {smis[0]}")
    if not conditions:
        raise SystemExit("no molecules survived; nothing to write")

    batch = collate_conditions(conditions)
    save_condition_file(batch, args.out)
    print(f"\nwrote conditions -> {args.out}  ({batch.num_graphs} graphs, "
          f"k = {int(batch.n_torsions[0])})")
    if bundle is not None:
        print("   embeddings baked. In the config set:")
        print("     embedding_conditioning: true")
        print(f"     embedding_conditioning_dim: {2 * bundle['hidden']}")

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
