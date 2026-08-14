"""Fill prior_dataset / prior_buffer from the handcrafted prior -- no MLE model needed.

In the crystal harness the prior sampler is a *trained* model: phase 1 fits it by MLE,
and everything downstream draws prior states from it. For a toy that is circular work.
``mxtaltools.conformers.prior.InternalPrior`` is already a generative model over internal
DoF -- one-body histograms per DoF type, plus an empirical joint per ring system -- so it
occupies exactly the ``prior_model`` slot and can emit prior states directly.

MLE warm-start is unaffected: it still runs, it just trains against a dataset written by
the handcrafted prior instead of by a learned one.

What comes out is a ``.pt`` holding torsion states in the sampler's own units -- deltas
from the reference conformer on ``[-1, 1]`` -- so the same file serves as
``prior_states_path`` (prior buffer) and as a prior_dataset for MLE.

    python build_prior_states.py --smiles CCCCO --n 20000 \
        --datasets ../../mxtaltools/mini_datasets/mini_CSD_dataset.pt \
        --prior-path conformer_prior.pt

**Typing caveat, deliberate and worth knowing.** ``torsion_key`` is the central bond
only, so every quadruple about a given bond -- heavy and hydrogen alike -- pools into one
histogram. The siblings about an sp3 axis sit at +/-120 deg from each other, so that
pooled distribution is close to 3-fold symmetric, which is what makes it legitimate to
apply a draw from it as a delta on an arbitrary representative quadruple. It is an
approximation, not an identity, and it is the same coarseness that makes the prior
transferable in the first place.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from energies.conformer_torsions import ConformerTorsions


def wrap(x):
    """Wrap to the state space (-1, 1]. Period 2, not 2*pi."""
    return (x + 1.0) % 2.0 - 1.0


def fit_or_load(prior_path: Path, dataset_paths, fatten: float):
    """A fitted InternalPrior, from cache if present or from the datasets if not."""
    from mxtaltools.conformers.prior import InternalPrior

    if prior_path is not None and prior_path.exists():
        prior = torch.load(prior_path, weights_only=False)
        prior.fatten = fatten
        print(f"prior: loaded fit from {prior_path} "
              f"({prior.n_fitted} molecules, {len(prior.torsions)} torsion types)")
        return prior

    if not dataset_paths:
        raise SystemExit("no --prior-path cache and no --datasets to fit on")

    mols = []
    for p in dataset_paths:
        blob = torch.load(Path(p), weights_only=False)
        mols.extend(blob)
        print(f"  {len(blob):>6} molecules from {p}")
    prior = InternalPrior(fatten=fatten).fit(mols)
    if prior_path is not None:
        torch.save(prior, prior_path)
        print(f"prior: cached fit to {prior_path}")
    return prior


def torsion_histograms(energy, prior):
    """One Histogram1D (or None) per sampled torsion, with its reference dihedral.

    Returns ``[(hist_or_None, phi_ref, key)]``, one entry per state dimension. A None
    means that torsion type was absent from the fit; the caller falls back to uniform for
    that dimension and says so, because a silent fallback here would be indistinguishable
    from a prior that simply had nothing to say.
    """
    keys = energy.atom_keys
    mask = energy.mask.cpu().numpy()
    ph0 = energy.ph0.cpu().numpy()

    out = []
    for j, (u, v) in enumerate(energy.rotatable):
        key = prior.torsion_key(keys[u], keys[v])
        rows = np.flatnonzero(mask[:, j] != 0)
        # any quadruple about this axis is an equally valid representative -- see the
        # module docstring on why the pooled histogram makes that defensible
        out.append((prior.torsions.get(key), float(ph0[rows[0]]), key))
    return out


def draw_states(energy, prior, n: int, rng):
    """``[n, k]`` torsion states on [-1, 1], drawn from the handcrafted prior."""
    hists = torsion_histograms(energy, prior)
    x = np.empty((n, len(hists)), dtype=np.float64)
    n_uniform = 0
    print(f"\n{'dim':>4} {'axis':>10} {'torsion type':>26} {'source':>10} {'obs':>8}")
    sym = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
    for j, (hist, phi_ref, key) in enumerate(hists):
        u, v = energy.rotatable[j]
        axis = f"{sym.get(int(energy.atom_keys[u][0]), '?')}{u}-" \
               f"{sym.get(int(energy.atom_keys[v][0]), '?')}{v}"
        pretty = "-".join(f"{sym.get(e, e)}{d}" for e, d in key)
        if hist is None:
            x[:, j] = rng.uniform(-1.0, 1.0, n)
            n_uniform += 1
            print(f"{j:>4} {axis:>10} {pretty:>26} {'UNIFORM':>10} {0:>8}")
        else:
            phi = hist.sample(n, prior.fatten, rng)
            x[:, j] = wrap((phi - phi_ref) / np.pi)
            print(f"{j:>4} {axis:>10} {pretty:>26} {'prior':>10} "
                  f"{int(hist.counts.sum()):>8}")
    if n_uniform:
        print(f"\n  {n_uniform}/{len(hists)} dimensions fell through to uniform")
    return torch.from_numpy(x), n_uniform


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", default="CCCCO")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--datasets", nargs="*", default=[],
                    help=".pt molecule lists to fit the prior on")
    ap.add_argument("--prior-path", type=Path, default=None,
                    help="cache for the fitted prior; loaded if it exists, else written")
    ap.add_argument("--fatten", type=float, default=0.15,
                    help="uniform mixing weight. A MODELLING CHOICE -- 0.15 is the "
                         "InternalPrior default, not a tuned value")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    # see build_conformer_buffer.py: `torsion` is explicit, not a default. The fitted
    # InternalPrior draw and the period-2 wrapping below both assume torsion-only state.
    energy = ConformerTorsions(smiles=args.smiles, device="cpu", epsilon=args.epsilon,
                               level="torsion")
    print(energy.describe())
    prior = fit_or_load(args.prior_path, args.datasets, args.fatten)
    states, n_uniform = draw_states(energy, prior, args.n, rng)

    # energies are diagnostic only -- how plausible is the prior? The crystal prior runs
    # <10% reasonable samples, so a low hit rate here is expected, not a failure
    e = torch.cat([energy.energy(states[i:i + 4096]) for i in range(0, len(states), 4096)])
    e_unif = energy.energy(torch.rand(len(states), energy.data_ndim) * 2 - 1)
    print(f"\nprior   E: median {e.median():+8.3f}   p10 {torch.quantile(e, 0.1):+8.3f}")
    print(f"uniform E: median {e_unif.median():+8.3f}   p10 {torch.quantile(e_unif, 0.1):+8.3f}")

    out = args.out or Path(f"conformer_prior_states_{args.smiles.replace('/', '_')}.pt")
    torch.save(dict(states=states, energies=e, smiles=args.smiles,
                    k=energy.data_ndim, fatten=args.fatten,
                    n_uniform_dims=n_uniform, source="InternalPrior",
                    datasets=[str(d) for d in args.datasets]), out)
    print(f"\nwrote {out}  ({len(states)} states)")


if __name__ == "__main__":
    main()
