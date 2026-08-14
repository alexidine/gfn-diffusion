"""Sample random QM9 molecules and pin them to the standardized frame, for anchor generation.

Step 2a of the conditional-crystal plan. The crystal search generates anchors from random
init for whatever molecules it is handed; those molecules must ALREADY sit at the
``orient_molecule(mode='std')`` fixed point, because the GFN re-standardizes at rollout and
at buffer admission (see build_qm9_conditions.py's header). A molecule that moves under a
second standardization would have its generated ``aunit_orientation`` silently
reinterpreted the moment the anchor is admitted.

Note what is NOT a problem here: standardization MIRRORS a majority of molecules, which is
fatal when reusing an EXPERIMENTAL crystal's stored parameters (a rotation vector cannot
express an improper transform -- build_qm9_conditions.py's crystal_valid == 0 case, the
reason that ladder capped at 8 molecules). Anchors are generated fresh AFTER the frame is
fixed, so there is no stored orientation to invalidate and the whole QM9 set is usable.

CPU-only. Usage:
    python prep_qm9_anchor_mols.py --n-mols 200 --out D:\\crystal_datasets\\conditional\\priors\\qm9_anchor_mols_200.pt
"""
import argparse
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from mxtaltools.dataset_utils.utils import collate_data_list

DEFAULT_SRC = Path(r"D:\crystal_datasets\csd_free_qm9_dataset.pt")
MO3ENET_VOCAB = {1, 6, 7, 8, 9}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help="molecule pool (list of MolData)")
    p.add_argument("--n-mols", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixed-point-tol", type=float, default=1e-4,
                   help="max allowed per-atom displacement (A) under a SECOND standardization")
    p.add_argument("--chunks", type=int, default=1,
                   help="split the kept molecules into K DISJOINT files for parallel "
                        "crystal searches. Splitting here rather than via run_search's "
                        "mol_seed is deliberate: that path draws with replacement "
                        "(np.random.randint), so separate seeds neither partition the pool "
                        "nor guarantee distinct molecules across jobs.")
    return p.parse_args()


def main():
    args = parse_args()
    pool = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"pool: {len(pool)} molecules from {args.src.name}")

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(pool), generator=g)[:args.n_mols]
    mols = [pool[int(i)].clone() for i in idx]
    print(f"sampled {len(mols)} molecules (seed {args.seed}, without replacement)")

    batch = collate_data_list(mols)

    # Vocabulary guard. The pool should be all-QM9 and therefore inside Mo3ENet's
    # [1,6,7,8,9], but a silently wider pool would only surface much later, as a hard
    # failure inside build_qm9_conditions.py.
    types = set(int(v) for v in batch.z.flatten().tolist())
    bad = types - MO3ENET_VOCAB
    if bad:
        raise SystemExit(f"atom types {sorted(bad)} outside Mo3ENet's vocabulary {sorted(MO3ENET_VOCAB)}")
    print(f"atom types {sorted(types)} -- within the encoder's vocabulary")

    batch.orient_molecule(mode="std")
    pos_once = batch.pos.clone()

    # std-orientation is NOT idempotent in general: a molecule whose inertia frame is
    # near-degenerate (symmetric tops, high-symmetry cages) can pick a different axis
    # assignment on the second pass. Those are exactly the molecules whose generated
    # anchors would be silently reframed at buffer admission, so measure it rather than
    # assume, and drop them.
    probe = collate_data_list(batch.to_data_list())
    probe.orient_molecule(mode="std")
    disp = (probe.pos - pos_once).norm(dim=-1)

    per_mol = torch.zeros(batch.num_graphs)
    per_mol.scatter_reduce_(0, batch.batch, disp, reduce="amax", include_self=False)
    unstable = (per_mol > args.fixed_point_tol).nonzero().flatten().tolist()

    print(f"fixed-point check: max per-atom drift {float(per_mol.max()):.3e} A, "
          f"median {float(per_mol.median()):.3e} A")
    print(f"  unstable (> {args.fixed_point_tol:g} A): {len(unstable)} of {batch.num_graphs}")

    keep = [i for i in range(batch.num_graphs) if i not in set(unstable)]
    if not keep:
        raise SystemExit("every sampled molecule failed the fixed-point check")

    out_list = batch.to_data_list()
    kept = [out_list[i] for i in keep]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.chunks > 1:
        if args.chunks > len(kept):
            raise SystemExit(f"--chunks {args.chunks} exceeds {len(kept)} kept molecules")
        # contiguous slices of an already-shuffled list: disjoint by construction, and
        # chunk k is reproducible from (seed, chunks, k) alone
        per = -(-len(kept) // args.chunks)
        written = []
        for k in range(args.chunks):
            part = kept[k * per:(k + 1) * per]
            if not part:
                continue
            path = args.out.with_name(f"{args.out.stem}_chunk{k}{args.out.suffix}")
            torch.save(part, path)
            written.append((path, len(part)))
        total = sum(n for _, n in written)
        assert total == len(kept), f"chunking lost molecules: {total} != {len(kept)}"
        ids = set()
        for path, _ in written:
            ids |= set(str(m.identifier) for m in torch.load(path, weights_only=False))
        assert len(ids) == len(kept), "chunks overlap -- molecules appear in more than one"
        # the combined file too: build_anchor_conditions.py needs ONE molecule set to
        # embed and to check identifier parity against the merged anchors
        torch.save(kept, args.out)
        print()
        print(f"KEPT {len(kept)} standardized molecules -> {len(written)} disjoint chunks "
              f"of ~{per} (verified: no overlap, none lost)")
        for path, n in written[:4]:
            print(f"   {path.name}  {n}")
        if len(written) > 4:
            print(f"   ... {len(written) - 4} more")
        print(f"   {args.out.name}  {len(kept)}  (combined, for build_anchor_conditions.py)")
        return

    torch.save(kept, args.out)
    print()
    print(f"KEPT {len(kept)} standardized molecules -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.2f} MB)")
    n_atoms = torch.tensor([int(m.num_atoms) for m in kept], dtype=torch.float)
    print(f"atoms/molecule: min {int(n_atoms.min())} med {int(n_atoms.median())} "
          f"max {int(n_atoms.max())}")
    print(f"distinct SMILES: {len(set(str(m.smiles) for m in kept))}")


if __name__ == "__main__":
    main()
