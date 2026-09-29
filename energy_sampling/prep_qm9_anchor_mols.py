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

Continuing a chunk family with the rest of the pool (every molecule not already in it, fixed-size chunks numbered on
from the family's last one, written under the family's stem beside a separate combined file):
    python prep_qm9_anchor_mols.py --n-mols 0 --exclude <family combined .pt> --chunk-size 780 --chunk-start 50
        --chunk-stem qm9_cluster_mols --out <dir>\\qm9_cluster_mols_rest.pt
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
    p.add_argument("--n-mols", type=int, default=200,
                   help="molecules to draw; 0 draws every molecule left in the pool")
    p.add_argument("--exclude", type=Path, nargs="+", default=[],
                   help="molecule files (lists of MolData) whose identifiers leave the pool before the draw, so "
                        "the new set is disjoint from sets already searched")
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
    p.add_argument("--chunk-size", type=int, default=0,
                   help="molecules per chunk file, in place of --chunks equal parts; the last chunk holds the "
                        "remainder")
    p.add_argument("--chunk-start", type=int, default=0,
                   help="number of the first chunk file, so a new set can continue an existing chunk family")
    p.add_argument("--chunk-stem", default=None,
                   help="chunk file stem (default: the --out stem): chunk k is <stem>_chunk<k> beside --out. A "
                        "continuation names the family's stem and its own --out, so the family's combined file "
                        "is not replaced")
    return p.parse_args()


def _refuse_to_replace(path, mols):
    """A file already at path is replaced only by the same molecules in the same order: a wrong --chunk-start or
    stem would otherwise overwrite chunks that searches and priors already name."""
    if path.exists():
        old = [str(m.identifier) for m in torch.load(path, map_location="cpu", weights_only=False)]
        if old != [str(m.identifier) for m in mols]:
            raise SystemExit(f"{path} exists and holds other molecules; not replacing it")


def main():
    args = parse_args()
    pool = torch.load(args.src, map_location="cpu", weights_only=False)
    print(f"pool: {len(pool)} molecules from {args.src.name}")

    excluded_smiles = set()
    if args.exclude:
        drop = set()
        for path in args.exclude:
            drop |= {str(m.identifier) for m in torch.load(path, map_location="cpu", weights_only=False)}
        # an excluded identifier the pool does not hold means the files name molecules differently, and the
        # exclusion would then remove nothing without saying so
        absent = drop - {str(m.identifier) for m in pool}
        if absent:
            raise SystemExit(f"{len(absent)} of {len(drop)} excluded identifiers are not in the pool, "
                             f"e.g. {sorted(absent)[:3]}")
        excluded_smiles = {str(m.smiles) for m in pool if str(m.identifier) in drop}
        pool = [m for m in pool if str(m.identifier) not in drop]
        print(f"excluded {len(drop)} molecules named in {', '.join(p.name for p in args.exclude)}; "
              f"{len(pool)} remain")

    g = torch.Generator().manual_seed(args.seed)
    n_draw = len(pool) if args.n_mols <= 0 else args.n_mols
    idx = torch.randperm(len(pool), generator=g)[:n_draw]
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

    unstable_set = set(unstable)
    keep = [i for i in range(batch.num_graphs) if i not in unstable_set]
    if not keep:
        raise SystemExit("every sampled molecule failed the fixed-point check")

    out_list = batch.to_data_list()
    # to_data_list returns views into the collated batch, and torch.save writes a view's whole storage: without the
    # clone every chunk file carries every molecule's coordinates
    kept = [out_list[i].clone() for i in keep]
    if excluded_smiles:
        shared = sum(str(m.smiles) in excluded_smiles for m in kept)
        print(f"{shared} kept molecules share a SMILES with an excluded one (stereoisomers or repeated graphs); "
              f"a molecule-level split should group them")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.chunks > 1 and args.chunk_size > 0:
        raise SystemExit("--chunks and --chunk-size are alternatives; give one")
    if args.chunks > 1 or args.chunk_size > 0:
        if args.chunks > len(kept):
            raise SystemExit(f"--chunks {args.chunks} exceeds {len(kept)} kept molecules")
        # contiguous slices of an already-shuffled list: disjoint by construction, and
        # chunk k is reproducible from (seed, excluded files, chunks or chunk size, k) alone
        per = args.chunk_size if args.chunk_size > 0 else -(-len(kept) // args.chunks)
        stem = args.chunk_stem or args.out.stem
        parts = [kept[i:i + per] for i in range(0, len(kept), per)]
        paths = [args.out.with_name(f"{stem}_chunk{args.chunk_start + j}{args.out.suffix}")
                 for j in range(len(parts))]
        for path, part in zip(paths, parts):
            _refuse_to_replace(path, part)
        _refuse_to_replace(args.out, kept)
        written = []
        for path, part in zip(paths, parts):
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
            print(f"   {path.name}  {n}  ({path.stat().st_size / 1e6:.2f} MB)")
        if len(written) > 4:
            print(f"   ... {len(written) - 5} more, then {written[-1][0].name}  {written[-1][1]}")
        print(f"   {args.out.name}  {len(kept)}  (combined, for build_anchor_conditions.py; "
              f"{args.out.stat().st_size / 1e6:.2f} MB)")
        print(f"distinct SMILES: {len(set(str(m.smiles) for m in kept))}")
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
