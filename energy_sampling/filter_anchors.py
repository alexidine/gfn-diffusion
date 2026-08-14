"""Drop physically invalid structures from a generated anchor set.

Local optimization from random init produces a small tail that is not a crystal: cells
where a clash survived the relaxation (elj > 0) and cells packed past what molecular
volume allows (packing_coeff > 1 is impossible outright). At 9,864 anchors these are ~1.5%
-- rare enough that a 400-structure smoke run showed none of them, which is exactly why
this runs on the full set rather than on a sample.

Writes a NEW file and leaves the input untouched, so a threshold can be revisited.

Usage:
    python filter_anchors.py --in  D:\\...\\qm9_sg2_anchors_elj.pt \\
                             --out D:\\...\\qm9_sg2_anchors_elj_valid.pt
"""
import argparse
import warnings
from collections import Counter
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from mxtaltools.dataset_utils.utils import collate_data_list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-packing", type=float, default=0.85,
                   help="drop above this; >1.0 is outright impossible, and molecular "
                        "crystals essentially never exceed ~0.8")
    p.add_argument("--min-packing", type=float, default=None,
                   help="OFF by default. A very low packing coefficient is a physically "
                        "valid elj minimum but a near-gas, not a crystal -- dropping it is "
                        "a usefulness call, not a validity one, so it is opt-in.")
    return p.parse_args()


def main():
    args = parse_args()
    data = torch.load(args.inp, map_location="cpu", weights_only=False)
    batch = collate_data_list([s.clone() for s in data])
    n = batch.num_graphs

    elj = batch.elj.flatten()
    pc = batch.packing_coeff.flatten()

    finite = torch.isfinite(elj) & torch.isfinite(pc)
    clash = elj > 0
    overpacked = pc > args.max_packing
    reasons = Counter()

    keep = finite & ~clash & ~overpacked
    if args.min_packing is not None:
        underpacked = pc < args.min_packing
        keep = keep & ~underpacked
        reasons["under-packed"] = int((underpacked & finite & ~clash & ~overpacked).sum())

    reasons["non-finite"] = int((~finite).sum())
    reasons["elj > 0 (clash)"] = int((clash & finite).sum())
    reasons[f"packing > {args.max_packing}"] = int((overpacked & finite & ~clash).sum())

    print(f"input : {n} structures from {args.inp.name}")
    for r, c in reasons.items():
        if c:
            print(f"  drop {r:24s} {c:5d}  ({100 * c / n:.2f}%)")
    n_keep = int(keep.sum())
    print(f"KEPT  : {n_keep}  ({100 * n_keep / n:.2f}%)   dropped {n - n_keep}")
    print()

    # What the opt-in floor would cost, so the call can be made on numbers.
    if args.min_packing is None:
        for thr in (0.30, 0.40, 0.50):
            c = int((keep & (pc < thr)).sum())
            print(f"  (--min-packing {thr:.2f} would drop a further {c} of the kept, "
                  f"{100 * c / max(n_keep, 1):.1f}%)")
        print()

    kept = [s for s, k in zip(data, keep.tolist()) if k]

    # Per-molecule coverage after filtering: an anchor set is per-CONDITION, so a molecule
    # left with nothing is a condition with no anchor at all.
    counts = Counter(str(s.identifier) for s in kept)
    before = Counter(str(s.identifier) for s in data)
    lost_all = [m for m in before if counts.get(m, 0) == 0]
    v = torch.tensor(sorted(counts.values()), dtype=torch.float)
    print(f"molecules: {len(counts)} of {len(before)} retain >=1 anchor "
          f"({len(lost_all)} lost every anchor)")
    print(f"anchors per molecule: min {int(v.min())} med {int(v.median())} max {int(v.max())}")

    e_keep, p_keep = elj[keep], pc[keep]
    print()
    print(f"after filter -- elj     [{e_keep.min():.1f}, {e_keep.max():.1f}] "
          f"med {e_keep.median():.1f}")
    print(f"                packing [{p_keep.min():.3f}, {p_keep.max():.3f}] "
          f"med {p_keep.median():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kept, args.out)
    print()
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
