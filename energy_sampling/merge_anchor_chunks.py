"""Merge parallel crystal-search chunk outputs into one anchor set.

Each chunk was generated from a DISJOINT molecule file, so the merge is a concatenation --
but it verifies that rather than assuming it, because two chunks sharing a molecule would
silently double-weight that condition in the prior and in every per-condition statistic
downstream.

Reports which chunks are missing or short instead of quietly merging whatever landed: a
preempted cluster job leaves a partial .pt that loads perfectly well.

Usage:
    python merge_anchor_chunks.py --dir D:\\...\\anchors --tag qm9c100k --chunks 50 \\
        --out D:\\...\\anchors\\qm9c100k_all.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--chunks", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expect-per-chunk", type=int, default=None,
                   help="flag chunks holding fewer than this (a preempted job)")
    p.add_argument("--allow-missing", action="store_true",
                   help="merge anyway when chunks are absent or short")
    return p.parse_args()


def main():
    args = parse_args()
    items, missing, short = [], [], []
    per_chunk_ids = {}

    for k in range(args.chunks):
        path = args.dir / f"{args.tag}_chunk{k}.pt"
        if not path.exists():
            missing.append(k)
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        if args.expect_per_chunk and len(data) < args.expect_per_chunk:
            short.append((k, len(data)))
        per_chunk_ids[k] = set(str(s.identifier) for s in data)
        items.extend(data)

    print(f"chunks found : {args.chunks - len(missing)} of {args.chunks}")
    if missing:
        print(f"  MISSING    : {missing}")
    if short:
        print(f"  SHORT      : {short}  (expected >= {args.expect_per_chunk})")

    # disjointness is the property the whole parallel scheme rests on
    overlaps = []
    seen = {}
    for k, ids in per_chunk_ids.items():
        for mol in ids:
            if mol in seen:
                overlaps.append((seen[mol], k, mol))
            else:
                seen[mol] = k
    if overlaps:
        print(f"  OVERLAP    : {len(overlaps)} molecules in >1 chunk, e.g. {overlaps[:3]}")

    if (missing or short or overlaps) and not args.allow_missing:
        raise SystemExit("refusing to merge -- pass --allow-missing to override")

    print(f"merged       : {len(items)} structures over {len(seen)} molecules")
    if items:
        counts = Counter(str(s.identifier) for s in items)
        v = torch.tensor(sorted(counts.values()), dtype=torch.float)
        print(f"per-molecule : min {int(v.min())} med {int(v.median())} max {int(v.max())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print()
    print("next: filter_anchors.py --in <this> --out <..._valid.pt>")
    print("      build_anchor_conditions.py --anchors <..._valid.pt> "
          "--molecules <the merged molecule set> --holdout-n N")


if __name__ == "__main__":
    main()
