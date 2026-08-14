"""Extract unconditional crystal LATENTS for one space group / Z' from the featurized CSD.

Step 1 of the conditional-crystal plan: the prior is the molecule-agnostic latent
distribution over all CSD structures at a given (SG, Z'), which is what makes it usable
with QM9 molecules whose elements the crystal encoder can embed. Latents are
dimensionless -- ``latent_transform`` divides cell lengths by the molecular radius and
uses fractional centroids -- so latents harvested from large CSD molecules are directly
comparable to those of small QM9 molecules. That is the whole reason this works.

Only latents plus light provenance are stored, NOT the crystals: the molecules are the
part we cannot reuse, and keeping them would be ~100x the bytes for nothing.

Usage (CPU-only, no GPU needed):
    python extract_csd_latents.py --sg 2 --zp 1 --out D:\\crystal_datasets\\conditional\\priors\\csd_sg2_zp1_latents.pt
"""
import argparse
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

from mxtaltools.dataset_utils.utils import collate_data_list

DEFAULT_CHUNK_DIR = Path(r"D:\crystal_datasets\CSD_featurized_chunks")

# latent_params() returns [n, 12] for Z'=1: 3 normed aunit lengths, 3 cell angles,
# 3 fractional aunit centroid, 3 spherical rotvec.
LATENT_NAMES = ["len_a", "len_b", "len_c", "ang_al", "ang_be", "ang_ga",
                "cen_0", "cen_1", "cen_2", "rot_0", "rot_1", "rot_2"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sg", type=int, default=2, help="space group index")
    p.add_argument("--zp", type=int, default=1, help="Z prime")
    p.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="limit chunks, for a quick smoke run")
    p.add_argument("--batch-size", type=int, default=256,
                   help="structures per latent_params() call")
    p.add_argument("--keep-nonstandard", action="store_true",
                   help="keep nonstandard_symmetry entries (dropped by default)")
    return p.parse_args()


def main():
    args = parse_args()
    chunks = sorted(args.chunk_dir.glob("_chunk_*.pkl"))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    if not chunks:
        raise SystemExit(f"no chunks found under {args.chunk_dir}")

    lat_blocks, identifiers = [], []
    meta = {k: [] for k in ("packing_coeff", "density", "radius", "num_atoms",
                            "mol_volume", "cell_volume")}
    n_total = n_match = n_dropped_illdef = n_dropped_nonstd = n_dropped_bad = 0

    t0 = time.time()
    pending = []

    def flush(pending):
        """Latents for one accumulated group. Returns count actually kept."""
        nonlocal n_dropped_bad
        if not pending:
            return 0
        batch = collate_data_list([s.clone() for s in pending])
        lat = batch.latent_params()
        # A non-finite latent means the structure's cell params were unusable; drop it
        # rather than poisoning the prior with a NaN row. Counted and reported, never
        # silently absorbed.
        good = torch.isfinite(lat).all(dim=1)
        n_bad = int((~good).sum())
        if n_bad:
            n_dropped_bad += n_bad
        lat_blocks.append(lat[good].clone())
        kept = [s for s, g in zip(pending, good.tolist()) if g]
        for s in kept:
            identifiers.append(str(getattr(s, "identifier", "")))
            for k in meta:
                v = getattr(s, k, None)
                meta[k].append(float(v) if v is not None else float("nan"))
        return len(kept)

    for ci, path in enumerate(chunks):
        data = torch.load(path, map_location="cpu", weights_only=False)
        n_total += len(data)
        for s in data:
            if int(s.sg_ind) != args.sg or int(s.z_prime) != args.zp:
                continue
            n_match += 1
            if not bool(getattr(s, "is_well_defined", True)):
                n_dropped_illdef += 1
                continue
            if not args.keep_nonstandard and bool(getattr(s, "nonstandard_symmetry", False)):
                n_dropped_nonstd += 1
                continue
            pending.append(s)
            if len(pending) >= args.batch_size:
                flush(pending)
                pending = []
        if (ci + 1) % 50 == 0:
            n_so_far = sum(b.shape[0] for b in lat_blocks)
            print(f"  chunk {ci + 1}/{len(chunks)}  kept {n_so_far}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    flush(pending)

    latents = torch.cat(lat_blocks, dim=0) if lat_blocks else torch.empty(0, 12)

    # The latent box is a hard clip inside latent_params(), so a structure outside the
    # parameterisation's range lands ON the rail instead of raising. Report it: a large
    # railed fraction means the prior is piling mass on the boundary.
    railed = (latents.abs() >= 1.0 - 1e-6)
    railed_rows = int(railed.any(dim=1).sum())

    print()
    print(f"scanned          : {len(chunks)} chunks, {n_total} entries")
    print(f"SG={args.sg} Z'={args.zp}     : {n_match}")
    print(f"  dropped ill-defined  : {n_dropped_illdef}")
    print(f"  dropped nonstandard  : {n_dropped_nonstd}")
    print(f"  dropped non-finite   : {n_dropped_bad}")
    print(f"KEPT             : {latents.shape[0]}")
    print(f"rows on a rail   : {railed_rows} ({100 * railed_rows / max(len(latents), 1):.2f}%)")
    print()
    print("per-dim summary:")
    for i, nm in enumerate(LATENT_NAMES[:latents.shape[1]]):
        col = latents[:, i]
        print(f"  {nm:8s} mean {col.mean():+.4f}  std {col.std():.4f}  "
              f"[{col.min():+.3f}, {col.max():+.3f}]  railed {int(railed[:, i].sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latents,
        "latent_names": LATENT_NAMES[:latents.shape[1]],
        "identifier": identifiers,
        "meta": {k: torch.tensor(v) for k, v in meta.items()},
        "sg_ind": args.sg,
        "z_prime": args.zp,
        "source": str(args.chunk_dir),
        "n_chunks": len(chunks),
        "kept_nonstandard": bool(args.keep_nonstandard),
        "counts": {"scanned": n_total, "matched": n_match,
                   "dropped_ill_defined": n_dropped_illdef,
                   "dropped_nonstandard": n_dropped_nonstd,
                   "dropped_non_finite": n_dropped_bad},
    }, args.out)
    print()
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
