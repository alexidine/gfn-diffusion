"""Turn a generated anchor set into the ``molecules_path`` / ``prior_path`` pair a run needs.

The anchor search produces bare crystals. Two things are still missing before train.py can
consume them:

  * **Embeddings.** With ``embedding_conditioning`` on, MolecularCrystal raises unless every
    batch carries ``embedding`` -- and ``init_prior_dataset`` re-analyzes the prior's
    energies at startup, so the PRIOR needs them too, not just the conditions.
  * **The file envelope.** ``molecules_path`` is unwrapped from ``{'prior': batch}``
    (train.py:_load_condition_file) and ``prior_path`` is read from ``'equalized_prior'``
    (train.py:init_prior_dataset).

Embeddings are computed ONCE on the canonical standardized molecules and broadcast to
anchors by identifier, rather than re-encoded per anchor: the embedding is a property of
the molecule, not of the crystal it sits in, and this makes every anchor of a given
molecule share a bit-identical condition vector. Verified safe because an anchor's
molecular geometry still matches its source molecule and is still a std-orient fixed point.

The conditions file gets ONE entry per molecule (the lowest-energy anchor as carrier) so
condition sampling is exactly uniform over molecules; the prior gets the full set.

NO ``thermal_scaling_factor`` is written. train.py:1700 tests for the key's PRESENCE and
lets it REPLACE lj_coeff for the whole run, so writing one silently rescales every energy.
Omitting it keeps ``energy_config.temperature`` in RAW ELJ units -- see the reminder
printed at the end.

Usage:
    python build_anchor_conditions.py --anchors D:\\...\\qm9_sg2_anchors_elj_valid.pt \\
        --molecules D:\\...\\qm9_anchor_mols_200.pt --out-dir D:\\...\\priors --tag qm9a198
"""
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

from mxtaltools.common.training_utils import load_molecule_autoencoder
from mxtaltools.dataset_utils.utils import collate_data_list

DEFAULT_ENCODER = r"D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15"
MO3ENET_ATOM_TYPES = {1, 6, 7, 8, 9}


@torch.no_grad()
def embed(items, encoder, device, batch_size):
    """Frozen Mo3ENet encoding, mirroring build_qm9_conditions.embed exactly.

    Returns [n, 3, bottleneck] -- the EQUIVARIANT latent, kept unscalarized so a future
    augmentation can rotate it with the molecule. `encode` divides pos in place and asserts
    the batch is centered, so it only ever sees a clone, re-centered on ALL atoms (the
    autoencoder was trained on global centroids, while orient_molecule centers on heavy
    atoms).
    """
    out = []
    for start in tqdm(range(0, len(items), batch_size), desc="  embedding"):
        chunk = collate_data_list([m.clone() for m in items[start:start + batch_size]]).to(device)
        chunk.recenter_molecules(center_on_heavy_atoms=False)
        out.append(encoder.encode(chunk).clone().cpu())
    return torch.cat(out, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--molecules", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--encoder", type=Path, default=Path(DEFAULT_ENCODER))
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu is ample for a few hundred molecules and avoids GPU contention")
    p.add_argument("--embed-batch-size", type=int, default=100)
    p.add_argument("--holdout-n", type=int, default=0,
                   help="carve this many DISTINCT SMILES into a separate eval-only "
                        "condition file; the rest become the training conditions")
    p.add_argument("--holdout-seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    anchors = torch.load(args.anchors, map_location="cpu", weights_only=False)
    mols = torch.load(args.molecules, map_location="cpu", weights_only=False)
    print(f"anchors   : {len(anchors)} from {args.anchors.name}")
    print(f"molecules : {len(mols)} from {args.molecules.name}")

    mol_ids = [str(m.identifier) for m in mols]
    row_of = {m: i for i, m in enumerate(mol_ids)}
    if len(row_of) != len(mol_ids):
        raise SystemExit("molecule identifiers are not unique -- condition_id would collide")

    # Identifier parity is prep-side responsibility (init_identifiers' docstring): a prior
    # row whose identifier is absent from the registry cannot be matched to a condition.
    anchor_ids = set(str(a.identifier) for a in anchors)
    missing = anchor_ids - set(mol_ids)
    if missing:
        raise SystemExit(f"{len(missing)} anchor identifiers absent from the molecule file, "
                         f"e.g. {sorted(missing)[:3]}")
    unused = set(mol_ids) - anchor_ids
    if unused:
        print(f"  NOTE {len(unused)} molecules have no surviving anchor; dropping them from "
              f"the condition set so every condition has support")
        keep_ids = [m for m in mol_ids if m in anchor_ids]
    else:
        keep_ids = mol_ids

    types = set(int(v) for m in mols for v in m.z.flatten().tolist())
    if types - MO3ENET_ATOM_TYPES:
        raise SystemExit(f"atom types {sorted(types - MO3ENET_ATOM_TYPES)} outside the "
                         f"encoder's vocabulary")

    print(f"loading frozen encoder {args.encoder.name}")
    encoder = load_molecule_autoencoder(str(args.encoder), "cpu").to(args.device)
    keep_mols = [mols[row_of[m]] for m in keep_ids]
    emb = embed(keep_mols, encoder, args.device, args.embed_batch_size)
    flat_dim = int(emb.shape[1] * emb.shape[2])
    print(f"  embeddings {tuple(emb.shape)}  -> flat condition dim {flat_dim}")
    if not torch.isfinite(emb).all():
        raise SystemExit("non-finite embedding values")
    # Distinct molecules must get distinct conditions, or Z(c) has nothing to separate.
    flat = emb.reshape(len(keep_mols), -1)
    dmin = torch.cdist(flat, flat).fill_diagonal_(float("inf")).min()
    print(f"  min pairwise embedding distance {float(dmin):.4f} "
          f"(0 would mean two conditions are indistinguishable)")

    emb_row = {m: i for i, m in enumerate(keep_ids)}

    # ---- train / hold-out split, at the SMILES level ---------------------------------
    # Splitting by identifier would leak if two identifiers ever shared a SMILES: the
    # held-out molecule would be the same chemistry the model trained on, and the
    # generalization number would be measuring memorization.
    smiles_of = {m: str(mols[row_of[m]].smiles) for m in keep_ids}
    by_smiles = defaultdict(list)
    for m in keep_ids:
        by_smiles[smiles_of[m]].append(m)
    n_smiles = len(by_smiles)
    print(f"distinct SMILES among {len(keep_ids)} molecules: {n_smiles}"
          f"{'' if n_smiles == len(keep_ids) else '  (SOME SHARED -- split groups them)'}")

    test_ids = []
    if args.holdout_n > 0:
        if args.holdout_n >= n_smiles:
            raise SystemExit(f"--holdout-n {args.holdout_n} leaves no training conditions "
                             f"(only {n_smiles} distinct SMILES)")
        keys = sorted(by_smiles)
        g = torch.Generator().manual_seed(args.holdout_seed)
        pick = torch.randperm(len(keys), generator=g)[:args.holdout_n]
        for i in pick.tolist():
            test_ids.extend(by_smiles[keys[i]])
    test_set = set(test_ids)
    train_ids = [m for m in keep_ids if m not in test_set]
    if test_ids:
        assert not (set(smiles_of[m] for m in train_ids)
                    & set(smiles_of[m] for m in test_ids)), "SMILES leaked across the split"
        print(f"split: {len(train_ids)} training molecules / {len(test_ids)} held out "
              f"(seed {args.holdout_seed}, no shared SMILES)")

    # ---- prior: anchors of the TRAINING molecules only, embedding from their molecule -
    train_set = set(train_ids)
    prior_items, carrier = [], {}
    for a in anchors:
        k = str(a.identifier)
        s = a.clone()
        s.embedding = emb[emb_row[k]].unsqueeze(0).clone()
        e = float(a.elj)
        if k not in carrier or e < carrier[k][0]:
            carrier[k] = (e, s)          # lowest-energy anchor, used as the condition carrier
        if k in train_set:
            prior_items.append(s)
    prior_batch = collate_data_list(prior_items)
    print(f"prior batch: {prior_batch.num_graphs} graphs (training molecules only), "
          f"embedding {tuple(prior_batch.embedding.shape)}")

    # ---- conditions: one carrier per molecule ----------------------------------------
    cond_batch = collate_data_list([carrier[m][1].clone() for m in train_ids])
    print(f"cond batch : {cond_batch.num_graphs} graphs (one per training molecule)")
    test_batch = (collate_data_list([carrier[m][1].clone() for m in test_ids])
                  if test_ids else None)
    if test_batch is not None:
        print(f"test batch : {test_batch.num_graphs} graphs (one per held-out molecule)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cond_path = args.out_dir / f"{args.tag}_conditions.pt"
    prior_path = args.out_dir / f"{args.tag}_prior.pt"

    meta = {
        "n_molecules": len(train_ids),
        "n_structures": int(prior_batch.num_graphs),
        "n_holdout_molecules": len(test_ids),
        "holdout_seed": args.holdout_seed if test_ids else None,
        "embedding_dim": flat_dim,
        "frame": "orient_molecule(mode=std) fixed point",
        "sg_ind": 2,
        "z_prime": 1,
        "energy_function": "elj",
        "anchors_source": str(args.anchors),
        "molecules_source": str(args.molecules),
        "encoder": str(args.encoder),
        "provenance": "anchors generated from random init by crystal_search under elj; "
                      "NOT experimental structures",
    }
    torch.save({"prior": cond_batch, **meta}, cond_path)
    torch.save({"prior": prior_batch, "equalized_prior": prior_batch, **meta}, prior_path)
    print()
    print(f"wrote {cond_path}  ({cond_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {prior_path} ({prior_path.stat().st_size / 1e6:.1f} MB)")
    if test_batch is not None:
        # eval-only: held-out conditions are forward-sampled, never trained on, so this
        # gets no prior/equalized_prior companion
        test_path = args.out_dir / f"{args.tag}_test_conditions.pt"
        torch.save({"prior": test_batch, **meta, "split": "holdout"}, test_path)
        print(f"wrote {test_path} ({test_path.stat().st_size / 1e6:.1f} MB)")
    print()
    print("NOT set: thermal_scaling_factor. train.py:1700 fires on the KEY'S PRESENCE and "
          "lets it replace lj_coeff for the whole run. Without it, energies stay in RAW "
          "ELJ units and effective kT is energy_config.temperature exactly as written -- "
          "so configs must use the RAW value (the qm9_aug11 arms use 6.9, matching "
          "mipcas's nominal 2.5 / 0.3636), NOT a nominal 2.5.")
    print(f"Config keys: molecules_path: {cond_path}")
    print(f"             prior_path:     {prior_path}")
    print(f"             embedding_conditioning: true, embedding_conditioning_dim: {flat_dim}")


if __name__ == "__main__":
    main()
