"""Turn the qm9_full_sep29 search (MXtalTools configs/crystal_searches/qm9_full_sep29) into the
``molecules_path`` / ``prior_path`` / ``test_molecules_path`` files a conditional run reads, in the envelope
build_anchor_conditions.py writes: ``{'prior': batch}`` for the condition files (one carrier per molecule, its
lowest-energy kept crystal), ``{'prior': batch, 'equalized_prior': batch}`` for the prior, the held-out file carrying
``"split": "holdout"`` and no prior companion. Every row carries its molecule's frozen Mo3ENet ``embedding``,
computed once per molecule and broadcast by identifier.

Stages, each resumable from its files in --work-dir:
  1. embed: every molecule of the chunk family, once (work/embeddings.pt).
  2. chunks (parallel processes, one chunk each; work/chunk<k>.pt + chunk<k>_audit.json):
     - drop crystals whose stored cell lengths and angles describe no cell (a non-positive-definite metric, which
       standardize_cells cannot handle; the energy was computed on the stored box, whose parameters then disagree);
     - standardize_cells (MXtalTools), dropping rows it cannot place;
     - per molecule, near-duplicates: in ascending eLJ, a crystal within --dup-cut envwise RDF distance of one already
       kept is dropped. Chunks below --diverse-below compute every RDF; above it, only crystals in a pair within
       --screen-de eLJ and --screen-dpc packing coefficient are compared, a screen whose recall the full chunks
       measure (audit: dup_pairs, dup_pairs_screened);
     - chunks below --diverse-below keep the --keep most diverse survivors: farthest-point sampling in RDF distance,
       seeded with the lowest-eLJ crystal.
  3. assemble: re-describe every crystal in the trainer's chart (to_trainer_chart: handedness +1, centroid x in
     [0, 1/2]; the search draws handedness at random and the trainer reads every row at +1), then
     hold out a random --holdout-frac of the molecules (distinct SMILES drawn under --holdout-seed; all of
     their crystals leave the prior), write the three training files, <tag>_test_prior.pt (every kept crystal of the
     held-out molecules, in the prior's envelope, for offline evaluation; the trainer does not read it) and a
     provenance file (search chunk, seed and row of every prior and held-out row).

CPU only. Usage (from energy_sampling/):
    python build_qm9_full_prior.py --search-dir D:\\...\\anchors\\qm9_full_sep29 --mol-dir D:\\...\\priors
        --work-dir D:\\...\\anchors\\qm9_full_sep29\\prior_build --out-dir D:\\...\\priors --tag qm9full --workers 12
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

DEFAULT_ENCODER = r"D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-dir", type=Path, required=True, help="the battery's outputs, qm9full_c<k>_<seed>.pt")
    p.add_argument("--mol-dir", type=Path, required=True, help="the chunk family qm9_cluster_mols_chunk<k>.pt")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--holdout-frac", type=float, default=0.05,
                   help="fraction of the molecules (distinct SMILES) held out, with all their crystals")
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=205)
    p.add_argument("--chunks", type=str, default=None, help="comma list: build only these chunks (testing)")
    p.add_argument("--seeds-of-chunk0", type=int, default=4)
    p.add_argument("--diverse-below", type=int, default=50, help="chunks below this get the diversity selection")
    p.add_argument("--keep", type=int, default=10)
    p.add_argument("--dup-cut", type=float, default=0.01)
    p.add_argument("--screen-de", type=float, default=1.0, help="eLJ raw units")
    p.add_argument("--screen-dpc", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--encoder", type=Path, default=Path(DEFAULT_ENCODER))
    p.add_argument("--embed-batch-size", type=int, default=200)
    return p.parse_args()


# ------------------------------------------------------------------------------------------------ stage 1: embed
def stage_embed(args):
    path = args.work_dir / "embeddings.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from build_anchor_conditions import MO3ENET_ATOM_TYPES, embed
    from mxtaltools.common.training_utils import load_molecule_autoencoder
    mols = []
    for k in range(args.n_chunks):
        mols.extend(torch.load(args.mol_dir / f"qm9_cluster_mols_chunk{k}.pt", map_location="cpu",
                               weights_only=False))
    ids = [str(m.identifier) for m in mols]
    if len(set(ids)) != len(ids):
        raise SystemExit("molecule identifiers are not unique across the chunk family")
    types = set(int(v) for m in mols for v in m.z.flatten().tolist())
    if types - MO3ENET_ATOM_TYPES:
        raise SystemExit(f"atom types {sorted(types - MO3ENET_ATOM_TYPES)} outside the encoder's vocabulary")
    encoder = load_molecule_autoencoder(str(args.encoder), "cpu")
    t = time.time()
    emb = embed(mols, encoder, "cpu", args.embed_batch_size)
    if not torch.isfinite(emb).all():
        raise SystemExit("non-finite embedding values")
    out = dict(identifiers=ids, smiles=[str(m.smiles) for m in mols], embedding=emb, encoder=str(args.encoder))
    torch.save(out, path)
    print(f"embedded {len(ids)} molecules in {time.time() - t:.0f} s -> {tuple(emb.shape)}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ stage 2: chunks
def _metric_ok(c):
    a, b, g = torch.cos(c.cell_angles.double().flatten())
    return float(1 - a * a - b * b - g * g + 2 * a * b * g) > 1e-6


def _standardize(crystals):
    """standardize_cells on the crystals; rows it raises on or cannot place are returned as failures. Collated
    without the coordinator's RDF_DROP exclusions, which would strip elj and lj from the stored rows."""
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.dataset_utils.utils import collate_data_list
    try:
        std, info = standardize_cells(collate_data_list([c.clone() for c in crystals]), on_failure='flag')
        ok = np.asarray(info['ok'], dtype=bool)
        std_list = std.batch_to_list()
        return [s for s, o in zip(std_list, ok) if o], [i for i, o in enumerate(ok) if not o]
    except Exception:  # one bad cell spoils the batch call: place the rows one at a time
        kept, bad = [], []
        for i, c in enumerate(crystals):
            try:
                std, info = standardize_cells(collate_data_list([c.clone()]), on_failure='flag')
                if bool(np.asarray(info['ok'])[0]):
                    kept.append(std.batch_to_list()[0])
                else:
                    bad.append(i)
            except Exception:
                bad.append(i)
        return kept, bad


def _fps(D, n_keep):
    """farthest-point order in distance matrix D [m, m], starting from index 0; returns n_keep indices"""
    sel = [0]
    mind = D[0].clone()
    mind[0] = -1
    while len(sel) < min(n_keep, len(D)):
        j = int(torch.argmax(mind))
        sel.append(j)
        mind = torch.minimum(mind, D[j])
        mind[sel] = -1
    return sel


def process_chunk(k, a):
    """a: dict of settings. Writes work/chunk<k>.pt (collated kept crystals with embeddings) and its audit."""
    torch.set_num_threads(1)
    from mxtaltools.crystal_search.coordinator import compute_rdfs, rdf_distance_matrix
    from mxtaltools.dataset_utils.utils import collate_data_list
    work = Path(a['work_dir'])
    out_path, audit_path = work / f"chunk{k}.pt", work / f"chunk{k}_audit.json"
    if out_path.exists() and audit_path.exists():
        return json.loads(audit_path.read_text())
    t0 = time.time()
    emb = torch.load(work / "embeddings.pt", weights_only=False)
    emb_row = {m: i for i, m in enumerate(emb['identifiers'])}
    seeds = range(a['seeds_of_chunk0']) if k == 0 else [0]
    diverse = k < a['diverse_below']
    by_mol = defaultdict(list)          # identifier -> [(crystal, seed, row)]
    audit = Counter()
    for s in seeds:
        for r, c in enumerate(torch.load(Path(a['search_dir']) / f"qm9full_c{k}_{s}.pt", weights_only=False,
                                         map_location="cpu")):
            audit['loaded'] += 1
            if not _metric_ok(c):
                audit['dropped_invalid_cell_parameters'] += 1
                continue
            by_mol[str(c.identifier)].append((c, s, r))
    kept_rows, provenance = [], []
    lat_edge = 0
    per_mol_kept = []
    for ident, rows in by_mol.items():
        std, bad = _standardize([c for c, _, _ in rows])
        audit['dropped_standardize'] += len(bad)
        badset = set(bad)
        rows = [rw for i, rw in enumerate(rows) if i not in badset]
        e = torch.tensor([float(c.elj) for c, _, _ in rows])
        pc = torch.tensor([float(c.packing_coeff) for c, _, _ in rows])
        order = torch.argsort(e)
        std = [std[i] for i in order.tolist()]
        rows = [rows[i] for i in order.tolist()]
        e, pc = e[order], pc[order]
        n = len(rows)
        # which crystals need RDFs: all on diverse chunks; else those in a screened near pair
        if diverse:
            need = torch.ones(n, dtype=torch.bool)
        else:
            close = ((e[:, None] - e[None]).abs() < a['screen_de']) & ((pc[:, None] - pc[None]).abs() < a['screen_dpc'])
            close.fill_diagonal_(False)
            need = close.any(1)
        D = None
        if need.any():
            idx = torch.nonzero(need).flatten()
            R = compute_rdfs([std[i] for i in idx.tolist()], batch_size=max(1, len(idx)))
            Dn = rdf_distance_matrix(R, R)
            D = torch.full((n, n), float('inf'))
            D[idx[:, None], idx[None]] = Dn
            if diverse:  # recall of the screen, measured where every pair is known
                dup = (Dn < a['dup_cut'])
                dup.fill_diagonal_(False)
                scr = ((e[:, None] - e[None]).abs() < a['screen_de']) & ((pc[:, None] - pc[None]).abs() < a['screen_dpc'])
                audit['dup_pairs'] += int(dup.triu(1).sum())
                audit['dup_pairs_screened'] += int((dup & scr).triu(1).sum())
        keep = []
        for i in range(n):   # ascending eLJ: the first of a near-duplicate group is the lowest
            if D is not None and keep and bool((D[i, keep] < a['dup_cut']).any()):
                audit['dropped_duplicate'] += 1
                continue
            keep.append(i)
        if diverse and len(keep) > a['keep']:
            sel = _fps(D[keep][:, keep], a['keep'])
            audit['dropped_diversity'] += len(keep) - len(sel)
            keep = [keep[j] for j in sorted(sel)]
        per_mol_kept.append(len(keep))
        for i in keep:
            c = std[i]
            c.embedding = emb['embedding'][emb_row[ident]].unsqueeze(0).clone()
            kept_rows.append(c)
            provenance.append((ident, rows[i][1], rows[i][2], float(e[i])))
    batch = collate_data_list(kept_rows)
    probe = batch.clone()  # latent_params() canonicalizes before transforming; count on a canonicalized copy
    probe.canonicalize_zp_aunits()
    probe.canonicalize_free_axes()
    lat = probe.latent_transform(cell_params=probe.full_cell_parameters())
    lat_edge = int((lat.abs() > 1).any(1).sum())
    del probe
    torch.save(dict(batch=batch, provenance=provenance), out_path)
    audit = dict(audit, chunk=k, molecules=len(by_mol), kept=len(kept_rows), rows_outside_latent_box=lat_edge,
                 kept_per_molecule=dict(Counter(per_mol_kept)), seconds=round(time.time() - t0, 1))
    audit_path.write_text(json.dumps(audit))
    return audit


def stage_chunks(args):
    chunks = [int(c) for c in args.chunks.split(',')] if args.chunks else list(range(args.n_chunks))
    a = dict(work_dir=str(args.work_dir), search_dir=str(args.search_dir), seeds_of_chunk0=args.seeds_of_chunk0,
             diverse_below=args.diverse_below, keep=args.keep, dup_cut=args.dup_cut, screen_de=args.screen_de,
             screen_dpc=args.screen_dpc)
    audits = []
    # chunk 0 (200 starts per molecule) is the longest: start it first
    order = sorted(chunks, key=lambda c: (c != 0, c))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_chunk, k, a): k for k in order}
        for f in as_completed(futs):
            au = f.result()
            audits.append(au)
            print(f"  chunk {au['chunk']:>3}: {au['loaded']} loaded -> {au['kept']} kept "
                  f"({au.get('dropped_duplicate', 0)} duplicates, {au.get('dropped_diversity', 0)} by diversity, "
                  f"{au.get('dropped_invalid_cell_parameters', 0)} invalid cells, "
                  f"{au.get('dropped_standardize', 0)} not standardized) in {au['seconds']} s "
                  f"[{len(audits)}/{len(order)}]", flush=True)
    return sorted(audits, key=lambda x: x['chunk'])


# ------------------------------------------------------------------------------------------------ stage 3: assemble
FIT_TOL = 1e-2  # Angstrom RMSD: the source molecule must superimpose on one of the two copies this closely


def _proper_fit(M, X, batch, n):
    """Per graph, the proper rotation R (det +1) minimising |R m_i - x_i| over its atoms (batched Kabsch), and the
    RMSD it leaves. M, X: [n_atoms, 3], each graph's atoms centred; batch: [n_atoms] graph index."""
    H = torch.zeros(n, 3, 3, dtype=torch.float64).index_add_(0, batch, M[:, :, None] * X[:, None, :])
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    res = ((R[batch] @ M[:, :, None])[..., 0] - X).pow(2).sum(-1)
    cnt = torch.zeros(n, dtype=torch.float64).index_add_(0, batch, torch.ones_like(res))
    return R, (torch.zeros(n, dtype=torch.float64).index_add_(0, batch, res) / cnt).sqrt()


def to_trainer_chart(b, mol_pos):
    """Re-describe every P-1, Z'=1 crystal in the chart the trainer reads, in place: the stored molecule IS the source
    molecule (mol_pos[identifier], its std-orientation fixed point, the molecule the conditions were embedded from),
    handedness +1, centroid x in [0, 1/2], y and z in [0, 1). The trainer builds every crystal at handedness +1
    (energies/molecular_crystal.py::MolecularCrystal.init_blank_crystal_batch) and ignores a row's stored handedness;
    the search draws handedness at random and stores the molecule mirrored on a quarter of its rows, so its rows cannot
    be read as they are (a handedness -1 row: eLJ a median ~280 raw units higher; flipping the flag alone leaves the
    mirrored rows' molecule the enantiomer). So each row is refitted: its asymmetric unit is posed as the search scored
    it (pose_aunit, std_orientation=True, with its handedness); the source molecule is fitted onto it by a proper
    rotation, or, in P-1, onto its inversion partner at -centroid, whichever it superimposes on (FIT_TOL); a centroid
    with x > 1/2 moves by -1/2 (the same crystal, translated). Returns (rows refitted onto the inversion partner, rows
    moved in x, rows no copy fitted -- which are left out by the caller, via the returned keep mask)."""
    from mxtaltools.common.geometry_utils import rotmat2rotvec
    from mxtaltools.crystal_building.utils import canonicalize_rotvec
    n = b.num_graphs
    assert bool((b.sg_ind.reshape(-1) == 2).all()) and bool((b.z_prime.reshape(-1) == 1).all()), 'P-1, Z\'=1 only'
    src = torch.cat([mol_pos[i] for i in b.identifier]).double()
    assert src.shape == b.pos.shape, 'a row and its source molecule differ in atom count'
    posed = b.clone()
    posed.pose_aunit(std_orientation=True)
    batch, heavy = b.batch, (b.z.reshape(-1) > 1).double()[:, None]

    def centred(P):
        w = torch.zeros(n, 1, dtype=torch.float64).index_add_(0, batch, heavy)
        c = torch.zeros(n, 3, dtype=torch.float64).index_add_(0, batch, P * heavy) / w
        return P - c[batch]

    X = centred(posed.pos.double())
    M = centred(src)
    R1, e1 = _proper_fit(M, X, batch, n)
    R2, e2 = _proper_fit(M, -X, batch, n)
    use1, use2 = e1 < FIT_TOL, (e1 >= FIT_TOL) & (e2 < FIT_TOL)
    keep = use1 | use2
    R = torch.where(use1[:, None, None], R1, R2)
    c = b.aunit_centroid[:, :3].double().clone()
    c[use2] = -c[use2]
    c = c - torch.floor(c)
    shift = c[:, 0] > 0.5
    c[shift, 0] -= 0.5
    rv = canonicalize_rotvec(rotmat2rotvec(R.float(), warn_on_bad_determinant=False))
    b.pos = torch.cat([mol_pos[i] for i in b.identifier]).to(b.pos.dtype)
    b.aunit_centroid = c.to(b.aunit_centroid.dtype)
    b.aunit_orientation = rv.to(b.aunit_orientation.dtype)
    b.aunit_handedness = torch.ones_like(b.aunit_handedness)
    del posed
    return keep, int(use2.sum()), int(shift.sum()), int((~keep).sum())


def _merge(batches):
    """one batch from many, by MXtalTools append_batch in a balanced tree (log2 copies, not n)"""
    batches = [b for b in batches if b is not None and b.num_graphs > 0]
    while len(batches) > 1:
        batches = [batches[i].append_batch(batches[i + 1]) if i + 1 < len(batches) else batches[i]
                   for i in range(0, len(batches), 2)]
    return batches[0]


def stage_assemble(args, audits, emb):
    # hold out distinct SMILES, as build_anchor_conditions.py does: two identifiers sharing one would otherwise put the
    # same chemistry on both sides of the split
    by_smiles = defaultdict(list)
    for m, s in zip(emb['identifiers'], emb['smiles']):
        by_smiles[s].append(m)
    keys = sorted(by_smiles)
    n_hold = int(round(args.holdout_frac * len(keys)))
    pick = torch.randperm(len(keys), generator=torch.Generator().manual_seed(args.holdout_seed))[:n_hold]
    held_ids = {m for i in pick.tolist() for m in by_smiles[keys[i]]}
    print(f"held out {len(held_ids):,} molecules ({n_hold:,} of {len(keys):,} distinct SMILES, seed "
          f"{args.holdout_seed})", flush=True)
    prior_parts, cond_parts, test_parts, test_prior_parts, prov, test_prov = [], [], [], [], [], []
    per_mol_rows = Counter()
    n_flipped = n_shifted = n_unfit = 0
    for k in range(args.n_chunks):
        d = torch.load(args.work_dir / f"chunk{k}.pt", weights_only=False)
        b, pv = d['batch'], d['provenance']
        mol_pos = {str(m.identifier): m.pos for m in torch.load(args.mol_dir / f"qm9_cluster_mols_chunk{k}.pt",
                                                                 map_location="cpu", weights_only=False)}
        keep, flipped, shifted, unfit = to_trainer_chart(b, mol_pos)
        n_flipped += flipped
        n_shifted += shifted
        n_unfit += unfit
        if unfit:
            rows = torch.nonzero(keep).flatten().tolist()
            b, pv = b.subsample_new_batch(rows), [pv[i] for i in rows]
        ids = [p[0] for p in pv]
        energy = [p[3] for p in pv]
        best = {}
        for i, (m, en) in enumerate(zip(ids, energy)):
            if m not in best or en < energy[best[m]]:
                best[m] = i
        train_rows = [i for i, m in enumerate(ids) if m not in held_ids]
        held_rows = [i for i, m in enumerate(ids) if m in held_ids]
        prior_parts.append(b.subsample_new_batch(train_rows) if train_rows else None)
        test_prior_parts.append(b.subsample_new_batch(held_rows) if held_rows else None)
        prov.extend((pv[i][0], k, pv[i][1], pv[i][2]) for i in train_rows)
        test_prov.extend((pv[i][0], k, pv[i][1], pv[i][2]) for i in held_rows)
        for i in train_rows:
            per_mol_rows[ids[i]] += 1
        cond_rows = sorted(best[m] for m in best if m not in held_ids)
        test_rows = sorted(best[m] for m in best if m in held_ids)
        cond_parts.append(b.subsample_new_batch(cond_rows) if cond_rows else None)
        test_parts.append(b.subsample_new_batch(test_rows) if test_rows else None)
        del d, b
    prior, cond, test = _merge(prior_parts), _merge(cond_parts), _merge(test_parts)
    test_prior = _merge(test_prior_parts)
    del prior_parts, cond_parts, test_parts, test_prior_parts
    print(f"trainer chart: {n_flipped:,} rows refitted onto their inversion partner, {n_shifted:,} moved by -1/2 in x, "
          f"{n_unfit:,} left out (the source molecule fitted neither copy within {FIT_TOL} A)", flush=True)

    # the trainer assumes these; check them here, where they are cheap
    p_ids, c_ids, t_ids = set(prior.identifier), set(cond.identifier), set(test.identifier)
    assert p_ids == c_ids, "every training molecule needs rows in both the prior and the conditions"
    assert not (t_ids & c_ids), "a held-out molecule is also a training condition"
    assert len(c_ids) == cond.num_graphs and len(t_ids) == test.num_graphs, "one carrier per molecule"
    assert t_ids == held_ids, f"{len(held_ids - t_ids)} held-out molecules have no kept crystal"
    assert set(test_prior.identifier) == held_ids and not (p_ids & held_ids), "held-out crystals leaked into the prior"
    flat_dim = int(emb['embedding'].shape[1] * emb['embedding'].shape[2])
    for name, bt in (('prior', prior), ('conditions', cond), ('test', test), ('test prior', test_prior)):
        assert bt.embedding.reshape(bt.num_graphs, -1).shape[1] == flat_dim, name

    mxt_commit = os.popen(f'git -C "{Path(__file__).resolve().parents[2] / "mxtaltools"}" rev-parse --short=12 HEAD'
                          ).read().strip() or 'unknown'
    gfn_commit = os.popen(f'git -C "{Path(__file__).resolve().parent}" rev-parse --short=12 HEAD').read().strip()
    meta = {
        "n_molecules": len(c_ids),
        "n_structures": int(prior.num_graphs),
        "n_holdout_molecules": len(t_ids),
        "n_holdout_structures": int(test_prior.num_graphs),
        "holdout_frac": args.holdout_frac,
        "holdout_seed": args.holdout_seed,
        "embedding_dim": flat_dim,
        "frame": "orient_molecule(mode=std) fixed point",
        "sg_ind": 2,
        "z_prime": 1,
        "energy_function": "elj",
        "anchors_source": str(args.search_dir),
        "molecules_source": str(args.mol_dir / "qm9_cluster_mols_chunk<k>.pt"),
        "encoder": emb['encoder'],
        "provenance": "anchors generated from random init by crystal_search under elj (qm9_full_sep29); "
                      "NOT experimental structures",
        "build": {"script": "build_qm9_full_prior.py", "gfn_commit": gfn_commit, "mxtaltools_commit": mxt_commit,
                  "standardize": "mxtaltools.crystal_search.standardize.standardize_cells",
                  "dup_cut_rdf_envwise": args.dup_cut, "screen_de": args.screen_de, "screen_dpc": args.screen_dpc,
                  "diverse_below_chunk": args.diverse_below, "keep": args.keep,
                  "trainer_chart": "source molecule refitted, handedness +1, centroid x in [0, 1/2] "
                                   f"(to_trainer_chart); {n_flipped} onto the inversion partner, {n_shifted} shifted, "
                                   f"{n_unfit} left out"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cond_path, prior_path = args.out_dir / f"{args.tag}_conditions.pt", args.out_dir / f"{args.tag}_prior.pt"
    test_path, prov_path = args.out_dir / f"{args.tag}_test_conditions.pt", args.out_dir / f"{args.tag}_prior_provenance.pt"
    test_prior_path = args.out_dir / f"{args.tag}_test_prior.pt"
    torch.save({"prior": cond, **meta}, cond_path)
    torch.save({"prior": prior, "equalized_prior": prior, **meta}, prior_path)
    torch.save({"prior": test, **meta, "split": "holdout"}, test_path)
    # every kept crystal of the held-out molecules, for offline evaluation; no run reads it
    torch.save({"prior": test_prior, "equalized_prior": test_prior, **meta, "split": "holdout"}, test_prior_path)
    torch.save({"columns": ("identifier", "chunk", "seed", "row"), "prior_rows": prov, "test_prior_rows": test_prov},
               prov_path)
    v = torch.tensor(sorted(per_mol_rows.values()), dtype=torch.float)
    print(f"\nprior      : {prior.num_graphs:,} rows over {len(p_ids):,} molecules "
          f"(per molecule min {int(v.min())} median {int(v.median())} max {int(v.max())})")
    print(f"conditions : {cond.num_graphs:,} carriers; held out: {test.num_graphs:,} carriers, "
          f"{test_prior.num_graphs:,} crystals")
    for pth in (cond_path, prior_path, test_path, test_prior_path, prov_path):
        print(f"wrote {pth} ({pth.stat().st_size / 1e6:.1f} MB)")
    print(f"config keys: molecules_path {cond_path.name}, prior_path {prior_path.name}, test_molecules_path "
          f"{test_path.name}; embedding_conditioning true, embedding_conditioning_dim {flat_dim}")


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    emb = stage_embed(args)
    audits = stage_chunks(args)
    tot = Counter()
    for au in audits:
        for key, val in au.items():
            if isinstance(val, (int, float)) and key not in ('chunk', 'seconds'):
                tot[key] += val
    print("\nall chunks:", dict(tot))
    if tot['dup_pairs']:
        print(f"screen recall on the fully compared chunks: {tot['dup_pairs_screened']} of {tot['dup_pairs']} "
              f"near-duplicate pairs pass the eLJ/packing screen")
    if args.chunks:
        print("--chunks given: stopping before assembly")
        return
    stage_assemble(args, audits, emb)


if __name__ == "__main__":
    main()
