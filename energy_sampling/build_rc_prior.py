"""Turn the rc_oct22 search (MXtalTools configs/crystal_searches/rc_oct22: random-start eLJ searches, P-1, Z'=1, on
molecules with one random conformer each, five chunk families) into the condition, prior and held-out files a
conditional run reads, in build_qm9_full_prior.py's envelope: ``{'prior': batch}`` for the condition files (one carrier
per molecule, its lowest-energy kept crystal), ``{'prior': batch, 'equalized_prior': batch}`` for the prior, the
held-out files carrying ``"split": "holdout"``. Every row carries its molecule's frozen Mo3ENet ``embedding``, computed
from the molecule's random-conformer geometry (the geometry the search used).

A unit is one chunk of one family. It enters a build only when all --seeds of its seed runs are complete (each run's
progress file has cursor = num_samples = rows written, and the output holds that many rows); the others are listed and
wait for a later build, which reuses the work of every unit already processed (work files carry the settings that made
them, and a build with other settings refuses them).

ONE MOLECULE, ONE IDENTIFIER, ONE SIDE. A molecule is its canonical stereo-free SMILES (smiles_mixture.canon), not the
string its rows carry: the PubChem pool kept the isotopic hydrogens of deuterated entries as explicit [H] (fixed in
smiles_pool.py since), so about 43,000 pubrc strings are second spellings of molecules searched under another identifier,
QM9 molecules among them. Over the whole selection (not only the units built), the registry stage keeps one identifier
per molecule -- the earliest family's, an already canonical string first -- drops every non-qm9rc identifier of a QM9
molecule (smiles_mixture.qm9_reference), and holds a molecule out when any of its spellings is held out in a selection
file (--qm9-split for qm9rc: the qm9full prior's split; --new-split: smiles_mixture.py's salted-hash split). Sides are
never drawn here, so the held-out set does not move between builds and no qm9full held-out molecule trains.

Stages, each resumable from its files in --work-dir:
  0. registry (work/registry.pt): identifier -> kept or dropped (and why), canonical SMILES, side.
  1. units (parallel processes, one unit each, a fresh process per unit; work/<family>_c<k>.pt + _audit.json):
     - leave out every molecule whose standard frame rests on a near-zero decision (frame_margins < --frame-margin):
       the trainer re-standardizes each molecule on its own device and can land such a molecule in another frame;
     - embed the chunk's molecules (Mo3ENet, in near-equal batches of at most --embed-batch-size in chunk order: the
       embedding depends on the batch it is computed in, and a short tail batch would differ by several percent);
     - gather the seeds' crystals; drop those with a non-finite eLJ, those whose cell parameters describe no cell, and
       rows standardize_cells cannot place;
     - per molecule, in ascending eLJ, drop a crystal within --dup-cut envwise RDF distance of one already kept. Only
       crystals in a pair within --screen-de eLJ and --screen-dpc packing coefficient are compared (the rule of
       build_qm9_full_prior.py's screened chunks); every --full-rdf-every-th unit also computes every pair, to measure
       the rule's recall (audit: dup_pairs, dup_pairs_screened) without changing what it keeps. Every survivor is kept
       (no diversity cap);
     - re-describe each kept crystal in the trainer's chart (build_qm9_full_prior.to_trainer_chart: the source molecule
       refitted, handedness +1, centroid x in [0, 1/2]); a row the source molecule fits neither copy of is left out.
  2. assemble, per family: the registry's kept identifiers and sides; writes
     <tag>_<family>_{conditions,prior,test_conditions,test_prior,prior_provenance}.pt, and checks the prior as the
     trainer reads it (trainer_read_check).
  3. --combine FAMILY ...: concatenate those families' files into <tag>_{conditions,prior,test_conditions,
     test_prior}.pt, the files a run names, re-checking one identifier per molecule and disjoint sides by canonical
     SMILES. One file at a time; the combined prior of installment 1 (about 7M rows) still needs tens of GB.

CPU only. Usage (from energy_sampling/):
    python build_rc_prior.py --search-dir D:\\...\\anchors\\rc_oct22 --mol-dir D:\\...\\rc_oct22\\mols
        --work-dir D:\\...\\anchors\\rc_oct22_prior_build --out-dir D:\\...\\priors\\rc_oct22 --tag rcoct22i1 --workers 8
    python build_rc_prior.py ... --combine qm9rc csdrc zincrc gdbrc pubrc
"""
import argparse
import csv
import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

FAMILIES = ('qm9rc', 'csdrc', 'zincrc', 'gdbrc', 'pubrc')
SOURCE_FAMILY = {'csd': 'csdrc', 'zinc20A': 'zincrc', 'gdb11': 'gdbrc', 'pubchem': 'pubrc'}
DEFAULT_ENCODER = r"D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15"
QM9_SPLIT = r"D:\crystal_datasets\smiles_sources\selection\qm9_oct22\selected.tsv"
NEW_SPLIT = r"D:\crystal_datasets\smiles_sources\selection\new_oct22\selected.tsv"
QM9_XYZ = r"D:\crystal_datasets\qm9_xyz"
FILES = ('conditions', 'prior', 'test_conditions', 'test_prior')
TRAINER_READ_TOL = 0.05   # raw eLJ units; a misread row (mirrored, refitted wrong) is off by tens to hundreds


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-dir", type=Path, required=True, help="the battery's outputs, <family>_c<k>_<seed>.pt")
    p.add_argument("--mol-dir", type=Path, required=True, help="the chunk files <family>_mols_chunk<k>.pt")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", required=True, help="name the build (e.g. with its installment): run problem identity "
                                                "follows the prior's path")
    p.add_argument("--families", nargs='+', default=list(FAMILIES))
    p.add_argument("--units", nargs='+', default=None, help="family:chunk ... -- build only these (testing)")
    p.add_argument("--seeds", type=int, default=5, help="seed runs per chunk")
    p.add_argument("--qm9-split", type=Path, default=Path(QM9_SPLIT))
    p.add_argument("--new-split", type=Path, default=Path(NEW_SPLIT))
    p.add_argument("--qm9-xyz", type=Path, default=Path(QM9_XYZ), help="for the QM9 reference set")
    p.add_argument("--dup-cut", type=float, default=0.01)
    p.add_argument("--screen-de", type=float, default=1.0, help="eLJ raw units")
    p.add_argument("--screen-dpc", type=float, default=0.005)
    p.add_argument("--full-rdf-every", type=int, default=20, help="every n-th chunk also compares every pair (0: never)")
    p.add_argument("--workers", type=int, default=8, help="each unit process peaks near 4 GB")
    p.add_argument("--encoder", type=Path, default=Path(DEFAULT_ENCODER))
    p.add_argument("--embed-batch-size", type=int, default=200)
    p.add_argument("--frame-margin", type=float, default=1e-4,
                   help="drop molecules whose standard frame rests on a decision closer than this (frame_margins)")
    p.add_argument("--combine", nargs='+', default=None, metavar='FAMILY',
                   help="only concatenate these families' per-family files into the combined files")
    p.add_argument("--combined-tag", default=None, help="--combine: name of the combined files (default --tag)")
    p.add_argument("--rows-per-molecule", type=int, default=0,
                   help="--combine: keep at most this many crystals per molecule (0: all; thin_rows)")
    p.add_argument("--molecule-fraction", type=float, default=1.0,
                   help="--combine: keep the training molecules whose molecule_u is below this (the held-out files "
                        "are untouched; a smaller fraction keeps a subset of a larger one's molecules)")
    args = p.parse_args(argv)
    if not 0.0 < args.molecule_fraction <= 1.0:
        p.error("--molecule-fraction must be in (0, 1]")
    return args


def _atomic_save(obj, path):
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_text(text, path):
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_split(path):
    """{smiles: 'train' | 'held_out'} from a selection file's smiles and split columns."""
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        i, j = header.index('smiles'), header.index('split')
        return {row[i]: row[j] for row in r if row}


# ------------------------------------------------------------------------------------------------ registry
def molecule_registry(args):
    """{identifier: (kept, reason, canonical SMILES, side)} over every molecule of both selection files: one identifier
    per canonical molecule (earliest family, then an already canonical string, then the smallest string), none for a QM9
    molecule outside qm9rc, and 'held_out' for a molecule any of whose spellings is held out."""
    path = args.work_dir / "registry.pt"
    key = {str(p): os.path.getsize(p) for p in (args.qm9_split, args.new_split)}
    if path.exists():
        d = torch.load(path, weights_only=False)
        if d['key'] == key:
            return d['entries']
    from data_processing.smiles_mixture import canon, qm9_reference
    rows = [(s, 'qm9rc', side) for s, side in read_split(args.qm9_split).items()]
    with open(args.new_split, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        i, j, k = header.index('smiles'), header.index('source'), header.index('split')
        rows += [(row[i], SOURCE_FAMILY[row[j]], row[k]) for row in r if row]
    t = time.time()
    canonical = {s: canon(s) for s in {s for s, _, _ in rows}}
    held = {canonical[s] for s, _, side in rows if side == 'held_out'}
    qm9 = qm9_reference(str(args.qm9_xyz))
    best = {}
    for s, fam, _ in rows:
        c = canonical[s]
        if fam != 'qm9rc' and c in qm9:
            continue
        rank = (FAMILIES.index(fam), s != c, s)
        if c not in best or rank < best[c]:
            best[c] = rank
    entries = {}
    for s, fam, _ in rows:
        c = canonical[s]
        side = 'held_out' if c in held else 'train'
        if fam != 'qm9rc' and c in qm9:
            entries[f"{s}_c0"] = (False, 'qm9_molecule', c, side)
        elif best[c][2] != s:
            entries[f"{s}_c0"] = (False, 'second_spelling', c, side)
        else:
            entries[f"{s}_c0"] = (True, 'kept', c, side)
    _atomic_save(dict(key=key, entries=entries), path)
    n = Counter((fam, entries[f"{s}_c0"][1]) for s, fam, _ in rows)
    print(f"registry: {len(rows):,} selected strings, {len(best):,} molecules, {sum(e[0] for e in entries.values()):,} "
          f"kept identifiers; dropped {dict((f'{f}:{w}', v) for (f, w), v in sorted(n.items()) if w != 'kept')} "
          f"({time.time() - t:.0f} s)", flush=True)
    return entries


# ------------------------------------------------------------------------------------------------ units
def run_complete(search_dir, run):
    """True when the run's outputs exist and its progress file says every start was relaxed and written; False for a
    file still arriving (unreadable progress)."""
    prog = Path(search_dir) / f"{run}_progress.json"
    if not prog.exists() or not (Path(search_dir) / f"{run}.pt").exists():
        return False
    try:
        p = json.loads(prog.read_text())
    except (ValueError, OSError):
        return False
    return p['cursor'] >= p['num_samples'] and p['n_out'] == p['num_samples']


def find_units(args):
    """(complete units, incomplete units) as sorted [(family, chunk)] over the chunk files of --mol-dir."""
    done, waiting = [], []
    for fam in args.families:
        ks = sorted(int(m.group(1)) for p in args.mol_dir.glob(f"{fam}_mols_chunk*.pt")
                    if (m := re.fullmatch(rf"{fam}_mols_chunk(\d+)\.pt", p.name)))
        for k in ks:
            ok = all(run_complete(args.search_dir, f"{fam}_c{k}_{s}") for s in range(args.seeds))
            (done if ok else waiting).append((fam, k))
    if args.units:
        want = {(u.split(':')[0], int(u.split(':')[1])) for u in args.units}
        done, waiting = [u for u in done if u in want], [u for u in waiting if u in want]
    return done, waiting


def _settings(a):
    return {k: a[k] for k in ('seeds', 'dup_cut', 'screen_de', 'screen_dpc', 'encoder', 'embed_batch_size',
                              'frame_margin')}


def frame_margins(mols):
    """{identifier: margin} of the standard frame's two decisions (mxtaltools align_mol_batch_to_standard_axes: heavy
    atoms, unit mass, heavy-atom centroid): each principal axis takes the sign of its overlap with the direction to the
    farthest heavy atom. The margin is the smaller of (a) the least |overlap| of an axis with that direction and (b) the
    relative gap between the farthest and second-farthest heavy-atom distances. Near zero, the decision rests on
    rounding, and the trainer, which re-standardizes every molecule on its own device, can land on another frame: a
    near-planar heavy skeleton (the plane normal is perpendicular to every in-plane reference) then has its hydrogens
    mirrored, and the same latent builds a different crystal (measured: 67 of ~3,800 molecules of two test chunks,
    re-standardized on the GPU, moved ~2 A and scored up to 23 nats off; their margins were all below 1e-5, so the
    default --frame-margin 1e-4 keeps a tenfold distance; it leaves out 3-5% of a chunk's molecules, 5% of qm9rc's,
    nearly all with margins below 3e-5: near-planar heavy skeletons, which this frame convention cannot pin down)."""
    out = {}
    for m in mols:
        p = m.pos.double()[m.z.reshape(-1) > 1]
        if len(p) < 3:
            out[str(m.identifier)] = 0.0
            continue
        p = p - p.mean(0)
        r = p.norm(dim=1)
        rs = torch.sort(r, descending=True).values
        _, V = torch.linalg.eigh(torch.eye(3, dtype=torch.float64) * (p * p).sum() - p.T @ p)
        overlap = float((V.T @ (p[torch.argmax(r)] / r.max())).abs().min())
        out[str(m.identifier)] = min(overlap, float((rs[0] - rs[1]) / rs[0]))
    return out


def embed_unit(mols, a, path):
    """Mo3ENet embeddings of a chunk's molecules, in near-equal batches of at most embed_batch_size, chunk order."""
    ids = [str(m.identifier) for m in mols]
    if path.exists():
        d = torch.load(path, weights_only=False)
        if d['identifiers'] != ids or d.get('settings') != {k: _settings(a)[k] for k in ('encoder', 'embed_batch_size')}:
            raise SystemExit(f"{path.name} was made from other molecules or settings; use a fresh --work-dir")
        return d
    from build_anchor_conditions import MO3ENET_ATOM_TYPES, embed
    from mxtaltools.common.training_utils import load_molecule_autoencoder
    types = set(int(v) for m in mols for v in m.z.flatten().tolist())
    if types - MO3ENET_ATOM_TYPES:
        raise SystemExit(f"atom types {sorted(types - MO3ENET_ATOM_TYPES)} outside the encoder's vocabulary")
    encoder = load_molecule_autoencoder(str(a['encoder']), "cpu")
    parts = torch.arange(len(mols)).tensor_split(-(-len(mols) // a['embed_batch_size']))
    e = torch.cat([embed([mols[i] for i in p.tolist()], encoder, "cpu", len(p)) for p in parts])
    if not torch.isfinite(e).all():
        raise SystemExit(f"{path.name}: non-finite embedding values")
    out = dict(identifiers=ids, embedding=e, settings={k: _settings(a)[k] for k in ('encoder', 'embed_batch_size')})
    _atomic_save(out, path)
    return out


def process_unit(fam, k, a):
    """Writes work/<family>_c<k>.pt (the kept crystals, collated, in the trainer's chart, with embeddings, and their
    provenance) and its audit. a: dict of settings."""
    torch.set_num_threads(1)
    from build_qm9_full_prior import _metric_ok, _standardize, to_trainer_chart
    from mxtaltools.crystal_search.coordinator import compute_rdfs, rdf_distance_matrix
    from mxtaltools.dataset_utils.utils import collate_data_list
    work, stem = Path(a['work_dir']), f"{fam}_c{k}"
    out_path, audit_path = work / f"{stem}.pt", work / f"{stem}_audit.json"
    if out_path.exists() and audit_path.exists():
        au = json.loads(audit_path.read_text())
        if au.get('settings') != _settings(a):
            raise SystemExit(f"{stem} was built with other settings ({au.get('settings')}); use a fresh --work-dir")
        return au
    t0 = time.time()
    mols = torch.load(Path(a['mol_dir']) / f"{fam}_mols_chunk{k}.pt", map_location="cpu", weights_only=False)
    mol_pos = {str(m.identifier): m.pos for m in mols}
    margin = frame_margins(mols)
    unstable = {m for m, v in margin.items() if v < a['frame_margin']}
    emb = embed_unit(mols, a, work / f"{stem}_embeddings.pt")
    emb_row = {m: i for i, m in enumerate(emb['identifiers'])}
    t_embed = time.time() - t0

    by_mol = defaultdict(list)          # identifier -> [(crystal, seed, row)]
    audit = Counter()
    for s in range(a['seeds']):
        run = Path(a['search_dir']) / f"{stem}_{s}"
        rows = torch.load(f"{run}.pt", weights_only=False, map_location="cpu")
        n_out = json.loads(Path(f"{run}_progress.json").read_text())['n_out']
        if len(rows) != n_out:   # the progress file vouches for exactly n_out rows
            raise SystemExit(f"{run.name}.pt holds {len(rows)} rows, its progress file {n_out}")
        for r, c in enumerate(rows):
            audit['loaded'] += 1
            ident = str(c.identifier)
            if ident not in mol_pos:
                raise SystemExit(f"{run.name}.pt row {r}: {ident} is not a molecule of {fam}_mols_chunk{k}.pt")
            if ident in unstable:
                audit['dropped_unstable_frame_rows'] += 1
                continue
            if not bool(torch.isfinite(torch.as_tensor(c.elj)).all()):
                audit['dropped_nonfinite_energy'] += 1
                continue
            if not _metric_ok(c):
                audit['dropped_invalid_cell_parameters'] += 1
                continue
            if float(c.elj) >= 0:
                audit['unbound_elj_nonnegative_kept'] += 1
            by_mol[ident].append((c, s, r))
        del rows
    full = bool(a['full_rdf_every']) and k % a['full_rdf_every'] == 0
    kept_rows, provenance, per_mol_kept = [], [], []
    for ident, rows in by_mol.items():
        std, bad = _standardize([c for c, _, _ in rows])
        audit['dropped_standardize'] += len(bad)
        badset = set(bad)
        rows = [rw for i, rw in enumerate(rows) if i not in badset]
        if not rows:
            continue
        e = torch.tensor([float(c.elj) for c, _, _ in rows])
        pc = torch.tensor([float(c.packing_coeff) for c, _, _ in rows])
        order = torch.argsort(e)
        std, rows = [std[i] for i in order.tolist()], [rows[i] for i in order.tolist()]
        e, pc = e[order], pc[order]
        n = len(rows)
        # the rule (build_qm9_full_prior.py's screened chunks): RDFs only for crystals in a pair within the eLJ and
        # packing screen, and any two of those crystals may be duplicates. A full unit computes every RDF to measure
        # the rule's recall, but decides by the same rule, so every unit is reduced alike.
        screened = ((e[:, None] - e[None]).abs() < a['screen_de']) & ((pc[:, None] - pc[None]).abs() < a['screen_dpc'])
        screened.fill_diagonal_(False)
        need_s = screened.any(1)
        need = torch.ones(n, dtype=torch.bool) if full else need_s
        D = None
        if need.any():
            idx = torch.nonzero(need).flatten()
            R = compute_rdfs([std[i] for i in idx.tolist()], batch_size=max(1, len(idx)))
            Dn = rdf_distance_matrix(R, R)
            D = torch.full((n, n), float('inf'))
            D[idx[:, None], idx[None]] = Dn
            seen = need_s[:, None] & need_s[None]
            if full:   # recall: of all near-duplicate pairs, those the rule compares
                dup = D < a['dup_cut']
                dup.fill_diagonal_(False)
                audit['dup_pairs'] += int(dup.triu(1).sum())
                audit['dup_pairs_screened'] += int((dup & seen).triu(1).sum())
            D = torch.where(seen, D, torch.full_like(D, float('inf')))
        keep = []
        for i in range(n):   # ascending eLJ: the first of a near-duplicate group is the lowest
            if D is not None and keep and bool((D[i, keep] < a['dup_cut']).any()):
                audit['dropped_duplicate'] += 1
                continue
            keep.append(i)
        per_mol_kept.append(len(keep))
        for i in keep:
            c = std[i]
            c.embedding = emb['embedding'][emb_row[ident]].unsqueeze(0).clone()
            kept_rows.append(c)
            provenance.append((ident, rows[i][1], rows[i][2], float(e[i])))
    batch = collate_data_list(kept_rows)
    keepmask, flipped, shifted, unfit = to_trainer_chart(batch, mol_pos)
    if unfit:
        sel = torch.nonzero(keepmask).flatten().tolist()
        batch, provenance = batch.subsample_new_batch(sel), [provenance[i] for i in sel]
    probe = batch.clone()  # latent_params() canonicalizes before transforming; count on a canonicalized copy
    probe.canonicalize_zp_aunits()
    probe.canonicalize_free_axes()
    lat = probe.latent_transform(cell_params=probe.full_cell_parameters())
    lat_edge = int((lat.abs() > 1).any(1).sum())
    del probe
    _atomic_save(dict(batch=batch, provenance=provenance, settings=_settings(a)), out_path)
    audit = dict(audit, family=fam, chunk=k, molecules=len(mols), molecules_unstable_frame=len(unstable),
                 molecules_with_crystals=len(by_mol),
                 molecules_kept=len({p[0] for p in provenance}), kept=len(provenance), full_rdf=full,
                 trainer_chart_inversion_partner=flipped, trainer_chart_shifted=shifted,
                 dropped_trainer_chart_unfit=unfit, rows_outside_latent_box=lat_edge,
                 kept_per_molecule=dict(sorted(Counter(per_mol_kept).items())), settings=_settings(a),
                 seconds=round(time.time() - t0, 1), seconds_embed=round(t_embed, 1))
    _atomic_text(json.dumps(audit), audit_path)
    return audit


def stage_units(args, units):
    a = dict(work_dir=str(args.work_dir), search_dir=str(args.search_dir), mol_dir=str(args.mol_dir),
             seeds=args.seeds, dup_cut=args.dup_cut, screen_de=args.screen_de, screen_dpc=args.screen_dpc,
             full_rdf_every=args.full_rdf_every, encoder=str(args.encoder), embed_batch_size=args.embed_batch_size,
             frame_margin=args.frame_margin)
    audits = []
    # a fresh process per unit: a worker keeps a unit's peak memory (near 4 GB) otherwise
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(process_unit, f, k, a): (f, k) for f, k in units}
        for fut in as_completed(futs):
            au = fut.result()
            audits.append(au)
            print(f"  {au['family']} chunk {au['chunk']:>3}: {au['loaded']} loaded -> {au['kept']} kept over "
                  f"{au['molecules_kept']} molecules ({au.get('dropped_duplicate', 0)} duplicates, "
                  f"{au.get('dropped_standardize', 0)} not standardized, {au['dropped_trainer_chart_unfit']} unfit) "
                  f"in {au['seconds']} s [{len(audits)}/{len(units)}]", flush=True)
    return sorted(audits, key=lambda x: (FAMILIES.index(x['family']), x['chunk']))


# ------------------------------------------------------------------------------------------------ assemble
def _merge(batches):
    from build_qm9_full_prior import _merge as merge
    return merge(batches)


def _commits():
    mxt = os.popen(f'git -C "{Path(__file__).resolve().parents[2] / "mxtaltools"}" rev-parse --short=12 HEAD'
                   ).read().strip() or 'unknown'
    gfn = os.popen(f'git -C "{Path(__file__).resolve().parent}" rev-parse --short=12 HEAD').read().strip()
    return gfn, mxt


def trainer_read_check(prior, n=32, seed=0):
    """eLJ rebuilt from n random prior rows as the trainer builds them (handedness +1, analyze with
    std_orientation=False) must equal the stored eLJ: the refit into the trainer's chart is right only if it does."""
    from train import mirrored_rows
    z1, zn = mirrored_rows(prior)
    assert z1.numel() == 0 and zn.numel() == 0, f"{z1.numel() + zn.numel()} prior rows are not at handedness +1"
    k = torch.randperm(prior.num_graphs, generator=torch.Generator().manual_seed(seed))[:n]
    s = prior.subsample_new_batch(k).clone()
    s.lj_coeff = torch.ones(s.num_graphs)
    e = s.analyze(['elj'], cutoff=10, supercell_size=10, std_orientation=False)['elj'].double()
    dev = float((e - prior.elj.reshape(-1)[k].double()).abs().max())
    assert dev < TRAINER_READ_TOL, f"trainer-read eLJ is {dev:.3g} raw units from the stored eLJ"
    return dict(rows=int(k.numel()), max_abs_diff=dev, tol=TRAINER_READ_TOL)


def stage_assemble(args, units, waiting, audits, registry):
    gfn, mxt = _commits()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fam in args.families:
        ks = [k for f, k in units if f == fam]
        if not ks:
            continue
        prior_parts, cond_parts, test_parts, test_prior_parts, prov, test_prov = [], [], [], [], [], []
        per_mol_rows, dropped = Counter(), Counter()
        for k in ks:
            d = torch.load(args.work_dir / f"{fam}_c{k}.pt", weights_only=False)
            b, pv = d['batch'], d['provenance']
            ids = [p[0] for p in pv]
            assert ids == [str(i) for i in b.identifier], f"{fam} chunk {k}: provenance and rows disagree"
            for m in set(ids):
                if m not in registry:
                    raise SystemExit(f"{fam} chunk {k}: {m} is in no selection file")
                if not registry[m][0]:
                    dropped[registry[m][1]] += 1
            best = {}
            for i, m in enumerate(ids):
                if registry[m][0] and (m not in best or pv[i][3] < pv[best[m]][3]):
                    best[m] = i
            side = {m: registry[m][3] for m in best}
            train_rows = [i for i, m in enumerate(ids) if m in side and side[m] == 'train']
            held_rows = [i for i, m in enumerate(ids) if m in side and side[m] == 'held_out']
            cond_rows = sorted(best[m] for m in best if side[m] == 'train')
            test_rows = sorted(best[m] for m in best if side[m] == 'held_out')
            for parts, rows in ((prior_parts, train_rows), (test_prior_parts, held_rows), (cond_parts, cond_rows),
                                (test_parts, test_rows)):
                parts.append(b.subsample_new_batch(rows) if rows else None)
            prov.extend((pv[i][0], fam, k, pv[i][1], pv[i][2]) for i in train_rows)
            test_prov.extend((pv[i][0], fam, k, pv[i][1], pv[i][2]) for i in held_rows)
            for i in train_rows:
                per_mol_rows[ids[i]] += 1
            del d, b
        prior, cond, test, test_prior = (_merge(x) for x in (prior_parts, cond_parts, test_parts, test_prior_parts))
        del prior_parts, cond_parts, test_parts, test_prior_parts
        p_ids, c_ids, t_ids, tp_ids = (set(x.identifier) for x in (prior, cond, test, test_prior))
        assert p_ids == c_ids, "every training molecule needs rows in both the prior and the conditions"
        assert t_ids == tp_ids, "every held-out molecule needs its carrier and its crystals"
        assert len(c_ids) == cond.num_graphs and len(t_ids) == test.num_graphs, "one carrier per molecule"
        canon_of = {m: registry[m][2] for m in p_ids | t_ids}
        assert not ({canon_of[m] for m in c_ids} & {canon_of[m] for m in t_ids}), "a molecule on both sides"
        assert len({canon_of[m] for m in c_ids | t_ids}) == len(c_ids | t_ids), "one molecule under two identifiers"
        assert all(registry[m][3] == 'train' for m in c_ids) and all(registry[m][3] == 'held_out' for m in t_ids)
        flat_dim = int(prior.embedding.reshape(prior.num_graphs, -1).shape[1])
        for name, bt in (('conditions', cond), ('test conditions', test), ('test prior', test_prior)):
            assert bt.embedding.reshape(bt.num_graphs, -1).shape[1] == flat_dim, name
        check = trainer_read_check(prior)
        print(f"{fam}: eLJ rebuilt as the trainer reads the rows (handedness +1, std_orientation=False) against the "
              f"stored eLJ on {check['rows']} random prior rows: max |diff| {check['max_abs_diff']:.2e}", flush=True)
        fa = [au for au in audits if au['family'] == fam]
        meta = {
            "n_molecules": len(c_ids), "n_structures": int(prior.num_graphs),
            "n_holdout_molecules": len(t_ids), "n_holdout_structures": int(test_prior.num_graphs),
            "embedding_dim": flat_dim, "frame": "orient_molecule(mode=std) fixed point",
            "sg_ind": 2, "z_prime": 1, "energy_function": "elj",
            "family": fam, "chunks": ks, "chunks_waiting": [k for f, k in waiting if f == fam],
            "anchors_source": str(args.search_dir), "molecules_source": str(args.mol_dir / f"{fam}_mols_chunk<k>.pt"),
            "split_source": [str(args.qm9_split), str(args.new_split)],
            "molecules_dropped": dict(dropped),
            "encoder": str(args.encoder),
            "provenance": "anchors generated from random init by crystal_search under elj (rc_oct22, one random "
                          "conformer per molecule); NOT experimental structures",
            "build": {"script": "build_rc_prior.py", "gfn_commit": gfn, "mxtaltools_commit": mxt,
                      "settings": fa[0]['settings'] if fa else None, "full_rdf_every": args.full_rdf_every,
                      "diversity_cap": None,
                      "trainer_chart": "source molecule refitted, handedness +1, centroid x in [0, 1/2] "
                                       "(build_qm9_full_prior.to_trainer_chart)",
                      "trainer_read_check": check,
                      "audit": {key: sum(au.get(key, 0) for au in fa) for key in
                                ('loaded', 'kept', 'dropped_duplicate', 'dropped_standardize',
                                 'dropped_invalid_cell_parameters', 'dropped_nonfinite_energy',
                                 'dropped_trainer_chart_unfit', 'unbound_elj_nonnegative_kept',
                                 'rows_outside_latent_box', 'dup_pairs', 'dup_pairs_screened',
                                 'molecules_unstable_frame', 'dropped_unstable_frame_rows')}},
        }
        stem = args.out_dir / f"{args.tag}_{fam}"
        _atomic_save({"prior": cond, **meta}, Path(f"{stem}_conditions.pt"))
        _atomic_save({"prior": prior, "equalized_prior": prior, **meta}, Path(f"{stem}_prior.pt"))
        _atomic_save({"prior": test, **meta, "split": "holdout"}, Path(f"{stem}_test_conditions.pt"))
        _atomic_save({"prior": test_prior, "equalized_prior": test_prior, **meta, "split": "holdout"},
                     Path(f"{stem}_test_prior.pt"))
        _atomic_save({"columns": ("identifier", "family", "chunk", "seed", "row"), "prior_rows": prov,
                      "test_prior_rows": test_prov}, Path(f"{stem}_prior_provenance.pt"))
        v = torch.tensor(sorted(per_mol_rows.values()), dtype=torch.float)
        print(f"{fam}: {len(ks)} chunks; prior {prior.num_graphs:,} rows over {len(p_ids):,} molecules (per molecule "
              f"min {int(v.min())} median {int(v.median())} max {int(v.max())}); held out {len(t_ids):,} molecules, "
              f"{test_prior.num_graphs:,} crystals; dropped identifiers {dict(dropped)} -> {stem}_*.pt", flush=True)
        del prior, cond, test, test_prior


def thin_rows(b, k):
    """Row indices keeping at most k crystals per molecule: its lowest-eLJ crystal (the carrier) and k - 1 of the others
    drawn at random, seeded by the identifier, so a molecule thins the same way in every file and build."""
    import hashlib
    rows = defaultdict(list)
    for i, m in enumerate(b.identifier):
        rows[str(m)].append(i)
    e = b.elj.reshape(-1)
    keep = []
    for m, idx in rows.items():
        idx = sorted(idx, key=lambda i: float(e[i]))
        rest = idx[1:]
        g = torch.Generator().manual_seed(int(hashlib.blake2b(m.encode(), digest_size=8).hexdigest(), 16) % (2 ** 63))
        keep += [idx[0]] + [rest[j] for j in torch.randperm(len(rest), generator=g)[:k - 1].tolist()]
    return sorted(keep)


def molecule_u(m):
    """A molecule's fixed draw in [0, 1), from its identifier on a hash stream of its own (not thin_rows'): --molecule-
    fraction f keeps the training molecules with molecule_u below f, the same ones in every file and build."""
    import hashlib
    return int(hashlib.blake2b(m.encode(), digest_size=8, person=b'rc_molsub').hexdigest(), 16) / 2.0 ** 64


def combine(args):
    """The combined files a run names, from the chosen families' per-family files, one file name at a time; with
    --rows-per-molecule, the prior and held-out prior keep at most that many crystals per molecule (thin_rows); with
    --molecule-fraction, the conditions and prior keep only the training molecules that molecule_u selects."""
    from data_processing.smiles_mixture import canon
    metas, ids = [], {}
    out_tag = args.combined_tag or args.tag
    for name in ('conditions', 'test_conditions', 'prior', 'test_prior'):
        parts = []
        for fam in args.combine:
            d = torch.load(args.out_dir / f"{args.tag}_{fam}_{name}.pt", weights_only=False)
            b = d['prior']
            if args.molecule_fraction < 1.0 and name in ('conditions', 'prior'):
                sel = [i for i, m in enumerate(b.identifier) if molecule_u(str(m)) < args.molecule_fraction]
                assert sel, f"{fam} {name}: --molecule-fraction {args.molecule_fraction} keeps no molecule"
                b = b.subsample_new_batch(sel)
            if args.rows_per_molecule and name in ('prior', 'test_prior'):
                b = b.subsample_new_batch(thin_rows(b, args.rows_per_molecule))
            parts.append(b)
            if name == 'prior':
                metas.append({k: v for k, v in d.items() if k not in ('prior', 'equalized_prior')})
            del d, b
        out = _merge(parts)
        del parts
        ids[name] = list(out.identifier)
        if name in ('conditions', 'test_conditions'):   # one identifier per molecule, sides disjoint, across families
            assert len(set(ids[name])) == len(ids[name]), f"{name}: an identifier in two families"
            cs = [canon(str(s)) for s in out.smiles]
            assert len(set(cs)) == len(cs), f"{name}: one molecule under two identifiers"
            ids[name + '_canon'] = set(cs)
        if name == 'prior':
            assert set(ids['prior']) == set(ids['conditions']), "prior and conditions cover different molecules"
        if name == 'test_prior':
            assert set(ids['test_prior']) == set(ids['test_conditions'])
            assert not (ids['conditions_canon'] & ids['test_conditions_canon']), "a molecule on both sides"
            dims = {m['embedding_dim'] for m in metas}
            assert len(dims) == 1, f"the families have different embedding widths: {dims}"
        meta = {"n_molecules": sum(m['n_molecules'] for m in metas) if metas else None,
                "n_structures": sum(m['n_structures'] for m in metas) if metas else None,
                "families": list(args.combine), "rows_per_molecule_cap": args.rows_per_molecule or None,
                "molecule_fraction": args.molecule_fraction, "rows_in_file": out.num_graphs,
                "molecules_in_file": len(set(ids[name])),
                "frame": "orient_molecule(mode=std) fixed point", "sg_ind": 2,
                "z_prime": 1, "energy_function": "elj",
                "provenance": "rc_oct22 per-family files combined by build_rc_prior.py --combine"}
        if metas:
            meta.update(embedding_dim=metas[0]['embedding_dim'], encoder=metas[0]['encoder'],
                        family_builds={m['family']: {k: m[k] for k in ('chunks', 'chunks_waiting', 'n_molecules',
                                                                      'n_structures', 'build')} for m in metas})
        path = Path(f"{args.out_dir / out_tag}_{name}.pt")
        if name in ('prior', 'test_prior'):
            obj = {"prior": out, "equalized_prior": out, **meta}
        else:
            obj = {"prior": out, **meta}
        if name.startswith('test'):
            obj["split"] = "holdout"
        _atomic_save(obj, path)
        print(f"  {path.name}: {out.num_graphs:,} rows", flush=True)
        del out, obj
    print(f"combined {', '.join(args.combine)}: {len(ids['conditions']):,} training molecules, "
          f"{len(ids['prior']):,} prior rows, {len(ids['test_conditions']):,} held out; config keys: molecules_path "
          f"{out_tag}_conditions.pt, prior_path {out_tag}_prior.pt, test_molecules_path {out_tag}_test_conditions.pt; "
          f"embedding_conditioning true, embedding_conditioning_dim {metas[0]['embedding_dim']}")


def main(argv=None):
    args = parse_args(argv)
    if args.combine:
        combine(args)
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    registry = molecule_registry(args)
    units, waiting = find_units(args)
    print(f"{len(units)} complete units, {len(waiting)} waiting for runs to finish"
          + (f" (e.g. {', '.join(f'{f}:{k}' for f, k in waiting[:6])})" if waiting else ''), flush=True)
    if not units:
        raise SystemExit("no complete units")
    audits = stage_units(args, units)
    tot = Counter()
    for au in audits:
        for key, val in au.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool) and key not in ('chunk', 'seconds'):
                tot[key] += val
    print("\nall units:", dict(tot))
    if tot['dup_pairs']:
        print(f"screen recall on the fully compared units: {tot['dup_pairs_screened']} of {tot['dup_pairs']} "
              f"near-duplicate pairs pass the eLJ/packing screen")
    stage_assemble(args, units, waiting, audits, registry)


if __name__ == "__main__":
    main()
