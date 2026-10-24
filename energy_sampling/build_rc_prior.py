"""Turn the rc_oct22 search (MXtalTools configs/crystal_searches/rc_oct22: random-start eLJ searches, P-1, Z'=1, on
molecules with one random conformer each, five chunk families) into the condition, prior and held-out files a
conditional run reads, in build_qm9_full_prior.py's envelope: ``{'prior': batch}`` for the condition files (one carrier
per molecule, its lowest-energy kept crystal), ``{'prior': batch, 'equalized_prior': batch}`` for the prior, the
held-out files carrying ``"split": "holdout"``. Every row carries its molecule's frozen Mo3ENet ``embedding``, computed
from the molecule's random-conformer geometry (the geometry the search used).

A unit is one chunk of one family. It enters a build only when all --seeds of its seed runs are complete (each run's
progress file has cursor = num_samples = rows written); the others are listed and wait for a later build, which reuses
the work of every unit already processed.

Stages, each resumable from its files in --work-dir:
  1. units (parallel processes, one unit each; work/<family>_c<k>.pt + _audit.json):
     - embed the chunk's molecules (batches of --embed-batch-size in chunk order, as build_qm9_full_prior.py does);
     - gather the seeds' crystals; drop those with a non-finite eLJ, those whose cell parameters describe no cell, and
       rows standardize_cells cannot place;
     - per molecule, in ascending eLJ, drop a crystal within --dup-cut envwise RDF distance of one already kept. Only
       crystals in a pair within --screen-de eLJ and --screen-dpc packing coefficient are compared; every
       --full-rdf-every-th unit compares every pair, measuring that screen's recall (audit: dup_pairs,
       dup_pairs_screened). Every survivor is kept (no diversity cap);
     - re-describe each kept crystal in the trainer's chart (build_qm9_full_prior.to_trainer_chart: the source molecule
       refitted, handedness +1, centroid x in [0, 1/2]); a row the source molecule fits neither copy of is left out.
  2. assemble, per family: each molecule's side from the selection files, never drawn here (--qm9-split for qm9rc:
     the qm9full prior's split; --new-split for the other families: smiles_mixture.py's salted-hash split), so the
     held-out set neither depends on which chunks a build holds nor lets a qm9full held-out molecule into training;
     writes <tag>_<family>_{conditions,prior,test_conditions,test_prior,prior_provenance}.pt.
  3. --combine FAMILY ...: concatenate those families' files into <tag>_{conditions,prior,test_conditions,
     test_prior}.pt, the files a run names. A separate step: the combined prior holds every row in memory at once.

CPU only. Usage (from energy_sampling/):
    python build_rc_prior.py --search-dir D:\\...\\anchors\\rc_oct22 --mol-dir D:\\...\\rc_oct22\\mols
        --work-dir D:\\...\\anchors\\rc_oct22_prior_build --out-dir D:\\...\\priors\\rc_oct22 --tag rcoct22 --workers 12
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
DEFAULT_ENCODER = r"D:\crystal_datasets\model_checkpoints\_best_autoencoder_experiments_dev_26-09-13-48-15"
QM9_SPLIT = r"D:\crystal_datasets\smiles_sources\selection\qm9_oct22\selected.tsv"
NEW_SPLIT = r"D:\crystal_datasets\smiles_sources\selection\new_oct22\selected.tsv"
FILES = ('conditions', 'prior', 'test_conditions', 'test_prior')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-dir", type=Path, required=True, help="the battery's outputs, <family>_c<k>_<seed>.pt")
    p.add_argument("--mol-dir", type=Path, required=True, help="the chunk files <family>_mols_chunk<k>.pt")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--families", nargs='+', default=list(FAMILIES))
    p.add_argument("--units", nargs='+', default=None, help="family:chunk ... -- build only these (testing)")
    p.add_argument("--seeds", type=int, default=5, help="seed runs per chunk")
    p.add_argument("--qm9-split", type=Path, default=Path(QM9_SPLIT))
    p.add_argument("--new-split", type=Path, default=Path(NEW_SPLIT))
    p.add_argument("--dup-cut", type=float, default=0.01)
    p.add_argument("--screen-de", type=float, default=1.0, help="eLJ raw units")
    p.add_argument("--screen-dpc", type=float, default=0.005)
    p.add_argument("--full-rdf-every", type=int, default=20, help="every n-th chunk compares every pair (0: never)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--encoder", type=Path, default=Path(DEFAULT_ENCODER))
    p.add_argument("--embed-batch-size", type=int, default=200)
    p.add_argument("--combine", nargs='+', default=None, metavar='FAMILY',
                   help="only concatenate these families' per-family files into the combined files")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------ units
def run_complete(search_dir, run):
    """True when the run's outputs exist and its progress file says every start was relaxed and written."""
    prog = Path(search_dir) / f"{run}_progress.json"
    if not prog.exists() or not (Path(search_dir) / f"{run}.pt").exists():
        return False
    p = json.loads(prog.read_text())
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


def _atomic_save(obj, path):
    tmp = Path(str(path) + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def embed_unit(mols, a, path):
    if path.exists():
        return torch.load(path, weights_only=False)
    from build_anchor_conditions import MO3ENET_ATOM_TYPES, embed
    from mxtaltools.common.training_utils import load_molecule_autoencoder
    types = set(int(v) for m in mols for v in m.z.flatten().tolist())
    if types - MO3ENET_ATOM_TYPES:
        raise SystemExit(f"atom types {sorted(types - MO3ENET_ATOM_TYPES)} outside the encoder's vocabulary")
    encoder = load_molecule_autoencoder(str(a['encoder']), "cpu")
    e = embed(mols, encoder, "cpu", a['embed_batch_size'])
    if not torch.isfinite(e).all():
        raise SystemExit(f"{path.name}: non-finite embedding values")
    out = dict(identifiers=[str(m.identifier) for m in mols], embedding=e, encoder=str(a['encoder']))
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
        return json.loads(audit_path.read_text())
    t0 = time.time()
    mols = torch.load(Path(a['mol_dir']) / f"{fam}_mols_chunk{k}.pt", map_location="cpu", weights_only=False)
    mol_pos = {str(m.identifier): m.pos for m in mols}
    emb = embed_unit(mols, a, work / f"{stem}_embeddings.pt")
    emb_row = {m: i for i, m in enumerate(emb['identifiers'])}
    t_embed = time.time() - t0

    by_mol = defaultdict(list)          # identifier -> [(crystal, seed, row)]
    audit = Counter()
    for s in range(a['seeds']):
        for r, c in enumerate(torch.load(Path(a['search_dir']) / f"{stem}_{s}.pt", weights_only=False,
                                         map_location="cpu")):
            audit['loaded'] += 1
            ident = str(c.identifier)
            if ident not in mol_pos:
                raise SystemExit(f"{stem}_{s}.pt row {r}: {ident} is not a molecule of {fam}_mols_chunk{k}.pt")
            if not bool(torch.isfinite(torch.as_tensor(c.elj)).all()):
                audit['dropped_nonfinite_energy'] += 1
                continue
            if not _metric_ok(c):
                audit['dropped_invalid_cell_parameters'] += 1
                continue
            if float(c.elj) >= 0:
                audit['unbound_elj_nonnegative_kept'] += 1
            by_mol[ident].append((c, s, r))
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
    _atomic_save(dict(batch=batch, provenance=provenance), out_path)
    audit = dict(audit, family=fam, chunk=k, molecules=len(mols), molecules_with_crystals=len(by_mol),
                 molecules_kept=len({p[0] for p in provenance}), kept=len(provenance), full_rdf=full,
                 trainer_chart_inversion_partner=flipped, trainer_chart_shifted=shifted,
                 dropped_trainer_chart_unfit=unfit, rows_outside_latent_box=lat_edge,
                 kept_per_molecule=dict(sorted(Counter(per_mol_kept).items())),
                 seconds=round(time.time() - t0, 1), seconds_embed=round(t_embed, 1))
    tmp = Path(str(audit_path) + '.tmp')
    tmp.write_text(json.dumps(audit))
    os.replace(tmp, audit_path)
    return audit


def stage_units(args, units):
    a = dict(work_dir=str(args.work_dir), search_dir=str(args.search_dir), mol_dir=str(args.mol_dir),
             seeds=args.seeds, dup_cut=args.dup_cut, screen_de=args.screen_de, screen_dpc=args.screen_dpc,
             full_rdf_every=args.full_rdf_every, encoder=str(args.encoder), embed_batch_size=args.embed_batch_size)
    audits = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
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
def read_split(path):
    """{smiles: 'train' | 'held_out'} from a selection file's smiles and split columns."""
    with open(path, encoding='utf-8') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        i, j = header.index('smiles'), header.index('split')
        return {row[i]: row[j] for row in r if row}


def _merge(batches):
    from build_qm9_full_prior import _merge as merge
    return merge(batches)


TRAINER_READ_TOL = 0.05   # raw eLJ units; a misread row (mirrored, refitted wrong) is off by tens to hundreds


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


def _commits():
    mxt = os.popen(f'git -C "{Path(__file__).resolve().parents[2] / "mxtaltools"}" rev-parse --short=12 HEAD'
                   ).read().strip() or 'unknown'
    gfn = os.popen(f'git -C "{Path(__file__).resolve().parent}" rev-parse --short=12 HEAD').read().strip()
    return gfn, mxt


def stage_assemble(args, units, waiting, audits):
    splits = {'qm9rc': read_split(args.qm9_split)}
    new = read_split(args.new_split)
    gfn, mxt = _commits()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fam in args.families:
        ks = [k for f, k in units if f == fam]
        if not ks:
            continue
        side_of = splits.get(fam, new)
        prior_parts, cond_parts, test_parts, test_prior_parts, prov, test_prov = [], [], [], [], [], []
        per_mol_rows, sides = Counter(), {}
        for k in ks:
            d = torch.load(args.work_dir / f"{fam}_c{k}.pt", weights_only=False)
            b, pv = d['batch'], d['provenance']
            smiles = [str(s) for s in b.smiles]
            for ident, smi in zip(b.identifier, smiles):
                if smi not in side_of:
                    raise SystemExit(f"{fam} chunk {k}: {smi} has no side in the selection file")
                sides[str(ident)] = side_of[smi]
            ids = [p[0] for p in pv]
            assert ids == [str(i) for i in b.identifier], f"{fam} chunk {k}: provenance and rows disagree"
            best = {}
            for i, (m, en) in enumerate(zip(ids, (p[3] for p in pv))):
                if m not in best or en < pv[best[m]][3]:
                    best[m] = i
            train_rows = [i for i, m in enumerate(ids) if sides[m] == 'train']
            held_rows = [i for i, m in enumerate(ids) if sides[m] == 'held_out']
            prior_parts.append(b.subsample_new_batch(train_rows) if train_rows else None)
            test_prior_parts.append(b.subsample_new_batch(held_rows) if held_rows else None)
            prov.extend((pv[i][0], fam, k, pv[i][1], pv[i][2]) for i in train_rows)
            test_prov.extend((pv[i][0], fam, k, pv[i][1], pv[i][2]) for i in held_rows)
            for i in train_rows:
                per_mol_rows[ids[i]] += 1
            cond_rows = sorted(best[m] for m in best if sides[m] == 'train')
            cond_parts.append(b.subsample_new_batch(cond_rows) if cond_rows else None)
            test_rows = sorted(best[m] for m in best if sides[m] == 'held_out')
            test_parts.append(b.subsample_new_batch(test_rows) if test_rows else None)
            del d, b
        prior, cond, test, test_prior = (_merge(x) for x in (prior_parts, cond_parts, test_parts, test_prior_parts))
        del prior_parts, cond_parts, test_parts, test_prior_parts
        held_ids = {m for m, s in sides.items() if s == 'held_out'}
        p_ids, c_ids, t_ids = set(prior.identifier), set(cond.identifier), set(test.identifier)
        assert p_ids == c_ids, "every training molecule needs rows in both the prior and the conditions"
        assert not (t_ids & c_ids), "a held-out molecule is also a training condition"
        assert len(c_ids) == cond.num_graphs and len(t_ids) == test.num_graphs, "one carrier per molecule"
        assert t_ids == held_ids, "a held-out molecule with kept crystals has no carrier"
        assert set(test_prior.identifier) == held_ids and not (p_ids & held_ids), "held-out crystals leaked into the prior"
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
            "split_source": str(args.qm9_split if fam == 'qm9rc' else args.new_split),
            "encoder": str(args.encoder),
            "provenance": "anchors generated from random init by crystal_search under elj (rc_oct22, one random "
                          "conformer per molecule); NOT experimental structures",
            "build": {"script": "build_rc_prior.py", "gfn_commit": gfn, "mxtaltools_commit": mxt, "seeds": args.seeds,
                      "dup_cut_rdf_envwise": args.dup_cut, "screen_de": args.screen_de,
                      "screen_dpc": args.screen_dpc, "full_rdf_every": args.full_rdf_every, "diversity_cap": None,
                      "trainer_chart": "source molecule refitted, handedness +1, centroid x in [0, 1/2] "
                                       "(build_qm9_full_prior.to_trainer_chart)",
                      "trainer_read_check": check,
                      "audit": {key: sum(au.get(key, 0) for au in fa) for key in
                                ('loaded', 'kept', 'dropped_duplicate', 'dropped_standardize',
                                 'dropped_invalid_cell_parameters', 'dropped_nonfinite_energy',
                                 'dropped_trainer_chart_unfit', 'unbound_elj_nonnegative_kept',
                                 'rows_outside_latent_box', 'dup_pairs', 'dup_pairs_screened')}},
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
              f"{test_prior.num_graphs:,} crystals -> {stem}_*.pt", flush=True)


def combine(args):
    """The combined files a run names, from the chosen families' per-family files."""
    metas, parts = [], {name: [] for name in FILES}
    for fam in args.combine:
        for name in FILES:
            d = torch.load(args.out_dir / f"{args.tag}_{fam}_{name}.pt", weights_only=False)
            parts[name].append(d['prior'])
            if name == 'prior':
                metas.append({k: v for k, v in d.items() if k not in ('prior', 'equalized_prior')})
    dims = {m['embedding_dim'] for m in metas}
    assert len(dims) == 1, f"the families have different embedding widths: {dims}"
    meta = {"n_molecules": sum(m['n_molecules'] for m in metas), "n_structures": sum(m['n_structures'] for m in metas),
            "n_holdout_molecules": sum(m['n_holdout_molecules'] for m in metas),
            "n_holdout_structures": sum(m['n_holdout_structures'] for m in metas), "embedding_dim": dims.pop(),
            "frame": metas[0]['frame'], "sg_ind": 2, "z_prime": 1, "energy_function": "elj",
            "families": {m['family']: {k: m[k] for k in ('chunks', 'chunks_waiting', 'n_molecules', 'n_structures',
                                                           'split_source', 'build')} for m in metas},
            "encoder": metas[0]['encoder'], "provenance": metas[0]['provenance']}
    out = {name: _merge(parts[name]) for name in FILES}
    stem = args.out_dir / args.tag
    _atomic_save({"prior": out['conditions'], **meta}, Path(f"{stem}_conditions.pt"))
    _atomic_save({"prior": out['prior'], "equalized_prior": out['prior'], **meta}, Path(f"{stem}_prior.pt"))
    _atomic_save({"prior": out['test_conditions'], **meta, "split": "holdout"}, Path(f"{stem}_test_conditions.pt"))
    _atomic_save({"prior": out['test_prior'], "equalized_prior": out['test_prior'], **meta, "split": "holdout"},
                 Path(f"{stem}_test_prior.pt"))
    print(f"combined {', '.join(args.combine)}: {meta['n_molecules']:,} training molecules, {meta['n_structures']:,} "
          f"prior rows, {meta['n_holdout_molecules']:,} held out -> {stem}_*.pt; config keys: molecules_path "
          f"{stem.name}_conditions.pt, prior_path {stem.name}_prior.pt, test_molecules_path "
          f"{stem.name}_test_conditions.pt; embedding_conditioning true, embedding_conditioning_dim "
          f"{meta['embedding_dim']}")


def main(argv=None):
    args = parse_args(argv)
    if args.combine:
        combine(args)
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
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
    stage_assemble(args, units, waiting, audits)


if __name__ == "__main__":
    main()
