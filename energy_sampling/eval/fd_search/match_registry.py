"""Assign search endpoints to the campaign registry's basins; the campaign random stream's rows as the reference.

usage, from gfn_diffusion/energy_sampling (CPU; the registry, ~0.9 GB in memory, is loaded once per process):
    python -m eval.fd_search.match_registry [--run-name NAME ...] [--reference] [--validate N] [--summary]

- ``--run-name NAME``: the endpoints of run NAME (``run_fd_search``) against the registry; writes
  ``OUT_ROOT/NAME/matches.pt`` (refused if it exists).
- ``--reference``: the campaign random stream's own rows (``shards/random_*``) labelled by the registry's hits;
  writes ``OUT_ROOT/campaign_random_reference/reference.pt`` (a new directory; refused if it exists, then read).
- ``--validate N``: N in-window campaign random rows, re-matched by this module's rule, against their registry
  labels; and their eLJ recomputed from the stored parameters and the local conformer against the shard energy.
- ``--summary``: one table over the named runs and the reference, written to the new directory
  ``OUT_ROOT/<--summary-name>/summary.json`` (default ``summary_<run names>``; refused if it exists).

New-basin labels (``n_basins + k``) are local to one ``match`` call: each arm, and each 1,000-row block of
``relabel_reference``, numbers its own new basins from ``n_basins``, so a new-basin label names nothing across arms
or blocks; only counts of new basins compare.

The identity rule is the campaign's (``crystal_search.coordinator`` at HEAD, read only; the functions are those of
6e9c2ba9 except for docstrings): a row is a crystal if ``physical`` (lj below 1e4, angular factor above 0.1); a
crystal within ``window_kT`` (coord.yaml, 3 kT) of the reference (the lowest basin energy, -670.5643) is rebuilt
from its parameters and the campaign conformer (``rebuild_crystals``), re-expressed in its reduced cell
(``standardize.standardize_cells``), its atomwise RDF computed (``compute_rdfs``) and assigned to the nearest
registry leader within ``identity_cut`` (``rdf_distance_matrix``). Rows nearer no leader than the cut open new
basins among themselves by the campaign's leader clustering in row order (``_assign``); a new basin is labelled
``n_basins + k``. Bands: E <= reference + w kT for w in coord.yaml's ``bands_kT`` (2 kT, 1 kT).

Reference labels: the registry records, per shard, one hit per admitted row in row order (the shard's physical rows
at or below the admission window's edge at ingestion); each shard's hits are matched to its lowest-energy physical
rows, as many as the hits, whose energies must equal the hits' in row order (``coordinator.backfill_lineage``'s
rule). Every row within the final window was admitted (the edge only fell).
"""
import argparse
import gc
import glob
import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch

from eval.fd_search import campaign

REF_NAME = 'campaign_random_reference' if campaign.CAMPAIGN == 'nehzor' else f'campaign_random_reference_{campaign.CAMPAIGN}'


def load_registry():
    return torch.load(os.path.join(campaign.CAMPAIGN_DIR, 'registry.pt'), weights_only=False)


def _ns(cc):
    return SimpleNamespace(mol_path=campaign.LOCAL_MOL, sg=int(cc['sg']), z_prime=int(cc['z_prime']))


def energy_reference(reg, cc):
    return float(cc['energy_ref']) if cc.get('energy_ref') is not None else float(min(reg['basin_E']))


#: leaders per call of ``coordinator.rdf_distance_matrix``: it copies its second argument twice (normalised CDFs),
#: so the 3,157 leaders (380 MB) at once would add about 1.1 GB; a pair's distance does not depend on the blocking
LEADER_BLOCK = 512
#: rows rebuilt and analysed at once, and crystals per RDF analysis call. The campaign used rdf_batch 64; the RDFs do
#: not depend on it (checked bit for bit, 16 against 64). Measured 2026-10-26: compute_rdfs at batch 64 took a
#: process from 1.3 to 3.7 GB on 300 crystals; at 16, in blocks of 128 rebuilt rows, it levelled at 2.1 GB
#: (without the registry, about 0.45 GB more)
ROW_BLOCK = 128
RDF_BATCH = 16


def nearest_leader(rdfs, leaders, block: int = LEADER_BLOCK):
    """(distance [n], index [n]) of each RDF's nearest leader by ``coordinator.rdf_distance_matrix``, over blocks of
    leaders; ties keep the lower index, as ``min`` over the full matrix does."""
    from mxtaltools.crystal_search import coordinator
    bd = torch.full((len(rdfs),), float('inf'))
    bj = torch.zeros(len(rdfs), dtype=torch.long)
    for lo in range(0, len(leaders), block):
        d = coordinator.rdf_distance_matrix(rdfs, leaders[lo:lo + block])
        v, j = d.min(1)
        take = v < bd
        bd = torch.where(take, v, bd)
        bj = torch.where(take, j + lo, bj)
    return bd, bj


def match(reg, cc, params, hand, E, lj, log=print):
    """Basin labels of rows (params [n, 12], hand [n, 1], eLJ E [n], lj [n]) by the campaign's identity rule.

    Returns a dict of [n] tensors: physical, in_window, basin (registry index; new basins n_basins + k; -1 outside
    the window or not physical), nearest (nearest leader), dist (its RDF distance; NaN outside the window), new (a
    new basin), std_ok (standardize_cells handled the row)."""
    from mxtaltools.crystal_search import coordinator
    from mxtaltools.crystal_search.standardize import standardize_cells
    from mxtaltools.dataset_utils.utils import collate_data_list
    params, hand = params.float(), hand.float().reshape(len(params), -1)
    E, lj = E.float(), lj.float()
    n = len(E)
    ref, kT, cut = energy_reference(reg, cc), float(cc['kT']), float(cc['identity_cut'])
    phys = coordinator.physical(params, lj)
    win = phys & (E <= ref + float(cc['window_kT']) * kT)
    idx = torch.nonzero(win).flatten()
    nb = len(reg['basin_E'])
    basin = torch.full((n,), -1, dtype=torch.long)
    nearest = torch.full((n,), -1, dtype=torch.long)
    dist = torch.full((n,), float('nan'))
    new = torch.zeros(n, dtype=torch.bool)
    std_ok = torch.zeros(n, dtype=torch.bool)
    if len(idx):
        bd = torch.empty(len(idx))
        bj = torch.empty(len(idx), dtype=torch.long)
        far = {}  # RDFs of rows nearer no leader than the cut (candidates for new basins)
        for lo in range(0, len(idx), ROW_BLOCK):  # blocks bound the RDF analysis's memory (see ROW_BLOCK)
            j = idx[lo:lo + ROW_BLOCK]
            crystals = coordinator.rebuild_crystals(_ns(cc), params[j], hand[j])
            std, info = standardize_cells(collate_data_list(crystals), on_failure='flag')
            std_ok[j] = torch.as_tensor(info['ok'])
            rdfs = coordinator.compute_rdfs(std.batch_to_list(), RDF_BATCH, cc['rdf_mode'])
            del crystals, std
            gc.collect()
            bd[lo:lo + len(j)], bj[lo:lo + len(j)] = nearest_leader(rdfs, reg['leaders'])
            for k in torch.nonzero(bd[lo:lo + len(j)] >= cut).flatten().tolist():
                far[lo + k] = rdfs[k].clone()
        nearest[idx], dist[idx] = bj, bd
        new_leaders = []  # rdf rows of new basins, in opening order (leader clustering, as coordinator._assign)
        for k, i in enumerate(idx.tolist()):
            if bd[k] < cut:
                basin[i] = int(bj[k])
                continue
            if new_leaders:
                dn = coordinator.rdf_distance_matrix(far[k][None], torch.stack(new_leaders))[0]
                j = int(dn.argmin())
                if dn[j] < cut:
                    basin[i], new[i] = nb + j, True
                    continue
            new_leaders.append(far[k])
            basin[i], new[i] = nb + len(new_leaders) - 1, True
        log(f'  matched {len(idx)} in-window rows of {n}: {int((~new[idx]).sum())} to registry basins, '
            f'{int(new.sum())} to {len(new_leaders)} new basins')
    return dict(physical=phys, in_window=win, basin=basin, nearest=nearest, dist=dist, new=new, std_ok=std_ok,
                energy_ref=ref, kT=kT, identity_cut=cut)


# ---------------------------------------------------------------------------
# the campaign random stream's rows
# ---------------------------------------------------------------------------

def campaign_reference(reg, cc, stream='random', log=print):
    """Every row of the stream's shards with its registry label (-1: not admitted). Returns a dict of arrays."""
    from mxtaltools.crystal_search import coordinator
    h = reg['hits']
    groups = {}
    for i, (st, run, cur) in enumerate(zip(h['stream'], h['run'], h['cursor'])):
        if st == stream:
            groups.setdefault((run, int(cur)), []).append(i)
    out = {k: [] for k in ('run', 'cursor', 'row', 'E', 'lj', 'basin', 'row_evals_share')}
    params, hand = [], []
    n_bad = 0
    files = sorted(glob.glob(os.path.join(campaign.CAMPAIGN_DIR, 'shards', f'{stream}_*', '*.pt')))
    runs = sorted({os.path.basename(os.path.dirname(f)) for f in files})
    for f in files:
        s = torch.load(f, weights_only=False)
        E, lj = s['energy'].float(), s['lj'].float()
        n = len(E)
        lab = torch.full((n,), -1, dtype=torch.long)
        hits = groups.get((s['run'], int(s['cursor'])), [])
        if hits:
            phys = torch.nonzero(coordinator.physical(s['params'], s['lj'])).flatten()
            e = E[phys]
            ok = len(hits) <= len(phys)
            if ok:
                edge = torch.sort(e).values[len(hits) - 1]
                adm = phys[e <= edge]
                ok = len(adm) == len(hits) and [float(x) for x in E[adm].tolist()] == [h['energy'][j] for j in hits]
            if ok:
                lab[adm] = torch.tensor([h['basin'][j] for j in hits])
            else:
                n_bad += len(hits)
        out['run'].append(torch.full((n,), runs.index(s['run'])))
        out['cursor'].append(torch.full((n,), int(s['cursor'])))
        out['row'].append(torch.arange(n))
        out['E'].append(E)
        out['lj'].append(lj)
        out['basin'].append(lab)
        out['row_evals_share'].append(torch.full((n,), float(s['row_evals']) / max(1, int(s['n_relaxations']))))
        params.append(s['params'].float())
        hand.append(s['handedness'].float())
    ref = {k: torch.cat(v) for k, v in out.items()}
    ref.update(params=torch.cat(params), hand=torch.cat(hand), runs=runs, unmatched_hits=n_bad,
               n_hits=sum(len(v) for v in groups.values()))
    log(f'  reference: {len(ref["E"])} rows from {len(files)} shards, {ref["n_hits"]} registry hits, '
        f'{n_bad} hits not matched to rows')
    return ref


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def rarefied_basins(labels: np.ndarray, N: int, n: int) -> float:
    """Expected number of distinct basins among n rows drawn without replacement from N rows whose in-band labels
    are ``labels`` (one per in-band row; out-of-band rows carry no basin): sum_b 1 - C(N - n_b, n) / C(N, n)."""
    if n >= N:
        return float(len(np.unique(labels)))
    _, counts = np.unique(labels, return_counts=True)
    lg = math.lgamma
    base = lg(N + 1) - lg(n + 1) - lg(N - n + 1)
    tot = 0.0
    for c in counts:
        if N - c < n:
            tot += 1.0
        else:
            tot += 1.0 - math.exp(lg(N - c + 1) - lg(n + 1) - lg(N - c - n + 1) - base)
    return tot


def arm_stats(E, basin, new, row_evals, ref_E, kT, bands, phys):
    """Per-band in-band share (with its binomial standard error), distinct basins, new basins. A row is in a band
    if it is a crystal (``coordinator.physical``; the campaign counts only admitted rows) within the band."""
    n = len(E)
    out = dict(n=n, row_evals_per_relaxation=float(row_evals.double().mean()) if row_evals is not None else None)
    for name, w in bands.items():
        m = phys & (E <= ref_E + w * kT)
        k = int(m.sum())
        p = k / n
        lab = basin[m & (basin >= 0)]
        out[name] = dict(in_band=k, share=p, share_se=math.sqrt(max(p * (1 - p), 1e-12) / n),
                         basins=int(len(torch.unique(lab))), new_basins=int(len(torch.unique(basin[m & new]))),
                         new_rows=int((m & new).sum()))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _load_run(name):
    d = campaign.run_dir(name)
    files = sorted(glob.glob(os.path.join(d, 'vb_*.pt')))
    if not files:
        raise FileNotFoundError(f'no vb_*.pt in {d}')
    recs = [torch.load(f, weights_only=False) for f in files]
    cat = lambda k: torch.cat([r[k] for r in recs])  # noqa: E731
    return d, recs, dict(params=cat('params'), hand=cat('hand'), E=cat('elj'), lj=cat('lj'),
                         row_evals=cat('row_evals'), arm=recs[0]['arm'],
                         wall=sum(r['wall'] for r in recs),
                         evals_run=sum(sum(r['evals_run']) for r in recs))


def _reference(reg, cc, log):
    path = os.path.join(campaign.OUT_ROOT, REF_NAME, 'reference.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    d = campaign.new_run_dir(REF_NAME)
    ref = campaign_reference(reg, cc, log=log)
    torch.save(ref, path)
    log(f'  wrote {path}')
    return ref


def relabel_reference(reg, cc, ref, log=print, block=1000):
    """The reference rows within the widest band, labelled by this module's rule (nearest final leader within the
    cut), as the arms are; the registry's own labels follow arrival order (a row joined the nearest leader existing
    when it was ingested). Written once to ``reference_nearest.pt`` beside ``reference.pt``, then read."""
    import time
    path = os.path.join(campaign.OUT_ROOT, REF_NAME, 'reference_nearest.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    ref_E, kT = energy_reference(reg, cc), float(cc['kT'])
    band = max(float(v) for v in cc['bands_kT'].values())
    idx = torch.nonzero(ref['E'] <= ref_E + band * kT).flatten()
    lab = torch.full((len(ref['E']),), -1, dtype=torch.long)
    dist = torch.full((len(ref['E']),), float('nan'))
    new = torch.zeros(len(ref['E']), dtype=torch.bool)
    t0 = time.time()
    for lo in range(0, len(idx), block):
        j = idx[lo:lo + block]
        m = match(reg, cc, ref['params'][j], ref['hand'][j], ref['E'][j], ref['lj'][j], log=lambda *a: None)
        lab[j], dist[j], new[j] = m['basin'], m['dist'], m['new']
        log(f'  relabel: {min(lo + block, len(idx))}/{len(idx)} rows, {time.time() - t0:.0f}s')
    out = dict(basin=lab, dist=dist, new=new, band_kT=band, n_rows=len(idx),
               agreement=float((lab[idx] == ref['basin'][idx]).double().mean()), wall_s=time.time() - t0)
    torch.save(out, path)
    log(f'  wrote {path}: agreement with the registry labels {out["agreement"]:.4f}')
    return out


def validate(reg, cc, ref, n, seed=0, log=print):
    """Re-match n in-window labelled campaign rows; recompute their eLJ. Returns a dict of agreement figures."""
    from mxtaltools.crystal_search import coordinator
    from mxtaltools.dataset_utils.utils import collate_data_list
    lab = ref['basin']
    cand = torch.nonzero(lab >= 0).flatten()
    g = torch.Generator().manual_seed(seed)
    pick = cand[torch.randperm(len(cand), generator=g)[:n]]
    m = match(reg, cc, ref['params'][pick], ref['hand'][pick], ref['E'][pick], ref['lj'][pick], log=log)
    agree = (m['basin'] == lab[pick])
    inw = m['in_window']
    # eLJ from the stored parameters and the local conformer, analysed as the stage's final analysis does
    crystals = coordinator.rebuild_crystals(_ns(cc), ref['params'][pick], ref['hand'][pick])
    es = []
    for lo in range(0, len(crystals), RDF_BATCH):
        b = collate_data_list([c.clone() for c in crystals[lo:lo + RDF_BATCH]])
        with torch.no_grad():
            b.analyze(computes=['lj', 'elj'], cutoff=10, rdf_cutoff=10, supercell_size=10, return_cluster=False,
                      repulsion=1.0, surface_padding=0, assign_outputs=True, elementwise=False, atomwise=True,
                      bins=100)
        es.append(b.elj.detach().double().reshape(-1))
    e = torch.cat(es)
    dE = (e - ref['E'][pick].double()).abs()
    return dict(n=len(pick), n_in_window=int(inw.sum()), basin_agreement_in_window=float(agree[inw].double().mean()),
                n_disagree_in_window=int((~agree & inw).sum()),
                disagree_dist=[round(float(x), 4) for x in m['dist'][~agree & inw].tolist()][:20],
                disagree_E_above_ref_kT=[round((float(x) - m['energy_ref']) / m['kT'], 2)
                                         for x in ref['E'][pick][~agree & inw].tolist()][:20],
                elj_abs_diff_max=float(dE.max()), elj_abs_diff_median=float(dE.median()),
                elj_rel_diff_max=float((dE / ref['E'][pick].double().abs()).max()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-name', action='append', default=[])
    ap.add_argument('--reference', action='store_true')
    ap.add_argument('--validate', type=int, default=0)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--relabel-reference', action='store_true',
                    help="label the reference's in-band rows by the arms' rule (nearest final leader)")
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--summary-name', default=None,
                    help="the new directory OUT_ROOT/<summary-name>/ the table is written to as summary.json "
                         "(default summary_<run names>; an existing directory is refused before any matching)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    def log(*a):
        print(*a, flush=True)

    writes_table = bool(args.run_name) or args.summary or args.reference
    summary_name = args.summary_name or f'summary_{"_".join(args.run_name) or "reference"}'
    if writes_table and os.path.exists(os.path.join(campaign.OUT_ROOT, summary_name)):
        raise FileExistsError(f'{os.path.join(campaign.OUT_ROOT, summary_name)} exists; pass another --summary-name')

    cc = campaign.coord_config()
    reg = load_registry()
    if reg['identity_cut'] != cc['identity_cut'] or reg['rdf_mode'] != cc['rdf_mode']:
        raise ValueError('registry and coord.yaml disagree on the identity rule')
    ref_E, kT = energy_reference(reg, cc), float(cc['kT'])
    bands = {k: float(v) for k, v in cc['bands_kT'].items()}
    log(f'registry: {len(reg["basin_E"])} basins, reference {ref_E:.4f}, kT {kT:.4f}, cut {cc["identity_cut"]} '
        f'({cc["rdf_mode"]})')
    ref = _reference(reg, cc, log) if (args.reference or args.validate or args.summary
                                       or args.relabel_reference) else None
    near_path = os.path.join(campaign.OUT_ROOT, REF_NAME, 'reference_nearest.pt')
    near = relabel_reference(reg, cc, ref, log) if args.relabel_reference else (
        torch.load(near_path, weights_only=False) if os.path.exists(near_path) else None)
    if args.validate:
        v = validate(reg, cc, ref, args.validate, log=log)
        log('validation: ' + json.dumps(v))
    table = []
    for name in args.run_name:
        d, recs, run = _load_run(name)
        mpath = os.path.join(d, 'matches.pt')
        vb_files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(d, 'vb_*.pt')))]
        if os.path.exists(mpath):
            m = torch.load(mpath, weights_only=False)
            if m.get('vb_files', vb_files) != vb_files or len(m['basin']) != len(run['E']):
                raise ValueError(f'{mpath} was made from other virtual batches than {d} now holds; '
                                 f'match this run under a new name')
            log(f'{name}: read {mpath}')
        else:
            log(f'{name}: matching {len(run["E"])} endpoints')
            m = match(reg, cc, run['params'], run['hand'], run['E'], run['lj'], log=log)
            m['vb_files'] = vb_files
            torch.save(m, mpath)
        st = arm_stats(run['E'], m['basin'], m['new'], run['row_evals'], ref_E, kT, bands, m['physical'])
        st.update(name=name, arm=run['arm'], s_per_relaxation=run['wall'] / len(run['E']),
                  evals_run_per_relaxation=run['evals_run'] / len(run['E']))
        table.append(st)
    labs = []
    if ref is not None and (args.summary or args.reference):
        from mxtaltools.crystal_search import coordinator
        E, N = ref['E'], len(ref['E'])
        rphys = coordinator.physical(ref['params'], ref['lj'])
        labs = [('registry labels', ref['basin'])]
        if near is not None:  # the arms' rule; preferred for the comparison (ref@n)
            labs.append(('nearest-leader labels', near['basin']))
        for what, lab in labs:
            rs = arm_stats(E, lab, torch.zeros_like(lab, dtype=torch.bool), ref['row_evals_share'], ref_E, kT, bands,
                           rphys)
            rs.update(name=f'campaign random ({what})', arm='latent (campaign)')
            table.append(rs)
        what, lab = labs[-1]
        for st in table:
            if st['arm'] == 'latent (campaign)':
                continue
            for b, w in bands.items():
                m = (E <= ref_E + w * kT) & (lab >= 0)
                st[b]['reference_basins_at_n'] = rarefied_basins(lab[m].numpy(), N, st['n'])
            st['reference_labels'] = what
    if table:
        log('\nPer arm: n relaxations; s/relax (CPU, 4 threads; ours) ; row-evals/relax (campaign accounting); '
            'in-band share +- SE and distinct basins per band; ref@n = expected distinct basins of the campaign '
            f'random stream rarefied to the same number of relaxations ({labs[-1][0] if labs else "-"}).')
        for st in table:
            parts = [f"{st['name']:<28} {st['arm']:<18} n={st['n']:>7}"]
            if st.get('s_per_relaxation') is not None:
                parts.append(f"s/relax={st['s_per_relaxation']:.2f}")
            parts.append(f"evals/relax={st['row_evals_per_relaxation']:.1f}")
            for b in bands:
                x = st[b]
                s = f"{b}: {100 * x['share']:.1f}+-{100 * x['share_se']:.1f}% basins={x['basins']}"
                if 'reference_basins_at_n' in x:
                    s += f" (ref@n {x['reference_basins_at_n']:.1f})"
                if x.get('new_basins'):
                    s += f" new={x['new_basins']}"
                parts.append(s)
            log('  '.join(parts))
        sdir = campaign.new_run_dir(summary_name)  # refuses an existing directory
        with open(os.path.join(sdir, 'summary.json'), 'x') as fh:
            json.dump(table, fh, indent=1, default=float)
        log(f'wrote {os.path.join(sdir, "summary.json")}')


if __name__ == '__main__':
    main()
