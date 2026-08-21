"""
Stage 0: does a landscape contain the known polymorphs it should?

This is the PRECONDITION for every other stage. A landscape that cannot recover
its own known forms cannot be used to judge anyone else's structures, and the
failure is silent -- his structures simply come out "unmatched", which reads as a
statement about them rather than about the search.

Run it on every new search before reporting anything from it.

TWO THINGS THIS EXISTS TO GET RIGHT, both of which caught real errors:

  * ASK THE RAW SEARCH, NOT THE THINNED PRIOR. `collate_prior.py` keeps only
    19-25% of a search: `greedy_bottom_up_anchors2` thins on
    torch.cdist(latent_params()) -- a CELL-PARAMETER radius -- and additionally
    requires 0.55 < packing_coeff < 0.95 and energy <= min + 6kT. None of that is
    an RDF criterion, so a structure can be discarded while sitting far in RDF
    space from everything retained. `--raw` streams the search output instead.
  * COMPARE LIKE-FOR-LIKE CONFORMERS. The reference must be
    `std_opt_acridine_polymorphs.pt`, which carries the same
    `opt_acridine_conformer.pt` as every prior. The `std_` file carries an older
    conformer (aromatic C-C 1.3668 vs 1.4027 A) and silently compares a different
    molecule. config.yaml points at the right one.

RDF distance ranks candidates; COMPACK decides identity. Do NOT use
`10**log_noise_range[1]` as an RDF threshold -- it is a latent-space radius and
has nothing to say about RDF distance.

    python -m energy_sampling.eval.nikos_comparison.landscape_check --prior sg9_zp2
    python -m energy_sampling.eval.nikos_comparison.landscape_check --prior sg14_zp2 --raw
"""
import argparse
import glob
import os

import torch
import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS, searched_combos

#: how much of the 20-molecule shell must match to call it the same structure
FOUND_AT = 15


def polymorph_rdfs(path, device='cpu'):
    poly = torch.load(path, weights_only=False, map_location='cpu').cpu()
    lst = poly.batch_to_list()
    out = []
    for i in range(0, len(lst), 3):
        sub = collate_data_list([e.clone() for e in lst[i:i + 3]],
                                exclude_keys=['rdf'])
        with torch.no_grad():
            o = sub.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        out.append(o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf'])
    return poly, lst, list(poly.identifier), torch.cat(out, 0)


def stream_distances(sources, target_rdf, bins, batch=100, desc='samples'):
    """
    Distance from each target to every sample, without holding all the RDFs.

    `sources` is a list of (loader, label); a loader returns a list of crystals.
    Holding 65k RDFs at once is ~2.4 GB, so each block is reduced to distances
    and dropped.
    """
    dists = [[] for _ in range(len(target_rdf))]
    refs, n = [], 0
    for load, label in tqdm.tqdm(sources, desc=desc):
        lst = load()
        if lst is None:
            continue
        for s in range(0, len(lst), batch):
            part = lst[s:s + batch]
            b = collate_data_list([e.clone() for e in part],
                                  exclude_keys=['rdf', 'fingerprint'])
            with torch.no_grad():
                o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
            r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
            for k in range(len(target_rdf)):
                dists[k].append(compute_rdf_distance(target_rdf[k], r, bins).cpu())
            refs.extend([(label, s + j) for j in range(len(part))])
            n += len(part)
            del b, o, r
        del lst
    return [torch.cat(d) for d in dists], refs, n


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--prior', required=True,
                    help='name from config `priors`, or a chunk prefix with --raw')
    ap.add_argument('--raw', action='store_true',
                    help='stream the raw search chunks instead of the thinned '
                         'prior_batch -- the right target for a containment question')
    ap.add_argument('--chunks-dir', default=None,
                    help='where the raw chunks live (default <prior dir>/prior_chunks)')
    ap.add_argument('--run', default=None,
                    help='raw chunk prefix, e.g. may_acridine_sg14_zp2')
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--all-polymorphs', action='store_true',
                    help='test every known form, not just those in this SG/Z-prime')
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    poly_path = cfg['polymorphs']
    if os.path.basename(poly_path).startswith('std_acridine'):
        raise ValueError(
            f"config points at {os.path.basename(poly_path)}, which carries the OLD "
            f"conformer. Use std_opt_acridine_polymorphs.pt -- see config.yaml.")
    poly, plist, pids, prdf = polymorph_rdfs(poly_path)
    bins = torch.linspace(0, 10, prdf.shape[-1])

    combos = searched_combos(cfg)
    combo = next((c for c, n in combos.items() if n == cli.prior), None)
    if combo is None and not cli.all_polymorphs:
        raise ValueError(f"{cli.prior!r} has no `searched` row in config, so its "
                         f"SG/Z' is unknown; pass --all-polymorphs")
    if cli.all_polymorphs or combo is None:
        want = list(range(len(pids)))
    else:
        want = [i for i in range(len(pids))
                if (int(poly.sg_ind[i]), int(poly.z_prime[i])) == combo]
        if not want:
            print(f"no known polymorph has sg={combo[0]} Z'={combo[1]}; "
                  f"nothing to validate against")
            return
    print(f"{cli.prior}: testing {[pids[i] for i in want]}")

    if cli.raw:
        root = cli.chunks_dir or os.path.join(os.path.dirname(poly_path),
                                              'prior_chunks')
        run = cli.run or f'may_acridine_{cli.prior}'
        files = sorted(glob.glob(os.path.join(root, run + '_*.pt')))
        if not files:
            raise FileNotFoundError(f"no chunks matching {run}_*.pt in {root}")

        def loader(f):
            def _load():
                try:
                    d = torch.load(f, weights_only=False, map_location='cpu')
                except Exception as e:           # zero-byte chunks exist
                    print(f"   !! unreadable chunk {os.path.basename(f)}: "
                          f"{type(e).__name__}")
                    return None
                return d if isinstance(d, list) else d.batch_to_list()
            return _load

        sources = [(loader(f), f) for f in files]
        label = f'RAW search ({len(files)} chunks)'
    else:
        pb = torch.load(cfg['priors'][cli.prior], weights_only=False,
                        map_location='cpu')['prior_batch'].cpu()
        sources = [(lambda pb=pb: pb.batch_to_list(), 'prior')]
        label = f'thinned prior ({pb.num_graphs})'

    dists, refs, n = stream_distances([s for s in sources],
                                      prdf[want], bins, desc=label)
    print(f"scored {n} samples from the {label}")

    work = os.path.join(cfg['out_dir'], '_landscape_check')
    os.makedirs(work, exist_ok=True)
    cwd = os.getcwd()
    rows = []
    try:
        os.chdir(work)
        for k, i in enumerate(want):
            d = dists[k]
            order = d.argsort()[:cli.topk]
            ref = collate_data_list([plist[i]])
            ref.mol2ucell()
            ref.write_cif(torch.arange(1), f'ref_{pids[i]}', 'unit cell')
            refp = os.path.abspath(f'ref_{pids[i]}_0.cif')

            by_src = {}
            for j in order.tolist():
                src, idx = refs[j]
                by_src.setdefault(src, []).append(idx)
            cands = []
            for src, idxs in by_src.items():
                lst = (torch.load(src, weights_only=False, map_location='cpu')
                       if src != 'prior' else None)
                if src == 'prior':
                    pb2 = torch.load(cfg['priors'][cli.prior], weights_only=False,
                                     map_location='cpu')['prior_batch'].cpu()
                    lst = pb2.batch_to_list()
                elif not isinstance(lst, list):
                    lst = lst.batch_to_list()
                cands.extend([lst[x] for x in idxs])
            cb = collate_data_list(cands, exclude_keys=['rdf', 'fingerprint'])
            m, r = cb.batch_compack(refp, torch.arange(cb.num_graphs), n_cpus=8)
            best = int(m.argmax())
            rows.append((pids[i], float(d[order[0]]), int(m[best]), float(r[best])))
    finally:
        os.chdir(cwd)

    print(f"\n{'polymorph':14s} {'best RDF':>9s} {'COMPACK':>9s} {'RMSD':>7s}  verdict")
    for pid, dnn, nm, rm in rows:
        verdict = ('FOUND' if nm >= FOUND_AT
                   else 'partial' if nm >= 8 else 'NOT FOUND')
        print(f"{pid:14s} {dnn:9.4f} {nm:6d}/20 {rm:7.3f}  {verdict}")
    ok = all(nm >= FOUND_AT for _, _, nm, _ in rows)
    print(f"\n{cli.prior}: {'USABLE' if ok else 'DOES NOT RECOVER ITS OWN FORMS'} "
          f"-- {'results from it can be trusted' if ok else 'a non-match here says nothing about the query structure'}")
    torch.save({'prior': cli.prior, 'raw': cli.raw, 'n_scored': n, 'rows': rows},
               os.path.join(cfg['out_dir'],
                            f"landscape_check_{cli.prior}{'_raw' if cli.raw else ''}.pt"))


if __name__ == '__main__':
    main()
