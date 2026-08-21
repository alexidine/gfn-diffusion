"""
Stage D: break each landscape into distinct structural modes, and place his
structures (and the known polymorphs) on them.

This is the paper1 "structural modes" picture adapted to a landscape that was
never thermally sampled.

WHY NOT paper1's CLUSTERING. `paper1_results/utils.py::clustering` drives
`mean_shift_density` from a KDE over the RDF distance matrix -- density is the
thing that defines a basin there, which is right for GFN samples drawn in
proportion to exp(-E/kT). Our brute-force priors are NOT that: `collate_prior.py`
ran `greedy_bottom_up_anchors2`, which deletes samples within a thermal radius of
one already kept. Local density in a thinned set measures the thinning radius,
not the free energy. Running the paper1 path here would read our own
preprocessing back out as if it were physics.

So the modes here are defined by SHAPE ALONE: average-linkage agglomerative
clustering on the precomputed RDF distance matrix, cut at a distance threshold.
No density, no kernel, no assumption about how the samples were drawn.

READ MODE SIZE CORRECTLY. Because the prior is thinned, a mode's member count is
the diversity retained in that region, NOT a population or a probability. Modes
are therefore ranked and reported by ENERGY, and `n_members` is labelled as
extent. Do not turn it into a weight.

The threshold is swept and reported rather than assumed, and two independent
checks decide whether a cut is meaningful:

  * the known polymorphs a prior was built to find must land in DIFFERENT modes
    (if they collapse together, the cut is too coarse to separate real forms);
  * a structure of his that COMPACK-confirmed against a prior sample must land in
    the SAME mode as that sample (if not, the modes do not respect identity).

Run (after compare.py, which caches the prior RDFs):
    python -m energy_sampling.eval.nikos_comparison.modes --device cpu
"""
import argparse
import csv
import os

import numpy as np
import torch
import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS, searched_combos

#: thresholds swept, in RDF distance. The known polymorphs sit 0.059-0.143 from
#: their nearest prior sample (controls.py), so a meaningful cut lives near there.
THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]

#: which known polymorphs each prior was built to find (collate_prior.py TARGETS)
PRIOR_TARGETS = {
    'sg14_zp1': ['ACRDIN04', 'ACRDIN12'],
    'sg14_zp2': ['ACRDIN07', 'ACRDIN06'],
    'sg9_zp2': ['ACRDIN05', 'ACRIDIN_VIII'],
}


def full_distance_matrix(rdf, cache_path, block=256):
    """
    N x N RDF distance matrix, cached.

    Measured on CPU: ~22 min for 12624 samples, ~43 min for 17925.
    """
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False, map_location='cpu')
        if blob['n'] == len(rdf) and blob['rdf_kwargs'] == RDF_KWARGS:
            print(f"   loaded cached distance matrix from "
                  f"{os.path.basename(cache_path)}")
            return blob['dmat']
        print(f"   cached matrix does not match (n={blob['n']} vs {len(rdf)})"
              f" -- recomputing")
    n = len(rdf)
    bins = torch.linspace(0, 10, rdf.shape[-1])
    dmat = torch.zeros(n, n, dtype=torch.float32)
    for i in tqdm.tqdm(range(n), desc=os.path.basename(cache_path)):
        dmat[i] = compute_rdf_distance(rdf[i], rdf, bins).float()
    #: symmetrise: the EMD is symmetric in principle, so any asymmetry here is
    #: numerical. Report it rather than hiding it inside the average.
    asym = float((dmat - dmat.T).abs().max())
    dmat = 0.5 * (dmat + dmat.T)
    dmat.fill_diagonal_(0.0)
    print(f"   max |d(i,j) - d(j,i)| = {asym:.2e} (symmetrised)")
    torch.save({'dmat': dmat, 'n': n, 'rdf_kwargs': RDF_KWARGS}, cache_path)
    return dmat


def build_linkage(dmat, cache_path):
    """
    Average-linkage hierarchy over the distance matrix, cached.

    One linkage supports every threshold, so the sweep costs nothing after this.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False, map_location='cpu')
        if blob['n'] == len(dmat):
            print(f"   loaded cached linkage from {os.path.basename(cache_path)}")
            return blob['Z']
    condensed = squareform(dmat.numpy().astype(np.float64), checks=False)
    Z = linkage(condensed, method='average')
    torch.save({'Z': Z, 'n': len(dmat)}, cache_path)
    return Z


def assign_external(query_rdf, prior_rdf, labels):
    """
    Put each external structure on a mode by its NEAREST prior sample.

    Nearest-member (not centroid): a mode is an extended region of shape space,
    and what we are asking is whether the structure falls inside that region at
    all, not whether it sits near its middle.
    """
    bins = torch.linspace(0, 10, prior_rdf.shape[-1])
    out = []
    for i in range(len(query_rdf)):
        d = compute_rdf_distance(query_rdf[i], prior_rdf, bins)
        j = int(d.argmin())
        out.append({'nn_index': j, 'nn_dist': float(d[j]), 'mode': int(labels[j])})
    return out


def sweep(Z, dmat, energies, poly_assign, poly_ids, prior_name):
    """n_modes vs threshold, with the polymorph-separation check at each."""
    from scipy.cluster.hierarchy import fcluster
    targets = PRIOR_TARGETS.get(prior_name, [])
    rows = []
    print(f"   {'thresh':>7s} {'modes':>6s} {'largest':>8s} "
          f"{'modes w/ >1 member':>19s}  targeted polymorphs")
    for t in THRESHOLDS:
        lab = fcluster(Z, t=t, criterion='distance')
        n_modes = len(np.unique(lab))
        sizes = np.bincount(lab)[1:]
        tgt_modes = {pid: int(lab[poly_assign[k]['nn_index']])
                     for k, pid in enumerate(poly_ids) if pid in targets}
        sep = ('separate' if len(set(tgt_modes.values())) == len(tgt_modes)
               else 'COLLAPSED')
        rows.append({'threshold': t, 'n_modes': n_modes,
                     'largest': int(sizes.max()),
                     'n_multi': int((sizes > 1).sum()),
                     'target_modes': tgt_modes, 'separated': sep})
        print(f"   {t:7.2f} {n_modes:6d} {sizes.max():8d} {int((sizes>1).sum()):19d}"
              f"  {tgt_modes} {sep}")
    return rows


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threshold', type=float, default=None,
                    help='RDF distance to cut the hierarchy at; if omitted, the '
                         'sweep is reported and the smallest threshold that keeps '
                         'the targeted polymorphs in separate modes is used')
    ap.add_argument('--priors', nargs='*', default=None,
                    help='which landscapes to decompose (default: those a '
                         'structure of his actually lands in)')
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    from scipy.cluster.hierarchy import fcluster

    levels = torch.load(os.path.join(cfg['out_dir'], 'nikos_levels.pt'),
                        weights_only=False, map_location='cpu')
    ef = levels['energy_function']
    manifest = levels['manifest']
    his = levels['l1']
    searched = searched_combos(cfg)

    #: the known polymorphs, scored and RDF'd the same way
    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    predictor = load_mace_model(cfg['mace_model'], cli.device, torch.float32)
    from energy_sampling.eval.nikos_comparison.compare import calibrate_energy_offset
    from energy_sampling.eval.nikos_comparison.controls import polymorph_batch
    offset, _ = calibrate_energy_offset(cfg['priors'], ef, predictor, cli.device)
    poly, poly_en, poly_rdf, _ = polymorph_batch(cfg['polymorphs'], ef, predictor,
                                                 cli.device, offset)
    poly_ids = list(poly.identifier)

    wanted = cli.priors or sorted({searched[(int(his.sg_ind[i]), int(his.z_prime[i]))]
                                   for i in range(his.num_graphs)})
    print(f"decomposing landscapes: {wanted}")

    summary = {}
    for pname in wanted:
        print(f"\n===== {pname} =====")
        pb = torch.load(cfg['priors'][pname], weights_only=False,
                        map_location='cpu')['prior_batch'].cpu()
        rdf = torch.load(os.path.join(cfg['out_dir'], f'prior_rdf_{pname}.pt'),
                         weights_only=False, map_location='cpu')['rdf']
        energies = pb[ef]
        dmat = full_distance_matrix(
            rdf, os.path.join(cfg['out_dir'], f'prior_dmat_{pname}.pt'))
        Z = build_linkage(dmat, os.path.join(cfg['out_dir'],
                                             f'prior_linkage_{pname}.pt'))

        poly_assign = assign_external(poly_rdf, rdf, np.zeros(len(rdf), dtype=int))
        rows = sweep(Z, dmat, energies, poly_assign, poly_ids, pname)

        if cli.threshold is not None:
            thresh = cli.threshold
        else:
            ok = [r for r in rows if r['separated'] == 'separate']
            if not ok:
                raise ValueError(
                    f"{pname}: no swept threshold keeps the targeted polymorphs "
                    f"{PRIOR_TARGETS.get(pname)} in separate modes; the mode "
                    f"decomposition cannot separate known forms and must not be "
                    f"used to judge his structures")
            thresh = max(r['threshold'] for r in ok)
        labels = fcluster(Z, t=thresh, criterion='distance')
        print(f"   chosen threshold {thresh:.2f} -> {len(np.unique(labels))} modes")

        #: mode table, ranked by ENERGY. n_members is extent, not population.
        modes = []
        for m in np.unique(labels):
            members = np.flatnonzero(labels == m)
            rep = members[int(torch.argmin(energies[members]))]
            modes.append({'mode': int(m), 'n_members': len(members),
                          'min_energy': float(energies[members].min()),
                          'rep_index': int(rep)})
        modes.sort(key=lambda r: r['min_energy'])
        print(f"   top modes by energy (n_members = RETAINED DIVERSITY, "
              f"not population):")
        print(f"      {'rank':>4s} {'mode':>5s} {'n_memb':>7s} {'min E':>8s}")
        for r, mrow in enumerate(modes[:10], 1):
            print(f"      {r:4d} {mrow['mode']:5d} {mrow['n_members']:7d} "
                  f"{mrow['min_energy']:8.2f}")

        rank_of_mode = {m['mode']: i + 1 for i, m in enumerate(modes)}
        his_assign = assign_external(his.rdf, rdf, labels)
        poly_assign = assign_external(poly_rdf, rdf, labels)

        print(f"   known polymorphs:")
        for k, pid in enumerate(poly_ids):
            a = poly_assign[k]
            tag = ' <-- targeted here' if pid in PRIOR_TARGETS.get(pname, []) else ''
            print(f"      {pid:14s} mode {a['mode']:5d} "
                  f"(energy rank {rank_of_mode[a['mode']]:3d}) "
                  f"d={a['nn_dist']:.4f}{tag}")

        summary[pname] = {'threshold': thresh, 'sweep': rows, 'modes': modes,
                          'labels': labels, 'his': his_assign,
                          'poly': poly_assign, 'poly_ids': poly_ids,
                          'rank_of_mode': rank_of_mode}

    _validate(summary, cfg, levels, searched)
    _write_table(cfg, summary, levels, searched, offset, poly_en, poly_ids)
    dst = os.path.join(cfg['out_dir'], 'nikos_modes.pt')
    torch.save({'summary': summary, 'thresholds': THRESHOLDS,
                'manifest': manifest}, dst)
    print(f"\nwrote {dst}")


def _validate(summary, cfg, levels, searched):
    """
    Does the decomposition respect identity we already confirmed?

    A structure of his that COMPACK-matched a prior sample on a full 20-molecule
    shell must land in the same mode as that sample. If it does not, the modes
    are not tracking structural identity and nothing built on them is safe.
    """
    path = os.path.join(cfg['out_dir'], 'nikos_matches.pt')
    if not os.path.exists(path):
        print("\nvalidation skipped: no nikos_matches.pt (run compare.py)")
        return
    m = torch.load(path, weights_only=False, map_location='cpu')
    his = levels['l1']
    checked = failed = 0
    for pname, cp in m.get('compack', {}).items():
        if pname not in summary:
            continue
        labels, assign = summary[pname]['labels'], summary[pname]['his']
        for i in range(len(assign)):
            nm, nn = cp['n_matched'][i], cp['neighbour_inds'][i]
            if int(nm.max()) < 20:
                continue
            j = int(nn[int(nm.argmax())])
            checked += 1
            if int(labels[j]) != assign[i]['mode']:
                failed += 1
                print(f"   MISMATCH {his.identifier[i]} assigned mode "
                      f"{assign[i]['mode']} but its 20/20 COMPACK partner "
                      f"(prior {pname} #{j}) is in mode {int(labels[j])}")
    if checked == 0:
        print("\nvalidation: no 20/20 COMPACK matches to check against")
    elif failed:
        raise AssertionError(
            f"{failed} of {checked} confirmed-identical pairs landed in different "
            f"modes; the decomposition does not respect structural identity")
    else:
        print(f"\nvalidation: all {checked} confirmed-identical pairs "
              f"(20/20 COMPACK) share a mode with their partner")


def _write_table(cfg, summary, levels, searched, offset, poly_en, poly_ids):
    """One row per structure of his: which mode of its own landscape it lands in."""
    his, ef, manifest = levels['l1'], levels['energy_function'], levels['manifest']
    path = os.path.join(cfg['out_dir'], 'nikos_modes.csv')
    rows = []
    for i, key in enumerate(his.identifier):
        rec = manifest[key]
        pname = searched[(int(his.sg_ind[i]), int(his.z_prime[i]))]
        if pname not in summary:
            continue
        a = summary[pname]['his'][i]
        rows.append({
            'key': key, 'file': rec['file_name'], 'sg_label': rec['sg_label'],
            'nikos_rank': rec['rank'], 'landscape': pname,
            'mode': a['mode'],
            'mode_energy_rank': summary[pname]['rank_of_mode'][a['mode']],
            'n_modes': len(summary[pname]['modes']),
            'dist_to_mode': round(a['nn_dist'], 5),
            'mode_min_energy': round(next(m['min_energy'] for m in
                                          summary[pname]['modes']
                                          if m['mode'] == a['mode']), 3),
            'his_energy_unrelaxed': round(float(his[ef][i]) + offset, 3),
        })
    rows.sort(key=lambda r: (r['landscape'], r['mode_energy_rank']))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")

    for pname in summary:
        sub = [r for r in rows if r['landscape'] == pname]
        if not sub:
            continue
        occupied = sorted({r['mode_energy_rank'] for r in sub})
        print(f"  {pname}: his {len(sub)} structures occupy {len(occupied)} of "
              f"{sub[0]['n_modes']} modes; energy ranks {occupied[:12]}")
        best = min(sub, key=lambda r: r['mode_energy_rank'])
        print(f"     lowest-energy mode reached: rank {best['mode_energy_rank']} "
              f"by {best['key']} (his rank {best['nikos_rank']})")


if __name__ == '__main__':
    main()
