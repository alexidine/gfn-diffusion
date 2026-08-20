"""
Stage C: match Nikos' structures against the landmarks on our landscapes.

Two directions, both worth reading:

  HIS -> OURS   for each of his structures, the nearest sample in each of our
                brute-force landscapes, how far away in RDF space, and how its
                energy ranks against ours.
  OURS -> HIS   which of our basins have a counterpart of his, and which do not.

Matching is done in RDF space (`compute_rdf_distance`, envwise, the same
definition used to build the priors) and then confirmed with COMPACK
(ccdc PackingSimilarity, 20-molecule shell), which is what makes "the same
structure" a defensible statement rather than a small number.

RDF is divided by z_prime and MACE energy by (sym_mult * z_prime), so both are
per-molecule and comparable across Z'. That is what lets his Z'=2/3 structures be
compared against our Z'=1 landscape at all -- and a Z'=2 structure that is really
a Z'=1 structure in a doubled cell is exactly the kind of thing this should
catch.

TWO CALIBRATIONS, both measured here rather than assumed:

  ENERGY. The MACE lattice energy currently computed by MolCrystalData does NOT
  reproduce the `mace` stored in our own prior files: recomputing a prior sample
  gives -11898.93 kJ/mol where the file says -62.80. The gap is a per-molecule
  constant -- 11836.127 kJ/mol, std 0.003 over 18 samples spanning Z'=1/2 and
  space groups 9 and 14 -- which is the signature of the atomic E0 sum being
  included in the crystal leg and omitted from the gas-phase leg. It therefore
  cancels exactly in any DIFFERENCE of two freshly computed energies, and does
  not cancel in an absolute value or against a stored one. `calibrate_energy_offset`
  measures it per run and asserts it is constant; his energies are reported on the
  stored scale so they can be read against our landscapes.

  DISTANCE. What counts as "the same structure" in RDF space is not a guess: each
  prior file carries the `log_noise_range` that `collate_prior.py` calibrated
  thermally and used as its own basin-thinning cutoff. 10**log_noise_range[1] is
  that cutoff (~0.074 for sg14_zp2), and is what `rdf_match_cut` uses here.

COVERAGE. Our searches cover sg14-Z'1, sg14-Z'2, sg9-Z'2 (sg19-Z'3 exists only as
raw chunks and is not included). His set spans 17 space groups at Z'=2/3. For
every group we did not search, "no match" means WE NEVER LOOKED THERE -- not that
we looked and missed. `matched_landscape` in the output table marks the rows
where the comparison is a real test.

Run (after ingest.py and levels.py):
    python -m energy_sampling.eval.nikos_comparison.compare --config config.yaml
"""
import argparse
import csv
import os

import torch
import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS

#: (space group, Z') combinations our brute-force searches actually cover, and
#: the prior that covers each. A row of the output table is only a real test of
#: our landscape if his structure lands in one of these.
SEARCHED = {
    (14, 1): 'sg14_zp1',
    (14, 2): 'sg14_zp2',
    (9, 2): 'sg9_zp2',
}

#: How constant the energy offset must be across sampled prior structures before
#: it may be applied as a single number, in kJ/mol. Measured std is ~0.003.
OFFSET_TOL = 0.5


def calibrate_energy_offset(prior_files, energy_function, predictor, device,
                            n_per_prior=6):
    """
    stored energy - freshly computed energy, measured across the priors.

    Must be a constant; if it is not, the two are not related by a reference
    shift and nothing here may be compared to a stored value. Returns
    (offset, per-sample values).
    """
    vals = []
    for name, path in prior_files.items():
        pb = torch.load(path, weights_only=False,
                        map_location='cpu')['prior_batch'].cpu()
        lst = pb.batch_to_list()
        idx = torch.linspace(0, len(lst) - 1, n_per_prior).long().tolist()
        sub = collate_data_list([lst[i] for i in idx], exclude_keys=[energy_function])
        with torch.no_grad():
            fresh = sub.analyze([energy_function], assign_outputs=False,
                                predictor=predictor)[energy_function]
        vals.append(pb[energy_function][idx].cpu() - fresh.cpu())
    vals = torch.cat(vals)
    offset, spread = float(vals.mean()), float(vals.std())
    print(f"energy offset (stored - recomputed): {offset:.4f} kJ/mol, "
          f"std {spread:.5f} over {len(vals)} samples")
    if spread > OFFSET_TOL:
        raise AssertionError(
            f"the stored and recomputed {energy_function} energies do not differ "
            f"by a constant (std {spread:.4f} > {OFFSET_TOL} kJ/mol); they are not "
            f"on scales that can be reconciled by a shift, so his energies cannot "
            f"be placed against our landscapes")
    return offset, vals


def ensure_rdf(batch, cache_path, device, batch_size=200):
    """
    RDFs for one of our priors, cached.

    `collate_prior.py` deletes `rdf` from the prior batches before saving, so
    this is always a recompute on first run. The cache is keyed by file and by
    size -- delete it if the prior is rebuilt.

    Deliberately NOT `adaptive_batched_analysis`: that accumulates a full Data
    object per structure (positions, symmetry operators and a 91x100 RDF each)
    and re-collates them at the end, which exhausts host RAM partway through the
    second prior. Only the RDF tensor is wanted here, so each chunk is reduced to
    its RDF and the rest dropped.
    """
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False, map_location='cpu')
        if blob['n'] == batch.num_graphs and blob['rdf_kwargs'] == RDF_KWARGS:
            print(f"   loaded cached RDFs from {os.path.basename(cache_path)}")
            return blob['rdf']
        print(f"   cache {os.path.basename(cache_path)} does not match "
              f"(n={blob['n']} vs {batch.num_graphs}) -- recomputing")

    data_list = batch.batch_to_list()
    chunks = []
    with torch.no_grad():
        for start in tqdm.tqdm(range(0, len(data_list), batch_size),
                               desc=os.path.basename(cache_path)):
            sub = collate_data_list(data_list[start:start + batch_size]).to(device)
            out = sub.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
            chunks.append(out['rdf'][0].detach().cpu().clone()
                          if isinstance(out['rdf'], (tuple, list))
                          else out['rdf'].detach().cpu().clone())
            del sub, out
    rdf = torch.cat(chunks, dim=0)
    del chunks
    if rdf.shape[0] != batch.num_graphs:
        raise AssertionError(f"computed {rdf.shape[0]} RDFs for "
                             f"{batch.num_graphs} structures")
    torch.save({'rdf': rdf, 'n': batch.num_graphs, 'rdf_kwargs': RDF_KWARGS},
               cache_path)
    return rdf


def cross_distances(query_rdf, ref_rdf, bins):
    """(n_query, n_ref) RDF distance matrix."""
    return torch.stack([compute_rdf_distance(query_rdf[i], ref_rdf, bins)
                        for i in tqdm.tqdm(range(len(query_rdf)), desc='rdf dist')])


def compack_confirm(query_batch, ref_batch, neighbour_inds, work_dir, n_cpus=8):
    """
    COMPACK each query structure against its RDF neighbours in `ref_batch`.

    `batch_compack` compares many test structures against ONE reference and
    writes `compack_*.cif` into the working directory, so this runs one call per
    query structure with that structure as the reference.

    Returns (rmsds, n_matched), both (n_query, k). An entry is 0/0 where the
    comparison failed inside the similarity engine -- that is a failure to
    compare, NOT a match of zero RMSD, and must not be read as a perfect hit.
    """
    os.makedirs(work_dir, exist_ok=True)
    cwd = os.getcwd()
    n_query, k = neighbour_inds.shape
    rmsds = torch.zeros(n_query, k)
    matched = torch.zeros(n_query, k, dtype=torch.long)
    ref_list = ref_batch.batch_to_list()
    query_list = query_batch.batch_to_list()
    try:
        os.chdir(work_dir)
        for i in tqdm.tqdm(range(n_query), desc='compack'):
            q = collate_data_list([query_list[i]])
            q.mol2ucell()
            q.write_cif(torch.arange(1), f'query_{i}', mode='unit cell')
            ref_path = os.path.abspath(f'query_{i}_0.cif')

            cand = collate_data_list([ref_list[int(j)] for j in neighbour_inds[i]])
            m, r = cand.batch_compack(ref_path, torch.arange(k), n_cpus=n_cpus)
            rmsds[i] = torch.as_tensor(r, dtype=torch.float32)
            matched[i] = torch.as_tensor(m, dtype=torch.long)
    finally:
        os.chdir(cwd)
    return rmsds, matched


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--device', default=None, help="override config device")
    ap.add_argument('--level', default='l1', choices=['l0', 'l1', 'l2'],
                    help='which level to COMPACK-confirm and tabulate matches for '
                         '(all present levels are matched in RDF space regardless)')
    ap.add_argument('--compack-k', type=int, default=5,
                    help='RDF neighbours per structure to confirm; 0 disables')
    ap.add_argument('--rdf-batch', type=int, default=200)
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    device = cli.device or cfg['device']
    ef = cfg['energy_function']

    if ef == 'mace':
        from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
        predictor = load_mace_model(cfg['mace_model'], device, torch.float32)
    else:
        raise NotImplementedError(f"energy_function {ef!r}: only 'mace' is wired up")

    #: map_location: std_acridine_polymorphs.pt carries cuda tensors, and
    #: torch.load raises outright on a machine with no GPU.
    levels = torch.load(os.path.join(cfg['out_dir'], 'nikos_levels.pt'),
                        weights_only=False, map_location='cpu')
    manifest = levels['manifest']
    if levels['rdf_kwargs'] != RDF_KWARGS:
        raise ValueError(f"levels were built with rdf_kwargs {levels['rdf_kwargs']}, "
                         f"this run uses {RDF_KWARGS}; the distances would not be "
                         f"comparable")
    available = {k: levels[k] for k in ('l0', 'l1', 'l2') if levels.get(k) is not None}
    print(f"levels present: {sorted(available)}")

    offset, _ = calibrate_energy_offset(cfg['priors'], ef, predictor, device)

    """our landscapes"""
    priors, prior_rdfs, match_cuts = {}, {}, {}
    for name, path in cfg['priors'].items():
        print(f"prior {name}")
        blob = torch.load(path, weights_only=False, map_location='cpu')
        batch = blob['prior_batch'].cpu()
        #: the thermally calibrated basin cutoff collate_prior.py thinned with
        match_cuts[name] = float(10 ** blob['log_noise_range'][1])
        cache = os.path.join(cfg['out_dir'], f'prior_rdf_{name}.pt')
        prior_rdfs[name] = ensure_rdf(batch, cache, device, cli.rdf_batch)
        priors[name] = batch
        print(f"   {batch.num_graphs} samples, sg {sorted(set(batch.sg_ind.tolist()))}, "
              f"Z' {sorted(set(batch.z_prime.tolist()))}, "
              f"{ef} min {batch[ef].min():.2f} kJ/mol, "
              f"rdf match cutoff {match_cuts[name]:.4f}")

    """the known polymorphs, as a third reference layer"""
    poly = torch.load(cfg['polymorphs'], weights_only=False,
                      map_location='cpu').cpu()
    with torch.no_grad():
        poly = adaptive_batched_analysis(poly, analyses=['rdf', ef], state={},
                                         initial_batch_size=4, max_batch_size=4,
                                         predictor=predictor, device=device,
                                         show_tqdm=False, **RDF_KWARGS)
    poly = poly.to('cpu')
    poly[ef] = poly[ef] + offset

    """match"""
    bins = torch.linspace(0, 10, next(iter(prior_rdfs.values())).shape[-1])
    results = {}
    for lname, lbatch in available.items():
        results[lname] = {}
        for pname, prdf in prior_rdfs.items():
            d = cross_distances(lbatch.rdf, prdf, bins)
            results[lname][pname] = d
            nn = d.min(dim=1).values
            print(f"{lname} vs {pname}: nearest-neighbour RDF distance "
                  f"median {nn.median():.4f}, best {nn.min():.4f}; "
                  f"{int((nn < match_cuts[pname]).sum())}/{len(nn)} within the "
                  f"{match_cuts[pname]:.4f} basin cutoff")
        results[lname]['polymorphs'] = cross_distances(lbatch.rdf, poly.rdf, bins)

    """COMPACK confirmation on the requested level"""
    compack = {}
    level = cli.level if cli.level in available else sorted(available)[-1]
    if cli.compack_k and cli.level not in available:
        print(f"COMPACK: level {cli.level} was not built; using {level} instead")
    if cli.compack_k:
        lbatch = available[level]
        for pname, prior in priors.items():
            d = results[level][pname]
            k = min(cli.compack_k, d.shape[1])
            nn = d.argsort(dim=1)[:, :k]
            print(f"COMPACK {level} vs {pname}: {lbatch.num_graphs} x {k}")
            rmsds, matched = compack_confirm(
                lbatch, prior, nn,
                os.path.join(cfg['out_dir'], '_compack', f'{level}_{pname}'))
            compack[pname] = {'neighbour_inds': nn, 'rmsd': rmsds,
                              'n_matched': matched}

    out = os.path.join(cfg['out_dir'], 'nikos_matches.pt')
    torch.save({'rdf_distances': results, 'compack': compack,
                'compack_level': level, 'energy_offset': offset,
                'match_cuts': match_cuts, 'manifest': manifest,
                'energy_function': ef, 'prior_names': list(priors)}, out)
    print(f"\nwrote {out}")

    write_table(cfg, levels, available, priors, results, compack, level,
                poly, offset, match_cuts)


def write_table(cfg, levels, available, priors, results, compack, level,
                poly, offset, match_cuts):
    """One row per structure of his: where it sits, and what it matched."""
    ef = levels['energy_function']
    manifest = levels['manifest']
    l0 = levels['l0']
    path = os.path.join(cfg['out_dir'], 'nikos_comparison.csv')
    best_ours = {n: float(b[ef].min()) for n, b in priors.items()}

    rows = []
    for i, key in enumerate(l0.identifier):
        rec = manifest[key]
        sg, zp = int(l0.sg_ind[i]), int(l0.z_prime[i])
        row = {
            'key': key,
            'file': rec['file_name'],
            'sg_label': rec['sg_label'],
            'sg_ind': sg,
            'z_prime': zp,
            'nikos_rank': rec['rank'],
            'matched_landscape': SEARCHED.get((sg, zp), ''),
            'mirror_flip': rec['mirror_flip'],
            'nonstandard_setting': rec['nonstandard_symmetry'],
            'l0_l1_rdf_gap': round(float(levels['l0_l1_rdf_gap'][i]), 5),
        }
        #: energies shifted onto the stored-prior scale, so they read against our
        #: landscapes; see calibrate_energy_offset.
        for lname, lbatch in available.items():
            row[f'{lname}_{ef}'] = round(float(lbatch[ef][i]) + offset, 3)
            row[f'{lname}_packing_coeff'] = round(float(lbatch.packing_coeff[i]), 4)
        for pname in priors:
            d = results[level][pname][i]
            j = int(d.argmin())
            row[f'{pname}_rdf_dist'] = round(float(d[j]), 5)
            row[f'{pname}_within_cut'] = bool(float(d[j]) < match_cuts[pname])
            row[f'{pname}_nn_index'] = j
            row[f'{pname}_nn_{ef}'] = round(float(priors[pname][ef][j]), 3)
            row[f'{pname}_best_ours_{ef}'] = round(best_ours[pname], 3)
            if pname in compack:
                rms = compack[pname]['rmsd'][i]
                nm = compack[pname]['n_matched'][i]
                valid = nm > 0  # 0 matched == comparison failed, not a hit
                if bool(valid.any()):
                    b = int(rms.masked_fill(~valid, float('inf')).argmin())
                    row[f'{pname}_compack_rmsd'] = round(float(rms[b]), 4)
                    row[f'{pname}_compack_nmatched'] = int(nm[b])
                else:
                    row[f'{pname}_compack_rmsd'] = ''
                    row[f'{pname}_compack_nmatched'] = 0
        dp = results[level]['polymorphs'][i]
        jp = int(dp.argmin())
        row['nearest_polymorph'] = poly.identifier[jp]
        row['nearest_polymorph_rdf_dist'] = round(float(dp[jp]), 5)
        rows.append(row)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")

    n_matched = sum(1 for r in rows if r['matched_landscape'])
    print(f"\n{n_matched} / {len(rows)} of his structures fall in an SG/Z' "
          f"combination we actually searched; for the other {len(rows) - n_matched}, "
          f"a non-match is absence of evidence, not evidence of absence")
    for pname in priors:
        hits = [r for r in rows if r.get(f'{pname}_within_cut')]
        real = [r for r in hits if r['matched_landscape'] == pname]
        print(f"  {pname}: {len(hits)} of his structures within the basin cutoff "
              f"({len(real)} of them in the matching SG/Z')")
    print("
DO NOT READ THE CUTOFF COLUMN ON ITS OWN. It is collate_prior's "
          "thermal THINNING radius, not an identity criterion for structures "
          "ingested from outside: 5 of our own 6 targeted known polymorphs also "
          "fall outside it. Run controls.py for the range that 'present in our "
          "landscape' actually spans, and compare energies only against the "
          "matching relaxation stage.")


if __name__ == '__main__':
    main()
