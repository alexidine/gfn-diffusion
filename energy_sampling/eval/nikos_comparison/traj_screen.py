"""
Did any UNSEEDED trajectory pass through a known form mid-flight?

Endpoints cannot separate "went near it and kept going" from "never went near it",
and those mean different things: the first says the basin is reachable but not
retained, the second says random starts do not funnel there at all. Only the
trajectories can tell them apart, and `save_trajs` wrote them for a subset of arms.

TWO-STAGE, because ~5M states is too many to reconstruct:

  1. ENERGY, from the stored `mace` array -- no reconstruction at all. A state
     cannot BE ACRDIN07 without scoring like ACRDIN07. Energy is also invariant to
     the latent representation, which latent distance is NOT: standardised
     orientation mirrors ~60% of the time, so identical structures can sit far
     apart in latent space and a latent screen would miss them.
  2. RDF on the survivors, rebuilt via `set_cell_parameters` from each file's own
     `base_crystal`.

Stage 1 is NECESSARY, NOT SUFFICIENT -- many structures share an energy -- so its
count is an upper bound on near-misses and stage 2 decides.

The comparison is against `--closest`, the best RDF distance the ENDPOINTS reached
(0.1333 for ACRDIN07, 0.1391 for ACRDIN06 across the unseeded modes). If the
trajectories get no closer than the endpoints, random starts never approached these
basins. If they get much closer, the basin was visited and left.

    python -m energy_sampling.eval.nikos_comparison.traj_screen
"""
import argparse
import glob
import os
import re

import torch
import tqdm

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, known_form_rdfs)

#: the forms' own scores on the search scale, calibrated in-run (offset 11836.1328)
FORM_E = {'ACRDIN07': -59.70, 'ACRDIN06': -58.33}
#: best the ENDPOINTS reached, unseeded modes only
ENDPOINT_BEST = {'ACRDIN07': 0.1333, 'ACRDIN06': 0.1391}
#: a confirmed 20/20 match sits at RDF 0.044-0.049 and a non-match at 0.13+, so
#: differences of a few thousandths up here carry no meaning. Report "closer" only
#: past this, or a 0.0016 wobble reads as a result.
MATERIAL = 0.01
#: where a real match lands, for scale in the output
MATCH_RANGE = '0.044-0.049'

#: The band is CALIBRATED AGAINST A KNOWN MATCH, not chosen for tidiness. The
#: seeded outputs that COMPACK confirms at 20/20 (RMSD 0.165 / 0.144) score -60.34
#: and -59.15 against form energies of -59.70 and -58.33 -- offsets of 0.64 and
#: 0.82 kJ/mol. A structure can therefore BE a form and still sit ~0.8 away in
#: energy, because the optimiser relaxes it within its own basin.
#:
#: The first version of this screen used 0.5 and reported "no trajectory came
#: close". That band EXCLUDES both confirmed matches, so the negative result was
#: guaranteed by construction and meant nothing. Any change to this number must be
#: re-checked against those two offsets.
CALIBRATION_OFFSETS = {'ACRDIN07': 0.64, 'ACRDIN06': 0.82}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'opt_outs'))
    ap.add_argument('--tol', type=float, default=2.0,
                    help='kJ/mol band around each form; must exceed every value '
                         'in CALIBRATION_OFFSETS or the screen is blind')
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--skip-seeded', action='store_true', default=True)
    cli = ap.parse_args()

    worst = max(CALIBRATION_OFFSETS.values())
    if cli.tol <= worst:
        raise SystemExit(
            f"--tol {cli.tol} is at or below the largest confirmed-match energy "
            f"offset ({worst}); this screen would reject a structure already known "
            f"to be a 20/20 COMPACK match. Raise it.")
    print(f"energy band +/-{cli.tol} kJ/mol; confirmed matches sit at "
          f"{CALIBRATION_OFFSETS} so the band admits them")

    names, form_rdf = known_form_rdfs(list(FORM_E))
    bins = None
    best = {n: (float('inf'), None) for n in names}
    n_states = n_band = 0

    files = sorted(glob.glob(os.path.join(cli.out_dir, 'aug21*_traj*.pt')))
    if cli.skip_seeded:
        files = [f for f in files if 'seed' not in os.path.basename(f)]
    print(f"{len(files)} unseeded trajectory files")

    for p in tqdm.tqdm(files, desc='traj'):
        rec = torch.load(p, weights_only=False, map_location='cpu')
        if not (isinstance(rec, dict) and 'params' in rec and 'base_crystal' in rec):
            continue
        e, pr = rec['mace'], rec['params']          # (steps, batch), (steps, batch, n)
        n_states += int(e.numel())
        m = torch.zeros_like(e, dtype=torch.bool)
        for v in FORM_E.values():                    # UNION of the two bands
            m |= (e - v).abs() < cli.tol
        m &= torch.isfinite(e)
        idx = m.nonzero()
        n_band += len(idx)
        if not len(idx):
            del rec
            continue
        sel = pr[idx[:, 0], idx[:, 1]]               # (n_sel, n_params)
        base = rec['base_crystal']
        for s in range(0, len(sel), cli.chunk):
            part = sel[s:s + cli.chunk]
            b = collate_data_list([base.clone() for _ in range(len(part))],
                                  exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
            b.set_cell_parameters(part.to(b.device), skip_box_analysis=False)
            with torch.no_grad():
                o = b.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
            r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
            r = r.cpu()
            if bins is None:
                bins = torch.linspace(0, 10, r.shape[-1])
            for k, n in enumerate(names):
                d = compute_rdf_distance(form_rdf[k], r, bins)
                j = int(d.argmin())
                if float(d[j]) < best[n][0]:
                    stem = re.sub(r'_\d+_traj\d+_\d+\.pt$', '',
                                  os.path.basename(p))
                    best[n] = (float(d[j]),
                               f"{stem} step {int(idx[s + j, 0])} "
                               f"traj {int(idx[s + j, 1])} "
                               f"E={float(e[idx[s + j, 0], idx[s + j, 1]]):.2f}")
            del b, o, r
        del rec

    print(f"\n{n_states:,} trajectory states; {n_band:,} inside an energy band "
          f"(+/-{cli.tol} kJ/mol) and reconstructed")
    print(f"\n{'form':12s} {'best in-flight':>15} {'endpoints':>11}  where")
    for n in names:
        d, where = best[n]
        cmp = ('CLOSER in flight' if d < ENDPOINT_BEST[n] - MATERIAL
               else 'no closer than the endpoints')
        print(f"{n:12s} {d:15.4f} {ENDPOINT_BEST[n]:11.4f}  {cmp}")
        print(f"{'':12s} {where}")


if __name__ == '__main__':
    main()
