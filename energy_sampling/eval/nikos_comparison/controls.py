"""
The scale the comparison must be read against.

Neither an energy nor an RDF distance from `compare.py` means anything on its
own. Both need a reference measured the same way on structures whose answer we
already know -- the seven known acridine polymorphs. Without these two controls
the results mislead in both directions, and did:

  ENERGY. His structures score POSITIVE MACE lattice energy as given (+23 to +91
  kJ/mol), against prior minima at -62.8. That reads as "his structures are
  unbound" until you score the known EXPERIMENTAL polymorphs the same way and
  find they are positive too (+11 to +53). The difference is relaxation, not
  quality: `std_acridine_polymorphs.pt` is the experimental cell with our rigid
  conformer and is NOT relaxed, while `std_opt_acridine_polymorphs.pt` is the
  same structures after rigid-body relaxation and scores -55 to -60. So an
  unrelaxed structure scoring positive is the NORMAL result on this surface, and
  his structures may only be compared against the unrelaxed column until L2
  exists.

  DISTANCE. `collate_prior.py` calibrated `log_noise_range` thermally and used
  10**log_noise_range[1] (0.056-0.076) to thin the priors. That is a thinning
  radius for samples drawn FROM the prior, and it is too strict as an identity
  criterion for a structure ingested from outside: most of our own known
  polymorphs fall OUTSIDE the cutoff of the prior that was built to target them.
  What "present in our landscape" actually looks like in RDF distance is the
  range these controls measure, not the thinning cutoff.

Run:
    python -m energy_sampling.eval.nikos_comparison.controls --device cpu
"""
import argparse
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS

#: which prior each known polymorph was searched for, from collate_prior.TARGETS
TARGETS = {
    'sg14_zp1': ['ACRDIN04', 'ACRDIN12'],       # forms II and IX
    'sg14_zp2': ['ACRDIN07', 'ACRDIN06'],       # forms III and VII
    'sg9_zp2': ['ACRDIN05', 'ACRIDIN_VIII'],    # forms VI and VIII
}

#: the same polymorphs before and after rigid-body relaxation. The pair is the
#: point: it brackets where an unrelaxed and a relaxed structure sit.
STAGES = {
    'unrelaxed (= our L1)': 'std_acridine_polymorphs.pt',
    'rigid-body relaxed (= our L2)': 'std_opt_acridine_polymorphs.pt',
}


def polymorph_batch(path, energy_function, predictor, device, offset):
    poly = torch.load(path, weights_only=False, map_location='cpu').cpu()
    lst = poly.batch_to_list()
    ens, rdfs, vdw = [], [], []
    for i in range(0, len(lst), 3):
        sub = collate_data_list(lst[i:i + 3],
                                exclude_keys=[energy_function, 'rdf']).to(device)
        with torch.no_grad():
            out = sub.analyze([energy_function, 'rdf', 'vdw_max'],
                              assign_outputs=False, predictor=predictor,
                              **RDF_KWARGS)
        r = out['rdf'][0] if isinstance(out['rdf'], (tuple, list)) else out['rdf']
        ens.append(out[energy_function].cpu() + offset)
        vdw.append(out['vdw_max'].cpu())
        rdfs.append(r.detach().cpu())
    return poly, torch.cat(ens), torch.cat(rdfs, dim=0), torch.cat(vdw)


def main():
    from energy_sampling.utils import load_yaml
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=os.path.join(os.path.dirname(__file__),
                                                     'config.yaml'))
    ap.add_argument('--device', default=None)
    cli = ap.parse_args()

    cfg = load_yaml(cli.config)
    device = cli.device or cfg['device']
    ef = cfg['energy_function']
    root = os.path.dirname(cfg['polymorphs'])

    from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
    predictor = load_mace_model(cfg['mace_model'], device, torch.float32)

    from energy_sampling.eval.nikos_comparison.compare import calibrate_energy_offset
    offset, _ = calibrate_energy_offset(cfg['priors'], ef, predictor, device)

    cuts, prior_rdf = {}, {}
    for name, path in cfg['priors'].items():
        blob = torch.load(path, weights_only=False, map_location='cpu')
        cuts[name] = float(10 ** blob['log_noise_range'][1])
        cache = os.path.join(cfg['out_dir'], f'prior_rdf_{name}.pt')
        if not os.path.exists(cache):
            raise FileNotFoundError(f"{cache} missing -- run compare.py first")
        prior_rdf[name] = torch.load(cache, weights_only=False,
                                     map_location='cpu')['rdf']

    out = {'offset': offset, 'cuts': cuts, 'stages': {}}
    for stage, fname in STAGES.items():
        path = os.path.join(root, fname)
        poly, ens, rdfs, vdw = polymorph_batch(path, ef, predictor, device, offset)
        bins = torch.linspace(0, 10, rdfs.shape[-1])
        print(f"\n===== known polymorphs, {stage} =====")
        print(f"      ({fname})")
        rows = []
        for i, ident in enumerate(poly.identifier):
            targeted = [p for p, ids in TARGETS.items() if ident in ids]
            line = {'identifier': ident, 'sg': int(poly.sg_ind[i]),
                    'zp': int(poly.z_prime[i]), 'energy': float(ens[i]),
                    'vdw_max': float(vdw[i]), 'targeted_by': targeted, 'dists': {}}
            for pname, prdf in prior_rdf.items():
                line['dists'][pname] = float(
                    compute_rdf_distance(rdfs[i], prdf, bins).min())
            rows.append(line)
            tgt = targeted[0] if targeted else None
            d_tgt = line['dists'][tgt] if tgt else None
            print(f"  {ident:14s} sg={line['sg']:3d} zp={line['zp']} "
                  f"{ef}={line['energy']:8.2f} kJ/mol  "
                  + (f"dist to its own prior ({tgt}) = {d_tgt:.4f} "
                     f"[cutoff {cuts[tgt]:.4f}: "
                     f"{'inside' if d_tgt < cuts[tgt] else 'OUTSIDE'}]"
                     if tgt else "no prior targets this form"))
        out['stages'][stage] = rows
        e = torch.tensor([r['energy'] for r in rows])
        print(f"  --> energy range {e.min():.2f} to {e.max():.2f} kJ/mol")
        td = torch.tensor([r['dists'][r['targeted_by'][0]]
                           for r in rows if r['targeted_by']])
        print(f"  --> distance to own prior: {td.min():.4f} to {td.max():.4f} "
              f"({int((td < torch.tensor([cuts[r['targeted_by'][0]] for r in rows if r['targeted_by']])).sum())}"
              f"/{len(td)} inside the thinning cutoff)")

    dst = os.path.join(cfg['out_dir'], 'controls.pt')
    torch.save(out, dst)
    print(f"\nwrote {dst}")
    print("\nREAD THE COMPARISON AGAINST THESE. His structures currently exist "
          "only at L0/L1, so they are comparable ONLY to the unrelaxed row.")


if __name__ == '__main__':
    main()
