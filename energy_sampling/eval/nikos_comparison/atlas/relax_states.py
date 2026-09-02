"""
Before/after rigid-body relaxation for every named structure, for the figures.

The atlas has been showing a mixture of relaxation states on one energy axis, which
is not comparable. Rather than pick a state, show BOTH and draw the move -- the
offset is real information, and it is the single biggest confound in every energy
comparison in this document.

Measured (relaxed vs relaxed, 120 steps, production stage-2 settings):

    nik00000 -> ACRDIN12   gap -0.00 kJ/mol, RDF 0.0033, COMPACK 20/20 rmsd 0.010
                           => THE SAME STRUCTURE, to machine precision
    nik00002 -> ACRDIN04   gap +0.78 kJ/mol, RDF 0.047,  COMPACK 20/20 rmsd 0.125
                           => same basin, NOT the same point

So the earlier merge was right for one pair and wrong for the other. Both are drawn
separately here, with their relaxation arrows, and the reader can see which
coincides and which does not.

⚠ RELAXATION IS RIGID-BODY: cell + pose, molecule held at our reference conformer.
All-atom is still not done anywhere in this analysis.
"""
import json
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import ROOT, harmonize

SP = os.path.dirname(os.path.abspath(__file__))
MACE_MODEL = r"D:\crystal_datasets\acr_112025_mh1_stagetwo.model"
OPT = dict(optimizer_func='rprop', init_lr=0.01, max_num_steps=120,
           convergence_eps=1e-5, grad_norm_clip=0.1, optim_target='mace',
           cutoff=10, enforce_reduced=True, anneal_lr=True,
           compression_factor=0.0, target_packing_coeff=None, show_tqdm=False)
POLY = ['ACRDIN04', 'ACRDIN12']
NIK = ['nik00000', 'nik00001', 'nik00002']


def main():
    pred = load_mace_model(MACE_MODEL, 'cpu', torch.float32)

    def bat(lst):
        b = collate_data_list([c.clone() for c in lst],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        return b

    def score(lst):
        with torch.no_grad():
            o = bat(lst).analyze(['mace'], assign_outputs=False, predictor=pred)
        v = o['mace']
        return (v[0] if isinstance(v, (tuple, list)) else v).flatten().cpu()

    def rdf(lst):
        with torch.no_grad():
            o = bat(lst).analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        return r.cpu()

    samp = torch.load(os.path.join(ROOT, 'opt_outs',
                                   'aug21seed_acridine_sg14_zp2_0.pt'),
                      weights_only=False, map_location='cpu')[:10]
    OFFSET = float((torch.tensor([float(c.mace) for c in samp])
                    - score(samp)).mean())

    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    lev = torch.load(os.path.join(ROOT, 'nikos_comparison', 'nikos_levels.pt'),
                     weights_only=False, map_location='cpu')
    l1 = lev['l1'].cpu()
    ids, ll = list(l1.identifier), l1.batch_to_list()

    names = POLY + NIK
    kinds = ['known'] * len(POLY) + ['nik'] * len(NIK)
    src = ([pl[pids.index(n)] for n in POLY] + [ll[ids.index(n)] for n in NIK])
    src = harmonize([c.clone() for c in src])

    e0 = (score(src) + OFFSET).tolist()
    pc0 = [float(c.packing_coeff) for c in src]
    r0 = rdf(src)

    rel = collate_data_list(
        bat(src).optimize_crystal_parameters(predictor=pred, **OPT)).cpu()
    rl = rel.batch_to_list()
    e1 = (score(rl) + OFFSET).tolist()
    pc1 = [float(c.packing_coeff) for c in rl]
    r1 = rdf(rl)

    #: distances to the low-energy pool, for out-of-sample placement in the MDS
    blob = torch.load(os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt'),
                      weights_only=False)
    from energy_sampling.eval.nikos_comparison.summarize_search import (
        chunk_by_cluster_cost, load_arms, physical)
    groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                       'may_acridine_sg14_zp1_*.pt')
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    Ea = torch.tensor([float(c.mace) for c in keep])
    pool = harmonize([keep[i] for i in
                      (Ea <= -60.0).nonzero().flatten().tolist()])
    rp = []
    for lo, hi in chunk_by_cluster_cost(pool, 1_500_000):
        rp.append(rdf(pool[lo:hi]))
    rp = torch.cat(rp)
    bins = torch.linspace(0, 10, rp.shape[-1])

    out = []
    for i, nm in enumerate(names):
        d0 = compute_rdf_distance(r0[i], rp, bins).tolist()
        d1 = compute_rdf_distance(r1[i], rp, bins).tolist()
        out.append(dict(name=nm, kind=kinds[i],
                        e0=e0[i], e1=e1[i], pc0=pc0[i], pc1=pc1[i],
                        d0=d0, d1=d1))
        print(f"{nm:10s} {kinds[i]:6s} E {e0[i]:8.2f} -> {e1[i]:8.2f}   "
              f"pc {pc0[i]:.3f} -> {pc1[i]:.3f}   "
              f"moved {abs(e1[i] - e0[i]):5.2f} kJ/mol")
    json.dump(out, open(os.path.join(SP, 'relax_states.json'), 'w'))
    print(f"\nwrote relax_states.json")


if __name__ == '__main__':
    main()
