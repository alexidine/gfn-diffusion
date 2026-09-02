"""
Do Nikos' structures actually RELAX ONTO the polymorphs they match?

The atlas currently merges nik00002 into ACRDIN04 and nik00000 into ACRDIN12 and
draws each once, at the POLYMORPH's relaxed energy. That merge was asserted by me
from the COMPACK identity, not measured -- and the identity is looser than the
others in the set (rmsd 0.258 / 0.305 A against 0.137-0.156 for the rest). If his
structures relax somewhere else, they are near-neighbours of the known forms rather
than the same point, and the figure is wrong to collapse them.

Relax his three sg14-Z'1 structures with the production stage-2 optimiser, then ask:

    energy    does it reach the polymorph's relaxed energy?
    RDF       does it land ON the polymorph, or beside it?
    COMPACK   is it still a 20/20 match after relaxing?

MERGE IS JUSTIFIED only if the relaxed structure lands on the polymorph in all
three. Otherwise the figure must show them separately, with the offset visible.

The __main__ guard is REQUIRED: compack_confirm spawns mp.Pool on Windows.
"""
import os

import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model

from energy_sampling.eval.nikos_comparison.compare import compack_confirm
from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import ROOT, harmonize

MACE_MODEL = r"D:\crystal_datasets\acr_112025_mh1_stagetwo.model"
OPT = dict(optimizer_func='rprop', init_lr=0.01, max_num_steps=120,
           convergence_eps=1e-5, grad_norm_clip=0.1, optim_target='mace',
           cutoff=10, enforce_reduced=True, anneal_lr=True,
           compression_factor=0.0, target_packing_coeff=None, show_tqdm=True)
PAIRS = {'nik00002': 'ACRDIN04', 'nik00000': 'ACRDIN12', 'nik00001': None}
WORK = os.path.join(ROOT, 'nikos_comparison', '_relax_nikos')


def main():
    pred = load_mace_model(MACE_MODEL, 'cpu', torch.float32)

    def score(lst):
        b = collate_data_list([c.clone() for c in lst],
                              exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        b.aunit_handedness = b.aunit_handedness.abs()
        with torch.no_grad():
            o = b.analyze(['mace'], assign_outputs=False, predictor=pred)
        v = o['mace']
        return (v[0] if isinstance(v, (tuple, list)) else v).flatten().cpu()

    samp = torch.load(os.path.join(ROOT, 'opt_outs',
                                   'aug21seed_acridine_sg14_zp2_0.pt'),
                      weights_only=False, map_location='cpu')[:10]
    off = torch.tensor([float(c.mace) for c in samp]) - score(samp)
    assert float(off.std()) < 0.5
    OFFSET = float(off.mean())
    print(f"energy offset {OFFSET:.4f}\n")

    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    lev = torch.load(os.path.join(ROOT, 'nikos_comparison', 'nikos_levels.pt'),
                     weights_only=False, map_location='cpu')
    l1 = lev['l1'].cpu()
    ids, ll = list(l1.identifier), l1.batch_to_list()

    names = [n for n in PAIRS if n in ids]
    his = harmonize([ll[ids.index(n)].clone() for n in names])
    b = collate_data_list([c.clone() for c in his],
                          exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    b.aunit_handedness = b.aunit_handedness.abs()
    e_before = (score(his) + OFFSET).tolist()
    print(f"relaxing {len(names)} structures, {OPT['max_num_steps']} steps")
    out = b.optimize_crystal_parameters(predictor=pred, **OPT)
    rel = collate_data_list(out).cpu().batch_to_list()
    e_after = (score(rel) + OFFSET).tolist()

    def rdf(lst):
        bb = collate_data_list([c.clone() for c in lst],
                               exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
        bb.aunit_handedness = bb.aunit_handedness.abs()
        with torch.no_grad():
            o = bb.analyze(['rdf'], assign_outputs=False, **RDF_KWARGS)
        r = o['rdf'][0] if isinstance(o['rdf'], (tuple, list)) else o['rdf']
        return r.cpu()

    r_rel = rdf(rel)
    tgt = [PAIRS[n] for n in names if PAIRS[n]]
    bins = torch.linspace(0, 10, r_rel.shape[-1])
    #: ⚠ THE TARGET MUST BE RELAXED TOO. Scoring the CSD geometry and comparing it
    #: to a relaxed structure is the same unfair comparison the figure just had to
    #: fix -- it made nik00000 look 1.46 kJ/mol BELOW its target when in fact the
    #: target had simply never been relaxed. Relax the polymorphs on identical
    #: settings and compare relaxed-to-relaxed.
    tb = collate_data_list([pl[pids.index(t)].clone() for t in tgt],
                           exclude_keys=['rdf', 'fingerprint', 'rdf_bins'])
    tb.aunit_handedness = tb.aunit_handedness.abs()
    e_tgt_raw = {t: float(v) for t, v in
                 zip(tgt, score([pl[pids.index(t)] for t in tgt]) + OFFSET)}
    t_rel = collate_data_list(
        tb.optimize_crystal_parameters(predictor=pred, **OPT)).cpu().batch_to_list()
    e_tgt = {t: float(v) for t, v in zip(tgt, score(t_rel) + OFFSET)}
    r_tgt = rdf(t_rel)
    print()
    print("polymorphs relaxed on the same settings:")
    for t in tgt:
        print(f"   {t:10s} {e_tgt_raw[t]:8.2f} -> {e_tgt[t]:8.2f}")

    print(f"\n{'structure':10s} {'target':10s} {'E before':>9} {'E after':>8} "
          f"{'E target':>9} {'gap':>7} {'RDF to target':>14}")
    for i, n in enumerate(names):
        t = PAIRS[n]
        if t is None:
            print(f"{n:10s} {'--':10s} {e_before[i]:9.2f} {e_after[i]:8.2f} "
                  f"{'--':>9} {'--':>7} {'--':>14}")
            continue
        k = tgt.index(t)
        d = float(compute_rdf_distance(r_tgt[k], r_rel[i:i + 1], bins)[0])
        print(f"{n:10s} {t:10s} {e_before[i]:9.2f} {e_after[i]:8.2f} "
              f"{e_tgt[t]:9.2f} {e_after[i] - e_tgt[t]:+7.2f} {d:14.4f}")

    #: and does it still match after relaxing?
    q = collate_data_list([c.clone() for c in t_rel])
    q.aunit_handedness = q.aunit_handedness.abs()
    sub = collate_data_list([c.clone() for c in rel])
    nb = torch.tensor([[names.index(n) for n in names if PAIRS[n] == t]
                       for t in tgt])
    rm, mt = compack_confirm(q, sub, nb, WORK)
    print(f"\n{'target':10s} {'matched':>9} {'rmsd':>7}   (after relaxation)")
    for k, t in enumerate(tgt):
        print(f"{t:10s} {int(mt[k, 0]):6d}/20 {float(rm[k, 0]):7.3f}")
    print("\nmerge is justified only if E lands on target, RDF is small, and the")
    print("match holds at 20/20. Otherwise show them as separate points.")


if __name__ == '__main__':
    main()
