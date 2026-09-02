"""
Does Nikos' ACCEPTED pool contain a match for every known acridine polymorph?

His set is supposed to. The stored `direct_compack.pt` answers this only for the OLD
four-structure `from_nikos` set -- it was never re-run after the accepted pool
arrived, so the question has not actually been asked of the current data.

Each known polymorph is the COMPACK reference; every structure in his pool is tested
against it. Reported per polymorph: the best match in his set, and whether that is a
full 20/20, a partial, or nothing.

⚠ RELAXATION. His structures are L1 -- reprojected onto our reference conformer but
NEVER relaxed. If his came from a different method's optimisation they will sit near
but not on the experimental geometry, and a partial match (say 12-18/20 at high RMSD)
is exactly what that looks like. A near-miss here is therefore NOT evidence his pool
lacks the form; a 1-5/20 with no near neighbour is.

Both levels are run where available: L0 as he supplied them, L1 reprojected onto our
conformer. If a form matches at L0 but not L1, the reprojection is the problem, not
his structure.

The __main__ guard is REQUIRED: batch_compack spawns mp.Pool and Windows re-imports.
"""
import json
import os

import torch

from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.compare import compack_confirm
from energy_sampling.eval.nikos_comparison.summarize_search import ROOT, harmonize

WORK = os.path.join(ROOT, 'nikos_comparison', '_nikos_vs_poly')


def main():
    lev = torch.load(os.path.join(ROOT, 'nikos_comparison', 'nikos_levels.pt'),
                     weights_only=False, map_location='cpu')
    print("levels available:", [k for k in lev if hasattr(lev[k], 'batch_to_list')])
    poly = torch.load(os.path.join(ROOT, 'std_opt_acridine_polymorphs.pt'),
                      weights_only=False, map_location='cpu').cpu()
    pl, pids = poly.batch_to_list(), list(poly.identifier)
    print(f"{len(pids)} known polymorphs: {pids}")

    #: l2 first -- it is the like-for-like comparison and the one that
    #: decides whether a partial match at l1 was ever about relaxation
    #: results persisted, not just printed -- this table is the headline answer
    #: of the sub-project and it was living only in terminal scrollback
    out = {}
    for level in ('l2', 'l1', 'l0'):
        if level not in lev or not hasattr(lev[level], 'batch_to_list'):
            print(f"\n[{level}] not present, skipped")
            continue
        b = lev[level].cpu()
        his = b.batch_to_list()
        ids = list(b.identifier)
        print(f"\n=== level {level}: {len(his)} of his structures ===")

        cand = harmonize([c.clone() for c in his])
        sub = collate_data_list(cand)
        #: at l2 his structures sit at OUR minima, so the reference must sit at our
        #: minima too -- otherwise a real basin match reads as a near miss purely
        #: because the experimental geometry was never relaxed.
        refs, tag = pl, 'experimental (unrelaxed)'
        rp = os.path.join(ROOT, 'nikos_comparison', 'polymorphs_l2.pt')
        if level == 'l2' and os.path.exists(rp):
            refs = torch.load(rp, weights_only=False,
                              map_location='cpu').cpu().batch_to_list()
            tag = 'RELAXED'
        elif level == 'l2':
            print("  ! polymorphs_l2.pt missing -- relaxed-his vs UNRELAXED refs; "
                  "run relax_l2_chunked.py --polymorphs for like-for-like")
        print(f"  references: {tag}")
        q = collate_data_list([c.clone() for c in refs])
        q.aunit_handedness = q.aunit_handedness.abs()
        nb = torch.arange(len(cand)).repeat(len(refs), 1)
        rmsds, matched = compack_confirm(
            q, sub, nb, WORK + '_' + level + '_' + tag.split()[0])

        print(f"{'polymorph':14s} {'best':>6} {'rmsd':>7}  {'which':14s} "
              f"{'#>=15':>6} {'#>=20':>6}  verdict")
        for k, name in enumerate(pids):
            m = matched[k].clone()
            r = rmsds[k].clone()
            fail = (m == 0) & (r == 0.0)      # engine failure, not a zero-RMSD hit
            m[fail] = -1
            j = int(m.argmax())
            best, brms = int(m[j]), float(r[j])
            n15 = int((m >= 15).sum())
            n20 = int((m >= 20).sum())
            if best >= 20:
                v = 'MATCHED'
            elif best >= 12:
                v = 'partial -- could be relaxation'
            else:
                v = 'NOT PRESENT'
            print(f"{name:14s} {best:4d}/20 {brms:7.3f}  {ids[j]:14s} "
                  f"{n15:6d} {n20:6d}  {v}")
            out.setdefault(level, {})[name] = dict(
                best=best, rmsd=round(brms, 4), which=ids[j], n15=n15, n20=n20,
                verdict=v, refs=tag)
        print(f"   ({int(((matched == 0) & (rmsds == 0.0)).sum())} of "
              f"{matched.numel()} comparisons failed inside the engine and were "
              f"excluded, not counted as non-matches)")


    dest = os.path.join(ROOT, 'nikos_comparison', 'nikos_vs_polymorphs.json')
    json.dump(out, open(dest, 'w'), indent=1)
    print("")
    print(f"wrote {dest}")


if __name__ == '__main__':
    main()
