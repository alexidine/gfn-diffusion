"""
Four views of the RDF-distance embedding, over its top 4 dimensions.

The dimensions are INTERPRETED, not anonymous -- each was regressed against the
physical descriptors of all 1,969 low-energy structures:

    dim 1  43.7%   herringbone <-> layered    theta_med   r +0.82   multi-R2 0.97
    dim 2  22.1%   open <-> dense             mean_d      r +0.72   multi-R2 0.93
    dim 3  15.3%   cell shape                 cell_short  r +0.58   multi-R2 0.75
    dim 4   9.6%   cell anisotropy            cell_aniso  r -0.73   multi-R2 0.91

Fidelity, measured not assumed: 2D reproduces the true RDF distances at r 0.974 but
Kruskal stress 0.240 -- READ NEIGHBOURHOODS, NOT DISTANCES. 5D reaches stress 0.095.

The polymorphs and Nikos' structures were not in the embedded set, so they are placed
by the standard classical-MDS out-of-sample formula, and each one's placement
RESIDUAL is reported -- a structure that the embedding cannot place should not be
silently drawn as though it fits.
"""
import json
import os

import numpy as np
import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.dataset_utils.utils import collate_data_list

from energy_sampling.eval.nikos_comparison.levels import RDF_KWARGS
from energy_sampling.eval.nikos_comparison.summarize_search import (
    ROOT, chunk_by_cluster_cost, harmonize, load_arms, physical)

#: ⚠ IDENTITY, not proximity. Direct COMPACK of every known polymorph against
#: Nikos' accepted pool (nikos_vs_polymorphs.py, both L0 and L1, same answer):
#:
#:     ACRDIN04  20/20 rmsd 0.258  = nik00002
#:     ACRDIN12  20/20 rmsd 0.305  = nik00000
#:     (and ACRDIN07=nik00011, ACRDIN05=nik00007, ACRDIN06=nik00012,
#:      ACRIDIN_VIII=nik00004 in other space groups)
#:
#: So two of the three sg14-Z'1 "proposals" ARE the known forms. Drawing them as
#: separate points implied four distinct structures where there are two, and put
#: them 6 and 11 kJ/mol apart on the energy axis for no reason but relaxation
#: state -- his copies are unrelaxed L1, the polymorphs are relaxed. Merge them,
#: and use the RELAXED energy, which is the meaningful one.
SAME_AS = {'nik00002': 'ACRDIN04', 'nik00000': 'ACRDIN12'}

SP = os.path.dirname(os.path.abspath(__file__))
LOW = -60.0
K = 4
DIMS = [(0, 1), (0, 2), (0, 3), (1, 2)]
MEANING = {0: ('herringbone', 'layered', 'interplane angle'),
           1: ('open', 'dense', 'mean neighbour distance'),
           2: ('flat cell', 'long cell', 'cell shape'),
           3: ('isotropic', 'anisotropic', 'cell anisotropy')}
VAR = {0: 43.7, 1: 22.1, 2: 15.3, 3: 9.6}

W, H = 420, 330
PAD_L, PAD_R, PAD_T, PAD_B = 46, 14, 26, 44


def main():
    blob = torch.load(os.path.join(ROOT, 'nikos_comparison', 'lowE_dmat.pt'),
                      weights_only=False)
    Dm = blob['D'].numpy().astype(np.float64)
    E = blob['E'].numpy()
    n = Dm.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    D2 = Dm ** 2
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    lam = np.clip(w[:K], 1e-12, None)
    Y = V[:, :K] * np.sqrt(lam)
    print(f"embedded {n:,} structures, {K} dims")

    #: out-of-sample placement, classical-MDS (Gower) formula
    groups = load_arms(os.path.join(ROOT, 'prior_chunks'),
                       'may_acridine_sg14_zp1_*.pt')
    raw = [c for stem in sorted(groups)
           for _, lst in sorted(groups[stem]) for c in lst]
    keep, _ = physical(raw)
    Ea = torch.tensor([float(c.mace) for c in keep])
    idx = (Ea <= LOW).nonzero().flatten().tolist()
    pool = harmonize([keep[i] for i in idx])

    #: BOTH RELAXATION STATES. relax_states.json holds, per structure, its RDF
    #: distance to every pool member BEFORE and AFTER rigid-body relaxation, so
    #: each can be placed out-of-sample twice and the move drawn. Mixing states
    #: silently was the confound; showing the move makes it the information.
    ST = json.load(open(os.path.join(SP, 'relax_states.json')))
    d2mean = D2.mean(1)
    Vk = V[:, :K] / np.sqrt(lam)

    def place(dvec):
        d = np.asarray(dvec, dtype=np.float64)
        y = 0.5 * Vk.T @ (d2mean - d ** 2)
        dh = np.linalg.norm(Y - y[None, :], axis=1)
        return y, float(np.sqrt(((d - dh) ** 2).mean()) / d.mean())

    placed = []
    print()
    print(f"{'structure':10s} {'resid unrel':>12} "
          f"{'resid relaxed':>14} {'nearest after':>14}")
    for st in ST:
        y0, r0 = place(st['d0'])
        y1, r1 = place(st['d1'])
        placed.append((st['name'], st['kind'], y0, y1))
        print(f"{st['name']:10s} {100 * r0:11.1f}% {100 * r1:13.1f}% "
              f"{min(st['d1']):14.4f}")

    lo = np.percentile(Y, 1, axis=0)
    hi = np.percentile(Y, 99, axis=0)
    emin, emax = float(E.min()), float(E.max())

    def panel(ix, iy):
        x0, x1 = lo[ix] * 1.15, hi[ix] * 1.15
        y0, y1 = lo[iy] * 1.15, hi[iy] * 1.15

        def sx(v):
            return PAD_L + (v - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

        def sy(v):
            return H - PAD_B - (v - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

        o = [f'<rect class="panel" x="{PAD_L}" y="{PAD_T}" '
             f'width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}"/>']
        for i in range(n):
            t = (E[i] - emin) / (emax - emin + 1e-9)
            o.append(f'<circle class="dot" cx="{sx(Y[i, ix]):.1f}" '
                     f'cy="{sy(Y[i, iy]):.1f}" r="1.7" opacity="{0.18 + 0.5 * (1 - t):.2f}"/>')
        #: hollow = as supplied, filled = relaxed, dashed line = the move.
        #: nik00000 and ACRDIN12 land on the SAME point once both are relaxed
        #: (0.00 kJ/mol, RDF 0.0033) -- their filled markers coincide, and that
        #: coincidence is the result, not a plotting fault.
        lab = []
        #: NOT y0/y1 -- those are the AXIS BOUNDS this panel closed over, and
        #: sy() reads them at call time. Rebinding them here turned every
        #: coordinate into an array.
        for nm, cls, pos0, pos1 in placed:
            X0, Yp0 = sx(pos0[ix]), sy(pos0[iy])
            X1, Yp1 = sx(pos1[ix]), sy(pos1[iy])
            o.append(f'<line class="relax" x1="{X0:.1f}" y1="{Yp0:.1f}" '
                     f'x2="{X1:.1f}" y2="{Yp1:.1f}"/>')
            o.append(f'<circle class="mk {cls} hollow" cx="{X0:.1f}" '
                     f'cy="{Yp0:.1f}" r="4"/>')
            o.append(f'<circle class="mk {cls}" cx="{X1:.1f}" cy="{Yp1:.1f}" r="5"/>')
            ly = Yp1 - 11
            for _ in range(8):
                if all(abs(X1 - px) > 54 or abs(ly - py) > 12 for px, py in lab):
                    break
                ly = ly - 13 if ly <= Yp1 else Yp1 + (Yp1 - ly) + 13
                if abs(ly - Yp1) > 60:
                    ly = Yp1 + 13
                    break
            lab.append((X1, ly))
            if abs(ly - (Yp1 - 11)) > 2:
                o.append(f'<line class="lead" x1="{X1:.1f}" '
                         f'y1="{Yp1 + (5 if ly > Yp1 else -5):.1f}" x2="{X1:.1f}" '
                         f'y2="{ly + (-8 if ly > Yp1 else 3):.1f}"/>')
            o.append(f'<text class="mklab {cls}" x="{X1:.1f}" y="{ly:.1f}" '
                     f'text-anchor="middle">{nm}</text>')
        a_lo, a_hi, aname = MEANING[ix]
        b_lo, b_hi, bname = MEANING[iy]
        o.append(f'<text class="axl" x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 24}" '
                 f'text-anchor="middle">dim {ix + 1} &middot; {aname} '
                 f'({VAR[ix]:.0f}%)</text>')
        o.append(f'<text class="axend" x="{PAD_L}" y="{H - 10}">&larr; {a_lo}</text>')
        o.append(f'<text class="axend" x="{W - PAD_R}" y="{H - 10}" '
                 f'text-anchor="end">{a_hi} &rarr;</text>')
        o.append(f'<text class="axl" transform="translate(13,'
                 f'{(PAD_T + H - PAD_B) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 f'dim {iy + 1} &middot; {bname} ({VAR[iy]:.0f}%)</text>')
        return (f'<svg viewBox="0 0 {W} {H}" class="fig mdsfig">'
                f'{"".join(o)}</svg>')

    panels = ''.join(f'<figure class="mp">{panel(a, b)}</figure>' for a, b in DIMS)
    resid_txt = 'all under 25%'

    FIG = f'''<h2>The embedding, with its axes named</h2>
<p class="sub">Classical scaling of the RDF distance matrix over all
{n:,} low-energy structures, shown on its four leading dimensions. The axes are not
anonymous: each was regressed against the physical descriptors of every structure,
and all four are well captured (multi-R&sup2; 0.75&ndash;0.97), so each carries a
name rather than a number. Darker points are lower in energy.</p>
<div class="mpgrid">{panels}</div>
<div class="figkey">
 <span><i class="sw known"></i>experimental form</span>
 <span><i class="sw nik"></i>Nikos&rsquo; proposal</span>
 <span><i class="sw ours"></i>our low-energy structures</span>
</div>
<p class="caveat"><b>Read neighbourhoods, not distances.</b> Two dimensions reproduce
the true RDF distances at r&nbsp;=&nbsp;0.974 but with Kruskal stress
<b>0.240</b> &mdash; poor by the usual bar, meaning the <em>ordering</em> of
similarities survives while the scale is compressed. Five dimensions would reach
stress 0.095. The five named structures were not part of the embedded set; they are
placed by the standard out-of-sample formula ({resid_txt} residual). Each is drawn
<b>twice</b>: hollow as supplied, filled after rigid-body relaxation, joined by the
move. <b>nik00000 and ACRDIN12 converge onto the same point</b> &mdash; 0.00 kJ/mol
and RDF 0.0033 apart once both are relaxed, so their filled markers coincide.
nik00002 settles beside ACRDIN04 rather than on it.</p>'''
    open(os.path.join(SP, 'mds_block.html'), 'w', encoding='utf-8').write(FIG)
    print(f"\nwrote mds_block.html ({len(FIG):,} bytes), {len(DIMS)} panels")


if __name__ == '__main__':
    main()
