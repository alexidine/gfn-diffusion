"""
Landscape figure: two panels, both on axes that mean something.

MEASURED FIRST, then drawn. The abstract embeddings do not carry this landscape:

    PCA on 12 latent params      PC1+PC2 = 38.7%   -> not a map
    classical MDS on RDF dist    stress-1 = 0.249  -> poor by convention, though
                                 distance correlation 0.962, so ORDER survives
                                 and scale does not

Interpretable axes beat both for the questions actually being asked:

    descriptor        |r| with E   gamma/beta d'
    coordination        0.43          0.02        <- best energy predictor
    interplane angle    0.25          2.68        <- best family discriminator
    slip                0.10          0.92
    stack distance      0.02          0.05        <- predicts NOTHING, a non-variable

So: panel A is the field-standard energy-vs-density landscape over all 53,582
structures; panel B is coordination vs interplane angle, the two descriptors that
carry energy and family respectively.
"""
import json
import math
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(SP, 'projection.json')))

W, H = 560, 400
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 22, 52


def sx(v, lo, hi):
    return PAD_L + (v - lo) / (hi - lo) * (W - PAD_L - PAD_R)


def sy(v, lo, hi):
    return H - PAD_B - (v - lo) / (hi - lo) * (H - PAD_T - PAD_B)


def axes(xlo, xhi, ylo, yhi, xlab, ylab, xticks, yticks, xfmt="{:.2f}",
         yfmt="{:.0f}"):
    o = [f'<rect class="panel" x="{PAD_L}" y="{PAD_T}" '
         f'width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}"/>']
    for t in xticks:
        x = sx(t, xlo, xhi)
        o.append(f'<line class="grid" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" '
                 f'y2="{H - PAD_B}"/>')
        o.append(f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 16}" '
                 f'text-anchor="middle">{xfmt.format(t)}</text>')
    for t in yticks:
        y = sy(t, ylo, yhi)
        o.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" '
                 f'x2="{W - PAD_R}" y2="{y:.1f}"/>')
        o.append(f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" '
                 f'text-anchor="end">{yfmt.format(t)}</text>')
    o.append(f'<text class="axlab" x="{(PAD_L + W - PAD_R) / 2:.0f}" '
             f'y="{H - 12}" text-anchor="middle">{xlab}</text>')
    o.append(f'<text class="axlab" transform="translate(14,{(PAD_T + H - PAD_B) / 2:.0f}) '
             f'rotate(-90)" text-anchor="middle">{ylab}</text>')
    return o


# ---------- panel A: the standard CSP landscape, energy vs density ----------
ae, apc = P['all_e'], P['all_pc']
XLO, XHI, YLO, YHI = 0.55, 0.80, -63.5, -30.0
NX, NY = 96, 64
grid = {}
for e, p in zip(ae, apc):
    if not (XLO <= p <= XHI and YLO <= e <= YHI):
        continue
    gx = int((p - XLO) / (XHI - XLO) * NX)
    gy = int((e - YLO) / (YHI - YLO) * NY)
    grid[(gx, gy)] = grid.get((gx, gy), 0) + 1
mx = max(grid.values())
cw = (W - PAD_L - PAD_R) / NX
ch = (H - PAD_T - PAD_B) / NY
cells = []
for (gx, gy), c in grid.items():
    op = 0.10 + 0.90 * (math.log(c + 1) / math.log(mx + 1))
    x = PAD_L + gx * cw
    y = H - PAD_B - (gy + 1) * ch
    cells.append(f'<rect class="cell" x="{x:.1f}" y="{y:.1f}" '
                 f'width="{cw + .6:.1f}" height="{ch + .6:.1f}" opacity="{op:.3f}"/>')

#: BOTH RELAXATION STATES, with the move drawn. Mixing states on one energy axis
#: is not a comparison -- and the offsets are not small:
#:
#:     ACRDIN04  -60.07 -> -60.88   moved  0.82      pc 0.701 -> 0.685
#:     ACRDIN12  -58.63 -> -60.09          1.46         0.680 -> 0.669
#:     nik00002  -53.93 -> -60.11          6.18         0.743 -> 0.687
#:     nik00000  -47.67 -> -60.09         12.43         0.721 -> 0.669
#:     nik00001   -4.67 -> -43.91         39.24         0.626 -> 0.606
#:
#: His move 6-39 kJ/mol, the polymorphs 1-1.5, because his arrive unrelaxed while
#: the CSD geometries already sit near a minimum. Every mixed-state number earlier
#: in this project was distorted by exactly that asymmetry.
#:
#: NOT MERGED any more. Relaxed-to-relaxed: nik00000 lands ON ACRDIN12 (gap 0.00,
#: RDF 0.0033, COMPACK rmsd 0.010 -- the same structure); nik00002 lands NEAR
#: ACRDIN04 but 0.78 kJ/mol above it at rmsd 0.125 -- same basin, different point.
#: Drawing them separately lets that distinction show.
ST = json.load(open(os.path.join(SP, 'relax_states.json')))

panelA = ['<svg viewBox="0 0 {} {}" class="fig landfig">'.format(W, H)]
panelA += axes(XLO, XHI, YLO, YHI, 'packing coefficient',
               'lattice energy  (kJ/mol)',
               [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
               [-60, -50, -40, -30])
panelA += cells

yfloor = sy(-62.81, YLO, YHI)
panelA.append(f'<line class="floor" x1="{PAD_L}" y1="{yfloor:.1f}" '
              f'x2="{W - PAD_R}" y2="{yfloor:.1f}"/>')
panelA.append(f'<text class="glab floorlab" x="{PAD_L + 5}" y="{yfloor - 5:.1f}">'
              f'our floor &#8722;62.81</text>')

labA = []
for st in sorted(ST, key=lambda t: t['pc1']):
    cls, nm = st['kind'], st['name']
    x0, x1 = sx(st['pc0'], XLO, XHI), sx(st['pc1'], XLO, XHI)
    off0 = st['e0'] > YHI
    y0 = sy(YHI, YLO, YHI) + 7 if off0 else sy(st['e0'], YLO, YHI)
    y1 = sy(st['e1'], YLO, YHI)
    #: the move itself
    panelA.append(f'<line class="relax" x1="{x0:.1f}" y1="{y0:.1f}" '
                  f'x2="{x1:.1f}" y2="{y1:.1f}"/>')
    #: unrelaxed = hollow, relaxed = filled
    if off0:
        panelA.append(f'<path class="mk {cls} hollow" d="M{x0 - 5:.1f} '
                      f'{y0 + 4:.1f} L{x0 + 5:.1f} {y0 + 4:.1f} L{x0:.1f} '
                      f'{y0 - 4:.1f} Z"/>')
        panelA.append(f'<text class="offlab {cls}" x="{x0:.1f}" '
                      f'y="{y0 - 9:.1f}" text-anchor="middle">'
                      f'{nm} unrelaxed &#8593; {st["e0"]:.1f}</text>')
    else:
        panelA.append(f'<circle class="mk {cls} hollow" cx="{x0:.1f}" '
                      f'cy="{y0:.1f}" r="4"/>')
    panelA.append(f'<circle class="mk {cls}" cx="{x1:.1f}" cy="{y1:.1f}" r="5"/>')
    ly = y1 + 15
    for _ in range(8):
        if all(abs(x1 - px) > 64 or abs(ly - py) > 12 for px, py in labA):
            break
        ly += 13
    labA.append((x1, ly))
    panelA.append(f'<text class="mklab {cls}" x="{x1:.1f}" y="{ly:.1f}" '
                  f'text-anchor="middle">{nm} {st["e1"]:.2f}</text>')
panelA.append('</svg>')

# ---------- panel B: packing coefficient vs spread of interplane angles ------
#: AXES CHOSEN BY SEARCH, not by first impression. 24 descriptors, 276 pairs,
#: scored on family separation, energy correlation, and SPREAD (median
#: nearest-neighbour gap in normalised coords -- how much the points collide).
#:
#:   pair                        family   |r| E   spread   note
#:   theta_std x beta_angle        1.66    0.28      7.5   <- used
#:   theta_std x mean_d            1.63    0.50      5.9   better on energy
#:   theta_hi  x n_nbr             1.59    0.43      1.2   1st attempt, rank 60
#:
#: n_nbr takes only SIX distinct values over 34 basins, so everything collapsed
#: onto six lines -- that was the unreadability.
#:
#: packing_coeff scored top overall but is EXCLUDED: it is a density, not a
#: packing degree of freedom, and it is already panel A's x-axis, so panel B
#: would have restated panel A. beta is the monoclinic shear angle -- a real cell
#: DoF, and for stacked layers it is what sets the offset between them.
import statistics as _st


#: panel B -- basins on (beta, theta_std) -- lived here. The coarse-grained
#: figure now draws that exact view over all 42 basins with energy colour and
#: labels, so this was a second, worse copy reading a stale 34-basin source.
#: Deleted rather than left dead: it still LOADED shells.json, which reads as
#: a live dependency on the retired clustering.

FIG = f'''<h2>Where the structures sit</h2>
<p class="sub">The field-standard landscape: every physical structure the search
produced, by density and energy.</p>
<div class="figs one">
 <figure class="fw">
  {''.join(panelA)}
  <figcaption><b>A &middot; the landscape.</b> All 53,582 physical structures, binned
  by density and energy; darker is more populated. Both experimental forms and
  Nikos&rsquo; three proposals are placed at their own density and energy. Neither
  known form is the minimum &mdash; they sit 2.7 and 4.2&nbsp;kJ/mol above our floor.
  Each named structure is drawn <b>twice</b> &mdash; hollow as supplied, filled after
  rigid-body relaxation, joined by the move. Nikos&rsquo; structures travel
  6&ndash;39&nbsp;kJ/mol because they arrive unrelaxed; the polymorphs travel
  1&ndash;1.5. Relaxed, <b>nik00000 lands exactly on ACRDIN12</b> (0.00&nbsp;kJ/mol
  apart, packing match RMSD 0.010&nbsp;&Aring;) and <b>nik00002 lands beside
  ACRDIN04</b>, 0.78&nbsp;kJ/mol above it in the same basin. Only <b>nik00001</b> is
  genuinely elsewhere; its unrelaxed energy is off the top of the axis.</figcaption>
 </figure>
</div>
<p class="caveat"><b>Stack distance is deliberately not plotted anywhere.</b> It
correlates 0.08 with energy and separates the motif families at
d&prime;&nbsp;=&nbsp;0.19 &mdash; across all 42 basins it spans 3.21 to
3.47&nbsp;&Aring; with a standard deviation of 0.05&nbsp;&Aring;. It is a
non-variable, which is why the atlas leads with it never changing rather than
giving it an axis.</p>'''

open(os.path.join(SP, 'figure_block.html'), 'w', encoding='utf-8').write(FIG)
print(f"wrote figure_block.html ({len(FIG):,} bytes); "
      f"{len(grid)} density cells over {len(P['all_e']):,} structures, "
      f"{len(ST)} named structures drawn twice")
