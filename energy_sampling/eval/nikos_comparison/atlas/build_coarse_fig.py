"""
The coarse-grained landscape: 42 basins, four views, coloured by depth.

1,969 sampled structures collapse to 42 objects because the landscape is genuinely
discrete -- within a basin the optimiser returns the SAME structure (true RDF
distance ~0.000), and between basins the separation is real. Each basin is drawn
once, at its own lowest-energy member.

Why this is more than a decluttered scatter: the basins were found by clustering RDF
distances, and they ALSO separate on physical descriptors (3.8x their nearest-
neighbour spacing). Two independent routes, same objects. So the same 42 points can
be shown on RDF-derived axes and on physical axes, and mean the same thing on both.

Colour = minimum energy of the basin, in five bins, defined per theme in CSS so the
ramp survives light and dark. Area = how often the search landed there.
"""
import json
import math
import os

SP = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(SP, 'coarse_landscape.json')))

W, H = 430, 340
PAD_L, PAD_R, PAD_T, PAD_B = 52, 16, 20, 50

EMIN = min(g['emin'] for g in G)
EMAX = max(g['emin'] for g in G)
EDGES = [EMIN + (EMAX - EMIN) * f for f in (0.2, 0.4, 0.6, 0.8)]


def ebin(e):
    return sum(e > x for x in EDGES) + 1        # 1 = deepest, 5 = shallowest


VIEWS = [
    ('pc', 'emin', 'packing coefficient', 'lattice energy  (kJ/mol)',
     'the classic landscape, one point per basin'),
    ('beta', 'theta_std', 'monoclinic β angle  (deg)',
     'spread of interplane angles  (deg)',
     'two packing degrees of freedom'),
    ('mds0', 'mds1', 'RDF embedding dim 1  · herringbone → layered',
     'dim 2  · open → dense', 'the metric that defined the basins'),
    ('n_nbr', 'mean_d', 'coordination number',
     'mean neighbour distance  (Å)', 'how tightly each basin packs'),
]


def val(g, k):
    if k == 'mds0':
        return g['mds'][0]
    if k == 'mds1':
        return g['mds'][1]
    return g[k]


def panel(xk, yk, xlab, ylab, sub):
    xs = [val(g, xk) for g in G]
    ys = [val(g, yk) for g in G]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    px, py = (x1 - x0) * .12 or 1, (y1 - y0) * .12 or 1
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py

    def sx(v):
        return PAD_L + (v - x0) / (x1 - x0) * (W - PAD_L - PAD_R)

    def sy(v):
        return H - PAD_B - (v - y0) / (y1 - y0) * (H - PAD_T - PAD_B)

    o = [f'<rect class="panel" x="{PAD_L}" y="{PAD_T}" '
         f'width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}"/>']
    for t in range(5):
        gx = x0 + (x1 - x0) * t / 4
        gy = y0 + (y1 - y0) * t / 4
        o.append(f'<line class="grid" x1="{sx(gx):.1f}" y1="{PAD_T}" '
                 f'x2="{sx(gx):.1f}" y2="{H - PAD_B}"/>')
        o.append(f'<line class="grid" x1="{PAD_L}" y1="{sy(gy):.1f}" '
                 f'x2="{W - PAD_R}" y2="{sy(gy):.1f}"/>')
        fmt = '{:.2f}' if abs(x1 - x0) < 6 else '{:.0f}'
        o.append(f'<text class="tick" x="{sx(gx):.1f}" y="{H - PAD_B + 14}" '
                 f'text-anchor="middle">{fmt.format(gx)}</text>')
        fmt = '{:.2f}' if abs(y1 - y0) < 6 else '{:.0f}'
        o.append(f'<text class="tick" x="{PAD_L - 6}" y="{sy(gy) + 4:.1f}" '
                 f'text-anchor="end">{fmt.format(gy)}</text>')

    nmax = max(g['n'] for g in G)
    lab = []
    for g in sorted(G, key=lambda d: -d['n']):
        x, y = sx(val(g, xk)), sy(val(g, yk))
        r = 3.4 + 11.0 * math.sqrt(g['n'] / nmax)
        cls = f"b{ebin(g['emin'])}"
        shape = 'circle' if g['fam'] == 'gamma' else 'rect'
        if shape == 'circle':
            o.append(f'<circle class="bs {cls}" cx="{x:.1f}" cy="{y:.1f}" '
                     f'r="{r:.1f}"/>')
        else:
            o.append(f'<rect class="bs {cls}" x="{x - r:.1f}" y="{y - r:.1f}" '
                     f'width="{2 * r:.1f}" height="{2 * r:.1f}"/>')
        inb = [h for h in (g.get('holds') or []) if h.get('in_basin')]
        if inb:
            o.append(f'<circle class="ring" cx="{x:.1f}" cy="{y:.1f}" '
                     f'r="{r + 4:.1f}"/>')
            nm = ' / '.join(h['name'] for h in inb)
            ly = y - r - 8
            for _ in range(6):
                if all(abs(x - qx) > 78 or abs(ly - qy) > 11 for qx, qy in lab):
                    break
                ly -= 12
            lab.append((x, ly))
            o.append(f'<text class="bslab" x="{x:.1f}" y="{ly:.1f}" '
                     f'text-anchor="middle">{nm}</text>')

    o.append(f'<text class="axl" x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 24}" '
             f'text-anchor="middle">{xlab}</text>')
    o.append(f'<text class="axsub" x="{(PAD_L + W - PAD_R) / 2:.0f}" '
             f'y="{H - 11}" text-anchor="middle">{sub}</text>')
    o.append(f'<text class="axl" transform="translate(12,'
             f'{(PAD_T + H - PAD_B) / 2:.0f}) rotate(-90)" '
             f'text-anchor="middle">{ylab}</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="fig coarsefig">{"".join(o)}</svg>'


panels = ''.join(f'<figure class="mp">{panel(*v)}</figure>' for v in VIEWS)
held = [g for g in G
        if any(h.get('in_basin') for h in (g.get('holds') or []))]
unplaced = [h for g in G for h in (g.get('holds') or [])
            if not h.get('in_basin')]
deep = sorted(G, key=lambda d: d['emin'])[:3]
big = sorted(G, key=lambda d: -d['n'])[:2]

UNPLACED = ''
if unplaced:
    UNPLACED = ('<b>' + ', '.join(sorted({h['name'] for h in unplaced}))
                + '</b> is not ringed anywhere: its nearest member of any basin is '
                + '%.2f' % max(h['rdf'] for h in unplaced)
                + ' away, three times the 0.10 cut, so it belongs to no basin in '
                + 'this landscape.')

legend = ''.join(
    f'<span><i class="sw b{i + 1}"></i>{EMIN + (EMAX - EMIN) * i / 5:.1f}'
    f'&ndash;{EMIN + (EMAX - EMIN) * (i + 1) / 5:.1f}</span>' for i in range(5))

FIG = f'''<h2>The landscape, coarse-grained</h2>
<p class="sub">Inside a basin the optimiser returns the <em>same</em> structure &mdash;
true RDF distance ~0.000, local density in the embedding tracking the real metric at
r&nbsp;=&nbsp;0.91. That discreteness is what lets 1,969 samples become
<b>{len(G)} objects</b>, each drawn once at its own lowest-energy member. The basins
come from clustering RDF distances, and they separate on physical descriptors at
<b>3.8&times;</b> their nearest-neighbour spacing &mdash; two independent routes, the
same objects, so the same points can be read on either kind of axis.</p>
<div class="figkey">
 <span><svg viewBox="0 0 14 14" class="ic"><circle cx="7" cy="7" r="5"/></svg>&gamma;</span>
 <span><svg viewBox="0 0 14 14" class="ic"><rect x="2" y="2" width="10" height="10"/></svg>&beta;</span>
 <span>area = times found</span>
 {legend}
 <span><i class="sw ring"></i>contains a known form</span>
</div>
<div class="mpgrid">{panels}</div>
<p class="caveat"><b>{len(G)} basins, complete linkage at the calibrated 0.10 cut.</b>
Widest basin diameter 0.0937, so every pair inside every basin is within the range
where packing similarity is 100%. This one clustering defines every basin in
this document. Average linkage, the obvious alternative, gave 34 basins but let one
span 0.187 &mdash; wider than any real packing match &mdash; so it is not used here;
the count is genuinely soft between about 30 and 45 depending on that choice.
The deepest basins are {', '.join(str(g['basin']) for g in deep)} at
{deep[0]['emin']:.2f} to {deep[2]['emin']:.2f}&nbsp;kJ/mol; the most-found are
{big[0]['basin']} and {big[1]['basin']} with {big[0]['n']} and {big[1]['n']} hits at
{big[0]['emin']:.2f} and {big[1]['emin']:.2f}. <b>Depth and popularity are
different things</b>, which is visible in every panel: the large markers are not the
dark ones.{' ' + UNPLACED if UNPLACED else ''}</p>'''

open(os.path.join(SP, 'coarse_block.html'), 'w', encoding='utf-8').write(FIG)
print(f"wrote coarse_block.html ({len(FIG):,} bytes), {len(VIEWS)} views, "
      f"{len(G)} basins, {len(held)} holding named structures")
print(f"energy bins: {EMIN:.2f} .. {EMAX:.2f}")
