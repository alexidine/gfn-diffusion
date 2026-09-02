"""
Acridine packing atlas -- one energy axis, three provenances, P2(1)/c Z'=1.

Each plate is a 3 A SLAB through that structure's measured coordination shell,
viewed DOWN THE REFERENCE MOLECULE'S LONG AXIS onto (short in-plane, normal):

    x = v           offset along the short in-plane axis
    y = -w          offset along the plane normal      -> stack spacing is literal
    bar direction   the molecular plane's own trace in this view
    bar length      that plate's extent along the trace
    fade from |u|   depth within the slab

See the PROJECTION note below for why this view and not the obvious one: the
(long, normal) view hides the herringbone contact in ten of sixteen gamma basins.

⚠ THE ENERGY AXIS MIXES RELAXATION STATES. Our basins and the polymorphs are
relaxed; Nikos' structures are L1 -- reprojected onto our reference conformer and
NEVER relaxed. That difference flatters ours, and the page says so where the numbers
appear rather than leaving the reader to assume parity.
"""
import json
import math
import os

SP = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(SP, 'shells.json')))

VB = 150
SPAN = 8.6
PXA = VB / (2 * SPAN)
LHALF, WHALF = 4.3, 2.5     # acridine half-length and half-width, Angstrom
SLAB = 3.0

#: PROJECTION: down the reference molecule's LONG AXIS, onto (short in-plane, normal).
#:
#: The obvious view -- along the short axis, onto (long, normal) -- HIDES THE
#: HERRINGBONE. Measured: in a 3 A slab of that view only 6 of 16 gamma basins keep
#: an edge-to-face neighbour, so ten of them draw exactly like beta and the picture
#: contradicts its own label. Down the long axis keeps 14 of 16, and is the
#: conventional view for herringbone diagrams for that same reason.
#:
#: Bonus: the long axis now points into the page, so molecules foreshorten to about
#: half length and the higher bar count costs less ink than it sounds.
#:
#:   view              slab   gamma keeping >45deg   median bars   distinct
#:   (long, normal)     3.0        6/16                   4          34/34
#:   (short, normal)    3.0       14/16                   9          34/34   <- used


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def trace(r):
    """
    The molecular plane's silhouette in this projection.

    A flat plate viewed obliquely draws as a line along the intersection of its own
    plane with the view plane: t = f3 x e1. Its half-length is the plate's extent
    along t, from the long and short axes. Returns (dx, dy, half) in Angstrom, or
    None when the molecule is nearly face-on and has no line silhouette.
    """
    f1, f3 = r['L1'], r['L3']
    t = cross(f3, [1.0, 0.0, 0.0])          # lies in the (e2,e3) view plane
    n = math.sqrt(t[1] ** 2 + t[2] ** 2)
    if n < 1e-6:
        return None
    t = [0.0, t[1] / n, t[2] / n]
    f2 = cross(f3, f1)
    half = math.hypot(LHALF * sum(t[i] * f1[i] for i in range(3)),
                      WHALF * sum(t[i] * f2[i] for i in range(3)))
    return t[1], t[2], half


def bar_pts(cx, cy, dx, dy, half):
    hx, hy = dx * half * PXA, dy * half * PXA
    return cx - hx, cy + hy, cx + hx, cy - hy


def plate(shell):
    c = VB / 2
    out = []
    sub = [r for r in shell if abs(r['u']) <= SLAB]
    for r in sorted(sub, key=lambda r: -abs(r['u'])):
        tr = trace(r)
        op = 1.0 - 0.45 * min(abs(r['u']) / SLAB, 1.0)
        x, y = c + r['v'] * PXA, c - r['w'] * PXA
        if tr is None:                       # face-on: no line, show the footprint
            out.append(f'<circle class="face" cx="{x:.1f}" cy="{y:.1f}" '
                       f'r="{WHALF * PXA:.1f}" opacity="{op:.2f}"/>')
            continue
        dx, dy, half = tr
        x1, y1, x2, y2 = bar_pts(x, y, dx, dy, half)
        out.append(f'<line class="nbr" x1="{x1:.1f}" y1="{y1:.1f}" '
                   f'x2="{x2:.1f}" y2="{y2:.1f}" opacity="{op:.2f}"/>')
    #: the reference molecule: its own plane is (e1,e2), normal e3, so seen down e1
    #: it draws as a horizontal bar of its half-WIDTH
    out.append(f'<line class="ref" x1="{c - WHALF * PXA:.1f}" y1="{c:.1f}" '
               f'x2="{c + WHALF * PXA:.1f}" y2="{c:.1f}"/>')
    return (f'<svg viewBox="0 0 {VB} {VB}" role="img" aria-hidden="true">'
            f'<g class="mol">{"".join(out)}</g></svg>')


N_LOW = 10      # deepest basins to show
N_TOP = 3       # most-captured basins to show (union with the above)
E_LO, E_HI = -63.5, -44.0     # common energy axis for the position ticks

FAM = {'gamma': 'γ', 'beta': 'β', 'herringbone': 'H', 'sheet': 'S'}


def slip_of(shell):
    st = [r for r in shell if r['theta'] < 25 and 3.1 <= r['h'] <= 3.9]
    return min((r['s'] for r in st), default=None)


def n_slab(shell):
    return len([r for r in shell if abs(r['u']) <= SLAB])


def epos(e):
    return max(0.0, min(100.0, 100 * (e - E_LO) / (E_HI - E_LO)))


def card(kind, title, e, shell, stack, fam, rows_html, sub=''):
    slip = slip_of(shell)
    n_in = n_slab(shell)
    return f'''<figure class="plate {kind}">
 <div class="art">{plate(shell)}</div>
 <figcaption>
  <div class="hdr"><span class="who">{title}</span><span class="fam">{FAM[fam]}</span></div>
  {f'<div class="sub2">{sub}</div>' if sub else ''}
  <div class="energy"><span class="ev">{e:.2f}</span><span class="eu">kJ/mol</span></div>
  <div class="escale"><i style="left:{epos(e):.1f}%"></i></div>
  <dl>
   <div><dt>stack</dt><dd>{f"{stack:.2f}" if stack else "—"}<span class="unit">Å</span></dd></div>
   <div><dt>slip</dt><dd>{f"{slip:.2f}" if slip is not None else "—"}<span class="unit">Å</span></dd></div>
   {rows_html}
   <div><dt>in slab</dt><dd>{n_in}<span class="unit">of {len(shell)}</span></dd></div>
  </dl>
 </figcaption>
</figure>'''


# ---- the plate grid, from the SAME 42 basins the coarse figure uses --------
#: was built from shells.json (34 basins, AVERAGE linkage, one of them spanning
#: 0.187 -- wider than any real packing match). The document now uses ONE basin
#: definition throughout: complete linkage at the calibrated 0.10 cut, which bounds
#: every basin's diameter by the cut.
CG = json.load(open(os.path.join(SP, 'coarse_landscape.json')))
by_id = {g['basin']: g for g in CG}

#: a card is a "known form" card only if the basin CONTAINS the structure;
#: nearest-but-outside is a different claim and gets said differently below
held = [g for g in CG
        if any(h.get('in_basin') for h in (g.get('holds') or []))]
held_ids = {g['basin'] for g in held}
deep = [g for g in sorted(CG, key=lambda d: d['emin'])
        if g['basin'] not in held_ids][:6]
deep_ids = held_ids | {g['basin'] for g in deep}
pop = [g for g in sorted(CG, key=lambda d: -d['n'])
       if g['basin'] not in deep_ids][:6]
chosen = held + deep + pop
erank = {g['basin']: i + 1 for i, g in enumerate(sorted(CG, key=lambda d: d['emin']))}
nrank = {g['basin']: i + 1 for i, g in enumerate(sorted(CG, key=lambda d: -d['n']))}

items = []
for g in chosen:
    holds = g.get('holds') or []
    kind = 'ours'
    sub = 'our search &middot; %d hits &middot; %.1f%%' % (
        g['n'], 100 * g['n'] / 1969)
    title = 'basin %d' % g['basin']
    if holds:
        names = ' = '.join(h['name'] for h in holds)
        kind = 'known' if any(h['kind'] == 'known' for h in holds) else 'nik'
        far = [h for h in holds if not h.get('in_basin')]
        #: a structure 0.30 from its nearest basin member is NOT in that basin --
        #: say so on the card rather than let the assignment imply membership
        sub = names if not far else (
            '%s &middot; NOT a member &mdash; nearest is %.2f away, past the '
            '0.10 cut' % (names, far[0]['rdf']))
    rows = ('<div><dt>captures</dt><dd>%s</dd></div>'
            '<div><dt>rank E / n</dt><dd>%d / %d</dd></div>'
            % (format(g['n'], ','), erank[g['basin']], nrank[g['basin']]))
    items.append((g['emin'], card(kind, title, g['emin'], g['shell'],
                                  g['stack'], g['fam'], rows, sub=sub)))

items.sort(key=lambda t: t[0])
grid = ''.join(h for _, h in items)

other = [c for c in D['controls'] if not (c['sg'] == 14 and c['zp'] == 1)]
others = ''.join(f'''<figure class="known small">
 <div class="art">{plate(c['shell'])}</div>
 <figcaption><b>{c['name']}</b>
 <em>sg{c['sg']} Z&prime;{c['zp']} · {FAM[c['fam']]}</em>
 <em class="dim">{c['e']:.2f} kJ/mol{' · BROKEN' if c['e'] > -40 else ''}</em>
 </figcaption></figure>''' for c in other)

CSS = """
:root{
  --ink:#14181B; --ground:#EDF0EE; --panel:#F7F9F7; --rule:#C6CCC8;
  --muted:#636D68; --gold:#8A6A12; --gold-line:#B0891F;
  --teal:#2F6165; --teal-line:#4E8A8E;
  --mol:#2A3236; --ref:#0B0E10;
  --e1:#7A4B12; --e2:#A9761F; --e3:#C9A24B; --e4:#9DAAA6; --e5:#C3CCC8;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ink:#E5E9E6; --ground:#111519; --panel:#181F24; --rule:#2E363C;
    --muted:#8D9994; --gold:#E2BE5C; --gold-line:#D4AE4A;
    --teal:#79C0C4; --teal-line:#4E8A8E; --mol:#9FB0B8; --ref:#F2F6F4;
    --e1:#F0C669; --e2:#C79A3C; --e3:#8E7A3E; --e4:#5E6E72; --e5:#3A464C;
  }
}
:root[data-theme="dark"]{
  --ink:#E5E9E6; --ground:#111519; --panel:#181F24; --rule:#2E363C;
  --muted:#8D9994; --gold:#E2BE5C; --gold-line:#D4AE4A;
  --teal:#79C0C4; --teal-line:#4E8A8E; --mol:#9FB0B8; --ref:#F2F6F4;
    --e1:#F0C669; --e2:#C79A3C; --e3:#8E7A3E; --e4:#5E6E72; --e5:#3A464C;
}
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1240px; margin:0 auto;
  padding:clamp(2rem,5vw,4.5rem) clamp(1rem,4vw,2.5rem) 5rem}
header{border-bottom:1px solid var(--rule); padding-bottom:2rem}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.72rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin:0 0 1rem}
h1{font-family:Spectral,Georgia,serif; font-weight:600;
  font-size:clamp(2rem,5vw,3.05rem); line-height:1.08; margin:0 0 1rem;
  text-wrap:balance; letter-spacing:-.015em}
.lede{max-width:64ch; font-size:1.05rem; margin:0}
.meta{display:flex; flex-wrap:wrap; gap:1.9rem;
  font-family:"IBM Plex Mono",monospace; font-size:.78rem; color:var(--muted);
  margin-top:1.75rem}
.meta b{display:block; font-size:1.3rem; color:var(--ink); font-weight:500;
  font-variant-numeric:tabular-nums; line-height:1.25}
h2{font-family:Spectral,Georgia,serif; font-weight:600; font-size:1.45rem;
  margin:3.25rem 0 .4rem; letter-spacing:-.01em}
.sub{color:var(--muted); max-width:70ch; margin:0 0 1.4rem; font-size:.93rem}
.key{display:flex; flex-wrap:wrap; gap:1.25rem; margin:0 0 1.4rem;
  font-family:"IBM Plex Mono",monospace; font-size:.78rem}
.key span{display:flex; align-items:center; gap:.45rem; color:var(--muted)}
.key i{width:22px; height:3px; display:block}
.k-ours{background:var(--mol)} .k-known{background:var(--gold-line)}
.k-nikos{background:var(--teal-line)}
svg{display:block; width:100%; height:auto}
.mol line{stroke-linecap:round; fill:none}
.mol circle{fill:none; stroke:var(--mol); stroke-width:2.4}
.mol line.nbr{stroke:var(--mol); stroke-width:3.2}
.mol line.ref{stroke:var(--ref); stroke-width:5}
.art{background:var(--ground); border:1px solid var(--rule)}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(216px,1fr));
  gap:1.2rem}
.plate{margin:0; display:flex; flex-direction:column; gap:.55rem}
figcaption{display:flex; flex-direction:column; gap:.45rem}
.hdr{display:flex; justify-content:space-between; align-items:baseline; gap:.4rem}
.who{font-family:"IBM Plex Mono",monospace; font-size:.82rem; font-weight:500}
.fam{font-family:Spectral,Georgia,serif; font-size:1.3rem; font-weight:600;
  line-height:1}
.sub2{font-size:.7rem; color:var(--muted); line-height:1.35}
.energy{display:flex; align-items:baseline; gap:.3rem}
.ev{font-family:"IBM Plex Mono",monospace; font-size:1.22rem; font-weight:500;
  font-variant-numeric:tabular-nums}
.eu{font-size:.66rem; color:var(--muted)}
.escale{position:relative; height:3px; background:var(--rule)}
.escale i{position:absolute; top:-2px; width:3px; height:7px; background:var(--mol)}
.plate.known .art{border-color:var(--gold-line)}
.plate.known .mol line.ref{stroke:var(--gold)}
.plate.known .fam,.plate.known .ev,.plate.known .who{color:var(--gold)}
.plate.known .escale i{background:var(--gold-line)}
.plate.nikos .art{border-color:var(--teal-line)}
.plate.nikos .mol line.ref{stroke:var(--teal)}
.plate.nikos .fam,.plate.nikos .ev,.plate.nikos .who{color:var(--teal)}
.plate.nikos .escale i{background:var(--teal-line)}
dl{margin:0; display:grid; grid-template-columns:1fr 1fr; gap:.35rem .7rem}
dl div{display:flex; flex-direction:column}
dt{font-size:.62rem; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted)}
dd{margin:0; font-family:"IBM Plex Mono",monospace; font-size:.82rem;
  font-variant-numeric:tabular-nums}
.unit{color:var(--muted); font-size:.66rem; margin-left:.25rem}
.axes{display:grid; grid-template-columns:repeat(auto-fit,minmax(238px,1fr));
  gap:1.6rem; background:var(--panel); border:1px solid var(--rule); padding:1.5rem}
.axes h3{font-size:.9rem; margin:0 0 .3rem; font-weight:600}
.axes p{margin:0; font-size:.85rem; color:var(--muted)}
.axes .art{max-width:164px; margin-bottom:.7rem}
.knowns{display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
  gap:1rem}
.known.small figcaption{font-size:.76rem; display:flex; flex-direction:column;
  gap:.08rem; padding-top:.4rem}
.known.small b{font-family:"IBM Plex Mono",monospace; font-size:.74rem}
.known.small em{font-style:normal; color:var(--muted);
  font-family:"IBM Plex Mono",monospace; font-size:.7rem}
.known.small em.dim{opacity:.75}
.caveat{background:var(--panel); border-left:3px solid var(--teal-line);
  padding:1rem 1.25rem; margin:1.4rem 0 0; font-size:.89rem; max-width:72ch}
.caveat b{font-weight:600}
.qa{max-width:74ch; margin:0 0 1.5rem; padding-left:1.15rem; font-size:.92rem}
.qa li{margin:0 0 .7rem; line-height:1.55}
.qa li::marker{color:var(--muted); font-variant-numeric:tabular-nums}
.qa b{font-weight:600}
.tbl{border-collapse:collapse; font-size:.84rem; margin:.2rem 0 0;
  font-variant-numeric:tabular-nums}
.tbl th{text-align:left; font-weight:600; color:var(--muted); padding:.3rem .8rem;
  border-bottom:1px solid var(--rule); white-space:nowrap;
  font-size:.72rem; letter-spacing:.06em; text-transform:uppercase}
.tbl td{padding:.3rem .8rem; border-bottom:1px solid var(--rule)}
.tbl tr:last-child td{border-bottom:none}
.tbl .num{text-align:right; font-family:"IBM Plex Mono",monospace}
.tbl .mono{font-family:"IBM Plex Mono",monospace; color:var(--muted)}
.tbl .warn{color:var(--gold); font-weight:600}
.note{margin-top:3.25rem; padding-top:1.4rem; border-top:1px solid var(--rule);
  color:var(--muted); font-size:.88rem; max-width:72ch}
.note b{color:var(--ink); font-weight:600}
.figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr));
  gap:1.5rem}
.fw{margin:0; background:var(--panel); border:1px solid var(--rule); padding:1rem}
.fw figcaption{font-size:.83rem; color:var(--muted); margin-top:.6rem;
  display:block; line-height:1.5}
.fw figcaption b{color:var(--ink); font-weight:600}
svg.fig{width:100%; height:auto; overflow:visible}
.fig .panel{fill:var(--ground); stroke:var(--rule); stroke-width:1}
.fig .grid{stroke:var(--rule); stroke-width:.5; opacity:.6}
.fig .tick,.fig .axlab{fill:var(--muted); font-family:"IBM Plex Mono",monospace}
.fig .tick{font-size:10px} .fig .axlab{font-size:11px}
.fig .cell{fill:var(--mol)}
.fig .gline{stroke:var(--gold-line); stroke-width:1.2; stroke-dasharray:5 3}
.fig .floor{stroke:var(--mol); stroke-width:1.2; stroke-dasharray:2 3}
.fig .glab{fill:var(--gold); font-size:10px;
  font-family:"IBM Plex Mono",monospace}
.fig .glab.floorlab{fill:var(--muted)}
.fig .pt{stroke-width:1.4}
.fig .pt.ours{fill:var(--mol); stroke:var(--ref); fill-opacity:.55}
.fig .pt.known{fill:var(--gold-line); stroke:var(--gold); stroke-width:2}
.fig .pt.nik{fill:var(--teal-line); stroke:var(--teal); stroke-width:2}
.fig .ptlab{fill:var(--muted); font-size:9px;
  font-family:"IBM Plex Mono",monospace}
.fig .namelab{font-size:10px; font-weight:600;
  font-family:"IBM Plex Mono",monospace; paint-order:stroke;
  stroke:var(--panel); stroke-width:3px}
.fig .namelab.known{fill:var(--gold)}
.fig .namelab.nik{fill:var(--teal)}
.figkey{display:flex; flex-wrap:wrap; gap:.9rem; margin-top:.7rem;
  font-family:"IBM Plex Mono",monospace; font-size:.72rem; color:var(--muted)}
.figkey span{display:flex; align-items:center; gap:.35rem}
.figkey .ic{width:12px; height:12px; flex:none}
.figkey .ic circle,.figkey .ic rect{fill:none; stroke:var(--mol);
  stroke-width:1.8}
.figkey .sw{width:11px; height:11px; display:block; border-radius:50%}
.figkey .sw.ours{background:var(--mol); opacity:.6}
.figkey .sw.known{background:var(--gold-line)}
.figkey .sw.nik{background:var(--teal-line)}
.figs.one{grid-template-columns:1fr; max-width:660px}
.mpgrid{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:1.1rem}
.mp{margin:0; background:var(--panel); border:1px solid var(--rule); padding:.6rem}
.mdsfig .dot{fill:var(--mol)}
.mdsfig .mk{stroke-width:2; fill-opacity:.85}
.mdsfig .mk.known{fill:var(--gold-line); stroke:var(--gold)}
.mdsfig .mk.nik{fill:var(--teal-line); stroke:var(--teal)}
.mdsfig .mklab{font-size:9.5px; font-weight:600;
  font-family:"IBM Plex Mono",monospace; paint-order:stroke;
  stroke:var(--panel); stroke-width:3px}
.mdsfig .mklab.known{fill:var(--gold)}
.mdsfig .mklab.nik{fill:var(--teal)}
.mdsfig .axl{fill:var(--muted); font-size:10px;
  font-family:"IBM Plex Sans",sans-serif}
.mdsfig .axend{fill:var(--muted); font-size:8.5px; opacity:.8;
  font-family:"IBM Plex Mono",monospace}
.mdsfig .lead{stroke:var(--muted); stroke-width:1; opacity:.5}
.landfig .mk{stroke-width:2; fill-opacity:.85}
.landfig .mk.known{fill:var(--gold-line); stroke:var(--gold)}
.landfig .mk.nik{fill:var(--teal-line); stroke:var(--teal)}
.landfig .mklab{font-size:10px; font-weight:600;
  font-family:"IBM Plex Mono",monospace; paint-order:stroke;
  stroke:var(--panel); stroke-width:3px}
.landfig .mklab.known{fill:var(--gold)}
.landfig .mklab.nik{fill:var(--teal)}
.landfig .lead{stroke:var(--muted); stroke-width:1; opacity:.5}
.landfig .offscale{fill-opacity:1; stroke-width:1.5}
.landfig .relax,.mdsfig .relax{stroke:var(--muted); stroke-width:1.4;
  opacity:.75; stroke-dasharray:3 2}
.landfig .hollow,.mdsfig .hollow{fill:none; stroke-width:1.8}
.landfig .offlab{font-size:9px; font-family:"IBM Plex Mono",monospace;
  paint-order:stroke; stroke:var(--panel); stroke-width:3px}
.landfig .offlab.known{fill:var(--gold)}
.landfig .offlab.nik{fill:var(--teal)}
.coarsefig .bs{stroke:var(--panel); stroke-width:1.2}
.coarsefig .bs.b1{fill:var(--e1)}
.coarsefig .bs.b2{fill:var(--e2)}
.coarsefig .bs.b3{fill:var(--e3)}
.coarsefig .bs.b4{fill:var(--e4)}
.coarsefig .bs.b5{fill:var(--e5)}
.coarsefig .ring{fill:none; stroke:var(--gold); stroke-width:2}
.coarsefig .bslab{fill:var(--gold); font-size:9.5px; font-weight:600;
  font-family:"IBM Plex Mono",monospace; paint-order:stroke;
  stroke:var(--panel); stroke-width:3px}
.coarsefig .axl{fill:var(--ink); font-size:10.5px;
  font-family:"IBM Plex Sans",sans-serif}
.coarsefig .axsub{fill:var(--muted); font-size:9px; font-style:italic;
  font-family:"IBM Plex Sans",sans-serif}
.figkey .sw.b1{background:var(--e1)}
.figkey .sw.b2{background:var(--e2)}
.figkey .sw.b3{background:var(--e3)}
.figkey .sw.b4{background:var(--e4)}
.figkey .sw.b5{background:var(--e5)}
.figkey .sw.ring{background:none; border:2px solid var(--gold)}
.landfig .mk.both{fill:var(--gold-line); stroke:var(--gold)}
.landfig .mklab.both{fill:var(--gold)}
.mdsfig .mk.both{fill:var(--gold-line); stroke:var(--gold)}
.mdsfig .mklab.both{fill:var(--gold)}
"""

ex = by_id
axes = f"""<div class="axes">
 <div><div class="art">{plate((sorted(CG, key=lambda d: -d['n'])[0])['shell'])}</div>
  <h3>Read it as a projection</h3>
  <p>The heavy bar at centre is the reference molecule seen edge-on. Every other bar
  is a real neighbour at its measured position &mdash; nothing is idealised.</p></div>
 <div><div class="art">{plate((sorted(CG, key=lambda d: d['emin'])[0])['shell'])}</div>
  <h3>Up is the stack, across is the slip</h3>
  <p>The view looks down the reference molecule&rsquo;s long axis, so bars that cross
  each other are the edge-to-face herringbone contact. Vertical offset is separation
  along the plane normal.</p></div>
 <div><div class="art">{plate((max(CG, key=lambda d: len(d['shell'])))['shell'])}</div>
  <h3>A 3&nbsp;&Aring; slab, not the whole shell</h3>
  <p>Only molecules within 3&nbsp;&Aring; of the view plane are drawn &mdash; a slab
  thin enough to stay readable, thick enough that 14 of 16 &gamma; basins still show
  their herringbone partner.</p></div>
</div>"""

FIGBLOCK = open(os.path.join(SP, 'figure_block.html'),
                encoding='utf-8').read()
MDSBLOCK = open(os.path.join(SP, 'mds_block.html'),
                encoding='utf-8').read()
COARSE = open(os.path.join(SP, 'coarse_block.html'),
              encoding='utf-8').read()
CLUMPS = open(os.path.join(SP, 'clumps_block.html'),
              encoding='utf-8').read()

html = f"""<title>Acridine Packing Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&amp;family=IBM+Plex+Mono:wght@400;500&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
 <p class="eyebrow">acridine &middot; P2&#8321;/c &middot; Z&prime; = 1 &middot; MACE</p>
 <h1>Forty-two ways to stack a flat molecule</h1>
 <p class="lede">The low-energy landscape of acridine in one space group, enumerated
 to 99.7% of its probability mass. Every packing here is &pi;-stacked at
 3.2&ndash;3.5&nbsp;&Aring; &mdash; the stack never varies. What varies is the slip,
 the tilt between stacks, and the energy.</p>
 <div class="meta">
  <span>basins found<b>42</b></span>
  <span>structures<b>1,969</b></span>
  <span>mass covered<b>99.7%</b></span>
  <span>&pi;-stack<b>3.21&ndash;3.47&nbsp;&Aring;</b></span>
  <span>energy floor<b>&minus;62.81</b></span>
 </div>
</header>

<h2>How to read a plate</h2>
<p class="sub">Acridine is flat and elongated, so edge-on it collapses to a bar and
its long axis gives a real in-plane reference. Each plate looks along that frame.</p>
{axes}

<h2>The basins, drawn</h2>
<p class="sub">Every basin holding a named structure, plus the six deepest and six most-found of the rest &mdash; ordered by energy. Each plate is a 3&nbsp;&Aring; slab through that basin&rsquo;s lowest-energy member, viewed down the molecular long axis. The tick under each energy places it on a common &minus;63.5 to &minus;44 scale.</p>
<div class="key">
 <span><i class="k-ours"></i>our search</span>
 <span><i class="k-known"></i>experimental (CSD)</span>
 <span><i class="k-nikos"></i>Nikos&rsquo; proposals</span>
</div>
<div class="grid">{grid}</div>
<p class="caveat"><b>The energies are not strictly comparable.</b> Our basins and the
experimental forms are relaxed; Nikos&rsquo; three are L1 &mdash; reprojected onto our
reference conformer but never relaxed. That gap flatters ours, and the rigid-body
relaxation that would close it has not been run. Read his numbers as upper bounds,
not as a ranking against ours.</p>

{FIGBLOCK}

{COARSE}

{CLUMPS}

{MDSBLOCK}

<h2>The other known forms, for scale</h2>
<p class="sub">Outside this space group, so not part of the landscape above &mdash;
but they validate the descriptor: drawn by the same routine, the experimental
structures return &pi;-stacks of 3.38&ndash;3.50&nbsp;&Aring; with nothing fitted, and
the three unstacked herringbone forms come back correctly unstacked.</p>
<div class="knowns">{others}</div>

<p class="note"><b>The widest basin is not the deepest.</b> Basin 21 caught 1,167 of
1,969 structures yet sits 2.0&nbsp;kJ/mol above the floor, while the global minimum
&mdash; basin 17 &mdash; was found four times. Depth and catchment volume are close
to uncorrelated here, which is why the two real forms being rare (5 and 19 captures)
is unremarkable rather than suspicious. <b>ACRDIN08 is broken:</b> it scores
&minus;16.91 against &minus;55 to &minus;60 for every other known form, and moves 2.8&nbsp;&Aring;
in cell length under relaxation. Do not use it as a reference.</p>
</div>
"""

out_path = os.path.join(SP, 'acridine_packing_atlas.html')
with open(out_path, 'w', encoding='utf-8') as fh:
    fh.write(html)
print(f"wrote {out_path} ({len(html):,} bytes); {len(items)} cards in the main grid")
