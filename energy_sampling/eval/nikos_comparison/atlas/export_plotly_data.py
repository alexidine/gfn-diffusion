"""
Compact data bundle for the Plotly rendering of the atlas figures.

Everything the page needs, small enough to inline. The 53,582-structure landscape
is pre-binned here rather than shipped point-by-point -- a 2D histogram is what the
panel draws anyway, and binning server-side keeps the page from carrying 107k floats
to redraw the same picture.

Reads the same intermediates as the SVG renderers, so the two are guaranteed to be
showing the same numbers: coarse_landscape.json is THE basin definition.
"""
import json
import os

import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))

XLO, XHI, YLO, YHI = 0.55, 0.80, -63.5, -30.0
NX, NY = 64, 48

P = json.load(open(os.path.join(SP, 'projection.json')))
G = json.load(open(os.path.join(SP, 'coarse_landscape.json')))
ST = json.load(open(os.path.join(SP, 'relax_states.json')))

pc = np.asarray(P['all_pc'], float)
en = np.asarray(P['all_e'], float)
m = np.isfinite(pc) & np.isfinite(en)
#: clip rather than drop -- an out-of-range structure still happened, and silently
#: dropping it would understate the population the density is meant to show
inx = (pc >= XLO) & (pc <= XHI) & (en >= YLO) & (en <= YHI)
H, xe, ye = np.histogram2d(pc[m & inx], en[m & inx], bins=[NX, NY],
                           range=[[XLO, XHI], [YLO, YHI]])
print(f"landscape: {int(m.sum()):,} finite of {len(pc):,}; "
      f"{int(inx[m].sum()):,} inside the axes ({100 * inx[m].mean():.1f}%), "
      f"{int(H.sum()):,} binned into {NX}x{NY}, max cell {int(H.max())}")

basins = [dict(
    basin=g['basin'], n=g['n'], emin=round(g['emin'], 3),
    pc=round(g['pc'], 4), beta=round(g['beta'], 2),
    theta_std=round(g['theta_std'], 2), theta_med=round(g['theta_med'], 2),
    mds0=round(g['mds'][0], 4), mds1=round(g['mds'][1], 4),
    n_nbr=g['n_nbr'], mean_d=round(g['mean_d'], 3),
    stack=round(g['stack'], 3) if g['stack'] else None,
    aniso=round(g['aniso'], 3), diam=round(g['diam'], 4),
    fam=g['fam'], label=g['label'],
    #: ONLY genuine members. A structure further from its nearest member than the
    #: cut is not in the basin, and labelling the basin with its name puts that
    #: name at the BASIN's energy and density -- which is how nik00001 appeared at
    #: -60.2 here and at its own -43.9 on the landscape panel.
    holds=[h['name'] for h in (g.get('holds') or []) if h['in_basin']],
    holds_rdf=[round(h['rdf'], 4) for h in (g.get('holds') or [])
               if h['in_basin']],
) for g in sorted(G, key=lambda d: d['emin'])]

#: the whole low-energy stratum in the embedding, not just the 42 representatives
#: -- this is what shows how tightly the optimiser reconverges
M = json.load(open(os.path.join(SP, 'lowE_mds.json')))
T = json.load(open(os.path.join(SP, 'tight_regions.json')))
raw = dict(x=[round(float(v), 4) for v in np.asarray(M['pos'])[:, 0]],
           y=[round(float(v), 4) for v in np.asarray(M['pos'])[:, 1]],
           e=[round(float(v), 3) for v in M['e']],
           basin=[int(b) for b in T['basin']],
           stats=T['stats'])
assert len(raw['x']) == len(raw['basin']) == 1969, len(raw['x'])
print(f"raw embedding: {len(raw['x']):,} structures, "
      f"{len(set(raw['basin']))} basins, density r={raw['stats']['r']:.3f}")

UN = json.load(open(os.path.join(SP, 'unplaced.json')))
print(f"unplaced (no basin at the cut): "
      f"{', '.join(u['name'] for u in UN) if UN else 'none'}")

out = dict(
    raw=raw, unplaced=UN,
    dens=dict(z=H.T.astype(int).tolist(),
              x=[round(float(v), 5) for v in (xe[:-1] + np.diff(xe) / 2)],
              y=[round(float(v), 4) for v in (ye[:-1] + np.diff(ye) / 2)],
              n_total=int(m.sum()), n_shown=int(inx[m].sum())),
    basins=basins,
    named=[dict(name=s['name'], kind=s['kind'],
                pc0=round(s['pc0'], 4), pc1=round(s['pc1'], 4),
                e0=round(s['e0'], 3), e1=round(s['e1'], 3)) for s in ST],
    meta=dict(floor=round(min(g['emin'] for g in G), 2),
              n_basins=len(G), n_struct=1969, cut=0.10, linkage='complete'),
)
p = os.path.join(SP, 'plotly_data.json')
json.dump(out, open(p, 'w'))
print(f"wrote plotly_data.json ({os.path.getsize(p):,} bytes): "
      f"{len(basins)} basins, {len(out['named'])} named, "
      f"density {NX}x{NY}")
