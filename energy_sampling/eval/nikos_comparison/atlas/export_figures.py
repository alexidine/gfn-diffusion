"""
Static PNGs of the five atlas figures, via plotly.py + kaleido.

Same data and same encoding as the interactive page (page_charts.js) -- both read
plotly_data.json, so the PNGs and the artifact cannot show different numbers. The
one deliberate difference: a PNG has no hover, so every named structure keeps an
arrowed callout and nothing relies on pointing at it.

Encoding, constant across the four basin panels:
    colour = basin minimum energy   (sequential blue, deepest = darkest)
    area   = times the search found it
    symbol = packing family, gamma circle / beta square

Writes light-background figures by default (--dark for the dark set) at scale 2,
so a 900px-wide figure lands at 1800px -- enough for print.
"""
import argparse
import json
import os

import plotly.graph_objects as go

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, 'figures')

#: Two ramps, both documented steps of the one blue scale, near-surface end first.
#: FIELD is the continuous density: it runs the full range because an empty cell
#: SHOULD recede into the surface. MARK is for discrete points, where the palest
#: step would leave a real datum nearly invisible -- so it obeys the ordinal floor
#: (light: no lighter than step 250; dark: no darker than step 600).
FIELD_LIGHT = ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#184f95', '#0d366b']
FIELD_DARK = ['#0d366b', '#104281', '#1c5cab', '#2a78d6', '#5598e7', '#9ec5f4']
MARK_LIGHT = ['#86b6ef', '#5598e7', '#3987e5', '#2a78d6', '#184f95', '#0d366b']
MARK_DARK = ['#184f95', '#1c5cab', '#256abf', '#3987e5', '#5598e7', '#9ec5f4']

THEMES = dict(
    light=dict(surface='#fcfcfb', card='#ffffff', grid='#e8e8e3',
               ink='#0b0b0b', ink2='#52514e', ink3='#7d7c76',
               known='#eb6834', nik='#1baf7a',
               ramp=FIELD_LIGHT, mark=MARK_LIGHT),
    dark=dict(surface='#1a1a19', card='#201f1e', grid='#2f2f2c',
              ink='#ffffff', ink2='#c3c2b7', ink3='#8d8c83',
              known='#d95926', nik='#199e70',
              ramp=FIELD_DARK, mark=MARK_DARK),
)

FONT = 'IBM Plex Sans, Helvetica, Arial, sans-serif'
MONO = 'IBM Plex Mono, Consolas, monospace'
STAGGER = [34, 62, 90, 118, 146]

D = json.load(open(os.path.join(SP, 'plotly_data.json'), encoding='utf-8'))
BAS = D['basins']
NMAX = max(b['n'] for b in BAS)
EMIN = min(b['emin'] for b in BAS)
EMAX = max(b['emin'] for b in BAS)


def escale(t):
    r = t['mark']
    return [[i / (len(r) - 1), r[len(r) - 1 - i]] for i in range(len(r))]


def axis(t, title, **kw):
    a = dict(title=dict(text=title, font=dict(size=13, color=t['ink2'])),
             gridcolor=t['grid'], zeroline=False, linecolor=t['grid'],
             ticks='outside', tickcolor=t['grid'],
             tickfont=dict(color=t['ink3'], size=11))
    a.update(kw)
    return a


def base(t, xt, yt, **kw):
    L = dict(paper_bgcolor=t['card'], plot_bgcolor=t['card'],
             font=dict(family=FONT, size=12, color=t['ink2']),
             margin=dict(l=74, r=26, t=38, b=58),
             xaxis=axis(t, xt), yaxis=axis(t, yt),
             legend=dict(orientation='h', y=1.02, yanchor='bottom', x=0,
                         font=dict(size=11.5, color=t['ink2']),
                         bgcolor='rgba(0,0,0,0)'))
    L.update(kw)
    return L


def callouts(items, t, ymid=None):
    """Arrowed labels, staggered, pushed toward the middle of the panel.

    Four of the five named structures relax into a 0.8 kJ/mol neighbourhood, so
    plain text marks overlap whichever side they are placed on and plotly does no
    collision avoidance. Two rules keep them readable: the label goes INWARD (a
    point in the lower half gets its label above, and vice versa) so nothing is
    pushed off an edge, and within each half the offsets stagger so labels of
    near-identical points do not stack on each other.
    """
    if ymid is None and items:
        ys = [o['y'] for o in items]
        ymid = (min(ys) + max(ys)) / 2
    lower = [o for o in sorted(items, key=lambda o: o['y']) if o['y'] <= ymid]
    upper = [o for o in sorted(items, key=lambda o: -o['y']) if o['y'] > ymid]
    out = []
    for group, sign in ((lower, -1), (upper, 1)):
        for i, o in enumerate(group):
            out.append(dict(
                x=o['x'], y=o['y'], text=o['text'], showarrow=True,
                ax=0, ay=sign * STAGGER[i % len(STAGGER)],
                arrowhead=0, arrowwidth=1,
                arrowcolor=o.get('color') or t['ink3'], standoff=5,
                font=dict(family=MONO, size=10,
                          color=o.get('color') or t['ink2']),
                bgcolor=t['card'], borderpad=2, opacity=0.97))
    return out


def basin_fig(t, xk, yk, xt, yt, xkw=None, **kw):
    fig = go.Figure()
    for fam, sym, nm in (('gamma', 'circle', 'γ  stack + herringbone'),
                         ('beta', 'square', 'β  stacked layers')):
        S = [b for b in BAS if b['fam'] == fam]
        fig.add_trace(go.Scatter(
            x=[b[xk] for b in S], y=[b[yk] for b in S], mode='markers', name=nm,
            marker=dict(symbol=sym, size=[b['n'] for b in S], sizemode='area',
                        sizeref=(2.0 * NMAX) / (34 ** 2), sizemin=5,
                        color=[b['emin'] for b in S], colorscale=escale(t),
                        cmin=EMIN, cmax=EMAX,
                        line=dict(color=t['card'], width=1.5),
                        showscale=(fam == 'gamma'),
                        colorbar=dict(
                            title=dict(text='min E<br>kJ/mol',
                                       font=dict(size=11, color=t['ink3'])),
                            thickness=11, len=0.62, outlinewidth=0,
                            tickfont=dict(size=10, color=t['ink3'],
                                          family=MONO)))))
    #: holds already excludes non-members (export_plotly_data.py gates on
    #: in_basin), so a ring here always means "this basin contains that structure"
    held = [b for b in BAS if b['holds']]
    fig.add_trace(go.Scatter(
        x=[b[xk] for b in held], y=[b[yk] for b in held], mode='markers',
        showlegend=False, hoverinfo='skip',
        marker=dict(symbol='circle-open', size=26, color=t['ink3'],
                    line=dict(width=1.4))))
    L = base(t, xt, yt, **kw)
    if xkw:
        L['xaxis'] = axis(t, xt, **xkw)
    L['annotations'] = callouts(
        [dict(x=b[xk], y=b[yk], text=' = '.join(b['holds'])) for b in held], t)
    if D.get('unplaced'):
        #: the note lives below the x-axis title, so the bottom margin has to
        #: grow or plotly clips it silently
        L['margin'] = dict(L['margin'], b=92)
    for u in D.get('unplaced', []):
        L['annotations'].append(dict(
            xref='paper', yref='paper', x=1.0, y=-0.155, xanchor='right',
            yanchor='top', showarrow=False,
            text=(f"{u['name']} matches no basin at the 0.10 cut "
                  f"(nearest member {u['rdf']:.2f} away) and is not drawn"),
            font=dict(family=MONO, size=9.5, color=t['ink3'])))
    fig.update_layout(**L)
    return fig


def raw_embedding_fig(t, zoom=None):
    """All 1,969 low-energy structures in the embedding, not the 42 representatives.

    The point of drawing every one: within a basin the optimiser returns the SAME
    structure, so hundreds of points land on top of each other and the knots ARE the
    basins. Local density here tracks density in the true RDF metric at r = 0.911,
    so the tightness is a real measurement and not a projection artifact.
    """
    R = D['raw']
    fig = go.Figure(go.Scattergl(
        x=R['x'], y=R['y'], mode='markers',
        marker=dict(size=5, opacity=0.55,
                    color=R['e'], colorscale=escale(t), cmin=EMIN, cmax=EMAX,
                    line=dict(width=0),
                    colorbar=dict(title=dict(text='lattice E<br>kJ/mol',
                                             font=dict(size=11, color=t['ink3'])),
                                  thickness=11, len=0.62, outlinewidth=0,
                                  tickfont=dict(size=10, color=t['ink3'],
                                                family=MONO)))))
    n = len(R['x'])
    sub = (f"{n:,} structures &#183; {len(set(R['basin']))} basins &#183; "
           f"local density vs true RDF metric r = {R['stats']['r']:.3f}")
    L = base(t, 'RDF embedding dim 1', 'dim 2',
             margin=dict(l=74, r=26, t=52, b=58), showlegend=False)
    L['annotations'] = [dict(xref='paper', yref='paper', x=0, y=1.035,
                             xanchor='left', yanchor='bottom',
                             text=sub.replace('&#183;', '·'),
                             showarrow=False,
                             font=dict(family=MONO, size=11, color=t['ink3']))]
    if zoom:
        L['xaxis'] = axis(t, 'RDF embedding dim 1', range=zoom[0])
        L['yaxis'] = axis(t, 'dim 2', range=zoom[1])
    fig.update_layout(**L)
    return fig


def landscape_fig(t):
    YLO, YHI = -64.2, -28.5
    fig = go.Figure(go.Heatmap(
        x=D['dens']['x'], y=D['dens']['y'], z=D['dens']['z'],
        colorscale=([[0.0, t['card']], [0.0001, t['ramp'][0]]]
                    + [[0.0001 + (1 - 0.0001) * i / (len(t['ramp']) - 1), c]
                       for i, c in enumerate(t['ramp'])][1:]),
        zsmooth=False,
        colorbar=dict(title=dict(text='structures<br>per cell',
                                 font=dict(size=11, color=t['ink3'])),
                      thickness=11, len=0.62, outlinewidth=0,
                      tickfont=dict(size=10, color=t['ink3'], family=MONO))))
    #: co-located structures share one marker and one label -- nik00000 relaxes
    #: onto ACRDIN12 exactly, and two labels there would imply two basins
    merged, note, seen, drawn = {}, [], set(), set()
    for s in D['named']:
        merged.setdefault((round(s['pc1'], 3), round(s['e1'], 2)), []).append(s['name'])
    for s in sorted(D['named'], key=lambda d: d['e1']):
        col = t['known'] if s['kind'] == 'known' else t['nik']
        lab = 'experimental (CSD)' if s['kind'] == 'known' else "Nikos' proposals"
        off = s['e0'] > YHI                       # starts above the panel
        #: park it just inside the top rather than exactly on YHI -- a marker
        #: centred on the boundary is half-clipped and its label lands in the
        #: legend strip
        y0 = YHI - 0.9 if off else s['e0']
        fig.add_trace(go.Scatter(x=[s['pc0'], s['pc1']], y=[y0, s['e1']],
                                 mode='lines', showlegend=False, hoverinfo='skip',
                                 line=dict(color=col, width=1.4, dash='dot')))
        fig.add_trace(go.Scatter(
            x=[s['pc0']], y=[y0],
            mode='markers+text' if off else 'markers',
            text=[f"  {s['name']} unrelaxed {s['e0']:.1f} ↑"] if off else None,
            textposition='middle right',
            textfont=dict(family=MONO, size=10, color=col),
            marker=dict(symbol='triangle-up-open' if off else 'circle-open',
                        size=12 if off else 10, color=col, line=dict(width=1.8)),
            name=lab, legendgroup=lab, showlegend=lab not in seen))
        seen.add(lab)
        key = (round(s['pc1'], 3), round(s['e1'], 2))
        if key not in drawn:
            note.append(dict(x=s['pc1'], y=s['e1'],
                             text=' = '.join(merged[key]), color=col))
            drawn.add(key)
        fig.add_trace(go.Scatter(
            x=[s['pc1']], y=[s['e1']], mode='markers', showlegend=False,
            legendgroup=lab,
            marker=dict(symbol='circle', size=11, color=col,
                        line=dict(color=t['card'], width=1.5))))
    floor = D['meta']['floor']
    L = base(t, 'packing coefficient', 'MACE lattice energy  (kJ/mol)',
             xaxis=axis(t, 'packing coefficient', range=[0.55, 0.80]),
             yaxis=axis(t, 'MACE lattice energy  (kJ/mol)', range=[YLO, YHI]),
             shapes=[dict(type='line', xref='paper', x0=0, x1=1,
                          y0=floor, y1=floor,
                          line=dict(color=t['ink3'], width=1, dash='dash'))])
    L['annotations'] = [dict(xref='paper', x=0.012, y=floor, xanchor='left',
                             yanchor='bottom', text=f'energy floor {floor}',
                             showarrow=False,
                             font=dict(family=MONO, size=10, color=t['ink3']))
                        ] + callouts(note, t)
    fig.update_layout(**L)
    return fig


FIGS = [
    ('01_landscape', 1020, 640, lambda t: landscape_fig(t)),
    ('06_raw_embedding', 940, 700, lambda t: raw_embedding_fig(t)),
    ('07_raw_embedding_zoom', 940, 700, lambda t: raw_embedding_fig(
        t, zoom=[[0.075, 0.135], [-0.02, 0.07]])),
    ('02_depth_vs_density', 940, 580, lambda t: basin_fig(
        t, 'pc', 'emin', 'packing coefficient',
        'basin minimum energy  (kJ/mol)')),
    ('03_beta_vs_tilt_spread', 940, 580, lambda t: basin_fig(
        t, 'beta', 'theta_std', 'monoclinic β angle  (deg)',
        'spread of interplane angles  (deg)')),
    ('04_rdf_embedding', 940, 580, lambda t: basin_fig(
        t, 'mds0', 'mds1', 'RDF embedding dim 1', 'dim 2')),
    ('05_coordination_vs_distance', 940, 580, lambda t: basin_fig(
        t, 'n_nbr', 'mean_d', 'coordination number',
        'mean neighbour distance  (Å)', xkw=dict(dtick=1))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dark', action='store_true', help='dark-background set')
    ap.add_argument('--scale', type=float, default=2.0)
    a = ap.parse_args()
    mode = 'dark' if a.dark else 'light'
    t = THEMES[mode]
    os.makedirs(OUT, exist_ok=True)
    for name, w, h, build in FIGS:
        fig = build(t)
        p = os.path.join(OUT, f"{name}{'_dark' if a.dark else ''}.png")
        fig.write_image(p, width=w, height=h, scale=a.scale)
        print(f"{os.path.getsize(p) / 1e3:8.1f} kB  "
              f"{int(w * a.scale)}x{int(h * a.scale)}  {p}")


if __name__ == '__main__':
    main()
