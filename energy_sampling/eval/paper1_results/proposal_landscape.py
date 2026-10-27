"""A simplified landscape for a proposal: the paper figure's embedding panel alone, basins only.

The 100k bundle's window & reduced draws on the paper figure's cached UMAP layout, coloured by the paper figure's
basins (landscape_100k.bundle_landscape's own-mean-shift route at the given kernel d_cut: shares >= 0.8 to a mode,
basins under 100 per 10,000 draws dropped, the largest max_basins kept, numbered by the density at their mode; the
printed sizes must match that figure's), a subsample drawn for legibility, basin numbers written on the basins, and
the experimental form as a large X with one callout. No axes, no legend; built at its printed size.

usage: python -m energy_sampling.eval.paper1_results.proposal_landscape <uma bundle dir> <out.png> [d_cut] [label]
    [--basins]   (default: points coloured by local-density percentile, basins as outlines; --basins: by basin)"""
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from energy_sampling.eval.paper1_results.dens_engine import kernel_sums_from_hist, load_bundle
from energy_sampling.eval.paper1_results.landscape_100k import _dist_to_draws, embed_bundles, mean_shift_anchors

WIDTH_IN, DPI = 3.3, 300   # one column; exported at this size, so fonts are what they print at (height follows the map)
PT = DPI / 72                              # pixels per point at export


def bundle_basins(bundle, d_cut, max_n_clusters=4, min_basin_size=100, max_basins=4, device='cuda'):
    """indexed basin label (-1 = none) for every window & reduced draw, as bundle_landscape(basins_from='bundle',
    d_cut=d_cut) assigns them. Returns (rows, indexed)."""
    b = load_bundle(bundle)
    draws = b['draws']
    N = len(draws['energy'])
    rows = torch.nonzero(draws['window_reduced']).flatten()
    M = len(rows)
    dens = torch.from_numpy(np.asarray(kernel_sums_from_hist(bundle, d_cut)[:N]))[rows]
    ms_rows, anchors, _ = mean_shift_anchors(bundle, d_cut, max_n_clusters, device, dens=dens.numpy(),
                                             tag=f'_dcut{d_cut:.4f}')
    assert np.array_equal(ms_rows, rows.numpy())
    a_rows = rows[torch.as_tensor(anchors)].numpy()
    a_cdf = torch.from_numpy(np.stack([np.asarray(b['cdf'][int(r)]) for r in a_rows]))
    d_anchor = _dist_to_draws(b, a_cdf, draws['empty'][a_rows], device, cols=rows.numpy()).double()
    sigma0 = d_cut / 3
    w = torch.exp(-d_anchor ** 2 / (2 * sigma0 ** 2))
    tot = w.sum(0)
    share = torch.where(tot > 0, w / tot.clamp(min=1e-300), torch.zeros_like(w))
    best_share, best = share.max(0)
    lab = torch.where(best_share >= 0.8, best, torch.full_like(best, -1))
    for v in [v for v in torch.unique(lab) if v >= 0 and int((lab == v).sum()) < min_basin_size * M / 10000]:
        lab[lab == v] = -1
    ids = sorted((int(v) for v in torch.unique(lab) if v >= 0), key=lambda v: -int((lab == v).sum()))
    for v in ids[max_basins:]:
        lab[lab == v] = -1
    ids = ids[:max_basins]
    p_max = {}
    for v in ids:
        near = torch.nonzero((lab == v) & (d_anchor[v] <= sigma0)).flatten()
        p_max[v] = int(near[dens[near].argmax()]) if len(near) else int(d_anchor[v].argmin())
    ids.sort(key=lambda v: -float(dens[p_max[v]]))
    indexed = torch.full((M,), -1, dtype=torch.long)
    for new, v in enumerate(ids):
        indexed[lab == v] = new
    return rows.numpy(), indexed.numpy()


def basin_outline(pts, lo, hi, cover=0.9, smooth=0.02, n=400):
    """A smooth outline of a basin on the map: its points' 2-D histogram on an n x n grid over [lo, hi], Gaussian-smoothed
    (sigma = smooth x the map's larger side), contoured at the level whose enclosed region holds `cover` of the smoothed
    mass. Returns the outline's closed paths ([m, 2] arrays), largest first."""
    from contourpy import contour_generator
    from scipy.ndimage import gaussian_filter
    side = float(max(hi - lo))
    ex, ey = np.linspace(lo[0], hi[0], n + 1), np.linspace(lo[1], hi[1], n + 1)
    h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[ex, ey])
    h = gaussian_filter(h, smooth * side / ((hi[0] - lo[0]) / n))
    v = np.sort(h.ravel())[::-1]
    level = v[np.searchsorted(np.cumsum(v) / v.sum(), cover)]
    cx, cy = 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1])
    lines = contour_generator(x=cx, y=cy, z=h.T).lines(level)
    return sorted([ln for ln in lines if len(ln) > 10], key=len, reverse=True)


def proposal_fig(xy, basin, ref_xy, ref_label, density=None, n_show=20000, seed=0):
    """density: each draw's local-density percentile. Given, points are coloured by it (Turbo, densest on top) and each
    basin is drawn as an outline (basin_outline) numbered on its top edge; else points are coloured by basin."""
    rng = np.random.default_rng(seed)
    show = np.sort(rng.choice(len(xy), min(n_show, len(xy)), replace=False))
    colors = px.colors.qualitative.Vivid
    W = int(WIDTH_IN * DPI)
    fig = go.Figure()
    span = float(np.ptp(xy, axis=0).max())
    if density is not None:
        s = show[np.argsort(density[show])]
        fig.add_scatter(x=xy[s, 0], y=xy[s, 1], mode='markers', hoverinfo='skip', showlegend=False,
                        marker=dict(size=2.2 * PT, line_width=0, opacity=0.8, color=density[s], colorscale='Turbo',
                                    cmin=0, cmax=100,
                                    colorbar=dict(thickness=5 * PT, len=0.5, x=1.0, xanchor='left', y=0.36,
                                                  outlinewidth=0, tickvals=[10, 90], ticktext=['low', 'high'], ticks='',
                                                  tickfont=dict(size=10 * PT, family='Arial', color='black'),
                                                  title=dict(text='local<br>density<br> ', side='top',
                                                             font=dict(size=10 * PT, family='Arial', color='black')))))
        lo0, hi0 = xy.min(0) - 0.08 * span, xy.max(0) + 0.08 * span
        for k in range(int(basin.max()) + 1):
            paths = basin_outline(xy[basin == k], lo0, hi0)
            for e in paths:
                fig.add_scatter(x=e[:, 0], y=e[:, 1], mode='lines', hoverinfo='skip', showlegend=False,
                                line=dict(color='rgba(20, 20, 20, 0.85)', width=1.2 * PT, dash='dash'))
            top = paths[0][np.argmax(paths[0][:, 1])]
            fig.add_annotation(x=top[0], y=top[1], text=f"<b>{k + 1}</b>", showarrow=False,
                               font=dict(size=12 * PT, color='black', family='Arial'), bgcolor='white',
                               bordercolor='rgba(20, 20, 20, 0.85)', borderwidth=0.8 * PT, borderpad=0.6 * PT)
    else:
        order = [-1] + list(range(int(basin.max()) + 1))   # unassigned underneath
        for k in order:
            m = show[basin[show] == k]
            fig.add_scatter(x=xy[m, 0], y=xy[m, 1], mode='markers', hoverinfo='skip', showlegend=False,
                            marker=dict(size=2.2 * PT if k >= 0 else 1.6 * PT, line_width=0,
                                        opacity=0.55 if k >= 0 else 0.35,
                                        color=colors[k + 1] if k >= 0 else 'rgb(170, 170, 170)'))
        for k in range(int(basin.max()) + 1):   # the number at the basin's median point
            c = np.median(xy[basin == k], axis=0)
            fig.add_annotation(x=c[0], y=c[1], text=f"<b>{k + 1}</b>", showarrow=False,
                               font=dict(size=14 * PT, color='white', family='Arial'),
                               bgcolor=colors[k + 1], borderpad=0.8 * PT, opacity=0.95)
    lo, hi = xy.min(0), xy.max(0)
    if density is not None:   # the outlines reach past the points
        for k in range(int(basin.max()) + 1):
            for e in basin_outline(xy[basin == k], xy.min(0) - 0.08 * span, xy.max(0) + 0.08 * span):
                lo, hi = np.minimum(lo, e.min(0)), np.maximum(hi, e.max(0))
    for p, lab in zip(np.atleast_2d(ref_xy), ref_label):
        fig.add_scatter(x=[p[0]], y=[p[1]], mode='markers', hoverinfo='skip', showlegend=False,
                        marker=dict(symbol='x-thin', size=11 * PT, line=dict(width=1.6 * PT, color='black')))
        # callout where its label box (centred on the tail; ~0.064 span per character x 0.1 span) covers the fewest
        # points; the canvas grows to hold the box
        half = np.array([0.032 * span * len(lab), 0.05 * span])
        best, score = None, None
        for r in (0.18, 0.25, 0.32):
            for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                tail = p + r * span * np.array([np.cos(ang), np.sin(ang)])
                inside = np.all(np.abs(xy[show] - tail) < half, axis=1).sum()
                s = int(inside) * 1000 + r   # an empty spot, the nearest
                if score is None or s < score:
                    best, score = tail, s
        lo, hi = np.minimum(lo, best - half), np.maximum(hi, best + half)
        fig.add_annotation(x=p[0], y=p[1], ax=best[0], ay=best[1], xref='x', yref='y', axref='x', ayref='y',
                           text=lab, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=0.9 * PT,
                           arrowcolor='black', standoff=5 * PT, font=dict(size=13 * PT, color='black', family='Arial'))
    pad = 0.02 * span
    lo, hi = lo - pad, hi + pad
    # equal axis scales by sizing the canvas to the ranges (a scaleanchor would override them and clip the label)
    H = int(W * (hi[1] - lo[1]) / (hi[0] - lo[0]))
    fig.update_xaxes(visible=False, range=[lo[0], hi[0]])
    fig.update_yaxes(visible=False, range=[lo[1], hi[1]])
    # the colourbar sits right of the map: its width is added to the canvas
    cb = int(28 * PT) if density is not None else 0
    fig.update_layout(width=W + cb, height=H, margin=dict(l=0, r=cb, t=int(6 * PT), b=0), plot_bgcolor='white',
                      paper_bgcolor='white')
    return fig


if __name__ == '__main__':
    bundle, out = sys.argv[1], sys.argv[2]
    d_cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.242   # MIPCAS P^{10/20}, generate_figs.SCALE_RADII
    label = sys.argv[4] if len(sys.argv) > 4 else 'known form'
    by_basin = '--basins' in sys.argv   # colour by basin instead of density
    rows, basin = bundle_basins(bundle, d_cut)
    print(f"basin sizes {[int((basin == k).sum()) for k in range(basin.max() + 1)]}; unassigned "
          f"{100 * float((basin < 0).mean()):.1f}% of {len(rows)} draws")
    emb = embed_bundles([bundle])
    xy = emb['draws'][0][rows]
    density = None
    if not by_basin:   # percentile of each draw's kernel sum at d_cut among the draws (the paper's P^{10/20} shading)
        from scipy.stats import rankdata
        N = len(load_bundle(bundle)['draws']['energy'])
        s = np.asarray(kernel_sums_from_hist(bundle, d_cut)[:N])[rows]
        density = 100 * (rankdata(s) - 0.5) / len(s)
    fig = proposal_fig(xy, basin, emb['refs'], [label] * len(np.atleast_2d(emb['refs'])), density=density)
    fig.write_image(out, width=fig.layout.width, height=fig.layout.height, scale=1)   # kaleido ignores the layout size
    print('wrote', out)
