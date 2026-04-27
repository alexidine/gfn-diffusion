import tempfile
import webbrowser
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from scipy.stats import gaussian_kde

import plotly.colors as pc
import torch
from plotly.subplots import make_subplots
from scipy.signal import argrelmin
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from umap import UMAP

from mxtaltools.common.utils import get_point_density_knn
from mxtaltools.reporting.utils import lightweight_one_sided_violin
import plotly.express as px


def make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho, hard_assignment, num_clusters: int,
                      units: str):
    top_inds = torch.argsort(Zb, descending=True)[:num_clusters]

    a, b, = np.unique(hard_assignment.cpu().numpy(), return_counts=True)
    hard_p = b / b.sum()
    basin_ids = [f"{i + 1}" for i in range(num_clusters)]
    p_vals = basin_probs[top_inds].cpu().numpy()
    cluster_members = hard_p[top_inds]
    ref_state = Fb[top_inds].argmin()
    F_vals = (Fb[top_inds] - Fb[top_inds][ref_state]).cpu().numpy()
    Emean_vals = mean_E[top_inds].cpu().numpy()
    Emin_vals = min_ens[top_inds.cpu().numpy()]
    S_vals = (Sb[top_inds] - Sb[top_inds][ref_state]).cpu().numpy()
    rho_vals = mean_rho[top_inds].cpu().numpy()

    table_columns = {
        "Cluster": basin_ids,
        r"$p$": [f"{x:.3f}" for x in p_vals],
        r"$p_{\mathrm{hard}}$": [f"{x:.3f}" for x in cluster_members],
        rf"$\Delta F\ (\mathrm{{{units}}})$": [f"{x:.2f}" for x in F_vals],
        rf"$\langle E \rangle\ (\mathrm{{{units}}})$": [f"{x:.2f}" for x in Emean_vals],
        rf"$E_{{\min}}\ (\mathrm{{{units}}})$": [f"{x:.2f}" for x in Emin_vals],
        rf"$\Delta S_{{\mathrm{{eff}}}}\ (\mathrm{{{units}}})$": [f"{x:.2f}" for x in S_vals],
        r"$c_p$": [f"{x:.3f}" for x in rho_vals],
    }

    def column_colors(
            vals,
            colorscale="Blues",
            vmin=None,
            vmax=None,
            alpha=0.25,  # ← smaller = lighter
    ):
        vals = np.asarray(vals, dtype=float)

        if vmin is None:
            vmin = vals.min()
        if vmax is None:
            vmax = vals.max()

        denom = max(vmax - vmin, 1e-12)
        t = (vals - vmin) / denom

        # get darkest color of the scale
        scale = pc.get_colorscale(colorscale)
        base_color = scale[-1][1]

        # blend toward white
        return [
            pc.find_intermediate_color(
                "rgb(255,255,255)",
                base_color,
                ti * alpha,
                colortype="rgb",
            )
            for ti in t
        ]

    fill_colors = [
        ["rgb(245,245,245)"] * len(basin_ids),  # Basin (no shading)
        column_colors(p_vals, "Blues", alpha=0.35),
        column_colors(cluster_members, "Blues", alpha=0.35),
        column_colors(F_vals, "Reds"),
        column_colors(Emean_vals, "Blues"),
        column_colors(Emin_vals, "Blues"),
        column_colors(S_vals, "Purples", alpha=0.35),
        column_colors(rho_vals, "Blues"),
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_columns.keys()),
                    align="center",
                    font=dict(size=14, color="white"),
                    fill_color="rgb(50,50,50)",
                    height=32,
                ),
                cells=dict(
                    values=list(table_columns.values()),
                    align="center",
                    font=dict(size=13),
                    fill_color=fill_colors,  # "rgb(245,245,245)",
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        font_size=16,
        # title="Thermodynamic Properties of Dominant Basins",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def add_violin(fig, samples, name, color, row, col, ranges, n_kde, bw_factor):
    x_samp, y_samp = lightweight_one_sided_violin(samples,
                                                  n_kde,
                                                  bandwidth_factor=bw_factor,
                                                  data_min=ranges[0],
                                                  data_max=ranges[1])
    fig.add_scatter(
        x=x_samp.cpu().detach().numpy() if torch.is_tensor(x_samp) else x_samp,
        y=y_samp.cpu().detach().numpy() if torch.is_tensor(y_samp) else y_samp,
        mode='lines',
        fill='toself',  # 'tonexty' if i == 0 else 'tonexty',  # Fill to next y (which is 0)
        fillcolor=color,
        line=dict(color=color, width=1.2),
        name=name,
        legendgroup=name,
        showlegend=False,
        row=row, col=col)


def general_figs(fig_dict, sample_batch, sample_energy, units):
    fig = sample_batch.plot_batch_staircase(space='real', return_fig=True, show=False)
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))
    fig_dict['staircase_fig'] = fig

    fig_dict['std_marginals_fig'] = sample_batch.plot_batch_cell_params(space='latent',
                                                                        # ref_dist=data_batch.full_cell_parameters(),
                                                                        # quantiles=[0.1],
                                                                        # override_energy=sample_energy,
                                                                        return_fig=True,
                                                                        show=False)
    # fig_dict['std_marginals_fig'].update_traces(name="Prior Dataset", selector=dict(name="Reference"))
    fig_dict['std_marginals_fig'].update_annotations(font_size=20)

    fig_dict['density_funnel_fig'] = sample_batch.plot_batch_density_funnel(
        override_energy=sample_energy * sample_batch.num_atoms,
        return_fig=True, show=False,
        max_y_quantile=0.99,
        overwrite_yaxis_title=rf"Lattice Energy ({units})")

    fig_dict['energy_marginal'] = energy_marginal_fig(sample_energy)

    return fig_dict


def cluster_comparison_fig(top_cluster_inds,
                           sample_cp, sample_energy, hard_assignment,
                           sample_batch, num_clusters, sample_latents,
                           cluster_color,
                           ):
    nbins = 100
    n_cols = 8
    n_rows = num_clusters

    titles = ['' for _ in range(n_cols * n_rows)]
    titles[0] = 'Energy/Density'
    titles[1] = 'Box Aspect Ratio'
    titles[2] = 'u'
    titles[3] = 'v'
    titles[4] = 'w'
    titles[5] = 'Theta'
    titles[6] = 'Phi'
    titles[7] = 'r'

    fig = make_subplots(rows=num_clusters, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.02,
                        vertical_spacing=0.02)
    for ind in range(min(num_clusters, len(top_cluster_inds))):
        row = ind + 1
        # weights = basin_weights[:, top_cluster_inds[ind]]
        cluster_bools = hard_assignment == top_cluster_inds[ind]
        if not cluster_bools.sum() > 1:
            continue
        fig.add_trace(go.Histogram2dContour(
            x=sample_cp.numpy(),
            y=sample_energy.clip(max=0).numpy(),
            ncontours=12,
            showscale=False,  # colorbar and (i == D - 1 and j == 0),
            contours=dict(coloring='none', showlines=True, start=0.0001, end=0.1, size=0.04),
            line=dict(smoothing=1.0, color='grey', width=2),
            nbinsx=nbins,
            nbinsy=nbins,
            histnorm='probability',
            showlegend=False,
        ), row=row, col=1)
        x = sample_batch.packing_coeff[cluster_bools].numpy()
        y = sample_energy[cluster_bools].numpy()
        xy = np.vstack([x, y])
        try:
            c = get_point_density_knn(xy)
        except:
            c = np.ones(len(xy))
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker_color=c,
            marker_size=6,
            opacity=0.65,
            showlegend=False,
        ), row=row, col=1)

        len_rat = sample_batch.cell_lengths[:, 1] / torch.amax(sample_batch.cell_lengths[:, 0:3:2], dim=1)

        add_violin(fig, len_rat, name='', color='grey', row=row, col=2, n_kde=200, bw_factor=0.05,
                   ranges=[0, len_rat.amax()])
        add_violin(fig, len_rat[cluster_bools], name='', color=cluster_color[ind], row=row, col=2, n_kde=200,
                   bw_factor=0.05,
                   ranges=[0, len_rat.amax()])

        for cind in range(6):
            add_violin(fig, sample_latents[:, 6 + cind], name='', color='grey', row=row,
                       col=3 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])
            add_violin(fig, sample_latents[:, 6 + cind][cluster_bools], name='', color=cluster_color[ind], row=row,
                       col=3 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])

    x_range = [0.55, 0.95]
    y_range = [torch.amin(sample_energy), min(0, torch.quantile(sample_energy, 0.95))]
    for r in range(1, num_clusters + 1):
        fig.update_xaxes(range=x_range, row=r, col=1)
        fig.update_yaxes(range=y_range, row=r, col=1)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    # Dark axis styling
    axis_style = dict(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        tickcolor="black",
        tickwidth=1.5,
        mirror=False,
    )

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            show_y = (c == 1)
            fig.update_yaxes(
                showticklabels=show_y,
                ticks="outside" if show_y else "",
                showgrid=False,
                **axis_style,
                row=r,
                col=c,
            )

            show_x = (r == n_rows)
            fig.update_xaxes(
                showticklabels=show_x,
                ticks="outside" if show_x else "",
                showgrid=False,
                **axis_style,
                row=r,
                col=c,
            )

    fig.update_layout(font_size=16)

    return fig


def dim_reduction_fig(sample_embedding, marker_color, colorscale: Optional[str] = 'viridis'):
    "Umap visualization"
    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0],
                    y=sample_embedding[:, 1],
                    mode='markers', opacity=0.75,
                    marker_size=4,
                    showlegend=False,
                    show_colorbar=True,
                    marker_color=marker_color,
                    marker_colorscale=colorscale)

    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)

    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_layout(font=dict(size=16))
    fig.update_layout(plot_bgcolor='rgb(255,255,255)')

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )
    fig.update_layout(xaxis_title='CV1', yaxis_title='CV2')
    fig.update_layout(font_size=24)

    return fig


def dim_reduction_fig_old(dmat, hard_assignment, clusters_to_analyze, cluster_color, basin_inds,
                          n_neighbors=10, min_dist=0.05):
    "Umap visualization"
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric='precomputed')
    sample_embedding = umap_model.fit_transform(dmat.numpy().astype(np.float64))

    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0],
                    y=sample_embedding[:, 1],
                    mode='markers', opacity=0.75,
                    marker_size=4,
                    showlegend=False, marker_color='grey')
    masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]

    for ind in range(clusters_to_analyze):
        c_ind = mask_sorts[ind]
        m = masks[c_ind]
        fig.add_scatter(x=sample_embedding[m, 0],
                        y=sample_embedding[m, 1],
                        mode='markers', opacity=0.75,
                        marker_size=4,
                        name=f"Cluster {ind + 1}",
                        legendgroup=f"Cluster {ind + 1}",
                        showlegend=False, marker_color=cluster_color[ind],
                        # marker_line_color='grey',
                        # marker_line_width=0.5,
                        )
        fig.add_scatter(x=[sample_embedding[basin_inds[c_ind], 0]],
                        y=[sample_embedding[basin_inds[c_ind], 1]],
                        mode='markers', opacity=1.0,
                        name=f"Cluster {ind + 1}",
                        legendgroup=f"Cluster {ind + 1}",
                        showlegend=True, marker_color=cluster_color[ind],
                        marker_size=16, marker_line_color='black',
                        marker_line_width=6)

    fig.update_yaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)
    fig.update_xaxes(linecolor='black', mirror=True,
                     showgrid=True, zeroline=True)

    fig.update_xaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_yaxes(tickfont=dict(color="rgba(0,0,0,0)", size=1))
    fig.update_layout(font=dict(size=16))
    fig.update_layout(plot_bgcolor='rgb(255,255,255)')

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )
    fig.update_layout(xaxis_title='CV1', yaxis_title='CV2')
    fig.update_layout(font_size=24)

    return fig


def boltzmann_fig(sample_energy, kT, learned_log_Z, logp_est):
    boltzmann_logprobs = -sample_energy / kT - learned_log_Z  # unconditional boltzmann factor
    y = logp_est.cpu().detach()
    x = boltzmann_logprobs.cpu().detach()

    xmax = max(x.max(), y.max())

    fig = go.Figure()

    xy = np.vstack([x.cpu().detach().numpy(), y.cpu().detach().numpy()])
    try:
        c = get_point_density_knn(xy)
    except:
        c = np.ones(len(xy))

    # scatter
    fig.add_scatter(
        x=x,
        y=y,
        marker_color=c,
        mode='markers',
        marker=dict(
            size=6,
            color='rgba(31,119,180,0.6)',
            line=dict(width=0)
        ),
        opacity=0.45,
        name='data',
        showlegend=False,
    )

    # regression line
    quantiles = np.linspace(0, 0.1, 10)
    rs, slopes, intercepts = [], [], []

    for q in quantiles:
        xmin_q = torch.quantile(x, q)
        mask = x > xmin_q
        linreg = linregress(x[mask], y[mask])
        slopes.append(linreg.slope)
        intercepts.append(linreg.intercept)
        rs.append(linreg.rvalue)

    xx = np.linspace(torch.quantile(x, 0.025), xmax, 300)
    med_q = -1  # len(quantiles) // 2
    yy_med = slopes[med_q] * xx + intercepts[med_q]

    # y = x reference
    fig.add_scatter(
        x=xx,
        y=xx,
        mode='lines',
        line=dict(color='gray', width=5, dash='dash'),
        name='y = x',
        showlegend=False,
    )
    fig.add_scatter(
        x=xx,
        y=yy_med,
        mode="lines",
        line=dict(color="black", width=4),
        name="Median fit",
        showlegend=False,
    )

    yy_lo = slopes[-1] * xx + intercepts[-1]
    yy_hi = slopes[0] * xx + intercepts[0]

    fig.add_scatter(
        x=np.concatenate([xx, xx.numpy()[::-1]]),
        y=np.concatenate([yy_lo, yy_hi.numpy()[::-1]]),
        fill="toself",
        fillcolor="rgba(0,0,0,0.15)",
        line=dict(width=0),
        name="Tail sensitivity",
        showlegend=False,
    )

    # annotation
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        showarrow=False,
        align='left',
        text=(
            "Upper 90th Pctl. Regression<br>"
            f"Slope = {slopes[med_q]:.3g}<br>"
            f"Intercept = {intercepts[med_q]:.3g}<br>"
            f"R = {rs[med_q]:.3g}<br>"
        ),
        font=dict(size=20)
    )

    # layout
    fig.update_layout(
        template='simple_white',
        xaxis=dict(showline=True, zeroline=True, zerolinecolor='grey', zerolinewidth=0.5,
                   range=[torch.quantile(x, 0.025), torch.amax(x)]),  # scaleanchor='y',
        yaxis=dict(showline=True, zeroline=True, zerolinecolor='grey', zerolinewidth=0.5,
                   range=[torch.quantile(x, 0.025), torch.amax(x)]),
        legend=dict(orientation='h', y=1.05, x=0.5),  # , xanchor='center'),
        margin=dict(l=60, r=20, t=40, b=50)
    )
    fig.update_layout(xaxis_title='Log Boltzmann Weight',
                      yaxis_title='Sample Log Probability',
                      font_size=24,
                      )

    return fig


def rugged_pes_fig():
    import numpy as np

    import plotly.graph_objects as go

    # -----------------------
    # Parameters (tune these)
    # -----------------------
    np.random.seed(None)

    n_points = 2000
    x = np.linspace(-5, 5, n_points)

    n_large = np.random.randint(3, 6)  # large basins
    n_small = np.random.randint(15, 16)  # metastable minima

    large_amp = 3.0
    small_amp = 1.0

    large_width = 1.0
    small_width = 0.15

    noise_amp = 0.15
    noise_corr = 30  # larger = smoother noise

    # -----------------------
    # Helper
    # -----------------------
    def gaussian(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # -----------------------
    # Build potential
    # -----------------------
    V = np.zeros_like(x)

    # Large basins
    for _ in range(n_large):
        mu = np.random.uniform(-4, 4)
        amp = large_amp * np.random.uniform(0.7, 1.3)
        sig = large_width * np.random.uniform(0.8, 1.3)
        V -= amp * gaussian(x, mu, sig)

    # Metastable minima
    for _ in range(n_small):
        mu = np.random.uniform(-4.5, 4.5)
        amp = small_amp * np.random.uniform(0.5, 1.5)
        sig = small_width * np.random.uniform(0.7, 1.3)
        V -= amp * gaussian(x, mu, sig)

    # Correlated noise
    noise = np.random.randn(n_points)
    kernel = np.ones(noise_corr) / noise_corr
    noise = np.convolve(noise, kernel, mode="same")
    V += noise_amp * noise

    # Gentle confinement
    V += 0.05 * x ** 2

    # Normalize (optional)
    V -= V.min()

    # -----------------------
    # Plot
    # -----------------------
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=V,
        mode="lines",
        line=dict(color="black", width=2),
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
        yaxis=dict(showline=True, linewidth=1.5, linecolor="black"),
        margin=dict(l=60, r=20, t=20, b=60),
    )

    fig.show()


def parity_fig(
        x_raw,
        y_raw,
        x_label='Target (Pf + R)',
        y_label='Model (Pb + Z)',
        quantile_cut: float = 0.99,
):
    # tensors → cpu
    x = x_raw.detach().cpu().float()
    y = y_raw.detach().cpu().float()

    xn, yn = x.numpy(), y.numpy()

    # limits
    xmin = y.min()
    xmax = max(x.max(), y.max())
    pad = 0.02 * (xmax - xmin)
    xmin -= pad
    xmax += pad

    # density coloring
    xy = np.vstack([xn, yn])
    try:
        c = get_point_density_knn(xy)
    except Exception:
        c = np.ones(len(x))

    # full R
    r_full = linregress(xn, yn).rvalue
    from scipy.stats import spearmanr
    rho_full, p_full = spearmanr(xn, yn)

    # trimmed R
    lo, hi = np.quantile(xn, 1 - quantile_cut), np.quantile(xn, 1)
    mask = (xn >= lo) & (xn <= hi)
    r_cut = linregress(xn[mask], yn[mask]).rvalue

    # regression line (full data)
    slope, intercept = linregress(xn, yn).slope, linregress(xn, yn).intercept
    xx = np.linspace(xmin, xmax, 300)
    yy = slope * xx + intercept

    fig = go.Figure()

    fig.add_scatter(
        x=x, y=y, mode='markers',
        marker=dict(size=6, color=c, colorscale='Viridis', opacity=0.55, line=dict(width=0)),
        showlegend=False,
    )
    fig.add_scatter(
        x=xx, y=xx, mode='lines',
        line=dict(color='gray', width=3, dash='dash'), showlegend=False,
    )
    # fig.add_scatter(
    #     x=xx, y=yy, mode='lines',
    #     line=dict(color='black', width=4), showlegend=False,
    # )

    pct = int(quantile_cut * 100)
    fig.add_annotation(
        x=0.02, y=0.98, xref='paper', yref='paper',
        showarrow=False, align='left',
        text=f"R = {r_full:.3g}<br>R<sub>{pct}</sub> = {r_cut:.3g}",
        font=dict(size=24),
    )

    fig.update_layout(
        template='simple_white',
        xaxis=dict(title=x_label, range=[xmin, xmax], showline=True,
                   zeroline=True, zerolinecolor='grey', zerolinewidth=0.5),
        yaxis=dict(title=y_label, range=[xmin, xmax], showline=True,
                   zeroline=True, zerolinecolor='grey', zerolinewidth=0.5,
                   scaleanchor='x', scaleratio=1),
        margin=dict(l=70, r=20, t=40, b=60),
        font_size=24,
    )

    return fig


def sample_summary_table(sample_metrics, sample_energy, sample_inds,
                         metric_keys):
    metric_keys = [key for key in metric_keys if key not in ['is_local_en_minimum', 'is_local_density_maximum', 'var']]

    sample_metrics['energy'] = sample_energy
    table_columns = {"Sample #": [ind for ind in range(len(sample_inds))]}
    table_columns.update({
        f'{key}': [f"{np.nan_to_num(sample_metrics[key][ind]):.3g}" for ind in sample_inds]
        for key in metric_keys})

    def column_colors(
            vals,
            colorscale="RdBu",
            vmin=None,
            vmax=None,
            alpha=0.35,
    ):
        vals = np.asarray(vals, dtype=float)
        if vmin is None: vmin = vals.min()
        if vmax is None: vmax = vals.max()

        denom = max(vmax - vmin, 1e-12)
        # Normalize values between 0 and 1
        t = (vals - vmin) / denom

        # Sample the actual colorscale at point t
        actual_colors = pc.sample_colorscale(colorscale, t)

        # Optional: Apply alpha blending if you want them washed out
        return [
            pc.find_intermediate_color(
                "rgb(255,255,255)",
                c,
                alpha,  # Constant blend to make colors pastel
                colortype="rgb",
            )
            for c in actual_colors
        ]

    fill_colors = [
        column_colors(vals, "RdBu", alpha=0.35) for vals in table_columns.values()
    ]
    fill_colors[0] = ['rgb(255,255,255)' for _ in range(len(fill_colors[0]))]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_columns.keys()),
                    align="center",
                    font=dict(size=14, color="white"),
                    fill_color="rgb(50,50,50)",
                    height=32,
                ),
                cells=dict(
                    values=list(table_columns.values()),
                    align="center",
                    font=dict(size=13),
                    fill_color=fill_colors,  # "rgb(245,245,245)",
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        font_size=16,
        # title="Thermodynamic Properties of Dominant Basins",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def var_table(sample_metrics, sample_energy, sample_inds):
    metric_keys = ['energy', 'count']
    var = sample_metrics['var']
    ndims = var.shape[-1]
    for ind in range(ndims):
        metric_keys.append(f'var{ind}')
        sample_metrics.update({f'var{ind}': np.nan_to_num(var[:, ind]) / np.ptp(np.nan_to_num(var[:, ind]))})

    sample_metrics['energy'] = sample_energy
    table_columns = {"Sample #": [ind for ind in range(len(sample_inds))]}
    table_columns.update({
        f'{key}': [f"{np.nan_to_num(sample_metrics[key][ind]):.3g}" for ind in sample_inds]
        for key in metric_keys})

    def column_colors(
            vals,
            colorscale="RdBu",
            vmin=None,
            vmax=None,
            alpha=0.35,
    ):
        vals = np.asarray(vals, dtype=float)
        if vmin is None: vmin = vals.min()
        if vmax is None: vmax = vals.max()

        denom = max(vmax - vmin, 1e-12)
        # Normalize values between 0 and 1
        t = (vals - vmin) / denom

        # Sample the actual colorscale at point t
        actual_colors = pc.sample_colorscale(colorscale, t)

        # Optional: Apply alpha blending if you want them washed out
        return [
            pc.find_intermediate_color(
                "rgb(255,255,255)",
                c,
                alpha,  # Constant blend to make colors pastel
                colortype="rgb",
            )
            for c in actual_colors
        ]

    fill_colors = [
        column_colors(vals, "RdBu", alpha=0.35) for vals in table_columns.values()
    ]
    fill_colors[0] = ['rgb(255,255,255)' for _ in range(len(fill_colors[0]))]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(table_columns.keys()),
                    align="center",
                    font=dict(size=14, color="white"),
                    fill_color="rgb(50,50,50)",
                    height=32,
                ),
                cells=dict(
                    values=list(table_columns.values()),
                    align="center",
                    font=dict(size=13),
                    fill_color=fill_colors,  # "rgb(245,245,245)",
                    height=28,
                ),
            )
        ]
    )

    fig.update_layout(
        font_size=16,
        # title="Thermodynamic Properties of Dominant Basins",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def sparkbar_table(columns: dict,
                   title: str = "Summary",
                   subtitle: str = None,
                   open_browser: bool = True,
                   show_values: bool = True,
                   save_path: str = None,
                   col_colors: Optional[dict] = None):
    """
    Generate a styled HTML table with in-cell bars.

    Parameters
    ----------
    columns : dict
        {display_name: array-like of values}
        Each key becomes a column header, values become rows.
        Bars are scaled per-column to the min/max of that column.
    title : str
    subtitle : str, optional
    open_browser : bool

    Returns
    -------
    html : str
    """
    col_names = list(columns.keys())
    col_vals = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
    n_rows = len(next(iter(col_vals.values())))
    n_cols = len(col_names)

    # Normalize each column 0-1
    col_normed = {}
    for k, vals in col_vals.items():
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        col_normed[k] = (vals - vmin) / max(vmax - vmin, 1e-12)

    # Format numbers
    def fmt(val):
        if np.isnan(val):
            return "—"
        if abs(val) >= 1000:
            return f"{val:.1f}"
        elif abs(val) >= 1:
            return f"{val:.2f}"
        elif abs(val) >= 0.01:
            return f"{val:.3f}"
        else:
            return f"{val:.2e}"

    # Build rows
    rows_html = ""
    for i in range(n_rows):
        row_class = "even" if i % 2 == 0 else "odd"
        cells = f'<td class="idx-cell">{i}</td>'
        for k in col_names:
            pct = col_normed[k][i] * 100
            val = col_vals[k][i]
            val_text = fmt(val) if show_values else ""
            bg = ""
            if col_colors and k in col_colors:
                bg = f"background-color:{col_colors[k]};"
            cells += f'''<td class="bar-cell {row_class}" style="{bg}">
                    <div class="bar-bg" style="width:{pct:.1f}%;"></div>
                    <span class="bar-val">{val_text}</span>
                </td>'''
        rows_html += f"<tr>{cells}</tr>\n"

    header_cells = '<th class="idx-header">#</th>'
    for k in col_names:
        hbg = ""
        if col_colors and k in col_colors:
            hbg = f"background-color:{col_colors[k]};"
        header_cells += f'<th class="metric-header" style="{hbg}">{k}</th>'

    if subtitle is None:
        subtitle = f"{n_rows} rows &middot; {n_cols} columns"

    html = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}

      body {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        background: #ffffff;
        color: #222222;
        padding: 32px;
        min-height: 100vh;
      }}

      .container {{
        max-width: 95vw;
        margin: 0 auto;
      }}

      h1 {{
        font-size: 16px;
        font-weight: 600;
        color: #111111;
        margin-bottom: 2px;
      }}

      .subtitle {{
        font-size: 11px;
        color: #888888;
        margin-bottom: 16px;
        font-family: 'Courier New', monospace;
      }}

      table {{
        border-collapse: collapse;
        width: auto;
        font-size: 12px;
      }}

      th {{
        background: #f5f5f5;
        color: #333333;
        font-weight: 600;
        text-align: center;
        padding: 8px 8px;
        border-bottom: 2px solid #333333;
        border-top: 2px solid #333333;
        font-size: 11px;
        min-width: 80px;
        font-family: 'Courier New', monospace;
      }}

      .idx-header {{
        min-width: 36px;
        color: #999999;
      }}

      tr:hover {{
        background: #f9f9f9 !important;
      }}

      .idx-cell {{
        text-align: center;
        color: #999999;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        padding: 0 8px;
        border-right: 1px solid #e0e0e0;
      }}

      .bar-cell {{
        position: relative;
        padding: 0;
        height: 26px;
        border-bottom: 1px solid #eeeeee;
        border-right: 1px solid #eeeeee;
        overflow: hidden;
      }}

      .bar-cell.even {{
        background: #ffffff;
      }}

      .bar-cell.odd {{
        background: #fafafa;
      }}

      .bar-bg {{
        position: absolute;
        left: 0;
        top: 2px;
        bottom: 2px;
        border-radius: 1px;
        background: rgba(0, 0, 0, 0.18);
      }}

      .bar-val {{
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        height: 100%;
        padding: 0 8px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        color: #222222;
        white-space: nowrap;
      }}

      .table-wrap {{
        overflow-x: auto;
        border: none;
      }}

      .table-wrap table {{
        border: none;
        border-bottom: 2px solid #333333;
      }}
    </style>
    </head>
    <body>
      <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
        <div class="table-wrap">
          <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </body>
    </html>"""

    if open_browser:
        with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html)
            webbrowser.open('file://' + f.name)

    if save_path:
        from playwright.sync_api import sync_playwright
        with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html)
            html_path = f.name
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_path}")
            page.wait_for_load_state("networkidle")
            height = page.evaluate("document.body.scrollHeight")
            page.set_viewport_size({"width": 1200, "height": height + 64})
            page.screenshot(path=save_path, full_page=True)
            browser.close()

    return html


def energy_marginal_fig(sample_energy):
    # === Input energies ===
    energies_np = sample_energy[sample_energy < 0].cpu().detach().numpy()

    # === Histogram ===
    hist_y, hist_x = np.histogram(energies_np, bins=50, density=True)
    bin_centers = 0.5 * (hist_x[1:] + hist_x[:-1])
    nonzero = hist_y > 0

    # === KDE ===
    kde = gaussian_kde(energies_np, bw_method=0.3)
    x_kde = np.linspace(energies_np.min(), energies_np.max(), 500)
    y_kde = kde(x_kde)

    # === Trim fit to low-energy region ===
    quantile_cutoff = 0.99
    energy_cutoff = np.quantile(energies_np, quantile_cutoff)
    low_energy_mask = bin_centers <= energy_cutoff
    fit_mask = nonzero & low_energy_mask

    x_fit = bin_centers[fit_mask]
    log_y = np.log(hist_y[fit_mask])

    # === Linear fit to log P(E) ≈ -βE + const
    try:
        slope, intercept, _, _, _ = linregress(x_fit, log_y)
        beta_est = -slope
    except:
        slope, intercept = 1, 1

    # === Boltzmann fit in linear space
    boltzmann_y = np.exp(-beta_est * x_kde)
    boltzmann_y /= (np.trapz(boltzmann_y, x_kde) + 1e-6)
    log_fit = slope * bin_centers + intercept

    # === Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Probability Density', 'Log-Probability vs Energy'))

    # --- Left plot: Linear space
    fig.add_trace(go.Bar(
        x=bin_centers, y=hist_y,
        name='Histogram', opacity=0.5, showlegend=False,
        marker_color='blue'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_kde, y=y_kde,
        mode='lines', name='KDE', line=dict(width=2),
        showlegend=False,
        marker_color='red'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_kde, y=boltzmann_y,
        mode='lines', name=f'Boltzmann Fit (β ≈ {beta_est:.2f})',
        line=dict(dash='dot'),
        showlegend=False,
        marker_color='black'
    ), row=1, col=1)
    #
    # fig.add_trace(go.Scatter(
    #     x=[energy_cutoff, energy_cutoff],
    #     y=[0, max(y_kde.max(), hist_y.max())],
    #     mode='lines', name='Fit Cutoff',
    #     line=dict(color='gray', dash='dash')
    # ), row=1, col=1)

    # --- Right plot: Log-space
    fig.add_trace(go.Scatter(
        x=bin_centers[nonzero], y=np.log(hist_y[nonzero]),
        mode='markers+lines', name='log Histogram',
        showlegend=False, marker_color='blue',
        marker=dict(size=5), line=dict(width=2)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=bin_centers, y=log_fit,
        mode='lines', name=f'Linear Fit (β ≈ {beta_est:.2f})',
        line=dict(dash='dot', width=2),
        marker_color='black',
        showlegend=False
    ), row=1, col=2)

    # fig.add_trace(go.Scatter(
    #     x=[energy_cutoff, energy_cutoff],
    #     y=[min(log_y.min(), log_fit.min()), log_y.max()],
    #     mode='lines', name='Fit Cutoff',
    #     line=dict(color='gray', dash='dash')
    # ), row=1, col=2)

    # === Layout
    fig.update_layout(
        # height=500,
        # width=1000,
        template='plotly_white'
    )
    fig.update_layout(font_size=20)

    fig.update_xaxes(title_text='Energy', row=1, col=1)
    fig.update_yaxes(title_text='P(E)', row=1, col=1)

    fig.update_xaxes(title_text='Energy', row=1, col=2)
    fig.update_yaxes(title_text='log P(E)', row=1, col=2)

    return fig


def dual_energy_marginal_fig(sample_energy1, sample_energy2,
                             label1='Energy 1', label2='Energy 2'):
    colors = {'1': 'steelblue', '2': 'firebrick'}

    def process(sample_energy, color, label, fig, row, col):
        energies_np = sample_energy[sample_energy < 0].cpu().detach().numpy()

        hist_y, hist_x = np.histogram(energies_np, bins=50, density=True)
        bin_centers = 0.5 * (hist_x[1:] + hist_x[:-1])
        nonzero = hist_y > 0

        kde = gaussian_kde(energies_np, bw_method=0.3)
        x_kde = np.linspace(energies_np.min(), energies_np.max(), 500)
        y_kde = kde(x_kde)

        quantile_cutoff = 0.99
        energy_cutoff = np.quantile(energies_np, quantile_cutoff)
        low_energy_mask = bin_centers <= energy_cutoff
        fit_mask = nonzero & low_energy_mask
        x_fit = bin_centers[fit_mask]
        log_y = np.log(hist_y[fit_mask])

        try:
            slope, intercept, _, _, _ = linregress(x_fit, log_y)
            beta_est = -slope
        except:
            slope, intercept, beta_est = 1, 1, 1

        boltzmann_y = np.exp(-beta_est * x_kde)
        boltzmann_y /= (np.trapz(boltzmann_y, x_kde) + 1e-6)
        log_fit = slope * bin_centers + intercept

        # left: linear
        fig.add_trace(go.Bar(
            x=bin_centers, y=hist_y,
            name=label, opacity=0.4, showlegend=False,
            marker_color=color,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde, mode='lines',
            name=f'{label} KDE', line=dict(width=2, color=color),
            showlegend=False,
        ), row=row, col=col)
        # fig.add_trace(go.Scatter(
        #     x=x_kde, y=boltzmann_y, mode='lines',
        #     name=f'{label} Boltzmann (β≈{beta_est:.2f})',
        #     line=dict(dash='dot', width=1.5, color=color),
        #     showlegend=False,
        # ), row=row, col=1)
        #
        # # right: log
        # fig.add_trace(go.Scatter(
        #     x=bin_centers[nonzero], y=np.log(hist_y[nonzero]),
        #     mode='markers+lines',
        #     name=label, line=dict(width=2, color=color),
        #     marker=dict(size=5, color=color),
        #     showlegend=False,
        # ), row=row, col=2)
        # fig.add_trace(go.Scatter(
        #     x=bin_centers, y=log_fit, mode='lines',
        #     name=f'{label} fit (β≈{beta_est:.2f})',
        #     line=dict(dash='dot', width=1.5, color=color),
        #     showlegend=False,
        # ), row=row, col=2)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'LJ', f'UMA')  # ,
        # f'{label2} – P(E)', f'{label2} – log P(E)'),
    )
    process(sample_energy1, colors['1'], label1, fig, row=1, col=1)
    process(sample_energy2, colors['2'], label2, fig, row=1, col=2)

    fig.update_layout(template='plotly_white', font_size=20)
    for row in [1]:
        for col in [1, 2]:
            en = 'UMA' if col == 2 else 'LJ'
            fig.update_xaxes(title_text=f'Lattice Energy (kJ/mol)', row=row, col=col)
            fig.update_yaxes(title_text='P(E)', row=row, col=col)
            # fig.update_xaxes(title_text=f'{en} Energy (kJ/mol)', row=row, col=2)
            # fig.update_yaxes(title_text='log P(E)', row=row, col=2)
    # fig.update_xaxes(range = [min(sample_energy1.amin(), sample_energy2.amin()), max(sample_energy1.amax(), sample_energy2.amax())])

    return fig


def bivariate_energy_color(energy, free_energy, clip_quantile=0.01):
    """
    Blue = low energy AND low free energy (target basin)
    Red channel = normalized energy
    Green channel = normalized free energy
    Blue channel = 1 - max(r, g)  →  high when both are low
    """

    def robust_norm(x, clip_quantile):
        lo = np.quantile(x, clip_quantile)
        hi = np.quantile(x, 1 - clip_quantile)
        return np.clip((x - lo) / (hi - lo), 0, 1)

    r = robust_norm(energy, clip_quantile)
    g = robust_norm(free_energy, clip_quantile)
    b = 1 - np.maximum(r, g)

    rgb = np.stack([r, g, b], axis=1)
    return [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for c in rgb]


import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io, base64


def make_bivariate_colorbar_image(size=128):
    """Returns a base64 PNG of the 2D colorbar."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for py in range(size):
        for px in range(size):
            r = px / (size - 1)  # energy: left=low, right=high
            g = 1 - py / (size - 1)  # free energy: bottom=low, top=high
            b = max(0.0, 1 - max(r, g))
            arr[py, px] = [int(r * 255), int(g * 255), int(b * 255)]
    img = Image.fromarray(arr, 'RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def add_bivariate_colorbar(fig, x0=0.78, y0=0.02, size=0.18):
    """
    Inset a 2D colorbar square into an existing figure.
    x0, y0: bottom-left corner in paper coords (0-1)
    size: width and height in paper coords
    """
    b64 = make_bivariate_colorbar_image()

    fig.add_layout_image(
        source=f"data:image/png;base64,{b64}",
        xref="paper", yref="paper",
        x=x0, y=y0 + size,  # top-left anchor
        sizex=size, sizey=size,
        xanchor="left", yanchor="top",
        layer="above",
        sizing="stretch",
    )

    # border rect
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=x0, y0=y0, x1=x0 + size, y1=y0 + size,
        line=dict(color="black", width=0.8),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

    # axis labels
    mid = x0 + size / 2
    fig.add_annotation(
        xref="paper", yref="paper",
        x=mid, y=y0 - 0.03,
        text="E(x) →", showarrow=False,
        font=dict(size=9, family="Helvetica"), xanchor="center",
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=x0 - 0.02, y=y0 + size * 0.75,
        text="G(x) →", showarrow=False,
        font=dict(size=9, family="Helvetica"), xanchor="center",
        textangle=-90,
    )

    # # corner tick labels
    # for (x, y, label, xa, ya) in [
    #     (x0,        y0,        "●", "left",   "top"),     # low E, low ΔG  → blue
    #     (x0+size,   y0,        "●", "right",  "top"),     # high E, low ΔG → red
    #     (x0,        y0+size,   "●", "left",   "bottom"),  # low E, high ΔG → green
    #     (x0+size,   y0+size,   "●", "right",  "bottom"),  # high E, high ΔG→ black
    # ]:
    #     fig.add_annotation(
    #         xref="paper", yref="paper",
    #         x=x, y=y, text=label, showarrow=False,
    #         font=dict(size=7, color="rgba(0,0,0,0.4)"),
    #         xanchor=xa, yanchor=ya,
    #     )

    return fig


def rdf_embedding_fig(sample_embedding, uma_en, uma_free_energy, sorted_minima_inds, related_maxima, polymorph_inds,
                      basin_colors):
    # en_colors = bivariate_energy_color(uma_en.clip(max=0),
    #                                    uma_free_energy,
    #                                    clip_quantile=0.1)
    en_colors = uma_free_energy
    fig = make_subplots(rows=1, cols=2)  # , subplot_titles=['Basin Assignment','Basin Energy'])
    fig.add_scattergl(x=sample_embedding[:, 0],
                      y=sample_embedding[:, 1],
                      marker_colorscale='icefire',
                      marker_color=np.clip(en_colors, a_min=-np.inf, a_max=np.quantile(en_colors, 0.95)) - np.amin(
                          en_colors),
                      mode='markers',
                      marker_colorbar=dict(
                          title=dict(text='Free Energy (kJ/mol)', side='right'),
                          thickness=15,
                          len=0.8,
                          x=0.45,  # positions it between the two subplots
                      ),
                      opacity=0.5, marker_showscale=True,
                      showlegend=False, row=1, col=1)
    fig.add_scattergl(x=sample_embedding[:, 0],
                      y=sample_embedding[:, 1],
                      marker_color=basin_colors, mode='markers',
                      # marker_colorscale=basin_colors,
                      # marker_colorbar=dict(tickvals=list(range(1,len(uniques))), title='Basin'),
                      opacity=0.75,
                      showlegend=False, row=1, col=2)

    fig.update_layout(
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_zeroline=False, yaxis_zeroline=False,
        xaxis2_showgrid=False, yaxis2_showgrid=False,
        xaxis2_zeroline=False, yaxis2_zeroline=False,
        xaxis_title='CV1', yaxis_title='CV2',
        xaxis2_title='CV1',  # yaxis2_title='CV2',
        xaxis_showticklabels=False, yaxis_showticklabels=False,
        xaxis2_showticklabels=False, yaxis2_showticklabels=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    # fig.add_scattergl(x=sample_embedding[sorted_minima_inds, 0], y=sample_embedding[sorted_minima_inds, 1],
    #                   mode='markers', marker_color='white', marker_line_color='black', marker_line_width=4,
    #                   marker_size=22,
    #                   showlegend=False, row=1, col=1)
    # fig.add_scattergl(x=sample_embedding[related_maxima, 0], y=sample_embedding[related_maxima, 1],
    #                   mode='markers', marker_color='rgb(150, 150, 150)', marker_line_color='black', marker_line_width=4,
    #                   marker_size=22,
    #                   showlegend=False, row=1, col=1)
    # fig.add_scattergl(x=sample_embedding[related_maxima, 0], y=sample_embedding[related_maxima, 1],
    #                   mode='markers', marker_color='rgb(150, 150, 150)', marker_line_color='black', marker_line_width=4,
    #                   marker_size=22,
    #                   showlegend=False, row=1, col=2)

    # fig.add_scattergl(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
    #                   mode='markers', marker_color='red', marker_line_color='black', marker_line_width=4,
    #                   marker_size=22,
    #                   showlegend=False, row=1, col=1)

    fig.add_scattergl(x=sample_embedding[sorted_minima_inds, 0], y=sample_embedding[sorted_minima_inds, 1],
                      mode='markers', marker_color='white', marker_line_color='black', marker_line_width=4,
                      marker_size=28,
                      showlegend=False, row=1, col=2)
    fig.add_scattergl(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
                      mode='markers', marker_color='red', marker_line_color='black', marker_line_width=4,
                      marker_size=28, opacity=0.5,
                      showlegend=False, row=1, col=2)

    for rank, idx in enumerate(sorted_minima_inds):
        label = str(rank + 1)
        for col in [2]:
            fig.add_annotation(
                x=sample_embedding[idx, 0],
                y=sample_embedding[idx, 1],
                text=label,
                showarrow=False,
                font=dict(size=14, color='black', family='Arial Black'),
                row=1, col=col,
            )

    # fig = add_bivariate_colorbar(fig, 0.3, 0.70, size=0.2)
    fig.update_annotations(font_size=20)
    fig.update_layout(font_size=20)

    sxmin = np.quantile(sample_embedding[:, 0], 0.01) - 2
    symin = np.quantile(sample_embedding[:, 1], 0.01) - 2
    sxmax = np.quantile(sample_embedding[:, 0], 0.99) + 2
    symax = np.quantile(sample_embedding[:, 1], 0.99) + 2

    fig.update_xaxes(range=[sxmin, sxmax])
    fig.update_yaxes(range=[symin, symax])
    return fig


def combo_fig(num_polymorphs,
              n_basins,
              packing_coeffs,
              stats,
              uma_thermos,
              sample_inds,
              sample_embedding,
              p_maxima,
              polymorph_inds,
              new_min_inds,
              sample_energy,
              polymorph_colorscale,
              sample_colors,
              indexed_cluster_labels,
              basin_colorscale,
              basin_min_batch,
              polymorph_basin_index,
              ):
    marker_font_size = 16

    # Build subplot_titles aligned to the specs grid

    basin_positions, fig = make_specs_fig(n_basins)

    "embedding fig"
    point_size = ((uma_thermos['density'] / np.amax(uma_thermos['density'])) * 60).clip(min=8)

    embedding(fig, marker_font_size, new_min_inds, num_polymorphs, p_maxima, point_size, polymorph_inds, sample_colors,
              sample_embedding, sample_inds)

    fig_grid(basin_min_batch, basin_positions, fig, indexed_cluster_labels, n_basins, p_maxima, packing_coeffs,
             polymorph_basin_index, sample_energy, stats, uma_thermos)

    table_trace = new_new_table(basin_colorscale, num_polymorphs, polymorph_colorscale, stats, n_basins)
    fig.add_trace(table_trace, row=3, col=1)

    """
    Annotations and layout changes
    """
    #
    # for name, color, line_color, size, symbol in [
    #     ("Probability maximum", "rgb(150,150,150)", "black", 14, "circle"),
    #     ("Energy minimum", "white", "black", 14, "circle"),
    #     ("Experimental polymorph", "black", "black", 20, "x-thin"),
    # ]:
    #     fig.add_trace(go.Scatter(
    #         x=[None], y=[None],
    #         mode='markers',
    #         marker=dict(
    #             size=size,
    #             color=color,
    #             symbol=symbol,
    #             line=dict(color=line_color, width=4),
    #         ),
    #         name=name,
    #         showlegend=True,
    #     ), row=1, col=1)
    # fig.update_layout(
    #     legend=dict(
    #         x=0.01,
    #         y=0.99,
    #         xanchor='left',
    #         yanchor='top',
    #         bgcolor='rgba(255,255,255,0.7)',
    #         bordercolor='black',
    #         borderwidth=1,
    #         font=dict(size=11),
    #     ),
    # )
    # Get the domains of the top-left and bottom-left subplots of your 2x2 grid
    y_top = fig.layout.yaxis2.domain[1]  # top of top-row subplot
    y_bot = fig.layout.yaxis4.domain[0]  # bottom of bottom-row subplot
    x_left = fig.layout.xaxis2.domain[0]  # left edge of left-column subplots

    fig.add_annotation(
        text="E (kJ/mol)",
        xref="paper", yref="paper",
        x=x_left - 0.05,  # small offset left of the grid
        y=(y_top + y_bot) / 2,  # true midpoint of the grid
        showarrow=False,
        textangle=-90,
        font=dict(size=14),
        xanchor="center", yanchor="middle",
    )

    x_left = fig.layout.xaxis5.domain[0]  # left edge of Basin 3 (bottom-left of grid)
    x_right = fig.layout.xaxis6.domain[1]  # right edge of Basin 4 (bottom-right)
    y_bot = fig.layout.yaxis5.domain[0]  # bottom of bottom row of grid

    fig.add_annotation(
        text="Packing Coefficient",  # or whatever your x-axis quantity is
        xref="paper", yref="paper",
        x=(x_left + x_right) / 2,
        y=y_bot - 0.04,
        showarrow=False,
        font=dict(size=14),
        xanchor="center", yanchor="top",
    )
    fig.update_layout(
        xaxis1_title='CV1', yaxis1_title='CV2',
        font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgb(200,200,200)', zeroline=True)
    fig.update_yaxes(showgrid=True, gridcolor='rgb(200,200,200)', zeroline=True)

    all_cp = packing_coeffs  # or concatenate just the basin subsets if you prefer
    all_en = sample_energy

    x_pad = 0.075 * (all_cp.max() - all_cp.min())
    y_pad = 0.075 * (all_en.max() - all_en.min())
    x_range = [all_cp.min() - x_pad, all_cp.max() + x_pad]
    y_range = [all_en.min() - y_pad, all_en.max() + y_pad]

    # after all add_trace calls, before fig.show()
    for (row, col) in basin_positions:
        fig.update_xaxes(range=x_range, row=row, col=col)
        fig.update_yaxes(range=y_range, row=row, col=col)
    # fig.add_trace(go.Scatter(
    #     x=[None], y=[None],
    #     mode='markers',
    #     marker=dict(size=10, color='gray', opacity=0.5),
    #     name="Size ∝ P(x)",
    #     showlegend=True,
    # ), row=1, col=1)
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)

    fig.update_annotations(font_size=24)

    return fig


def make_specs_fig(n_basins):
    if n_basins > 4:
        total_rows = 3
        total_cols = 5
        subplot_titles = []
        specs = [
            [{"type": "scatter", "colspan": 2, "rowspan": 2}, None, {"type": "scatter"}, {"type": "scatter"},
             {"type": "scatter"}],

            [{"type": "scatter", "colspan": 2}, None, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],

            [{"type": "table", "colspan": 5}, None, None, None, None]
        ]
        basin_positions = [
            (1, 3), (1, 4), (1, 5),
            (2, 3), (2, 4), (2, 5),
        ]
        basin_idx = 0
        for r in range(total_rows):
            for c in range(total_cols):
                if specs[r][c] is not None:
                    if (r + 1, c + 1) in basin_positions:
                        basin_idx += 1
                        subplot_titles.append(f"Basin {basin_idx}")
                    else:
                        subplot_titles.append("")  # no title for non-basin panels

        fig = make_subplots(
            rows=total_rows, cols=total_cols,
            row_heights=[1/3, 1/3, 1/3],
            column_widths=[0.3, 0.3, 0.4 / 3, 0.4 / 3, 0.4 / 3],
            specs=specs,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )
    elif n_basins <= 4:
        total_rows = 3
        total_cols = 4
        subplot_titles = []
        specs = [
            [{"type": "scatter", "colspan": 2, "rowspan": 2}, None, {"type": "scatter"}, {"type": "scatter"}],

            [{"type": "scatter", "colspan": 2}, None, {"type": "scatter"}, {"type": "scatter"}],

            [{"type": "table", "colspan": 4}, None, None, None]
        ]
        basin_positions = [
            (1, 3), (1, 4),
            (2, 3), (2, 4),
        ]
        basin_idx = 0
        for r in range(total_rows):
            for c in range(total_cols):
                if specs[r][c] is not None:
                    if (r + 1, c + 1) in basin_positions:
                        basin_idx += 1
                        subplot_titles.append(f"Basin {basin_idx}")
                    else:
                        subplot_titles.append("")  # no title for non-basin panels

        fig = make_subplots(
            rows=total_rows, cols=total_cols,
            row_heights=[1 / 3, 1 / 3, 1 / 3],
            column_widths=[0.3, 0.3, 0.4 / 2, 0.4 / 2],
            specs=specs,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.1,
        )

    return basin_positions, fig


def embedding(fig, marker_font_size, new_min_inds, num_polymorphs, p_maxima, point_size, polymorph_inds, sample_colors,
              sample_embedding, sample_inds):
    fig.add_scatter(x=sample_embedding[sample_inds, 0],
                    y=sample_embedding[sample_inds, 1],
                    marker_color=sample_colors,
                    mode='markers',
                    marker_size=point_size,
                    marker_sizemode='area',
                    opacity=1.0,
                    showlegend=False, row=1, col=1)
    fig.add_scatter(x=sample_embedding[p_maxima, 0], y=sample_embedding[p_maxima, 1],
                    mode='markers+text', text=np.arange(len(p_maxima)) + 1,
                    marker_color='rgb(150, 150, 150)', marker_line_color='black', marker_line_width=4,
                    marker_size=28, opacity=0.8,
                    textposition='middle center', textfont=dict(size=marker_font_size, color='black'),
                    showlegend=False, row=1, col=1)
    fig.add_scatter(x=sample_embedding[new_min_inds, 0], y=sample_embedding[new_min_inds, 1],
                    mode='markers+text', text=np.arange(len(new_min_inds)) + 1,
                    marker_color='white', marker_line_color='black', marker_line_width=4,
                    marker_size=28, opacity=0.8,
                    textposition='middle center', textfont=dict(size=marker_font_size, color='black'),
                    showlegend=False, row=1, col=1)
    # fig.add_scatter(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
    #                 mode='markers+text', text=['I', 'II'] if num_polymorphs else ['I'],
    #                 marker_color='black', marker_line_color='black', marker_line_width=4,
    #                 textfont_color='white',
    #                 marker_size=28, opacity=0.8,
    #                 textposition='middle center', textfont=dict(size=marker_font_size),
    #                 showlegend=False, row=1, col=1)
    #
    # # Polymorph markers: X at true location, label offset with arrow
    # polymorph_labels = ['I', 'II'] if num_polymorphs else ['I']
    fig.add_scatter(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
                    mode='markers',
                    marker_symbol='x-thin',
                    marker_color='black',
                    marker_line_color='black',
                    marker_line_width=4,
                    marker_size=18,
                    opacity=0.75,
                    showlegend=False, row=1, col=1)
    #
    # # Compute a direction for the label offset that points away from the data centroid,
    # # so labels reliably land in empty space rather than on top of other callouts.
    # centroid = sample_embedding[sample_inds].mean(axis=0)
    # x_range = np.ptp(sample_embedding[sample_inds, 0])
    # y_range = np.ptp(sample_embedding[sample_inds, 1])
    # offset_frac = 0.12  # fraction of the plot span
    #
    # for idx, label in zip(polymorph_inds, polymorph_labels):
    #     px, py = sample_embedding[idx, 0], sample_embedding[idx, 1]
    #     dx, dy = px - centroid[0], py - centroid[1]
    #     norm = np.hypot(dx, dy) + 1e-12
    #     lx = px + (dx / norm) * offset_frac * x_range
    #     ly = py + (dy / norm) * offset_frac * y_range
    #
    #     fig.add_annotation(
    #         x=px, y=py,  # arrow head at the X
    #         ax=lx, ay=ly,  # label (tail) offset outward
    #         xref='x1', yref='y1',
    #         axref='x1', ayref='y1',
    #         text=f'<b>{label}</b>',
    #         showarrow=True,
    #         arrowhead=2,
    #         arrowsize=1,
    #         arrowwidth=2,
    #         arrowcolor='black',
    #         font=dict(size=marker_font_size, color='black'),
    #         bgcolor='rgba(255,255,255,0.85)',
    #         bordercolor='black',
    #         borderwidth=1,
    #         borderpad=3,
    #         row=1, col=1,
    #     )
    #

def fig_grid(basin_min_batch, basin_positions, fig, indexed_cluster_labels, n_basins, p_maxima, packing_coeffs,
             polymorph_basin_index, sample_energy, stats, uma_thermos):
    for i, basin_ind in enumerate(np.arange(n_basins)):  # enumerate(sorted_minima_inds[:4]):
        row, col = basin_positions[i]

        bb = indexed_cluster_labels == basin_ind
        fig.add_trace(go.Scatter(
            x=packing_coeffs[bb],
            y=sample_energy[bb],
            mode='markers',
            marker=dict(
                size=5,
                color=uma_thermos['density'][bb] / np.amax(uma_thermos['density']),
                colorscale='Viridis',
                cmin=0,
                cmax=1,
                opacity=0.6,
                showscale=(i == 0),  # only once
                colorbar_title="P(x)",
            ),
            showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[stats['sample_cp'][i]],
            y=[sample_energy[p_maxima[i]]],
            mode='markers',
            marker_color='rgb(150, 150, 150)', marker_line_color='black', marker_line_width=4,
            marker=dict(
                size=14,
                opacity=1.0,
                showscale=False,  # (i == 0)  # only once
            ),
            showlegend=False
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[basin_min_batch.packing_coeff[i]],
            y=[basin_min_batch.uma[i]],
            mode='markers',
            marker_color='white', marker_line_color='black', marker_line_width=4,
            marker=dict(
                size=14,
                opacity=1.0,
                showscale=False,  # (i == 0)  # only once
            ),
            showlegend=False
        ), row=row, col=col)
        if basin_ind in polymorph_basin_index:
            poly_ind = torch.argwhere(polymorph_basin_index == basin_ind ).flatten()[0]
            fig.add_trace(go.Scatter(
                x=[stats['sample_cp'][n_basins + poly_ind]],
                y=[stats['sample_energy'][n_basins + poly_ind]],
                mode='markers+text', text='I' if poly_ind == 0 else 'II',
                marker_color='black', marker_line_color='black', marker_line_width=4,
                textfont_color='white',
                marker=dict(
                    size=20,
                    opacity=1.0,
                    showscale=False,  # (i == 0)  # only once
                ),
                textposition='middle center', textfont=dict(size=14),
                showlegend=False
            ), row=row, col=col)


def new_new_table(basin_colorscale, num_polymorphs, polymorph_colorscale, stats, n_basins):
    "summary table"
    e_min = stats['sample_energy']
    p = stats['density'].numpy()
    p2 = stats['elj_density']
    # e_var = stats['local_en_var']
    cp = stats['sample_cp']
    row_labels = ["Min Energy (kJ/mol)",
                  "Normed UMA P(x*)",
                  "Normed LJ P(x*)",
                  "c<sub>p</sub>(E<sub>min</sub>)",
                  ]
    raw_values = [e_min,
                  p,
                  p2,
                  cp
                  ]  # , e_var]
    formatters = [
        lambda v: f"{v:.1f}" if not np.isnan(v) else "",
        lambda v: f"{v:.2f}" if not np.isnan(v) else "",
        lambda v: f"{v:.2f}" if not np.isnan(v) else "",
        lambda v: f"{v:.2f}" if not np.isnan(v) else "",

    ]
    # Which rows should have the *minimum* bolded vs the *maximum*?
    # For most quantities lower is "best"; for TΔS higher (less negative) is notable.
    bold_min_rows = {0}  # E
    bold_max_rows = {1}  # P
    header_vals = [""] + [f"Basin {i + 1}" for i in range(n_basins)]
    heads = ['I', 'II']
    for ind in range(num_polymorphs):
        header_vals.append(f"Polymorph {heads[ind]}")
    cell_vals = [row_labels]
    for i in range(n_basins + num_polymorphs):
        col = []
        for r, (vals, fmt) in enumerate(zip(raw_values, formatters)):
            txt = fmt(vals[i])
            is_min = not np.isnan(vals[i]) and vals[i] == np.nanmin(vals)
            is_max = not np.isnan(vals[i]) and vals[i] == np.nanmax(vals)
            if (r in bold_min_rows and is_min) or (r in bold_max_rows and is_max):
                txt = f"<b>{txt}</b>"
            col.append(txt)
        cell_vals.append(col)
    # Subtle alternating row shading
    white = "rgb(255,255,255)"
    light_grey = "rgb(245,245,245)"
    row_bg = [white if r % 2 == 0 else light_grey for r in range(len(row_labels))]
    header_fill = ["rgb(255, 255, 255)"] + [basin_colorscale[i + 1] for i in range(n_basins)]
    for ind in range(num_polymorphs):
        header_fill.append(polymorph_colorscale[ind])
    fill_colors = [row_bg] + [row_bg for _ in range(n_basins)]
    f2 = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            align="center",
            font=dict(size=22, color="black"),
            fill_color=header_fill,
            height=36,
        ),
        cells=dict(
            values=cell_vals,
            align="center",
            font=dict(size=18),
            fill_color=fill_colors,
            height=36,
        ),
    )])
    table_trace = f2.data[0]
    return table_trace


def pes_cartoon():
    # --- PES ---
    x = np.linspace(-6, 6, 2000)
    well_L = 8.0 * (1 - np.exp(-0.3 * (x + 2.5) ** 2))
    well_R = 9.5 * (1 - np.exp(-0.3 * (x - 2.5) ** 2))
    base = np.minimum(well_L, well_R) + 0.12 * x ** 2
    rng = np.random.default_rng(42)

    # Place sub-basins with controlled randomness
    n_sub = 30
    centers = rng.uniform(-5.5, 5.5, n_sub)
    depths = -rng.uniform(0.3, 2.5, n_sub)
    widths = rng.uniform(5, 18, n_sub)

    sub = sum(a * np.exp(-w * (x - c) ** 2) for a, w, c in zip(depths, widths, centers))
    V = base + sub
    V -= V.min()

    # --- Local minima ---
    local_min_idx = argrelmin(V, order=30)[0]
    local_min_x = x[local_min_idx]
    local_min_V = V[local_min_idx]
    order = np.argsort(local_min_V)
    local_min_x, local_min_V = local_min_x[order], local_min_V[order]

    # --- Distributions ---
    kT = 2.5

    # 1: cold, non-thermalized (trapped in all basins, mild energy weighting)
    p_cold = sum(
        np.exp(-0.3 * vm) * np.exp(-0.5 * ((x - xm) / 0.15) ** 2)
        for xm, vm in zip(local_min_x, local_min_V)
    )
    p_cold /= np.trapz(p_cold, x)

    # 2: correct Boltzmann weights, but only near minima
    p_local = sum(
        np.exp(-vm / kT) * np.exp(-0.5 * ((x - xm) / 0.25) ** 2)
        for xm, vm in zip(local_min_x, local_min_V)
    )
    p_local /= np.trapz(p_local, x)

    # 3: full Boltzmann
    logp = -V / kT
    logp -= logp.max()
    p_full = np.exp(logp)
    p_full /= np.trapz(p_full, x)

    # --- Plot ---
    # fig = make_subplots(
    #     rows=2, cols=3,
    #     row_heights=[0.38, 0.62],
    #     vertical_spacing=0.06,
    #     horizontal_spacing=0.07,
    #     subplot_titles=["Phase 1: Prior Distribution",
    #                     "Phase 2: Prior Thermalization",
    #                     "Phase 3: Global Thermalization"],
    # )
    fig = make_subplots(rows=2, cols=1,
                        row_heights=[0.62, 0.38])
    colors = ["#c44e52", "#dd8452", "#4c72b0"]

    for col, (p, c) in enumerate(zip([p_cold, p_local, p_full], colors), 1):
        fig.add_trace(go.Scatter(x=x, y=p + col * 0.25, fill='toself',
                                 fillcolor=f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.25)",
                                 line=dict(color=c, width=2), showlegend=False), row=1, col=1, )
        fig.add_trace(go.Scatter(x=x, y=V, line=dict(color="#555", width=2),
                                 showlegend=False), row=2, col=1)

    for name, c in zip(["Prior Distribution", "Thermalized Prior", "Equilibrium Distribution"], colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=c, width=3),
            name=name,
        ))
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ), )

    for col in range(1, 4):
        for row in [1, 2]:
            fig.update_xaxes(showticklabels=False, row=row, col=col)
            fig.update_yaxes(showticklabels=False, row=row, col=col)

    fig.update_yaxes(title_text="P(x)", row=1, col=1)
    fig.update_yaxes(title_text="V(x)", row=2, col=1)
    fig.update_xaxes(title_text="x ", row=2, col=2)
    fig.update_layout(  # width=1000, height=550,
        template="plotly_white",
        margin=dict(l=60, r=30, t=50, b=50))
    return fig


def plot_dual_density_contour(
        x_a, y_a,
        x_b, y_b,
        bw: float = 0.15,
        label_a: str = "Distribution A",
        label_b: str = "Distribution B",
        ncontours: int = 8,
        color_a: str = "steelblue",
        color_b: str = "firebrick",
        grid_size: int = 100,
        fill_opacity: float = 0.10,
        renderer=None,
        show: bool = True,
        return_fig: bool = False,
        yaxis_title: str = "Energy per Atom / Arb Units",
        xaxis_title: str = "Packing Coefficient",
        max_y_quantile: float = 0.99,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
):
    from plotly.subplots import make_subplots
    import matplotlib.colors as mcolors

    all_y = np.concatenate([y_a, y_b])
    all_x = np.concatenate([x_a, x_b])
    if y_max is None: y_max = np.quantile(all_y, max_y_quantile)
    if y_min is None: y_min = np.amin(all_y) - np.ptp(all_y) * 0.05
    if x_min is None: x_min = max(0.0, np.amin(all_x) * 0.95)
    if x_max is None: x_max = min(1.0, np.amax(all_x) * 1.05)

    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xi, yi)
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    def compute_kde(x, y, bw_method):
        kde = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
        z = kde(grid_points).reshape(grid_size, grid_size)
        z = np.log1p(z / z.max() * 100)
        return z / z.max()

    def hex_to_rgba(color, alpha):
        r, g, b, _ = mcolors.to_rgba(color)
        return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})'

    def make_colorscale(hex_color, fill_opacity):
        rgba_zero = hex_to_rgba(hex_color, 0.0)
        rgba_fill = hex_to_rgba(hex_color, fill_opacity)
        return [[0.0, rgba_zero], [1.0, rgba_fill]]

    def add_traces(fig, z, color, name, row, col):
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=z,
            ncontours=ncontours,
            contours=dict(coloring='heatmap', showlines=False),
            colorscale=make_colorscale(color, fill_opacity),
            showscale=False, showlegend=False, hoverinfo='skip',
        ), row=row, col=col)
        fig.add_trace(go.Contour(
            x=xi, y=yi, z=z,
            ncontours=ncontours,
            contours=dict(coloring='none', showlines=True),
            line=dict(width=1.5, color=color),
            showscale=False, showlegend=False, hoverinfo='skip',
        ), row=row, col=col)

    z_a = compute_kde(x_a, y_a, bw)
    z_b = compute_kde(x_b, y_b, bw)

    fig = make_subplots(rows=1, cols=2, subplot_titles=[label_a, label_b],
                        shared_yaxes=True, horizontal_spacing=0.06)

    add_traces(fig, z_a, color_a, label_a, 1, 1)
    add_traces(fig, z_b, color_b, label_b, 1, 2)

    fig.update_layout(
        font=dict(family="Helvetica", size=12),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=20, t=40, b=50),
    )
    for ax in ['xaxis', 'xaxis2']:
        fig.update_layout(**{ax: dict(
            title=xaxis_title, range=[x_min, x_max],
            showgrid=True, gridcolor='rgba(0,0,0,0.15)', gridwidth=0.8,
            zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True,
        )})
    for ax in ['yaxis', 'yaxis2']:
        fig.update_layout(**{ax: dict(
            range=[y_min, y_max],
            showgrid=True, gridcolor='rgba(0,0,0,0.15)', gridwidth=0.8,
            zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True,
        )})
    fig.update_layout(yaxis_title=yaxis_title)

    if show:
        fig.show(renderer=renderer)
    if return_fig:
        return fig


def stacked_kde_histograms(
        scalars,
        index,
        label="Value",
        group_name="Group",
        colors=None,
        shared_range=True,
        height_per_group=150,
        width=700,
        side="positive",
        pointpos=-0.5,
        jitter=0.3,
        marker_size=3,
        bandwidth=None,
):
    """
    Stacked violin plots (one per unique index value) using go.Violin with points.

    Parameters
    ----------
    scalars : array-like, shape (n,)
    index : array-like, shape (n,)
    label : str
    group_name : str
    colors : list[str] or None
    shared_range : bool
    height_per_group : int
    width : int
    side : str
        'positive', 'negative', or 'both'
    pointpos : float
        Position of points relative to violin (-1 to 1).
    jitter : float
    marker_size : int

    Returns
    -------
    go.Figure
    """
    scalars = np.asarray(scalars, dtype=float)
    index = np.asarray(index)
    unique_groups = np.unique(index)
    k = len(unique_groups)

    if colors is None:
        base_colors = [
            "rgba(102,194,165,0.55)", "rgba(252,141,98,0.55)",
            "rgba(141,160,203,0.55)", "rgba(231,138,195,0.55)",
            "rgba(166,216,84,0.55)", "rgba(255,217,47,0.55)",
            "rgba(229,196,148,0.55)", "rgba(179,179,179,0.55)",
        ]
        colors = [base_colors[i % len(base_colors)] for i in range(k)]

    line_colors = [c.replace("0.55)", "1.0)") for c in colors]

    fig = make_subplots(
        rows=k, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"{group_name} {g}" for g in unique_groups],
    )

    for i, g in enumerate(unique_groups):
        mask = index == g
        data_g = scalars[mask]

        fig.add_trace(
            go.Violin(
                x=data_g,
                name=f"{group_name} {g}",
                side=side,
                bandwidth=bandwidth,
                orientation="h",
                fillcolor=colors[i],
                line=dict(color=line_colors[i], width=1.2),
                points="all",
                pointpos=pointpos,
                jitter=jitter,
                marker=dict(size=marker_size, color=line_colors[i], opacity=0.9),
                meanline_visible=True,
                showlegend=True,
            ),
            row=i + 1, col=1,
        )

        fig.update_yaxes(showticklabels=False, row=i + 1, col=1)

    if shared_range:
        global_min, global_max = scalars.min(), scalars.max()
        pad = 0.05 * (global_max - global_min)
        for i in range(k):
            fig.update_xaxes(range=[global_min - pad, global_max + pad], row=i + 1, col=1)

    fig.update_xaxes(title_text=label, row=k, col=1)
    fig.update_layout(
        height=height_per_group * k + 80,
        width=width,
        template="plotly_white",
        margin=dict(l=60, r=30, t=40, b=50),
    )
    return fig
