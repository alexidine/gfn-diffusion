import tempfile
import webbrowser
from typing import Optional

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
        quantile_cut: float = None
):
    """
    Pure parity plot with a single global regression.
    """

    # tensors → cpu
    x = x_raw.detach().cpu().float()
    y = y_raw.detach().cpu().float()

    # limits
    xmin = y.min()  # min(x.min(), y.min())
    xmax = max(x.max(), y.max())
    pad = 0.02 * (xmax - xmin)
    xmin -= pad
    xmax += pad

    # density coloring
    xy = np.vstack([x.numpy(), y.numpy()])
    try:
        c = get_point_density_knn(xy)
    except Exception:
        c = np.ones(len(x))

    if quantile_cut is not None:
        lo, hi = np.quantile(x.numpy(), quantile_cut), np.quantile(x.numpy(), 1 - quantile_cut)
        mask = (x.numpy() >= lo) & (x.numpy() <= hi)
        xr, yr = x.numpy()[mask], y.numpy()[mask]
    else:
        xr, yr = x.numpy(), y.numpy()

    linreg = linregress(xr, yr)
    slope, intercept, r = linreg.slope, linreg.intercept, linreg.rvalue
    xx = np.linspace(xmin, xmax, 300)
    yy = slope * xx + intercept

    fig = go.Figure()

    # scatter
    fig.add_scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=6,
            color=c,
            colorscale='Viridis',
            opacity=0.55,
            line=dict(width=0),
        ),
        showlegend=False,
    )

    # y = x reference
    fig.add_scatter(
        x=xx,
        y=xx,
        mode='lines',
        line=dict(color='gray', width=3, dash='dash'),
        showlegend=False,
    )

    # regression line
    fig.add_scatter(
        x=xx,
        y=yy,
        mode='lines',
        line=dict(color='black', width=4),
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
            f"Slope = {slope:.2g}<br>"
            f"Intercept = {intercept:.3g}<br>"
            f"R = {r:.2g}"
        ),
        font=dict(size=20)
    )

    # layout
    fig.update_layout(
        template='simple_white',
        xaxis=dict(
            title=x_label,
            range=[xmin, xmax],
            showline=True,
            zeroline=True,
            zerolinecolor='grey',
            zerolinewidth=0.5,
        ),
        yaxis=dict(
            title=y_label,
            range=[xmin, xmax],
            showline=True,
            zeroline=True,
            zerolinecolor='grey',
            zerolinewidth=0.5,
            scaleanchor='x',
            scaleratio=1,
        ),
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

    def process(sample_energy, color, label, fig, row):
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
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x_kde, y=y_kde, mode='lines',
            name=f'{label} KDE', line=dict(width=2, color=color),
            showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x_kde, y=boltzmann_y, mode='lines',
            name=f'{label} Boltzmann (β≈{beta_est:.2f})',
            line=dict(dash='dot', width=1.5, color=color),
            showlegend=False,
        ), row=row, col=1)

        # right: log
        fig.add_trace(go.Scatter(
            x=bin_centers[nonzero], y=np.log(hist_y[nonzero]),
            mode='markers+lines',
            name=label, line=dict(width=2, color=color),
            marker=dict(size=5, color=color),
            showlegend=False,
        ), row=row, col=2)
        fig.add_trace(go.Scatter(
            x=bin_centers, y=log_fit, mode='lines',
            name=f'{label} fit (β≈{beta_est:.2f})',
            line=dict(dash='dot', width=1.5, color=color),
            showlegend=False,
        ), row=row, col=2)

    fig = make_subplots(
        rows=2, cols=2,
        # subplot_titles=(f'{label1} – P(E)', f'{label1} – log P(E)',
        #                f'{label2} – P(E)', f'{label2} – log P(E)'),
    )
    process(sample_energy1, colors['1'], label1, fig, row=1)
    process(sample_energy2, colors['2'], label2, fig, row=2)

    fig.update_layout(template='plotly_white', font_size=20)
    for row in [1, 2]:
        en = 'UMA' if row == 2 else 'LJ'
        fig.update_xaxes(title_text=f'{en} Energy (kJ/mol)', row=row, col=1)
        fig.update_yaxes(title_text='P(E)', row=row, col=1)
        fig.update_xaxes(title_text=f'{en} Energy (kJ/mol)', row=row, col=2)
        fig.update_yaxes(title_text='log P(E)', row=row, col=2)
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
        x=x0 - 0.02, y=y0 + size,
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
    en_colors = bivariate_energy_color(uma_en.clip(max=0), uma_free_energy, clip_quantile=0.1)
    fig = make_subplots(rows=1, cols=2)  # , subplot_titles=['Basin Assignment','Basin Energy'])
    fig.add_scattergl(x=sample_embedding[:, 0],
                      y=sample_embedding[:, 1],
                      marker_color=en_colors, mode='markers',
                      opacity=0.5, marker_showscale=False,
                      showlegend=False, row=1, col=1)
    fig.add_scattergl(x=sample_embedding[:, 0],
                      y=sample_embedding[:, 1],
                      marker_color=basin_colors, mode='markers',
                      # marker_colorscale=basin_colors,
                      # marker_colorbar=dict(tickvals=list(range(1,len(uniques))), title='Basin'),
                      opacity=0.5,
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
    fig.add_scattergl(x=sample_embedding[related_maxima, 0], y=sample_embedding[related_maxima, 1],
                      mode='markers', marker_color='rgb(150, 150, 150)', marker_line_color='black', marker_line_width=4,
                      marker_size=22,
                      showlegend=False, row=1, col=2)

    # fig.add_scattergl(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
    #                   mode='markers', marker_color='red', marker_line_color='black', marker_line_width=4,
    #                   marker_size=22,
    #                   showlegend=False, row=1, col=1)
    fig.add_scattergl(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
                      mode='markers', marker_color='red', marker_line_color='black', marker_line_width=4,
                      marker_size=22,
                      showlegend=False, row=1, col=2)
    fig.add_scattergl(x=sample_embedding[sorted_minima_inds, 0], y=sample_embedding[sorted_minima_inds, 1],
                      mode='markers', marker_color='white', marker_line_color='black', marker_line_width=4,
                      marker_size=22,
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

    fig = add_bivariate_colorbar(fig, 0.3, 0.85, size=0.2)
    fig.update_annotations(font_size=20)
    fig.update_layout(font_size=20)

    return fig


def polymorph_summary_table(stats, sorted_minima_inds, polymorph_inds, basin_colors):
    n_basins = len(sorted_minima_inds)

    e_min = stats['sample_energy'].numpy()
    delta_e = e_min - e_min[0]
    g = stats['free_energy']
    delta_g = g - g.min()
    e_var = stats['local_en_var']
    cp = stats['sample_cp'].numpy()

    row_labels = ["E_min (kJ/mol)", "ΔE (kJ/mol)", "ΔG (kT)", "σ²_E", "c_p"]
    raw_values = [e_min, delta_e, delta_g, e_var, cp]
    formatters = [
        lambda v: f"{v:.1f}",
        lambda v: f"{v:.2f}",
        lambda v: f"{v:.2f}",
        lambda v: f"{v:.2f}",
        lambda v: f"{v:.2f}",
    ]

    header_vals = [""] + [f"Basin {i + 1}" for i in range(n_basins)]

    cell_vals = [row_labels]
    for i in range(n_basins):
        col = [fmt(vals[i]) for vals, fmt in zip(raw_values, formatters)]
        cell_vals.append(col)

    def row_colors(vals, colorscale="RdBu", alpha=0.35):
        vals = np.asarray(vals, dtype=float)
        t = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
        actual_colors = pc.sample_colorscale(colorscale, t)
        return [
            pc.find_intermediate_color("rgb(255,255,255)", c, alpha, colortype="rgb")
            for c in actual_colors
        ]

    e_min_colors = row_colors(e_min, "RdBu_r", alpha=0.4)
    delta_e_colors = row_colors(delta_e, "RdBu_r", alpha=0.4)
    delta_g_colors = row_colors(delta_g, "RdBu_r", alpha=0.4)
    var_colors = row_colors(e_var, "Oranges", alpha=0.35)
    white = "rgb(255,255,255)"
    cp_colors = [white] * n_basins
    header_fill = ["rgb(40,40,40)"] + [basin_colors[i + 1] for i in range(n_basins)]

    by_row = [e_min_colors, delta_e_colors, delta_g_colors, var_colors, cp_colors]
    label_col_colors = [white] * len(row_labels)
    fill_colors = [label_col_colors] + [
        [by_row[r][i] for r in range(len(row_labels))]
        for i in range(n_basins)
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            align="center",
            font=dict(size=14, color="black"),
            fill_color=header_fill,
            height=44,
        ),
        cells=dict(
            values=cell_vals,
            align="center",
            font=dict(size=13),
            fill_color=fill_colors,
            height=32,
        ),
    )])

    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


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
