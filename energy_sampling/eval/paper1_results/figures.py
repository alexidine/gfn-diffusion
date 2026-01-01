import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from scipy.stats import linregress
from umap import UMAP

from mxtaltools.common.utils import get_point_density
from mxtaltools.reporting.utils import lightweight_one_sided_violin
import numpy as np
import plotly.colors as pc


def make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho, hard_assignment, num_clusters: int):
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
        "Basin": basin_ids,
        r"$p$": [f"{x:.3f}" for x in p_vals],
        r"$p_{\mathrm{hard}}$": [f"{x:.3f}" for x in cluster_members],
        r"$\Delta F\ (\mathrm{kJ/mol})$": [f"{x:.2f}" for x in F_vals],
        r"$\langle E \rangle\ (\mathrm{kJ/mol})$": [f"{x:.2f}" for x in Emean_vals],
        r"$E_{\min}\ (\mathrm{kJ/mol})$": [f"{x:.2f}" for x in Emin_vals],
        r"$\Delta S_{\mathrm{eff}}\ (\mathrm{kJ/mol})$": [f"{x:.2f}" for x in S_vals],
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
    x_samp, y_samp = lightweight_one_sided_violin(samples + torch.randn_like(samples) * 1e-3,
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


def general_figs(fig_dict, sample_batch, sample_energy, data_batch):
    fig = sample_batch.plot_batch_staircase(space='real', return_fig=True, show=False)
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    fig.update_xaxes(title_font=dict(size=20))
    fig.update_yaxes(title_font=dict(size=20))
    fig_dict['staircase_fig'] = fig

    fig_dict['std_marginals_fig'] = sample_batch.plot_batch_cell_params(space='real',
                                                                        ref_dist=data_batch.full_cell_parameters(),
                                                                        # quantiles=[0.1],
                                                                        override_energy=sample_energy, return_fig=True,
                                                                        show=False)
    fig_dict['std_marginals_fig'].update_annotations(font_size=20)

    fig_dict['density_funnel_fig'] = sample_batch.plot_batch_density_funnel(
        override_energy=sample_energy * sample_batch.num_atoms,
        return_fig=True, show=False,
        max_y_quantile=0.99,
        overwrite_yaxis_title=r"Lattice Energy (kJ/mol)")
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

    fig = make_subplots(rows=num_clusters, cols=n_cols, subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.02)
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
            c = get_point_density(xy, bins=50)
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

        for cind in range(3):
            add_violin(fig, sample_latents[:, 6 + cind], name='', color='grey', row=row,
                       col=3 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])
            add_violin(fig, sample_latents[:, 6 + cind][cluster_bools], name='', color=cluster_color[ind], row=row,
                       col=3 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])

        for cind in range(3):
            add_violin(fig, sample_latents[:, 8 + cind], name='', color='grey', row=row,
                       col=6 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])
            add_violin(fig, sample_latents[:, 8 + cind][cluster_bools], name='', color=cluster_color[ind], row=row,
                       col=6 + cind, n_kde=200, bw_factor=0.1, ranges=[-1, 1])

    x_range = [0.55, 0.95]
    y_range = [torch.amin(sample_energy), min(0, torch.quantile(sample_energy, 0.95))]
    for r in range(1, num_clusters + 1):
        fig.update_xaxes(range=x_range, row=r, col=1)
        fig.update_yaxes(range=y_range, row=r, col=1)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
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


def dim_reduction_fig(dmat, hard_assignment, clusters_to_analyze, cluster_color, basin_inds,
                      n_neighbors=10, min_dist = 0.01):
    "Umap visualization"
    umap_model = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric='precomputed')
    sample_embedding = umap_model.fit_transform(dmat)

    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0],
                    y=sample_embedding[:, 1],
                    mode='markers', opacity=0.75,
                    marker_size=6,
                    showlegend=False, marker_color='grey')
    masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]

    for ind in range(clusters_to_analyze):
        c_ind = mask_sorts[ind]
        m = masks[c_ind]
        fig.add_scatter(x=sample_embedding[m, 0],
                        y=sample_embedding[m, 1],
                        mode='markers', opacity=0.75,
                        marker_size=8,
                        name=f"Cluster {ind + 1}",
                        legendgroup=f"Cluster {ind + 1}",
                        showlegend=False, marker_color=cluster_color[ind])
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
        c = get_point_density(xy, bins=50)
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
    med_q = -1 # len(quantiles) // 2
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
            f"Slope = {slopes[med_q]:.3f}<br>"
            f"Intercept = {intercepts[med_q]:.3f}<br>"
            f"R = {rs[med_q]:.3f}<br>"
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
