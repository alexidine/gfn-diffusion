
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from scipy.stats import linregress
from umap import UMAP

from mxtaltools.common.utils import get_point_density
from mxtaltools.reporting.utils import lightweight_one_sided_violin


def make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho,hard_assignment, num_clusters:int):
    top_inds = torch.argsort(Zb, descending=True)[:num_clusters]

    a, b, = hard_assignment.unique(return_counts=True)
    hard_p = b/b.sum()
    basin_ids   = [f"B{i+1}" for i in range(num_clusters)]
    p_vals      = basin_probs[top_inds].cpu().numpy()
    cluster_members = hard_p[top_inds].cpu().numpy()
    ref_state = Fb[top_inds].argmin()
    F_vals      = (Fb[top_inds] - Fb[top_inds][ref_state]).cpu().numpy()
    Emean_vals  = mean_E[top_inds].cpu().numpy()
    Emin_vals   = min_ens[top_inds.cpu().numpy()]
    S_vals      = (Sb[top_inds] - Sb[top_inds][ref_state]).cpu().numpy()
    rho_vals    = mean_rho[top_inds].cpu().numpy()

    table_columns = {
        "Basin": basin_ids,
        "p":      [f"{x:.3f}" for x in p_vals],
        "p (hard)":[f"{x:.3f}" for x in cluster_members],
        "ΔF":     [f"{x:.2f}" for x in F_vals],
        "<E>":    [f"{x:.2f}" for x in Emean_vals],
        "E_min":  [f"{x:.2f}" for x in Emin_vals],
        "ΔS_eff":  [f"{x:.2f}" for x in S_vals],
        "c_p":    [f"{x:.3f}" for x in rho_vals],
    }
    fig_table = go.Figure(
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
                    fill_color="rgb(245,245,245)",
                    height=28,
                ),
            )
        ]
    )

    fig_table.update_layout(
        title="Thermodynamic Properties of Dominant Basins",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig_table

def add_violin(fig, samples, name, color, row, col, ranges, n_kde, bw_factor):

    x_samp, y_samp = lightweight_one_sided_violin(samples + torch.randn_like(samples) * 1e-3,
                                                  n_kde,
                                                  bandwidth_factor=bw_factor,
                                                  data_min=ranges[0],
                                                  data_max=ranges[1])
    fig.add_scatter(
        x=x_samp,
        y=y_samp,
        mode='lines',
        fill='toself',  # 'tonexty' if i == 0 else 'tonexty',  # Fill to next y (which is 0)
        fillcolor=color,
        line=dict(color=color, width=1.2),
        name=name,
        legendgroup=name,
        showlegend=False,
        row=row, col=col)

def general_figs(fig_dict, sample_batch, sample_energy, data_batch):

    fig_dict['staircase_fig'] = sample_batch.plot_batch_staircase(space='real', return_fig=True, show=False)
    fig_dict['std_marginals_fig'] = sample_batch.plot_batch_cell_params(space='real', ref_dist=data_batch.full_cell_parameters(), quantiles=[0.1],
                                        override_energy=sample_energy, return_fig=True, show=False)
    fig_dict['density_funnel_fig'] = sample_batch.plot_batch_density_funnel(override_energy=sample_energy, return_fig=True, show=False)
    return fig_dict


def cluster_comparison_fig(top_cluster_inds,
                           sample_cp, sample_energy, hard_assignment,
                           sample_batch, num_clusters, sample_latents,
                           cluster_color,
                           ):

    nbins = 100

    fig = make_subplots(rows=num_clusters, cols=8)
    for ind in range(num_clusters):
        row = ind + 1
        #weights = basin_weights[:, top_cluster_inds[ind]]
        cluster_bools = hard_assignment == top_cluster_inds[ind]
        if not cluster_bools.sum() > 1:
            continue
        fig.add_trace(go.Histogram2dContour(
            x=sample_cp,
            y=sample_energy.clip(max=0),
            ncontours=12,
            showscale=False,  # colorbar and (i == D - 1 and j == 0),
            contours=dict(coloring='none', showlines=True, start=0.0001, end=0.1, size=0.04),
            line=dict(smoothing=1.0, color='grey', width=2),
            nbinsx=nbins,
            nbinsy=nbins,
            histnorm='probability',
            showlegend=False,
        ), row=row, col=1)
        x = sample_batch.packing_coeff[cluster_bools]
        y = sample_energy[cluster_bools]
        xy = np.vstack([x.cpu().detach().numpy(), y.cpu().detach().numpy()])
        try:
            c = get_point_density(xy, bins=50)
        except:
            c = np.ones(len(xy))
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker_color=c,
            marker_size=4,
            opacity=0.65,
            showlegend=False,
        ), row=row, col=1)

        len_rat = sample_batch.cell_lengths[:, 1] / torch.amax(sample_batch.cell_lengths[:, 0:3:2], dim=1)

        add_violin(fig, len_rat, name='', color='grey', row=row, col=2, n_kde=200, bw_factor=0.05, ranges=[0, 6])
        add_violin(fig, len_rat[cluster_bools], name='', color=cluster_color[ind], row=row, col=2, n_kde=200, bw_factor=0.05,
                   ranges=[0, 6])

        for cind in range(3):
            add_violin(fig, sample_latents[:, 6 + cind], name='', color='grey', row=row, col=3 + cind, n_kde=200,
                       bw_factor=0.05, ranges=[-1, 1])
            add_violin(fig, sample_latents[:, 6 + cind][cluster_bools], name='', color=cluster_color[cind], row=row, col=3 + cind,
                       n_kde=200, bw_factor=0.05, ranges=[-1, 1])

        for cind in range(3):
            add_violin(fig, sample_latents[:, 9 + cind], name='', color='grey', row=row, col=6 + cind, n_kde=200,
                       bw_factor=0.05, ranges=[-1, 1])
            add_violin(fig, sample_latents[:, 9 + cind][cluster_bools], name='', color=cluster_color[cind], row=row, col=6 + cind,
                       n_kde=200, bw_factor=0.05, ranges=[-1, 1])

    x_range = [0.55, 0.95]
    y_range = [torch.amin(sample_energy), min(0, torch.quantile(sample_energy, 0.95))]
    for r in range(1, num_clusters + 1):
        fig.update_xaxes(range=x_range, row=r, col=1)
        fig.update_yaxes(range=y_range, row=r, col=1)

    return fig

def dim_reduction_fig(sample_batch, hard_assignment, clusters_to_analyze, cluster_color, basin_inds):
    real_params = sample_batch.latent_params()
    whitened_cell_params = (real_params - real_params.mean(0)) / torch.maximum(real_params.std(0),
                                                                               torch.ones_like(real_params.std(0)))
    "Umap visualization"
    umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.01)
    sample_embedding = umap_model.fit_transform(whitened_cell_params)  # [low_en_bools])

    fig = go.Figure()
    #fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], mode='markers', opacity=0.25, showlegend=False)
    masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]

    for ind in range(clusters_to_analyze):
        c_ind = mask_sorts[ind]
        m = masks[c_ind]
        fig.add_scatter(x=sample_embedding[m, 0],
                        y=sample_embedding[m, 1],
                        mode='markers', opacity=0.75,
                        showlegend=False, marker_color=cluster_color[ind])
        fig.add_scatter(x=[sample_embedding[basin_inds[c_ind], 0]],
                        y=[sample_embedding[basin_inds[c_ind], 1]],
                        mode='markers', opacity=1.0,
                        showlegend=False, marker_color=cluster_color[ind],
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

    return fig

def boltzmann_fig(sample_energy, kT, gfn_model, logp_est):

    boltzmann_logprobs = -sample_energy / kT - gfn_model.flow_model().item()  # unconditional boltzmann factor
    y = logp_est.cpu().detach()
    x = boltzmann_logprobs.cpu().detach()
    y = y[(x > x.quantile(0.05))]
    x = x[(x > x.quantile(0.05))]
    linreg = linregress(x, y)

    # domain
    xmin = min(x.min(), y.min())
    xmax = max(x.max(), y.max())
    xx = np.linspace(xmin, xmax, 200)

    # regression line
    yy = linreg.slope * xx + linreg.intercept

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
            size=7,
            color='rgba(31,119,180,0.6)',
            line=dict(width=0)
        ),
        name='data'
    )

    # regression
    fig.add_scatter(
        x=xx,
        y=yy,
        mode='lines',
        line=dict(color='black', width=2),
        name='fit'
    )

    # y = x reference
    fig.add_scatter(
        x=xx,
        y=xx,
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='y = x'
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
            f"slope = {linreg.slope:.3f}<br>"
            f"intercept = {linreg.intercept:.3f}<br>"
            f"R = {linreg.rvalue:.3f}<br>"
            #f"p = {linreg.pvalue:.2e}"
        ),
        font=dict(size=12)
    )

    # layout
    fig.update_layout(
        template='simple_white',
        xaxis=dict(title='x', scaleanchor='y'),
        yaxis=dict(title='y'),
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
        margin=dict(l=60, r=20, t=40, b=50)
    )
    fig.update_layout(xaxis_title='Boltzmann Weight',
                      yaxis_title='Generated Sample Probability')

    return fig