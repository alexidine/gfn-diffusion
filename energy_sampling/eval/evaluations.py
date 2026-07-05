import os
from typing import Optional

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import torch
import wandb
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, KDTree
from scipy.spatial.distance import cdist
from scipy.stats import linregress, gaussian_kde
from scipy.stats import pearsonr
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from umap import UMAP

from energy_sampling.eval.utils import sample_eval_fwd_trajs, get_plotly_fig_size_mb
from energy_sampling.utils import sample_crystal_prior
from mxtaltools.common.utils import get_point_density, log_rescale_positive
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.reporting.figures import log_crystal_samples
from mxtaltools.reporting.utils import lightweight_one_sided_violin


def adjust_fig_filesize(fig_dict):
    for key in fig_dict.keys():
        fig = fig_dict[key]
        try:
            if get_plotly_fig_size_mb(fig) > 1:  # bigger than 1 MB
                fig.write_image(key + 'fig.png', width=720,
                                height=512)  # save the image rather than the fig, for size reasons
                fig_dict[key] = wandb.Image(key + 'fig.png')
        except:
            pass


def add_color_switcher(fig, color_fields, *, colorscales=None, clip=(1, 99), trace_index=2):
    # trace_index = index of the scatter you want to recolor
    # (0 = identity line, 1 = fit line, 2 = scatter)

    cmins = {k: np.percentile(v, clip[0]) for k, v in color_fields.items()}
    cmaxs = {k: np.percentile(v, clip[1]) for k, v in color_fields.items()}

    buttons = []
    for name, z in color_fields.items():
        buttons.append(dict(
            label=name,
            method="restyle",
            args=[
                {
                    "marker.color": [z.tolist()],
                    "marker.cmin": cmins[name],
                    "marker.cmax": cmaxs[name],
                    "marker.colorscale": (colorscales or {}).get(name, "Viridis"),
                    "marker.colorbar.title": name,
                },
                [trace_index],  # IMPORTANT
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=1.05,
            y=1.0,
            showactive=True,
        )],
        margin=dict(r=120),
    )
    return fig


@torch.no_grad()
def conditional_eval_step(energy_function,
                          gfn_model,
                          discretizer,
                          init_state,
                          mol_batch,
                          mols_to_sample: int = 10,
                          sample_sgs=None

                          ):
    gfn_model.eval()
    # pick some molecules
    rands = np.random.choice(mol_batch.num_graphs, mols_to_sample, replace=False)
    mols_to_sample_list = [elem for i, elem in enumerate(mol_batch.to_data_list()) if i in rands]
    # ensure a clean batch
    samples_per_mol = len(init_state) // mols_to_sample
    init_state = init_state[:samples_per_mol * mols_to_sample]
    # instantiate batch to sample
    mol_batch = collate_data_list(mols_to_sample_list * samples_per_mol)
    cond_inds = torch.tensor([_ for _ in range(mols_to_sample)] * (samples_per_mol), dtype=torch.long)

    if sample_sgs is not None:
        sg_inds = torch.tensor(
            np.random.choice(sample_sgs, mols_to_sample),
            dtype=torch.long,
            device=mol_batch.device
        )
        sg_inds = sg_inds.repeat(samples_per_mol)
    else:
        sg_inds = None

    (flow_states, samples, log_r, log_Z, log_Z_lb,
     log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_flow,
     gauss_params, log_T_tensor) = sample_eval_fwd_trajs(
        init_state, gfn_model, discretizer, energy_function, mol_batch,
        sg_inds=sg_inds,
    )

    metrics = {}
    fig_dict = conditional_fwd_figs(
        log_flow,
        log_pbs,
        log_pfs,
        log_r,
        sample_batch.detach().cpu(),
        cond_inds,
        log_T_tensor,
    )

    adjust_fig_filesize(fig_dict)
    metrics.update(fig_dict)

    gfn_model.train()
    return metrics


def _tb_direction_figs(fig_dict, prefix, log_r, log_pfs, log_pbs, log_flow,
                       flow_states, packing_coeff):
    fig_dict[f'{prefix} TB Parity Plot'], _ = flow_parity_plot(
        log_r, log_flow, log_pbs, log_pfs, packing_coeff)
    # _, fig_dict[f'{prefix} Pf Parity R Value'] = pf_parity_plot(
    #     log_pfs, log_pbs, log_r, log_flow)

    fig_dict[f'{prefix} VG Error'] = vargrad_error(log_r, log_pbs, log_pfs)

    tb_residual = torch.abs(log_r - log_flow - log_pfs.sum(-1) + log_pbs.sum(-1))
    fig_dict[f'{prefix} TB Residual vs R'] = xy_scatter_plot(
        log_r, tb_residual, 'Reward', 'TB Residual')

    fig_dict[f'{prefix} Lattice Latents Trajectories'] = visualize_latent_trajs(
        flow_states.detach().cpu().numpy(), 20, log_r.detach().cpu().numpy())


def eval_figs(fwd_stats,
              bwd_stats,
              sample_batch,
              prior_latent_params,
              energy_function,
              metrics,
              temperature_conditioning: bool = False
              ):
    fig_dict = {}  # todo add tb GP fig & binned residuals

    log_r = fwd_stats['log_r']
    try:
        fig_dict['Boltzmann Fit'], metrics['Boltzmann Temp Estimate'] = boltzmann_fig(log_r)
    except:  # some issues in the above I don't care to fix
        pass

    # --- per-direction TB diagnostics ---
    _tb_direction_figs(
        fig_dict, 'Forward',
        log_r=fwd_stats['log_r'],
        log_pfs=fwd_stats['log_pfs'],
        log_pbs=fwd_stats['log_pbs'],
        log_flow=fwd_stats['log_Z_learned'],
        flow_states=fwd_stats['flow_states'],
        packing_coeff=sample_batch.packing_coeff,
    )
    _tb_direction_figs(
        fig_dict, 'Backward',
        log_r=bwd_stats['log_r'],
        log_pfs=bwd_stats['log_pfs'],
        log_pbs=bwd_stats['log_pbs'],
        log_flow=bwd_stats['log_Z_learned'],  # constant log Z stands in for per-state flow
        flow_states=bwd_stats['flow_states'],
        packing_coeff=bwd_stats['packing_coeff'],
    )

    log_traj_params(
        fwd_stats['means_f'], fwd_stats['logvars_f'], fig_dict,
        fwd_stats['flow_states'], fwd_stats['means_b'],
        fwd_stats['logvars_b'], prefix='Fwd')
    log_traj_params(
        bwd_stats['means_f'], bwd_stats['logvars_f'], fig_dict,
        bwd_stats['flow_states'], bwd_stats['means_b'],
        bwd_stats['logvars_b'], prefix='Bwd')

    fig_dict['Lattice Latents Distribution'] = sample_batch.plot_batch_cell_params(
        space='latent', ref_dist=prior_latent_params, quantiles=[0.1],
        show=False, return_fig=True, override_energy=sample_batch[energy_function])
    fig_dict['Sample Scatter'] = sample_batch.plot_batch_density_funnel(
        show=False, return_fig=True, override_energy=energy_function)

    if temperature_conditioning:
        fig_dict['Z vs T'] = Z_vs_T(fwd_stats)

    return fig_dict, metrics


def Z_vs_T(fwd_stats):

    log_T = np.asarray(fwd_stats['log_T_tensor'])
    log_Z = np.asarray(fwd_stats['log_flow'][:, 0])

    # sort so any line/markers read left-to-right
    order = np.argsort(log_T)
    log_T, log_Z = log_T[order], log_Z[order]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=log_T, y=log_Z, mode="markers",
        marker=dict(
            size=5, color=log_Z, colorscale="Tealgrn", showscale=False,
            line=dict(width=0), opacity=0.85,
        ),
        name="learned log Z",
        hovertemplate="log T = %{x:.3f}<br>log Z = %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Learned log Z vs temperature", font=dict(size=18), x=0.02, xanchor="left"),
        xaxis=dict(title="log T", showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                   ticks="outside", ticklen=4, tickcolor="rgba(0,0,0,0.3)"),
        yaxis=dict(title="log Z", showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
                   ticks="outside", ticklen=4, tickcolor="rgba(0,0,0,0.3)"),
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#2a2a2a"),
        width=720, height=560,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
    )
    return fig

def log_traj_params(means_f, vars_f, fig_dict,
                    flow_states,
                    means_b, vars_b, prefix: str):
    fig_dict[f'{prefix} Gauss Params'] = mean_var_fig(vars_f, means_f,
                                                              vars_b, means_b)
    fig_dict[f'{prefix} Traj Mean Step Sizes'] = mean_flow_step_sizes(flow_states)


def mean_var_fig(logvars_f, means_f, logvars_b, means_b):
    fig = make_subplots(rows=2, cols=1)
    fig.add_scatter(y=np.nan_to_num(torch.exp(logvars_f).mean(0).cpu().detach().numpy()), name='Pf Var', row=2, col=1)
    fig.add_scatter(y=np.nan_to_num(means_f.abs().mean(0).cpu().detach().numpy()), name='Pf Mean', row=1, col=1)
    fig.add_scatter(y=np.nan_to_num(torch.exp(logvars_b).mean(0).cpu().detach().numpy()), name='Pb Var', row=2, col=1)
    fig.add_scatter(y=np.nan_to_num(means_b.abs().mean(0).cpu().detach().numpy()), name='Pb Mean', row=1, col=1)
    fig.update_layout(xaxis2_title='Trajectory Step')
    return fig


def log_buffer_kld(cell_klds, metrics, latent_klds):
    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    if len(cell_klds) == len(lattice_features):
        for ind, feat in enumerate(lattice_features):
            metrics[f'{feat} Cell KLD'] = cell_klds[ind]
            metrics[f'{feat} Latent KLD'] = latent_klds[ind]
        metrics['Mean Cell KLD'] = np.mean(cell_klds)
        metrics['Mean Latent KLD'] = np.mean(latent_klds)
        metrics['Max Cell KLD'] = np.max(cell_klds)
        metrics['Max Latent KLD'] = np.max(latent_klds)


def known_mode_coverage(energy_function, fig_dict, std_cell_params):
    # for some toy problems, we save the solution in the energy function
    known_modes = energy_function.crystal_modes.detach().cpu().numpy() if hasattr(energy_function, 'modes') else None
    if known_modes is not None:
        known_modes_std = energy_function.modes.detach().cpu()
        dists = torch.cdist(std_cell_params.cpu().detach().float(), known_modes_std.float())
        nearest_sample = dists.amin(0)
        cutoff = 1
        fig_dict['Mode Coverage'] = np.mean(nearest_sample.numpy() < cutoff)

        # Gaussian kernel: soft coverage
        sigma = 1.0  # can tune this
        kernel_vals = torch.exp(-0.5 * (dists / sigma) ** 2)  # [num_generated, num_modes]

        # Reduce over generated samples (axis=0) to get "best coverage" per mode
        coverage_per_mode = kernel_vals.max(dim=0).values  # [num_modes]

        fig_dict['Soft Mode Coverage'] = coverage_per_mode.mean()


def conditional_fwd_figs(log_flow, log_pbs, log_pfs, log_r,
                         sample_batch, cond_inds,
                         log_T_tensor):
    fig_dict = {}
    fig_dict['Conditional TB Parity Plot'], _ = conditional_flow_parity_plot(log_r, log_flow, log_pbs, log_pfs,
                                                                             cond_inds)
    fig_dict['Conditional VG Error'] = conditional_vargrad_error(log_r, log_pbs,
                                                                 log_pfs, cond_inds)

    # todo re-add molwise conditioning / flagging with cond_inds
    fig_dict['Conditional Lattice Features Distribution'] = sample_batch.plot_batch_cell_params(
        split_by_sg=True, split_by_zp=True, space='real', show=False, return_fig=True
    )
    fig_dict['Conditional Lattice Latents Distribution'] = sample_batch.plot_batch_cell_params(
        split_by_sg=True, split_by_zp=True, space='latent', show=False, return_fig=True
    )
    fig_dict['Conditional Sample Scatter'] = sample_batch.plot_batch_density_funnel(
        split_by_sg=True, show=False, return_fig=True)

    return fig_dict


def boltzmann_fig(log_r):
    energies = -log_r
    # === Input energies ===
    energies_np = energies.detach().cpu().numpy() if isinstance(energies, torch.Tensor) else energies

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
        name='Histogram', opacity=0.5, showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_kde, y=y_kde,
        mode='lines', name='KDE', line=dict(width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_kde, y=boltzmann_y,
        mode='lines', name=f'Boltzmann Fit (β ≈ {beta_est:.2f})',
        line=dict(dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[energy_cutoff, energy_cutoff],
        y=[0, max(y_kde.max(), hist_y.max())],
        mode='lines', name='Fit Cutoff',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

    # --- Right plot: Log-space
    fig.add_trace(go.Scatter(
        x=bin_centers[nonzero], y=np.log(hist_y[nonzero]),
        mode='markers+lines', name='log Histogram',
        marker=dict(size=5), line=dict(width=2)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=bin_centers, y=log_fit,
        mode='lines', name=f'Linear Fit (β ≈ {beta_est:.2f})',
        line=dict(dash='dot', width=2)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[energy_cutoff, energy_cutoff],
        y=[min(log_y.min(), log_fit.min()), log_y.max()],
        mode='lines', name='Fit Cutoff',
        line=dict(color='gray', dash='dash')
    ), row=1, col=2)

    # === Layout
    fig.update_layout(
        # height=500,
        # width=1000,
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Energy', row=1, col=1)
    fig.update_yaxes(title_text='P(E)', row=1, col=1)

    fig.update_xaxes(title_text='Energy', row=1, col=2)
    fig.update_yaxes(title_text='log P(E)', row=1, col=2)

    return fig, 1 / beta_est


def cluster_hist_fig(cluster_ind):
    uniques, counts = np.unique(cluster_ind, return_counts=True)
    fig = go.Figure()
    fig.add_bar(x=uniques, y=counts)
    fig.update_layout(xaxis_title='Cluster Ind', yaxis_title='Cluster Size')
    return fig


def embed_samples(ref_samples, ref_rewards, samples, sample_rewards, max_ref_samples: int, temperature: float = 0.1):
    """
    Cluster samples via agglomerative clustering
    Also, embed them in PC space
    :param ref_samples:
    :param ref_rewards:
    :param samples:
    :param sample_rewards:
    :return:
    """

    if ref_samples is not None:
        if len(ref_samples) > max_ref_samples:
            ref_rewards, ref_samples = subsample_batch(ref_rewards, ref_samples, max_ref_samples, temperature)

        samples_to_fit = np.concatenate([ref_samples, samples])
        energies_to_fit = -np.concatenate([ref_rewards, sample_rewards])
    else:
        samples_to_fit = samples
        energies_to_fit = -sample_rewards

    # embed the mixed reference + sampled distribution
    # todo cluster replace by agglomerative, but keep viz in umap
    # todo write a beautiful clustering module
    umap_model = UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
    sample_embedding = umap_model.fit_transform(samples_to_fit)

    # fit energy-weighted 2D gaussian kde on the UMAP data
    beta = 0.001
    weights = np.exp(-beta * (energies_to_fit - np.min(energies_to_fit)))  # stabilize exponent
    weights /= weights.sum()
    kde = gaussian_kde(sample_embedding.T, weights=weights, bw_method='scott')

    # Evaluate KDE on a 2D grid
    grid_size = 300
    x_min, y_min = sample_embedding.min(axis=0)
    x_max, y_max = sample_embedding.max(axis=0)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    density = kde(grid_coords).reshape(grid_size, grid_size)

    # Smooth for better watershed behavior
    density_smooth = gaussian_filter(density, sigma=1)
    density_inverted = -density_smooth  # so basins are valleys

    # -------------
    # Step 3: Watershed
    # -------------
    min_kde_range = 2  # np.sqrt(grid_size**2 / 500)  # to stuff M evenly spaced clusters
    peaks = peak_local_max(density_smooth, min_distance=int(min_kde_range), exclude_border=False)
    markers = np.zeros_like(density_smooth, dtype=int)
    for i, (x, y) in enumerate(peaks):
        markers[x, y] = i + 1  # watershed marker labels must be >0

    labels = watershed(density_inverted, markers=markers)

    # -------------
    # Step 4: Assign samples to regions
    # -------------
    # Build tree from grid to map each point to a label
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    label_flat = labels.ravel()
    tree = KDTree(grid_points)
    _, idx = tree.query(sample_embedding, k=1)
    cluster_assignments = label_flat[idx]

    unique_clusters = np.unique(cluster_assignments)
    anchor_inds = np.zeros(len(peaks), dtype=np.int32)
    for ind, clu in enumerate(unique_clusters):
        good_inds = np.argwhere(cluster_assignments == clu).flatten()
        ii = np.argmin(energies_to_fit[good_inds]).flatten()
        anchor_inds[ind] = good_inds[ii]

    return (sample_embedding[-len(samples):],
            sample_embedding[anchor_inds],
            samples_to_fit[anchor_inds],
            cluster_assignments[-len(samples):],
            energies_to_fit[anchor_inds],
            -sample_rewards,
            cluster_assignments[:-len(samples)],
            labels,
            (x_min, x_max, y_min, y_max))


def subsample_batch(ref_rewards, ref_samples, sample_size, temperature):  # todo replace this with our overlap method
    dists = cdist(ref_samples, ref_samples) + np.eye(len(ref_samples)) * 100
    closest_neighbor = np.amin(dists, axis=1)
    weight_rew = ref_rewards + closest_neighbor ** 2  # reward distinct samples
    weights = np.exp((weight_rew - weight_rew.max()) / temperature) + 1e-2
    weights /= weights.sum()
    inds_to_keep = np.random.choice(len(ref_samples), sample_size, replace=False, p=weights)
    ref_samples = ref_samples[inds_to_keep]
    ref_rewards = ref_rewards[inds_to_keep]
    return ref_rewards, ref_samples


"""
#possible alternative workflow

import numpy as np
import umap
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.segmentation import watershed
from sklearn.neighbors import KDTree
import plotly.express as px


"""


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions into finite polygons.

    Code adapted from:
    https://gist.github.com/pv/8036995
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points) * 2  # large enough

    # Map ridge points to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if -1 not in region:
            # Finite region
            new_regions.append([vor.vertices[i] for i in region])
            continue

        ridges = all_ridges[p1]
        new_region = []
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                # both vertices are finite
                new_region.append(vor.vertices[v2].tolist())
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(vor.vertices[v2].tolist())
            new_region.append(far_point.tolist())

        # Order region vertices
        vs = np.array(new_region)
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = vs[np.argsort(angles)]

        new_regions.append(new_region)

    return new_regions


def cluster_fig(sample_embedding, anchor_embedding, cluster_ind, anchor_energies,
                sample_energies, color_mode, watershed: Optional = None, watershed_range: Optional = None):
    """
    Figure for the clusters in PC space + Voronoi assignments
    :param sample_embedding:
    :param anchor_embedding:
    :param cluster_ind:
    :return:
    """
    if watershed is None:
        try:
            vor = Voronoi(anchor_embedding)
            polygons = voronoi_finite_polygons_2d(vor)
        except:
            polygons = None
    else:
        polygons = None

    energies = np.array(anchor_energies)
    norm_energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)

    colorscale = pc.get_colorscale("Jet")
    line_colors = [pc.sample_colorscale(colorscale, [v])[0] for v in norm_energies]

    x_all = np.concatenate([sample_embedding[:, 0], anchor_embedding[:, 0]])
    y_all = np.concatenate([sample_embedding[:, 1], anchor_embedding[:, 1]])

    colorscale_name = "rainbow"  # or "viridis", "plasma", etc.
    n_clusters = len(anchor_embedding)
    distinct_colors = pc.sample_colorscale(colorscale_name,
                                           [i / max((n_clusters + 2) - 1, 1) for i in range(n_clusters + 2)])
    cluster_to_color = {i: distinct_colors[i] for i in range(n_clusters + 2)}
    cluster_to_color.update({-1: 'rgb(0,0,0)'})  # black for noise dimension
    if color_mode == 'cluster':
        mapped_colors = [cluster_to_color[c] for c in cluster_ind]
    elif color_mode == 'energy':
        mapped_colors = sample_energies

    fig = go.Figure()
    if polygons is not None:
        for ind, poly in enumerate(polygons):
            cluster_color = cluster_to_color[ind]
            fig.add_trace(go.Scattergl(
                x=np.array(poly)[:, 0],
                y=np.array(poly)[:, 1],
                mode='lines',
                line=dict(color=cluster_color, width=1),
                fill='toself',
                fillcolor=cluster_color,
                opacity=0.1,
                hoverinfo='skip',
                showlegend=False,
                marker_showscale=False,
            ))

    if watershed is not None:
        custom_cscale = []
        for i in range(n_clusters):
            rel_pos = i / n_clusters
            custom_cscale.append([rel_pos, cluster_to_color[i]])

        x_min, x_max, y_min, y_max = watershed_range
        x_grid = np.linspace(x_min, x_max, watershed.shape[1])
        y_grid = np.linspace(y_min, y_max, watershed.shape[0])

        fig.add_trace(go.Heatmap(
            z=watershed / n_clusters,
            x=x_grid, y=y_grid,
            colorscale=custom_cscale,
            opacity=0.2,
            showscale=False
        ))

    fig.add_trace(go.Scattergl(x=anchor_embedding[:, 0],
                               y=anchor_embedding[:, 1],
                               mode='markers',
                               opacity=1,
                               name='Minima',
                               showlegend=False,
                               marker=dict(
                                   size=15,
                                   color=line_colors,
                                   # [cluster_to_color[ind] for ind in range(len(anchor_embedding))],  # Fill color
                                   line=dict(
                                       color='white',  # <-- variable border color!
                                       width=4
                                   )
                               )))

    fig.add_trace(go.Scattergl(x=sample_embedding[:, 0],
                               y=sample_embedding[:, 1],
                               mode='markers',
                               opacity=0.85,
                               name='Policy Samples',
                               showlegend=False,
                               marker=dict(
                                   size=6,
                                   colorscale=colorscale if color_mode == 'energy' else None,
                                   color=mapped_colors,
                                   colorbar=dict(title="Cluster Membership"),
                                   showscale=False,
                               )
                               ))

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=energies.min(),
            cmax=energies.max(),
            colorbar=dict(title='Anchor Energy'),
            color=[energies.min(), energies.max()],  # dummy scalar range
            showscale=True,
            size=0.1  # invisible
        ),
        showlegend=False
    ))

    fig.update_layout(
        xaxis_range=[x_all.min() - 1, x_all.max() + 1],
        yaxis_range=[y_all.min() - 1, y_all.max() + 1]
    )

    return fig



def get_buffer_stats(buffer):
    if len(buffer) > 0:
        samples_to_take = min(10000, len(buffer))
        # take samples according to the sampler weighting, rather than random trash in the buffer
        _, buffer_reward, buffer_batch, condition = buffer.sample(
            override_batch=samples_to_take,
            return_preload=True,
            # standardize_orientations=True
        )
        buffer_cell_params = buffer_batch.full_cell_parameters().cpu().detach().numpy()
        buffer_latent_params = buffer_batch.latent_params().cpu().detach().numpy()
        buffer_std_params_for_embedding = buffer_batch.latent_params().cpu().detach().numpy()
        reward = buffer_reward.cpu().detach().numpy()
        batch = buffer_batch.cpu().detach()
        sg_inds = buffer_batch.sg_ind.cpu().detach().numpy()
    else:
        buffer_cell_params, buffer_latent_params, buffer_std_params_for_embedding, reward, batch, sg_inds = None, None, None, None, None, None
    return buffer_cell_params, buffer_latent_params, buffer_std_params_for_embedding, reward, batch, sg_inds


def mean_flow_step_sizes(flow_states):
    mean_step_size = flow_states.diff(dim=1).abs().mean(0)
    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    fig = go.Figure(go.Heatmap(x=lattice_features, y=np.arange(1, flow_states.shape[1] + 1),
                               z=mean_step_size.cpu().detach()))
    return fig



def buffer_kld(buffer, metrics, sample_batch):
    (buffer_cell_params, buffer_latent_params,
     buffer_std_params, buffer_reward,
     buffer_batch, buffer_sg_inds) = get_buffer_stats(buffer)
    cell_params = sample_batch.zp1_cell_parameters().cpu().detach().numpy()
    latent_params = sample_batch.latent_params().cpu().detach().numpy()
    cell_klds = np.zeros(cell_params.shape[1])
    latent_klds = np.zeros(latent_params.shape[1])
    for ind in range(len(cell_klds)):
        cell_klds[ind] = compute_1d_kld(cell_params[:, ind], buffer_cell_params[:, ind])
        latent_klds[ind] = compute_1d_kld(latent_params[:, ind], buffer_latent_params[:, ind])
    log_buffer_kld(cell_klds, metrics, latent_klds)


def compute_1d_kld(p_data: np.ndarray,
                   q_data: np.ndarray,
                   n_kde_points=200,
                   bw_ratio=0.1,
                   epsilon: float = 1e-4):
    """
    :param p_data: reference distribution
    :param q_data: sample distribution
    :param n_bins:
    :param eps:
    :return:
    """
    data_range = [min(np.amin(p_data), np.amin(q_data)), max(np.amax(p_data), np.amax(q_data))]
    x_samp, y_samp = lightweight_one_sided_violin(
        q_data, n_kde_points,
        bandwidth_factor=bw_ratio,
        data_min=data_range[0],
        data_max=data_range[1],
    )
    x_ref, y_ref = lightweight_one_sided_violin(
        p_data, n_kde_points,
        bandwidth_factor=bw_ratio,
        data_min=data_range[0],
        data_max=data_range[1],
    )

    x_common = x_samp

    # Remove the arbitrary "width" scaling for probability normalization
    y_ref = np.maximum(y_ref, epsilon)
    y_q = np.maximum(y_samp, epsilon)
    P = y_ref / (np.trapz(y_ref, x_common) + epsilon)
    Q = y_q / (np.trapz(y_q, x_common) + epsilon)

    kl = np.trapz(P * np.log((P + epsilon) / (Q + epsilon)), x_common)
    return kl
    """
    #visually examine
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=x_samp, y=y_samp)
    fig.add_scatter(x=x_ref,y=y_ref)
    fig.show()
   
    """


def sample_backward_prior(args, buffer, sample_batch, num_samples):
    if buffer is not None:
        if len(buffer) > 0:
            prior_sample, _, _, _ = buffer.sample(override_batch=num_samples,
                                                  )
        else:
            prior_sample = sample_crystal_prior(sample_batch, args.bwd_loss_coeffs.pmle_std)
    else:
        prior_sample = sample_crystal_prior(sample_batch, args.bwd_loss_coeffs.pmle_std)
    return prior_sample


def to_loggable(v):
    if torch.is_tensor(v):
        v = v.detach().cpu()
        if v.numel() == 1:
            return v.item()
        else:
            return v.numpy()
    return v


def pf_parity_plot(log_pfs, log_pbs, log_r, log_flow):
    x = log_pfs.sum(-1).cpu().detach().numpy()
    y = (log_r + log_pbs.sum(-1) - log_flow).cpu().detach().numpy()
    r_value, _ = pearsonr(x, y)

    fig = go.Figure()
    fig.add_scatter(x=x,
                    y=y,
                    name=f'R = {r_value:.3f}',
                    showlegend=True,
                    marker_color=log_r.cpu().detach().numpy(),
                    marker_colorscale='Jet',
                    mode='markers',
                    )
    fig.update_layout(xaxis_title='Pf(t)', yaxis_title='Pb*R/Z')
    return fig, r_value


def xy_scatter_plot(
        x: torch.Tensor,
        y: torch.Tensor,
        xaxis_title: str = '',
        yaxis_title: str = '',
        c: Optional[torch.Tensor] = None,
        c_bins: int = 25,

):
    try:
        r_value, _ = pearsonr(x, y)
    except:
        r_value = 0

    if c is None:
        xy = np.vstack([x.cpu().detach().numpy(), y.cpu().detach().numpy()])
        try:
            c = get_point_density(xy, bins=c_bins)
        except:
            c = np.ones(len(xy))

    fig = go.Figure()
    fig.add_scatter(x=x.cpu().detach().numpy(),
                    y=y.cpu().detach().numpy(),
                    marker_color=c,
                    marker_colorscale='Jet',
                    name=f'R = {r_value:.3f}',
                    showlegend=True,
                    mode='markers',
                    )
    fig.update_layout(yaxis_title=yaxis_title,
                      xaxis_title=xaxis_title)
    return fig


def diverse_n_colors(n, colorscale='Viridis'):
    """
    Generate `n` visually distinct colors from a Plotly continuous colorscale.

    Args:
        n (int): Number of colors to generate.
        colorscale (str or list): Name of Plotly continuous colorscale, or a custom list.

    Returns:
        list[str]: List of color strings (e.g., 'rgb(68,1,84)')
    """
    if isinstance(colorscale, str):
        try:
            base_scale = pc.get_colorscale(colorscale)
        except ValueError:
            raise ValueError(f"Unknown Plotly colorscale: {colorscale}")
    else:
        base_scale = colorscale

    # Normalize to 0-1 spacing, but skip too-close colors by spreading out non-linearly
    positions = np.linspace(0, 1, n)

    # Use Plotly's built-in interpolation
    interpolated = pc.sample_colorscale(base_scale, positions, colortype='rgb')
    return interpolated


def visualize_latent_trajs(states, n_trajs, log_r):
    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    # 1d Histograms
    n_crystal_features = 12
    n_trajs = min(n_trajs, states.shape[0])
    steps = np.arange(states.shape[1])
    trajs = states

    # Center the normalization around 0
    # vmin = -max(abs(log_r[:n_trajs].min()), abs(log_r[:n_trajs].max()))
    # vmax = -vmin  # symmetric

    # Normalize to [0, 1], with 0 mapped to 0.5 in the colormap
    # norm_log_r = (log_r[:n_trajs] - vmin) / (vmax - vmin)
    # cmap = cm.get_cmap('bwr')
    # color_hex = [to_hex(cmap(val)) for val in norm_log_r]

    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        for j in range(n_trajs):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(go.Scatter(
                x=trajs[j, :, i], y=steps,
                name=f"Traj {j}",
                legendgroup=f"Traj {j}",
                opacity=0.5,
                mode='lines',
                marker_line_width=0.5,
                showlegend=True if i == 0 else False,
                marker_color=log_r,  # color_hex[j],
                marker_colorscale='Jet',
            ),
                row=row, col=col
            )
    custom_ranges = {i: [-1.1, 1.1] for i in range(len(lattice_features))}
    for i in range(len(lattice_features)):
        row = i // 3 + 1
        col = i % 3 + 1
        fig.update_xaxes(range=custom_ranges[i], row=row, col=col)
    return fig


def flow_parity_plot(log_r, log_Z_learned, log_pbs, log_pfs, packing_coeff):
    # Compute x and y
    x = (log_r + log_pbs.sum(-1)).cpu().detach().numpy()
    y = (log_Z_learned + log_pfs.sum(-1)).cpu().detach().numpy()
    log_importance_weight = ((log_r - log_Z_learned) - (log_pfs.sum(-1) - log_pbs.sum(-1))).cpu().detach().numpy()

    # Get symmetric limits
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    margin = 0.05 * (max_val - min_val)
    lim_low = min_val - margin
    lim_high = max_val + margin
    mae = np.mean(np.abs(x - y))
    r_value, _ = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, deg=1)
    # Build figure
    fig = go.Figure()

    # Identity line
    fig.add_trace(go.Scatter(
        x=[lim_low, lim_high], y=[lim_low, lim_high],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[lim_low, lim_high], y=[slope * lim_low + intercept, slope * lim_high + intercept],
        mode='lines',
        line=dict(color='red', dash='dot'),
        name=f'{slope:.3f}x + {intercept:.3f}',
        showlegend=True
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=6, opacity=0.7,
                    # color=log_r.cpu().detach(), colorscale='Jet'
                    ),
        name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
    ))

    # Layout adjustments
    fig.update_layout(
        title='Parity Plot',
        xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
        yaxis=dict(title='log_Z + log_pf', range=[lim_low, lim_high], scaleanchor='x'),  # , scaleratio=1),
        # width=600,
        # height=600,
        template='plotly_white'
    )
    color_fields = {
        "Reward": log_r,
        "Density": packing_coeff.clip(max=2),
        "Importance Weight": log_importance_weight,
    }
    fig = add_color_switcher(fig, color_fields)

    return fig, r_value


def conditional_flow_parity_plot(log_r, log_Z_learned, log_pbs, log_pfs, cond_inds):
    # Compute x and y
    x = (log_r + log_pbs.sum(-1)).cpu().detach().numpy()
    y = (log_Z_learned + log_pfs.sum(-1)).cpu().detach().numpy()

    # Get symmetric limits
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    margin = 0.05 * (max_val - min_val)
    lim_low = min_val - margin
    lim_high = max_val + margin
    mae = np.mean(np.abs(x - y))
    r_value, _ = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, deg=1)
    # Build figure
    fig = go.Figure()

    # Identity line
    fig.add_trace(go.Scatter(
        x=[lim_low, lim_high], y=[lim_low, lim_high],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[lim_low, lim_high], y=[slope * lim_low + intercept, slope * lim_high + intercept],
        mode='lines',
        line=dict(color='red', dash='dot'),
        name=f'{slope:.3f}x + {intercept:.3f}',
        showlegend=True
    ))
    n_elems = len(torch.unique(cond_inds))
    for ind in range(n_elems):
        bools = cond_inds == ind
        # Scatter points
        fig.add_trace(go.Scatter(
            x=x[bools], y=y[bools],
            mode='markers',
            marker=dict(size=6, opacity=0.7, color=ind, colorscale='Rainbow'),
            # name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
        ))

    # Layout adjustments
    fig.update_layout(
        title='Parity Plot',
        xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
        yaxis=dict(title='log_Z + log_pf', range=[lim_low, lim_high], scaleanchor='x'),  # , scaleratio=1),
        # width=600,
        # height=600,
        template='plotly_white'
    )

    return fig, r_value


def vargrad_error(log_r, log_pbs, log_pfs):
    # Compute x and y
    log_ratio = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).cpu().detach().numpy()
    log_z = log_ratio.mean()

    mae = np.abs(log_z - log_ratio).mean()
    fig = go.Figure(go.Histogram(x=(log_z - log_ratio), nbinsx=100, name=f'MAE={mae:.2f}', showlegend=True))
    fig.update_layout(xaxis_title='Log Ratio Error', yaxis_title='Count')

    return fig


def conditional_vargrad_error(log_r, log_pbs, log_pfs, cond_inds):
    # Compute x and y
    n_elems = len(torch.unique(cond_inds))
    fig = go.Figure()
    for ind in range(n_elems):
        bools = cond_inds == ind
        log_ratio = (log_r[bools] + log_pbs[bools].sum(-1) - log_pfs[bools].sum(-1)).cpu().detach().numpy()
        log_z = log_ratio.mean()

        fig.add_trace(go.Violin(
            x=(log_z - log_ratio), side='positive', orientation='h', width=4,
            showlegend=False, opacity=0.5))
    fig.update_layout(xaxis_title='Log Ratio Error', yaxis_title='Count')

    return fig


''' # version for anywhere-deployment
from scipy.stats import pearsonr

log_Z_learned = log_fs[:, 0]
x = (log_r + log_pbs.sum(-1)).cpu().detach().numpy()
y = (log_Z_learned + log_pfs.sum(-1)).cpu().detach().numpy()

# Get symmetric limits
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
margin = 0.05 * (max_val - min_val)
lim_low = min_val - margin
lim_high = max_val + margin
mae = np.mean(np.abs(x - y))
r_value, _ = pearsonr(x, y)
# Build figure
fig = go.Figure()

# Identity line
fig.add_trace(go.Scatter(
    x=[lim_low, lim_high], y=[lim_low, lim_high],
    mode='lines',
    line=dict(color='gray', dash='dash'),
    showlegend=False,
    hoverinfo='skip'
))

# Scatter points
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(size=6, opacity=0.7, color=log_r.cpu().detach()),
    name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
))

# Layout adjustments
fig.update_layout(
    title='Parity Plot',
    xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
    yaxis=dict(title='log_f + log_pf', range=[lim_low, lim_high], scaleanchor='x', scaleratio=1),
    # width=600,
    # height=600,
    template='plotly_white'
)
fig.show()

'''


def get_dimwise_coverage(test_samples, ref_samples, n_bins=24, tau=10, cmin=1):
    device = test_samples.device
    N, D = test_samples.shape
    q = torch.linspace(0, 1, n_bins + 1, device=device)
    edges = torch.empty(D, n_bins + 1, device=device)
    for j in range(D):
        edges[j] = torch.quantile(ref_samples[:, j], q)

    interior = edges[:, 1:-1]  # [D, B-1]
    per_dim_cov = torch.empty(D, device=device)
    expected = N / n_bins
    thresh = max(cmin, expected / tau)

    for j in range(D):
        idx = torch.bucketize(test_samples[:, j].contiguous(), interior[j], right=False)
        # idx in [0, B-1]
        counts = torch.bincount(idx, minlength=n_bins)
        covered = (counts >= thresh).float().mean()  # fraction of bins covered
        per_dim_cov[j] = covered

    ref_cov = torch.empty(D, device=device)
    for j in range(D):
        idx = torch.bucketize(ref_samples[:, j].contiguous(), interior[j], right=False)
        # idx in [0, B-1]
        counts = torch.bincount(idx, minlength=n_bins)
        covered = (counts >= thresh).float().mean()  # fraction of bins covered
        ref_cov[j] = covered

    return (per_dim_cov / ref_cov.clip(min=0.01)).clip(min=0, max=1)


@torch.no_grad()
def log_ess_frac(log_pf, log_pb, repeats):
    """
    Effective sample size fraction of the per-x importance weights, used as a
    support-aware convergence signal for the MLE-bound objective.

    Detects mode *dropping*: when the forward policy abandons buffer modes,
    those samples get tiny Pf, the weights concentrate, and ESS collapses.

    Args:
        log_pf: [B*repeats] trajectory-summed forward log-probs (log_pfs.sum(-1)),
                in the (x repeated K times) layout, K=repeats.
        log_pb: [B*repeats] trajectory-summed backward log-probs, same layout.
        repeats: K, number of backward trajectories per terminal state x.

    Returns:
        float in (-inf, 0]. 0.0 == uniform weights (healthy, full support).
        Strongly negative == degenerate weights (a few x carry all the mass).
        Interpret as log(ESS / B): exp(value) is the surviving sample fraction.
    """
    B = log_pf.numel() // repeats
    logw_paths = (log_pf - log_pb).view(B, repeats)

    # per-x marginal log importance weight: logmeanexp over the K paths
    log_w = torch.logsumexp(logw_paths, dim=-1) - np.log(repeats)  # [B]

    # normalize, then ESS = 1 / sum(w_n^2)  ->  log_ess = -logsumexp(2 log w_n)
    log_w_n = log_w - torch.logsumexp(log_w, dim=0)
    log_ess = -torch.logsumexp(2 * log_w_n, dim=0)

    return (log_ess - np.log(B)).item()


@torch.no_grad()
def sliced_wasserstein(sampled_latents, prior_latents, n_proj=50, p=1, generator=None):
    """
    Sliced-Wasserstein distance between sampler output and the prior/buffer,
    used as a support-aware monitoring curve.

    Detects mode *invention* (mass where the target has none) as well as
    dropping, because the random projections mix dimensions -- unlike per-dim
    binned coverage it sees joint structure. Sizes need NOT match; compared
    via quantiles, not sorted pairing.

    Args:
        sampled_latents: [N, D] latents from the sampler (std_params space).
        prior_latents:   [M, D] latents from the prior/buffer (same space).
        n_proj: number of random 1-D projection directions to average over.
        p: Wasserstein order (1 = mean abs quantile gap, 2 = RMS).
        generator: optional torch.Generator for reproducible projections.

    Returns:
        float >= 0. 0 == identical distributions (up to projection noise).
    """
    a = sampled_latents.detach()
    b = prior_latents.detach().to(a.device, a.dtype)
    D = a.shape[1]
    assert b.shape[1] == D, "latent dims must match"

    theta = torch.randn(D, n_proj, device=a.device, dtype=a.dtype, generator=generator)
    theta = theta / theta.norm(dim=0, keepdim=True)

    a_proj = a @ theta  # [N, n_proj]
    b_proj = b @ theta  # [M, n_proj]

    # 1-D OT per projection via quantile matching (handles N != M)
    n_q = min(a.shape[0], b.shape[0])
    qs = torch.linspace(0, 1, n_q, device=a.device, dtype=a.dtype)
    a_q = torch.quantile(a_proj, qs, dim=0)  # [n_q, n_proj]
    b_q = torch.quantile(b_proj, qs, dim=0)  # [n_q, n_proj]

    gap = (a_q - b_q).abs()
    if p == 1:
        return gap.mean().item()
    elif p == 2:
        return gap.pow(2).mean().sqrt().item()
    else:
        return (gap.pow(p).mean() ** (1.0 / p)).item()
