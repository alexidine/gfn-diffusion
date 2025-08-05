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

from energy_sampling.eval.utils import log_partition_function, get_plotly_fig_size_mb
from energy_sampling.utils import logmeanexp
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.reporting.figures import simple_cell_hist, simple_cell_scatter_fig, \
    log_crystal_samples


@torch.no_grad()
def eval_step(energy_function,
              gfn_model,
              discretizer,
              init_state,
              buffer,
              args,
              do_figures: bool = True,
              mol_batch=None,
              bwd_training: bool = False,
              add_to_buffer: bool = False):
    gfn_model.eval()

    (flow_states, samples, log_r, log_Z, log_Z_lb,
     log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_fs,
     f_means_f, f_vars_f, f_means_b, f_vars_b,
     log_T_tensor) = log_partition_function(
        init_state, gfn_model, discretizer, energy_function, mol_batch)

    metrics = log_eval_scalars_and_dists(energy_function, log_Z, log_Z_lb, log_Z_learned, log_r,
                                         sample_batch, log_T_tensor, args, buffer)

    if add_to_buffer:
        buffer.add(sample_batch.detach().cpu().to_data_list())  # add evaluation samples to buffer

    if do_figures:
        fig_dict = generate_fwd_figs(buffer,
                                     energy_function,
                                     condition,
                                     flow_states,
                                     gfn_model,
                                     init_state,
                                     log_fs,
                                     log_pbs,
                                     log_pfs,
                                     log_r,
                                     f_vars_f, f_means_f, f_vars_b, f_means_b,
                                     sample_batch.detach().cpu(),
                                     log_T_tensor)
        if bwd_training:
            fig_dict = generate_bwd_figs(fig_dict, buffer, gfn_model, init_state, discretizer)

        for key in fig_dict.keys():
            fig = fig_dict[key]
            try:
                if get_plotly_fig_size_mb(fig) > 1:  # bigger than 1 MB
                    fig.write_image(key + 'fig.png', width=720,
                                    height=512)  # save the image rather than the fig, for size reasons
                    fig_dict[key] = wandb.Image(key + 'fig.png')
            except:
                pass

        metrics.update(fig_dict)

    "Crystal samples"
    try:
        batch_to_log = collate_data_list(sample_batch.detach().cpu().to_data_list()[:6])
        batch_to_log.box_analysis()
        log_crystals(batch_to_log)
    except:  # sometimes it fails IDK
        pass

    gfn_model.train()
    return metrics


def log_crystals(sample_batch):
    cluster_batch = sample_batch.mol2cluster(cutoff=6,
                                             supercell_size=10,
                                             align_to_standardized_orientation=True)
    cluster_batch.construct_radial_graph(cutoff=6)
    lj_energy, normed_lj_energy = cluster_batch.compute_LJ_energy()
    cluster_batch.lj_pot = lj_energy
    samples_to_log, filenames = log_crystal_samples(sample_batch=cluster_batch, return_filenames=True)
    [wandb.log({f'crystal_sample_{ind}': samples_to_log[ind]}, commit=False) for ind in range(len(samples_to_log))]
    [os.remove(file) for file in filenames]  # delete this cif as a temporary file


def generate_fwd_figs(buffer, energy_function,
                      condition, flow_states,
                      gfn_model, init_state,
                      log_fs, log_pbs, log_pfs, log_r,
                      f_vars_f, f_means_f, f_vars_b, f_means_b, sample_batch,
                      log_T_tensor):
    fig_dict = {}

    buffer_cell_params, buffer_latent_params, buffer_std_params, buffer_reward, buffer_batch = get_buffer_stats(buffer)
    std_cell_params = sample_batch.cell_params_to_gen_basis().cpu().detach()

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

    (sample_embedding, anchor_embedding,
     anchor_states, sample_cluster_inds,
     anchor_energies, all_energies, buffer_cluster_inds,
     watershed_idx, watershed_range) = embed_samples(
        buffer_std_params,
        buffer_reward,
        std_cell_params.cpu().detach().numpy(),
        log_r.cpu().detach().numpy(),
        max_ref_samples=500
    )
    try:
        fig_dict['Boltzmann Fit'] = boltzmann_fig(log_r)
    except:  # some issues in the above I don't care to fix
        pass
    fig_dict['Sample Embedding'] = cluster_fig(sample_embedding, anchor_embedding, sample_cluster_inds, anchor_energies,
                                               all_energies, 'cluster', watershed_idx, watershed_range)
    fig_dict['Sample Embedding w Energy'] = cluster_fig(sample_embedding, anchor_embedding, sample_cluster_inds,
                                                        anchor_energies, all_energies, 'energy', watershed_idx,
                                                        watershed_range)

    # coverage metrics
    fig_dict['Num Sample Clusters'] = np.sum(np.unique(sample_cluster_inds) >= 0)
    fig_dict['Num Buffer Clusters'] = np.sum(np.unique(buffer_cluster_inds) >= 0)

    fig = cluster_hist_fig(sample_cluster_inds)
    fig_dict['Cluster Hist'] = fig

    # todo rewrite this for general conditioning
    # if energy_function.temperature_conditioning:
    #     fig_dict['temp/Learned Z vs T'] = Z_vs_T_fig(gfn_model, init_state)
    #     fig_dict['temp/T vs Energy'] = T_vs_E_fig(condition, sample_batch)
    fig_dict['Forward Gauss Params'] = mean_var_fig(f_vars_f, f_means_f,
                                                    f_vars_b, f_means_b)
    fig_dict['Mean Fwd F Drift'] = f_means_f.abs().mean()
    fig_dict['Mean Fwd B Drift'] = f_means_b.abs().mean()
    fig_dict['Mean Fwd F Var'] = f_vars_f.mean()
    fig_dict['Mean Fwd B Var'] = f_vars_b.mean()
    fig_dict['Traj Mean Step Sizes'] = mean_flow_step_sizes(flow_states)
    fig_dict['Pf vs Pb'] = Pf_vs_Pb_fig(log_pfs, log_pbs, log_r)
    fig_dict['TB Parity Plot'], fig_dict['Forward TB R Value'] = flow_parity_plot(log_r, log_fs[:, 0], log_pbs, log_pfs)
    fig_dict['VG Error'] = vargrad_error(log_r, log_pbs, log_pfs)  # todo tune this up for conditional modelling
    fig_dict['Lattice Latents Trajectories'] = visualize_latent_trajs(flow_states.cpu().detach().numpy(),
                                                                      20,
                                                                      log_r.cpu().detach().numpy())
    fig_dict['Lattice Features Distribution'], cell_klds = simple_cell_hist(sample_batch, buffer_cell_params,
                                                                 n_kde_points=200, bw_ratio=10, mode='cell')
    fig_dict['Lattice Latents Distribution'], latent_klds = simple_cell_hist(sample_batch, buffer_latent_params,
                                                                n_kde_points=200, bw_ratio=10, mode='latent')

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']

    for ind, feat in enumerate(lattice_features):
        fig_dict[f'{feat} Cell KLD'] = cell_klds[ind]
        fig_dict[f'{feat} Latent KLD'] = latent_klds[ind]
    fig_dict['Mean Cell KLD'] = np.mean(cell_klds)
    fig_dict['Mean Latent KLD'] = np.mean(latent_klds)

    log_T = len(torch.unique(log_T_tensor)) > 1
    fig_dict['Sample Scatter'] = simple_cell_scatter_fig(
        sample_batch,
        sample_cluster_inds,
        aux_scalar_name='Log Temp' if log_T else None,
        aux_array=log_T_tensor.cpu().detach().numpy() if log_T else None,
    )

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
    quantile_cutoff = 0.95
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
    boltzmann_y /= np.trapz(boltzmann_y, x_kde)
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
        title_text='Boltzmann Distribution Check (Linear & Log View)',
        #height=500,
        #width=1000,
        template='plotly_white'
    )

    fig.update_xaxes(title_text='Energy', row=1, col=1)
    fig.update_yaxes(title_text='P(E)', row=1, col=1)

    fig.update_xaxes(title_text='Energy', row=1, col=2)
    fig.update_yaxes(title_text='log P(E)', row=1, col=2)

    return fig


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
    umap_model = UMAP(n_components=2, n_neighbors=30, min_dist=0.01)
    sample_embedding = umap_model.fit_transform(samples_to_fit)

    # fit energy-weighted 2D gaussian kde on the UMAP data
    beta = 0.001
    weights = np.exp(-beta * (energies_to_fit - np.min(energies_to_fit)))  # stabilize exponent
    weights /= weights.sum()
    kde = gaussian_kde(sample_embedding.T, weights=weights, bw_method=0.1)

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
    min_kde_range = 2  #np.sqrt(grid_size**2 / 500)  # to stuff M evenly spaced clusters
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
                                   #[cluster_to_color[ind] for ind in range(len(anchor_embedding))],  # Fill color
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


#
# from scipy.spatial.distance import cdist
# X = buffer_std_params
# energies = -buffer_reward.cpu().detach().numpy()
#
# """
# cluster generation
# """
# k = 10000
# min_dist = 5.0
# energy_cutoff = -3
#
# idx_sorted = np.argsort(energies)
# anchors = []
# anchor_indices = []
#
# for idx in idx_sorted:
#     candidate = X[idx]
#     if energies[idx] > energy_cutoff:
#         break
#     if len(anchors) == 0:
#         anchors.append(candidate)
#         anchor_indices.append(idx)
#     else:
#         dists = cdist([candidate], anchors)
#         if np.min(dists) >= min_dist:
#             anchors.append(candidate)
#             anchor_indices.append(idx)
#     if len(anchors) >= k:
#         break
#
# """
# cluster assignment
# """
# dists = cdist(X, anchors)  # shape (N_samples, N_anchors)
# cluster_ind = np.argmin(dists, axis=1)
#
# import umap
# reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05)
# reduc
# er.fit(anchors)  # Learn manifold from anchors
# embedding = reducer.transform(X)  # Project full dataset
# anchor_embedding = reducer.transform(anchors)
#
# color_array = cluster_ind
# fig = go.Figure()
#
# fig.add_trace(go.Scattergl(x=embedding[:, 0],
#                            y=embedding[:, 1],
#                            mode='markers',
#                            opacity=0.85,
#                            name='Policy Samples',
#                            showlegend=True,
#                            marker=dict(
#                                size=6,
#                                color=color_array,
#                                colorscale="portland",
#                                colorbar=dict(title="Cluster Membership")
#                            )
#                            ))
#
# fig.add_trace(go.Scattergl(x=anchor_embedding[:, 0],
#                            y=anchor_embedding[:, 1],
#                            mode='markers',
#                            opacity=1,
#                            name='Known Modes',
#                            showlegend=True,
#                            marker=dict(
#                                size=15,
#                                color='green',  # Fill color
#                                line=dict(
#                                    color='black',  # Outline color
#                                    width=4  # Outline thickness
#                                )
#                            )
#                            ))
#
# fig.show()


def generate_bwd_figs(fig_dict, buffer, gfn_model, init_state, discretizer):
    terminal_state, b_log_r, crystal_batch, condition = buffer.sample(
        return_conditioning=True,
        override_batch=len(init_state))
    (backward_flow_states, b_log_pfs, b_log_pbs, b_log_fs,
     b_means_f, b_vars_f, b_means_b, b_vars_b) = gfn_model.get_trajectory_bwd(
        terminal_state.to(gfn_model.device), discretizer, condition.to(gfn_model.device), return_gauss_params=True)

    fig_dict['Backward Latents Trajectories'] = visualize_latent_trajs(
        backward_flow_states.cpu().detach().numpy(),
        n_trajs=20, log_r=b_log_r.cpu().detach().numpy())

    fig_dict['Backward Pf vs Pb'] = Pf_vs_Pb_fig(b_log_pfs, b_log_pbs, b_log_r)
    fig_dict['Backward TB Parity Plot'], fig_dict['Backward TB R Value'] = flow_parity_plot(b_log_r.to(b_log_fs.device),
                                                                                            b_log_fs[:, 0], b_log_pbs,
                                                                                            b_log_pfs)
    fig_dict['Backward Gauss Params'] = mean_var_fig(b_vars_f, b_means_f,
                                                     b_vars_b, b_means_b)
    fig_dict['Mean Bwd F Drift'] = b_means_f.abs().mean()
    fig_dict['Mean Bwd B Drift'] = b_means_b.abs().mean()
    fig_dict['Mean Bwd F Var'] = b_vars_f.mean()
    fig_dict['Mean Bwd B Var'] = b_vars_b.mean()

    fig_dict['Bwd Traj Mean Step Sizes'] = mean_flow_step_sizes(backward_flow_states)

    log_weight = b_log_r + b_log_pbs.sum(-1) - b_log_pfs.sum(-1)
    log_Z = logmeanexp(log_weight)
    log_Z_lb = log_weight.mean()
    fig_dict['Bwd Empirical log Z'] = log_Z.cpu().detach().numpy()
    fig_dict['Bwd Empirical log Z LB'] = log_Z_lb.cpu().detach().numpy()

    return fig_dict


def get_buffer_stats(buffer):
    if len(buffer) > 0:
        samples_to_take = min(10000, len(buffer))
        # take samples according to the sampler weighting, rather than random trash in the buffer
        buffer_latent_params, buffer_reward, buffer_batch = buffer.sample(
            temperature=torch.ones(samples_to_take), override_batch=samples_to_take)
        buffer_cell_params = buffer_batch.cell_parameters().cpu().detach().numpy()
        buffer_latent_params = buffer_batch.cell_params_to_gen_basis().cpu().detach().numpy()
        buffer_std_params_for_embedding = buffer_batch.cell_params_to_gen_basis().cpu().detach().numpy()
        reward = buffer_reward.cpu().detach().numpy()
        batch = buffer_batch.cpu().detach()
    else:
        buffer_cell_params, buffer_latent_params, buffer_std_params_for_embedding, reward, batch = None, None, None, None, None
    return buffer_cell_params, buffer_latent_params, buffer_std_params_for_embedding, reward, batch


def mean_var_fig(logvars_f, means_f, logvars_b, means_b):
    fig = make_subplots(rows=2, cols=1)
    fig.add_scatter(y=np.nan_to_num(torch.exp(logvars_f).mean(0).cpu().detach().numpy()), name='Pf Var', row=2, col=1)
    fig.add_scatter(y=np.nan_to_num(means_f.abs().mean(0).cpu().detach().numpy()), name='Pf Mean', row=1, col=1)
    fig.add_scatter(y=np.nan_to_num(torch.exp(logvars_b).mean(0).cpu().detach().numpy()), name='Pb Var', row=2, col=1)
    fig.add_scatter(y=np.nan_to_num(means_b.abs().mean(0).cpu().detach().numpy()), name='Pb Mean', row=1, col=1)
    fig.update_layout(xaxis2_title='Trajectory Step')
    return fig


def mean_flow_step_sizes(flow_states):
    mean_step_size = flow_states.diff(dim=1).abs().mean(0)
    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    fig = go.Figure(go.Heatmap(x=lattice_features, y=np.arange(1, flow_states.shape[1] + 1),
                               z=mean_step_size.cpu().detach()))
    return fig


def log_eval_scalars_and_dists(energy_function, log_Z, log_Z_lb, log_Z_learned, log_r,
                               sample_batch, log_T_tensor, args, buffer=None):
    """Scalar / distribution metrics"""
    metrics = {}
    # energies
    for key in sample_batch.keys():
        if 'energy' in key or 'pot' in key:
            val = sample_batch[key].mean().cpu().detach().item()
            metrics['Mean ' + key] = val

    # physical properties
    metrics['Mean Packing Coeff'] = sample_batch.packing_coeff.mean().cpu().detach().item()
    metrics['Packing Coeff'] = sample_batch.packing_coeff.clip(max=2).cpu().detach().numpy()
    metrics['Niggli Overlap'] = sample_batch.niggli_overlap.cpu().detach().numpy()
    metrics['Mean ellipsoid_overlap'] = sample_batch['ellipsoid_overlap'].mean().cpu().detach().item()

    # conditions
    metrics['Crystal Log Temperature'] = log_T_tensor.cpu().detach().numpy()
    metrics['Crystal Mean Log Temperature'] = log_T_tensor.mean().item()

    # training metrics
    metrics['Mean Sample Energy'] = sample_batch.gfn_energy.mean().cpu().detach().item()
    metrics['Sample Energy'] = sample_batch.gfn_energy.clip(max=50).cpu().detach().numpy()

    metrics['Mean Sample Reward'] = log_r.mean().cpu().detach().item()
    metrics['Sample Reward'] = log_r.clip(min=-50).cpu().detach().numpy()

    metrics['Empirical log Z'] = log_Z.cpu().detach().item()
    metrics['Empirical log Z LB'] = log_Z_lb.cpu().detach().item()
    metrics['log Z learned'] = log_Z_learned.cpu().detach().item()

    for elem in energy_function.__dict__.keys():
        thing = energy_function.__dict__[elem]
        if isinstance(thing, float) or isinstance(thing, int):
            metrics['energy_func/' + elem] = thing

    for elem in args.fwd_loss_coeffs.__dict__.keys():
        thing = args.fwd_loss_coeffs.__dict__[elem]
        if isinstance(thing, float) or isinstance(thing, int):
            metrics['loss_coeffs/' + 'fwd_' + elem] = thing

    for elem in args.bwd_loss_coeffs.__dict__.keys():
        thing = args.bwd_loss_coeffs.__dict__[elem]
        if isinstance(thing, float) or isinstance(thing, int):
            metrics['loss_coeffs/' + 'bwd_' + elem] = thing

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    std_params = sample_batch.cell_params_to_gen_basis()

    metrics['Total Var'] = std_params.var(dim=0).mean().cpu().detach().numpy()
    metrics['Total Mean'] = std_params.mean(dim=0).mean().cpu().detach().numpy()

    U, S, Vh = torch.linalg.svd(std_params - std_params.mean(0), full_matrices=False)
    eigvals = S ** 2
    explained_var_ratio = eigvals / eigvals.sum()
    loadings = Vh.T  # shape: (num_features, num_components)
    contrib_per_feature = (loadings ** 2) @ explained_var_ratio  # shape: (num_features,)
    d_eff = (explained_var_ratio ** 2).sum() ** -1

    metrics['Effective Dimension'] = d_eff.item()
    for ind, feat in enumerate(lattice_features):
        metrics[feat + '_mean'] = std_params[:, ind].mean().item()
        metrics[feat + '_var'] = std_params[:, ind].var().item()
        metrics[feat + '_expl_var_rat'] = contrib_per_feature[ind].item()

    cov = torch.cov(std_params.T)  # shape [12, 12]
    volume_proxy = torch.det(cov).clamp_min(1e-12).sqrt().item()
    metrics['Gaussian Proxy Hypervolume'] = np.log(volume_proxy)

    if hasattr(sample_batch, 'ellipsoid_overlap'):
        metrics['mean ellipsoid overlap'] = sample_batch.ellipsoid_overlap.mean().cpu().detach().numpy()
        metrics['ellipsoid overlap'] = sample_batch.ellipsoid_overlap.clip(min=1e-3).log10().cpu().detach().numpy()

    if buffer is not None:
        if len(buffer) > 0:  # todo adjust this to be according to the sampling routine
            metrics['Buffer Length'] = len(buffer)
            metrics['Buffer Quantiles'] = np.array([
                np.quantile(buffer.rewards_list, q=p)
                for p in np.linspace(0, 1, 50)
            ])
            metrics['Buffer Mean Score'] = np.mean(buffer.rewards_list)

    metrics = {k: to_loggable(v) for k, v in metrics.items()}
    return metrics


def to_loggable(v):
    if torch.is_tensor(v):
        v = v.detach().cpu()
        if v.numel() == 1:
            return v.item()
        else:
            return v.numpy()
    return v


def Z_vs_T_fig(gfn_model, init_state):
    log_temps = torch.linspace(-2, 2, 100).to(init_state.device)[:, None].flatten()
    Z_at_T = gfn_model.flow_model(
        gfn_model.conditions_embedding_model(log_temps[:, None])).cpu().detach().flatten()
    fig = go.Figure(go.Scatter(x=log_temps.cpu().detach(),
                               y=Z_at_T.cpu().detach(),
                               mode='lines'))
    fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Log Partition Function')
    return fig


def T_vs_E_fig(condition, sample_batch):
    fig = go.Figure()
    x = condition[:, 0].cpu().detach().numpy()
    y = sample_batch.gfn_energy.cpu().detach().numpy()
    fig.add_histogram2d(x=x,
                        y=np.log10(y - y.min() + 1e-3),
                        showscale=False,
                        nbinsx=50, nbinsy=50)
    fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Sample Energy')
    return fig


def Pf_vs_R_fig(pf, log_r):
    x = pf.sum(-1).cpu().detach().numpy()
    y = log_r.cpu().detach().numpy()
    r_value, _ = pearsonr(x, y)

    fig = go.Figure()
    fig.add_scatter(x=x,
                    y=y,
                    name=f'R = {r_value:.3f}',
                    showlegend=True,
                    mode='markers',
                    )
    fig.update_layout(xaxis_title='Trajectory Probability', yaxis_title='Terminal Reward')
    return fig


def Pf_vs_Pb_fig(pf, pb, log_r):
    if torch.is_tensor(pf):
        x = pf.sum(-1).cpu().detach().numpy()
        y = pb.sum(-1).cpu().detach().numpy()
    else:
        x = pf.sum(-1)
        y = pb.sum(-1)

    if torch.is_tensor(log_r):
        color = log_r.cpu().detach().numpy()
    else:
        color = log_r

    r_value, _ = pearsonr(x, y)

    fig = go.Figure()
    fig.add_scatter(x=y,
                    y=x,
                    marker_color=color,
                    name=f'R = {r_value:.3f}',
                    showlegend=True,
                    mode='markers',
                    )
    fig.update_layout(yaxis_title='Forward Prob', xaxis_title='Backward Prob')
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
                marker_color=log_r,  #color_hex[j],
                marker_colorscale='viridis',
            ),
                row=row, col=col
            )
    return fig


def flow_parity_plot(log_r, log_Z_learned, log_pbs, log_pfs):
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

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=6, opacity=0.7, color=log_r.cpu().detach(), colorscale='Jet'),
        name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
    ))

    # Layout adjustments
    fig.update_layout(
        title='Parity Plot',
        xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
        yaxis=dict(title='log_Z + log_pf', range=[lim_low, lim_high], scaleanchor='x'),#, scaleratio=1),
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
