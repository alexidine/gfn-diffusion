import os

import hdbscan
import numpy as np
import plotly.colors as pc
import torch
import wandb
from mxtaltools.reporting.online import simple_embedding_fig, simple_cell_hist, simple_cell_scatter_fig, \
    log_crystal_samples, simple_latent_hist
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from umap import UMAP

from plot_utils import get_plotly_fig_size_mb
from sample_metrics import compute_distribution_distances
from utils import logmeanexp


@torch.no_grad()
def log_partition_function(initial_state, gfn, discretizer, energy_function, mol_batch):
    condition = energy_function.get_conditioning_tensor(mol_batch)
    (states, log_pfs, log_pbs, log_fs,
     means_f, logvars_f, means_b, logvars_b) = gfn.get_trajectory_fwd(initial_state,
                                                                      discretizer,
                                                                      None,
                                                                      condition,
                                                                      return_gauss_params=True)
    log_r, sample_batch = energy_function.log_reward(
        states[:, -1], mol_batch=mol_batch,
        log_temperature=condition[:, 0],
        return_exp=True)
    log_weight = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)

    log_Z = logmeanexp(log_weight)
    log_Z_lb = log_weight.mean()
    log_Z_learned = log_fs[:, 0].mean()

    return (states, states[:, -1],
            log_r, log_Z, log_Z_lb, log_Z_learned,
            sample_batch, condition,
            log_pfs, log_pbs, log_fs,
            means_f, logvars_f, means_b, logvars_b)


@torch.no_grad()
def mean_log_likelihood(terminal_state, gfn, log_reward_fn, num_evals=10):
    bsz = terminal_state.shape[0]
    terminal_state = terminal_state.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(terminal_state, None, log_reward_fn)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
    return logmeanexp(log_weight, dim=1).mean()


@torch.no_grad()
def get_sample_metrics(samples, gt_samples=None, final_eval=False):
    if gt_samples is None:
        return

    return compute_distribution_distances(samples.unsqueeze(1), gt_samples.unsqueeze(1), final_eval)


@torch.no_grad()
def eval_step(energy_function,
              gfn_model,
              discretizer,
              init_state,
              buffer,
              do_figures: bool = True,
              mol_batch=None,
              bwd_training: bool = False,
              add_to_buffer: bool = False):
    gfn_model.eval()

    (flow_states, samples, log_r, log_Z, log_Z_lb,
     log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_fs,
     f_means_f, f_vars_f, f_means_b, f_vars_b) = log_partition_function(
        init_state, gfn_model, discretizer, energy_function, mol_batch)

    metrics = log_eval_scalars_and_dists(condition, energy_function, log_Z, log_Z_lb, log_Z_learned, log_r,
                                         sample_batch, buffer)

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
                                     sample_batch.detach().cpu())
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
        log_crystals(sample_batch)
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
                      f_vars_f, f_means_f, f_vars_b, f_means_b, sample_batch):
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

    sample_embedding, anchor_embedding, cluster_ind, anchor_energies, all_energies = embed_samples(
        buffer_std_params,
        buffer_reward,
        std_cell_params.cpu().detach().numpy(),
        log_r.cpu().detach().numpy(),
        sample_size=5000
        )

    fig_dict['Sample Embedding'] = cluster_fig(sample_embedding, anchor_embedding, cluster_ind, anchor_energies, all_energies, color_mode='cluster')
    fig_dict['Sample Embedding w Energy'] = cluster_fig(sample_embedding, anchor_embedding, cluster_ind, anchor_energies, all_energies, color_mode='energy')

    # coverage metrics
    fig_dict['Num Clusters'] = np.sum(np.unique(cluster_ind) > 0)
    fig_dict['Noise Fraction'] = np.mean(cluster_ind == -1)

    fig = cluster_hist_fig(cluster_ind)
    fig_dict['Cluster Hist'] = fig

    conditional = len(condition.unique()) != 1
    if conditional:
        fig_dict['temp/Learned Z vs T'] = Z_vs_T_fig(gfn_model, init_state)
        fig_dict['temp/T vs Energy'] = T_vs_E_fig(condition, sample_batch)
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
    fig_dict['Lattice Features Distribution'] = simple_cell_hist(sample_batch, buffer_cell_params)
    fig_dict['Lattice Latents Distribution'] = simple_latent_hist(sample_batch, buffer_latent_params)
    fig_dict['Sample Scatter'] = simple_cell_scatter_fig(
        sample_batch,
        (condition[:, 0].cpu().detach().numpy()) if condition is not None else None,
        aux_scalar_name='log_temperature' if condition is not None else None)

    return fig_dict


def cluster_hist_fig(cluster_ind):
    uniques, counts = np.unique(cluster_ind, return_counts=True)
    fig = go.Figure()
    fig.add_bar(x=uniques, y=counts)
    fig.update_layout(xaxis_title='Cluster Ind', yaxis_title='Cluster Size')
    return fig


def agglomerative_cluster(ens, samples, energy_cutoff: float = 0.0, min_dist: float = 5):
    # Can use full dataset or low-energy subset only
    mask = ens < energy_cutoff
    X_lowE = samples[mask]

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=min_dist,  # stop when all clusters > threshold apart
        #linkage='ward',
        linkage='complete',
    )
    labels = model.fit_predict(X_lowE)

    # Pick 1 lowest-energy point from each cluster as an anchor
    anchors = []
    anchor_indices = []

    for lbl in np.unique(labels):
        members = np.where(labels == lbl)[0]
        best_idx = members[np.argmin(ens[mask][members])]
        anchors.append(X_lowE[best_idx])
        anchor_indices.append(np.where(mask)[0][best_idx])
    return anchors, anchor_indices


def embed_samples(ref_samples, ref_rewards, samples, sample_rewards, sample_size: int, temperature: float = 0.1):
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
        if len(ref_samples) > sample_size:
            weights = np.exp((ref_rewards-ref_rewards.max()) / temperature) + 1e-2
            weights /= weights.sum()
            inds_to_keep = np.random.choice(len(ref_samples), sample_size, replace=False, p=weights)

            ref_samples = ref_samples[inds_to_keep]
            ref_rewards = ref_rewards[inds_to_keep]
        samples_to_fit = np.concatenate([ref_samples, samples])
        energies_to_fit = -np.concatenate([ref_rewards, sample_rewards])
    else:
        samples_to_fit = samples
        energies_to_fit = -sample_rewards

    # prewhiten data via PCA
    pca = PCA(n_components=12)
    pca_embedding = pca.fit_transform(samples_to_fit)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20,
                                min_samples=20,
                                metric='euclidean',
                                core_dist_n_jobs=-1)
    cluster_ind = clusterer.fit_predict(pca_embedding)  # -1 means "noise"
    sample_cluster_ind = cluster_ind[-len(samples):]

    n_clusters = np.sum(np.unique(cluster_ind) >= 0)

    if n_clusters > 0:
        # extract lowest energy sample
        cluster_anchor_inds = {}
        for cluster_id in np.unique(cluster_ind):
            if cluster_id == -1:
                continue  # skip noise
            in_cluster = np.where(cluster_ind == cluster_id)[0]
            best_ind = in_cluster[np.argmin(energies_to_fit[in_cluster])]
            cluster_anchor_inds[cluster_id] = best_ind
        anchor_inds = np.array(list(cluster_anchor_inds.values()))
    else:
        anchor_inds = np.array([np.argmin(energies_to_fit)])
        cluster_ind[anchor_inds] = 0  # make a cluster if there are none naturally

    # dimension reduction
    sampled_inds = np.random.choice(len(samples_to_fit), size=500, replace=False)
    samples_to_fit = np.concatenate([samples_to_fit[sampled_inds], samples_to_fit[anchor_inds]])  # ensure anchor inds get into the embedding
    reducer = UMAP(n_components=2, n_neighbors=10, min_dist=0.05,
                   metric='euclidean', densmap=False)
    fit_embedding = reducer.fit_transform(samples_to_fit)
    sample_embedding = reducer.transform(samples)

    return sample_embedding, fit_embedding[-len(anchor_inds):], sample_cluster_ind, energies_to_fit[np.array(anchor_inds)], -sample_rewards


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
        radius = np.ptp(vor.points)* 2  # large enough

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
                sample_energies, color_mode):
    """
    Figure for the clusters in PC space + Voronoi assignments
    :param sample_embedding:
    :param anchor_embedding:
    :param cluster_ind:
    :return:
    """
    try:
        vor = Voronoi(anchor_embedding)
        polygons = voronoi_finite_polygons_2d(vor)
    except:
        polygons = None

    energies = np.array(anchor_energies)
    norm_energies = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)

    colorscale = pc.get_colorscale("Jet")
    line_colors = [pc.sample_colorscale(colorscale, [v])[0] for v in norm_energies]

    x_all = np.concatenate([sample_embedding[:, 0], anchor_embedding[:, 0]])
    y_all = np.concatenate([sample_embedding[:, 1], anchor_embedding[:, 1]])

    colorscale_name = "rainbow"  # or "viridis", "plasma", etc.
    n_clusters = len(anchor_embedding)
    distinct_colors = pc.sample_colorscale(colorscale_name, [i / max(n_clusters - 1, 1) for i in range(n_clusters)])
    cluster_to_color = {i: distinct_colors[i] for i in range(n_clusters)}
    cluster_to_color.update({-1: 'rgb(0,0,0)'})  # black for noise dimension
    if color_mode == 'cluster':
        mapped_colors = [cluster_to_color[c] for c in cluster_ind]
    elif color_mode == 'energy':
        mapped_colors = sample_energies

    fig = go.Figure()

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

    fig.add_trace(go.Scattergl(x=anchor_embedding[:, 0],
                               y=anchor_embedding[:, 1],
                               mode='markers',
                               opacity=1,
                               name='Minima',
                               showlegend=False,
                               marker=dict(
                                   size=15,
                                   color=[cluster_to_color[ind] for ind in range(len(anchor_embedding))],  # Fill color
                                   line=dict(
                                       color=line_colors,  # <-- variable border color!
                                       width=4
                                   )
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


def log_eval_scalars_and_dists(condition, energy_function, log_Z, log_Z_lb, log_Z_learned, log_r,
                               sample_batch, buffer=None):
    """Scalar / distribution metrics"""
    metrics = {}
    metrics['Empirical log Z'] = log_Z.cpu().detach().numpy()
    metrics['Empirical log Z LB'] = log_Z_lb.cpu().detach().numpy()
    metrics['log Z learned'] = log_Z_learned.cpu().detach().numpy()
    metrics['Mean Cacking Coeff'] = sample_batch.packing_coeff.mean().cpu().detach().numpy()
    metrics['Packing Coeff'] = sample_batch.packing_coeff.clip(max=2).cpu().detach().numpy()
    metrics['Mean Silu Energy'] = sample_batch.silu_pot.mean().cpu().detach().numpy()
    metrics['Mean Sample Energy'] = sample_batch.gfn_energy.mean().cpu().detach().numpy()
    metrics['Sample Energy Distribution'] = sample_batch.gfn_energy.cpu().detach().numpy()
    metrics['Mean Sample Reward'] = log_r.mean().cpu().detach().numpy()
    metrics['sample Reward Distribution'] = log_r.cpu().detach().numpy()
    metrics['Crystal Log Temperature'] = condition[:, 0]
    metrics['Crystal Mean Log Temperature'] = condition[:, 0].mean()
    metrics['Crystal Min Temperature'] = energy_function.min_temperature
    metrics['Crystal Max Temperature'] = energy_function.max_temperature
    metrics['Crystal Static Temperature'] = energy_function.temperature
    metrics['Crystal Repulsion Factor'] = energy_function.lj_repulsion
    metrics['Ellipsoid Scale'] = energy_function.ellipsoid_scale
    metrics['Temperature Scaling Factor'] = energy_function.temperature_scaling_factor
    metrics['Density Loss Coefficient'] = energy_function.density_coeff

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    std_params = sample_batch.cell_params_to_gen_basis()

    metrics['Total Var'] = std_params.var(dim=0).sum().item()
    metrics['Total Mean'] = std_params.mean(dim=0).sum().item()

    eigvals = torch.linalg.svdvals(std_params - std_params.mean(0)) ** 2
    explained_var_ratio = eigvals / eigvals.sum()
    d_eff = (explained_var_ratio ** 2).sum() ** -1

    metrics['Effective Dimension'] = d_eff.item()
    for ind, feat in enumerate(lattice_features):
        metrics[feat + '_mean'] = std_params[:, ind].mean().item()
        metrics[feat + '_var'] = std_params[:, ind].var().item()
        metrics[feat + '_expl_var_rat'] = explained_var_ratio[ind].item()

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
                np.quantile(buffer.scores_np_list, q=p)
                for p in np.linspace(0, 1, 50)
            ])
            metrics['Buffer Mean Score'] = np.mean(buffer.scores_np_list)
    return metrics


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
    fig.add_scatter(x=x,
                    y=y,
                    marker_color=color,
                    name=f'R = {r_value:.3f}',
                    showlegend=True,
                    mode='markers',
                    )
    fig.update_layout(xaxis_title='Forward Prob', yaxis_title='Backward Prob')
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
        yaxis=dict(title='log_Z + log_pf', range=[lim_low, lim_high], scaleanchor='x', scaleratio=1),
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
