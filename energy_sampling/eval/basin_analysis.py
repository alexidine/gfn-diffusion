from energy_sampling.eval.basin_map_utils import agg_cluster, fit_cluster_classifier, dim_reduction, get_cluster_colors
from mxtaltools.dataset_utils.utils import collate_data_list
import plotly.graph_objects as go
import torch

chunk_path = r'a_nice_string'

opt1_trajectory = torch.load(chunk_path, weights_only=False)

opt_traj_samples = torch.stack([collate_data_list(elem).cell_params_to_gen_basis() for elem in
                                opt1_trajectory])  # [traj_len, num_samples, dim]
opt_traj_energies = torch.stack([collate_data_list(elem).lj_pot for elem in
                                 opt1_trajectory])  # [traj_len, num_samples, dim]
start_points = opt_traj_samples[0]
optimized_points = opt_traj_samples[-1]
optimized_energies = opt_traj_energies[-1]

cluster_labels = agg_cluster(optimized_points)
rf, cluster_labels_noised = fit_cluster_classifier(cluster_labels, optimized_energies, opt_traj_samples)
traj_cluster_pred = rf.predict(opt_traj_samples.flatten(0, 1))
traj_reshaped_pred = traj_cluster_pred.reshape(opt_traj_samples.shape[0], opt_traj_samples.shape[1])
opt_embedding, traj_embedding = dim_reduction(
    optimized_points, opt_traj_samples,
    cluster_labels, traj_reshaped_pred,
    supervise=False, sample_each=10)

point_colors, traj_colors = get_cluster_colors(cluster_labels_noised, traj_cluster_pred)

fig = go.Figure()
cind_hit = {int(ind): False for ind in np.unique(cluster_labels_noised)}
for ind in range(traj_embedding.shape[1]):
    traj_color = point_colors[ind]
    cind = cluster_labels_noised[ind]
    fig.add_scatter(x=traj_embedding[:, ind, 0], y=traj_embedding[:, ind, 1],
                    mode='lines+markers',
                    marker=dict(
                        color=traj_color,
                        size=6,
                        opacity=0.75
                    ),
                    line_color=traj_color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
                    name=f'Cluster_{cind}',
                    legendgroup=f'Cluster_{cind}',
                    showlegend=not cind_hit[cind]
                    )
    cind_hit[cind] = True

fig.add_scatter(x=opt_embedding[:, 0], y=opt_embedding[:, 1],
                mode='markers',
                marker=dict(
                    color=point_colors,
                    size=10,
                    line=dict(color='black', width=2)
                ),
                name='Optimized Samples')

fig.show(renderer='browser')

