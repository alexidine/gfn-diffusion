from tqdm import tqdm
import numpy as np
from energy_sampling.eval.basin_map_utils import agg_cluster, fit_cluster_classifier, dim_reduction, get_cluster_colors
from mxtaltools.dataset_utils.utils import collate_data_list
import plotly.graph_objects as go
import torch
import os
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import os

chunk_path = r'D:\crystal_datasets\mol0_basin'
os.chdir(chunk_path)
chunks = os.listdir()

if not os.path.exists('mol0_trajectories'):
    trajectories = []
    energies = []
    for c in tqdm(chunks):
        opt_traj = torch.load(c, weights_only=False)
        # extract initial state, final state, and final state energy
        traj, ens = [], []
        for i in range(len(opt_traj)):
            b = collate_data_list(opt_traj[i])
            traj.append(b.latent_params())
            ens.append(b.lj_pot)

        trajectories.append(traj)
        energies.append(ens)

    torch.save([trajectories, energies], 'mol0_trajectories')
else:
    out = torch.load('mol0_trajectories', weights_only=False)
    trajectories, energies = out

start_points = torch.cat([t[0] for t in trajectories])
optimized_points = torch.cat([t[-1] for t in trajectories])
optimized_energies = torch.cat([e[-1] for e in energies])
aa = 1

cluster_labels = agg_cluster(optimized_points)
rf, cluster_labels_noised = fit_cluster_classifier(cluster_labels,
                                                   optimized_energies,
                                                   start_points[None, ...])
# # traj_cluster_pred = rf.predict(opt_traj_samples.flatten(0, 1))
# # traj_reshaped_pred = traj_cluster_pred.reshape(opt_traj_samples.shape[0], opt_traj_samples.shape[1])
# opt_embedding, traj_embedding = dim_reduction(
#     optimized_points, start_points[None, ...],
#     cluster_labels_noised, cluster_labels_noised[None, ...],
#     supervise=True, sample_each=10)
#
# point_colors, traj_colors = get_cluster_colors(cluster_labels_noised, cluster_labels_noised)
#
# fig = go.Figure()
# cind_hit = {int(ind): False for ind in np.unique(cluster_labels_noised)}
# for ind in range(traj_embedding.shape[1]):
#     traj_color = point_colors[ind]
#     cind = cluster_labels_noised[ind]
#     fig.add_scatter(x=traj_embedding[:, ind, 0], y=traj_embedding[:, ind, 1],
#                     mode='lines+markers',
#                     marker=dict(
#                         color=traj_color,
#                         size=6,
#                         opacity=0.75
#                     ),
#                     line_color=traj_color.replace(')', ', 0.1)').replace('rgb', 'rgba'),
#                     name=f'Cluster_{cind}',
#                     legendgroup=f'Cluster_{cind}',
#                     showlegend=not cind_hit[cind]
#                     )
#     cind_hit[cind] = True
#
# fig.add_scatter(x=opt_embedding[:, 0], y=opt_embedding[:, 1],
#                 mode='markers',
#                 marker=dict(
#                     color=point_colors,
#                     size=10,
#                     line=dict(color='black', width=2)
#                 ),
#                 name='Optimized Samples')
#
# fig.show(renderer='browser')
#
# assert False
