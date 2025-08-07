"""
For crystal dataset analysis

"""
import numpy as np
import torch
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from umap import UMAP

from energy_sampling.eval.evaluations import cluster_fig
from mxtaltools.reporting.figures import simple_cell_scatter_fig, simple_cell_hist

from mxtaltools.dataset_utils.utils import collate_data_list

"""load up known samples"""
dataset = torch.load(r'D:\crystal_datasets\eval_qm9_sg2_dataset.pt', weights_only=False)

smi = dataset[0].smiles
dataset = [elem for elem in dataset if elem.smiles == smi]
batch = collate_data_list(dataset)


def cluster_samples(samples_to_fit, energies_to_fit, beta: float = 0.001, bw=0.1):
    umap_model = UMAP(n_components=2, n_neighbors=30, min_dist=0.01)
    sample_embedding = umap_model.fit_transform(samples_to_fit)

    weights = np.exp(-beta * (energies_to_fit - np.min(energies_to_fit)))  # stabilize exponent
    weights /= weights.sum()
    kde = gaussian_kde(sample_embedding.T, weights=weights, bw_method=bw)

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

    return sample_embedding, cluster_assignments, labels, anchor_inds, (x_min, x_max, y_min, y_max)


energies = batch.lj_pot

sample_embedding, cluster_assignments, labels, anchor_inds, watershed_range = \
    cluster_samples(batch.cell_params_to_gen_basis(), energies, beta=1.0, bw=0.1)

f3 = cluster_fig(sample_embedding, sample_embedding[anchor_inds],
                 cluster_assignments, energies[anchor_inds],
                 energies, 'cluster',
                 labels,
                 watershed_range)
f3.show(renderer='browser')

f1, _ = simple_cell_hist(batch, mode='cell')
f1a, _ = simple_cell_hist(batch, mode='latent')
f2 = simple_cell_scatter_fig(batch, aux_array=cluster_assignments)
f1.show(renderer='browser')
f1a.show(renderer='browser')
f2.show(renderer='browser')
aa = 1
