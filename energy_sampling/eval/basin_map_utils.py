from sklearn.cluster import AgglomerativeClustering
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from umap import UMAP
from matplotlib import cm


def agg_cluster(points):
    # clustering
    cluster_model = AgglomerativeClustering(n_clusters=None,
                                            linkage='ward',
                                            distance_threshold=10.0)
    cluster_labels = cluster_model.fit_predict(points)
    return cluster_labels


def fit_cluster_classifier(cluster_labels, optimized_energies, sample_trajs):
    # filtering
    cluster_energies = torch.tensor([
        torch.amin(optimized_energies[torch.tensor(cluster_labels) == label])
        for label in torch.unique(torch.tensor(cluster_labels))
    ])

    # cutoff is within 20% of the minimum binding energy
    energy_cutoff = torch.amin(cluster_energies) * 0.8
    valid_clusters = cluster_energies <= energy_cutoff
    valid_mask = np.isin(cluster_labels, torch.argwhere(valid_clusters).flatten().numpy())

    # Define classification dataset
    cluster_labels_noised = cluster_labels.copy()
    cluster_labels_noised[~valid_mask] = -1  # 'bad' class
    X_train = sample_trajs.flatten(0, 1)  # shape (n_valid_trajs, 12)
    y_train = cluster_labels_noised.repeat(sample_trajs.shape[0])  # integer cluster IDs

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    return rf, cluster_labels_noised


def dim_reduction(optimized_points, opt_traj_samples,
                  opt_labels, traj_labels, supervise=True, sample_each: int = 10,
                  neighbors: int=30):
    umap_model = UMAP(n_components=2, n_neighbors=neighbors, min_dist=0.01)
    X = torch.cat([optimized_points, opt_traj_samples[::sample_each].flatten(0, 1)], dim=0)
    Y = np.concatenate([opt_labels, traj_labels[::sample_each].flatten()])
    sample_embedding = umap_model.fit_transform(X, y=Y if supervise else None)

    end_embeddings = sample_embedding[:len(optimized_points)]
    traj_embedding = umap_model.transform(opt_traj_samples.flatten(0, 1))
    traj_embedding = traj_embedding.reshape(opt_traj_samples.shape[0], opt_traj_samples.shape[1], 2)
    return end_embeddings, traj_embedding


def get_cluster_colors(cluster_labels_noised, traj_cluster_pred):
    # Assign discrete colors

    # Assume cluster_labels_noised includes -1 and positive integers
    unique_labels = np.unique(cluster_labels_noised)
    n_classes = len(unique_labels[unique_labels != -1])

    # Get colors from 'tab20' or 'rainbow'
    cmap = cm.get_cmap('rainbow', n_classes)
    color_dict = {}

    # Assign colors to valid classes
    valid_labels = [label for label in unique_labels if label != -1]
    for i, label in enumerate(valid_labels):
        rgb = cmap(i)[:3]  # ignore alpha
        color_dict[label] = f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})'

    # Assign black or gray to -1
    color_dict[-1] = 'rgb(80,80,80)'  # or 'black'
    point_colors = [color_dict[label] for label in cluster_labels_noised]
    traj_colors = [color_dict.get(label, 'rgb(150,150,150)') for label in traj_cluster_pred]
    return point_colors, traj_colors
