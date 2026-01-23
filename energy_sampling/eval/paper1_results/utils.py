import gc
import os
import re
from collections import Counter
from collections import deque
from time import sleep
from typing import Optional

import hdbscan
import numpy as np
import pandas as pd
import plotly.colors as pc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from _plotly_utils.colors import qualitative, hex_to_rgb
from matplotlib import cm, colors
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pynndescent.distances import spearmanr
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.neighbors import NearestNeighbors
from torch_scatter import scatter
from tqdm import tqdm
from umap import UMAP

from energy_sampling.energies.molecular_crystal import density_penalty
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state, is_cuda_oom
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.common.geometry_utils import crystal_parameter_distmat
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

METRIC_REGISTRY = {}

# 2. Define the decorator
def register_metric(func):
    METRIC_REGISTRY[func.__name__] = func
    return func


def cluster_1d(X):
    X_ = X.reshape(-1, 1)
    bandwidth = estimate_bandwidth(X_, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X_)
    n_clusters = len(np.unique(labels))
    return n_clusters, labels


def plot_marginals(latents, labels, clusters_per_dim):
    fig = make_subplots(rows=4, cols=3)
    colors = qualitative.Dark24  # todo these heights are still messed up
    for ind in range(12):
        row = ind // 3 + 1
        col = ind % 3 + 1
        for c_ind in range(clusters_per_dim[ind]):
            good_inds = labels[:, ind] == c_ind
            fig.add_histogram(x=latents[good_inds, ind], nbinsx=50,
                              row=row, col=col,
                              marker_color=colors[c_ind], showlegend=False)
    fig.update_layout(barmode='overlay')
    fig.show()


def latent_dendro_fig(X, E, metric='cosine'):
    n = len(X)
    assert len(E) == n, "E must match number of samples in X"

    # --- hierarchical linkage on latent space ---
    Z = linkage(pdist(X, metric=metric), method="average")
    root = to_tree(Z)
    leaf_order = leaves_list(Z)

    # --- x positions for leaves ---
    x_pos = np.linspace(-1, 1, n)
    x_lookup = {leaf_id: x for leaf_id, x in zip(leaf_order, x_pos)}

    # --- colormap (optional) ---
    cmap = cm.get_cmap("viridis")
    E_norm = (E - E.min()) / (E.max() - E.min() + 1e-9)

    # --- recursively collect line segments ---
    def collect_segments(node):
        if node.is_leaf():
            x = x_lookup[node.id]
            y = E[node.id]
            e_color = E_norm[node.id]
            return [(x, y)], [], e_color

        left_pts, left_segs, e_left = collect_segments(node.left)
        right_pts, right_segs, e_right = collect_segments(node.right)
        left_x, left_y = left_pts[0]
        right_x, right_y = right_pts[0]

        # merge slightly above the highest child (for funnel shape)
        left_desc = [leaf.id for leaf in node.left.pre_order(lambda x: x)]
        right_desc = [leaf.id for leaf in node.right.pre_order(lambda x: x)]
        merge_y = max(E[left_desc + right_desc]) + 0.02 * (E.max() - E.min())

        e_branch = 0.5 * (e_left + e_right)
        segs = [
            ((left_x, left_y), (left_x, merge_y), e_branch),
            ((right_x, right_y), (right_x, merge_y), e_branch),
            ((left_x, merge_y), (right_x, merge_y), e_branch),
        ]
        x_mid = np.mean([left_x, right_x])
        return [(x_mid, merge_y)], left_segs + right_segs + segs, e_branch

    _, segments, _ = collect_segments(root)

    # --- build plotly figure ---
    fig = go.Figure()
    for (x0, y0), (x1, y1), e_color in segments:
        c = cmap(e_color)
        color = f"rgba({int(255 * c[0])},{int(255 * c[1])},{int(255 * c[2])},0.9)"
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=1.2),
            hoverinfo="skip", showlegend=False
        ))

    fig.update_layout(
        title="Latent-Space Funnel Dendrogram",
        template="plotly_white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Energy", autorange=True),
    )
    fig.show()


def cluster_dendro_fig(top_df):
    X = np.array(top_df["clusters"].tolist())
    E = np.array(top_df["mean_en"].tolist())
    P = np.array(top_df["coupling"].tolist())

    logP = -np.log(P + 1e-12)
    logP_norm = (logP - logP.min()) / (logP.max() - logP.min() + 1e-9)

    # --- hierarchical linkage ---
    Z = linkage(pdist(X, metric="hamming"), method="average")
    n = len(X)
    root = to_tree(Z)
    leaf_order = leaves_list(Z)

    # --- x positions for leaves ---
    x_pos = np.linspace(-1, 1, n)
    x_lookup = {leaf_id: x for leaf_id, x in zip(leaf_order, x_pos)}

    # --- colormap and line thickness ---
    cmap = cm.get_cmap("viridis")
    norm = colors.Normalize(vmin=logP.min(), vmax=logP.max())
    min_w, max_w = 0.5, 1.5  # min and max line thickness

    # --- recursively collect segments ---
    def collect_segments(node):
        if node.is_leaf():
            x = x_lookup[node.id]
            y = E[node.id]
            lp = logP[node.id]
            return [(x, y)], [], lp

        left_pts, left_segments, lp_left = collect_segments(node.left)
        right_pts, right_segments, lp_right = collect_segments(node.right)
        left_x, left_y = left_pts[0]
        right_x, right_y = right_pts[0]

        # funnel shape: merge slightly ABOVE highest child
        left_desc = [leaf.id for leaf in node.left.pre_order(lambda x: x)]
        right_desc = [leaf.id for leaf in node.right.pre_order(lambda x: x)]
        merge_y = max(E[left_desc + right_desc]) + 0.02 * (E.max() - E.min())

        lp_branch = 0.5 * (lp_left + lp_right)
        segs = [
            ((left_x, left_y), (left_x, merge_y), lp_branch),
            ((right_x, right_y), (right_x, merge_y), lp_branch),
            ((left_x, merge_y), (right_x, merge_y), lp_branch),
        ]
        x_mid = np.mean([left_x, right_x])
        return [(x_mid, merge_y)], left_segments + right_segments + segs, lp_branch

    _, segments, _ = collect_segments(root)

    # --- figure ---
    fig = go.Figure()

    for (x0, y0), (x1, y1), lp in segments:
        # w = min_w + (max_w - min_w) * norm(lp)
        lp_scaled = norm(lp) ** 3  # try 2–5 for stronger variation
        w = min_w + (max_w - min_w) * lp_scaled
        col = "rgba" + str(tuple(int(255 * c) for c in cmap(norm(lp))[:3]) + (0.9,))
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(  # color=col,
                color='grey',
                width=w
            ),
            hoverinfo="skip",
            showlegend=False,
        ))
    #
    # # --- add colorbar manually ---
    # colorbar_trace = go.Scatter(
    #     x=[None],
    #     y=[None],
    #     mode="markers",
    #     marker=dict(
    #         colorscale="Viridis",
    #         cmin=logP.min(),
    #         cmax=logP.max(),
    #         colorbar=dict(title="-log P", thickness=15, x=1.02),
    #         showscale=True
    #     ),
    #     hoverinfo="none",
    #     showlegend=False,
    # )
    # fig.add_trace(colorbar_trace)

    # --- layout ---
    fig.update_layout(
        title="Energy-Based Disconnectivity Graph (Funnel View)",
        template="plotly_white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(title="Energy", autorange=True),
    )

    fig.show()


def marginal_cluster_1d(samples):
    marginal_labels = np.zeros((len(samples), samples.shape[1]), dtype=np.int64)
    cluster_den = []
    for ind in range(marginal_labels.shape[-1]):
        _, marginal_labels[:, ind] = cluster_1d(samples[:, ind])
        _, num = np.unique(marginal_labels[:, ind], return_counts=True)
        cluster_den.append(num / len(marginal_labels))

    return marginal_labels


def coupling_ratio(row, marginal_ps):
    p_fact = np.prod([marginal_ps[i][c] for i, c in zip(row.dims, row.clusters)])
    return row.p / p_fact


def correlate_mask(labels, dims, clusters):
    """
    Return a boolean mask of samples matching a given correlate.

    Parameters
    ----------
    labels : (n_samples, d) int array
        Cluster labels for all samples.
    dims : iterable of int
        Dimension indices defining the correlate.
    clusters : iterable of int
        Cluster IDs corresponding to those dimensions.

    Returns
    -------
    mask : (n_samples,) bool array
    """
    mask = np.ones(labels.shape[0], dtype=bool)
    for i, c in zip(dims, clusters):
        mask &= (labels[:, i] == c)
    return mask


def top_joint_correlates(labels, k):
    n, d = labels.shape

    # Convert each row to a tuple to make it hashable
    tuples = [tuple(row) for row in labels]
    uniq, counts = np.unique(tuples, axis=0, return_counts=True)
    p = counts / counts.sum()
    df = pd.DataFrame({
        "dims": [tuple(range(d))] * len(uniq),
        "clusters": [tuple(u) for u in uniq],
        "n": counts,
        "p": p
    }).sort_values("p", ascending=False).reset_index(drop=True)

    return df.head(k)


def hierarchical_joint_df(labels, max_order=4, cutoff=0.01, descriptors_fn=None):
    """
    Discover high-probability joint cluster combinations up to max_order.

    Parameters
    ----------
    labels : (n_samples, d) int array
        Cluster labels per dimension.
    max_order : int
        Maximum correlation order (e.g., 4 → up to 4-body states).
    cutoff : float
        Minimum joint probability to keep (relative to total samples).
    descriptors_fn : callable, optional
        Function (indices, values, mask) -> dict of descriptors (e.g. energies).

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['order','dims','clusters','p','n','parent_key','descriptors']
    """
    n, d = labels.shape
    df_rows = []

    # helper to create unique keys
    def key_for(dims, clusts):
        return ",".join(map(str, dims)) + ":" + ",".join(map(str, clusts))

    # --- 1-body marginals ---
    for i in range(d):
        vals, counts = np.unique(labels[:, i], return_counts=True)
        for v, c in zip(vals, counts):
            p = c / n
            if p < cutoff:
                continue
            desc = descriptors_fn([i], [v], labels[:, i] == v) if descriptors_fn else {}
            df_rows.append({
                "order": 1,
                "dims": (i,),
                "clusters": (v,),
                "p": p,
                "n": c,
                "parent_key": None,
                "descriptors": desc
            })

    # --- iterative higher orders ---
    prev_level = [(row["dims"], row["clusters"]) for row in df_rows if row["order"] == 1]
    for order in range(2, max_order + 1):
        next_level = []
        seen = set()
        for dims_prev, clusts_prev in prev_level:
            used = set(dims_prev)
            mask = np.ones(n, dtype=bool)
            for i, v in zip(dims_prev, clusts_prev):
                mask &= (labels[:, i] == v)

            for j in range(d):
                if j in used:
                    continue
                for v in np.unique(labels[:, j]):
                    mask2 = mask & (labels[:, j] == v)
                    c = mask2.sum()
                    if c == 0:
                        continue
                    p = c / n
                    if p < cutoff:
                        continue
                    dims = tuple(sorted(list(dims_prev) + [j]))
                    clusts = tuple([clusts_prev[dims_prev.index(k)] if k in dims_prev else (v if k == j else None)
                                    for k in dims])
                    key = key_for(dims, clusts)
                    if key in seen:
                        continue
                    seen.add(key)
                    parent_key = key_for(dims_prev, clusts_prev)
                    # Look up parent's probability if it exists
                    parent_p = next((r["p"] for r in df_rows if key_for(r["dims"], r["clusters"]) == parent_key), None)
                    desc = descriptors_fn(dims, clusts, mask2) if descriptors_fn else {}
                    df_rows.append({
                        "order": order,
                        "dims": dims,
                        "clusters": clusts,
                        "p": p,
                        "n": c,
                        "parent_key": parent_key,
                        "parent_p": parent_p,
                        "descriptors": desc
                    })
                    next_level.append((dims, clusts))
        prev_level = next_level

    df = pd.DataFrame(df_rows)
    df["parent_p"] = df["parent_p"].fillna(df["p"])
    df["local_strength"] = df["p"] / df["parent_p"].replace(0, np.nan)
    df["id"] = [
        ",".join(map(str, row["dims"])) + ":" + ",".join(map(str, row["clusters"]))
        for _, row in df.iterrows()
    ]
    parent_map = {row["id"]: row["id"] for _, row in df.iterrows()}
    df["parent_id"] = ""
    for i, row in df.iterrows():
        parent_key = row["parent_key"]
        if parent_key in parent_map:
            df.at[i, "parent_id"] = parent_key

    # Human-readable labels
    df["label"] = [
        "{" + ", ".join(f"{d}:{c}" for d, c in zip(row.dims, row.clusters)) + "}"
        for _, row in df.iterrows()
    ]
    return df.sort_values(["order", "p"], ascending=[True, False]).reset_index(drop=True)


def get_highp_correlations(marginal_labels, n_samples, n_dims, clusters_per_dim, cutoff: float = 2.0,
                           max_depth: int = 4):
    p1 = [
        np.bincount(marginal_labels[:, i], minlength=clusters_per_dim[i]) / n_samples
        for i in range(n_dims)
    ]
    mask_cache = {
        (i, c): (marginal_labels[:, i] == c)
        for i in range(n_dims)
        for c in range(clusters_per_dim[i])
    }
    idx_cache = {key: np.flatnonzero(mask) for key, mask in mask_cache.items()}

    trees = []
    for i in range(marginal_labels.shape[1]):
        for c_i in range(clusters_per_dim[i]):
            tree = expand_tree_fast(i, c_i, p1, idx_cache, n_samples, cutoff=cutoff, max_depth=max_depth)
            trees.append(tree)

    corr_df = collect_correlations(trees)
    return corr_df


def compute_dim_weights(corr_df, use_pjoint=True, ratio_thresh=2.0, order_min=2):
    """
    Compute per-dimension weights from high-probability correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame with columns ['dims', 'clusters', 'ratio', 'p_joint'].
    ratio_thresh : float
        Minimum ratio to consider a correlation 'high'.
    order_min : int
        Minimum order (ignore 1-body terms).
    use_pjoint : bool
        Whether to weight by p_joint * ratio (default) or ratio alone.

    Returns
    -------
    pd.Series : normalized dimension weights (sum = 1)
    """
    df = corr_df.query("order >= @order_min and ratio > @ratio_thresh")

    weights = Counter()
    for _, row in df.iterrows():
        w = row.ratio * (row.p_joint if use_pjoint else 1.0)
        for d in row.dims:
            weights[d] += w

    # normalize
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total + 1e-12

    return pd.Series(weights).sort_index()


def collect_correlations(trees):
    """
    Flatten a list of correlation trees and dedupe them into a readable DataFrame.
    """
    found = {}

    def dfs(node, root):
        key = frozenset(node["indices"])
        order = len(node["indices"])
        p_joint = node["p_joint"]
        ratio = node["ratio"]

        # insert or update
        if key not in found or ratio > found[key]["ratio"]:
            found[key] = {
                "dims": [i for i, _ in sorted(node["indices"])],
                "clusters": [c for _, c in sorted(node["indices"])],
                "order": order,
                "p_joint": p_joint,
                "ratio": ratio,
                "root_dim": root[0],
                "root_cluster": root[1],
            }

        for ch in node["children"]:
            dfs(ch, root)

    # traverse all trees
    for t in trees:
        root = t["indices"][0]
        dfs(t, root)

    # convert to DataFrame
    df = pd.DataFrame(list(found.values()))
    df.sort_values(["order", "ratio"], ascending=[True, False], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def expand_tree_fast(root_dim, root_cluster, p1, idx_cache, N,
                     cutoff=5.0, max_depth=None):
    """
    Build the full n-body correlation tree starting from a single 1-body cluster.

    Parameters
    ----------
    root_dim : int
        Dimension index of the root cluster.
    root_cluster : int
        Cluster index within that dimension.
    p1 : list of np.ndarray
        Precomputed 1D marginal probabilities per dimension.
    idx_cache : dict[(int,int)] -> np.ndarray
        Cached indices (sample positions) for each (dimension, cluster) pair.
    N : int
        Total number of samples.
    cutoff : float
        Coupling ratio threshold for retaining a branch.
    max_depth : int or None
        Optional limit on how deep to expand.

    Returns
    -------
    dict : correlation tree node
    """
    root_idx = idx_cache[(root_dim, root_cluster)]
    p_root = len(root_idx) / N

    node = {
        "indices": [(root_dim, root_cluster)],
        "p_joint": p_root,
        "ratio": 1.0,
        "children": []
    }

    def recurse(indices, idx_set, prod_base, depth):
        if max_depth and depth >= max_depth:
            return []

        used_dims = [i for i, _ in indices]
        children = []

        for j in range(len(p1)):
            if j in used_dims:
                continue

            for c_j in range(len(p1[j])):
                # intersection of index sets
                new_idx = np.intersect1d(idx_set, idx_cache[(j, c_j)], assume_unique=True)
                if new_idx.size == 0:
                    continue

                p_joint = new_idx.size / N
                ratio = p_joint / (prod_base * p1[j][c_j] + 1e-12)

                if ratio > cutoff:
                    child = {
                        "indices": indices + [(j, c_j)],
                        "p_joint": p_joint,
                        "ratio": ratio,
                        "children": []
                    }
                    # recursively expand
                    child["children"] = recurse(
                        child["indices"], new_idx, prod_base * p1[j][c_j], depth + 1
                    )
                    children.append(child)

        return children

    node["children"] = recurse(
        [(root_dim, root_cluster)], root_idx, p1[root_dim][root_cluster], 1
    )
    return node


def estimate_logp_with_convergence(
        gfn_model, terminal_states,
        n_steps, tol=1e-3, window=5, max_repeats=200
):
    flows = []
    logp_history = []

    repeat = 0
    while True:
        repeat += 1
        # --- one backward trajectory sample ---
        discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)
        condition = torch.zeros((len(terminal_states), 1), device=gfn_model.device)

        with torch.no_grad():
            states, log_pfs, log_pbs, log_flow = gfn_model.get_traj_bwd(
                terminal_states.clone().to(gfn_model.device),
                discretizer, condition, return_gauss_params=False
            )
            delta = (log_pfs.sum(-1) - log_pbs.sum(-1)).cpu().detach()
            flows.append(delta)

        # --- recompute running log p_F(x) ---
        deltas = torch.stack(flows, dim=1)  # [n_states, n_repeats]
        logp_est = torch.logsumexp(deltas, dim=1) - np.log(len(flows))
        logp_history.append(logp_est)

        # --- convergence check every iteration after window ---
        if len(logp_history) > window:
            recent = torch.stack(logp_history[-window:], dim=0)
            diffs = (recent[-1] - recent[0]).abs().mean().item()
            if diffs < tol or len(flows) >= max_repeats:
                print(f"Converged after {len(flows)} repeats (Δ={diffs:.3e})")
                break

    return logp_est, torch.stack(logp_history, dim=0)


def get_sample_batch(batch_size, max_z_prime, device, n_steps, gfn_model):
    init_state = get_gfn_init_state(batch_size, 6 + 6 * max_z_prime, device)
    discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)

    condition = torch.zeros((batch_size, 1))  # unconditional sampling
    with torch.no_grad():
        (states, log_pfs, log_pbs, log_flow) = gfn_model.get_traj_fwd(init_state,
                                                                      discretizer,
                                                                      None,
                                                                      condition,
                                                                      return_gauss_params=False)
    return states[:, -1, :], log_pfs.sum(-1), log_pbs.sum(-1)


def sample_from_gfn(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model):
    samples = []
    pfs = []
    pbs = []
    counter = 0
    with tqdm(total=num_samples) as pbar:
        while (len(samples) * batch_size) < num_samples:
            states, log_pfs, log_pbs = get_sample_batch(batch_size, max_z_prime, device, n_steps, gfn_model)

            samples.append(states.cpu().detach())
            pfs.append(log_pfs.cpu().detach())
            pbs.append(log_pbs.cpu().detach())

            counter += batch_size
            pbar.update(batch_size)

    samples = torch.cat(samples)
    pfs = torch.cat(pfs)
    pbs = torch.cat(pbs)
    return samples.clip(max=1, min=-1), pfs, pbs


def analyze_samples(x, mol_list, max_z_prime, device, batch_size, sg_ind, zp, do_uma: bool = False, predictor=None, overwrite_latents: bool = True):
    num_samples = len(mol_list)
    samples = []
    cursor = 0
    already_oomed = False
    with tqdm(total=num_samples) as pbar:
        with torch.no_grad():
            while cursor < num_samples:
                try:
                    inds = np.arange(cursor, min(num_samples, cursor + batch_size))
                    for elem in mol_list:
                        elem.z_prime = zp
                    batch = collate_data_list([mol_list[ind] for ind in inds], max_z_prime=max_z_prime)
                    if overwrite_latents:
                        batch.reset_sg_info(sg_ind)
                        batch.latent_to_cell_params(x[inds])
                    batch = batch.to(device)
                    outs = batch.analyze(['lj', 'qlj', 'elj', 'silu', 'rdf'], cutoff=10, std_orientation=True)
                    if do_uma:
                        gas_en = batch.compute_lattice_gas_phase_uma(predictor,
                                                                     std_orientation=True).cpu().detach() * 96.485
                        cry_en = batch.compute_crystal_uma(predictor=predictor,
                                                           std_orientation=True).cpu().detach() * 96.485
                        batch.add_graph_attr(gas_en, 'uma_gas_pot')
                        batch.add_graph_attr(cry_en, 'uma_pot')

                    for key, value in outs.items():
                        if key != 'rdf':
                            batch.add_graph_attr(value, key)
                        else:
                            batch.add_graph_attr(value[0], key)

                    batch.to('cpu')
                    samples.extend(batch.batch_to_list())
                    del batch
                    cursor += batch_size
                    if (batch_size <= 10000) and (batch_size < num_samples) and not already_oomed:
                        batch_size += max(int(batch_size * 0.01), 1)
                    pbar.update(batch_size)
                except (RuntimeError, ValueError) as e:
                    if is_cuda_oom(e):
                        if batch_size == 1:
                            assert False, "Cascading OOM failure in molecule energy evaluation"
                        batch_size = max(int(batch_size * 0.65), 1)
                        print(f"OOM in energy evaluation: dropping batch size to {batch_size}")
                        gc.collect()
                        # del self.uma_predictor, mol_batch_i
                        torch.cuda.empty_cache()
                        # torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        # self.uma_predictor = init_uma_crystal_predictor(self.uma_path, device=self.device)
                        already_oomed = True
                        sleep(0.1)
                    else:
                        raise e

    return samples


def cluster_hdbscan_to_df(X: torch.Tensor,
                          lj_pot: torch.Tensor,
                          min_cluster_size: int = 10):
    """
    Run HDBSCAN on samples X [n, d] and summarize results in a DataFrame.

    Parameters
    ----------
    X : torch.Tensor [n, d]
        Sample coordinates.
    lj_pot : torch.Tensor [n]
        Per-sample energies.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.

    Returns
    -------
    pd.DataFrame with columns ['cluster_id', 'n', 'p', 'mean_en'].
    """

    # --- run clustering on CPU numpy array ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X.cpu().numpy())

    # cluster probabilities (membership strengths)

    # --- collect cluster masks and statistics ---
    df_records = []
    unique_labels = np.unique(labels)
    for cid in unique_labels:
        mask = labels == cid
        n_points = int(mask.sum())
        p_mean = n_points / len(X)

        if cid == -1:  # noise cluster
            mean_en = float(torch.tensor(log_rescale_positive(lj_pot[mask])).mean().cpu())
        else:
            mean_en = float(log_rescale_positive(lj_pot[mask]).mean().cpu())

        df_records.append(dict(cluster_id=cid, n=n_points, p=p_mean, mean_en=mean_en))

    top_df = pd.DataFrame(df_records, columns=['cluster_id', 'n', 'p', 'mean_en'])
    return top_df, labels


def graph_MC(energies, traj_len, kT, cval, samples=None, dmat=None):
    N = len(energies)
    beta_mc = 1 / kT
    k = int(samples.shape[1] * cval * np.log(len(samples)))

    if dmat is not None and samples is None:
        knn = dmat.topk(k + 1, largest=False).indices[:, 1:]
    elif samples is not None:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(samples.cpu().numpy())
        dists, inds = nbrs.kneighbors(samples.cpu().numpy())
        knn = torch.tensor(inds[:, 1:], device=samples.device)

    k = knn.shape[1]

    # walkers
    current = torch.arange(N, device=energies.device)
    best = current.clone()
    best_energy = energies[current]

    for _ in range(traj_len):
        # pick random neighbor for each walker
        rand_idx = torch.randint(k, (N,), device=energies.device)
        proposal = knn[current, rand_idx]

        dE = energies[proposal] - energies[current]

        accept = (dE < 0) | (torch.rand(N, device=energies.device) < torch.exp(-beta_mc * dE))
        current = torch.where(accept, proposal, current)

        # update best seen
        better = energies[current] < best_energy
        best = torch.where(better, current, best)
        best_energy = torch.where(better, energies[current], best_energy)

    return best


def first_hit_graph_MC(energies, basin_inds, traj_len, kT, cval, samples, dmat):
    N = len(energies)
    beta_mc = 1 / kT
    k = int(samples.shape[1] * cval * np.log(len(samples)))
    knn = dmat.topk(k + 1, largest=False).indices[:, 1:]

    k = knn.shape[1]

    state_is_basin = torch.zeros(len(energies), dtype=bool)
    state_is_basin.fill_(False)
    state_is_basin[basin_inds] = True

    # walkers
    current = torch.arange(N, device=energies.device)
    active = torch.ones(len(energies), dtype=bool)

    for _ in range(traj_len):
        if active.any() == False:
            break
        # pick random neighbor for each walker
        rand_idx = torch.randint(k, (N,), device=energies.device)
        proposal = knn[current, rand_idx]

        dE = energies[proposal] - energies[current]

        accept = (dE < 0) | (torch.rand(N, device=energies.device) < torch.exp(-beta_mc * dE))
        current = torch.where(accept & active, proposal, current)

        # update best seen
        new_hit = state_is_basin[current] & active
        active[new_hit] = False

    first_basin_hit = current
    first_basin_hit[active] = -1

    return first_basin_hit


def steepest_descent(samples, energies, cval, dmat):
    k = int(samples.shape[1] * cval * np.log(len(samples)))
    knn = dmat.topk(min(len(dmat), k + 1), largest=False).indices[:, 1:]
    minima = torch.empty(len(energies), dtype=torch.long)
    for i in range(len(energies)):
        minima[i] = graph_descent(i, knn, energies)

    return minima


def graph_descent(i, knn, energies):
    while True:
        neigh = knn[i]
        j = neigh[torch.argmin(energies[neigh])]
        if energies[j] < energies[i]:
            i = j
        else:
            return i


def steepest_descent_parallel(knn, energies):
    N = energies.shape[0]
    current = torch.arange(N, device=energies.device)
    active = torch.ones(N, dtype=torch.bool, device=energies.device)

    while active.any():
        neigh = knn[current]  # (N, k)
        neigh_E = energies[neigh]  # (N, k)
        best_idx = neigh_E.argmin(dim=1)  # (N,)
        j = neigh[torch.arange(N), best_idx]  # (N,)

        better = energies[j] < energies[current]
        move = active & better

        current = torch.where(move, j, current)
        active = move

    return current


def adaptive_steepest_descent_parallel(dmat, energies, cval_init, dim):
    N = energies.shape[0]
    current = torch.arange(N, device=energies.device)
    active = torch.ones(N, dtype=torch.bool, device=energies.device)

    minima_record = []

    cval = cval_init
    while len(current.unique()) > 1:
        k = int(dim * cval * np.log(len(energies)))
        knn = dmat.topk(min(len(dmat), k + 1), largest=False).indices[:, 1:]

        while active.any():
            neigh = knn[current]  # (N, k)
            neigh_E = energies[neigh]  # (N, k)
            best_idx = neigh_E.argmin(dim=1)  # (N,)
            j = neigh[torch.arange(N), best_idx]  # (N,)

            better = energies[j] < energies[current]
            move = active & better

            if not move.any():
                break

            current = torch.where(move, j, current)
            active = move

        minima_record.append(current)
        active = torch.ones(N, dtype=torch.bool, device=energies.device)
        cval *= 1.1

    return current


def basin_average(observable, basin_weights):
    w = basin_weights
    num = (w * observable.unsqueeze(1)).sum(0)
    den = w.sum(0).clamp(min=1e-12)
    return num / den


def get_committor_weights(clabel, basin_inds, num_committor_steps: int, sample_energy, num_samples, num_basins,
                          sample_latents, cval: float, mc_kT: float):
    basin_id = torch.full_like(clabel, -1)
    basin_id[basin_inds] = torch.arange(len(basin_inds), device=clabel.device)
    committed_record = torch.zeros((num_committor_steps, num_samples), dtype=torch.long)
    dmat = crystal_parameter_distmat(sample_latents)
    for ind in tqdm(range(num_committor_steps)):
        hits = first_hit_graph_MC(
            sample_energy,
            basin_inds,
            traj_len=5000,
            kT=mc_kT,
            cval=cval,
            samples=sample_latents,
            dmat=dmat)
        committed_record[ind] = torch.where(
            hits >= 0,
            basin_id[hits],
            -1
        )
    basin_weights = torch.zeros(num_samples, num_basins)

    flat = committed_record.view(-1)
    mask = flat >= 0

    basin_weights.index_add_(
        0,
        torch.arange(num_samples).repeat(num_committor_steps)[mask],
        torch.nn.functional.one_hot(flat[mask], num_basins).float()
    )

    basin_weights /= num_committor_steps
    return basin_weights


@torch.no_grad()
def get_gfn_samples(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model, energy_function, molecule, sg_ind,
                    zp):
    sample_latents, pfs, pbs = sample_from_gfn(num_samples, max_z_prime, device, n_steps, 1000, gfn_model)

    if energy_function == 'uma':
        pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
        predictor = init_uma_crystal_predictor(pred_path, device=device)
    else:
        predictor = None

    if isinstance(molecule, list):
        mol_list = molecule
    else:
        mol_list = [molecule]
    samples = analyze_samples(sample_latents, mol_list * num_samples, max_z_prime, device, batch_size, sg_ind,
                              zp,
                              do_uma=energy_function == 'uma', predictor=predictor)
    sample_batch = collate_data_list(samples, max_z_prime=max_z_prime)

    if energy_function == 'uma':
        sample_energy = sample_batch.uma_pot / (sample_batch.sym_mult * sample_batch.z_prime) - sample_batch.uma_gas_pot
    elif energy_function == 'elj':
        lj_mean, lj_std, uma_mean, uma_std = [-20.6, 5.7, -3.4, 1.5]
        atomwise_energy = sample_batch.elj / (sample_batch.num_atoms / sample_batch.z_prime)
        atomwise_fixed = (atomwise_energy - lj_mean) / lj_std * uma_std + uma_mean
        sample_energy = atomwise_fixed * (sample_batch.num_atoms / sample_batch.z_prime)
    else:
        sample_energy = sample_batch.lj

    return sample_batch, sample_latents, sample_energy.float(), sample_batch.packing_coeff, samples, pfs, pbs


#
# def get_cluster_weights(sample_latents, sample_energy, num_committor_steps,
#                         cval: float = 1,
#                         mc_cval: float = 0.5,
#                         mc_kT: float = 2.5):
#     "Soft clustering by committor analysis"
#     num_samples = len(sample_latents)
#     clabel = steepest_descent(sample_latents, sample_energy, cval=cval)
#     basin_inds = torch.unique(clabel)
#     num_basins = len(basin_inds)
#     basin_weights = get_committor_weights(clabel, basin_inds, num_committor_steps, sample_energy, num_samples,
#                                           num_basins, sample_latents, cval=mc_cval, mc_kT=mc_kT)
#     hard_assignment = torch.argmax(basin_weights, dim=1)
#     hard_assignment_prob = torch.amax(basin_weights, dim=1)
#     return basin_weights, hard_assignment, hard_assignment_prob, basin_inds

# import plotly.graph_objects as go
# masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
# mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
# sorted_masks = masks[mask_sorts]
# go.Figure(go.Histogram(x=basin_weights.amax(1), nbinsx=100)).show()
# go.Figure(go.Histogram(x=hard_assignment, nbinsx=len(hard_assignment.unique()))).show()
# sample_batch.plot_batch_cell_params(space='real',
#                                     aux_dists=[sample_batch.full_cell_parameters()[m] for m in
#                                                masks[mask_sorts[:10]] if
#                                                sum(m) > 1])


def cluster_thermo_analysis(basin_weights, sample_energy, kT, cp, cluster_labels, top_cluster_inds):
    min_ens = torch.tensor([sample_energy[cluster_labels == lab].amin() for lab in cluster_labels.unique()])[
        top_cluster_inds]

    # Basin partition weights
    Zb = basin_weights.sum(dim=0).clamp(min=1e-12)  # (B,)

    # Relative free energies
    Fb = -kT * torch.log(Zb / Zb.sum())  # (B,)
    basin_probs = Zb / Zb.sum()

    # Mean energies per basin
    mean_E = (basin_weights * sample_energy[:, None]).sum(dim=0) / Zb

    # Mean densities per basin
    mean_rho = (basin_weights * cp[:, None]).sum(dim=0) / Zb

    # Effective entropies (relative)
    Sb = (mean_E - Fb) / kT
    return min_ens, Zb, Fb, basin_probs, mean_rho, Sb, mean_E


def to_rgba(color, alpha=0.7):
    # already rgba → just replace alpha
    if color.startswith("rgba"):
        return re.sub(
            r'rgba\(([^,]+),([^,]+),([^,]+),[^)]+\)',
            rf'rgba(\1,\2,\3,{alpha})',
            color
        )

    # rgb(...)
    if color.startswith("rgb"):
        nums = re.findall(r'\d+', color)
        r, g, b = nums[:3]
        return f"rgba({r},{g},{b},{alpha})"

    # must be a hex
    r, g, b = hex_to_rgb(color)
    return f"rgba({r},{g},{b},{alpha})"


def get_color_set(n, alpha=0.7):
    if n <= len(qualitative.Plotly):
        colors = qualitative.Plotly[:n]
    elif n <= len(qualitative.Dark24):
        colors = qualitative.Dark24[:n]
    else:
        colors = pc.n_colors(
            'rgb(0,0,255)',
            'rgb(255,0,0)',
            n,
            colortype='rgb'
        )

    return [to_rgba(c, alpha) for c in colors]


def get_gfn_logprobs(batch_size, sample_latents, gfn_model, n_steps, max_repeats: int = 50, tol: float = 1e-2):
    num_batches = len(sample_latents) // batch_size + (1 if len(sample_latents) % batch_size else 0)
    num_samples = len(sample_latents)
    counter = 0
    logps = torch.zeros(len(sample_latents), dtype=torch.float32)

    with tqdm(total=num_samples) as pbar:
        with torch.no_grad():
            for b_ind in range(num_batches):
                inds = torch.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                terminal_states = sample_latents[inds, :]
                logp_est, _ = estimate_logp_with_convergence(
                    gfn_model, terminal_states, n_steps=n_steps, max_repeats=max_repeats, tol=tol, window=10
                )
                logps[inds] = logp_est
                counter += batch_size
                pbar.update(batch_size)

    return logps


def umap_hdbscan_clustering(dmat, sample_energy, n_components, n_neighbors, min_dist, min_cluster_size, min_samples,
                            kT):
    umap_model = UMAP(n_components=n_components,
                      n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      metric='precomputed')
    sample_embedding = umap_model.fit_transform(dmat.numpy().astype(np.float64))

    clusterer = hdbscan.HDBSCAN(
        metric='euclidean',  # use your distance matrix
        min_cluster_size=min_cluster_size,
        cluster_selection_method='eom',  # or 'leaf'
        min_samples=min_samples,
        # prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(sample_embedding)
    num_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    cluster_labels[cluster_labels == -1] = np.random.randint(low=0, high=num_clusters,
                                                             size=(cluster_labels == -1).sum())
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)

    "get soft assignments"
    k_sigma = 20  # smaller than p_k
    r_k = dmat.topk(k_sigma + 1, largest=False).values[:, -1]
    sigma = torch.median(r_k)

    p_k = min(100, len(dmat) - 1)
    p_knn = dmat.topk(p_k, largest=False).indices
    neighbor_clusters = cluster_labels[p_knn]
    dists = dmat[torch.arange(len(dmat))[:, None], p_knn]
    dist_weights = torch.exp(-dists / sigma)  # (N, p_k)

    basin_weights = scatter(
        dist_weights,
        neighbor_clusters,
        dim=1,
        dim_size=num_clusters,
        reduce='sum'
    )
    basin_weights /= basin_weights.sum(dim=1, keepdim=True)
    cluster_prob = torch.amax(basin_weights, dim=1)

    num_clusters = len(cluster_labels.unique())

    cluster_rep_inds = torch.empty(num_clusters, dtype=torch.long, device=cluster_labels.device)

    for k in range(num_clusters):
        mask = (cluster_labels == k)  # & (cluster_prob > 0.95)
        inds = torch.where(mask)[0]

        # pairwise distances within basin
        D = dmat[inds][:, inds]

        # Boltzmann weights (numerically stable)
        E = sample_energy[inds]
        w = torch.exp(-(E - E.min()) / kT)
        w = w / w.sum()

        # energy-weighted medoid
        score = (D * w[None, :]).sum(dim=1)
        cluster_rep_inds[k] = inds[score.argmin()]

    return basin_weights, cluster_labels, cluster_prob, cluster_rep_inds


def get_cluster_weights2(sample_latents, sample_energy, num_committor_steps,
                         mc_kT: float = 2.5):
    "Soft clustering by committor analysis"
    with torch.no_grad():
        energies = sample_energy.cuda()
        dmat = crystal_parameter_distmat(sample_latents).fill_diagonal_(0)
        dmat = dmat.cuda()

        traj_record = []
        cval_init = 0.1
        dim = 12
        N = energies.shape[0]

        k0 = int(dim * cval_init * np.log(N))
        knn0 = dmat.topk(min(N, k0 + 1), largest=False).indices[:, 1:]

        neigh_E = energies[knn0]  # (N, k)
        E0 = energies[:, None]  # (N, 1)
        dE = neigh_E - E0  # (N, k)
        weights = (-dE / mc_kT).softmax(dim=1)

        for c_step in tqdm(range(num_committor_steps)):
            """need a preliminary MC step"""
            choice = torch.multinomial(weights, num_samples=1).squeeze(1)
            # select nearby point to init traj
            j = knn0[torch.arange(N, device=energies.device), choice]
            current = j

            active = torch.ones(N, dtype=torch.bool, device=energies.device)

            minima_record = []
            cval_record = []
            cval = cval_init

            while len(current.unique()) > 1:
                k = int(dim * cval * np.log(len(energies)))
                knn = dmat.topk(min(len(dmat), k + 1), largest=False).indices[:, 1:]

                while active.any():
                    neigh = knn[current]  # (N, k)
                    neigh_E = energies[neigh]  # (N, k)
                    best_idx = neigh_E.argmin(dim=1)  # (N,)
                    j = neigh[torch.arange(N), best_idx]  # (N,)

                    better = energies[j] < energies[current]
                    move = active & better

                    current = torch.where(move, j, current)
                    active = move

                minima_record.append(current.clone().cpu().detach())
                active = torch.ones(N, dtype=torch.bool, device=energies.device)
                cval *= 1.1
                cval_record.append(cval)

            traj_record.append(minima_record)

    k_ind = 50
    final_assignments = torch.stack(
        [traj[k_ind] for traj in traj_record],  # (num_committor_steps, N)
        dim=0
    )  # CPU tensor is fine
    basin_inds = torch.unique(final_assignments)
    num_basins = len(basin_inds)
    basin_to_id = {int(b.item()): i for i, b in enumerate(basin_inds)}

    basin_ids = torch.empty_like(final_assignments)

    for b, bid in basin_to_id.items():
        basin_ids[final_assignments == b] = bid
    C, N = basin_ids.shape
    B = num_basins

    # counts[i, b] = number of committor runs where sample i ended in basin b
    counts = torch.zeros((N, B), dtype=torch.int32)

    for b in range(B):
        counts[:, b] = (basin_ids == b).sum(dim=0)

    basin_weights = counts / C
    hard_assignment = torch.argmax(basin_weights, dim=1)
    hard_assignment_prob = torch.amax(basin_weights, dim=1)

    return basin_weights, hard_assignment, hard_assignment_prob, basin_inds

    # import plotly.graph_objects as go
    # masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
    # mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    # sorted_masks = masks[mask_sorts]

    # sample_batch.plot_batch_cell_params(space='real',
    #                                     aux_dists=[sample_batch.full_cell_parameters()[m] for m in
    #                                                masks[mask_sorts[:10]] if
    #                                                sum(m) > 1])

    # go.Figure(go.Histogram(x=basin_weights.amax(1), nbinsx=100)).show()
    # go.Figure(go.Histogram(x=hard_assignment, nbinsx=len(hard_assignment.unique()))).show()


def bottom_up_cluster(xx, e, d_cut, e_cut, max_new_samples: int, device):
    # Sort by energy ascending
    sort_inds = torch.argsort(e.to(device))
    xx_sorted = xx.to(device)[sort_inds]
    e_sorted = e.to(device)[sort_inds]
    mask = e_sorted < e_cut

    xx_sorted_cuda = xx_sorted.to(device)
    blocked = torch.zeros(len(xx_sorted), dtype=torch.bool, device=device)
    keep = torch.zeros(len(xx_sorted), dtype=bool, device=device)
    d_cut_squared = d_cut * d_cut
    for i in range(len(xx_sorted)):
        if not mask[i]:
            break

        if blocked[i]:
            continue

        keep[i] = True
        if torch.sum(keep) == max_new_samples:
            break

        drow = ((xx_sorted_cuda - xx_sorted_cuda[i, None, :]) ** 2).sum(-1)  # faster, skips sqrt
        nearby = drow < d_cut_squared
        blocked |= nearby

    keep_inds = sort_inds[keep]

    return keep_inds.cpu()


def simple_dedupe(samples, d_cut, rdf_cut):
    "Cluster / dedupe"
    batch = collate_data_list(samples)
    latents = batch.latent_params()
    energy = batch.uma_pot / (batch.sym_mult * batch.z_prime) - batch.uma_gas_pot

    quant = torch.quantile(energy, 0.1)
    candidates = bottom_up_cluster(latents, energy, d_cut, quant, 1000000,
                                   device=latents.device)

    candidate_samples = [samples[ind] for ind in candidates]
    candidate_batch = collate_data_list(candidate_samples)
    candidate_energy = candidate_batch.uma_pot / (
            candidate_batch.sym_mult * candidate_batch.z_prime) - candidate_batch.uma_gas_pot
    out = candidate_batch.analyze(['rdf'])
    rdf_dists = compute_rdf_distmat(out['rdf'][0], out['rdf'][1]).fill_diagonal_(100)

    uniques = []
    for ind in range(candidate_batch.num_graphs):
        drow = rdf_dists[ind]
        if torch.any(drow < rdf_cut):
            match_inds = torch.argwhere(drow < rdf_cut).flatten()
            if torch.any(candidate_energy[match_inds] < candidate_energy[ind]):
                pass
            else:
                uniques.append(ind)
        else:
            uniques.append(ind)

    return [candidate_samples[ind] for ind in uniques]


def basin_min_energy(basin_id, energy):
    nb = basin_id.max().item() + 1
    Emin = torch.full((nb,), float('inf'), device=energy.device)
    Emin.scatter_reduce_(0, basin_id, energy.float(), reduce="amin")
    return Emin


def basin_barriers(basin_id, energy, knn):
    N, k = knn.shape
    i = torch.arange(N, device=energy.device)[:, None].expand(N, k)
    j = knn

    bi = basin_id[i]
    bj = basin_id[j]

    mask = bi != bj
    bi = bi[mask]
    bj = bj[mask]

    barrier = torch.maximum(energy[i][mask], energy[j][mask])

    # canonical basin pair ordering
    pair = torch.stack([
        torch.minimum(bi, bj),
        torch.maximum(bi, bj)
    ], dim=1)

    return pair, barrier


def reduce_min_barrier(pairs, barriers, num_basins):
    key = pairs[:, 0] * num_basins + pairs[:, 1]
    uniq, inv = torch.unique(key, return_inverse=True)

    min_barrier = torch.full((len(uniq),), float('inf'), device=barriers.device)
    min_barrier.scatter_reduce_(0, inv, barriers, reduce="amin")

    bi = uniq // num_basins
    bj = uniq % num_basins
    return bi, bj, min_barrier


def merge_edges_kT(bi, bj, barrier_ij, Emin, kT):
    delta = barrier_ij - torch.minimum(Emin[bi], Emin[bj])  # uphill merge rule
    # delta = barrier_ij - torch.maximum(Emin[bi], Emin[bj])  # downhill merge rule
    mask = delta < kT
    return bi[mask], bj[mask]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def merge_basins(num_basins, bi, bj):
    uf = UnionFind(num_basins)
    for i, j in zip(bi.tolist(), bj.tolist()):
        uf.union(i, j)

    rep = torch.tensor([uf.find(i) for i in range(num_basins)])
    _, new_labels = torch.unique(rep, return_inverse=True)
    return new_labels


def basin_min_energy_and_index(assignments, energy):
    """
    Returns:
        Emin  : (B,) minimum energy per basin
        idx   : (B,) index of sample achieving Emin
    """
    B = assignments.max().item() + 1

    # initialize
    Emin = torch.full((B,), float('inf'), device=energy.device)
    Emin.scatter_reduce_(0, assignments, energy, reduce="amin")

    # find indices (argmin per basin)
    idx = torch.full((B,), -1, dtype=torch.long, device=energy.device)
    for b in range(B):
        mask = assignments == b
        if mask.any():
            idx[b] = torch.argmin(energy[mask])
            idx[b] = torch.where(mask)[0][idx[b]]

    return Emin, idx


def kinetic_clustering(sample_latents, sample_energy, cval, kT):
    dmat = crystal_parameter_distmat(sample_latents)

    init_basins = steepest_descent(
        sample_latents,
        sample_energy,
        cval=cval,
        dmat=dmat
    ).long()

    init_basins_unique, init_basins_contiguous = torch.unique(init_basins, return_inverse=True)
    num_basins = len(init_basins_unique)  # Actual count

    N = dmat.shape[0]
    k = 5  # sample_latents.shape[1]
    knn = dmat.topk(min(dmat.shape[1], k + 1), largest=False).indices[:, 1:]

    Emin = basin_min_energy(init_basins_contiguous, sample_energy)
    pairs, barriers = basin_barriers(init_basins_contiguous, sample_energy, knn)
    bi, bj, barrier_ij = reduce_min_barrier(pairs, barriers, num_basins)

    merge_i, merge_j = merge_edges_kT(bi, bj, barrier_ij, Emin, kT)
    merged_basin_labels = merge_basins(num_basins, merge_i, merge_j)

    cluster_ind = merged_basin_labels[init_basins_contiguous]
    num_clusters = len(cluster_ind.unique())

    cluster_rep_inds = torch.empty(num_clusters, dtype=torch.long, device=cluster_ind.device)
    for k in range(num_clusters):
        mask = cluster_ind == k
        inds = torch.where(mask)[0]
        cluster_rep_inds[k] = inds[sample_energy[inds].argmin()]

    "estimate local fluctuations via distance weighted neighbors"
    k_sigma = 20  # smaller than p_k
    r_k = dmat.topk(k_sigma + 1, largest=False).values[:, -1]
    sigma = torch.median(r_k)

    p_k = min(100, len(dmat) - 1)
    p_knn = dmat.topk(p_k, largest=False).indices
    neighbor_clusters = cluster_ind[p_knn]
    dists = dmat[torch.arange(len(dmat))[:, None], p_knn]
    dist_weights = torch.exp(-dists / sigma)  # (N, p_k)

    basin_weights = scatter(
        dist_weights,
        neighbor_clusters,
        dim=1,
        dim_size=num_clusters,
        reduce='sum'
    )
    basin_weights /= basin_weights.sum(dim=1, keepdim=True)
    hard_cluster_probs = torch.amax(basin_weights, dim=1)

    return basin_weights, cluster_ind, hard_cluster_probs, cluster_rep_inds


'''

    import plotly.graph_objects as go
    masks = np.array([cluster_ind == ind for ind in np.unique(cluster_ind)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    sorted_masks = masks[mask_sorts]
    go.Figure(go.Histogram(x=basin_weights.amax(1), nbinsx=100)).show()
    go.Figure(go.Histogram(x=cluster_ind, nbinsx=len(cluster_ind.unique()))).show()
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in
                                                   masks[mask_sorts[:10]] if
                                                   sum(m) > 1])

'''


def compute_zp_order_penalty(bounding_energy, crystal_batch):
    # penalize the model for placing asymmetric units out of the canonical order (closest -> furthest from origin)
    per_aunit_centroids = crystal_batch.aunit_centroid.reshape(crystal_batch.num_graphs,
                                                               crystal_batch.max_z_prime, 3)
    idx = torch.arange(crystal_batch.max_z_prime, device=crystal_batch.device)[None, ...]
    mask = (idx >= (crystal_batch.z_prime[..., None]))[..., None].expand(-1, -1, 3)
    per_aunit_centroids[mask] = 1  # this will put lower Z' options always at the end
    origin_dists = per_aunit_centroids.norm(dim=2)
    overlaps = -origin_dists.diff(dim=1)
    zp_ordering_energy = F.relu(overlaps).mean(dim=-1) ** 2
    bounding_energy = bounding_energy + zp_ordering_energy
    return bounding_energy


def generator_reward(crystal_batch, raw_latents, max_z_prime,
                     energy_function, temperature,
                     energy_clip, lj_coeff=1, bounding_coeff=10, reduction_coeff=10, density_coeff=10):
    ens_dict = {}

    latents = crystal_batch.latent_params()
    if raw_latents is not None:
        bounding_energy = (F.relu(raw_latents - 1) ** 2 + F.relu(-(raw_latents + 1)) ** 2).sum(
            dim=-1)  # discourage exploration beyond clip range
    else:
        bounding_energy = torch.zeros_like(latents[:, 0])

    if max_z_prime > 1:
        bounding_energy = compute_zp_order_penalty(bounding_energy, crystal_batch)

    if energy_function in ['lj', 'qlj', 'elj', 'silu', 'uma']:
        density_energy = density_penalty(crystal_batch.packing_coeff)
        if energy_function == 'lj':
            mol_energy = crystal_batch.lj / crystal_batch.z_prime
        elif energy_function == 'qlj':
            mol_energy = crystal_batch.qlj / crystal_batch.z_prime
        elif energy_function == 'elj':
            mol_energy = crystal_batch.elj / crystal_batch.z_prime
        elif energy_function == 'silu':
            mol_energy = crystal_batch.silu / crystal_batch.z_prime
        elif energy_function == 'uma':
            mol_energy = crystal_batch.uma
        else:
            assert False

        lj_rescale = [-20.6, 5.7, -3.4, 1.5]  # mean and std by which to rescale LJ to align with uma
        if energy_function in ['lj', 'qlj', 'elj'] and lj_rescale is not None:
            # rescale functions with LJ-type minima to uma statistics
            lj_mean, lj_std, uma_mean, uma_std = lj_rescale
            atomwise_energy = mol_energy / (crystal_batch.num_atoms / crystal_batch.z_prime)
            atomwise_fixed = (atomwise_energy - lj_mean) / lj_std * uma_std + uma_mean
            mol_energy = atomwise_fixed * (crystal_batch.num_atoms / crystal_batch.z_prime)

        reduction_en = crystal_batch.compute(['reduction_en'])['reduction_en']
        reduction_energy = F.relu(reduction_en)  # punish positive energies

        ens_dict['reduction_energy'] = reduction_energy
        ens_dict['mol_energy'] = mol_energy
        ens_dict['density_energy'] = density_energy
        ens_dict['bounding_energy'] = bounding_energy
    else:
        reduction_energy = torch.zeros_like(bounding_energy)

    crystal_energy = lj_coeff * mol_energy + density_coeff * density_energy

    if energy_clip is not None:
        total_energy = (log_rescale_positive(crystal_energy,
                                             energy_clip) +
                        bounding_energy * bounding_coeff +
                        reduction_energy * reduction_coeff)
        total_energy = log_rescale_positive(total_energy, energy_clip + 0.1 * np.abs(energy_clip))
    else:
        total_energy = crystal_energy + bounding_energy * bounding_coeff + reduction_energy * reduction_coeff

    return -total_energy / temperature


@torch.no_grad()
def make_kinetic_graph(base_sample, sample_batch, n_points=10):
    ""
    edge_index, ens = kinetic_knn_interpolation(base_sample, n_points, sample_batch)
    E_src = ens[:, 0]  # .elj[edge_index[0,:]]
    E_dst = ens[:, -1]  # sample_batch.elj[edge_index[1,:]]
    gamma = 3.0  # try 3, 5, 10
    kT = 2.5
    Ecut = gamma * kT
    barrier = ens.amax(dim=1) - torch.minimum(E_src, E_dst)
    keep = barrier <= Ecut
    edges = edge_index[:, keep]
    N = sample_batch.num_graphs.shape[0]
    A = sp.coo_matrix(
        (np.ones(edges.shape[1]),
         (edges[0].cpu(), edges[1].cpu())),
        shape=(N, N)
    )
    A = A + A.T

    cluster_labels = connected_components(A, directed=False)[1]

    masks = np.array([cluster_labels == ind for ind in np.unique(cluster_labels)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    sorted_masks = masks[mask_sorts]
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in
                                                   masks[mask_sorts[:10]] if
                                                   sum(m) > 1])
    [print(sum(m)) for m in sorted_masks[:20]]
    degree_gamma = A.sum(axis=1)
    go.Figure(go.Histogram(x=np.array(degree_gamma).flatten())).show()


def kinetic_knn_interpolation(base_sample, n_points, lat_k, sample_batch, dmat, d_cut: float,
                              valid_sources: Optional[torch.Tensor] = None):
    'make latent space knn'
    latents = sample_batch.latent_params()
    lat_knn = dmat.topk(lat_k + 1, largest=False).indices[:, 1:]
    'get all edges to be interpolated'
    N, k = lat_knn.shape
    src = torch.arange(N, device=lat_knn.device).repeat_interleave(k)
    dst = lat_knn.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)  # (2, N*k)
    "prune long edges"
    dist = dmat[edge_index[0], edge_index[1]]
    edge_index = edge_index[:, dist < d_cut]

    "prune to valid sources"
    if valid_sources is not None:
        keep_edges = torch.isin(edge_index[0], valid_sources)
        edge_index = edge_index[:, keep_edges]

    'iterate over edges'
    num_trajs = edge_index.shape[1]
    batch_size = 500
    ens = torch.zeros(num_trajs, n_points, device='cpu', dtype=torch.float32)
    finished = False
    cursor = 0
    pbar = tqdm(total=num_trajs, unit="samples")
    while not finished:
        try:
            batch_inds = torch.arange(cursor, min(cursor + batch_size, num_trajs))
            src_lat = latents[edge_index[0, batch_inds]]
            tgt_lat = latents[edge_index[1, batch_inds]]
            paths = torch.linspace(0, 1, n_points)
            paths = src_lat[None] + paths[:, None, None] * (tgt_lat - src_lat)
            # paths = paths[1:-1,...]  # exclude endpoints
            paths = paths.clip(min=-1, max=1)
            x = paths.reshape(-1, paths.shape[2]).to('cuda')

            sample = base_sample.clone()
            traj_batch = collate_data_list([sample for _ in range(len(x))]).to('cuda')
            traj_batch.latent_to_cell_params(x)
            en = traj_batch.analyze(['elj'], cutoff=10)['elj'].cpu()
            ens[batch_inds, :] = en.reshape(paths.shape[:2]).T

            pbar.update(min(batch_size, num_trajs - cursor))  # safe final update

            cursor += batch_size
            if cursor >= num_trajs:
                finished = True
            else:
                batch_size = int(batch_size * 1.01)  # keep pushing the batch size between sets
                # print(f"Boosting batch size to {batch_size}")

        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                batch_size = max(int(batch_size * 0.9), 1)
                print(f"OOM error: dropping batch size to {batch_size}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sleep(0.1)

            else:
                raise e

    return edge_index, ens


def get_rmsdmat(sample_batch, radius: float = 5):
    sample_batch.pose_aunit(std_orientation=True)
    sample_batch.build_unit_cell()
    upos = sample_batch.unit_cell_pos.reshape(sample_batch.num_graphs,
                                              sample_batch.sym_mult[0] * sample_batch.num_atoms[0], 3)
    #
    # num_graphs = sample_batch.num_graphs
    # device = upos.device
    # if num_graphs > 10000:
    #     all_indices = []
    #     all_values = []
    #     for ind in range(num_graphs):
    #         # Calculate one full row of distances: (1, num_graphs)
    #         # Using .norm and .mean to get the RMSD-like average distance
    #         row_dist = (upos[ind, None, ...] - upos).norm(dim=-1).mean(-1)
    #
    #         # Apply hard cutoff
    #         mask = row_dist <= radius
    #
    #         # Get the indices where the condition is true
    #         # .nonzero() returns the indices of the neighbors
    #         neighbor_idxs = torch.nonzero(mask).flatten()
    #         vals = row_dist[neighbor_idxs]
    #
    #         # Store
    #         rows = torch.full_all((len(vals),), ind, dtype=torch.long, device=upos.device)
    #         all_indices.append(torch.stack([rows, neighbor_idxs], dim=0))
    #         all_values.append(vals)
    #
    #     indices = torch.cat(all_indices, dim=1)
    #     values = torch.cat(all_values)
    #
    #     # Create the sparse COO tensor
    #     sparse_rmsd = torch.sparse_coo_tensor(
    #         indices, values, size=(num_graphs, num_graphs)
    #     ).coalesce()
    # else:

    rmsdmat = torch.zeros(sample_batch.num_graphs, sample_batch.num_graphs)
    for ind in range(sample_batch.num_graphs):
        rmsdmat[ind] = (upos[ind, None, ...] - upos).norm(dim=-1).mean(-1)

    return rmsdmat


def get_directed_kinetic_edges(sample_inds, samples, sample_batch, results_dir, run_name, rmsdmat, kT, d_cut: float = 2,
                               lat_k: int = 50):
    nn_dist = rmsdmat.fill_diagonal_(100).amin(0).median()
    if True:  # not os.path.exists(results_dir + f'/{run_name}_kinetics.pt'):
        edges, ens = kinetic_knn_interpolation(samples[0], 10, lat_k, sample_batch, rmsdmat, d_cut,
                                               valid_sources=torch.as_tensor(sample_inds))
    else:
        edges, ens = torch.load(results_dir + f'/{run_name}_kinetics.pt')
    '''   # a cheaper way
    converged = False
    max_iter = 50
    iter = 0
    sources = torch.as_tensor(sample_inds)
    edges_list = []
    ens_list = []
    visited = torch.zeros(sample_batch.num_graphs, dtype=torch.bool, device=sources.device)
    while not converged and iter < max_iter:
        visited[sources] = True
        edges, ens = kinetic_knn_interpolation(samples[0], 10, lat_k, sample_batch, rmsdmat, d_cut, valid_sources=sources)
        edges_list.extend(edges.T)
        ens_list.extend(ens)

        # check for reachable neighbors
        dir_barrier = ens.amax(dim=1) - ens[:, 0]
        valid = torch.argwhere(dir_barrier < kT).flatten().unique()
        # stop if there are no more valid hops
        if len(valid) == 0:
            converged = True

        # barriers successfully hopped are the new source nodes
        sources = edges[1, valid]
        sources = sources[~visited[sources]]
        if len(sources) == 0:
            converged = True

        iter += 1
    '''
    # dist = rmsdmat[edges[0], edges[1]]

    "2A kNN edges, and linearly interpolated energy profiles"
    E_src = ens[:, 0]
    # E_dst = ens[:, 1]

    # directed barrier
    barrier_dir = ens.max(dim=1).values - E_src
    # dist = (rmsdmat[edges[0]] - rmsdmat[edges[1]]).norm(dim=1)
    Ecut = kT
    keep = barrier_dir <= Ecut
    edges_dir = edges[:, keep]  # directed edges i -> j
    # dist_dir = dist[keep]
    ens_dir = ens[keep]

    return edges_dir, ens_dir

@register_metric
def anisotropy(w, **kwargs):
    return np.log(w[-1] / w[0]) if len(w) >= 2 else 0

@register_metric
def d_eff(w, k, **kwargs):
    if len(w) < 2:
        return 0
    p = w / w.sum()
    return np.exp(-np.sum(p * np.log(p))) / k

@register_metric
def gap(w, **kwargs):
    return w[-1] / w[-2] if len(w) >= 2 else 0

@register_metric
def softness(w, **kwargs):
    return np.log(w[-1]) if len(w) >= 2 else 0

@register_metric
def gauss_entropy(w, **kwargs):
    return 0.5 * np.sum(np.log(w)) if len(w) >= 2 else 0

@register_metric
def local_dim(w, k, **kwargs):
    return len(w) / k if len(w) >= 2 else 0

@register_metric
def grad_mag(w, log_rho_local, log_rho_i, **kwargs):
    return (log_rho_local - log_rho_i).mean() if len(w) >= 2 else 0

@register_metric
def basin_radius(w, dists, **kwargs):
    return np.amax(dists, axis=-1)

@register_metric
def basin_std(w, dists, **kwargs):
    return np.std(dists, axis=-1)

@register_metric
def n_neighbors(w, dists, **kwargs):
    return dists.shape[-1]

@register_metric
def basin_mean_en(w, sample_energy_local, **kwargs):
    return np.mean(sample_energy_local, axis=-1)

@register_metric
def basin_min_en(w, sample_energy, sample_energy_local, **kwargs):
    return min(np.amin(sample_energy_local, axis=-1), sample_energy)

@register_metric
def basin_std_en(w, sample_energy_local, **kwargs):
    return np.std(sample_energy_local, axis=-1)

@register_metric
def is_local_en_minimum(w, sample_energy, sample_energy_local, **kwargs):
    return np.all(sample_energy <= sample_energy_local)

@register_metric
def log_rho(w, log_rho_i, **kwargs):
    return log_rho_i

@register_metric
def basin_max_rho( w, log_rho_local, log_rho_i, **kwargs):
    return max(np.amax(log_rho_local), log_rho_i)

@register_metric
def basin_mean_rho(w, log_rho_local, **kwargs):
    return np.mean(log_rho_local)

@register_metric
def is_local_rho_maximum(w, log_rho_local, log_rho_i, **kwargs):
    return np.all(log_rho_local <= log_rho_i)

@register_metric
def energy_smoothness(dists, sample_energy_local, **kwargs):
    # Correlation between distance from center and energy of neighbors
    # We use the absolute difference to see if energy 'drifts' predictably
    corr = spearmanr(dists, sample_energy_local)
    return corr

def local_analysis(k_values, sample_batch, sample_energy, dmat):
    N = sample_batch.num_graphs
    d2 = (dmat ** 2).numpy()
    sample_latents = sample_batch.latent_params()
    D = sample_latents.shape[1]

    all_results = {}
    for k in k_values:
        nn = NearestNeighbors(
            n_neighbors=k,
            metric='precomputed'
        )
        nn.fit(dmat)
        dists, inds = nn.kneighbors(dmat)

        metrics = k_nn_analysis(D, METRIC_REGISTRY, np.arange(N), d2, dists, inds, k, sample_energy)

        all_results[k] = metrics

    return all_results


def k_nn_analysis(D, METRICS, indices_to_compute, d2, dists, inds, k, sample_energy):
    N = len(indices_to_compute)
    metrics = {key: np.zeros(N) for key in METRICS}

    if isinstance(k, list):
        rk = [max(d) for d in dists]
        log_rho = np.log(k) - D * np.log(rk)

    elif isinstance(k, int):
        rk = dists[:, k - 1]
        # Local density estimate
        H = np.eye(k) - np.ones((k, k)) / k
        log_rho = np.log(k) - D * np.log(rk)

    for i in indices_to_compute:
        if isinstance(k, list):
            kk = k[i]
            H = np.eye(kk) - np.ones((kk, kk)) / kk
        elif isinstance(k, int):
            kk = k

        inds_i = inds[i]
        D2_local = d2[np.ix_(inds_i, inds_i)]

        # centered Gram matrix (intrinsic covariance)
        K = -0.5 * H @ D2_local @ H / (kk - 1)

        w = np.linalg.eigvalsh(K)
        w = w / (rk[i] ** 2)
        w = w[w > 1e-10]

        # Prepare context for all metrics
        context = {
            'w': w,
            'log_rho_local': log_rho[inds_i],
            'log_rho_i': log_rho[i],
            'dists': dists[i, :k],  # distances to neighbors
            'sample_energy_local': np.array(sample_energy[inds_i]),
            'sample_energy': np.array(sample_energy[i]),
            'k': k,
        }

        # Compute all metrics
        for name, func in METRICS.items():
            metrics[name][i] = func(**context)

    return metrics
