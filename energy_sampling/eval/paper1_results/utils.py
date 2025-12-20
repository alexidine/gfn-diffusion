import re
from collections import Counter

import hdbscan
import numpy as np
import pandas as pd
import plotly.colors as pc
import torch
from _plotly_utils.colors import qualitative, color_parser, hex_to_rgb
from matplotlib import cm, colors
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import estimate_bandwidth, MeanShift
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from energy_sampling.utils import uniform_discretizer, get_gfn_init_state
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor


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
    return states[:, -1, :]


def sample_from_gfn(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model):
    samples = []
    counter = 0
    with tqdm(total=num_samples) as pbar:
        while (len(samples) * batch_size) < num_samples:
            states = get_sample_batch(batch_size, max_z_prime, device, n_steps, gfn_model)

            samples.append(states.cpu().detach())

            counter += batch_size
            pbar.update(batch_size)

    samples = torch.cat(samples)
    return samples


def analyze_samples(x, mol_list, max_z_prime, device, batch_size, sg_ind, zp, do_uma: bool = False, predictor=None):
    num_batches = len(mol_list) // batch_size + (1 if len(mol_list) % batch_size else 0)
    num_samples = len(mol_list)
    samples = []
    counter = 0

    with tqdm(total=num_samples) as pbar:
        with torch.no_grad():
            for b_ind in range(num_batches):
                inds = torch.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                for elem in mol_list:
                    elem.z_prime = zp
                batch = collate_data_list([mol_list[ind] for ind in inds], max_z_prime=max_z_prime)
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
                counter += batch_size
                pbar.update(batch_size)

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


def first_hit_graph_MC(energies, basin_inds, traj_len, kT, cval, samples=None, dmat=None):
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


def steepest_descent(samples, energies, cval):
    k = int(samples.shape[1] * cval * np.log(len(samples)))
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(samples.cpu().numpy())
    dists, inds = nbrs.kneighbors(samples.cpu().numpy())
    knn = torch.tensor(inds[:, 1:], device=samples.device)

    minima = torch.empty(len(energies), dtype=torch.long)

    for i in tqdm(range(len(energies))):
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
    for ind in tqdm(range(num_committor_steps)):
        hits = first_hit_graph_MC(
            sample_energy,
            basin_inds,
            traj_len=5000,
            kT=mc_kT,
            cval=cval,
            samples=sample_latents)
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


def get_gfn_samples(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model, energy_function, molecule, sg_ind,
                    zp):
    sample_latents = sample_from_gfn(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model)
    if energy_function == 'uma':
        pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
        predictor = init_uma_crystal_predictor(pred_path, device=device)
    else:
        predictor = None
    samples = analyze_samples(sample_latents, molecule * num_samples, max_z_prime, device, batch_size, sg_ind,
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

    return sample_batch, sample_latents, sample_energy, sample_batch.packing_coeff, samples


def get_cluster_weights(sample_latents, sample_energy, num_samples, num_committor_steps, cval: float = 1,
                        mc_kT: float = 2.5):
    "Soft clustering by committor analysis"
    clabel = steepest_descent(sample_latents, sample_energy, cval=cval)
    basin_inds = torch.unique(clabel)
    num_basins = len(basin_inds)
    basin_weights = get_committor_weights(clabel, basin_inds, num_committor_steps, sample_energy, num_samples,
                                          num_basins, sample_latents, cval=cval, mc_kT=mc_kT)
    hard_assignment = torch.argmax(basin_weights, dim=1)
    hard_assignment_prob = torch.amax(basin_weights, dim=1)
    return basin_weights, hard_assignment, hard_assignment_prob, basin_inds

    # masks = np.array([hard_assignment == ind for ind in np.unique(hard_assignment)])
    # mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    # sorted_masks = masks[mask_sorts]
    # go.Figure(go.Histogram(x=basin_weights.amax(1), nbinsx=100)).show()
    # go.Figure(go.Histogram(x=hard_assignment, nbinsx=len(hard_assignment.unique()))).show()
    # sample_batch.plot_batch_cell_params(space='real',
    #                                     aux_dists=[sample_batch.full_cell_parameters()[m] for m in
    #                                                masks[mask_sorts[:10]] if
    #                                                sum(m) > 1])


def cluster_thermo_analysis(basin_weights, sample_energy, kT, cp, basin_inds, top_cluster_inds):
    min_ens = sample_energy[basin_inds][top_cluster_inds]

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


def get_gfn_logprobs(batch_size, sample_latents, gfn_model, n_steps, max_repeats: int = 50, tol:float=1e-2):
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

