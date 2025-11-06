import numpy as np
import pandas as pd
from _plotly_utils.colors import qualitative
from matplotlib import cm, colors
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import pdist
from sklearn.cluster import estimate_bandwidth, MeanShift


def cluster_1d(X):
    X_ = X.reshape(-1, 1)
    bandwidth = estimate_bandwidth(X_, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(X_)
    n_clusters = len(np.unique(labels))
    return n_clusters, labels

def plot_marginals(latents, labels, clusters_per_dim):
    fig = make_subplots(rows=4, cols=3)
    colors = qualitative.Dark24
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
