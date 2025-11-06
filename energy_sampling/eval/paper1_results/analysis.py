from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from energy_sampling.eval.paper1_results.utils import cluster_dendro_fig, marginal_cluster_1d, coupling_ratio, \
    correlate_mask, top_joint_correlates, latent_dendro_fig, plot_marginals
from energy_sampling.models import GFN
from energy_sampling.utils import get_gfn_init_state, uniform_discretizer
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.dataset_utils.utils import collate_data_list


def get_highp_correlations(marginal_labels, n_samples, n_dims, cutoff: float = 2.0, max_depth: int = 4):
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


def compute_dim_weights(corr_df, use_pjoint=True):
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
        condition = torch.zeros((batch_size, 1), device=gfn_model.device)

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


def analyze_samples(x, mol_list, max_z_prime, device, batch_size):
    num_batches = len(mol_list) // batch_size + (1 if len(mol_list) % batch_size else 0)

    samples = []
    counter = 0
    with tqdm(total=num_samples) as pbar:
        with torch.no_grad():
            for b_ind in range(num_batches):
                inds = torch.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                batch = collate_data_list([mol_list[ind] for ind in inds], max_z_prime=max_z_prime)
                batch.reset_sg_info(2)
                batch.latent_to_cell_params(x[inds])
                batch = batch.to(device)
                outs = batch.analyze(['lj', 'silu'], cutoff=10, std_orientation=True)
                batch.add_graph_attr(outs['lj'], 'lj_pot')
                batch.add_graph_attr(outs['silu'], 'silu_pot')
                batch.to('cpu')
                samples.extend(batch.batch_to_list())
                counter += batch_size
                pbar.update(batch_size)

    return samples


if __name__ == '__main__':
    device = 'cuda'
    num_samples = 20000
    batch_size = 500
    n_steps = 50  # critical to get this right!

    model_path = r"D:\crystal_datasets\nov_nic_4_3_train_thermalized.pt"  # r"C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\checkpoints\nov_nic_2_1_train_hit_prior.pt"
    config_path = r"D:\crystal_datasets\nov_nic_4_3_model_config.npy"  # r"C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\checkpoints\nov_nic_2_1_model_config.npy"
    molecule_path = r'D:/crystal_datasets/nicotinamide.pt'
    dataset_path = r'D:/crystal_datasets/opt_outputs/nicotinamide_sg_1_2.pt'

    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()

    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    dataset = [elem for elem in dataset if elem.sg_ind == 2]
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])
    data_batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    data_latents = data_batch.latent_params()

    sample_latents = sample_from_gfn(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model)
    samples = analyze_samples(sample_latents, molecule * len(sample_latents), max_z_prime, device, batch_size)
    sample_batch = collate_data_list(samples, max_z_prime=max_z_prime)

    """Analyses"""
    "Standard visualizations"
    sample_batch.plot_batch_staircase(space='real')
    sample_batch.plot_batch_cell_params(space='real', ref_dist=data_batch.full_cell_parameters())
    sample_batch.plot_batch_density_funnel()

    "Latent Space Analysis"
    latent_dendro_fig(sample_latents[sample_batch.lj_pot < 0].cpu().detach().numpy(),
                      sample_batch.lj_pot[sample_batch.lj_pot < 0].cpu().detach().numpy())

    "1D Marginal Clusters"
    marginal_labels = marginal_cluster_1d(sample_latents.cpu().detach().numpy())
    n_samples, n_dims = marginal_labels.shape
    clusters_per_dim = np.amax(marginal_labels, axis=0) + 1

    "High coupling n-dimensional correlation clusters for any n"
    corr_df = get_highp_correlations(marginal_labels, n_samples, n_dims, 2, 4)
    dim_weights = compute_dim_weights(corr_df)
    # masks = [
    #     correlate_mask(marginal_labels, row.dims, row.clusters)
    #     for _, row in corr_df[corr_df.order == 2].iterrows()
    # ]
    # sample_batch.plot_batch_cell_params(space='latent', aux_dists=[sample_batch.latent_params()[m] for m in masks])

    "High coupling n-dimensional correlation clusters for n=n_dims"
    plot_marginals(sample_latents, labels=marginal_labels, clusters_per_dim=clusters_per_dim)
    baseline_p = 1 / np.prod(clusters_per_dim)

    top_df = top_joint_correlates(marginal_labels, k=1000)
    marginal_ps = [np.bincount(marginal_labels[:, i]) / len(marginal_labels) for i in range(marginal_labels.shape[1])]
    top_df["coupling"] = top_df.apply(coupling_ratio, axis=1, args=(marginal_ps,))

    masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind in
             range(len(top_df))]
    top_df['mean_en'] = [log_rescale_positive(sample_batch.lj_pot[m]).mean().cpu().detach().item() for m in masks]

    cluster_dendro_fig(top_df[top_df.mean_en < 0])

    "Estimate state sampling probability"
    sort_inds = torch.argsort(sample_batch.lj_pot)[:batch_size]
    terminal_states = sample_latents[sort_inds, :]
    logp_est, _ = estimate_logp_with_convergence(
        gfn_model, terminal_states, n_steps=n_steps, max_repeats=500, tol=1e-2, window=10
    )

    boltzmann_logprobs = -(sample_batch.lj_pot / sample_batch.num_atoms)[
        sort_inds] - gfn_model.flow_model().item()  # unconditional boltzmann factor

    go.Figure(go.Scatter(x=logp_est.cpu().detach(), y=boltzmann_logprobs.cpu().detach(), mode='markers')).show()

    "Hierarchical joint probabilities"
    # df = hierarchical_joint_df(marginal_labels, max_order=3, cutoff=0.005)

    end = 1
