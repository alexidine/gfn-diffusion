import numpy as np
import plotly.graph_objects as go
import torch
from umap import UMAP

from energy_sampling.eval.paper1_results.utils import cluster_dendro_fig, marginal_cluster_1d, coupling_ratio, \
    correlate_mask, top_joint_correlates, latent_dendro_fig, plot_marginals, get_highp_correlations, \
    compute_dim_weights, estimate_logp_with_convergence, sample_from_gfn, analyze_samples
from energy_sampling.models import GFN
from mxtaltools.common.utils import log_rescale_positive
from mxtaltools.dataset_utils.utils import collate_data_list

if __name__ == '__main__':
    device = 'cuda'
    num_samples = 50000
    batch_size = 500
    n_steps = 50  # critical to get this right!

    model_path = r"D:\crystal_datasets\best_nov_nic_5_7_model_eval.pt"
    config_path = r"D:\crystal_datasets\nov_nic_5_7_model_config.npy"
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
    sample_batch.plot_batch_cell_params(space='real', ref_dist=data_batch.full_cell_parameters(), quantiles=[0.1])
    sample_batch.plot_batch_density_funnel()

    "1D Marginal Clusters"
    marginal_labels = marginal_cluster_1d(sample_latents.cpu().detach().numpy())
    n_samples, n_dims = marginal_labels.shape
    clusters_per_dim = np.amax(marginal_labels, axis=0) + 1

    "Latent Space Dendrogram"
    low_en_bools = sample_batch.lj_pot < torch.quantile(sample_batch.lj_pot, 0.1)
    latent_dendro_fig(sample_latents[low_en_bools].cpu().detach().numpy(),
                      sample_batch.lj_pot[low_en_bools].cpu().detach().numpy())

    "High coupling n-dimensional correlation clusters for any n"
    corr_df = get_highp_correlations(marginal_labels, n_samples, n_dims, clusters_per_dim, 2, 4)
    dim_weights = compute_dim_weights(corr_df, ratio_thresh=2.0, order_min=2)
    masks = [
        correlate_mask(marginal_labels, row.dims, row.clusters)
        for _, row in corr_df[corr_df.order == 2].iterrows()
    ]
    sample_batch.plot_batch_cell_params(space='latent', aux_dists=[sample_batch.latent_params()[m] for m in masks])

    "High coupling n-dimensional correlation clusters for n=n_dims"
    plot_marginals(sample_latents, labels=marginal_labels, clusters_per_dim=clusters_per_dim)
    baseline_p = 1 / np.prod(clusters_per_dim)

    top_df = top_joint_correlates(marginal_labels, k=500)
    marginal_ps = [np.bincount(marginal_labels[:, i]) / len(marginal_labels) for i in range(marginal_labels.shape[1])]
    top_df["coupling"] = top_df.apply(coupling_ratio, axis=1, args=(marginal_ps,))

    masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind in
             range(len(top_df))]
    top_df['mean_en'] = [log_rescale_positive(sample_batch.lj_pot[m]).mean().cpu().detach().item() for m in masks]

    cluster_dendro_fig(top_df[top_df.mean_en < -150])

    "Estimate state sampling probability"
    sort_inds = torch.argsort(sample_batch.lj_pot)[:batch_size]
    terminal_states = sample_latents[sort_inds, :]
    logp_est, _ = estimate_logp_with_convergence(
        gfn_model, terminal_states, batch_size, n_steps=n_steps, max_repeats=500, tol=1e-2, window=10
    )

    boltzmann_logprobs = -(sample_batch.lj_pot / sample_batch.num_atoms)[
        sort_inds] - gfn_model.flow_model().item()  # unconditional boltzmann factor

    go.Figure(go.Scatter(x=logp_est.cpu().detach(), y=boltzmann_logprobs.cpu().detach(), mode='markers')).show()

    "Hierarchical joint probabilities"
    # df = hierarchical_joint_df(marginal_labels, max_order=3, cutoff=0.005)

    "Dimension Reduction"
    umap_model = UMAP(n_components=2, n_neighbors=100, min_dist=0.01)
    sample_embedding = umap_model.fit_transform(sample_latents[low_en_bools])
    # masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind in
    #          range(20)]
    masks = [
        correlate_mask(marginal_labels, row.dims, row.clusters)[low_en_bools]
        for _, row in corr_df[corr_df.order == 2].iloc[:20].iterrows()
    ]
    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], mode='markers', opacity=0.25, showlegend=False)
    for ind, m in enumerate(masks):
        fig.add_scatter(x=sample_embedding[m, 0], y=sample_embedding[m, 1], mode='markers', opacity=1.0, showlegend=False, marker_color=ind)
    fig.show()

    end = 1
