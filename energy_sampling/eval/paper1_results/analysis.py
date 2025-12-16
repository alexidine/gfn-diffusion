import numpy as np
import plotly.graph_objects as go
import torch
from scipy.stats import linregress
from umap import UMAP

from energy_sampling.eval.paper1_results.utils import sample_from_gfn, analyze_samples, cluster_hdbscan_to_df, \
    estimate_logp_with_convergence
from energy_sampling.models import GFN
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)

if __name__ == '__main__':
    device = 'cuda'
    num_samples = 10000
    batch_size = 1000
    energy_function = 'elj'  # 'elj', 'lj' 'uma
    n_steps = 50  # critical to get this right!
    sg_ind = 14
    zp = 1  # todo fix zp>1 pre-processing

    model_path = rf"D:\crystal_datasets\nic3_sg{sg_ind}_zp{zp}_2_model_eval.pt"
    config_path = rf"D:\crystal_datasets\nic3_sg{sg_ind}_zp{zp}_2_model_config.npy"
    molecule_path = r"D:\crystal_datasets\nicoam\protonated_nicotinamide.pt"
    dataset_path = rf"D:\crystal_datasets\nicoam\nic_sg{sg_ind}_zp{zp}.pt"

    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()

    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])
    data_batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    data_latents = data_batch.latent_params()

    sample_latents = sample_from_gfn(num_samples, max_z_prime, device, n_steps, batch_size, gfn_model)
    if energy_function == 'uma':
        pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
        predictor = init_uma_crystal_predictor(pred_path, device=device)
    else:
        predictor = None

    samples = analyze_samples(sample_latents, molecule * len(sample_latents), max_z_prime, device, batch_size, sg_ind, zp,
                              do_uma=energy_function == 'uma', predictor=predictor)
    sample_batch = collate_data_list(samples, max_z_prime=max_z_prime)

    """Analyses"""
    if energy_function == 'uma':
        sample_energy = sample_batch.uma_pot / (sample_batch.sym_mult * sample_batch.z_prime) - sample_batch.uma_gas_pot
    elif energy_function == 'elj':
        sample_energy = sample_batch.elj
        sample_energy = ((sample_energy - -293) / 99) * 41 + -31
    else:
        sample_energy = sample_batch.lj

    # # sort_inds = torch.argsort(sample_energy)[:batch_size]
    # # terminal_states = sample_latents[sort_inds, :]
    # terminal_states = sample_latents[:batch_size, :]
    # logp_est, _ = estimate_logp_with_convergence(
    #     gfn_model, terminal_states, batch_size, n_steps=n_steps, max_repeats=500, tol=1e-2, window=10
    # )
    #
    # boltzmann_logprobs = -sample_energy[
    #     :batch_size] / 2.5 - gfn_model.flow_model().item()  # unconditional boltzmann factor
    # x = logp_est.cpu().detach()
    # y = boltzmann_logprobs.cpu().detach()
    # linreg = linregress(x, y)
    # go.Figure(go.Scatter(x=x, y=y, mode='markers')).show()

    "Dimension Reduction"
    real_params = sample_batch.full_cell_parameters()
    whitened_cell_params = (real_params - real_params.mean(0)) / torch.maximum(real_params.std(0), torch.ones_like(real_params.std(0)))
    umap_model = UMAP(n_components=6, n_neighbors=10, min_dist=0.001)
    sample_embedding = umap_model.fit_transform(whitened_cell_params)  # [low_en_bools])

    "Clustering in umap dims"
    clust_df, clust_labels = cluster_hdbscan_to_df(torch.Tensor(sample_embedding), sample_energy)
    clust_df = clust_df.sort_values('p', ascending=False)
    masks = [clust_labels == ind for ind in np.unique(clust_labels)]

    if energy_function == 'uma':
        # p21/c nicotinamide
        nic = torch.load(r"D:\crystal_datasets\protonated_nicoam\protonated_nonstandard_nicotinamide.pkl",
                         weights_only=False)
        nb = collate_data_list(nic)
        ref_out = nb.analyze(['elj'])
        elj_en = ref_out['elj'] / nb.num_atoms

        import plotly.graph_objects as go

        gas_en = nb.compute_lattice_gas_phase_uma(predictor, std_orientation=True).cpu().detach() * 96.485
        cry_en = nb.compute_crystal_uma(predictor=predictor, std_orientation=True).cpu().detach() * 96.485
        uma_en = cry_en / nb.sym_mult - gas_en

        fig = go.Figure(
            go.Scatter(x=sample_batch.packing_coeff, y=sample_energy, mode='markers'))
        fig.add_scatter(x=nb.packing_coeff, y=uma_en, mode='markers', marker_size=20)
        fig.show()

    "Umap visualization"
    umap_model = UMAP(n_components=2, n_neighbors=10, min_dist=0.001)
    sample_embedding = umap_model.fit_transform(whitened_cell_params)  # [low_en_bools])

    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], mode='markers', opacity=0.25, showlegend=False)
    for ind, m in enumerate(masks):
        fig.add_scatter(x=sample_embedding[m, 0], y=sample_embedding[m, 1], mode='markers', opacity=1.0,
                        showlegend=False, marker_color=ind)
    fig.show()

    m_sort = np.argsort([sum(m) for m in masks])
    sort_masks = [masks[ind] for ind in m_sort[::-1]]
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in sort_masks[:10] if
                                                   sum(m) > 1])

    "Standard visualizations"
    sample_batch.plot_batch_staircase(space='real')
    sample_batch.plot_batch_cell_params(space='real', ref_dist=data_batch.full_cell_parameters(), quantiles=[0.1],
                                        override_energy=sample_energy)
    sample_batch.plot_batch_density_funnel(override_energy=sample_energy)  # , color_flag=clust_labels)

    end = 1

'''
# other analyses

 
# GM business

from sklearn.mixture import GaussianMixture
X = sample_latents.clone().cpu().detach().numpy()[sample_energy < sample_energy.quantile(0.25)]
E = sample_energy.clone().cpu().detach().numpy()[sample_energy < sample_energy.quantile(0.25)]
K = 30   # deliberately too many
gmm = GaussianMixture(
    n_components=K,
    covariance_type="full",
    n_init=5,
    reg_covar=1e-6,
    random_state=0,
)
gmm.fit(X)
resp = gmm.predict_proba(X)     # (N, K)
Nk = resp.sum(axis=0)           # soft population per component
sample_batch.plot_batch_cell_params(space='latent', ref_dist=torch.tensor(gmm.means_))
E_k = (resp * E[:, None]).sum(axis=0) / Nk
go.Figure(go.Scatter(x=Nk, y=E_k, mode='markers')).show()
O = resp.T @ resp / len(X)
go.Figure(go.Heatmap(z=O)).show()

eff_components = (resp > 0.1).sum(axis=1)
go.Figure(go.Histogram(x=eff_components, nbinsx=50)).show()


    
    "1D Marginal Clusters"
    cell_params = sample_batch.full_cell_parameters()
    marginal_labels = marginal_cluster_1d(cell_params.cpu().detach().numpy())
    n_samples, n_dims = marginal_labels.shape
    clusters_per_dim = np.amax(marginal_labels, axis=0) + 1
    plot_marginals(cell_params, labels=marginal_labels, clusters_per_dim=clusters_per_dim)
    top_df = top_joint_correlates(marginal_labels, k=500)
    marginal_ps = [np.bincount(marginal_labels[:, i]) / len(marginal_labels) for i in range(marginal_labels.shape[1])]
    top_df["coupling"] = top_df.apply(coupling_ratio, axis=1, args=(marginal_ps,))

    masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind in
             range(len(top_df))]
    top_df['mean_en'] = [log_rescale_positive(sample_energy[m]).mean().cpu().detach().item() for m in masks]
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in masks[:10] if sum(m) > 1])

    cluster_dendro_fig(top_df[top_df.mean_en < -150])



    "1D Marginal Clusters"
    marginal_labels = marginal_cluster_1d(sample_latents.cpu().detach().numpy())
    n_samples, n_dims = marginal_labels.shape
    clusters_per_dim = np.amax(marginal_labels, axis=0) + 1

    "Latent Space Dendrogram"
    low_en_bools = sample_energy < torch.quantile(sample_energy, 0.1)
    latent_dendro_fig(sample_latents[low_en_bools].cpu().detach().numpy(),
                      sample_energy[low_en_bools].cpu().detach().numpy())

    "High coupling n-dimensional correlation clusters for any n"
    corr_df = get_highp_correlations(marginal_labels, n_samples, n_dims, clusters_per_dim, 2, 4)
    dim_weights = compute_dim_weights(corr_df, ratio_thresh=2.0, order_min=2)
    masks = [
        correlate_mask(marginal_labels, row.dims, row.clusters)
        for _, row in corr_df[corr_df.order == 2].iterrows()
    ]
    sample_batch.plot_batch_cell_params(space='latent',
                                        aux_dists=[sample_batch.latent_params()[m] for m in masks[:10] if sum(m) > 1])

    "High coupling n-dimensional correlation clusters for n=n_dims"
    plot_marginals(sample_latents, labels=marginal_labels, clusters_per_dim=clusters_per_dim)
    baseline_p = 1 / np.prod(clusters_per_dim)

    top_df = top_joint_correlates(marginal_labels, k=500)
    marginal_ps = [np.bincount(marginal_labels[:, i]) / len(marginal_labels) for i in range(marginal_labels.shape[1])]
    top_df["coupling"] = top_df.apply(coupling_ratio, axis=1, args=(marginal_ps,))

    masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind in
             range(len(top_df))]
    top_df['mean_en'] = [log_rescale_positive(sample_energy[m]).mean().cpu().detach().item() for m in masks]

    cluster_dendro_fig(top_df[top_df.mean_en < -150])

    "Estimate state sampling probability"
    if False:
        sort_inds = torch.argsort(sample_energy)[:batch_size]
        terminal_states = sample_latents[sort_inds, :]
        logp_est, _ = estimate_logp_with_convergence(
            gfn_model, terminal_states, batch_size, n_steps=n_steps, max_repeats=500, tol=1e-2, window=10
        )

        boltzmann_logprobs = -(sample_energy / sample_batch.num_atoms)[
            sort_inds] - gfn_model.flow_model().item()  # unconditional boltzmann factor

        go.Figure(go.Scatter(x=logp_est.cpu().detach(), y=boltzmann_logprobs.cpu().detach(), mode='markers')).show()

    "Hierarchical joint probabilities"
    # df = hierarchical_joint_df(marginal_labels, max_order=3, cutoff=0.005)

    "Top Cluster Analysis"
    masks = [correlate_mask(marginal_labels, top_df.loc[ind, "dims"], top_df.loc[ind, "clusters"]) for ind
             in
             range(20)]
    sample_batch.plot_batch_cell_params(space='latent', aux_dists=[sample_batch.latent_params()[m] for m in masks[:10]])
    top_df.p / top_df.p.sum()

'''
