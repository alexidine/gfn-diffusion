import os

import numpy as np
import torch

from energy_sampling.eval.paper1_results.figures import general_figs, cluster_comparison_fig, dim_reduction_fig, \
    boltzmann_fig, make_thermo_table
from energy_sampling.eval.paper1_results.utils import get_gfn_samples, \
    cluster_thermo_analysis, get_color_set, get_gfn_logprobs, \
    kinetic_clustering
from energy_sampling.models import GFN
from examples.crystal_search_reporting import batch_compack
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.dataset_utils.utils import collate_data_list
import plotly.graph_objects as go

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def save_outputs(sample_batch, sample_latents, sample_energy, sample_cp, samples, basin_weights, hard_assignment,
                 hard_assignment_prob, basin_inds, top_cluster_inds, logp_est, learned_log_Z):
    results = {
        "version": 1,

        # --- identity / invariants ---
        "meta": {
            "num_samples": num_samples,
            "max_z_prime": max_z_prime,
            "sg_ind": sg_ind,
            "zp": zp,
            "cval": cval,
            "n_steps": n_steps,
        },

        # --- GFN sampling outputs ---
        "samples": {
            "batch": sample_batch,  # heavy, but sometimes useful
            "latents": sample_latents,  # (N, D)
            "energy": sample_energy,  # (N,)
            "cp": sample_cp,  # (N,)
            "raw": samples,  # whatever get_gfn_samples returns
        },

        # --- committor / clustering ---
        "committor": {
            "basin_weights": basin_weights,  # (N, B)
            "hard_assignment": hard_assignment,  # (N,)
            "hard_assignment_prob": hard_assignment_prob,  # (N,)
            "basin_inds": basin_inds,  # (B,)
            "top_cluster_inds": top_cluster_inds,  # (B,)
        },

        # --- density / probabilities ---
        "density": {
            "logp_est": logp_est,  # (N,)
            "learned_log_Z": learned_log_Z,  # scalar
            "logp_settings": {
                "batch_size": batch_size,
                "max_repeats": max_repeats,
                "tol": tol,
            }
        },
    }
    torch.save(results, results_path)


def load_results():
    R = torch.load(results_path, weights_only=False)
    # ---- samples ----
    sample_batch = R["samples"].get("batch", None)
    sample_latents = R["samples"]["latents"].clip(max=1, min=-1)
    sample_energy = R["samples"]["energy"]
    sample_cp = R["samples"]["cp"]
    samples = R["samples"].get("raw", None)
    # ---- committor ----
    basin_weights = R["committor"]["basin_weights"]
    hard_assignment = R["committor"]["hard_assignment"]
    hard_assignment_prob = R["committor"]["hard_assignment_prob"]
    basin_inds = R["committor"]["basin_inds"]
    top_cluster_inds = R["committor"]["top_cluster_inds"]
    # ---- density ----
    logp_est = R["density"]["logp_est"]
    learned_log_Z = R["density"]["learned_log_Z"]
    return sample_batch, sample_latents, sample_energy, sample_cp, samples, basin_weights, hard_assignment, hard_assignment_prob, basin_inds, top_cluster_inds, logp_est, learned_log_Z


def sample_and_analyze():
    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()
    "Sample from GFN & process samples"
    sample_batch, sample_latents, sample_energy, sample_cp, samples = get_gfn_samples(
        num_samples, max_z_prime,
        device, n_steps, batch_size, gfn_model,
        energy_function, molecule, sg_ind, zp
    )

    basin_weights, hard_assignment, hard_assignment_prob, basin_inds = kinetic_clustering(
        sample_latents, sample_energy, cval, clust_kT)

    top_cluster_inds = torch.argsort(basin_weights.sum(0), descending=True).flatten()
    'Explicit Density Estimation'
    logp_est = get_gfn_logprobs(1000, sample_latents, gfn_model, n_steps, max_repeats, tol)
    learned_log_Z = gfn_model.flow_model().item()
    if save_results:
        if os.path.exists(results_path):
            if overwrite_results:
                save_outputs(sample_batch, sample_latents, sample_energy, sample_cp, samples, basin_weights,
                             hard_assignment, hard_assignment_prob, basin_inds, top_cluster_inds, logp_est,
                             learned_log_Z)
            else:
                pass
        else:
            save_outputs(sample_batch, sample_latents, sample_energy, sample_cp, samples, basin_weights,
                         hard_assignment, hard_assignment_prob, basin_inds, top_cluster_inds, logp_est, learned_log_Z)

    return sample_batch, sample_latents, sample_energy, sample_cp, samples, basin_weights, hard_assignment, hard_assignment_prob, basin_inds, top_cluster_inds, logp_est, learned_log_Z


if __name__ == '__main__':
    "Configs & args"
    '''
    # test nicotinamide config
    run_name = 'nic_test'
    device = 'cuda'
    num_samples = 20000
    batch_size = 1000
    energy_function = 'elj'  # 'elj', 'lj' 'uma
    n_steps = 50  # critical to get this right!
    sg_ind = 14
    zp = 1  # todo fix zp>1 pre-processing
    show_figs = False
    cval = 1
    kT = 2.5
    clusters_to_analyze = 8
    max_repeats = 50
    tol = 1e-2
    model_path = rf"D:\crystal_datasets\nic3_sg{sg_ind}_zp{zp}_2_model_eval.pt"
    config_path = rf"D:\crystal_datasets\nic3_sg{sg_ind}_zp{zp}_2_model_config.npy"
    molecule_path = r"D:\crystal_datasets\nicoam\protonated_nicotinamide.pt"
    dataset_path = rf"D:\crystal_datasets\nicoam\nic_sg{sg_ind}_zp{zp}.pt"
    results_path = rf"D:\crystal_datasets\gfn_results\{run_name}_sg{sg_ind}_zp{zp}.pt"
    '''
    # acridine lj config
    run_name = 'acr_lj'
    device = 'cuda'
    num_samples = 10000
    batch_size = 1000
    energy_function = 'elj'  # 'elj', 'lj' 'uma
    n_steps = 100  # critical to get this right!
    sg_ind = 2
    zp = 1  # todo fix zp>1 pre-processing
    cval = 0.1
    kT = 2.5
    clust_kT = 7.5
    clusters_to_analyze = 10
    max_repeats = 50
    tol = 1e-2
    model_path = rf"D:\crystal_datasets\acridine\best_acr_lj_sg{sg_ind}_zp{zp}_2_model_eval.pt"
    config_path = rf"D:\crystal_datasets\acridine\acr_lj_sg{sg_ind}_zp{zp}_2_model_config.npy"
    molecule_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
    dataset_path = rf"D:\crystal_datasets\acridine\acridine_sg{sg_ind}_zp{zp}.pt"
    results_path = rf"D:\crystal_datasets\gfn_results\{run_name}_sg{sg_ind}_zp{zp}.pt"
    reload_results = True
    show_figs = True
    write_figs = False
    save_results = True
    overwrite_results = True
    do_general_vis = True
    do_clustering = True
    do_dimension_reduction = True
    do_explicit_probs = True

    "Load Relevant Dataset"
    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])
    data_batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    data_latents = data_batch.latent_params()

    if reload_results and os.path.exists(results_path):
        (sample_batch, sample_latents, sample_energy, sample_cp, samples,
         basin_weights, cluster_labels, cluster_prob, basin_inds,
         top_cluster_inds, logp_est, learned_log_Z) = load_results()
    else:
        "Load GFN"
        (sample_batch, sample_latents, sample_energy, sample_cp, samples,
         basin_weights, cluster_labels, cluster_prob,
         basin_inds, top_cluster_inds, logp_est, learned_log_Z) = sample_and_analyze()

    """
    Make Figures
    """
    basin_weights, cluster_labels, cluster_prob, basin_inds = kinetic_clustering(
        sample_latents, sample_energy, cval, clust_kT)
    top_cluster_inds = torch.argsort(basin_weights.sum(0), descending=True).flatten()
    confident_cluster_labels = cluster_labels.clone()
    confident_cluster_labels[cluster_prob < 0.9] = -1

    masks = np.array([cluster_labels == ind for ind in np.unique(cluster_labels)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    sorted_masks = masks[mask_sorts]
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in
                                                   masks[mask_sorts[:10]] if
                                                   sum(m) > 1])

    clusters_to_analyze = min(len(top_cluster_inds), clusters_to_analyze)
    cluster_color = get_color_set(clusters_to_analyze, alpha=0.7)

    fig_dict = {}
    if do_general_vis:  # Standard visualizations
        fig_dict = general_figs(fig_dict, sample_batch, sample_energy, data_batch)

    if do_clustering:  # Clustering & thermodynamic analysis
        min_ens, Zb, Fb, basin_probs, mean_rho, Sb, mean_E = cluster_thermo_analysis(basin_weights, sample_energy, kT,
                                                                                     sample_cp, basin_inds,
                                                                                     top_cluster_inds)

        fig_dict['clusters'] = cluster_comparison_fig(top_cluster_inds,
                                                      sample_cp, sample_energy, cluster_labels,
                                                      sample_batch, 6, sample_latents,
                                                      cluster_color,
                                                      )
        fig_dict['Thermo Table'] = make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho,
                                                     cluster_labels, clusters_to_analyze)

    if do_dimension_reduction:
        assert do_clustering, "Need clusters for dim reduction fig"
        fig_dict['Dim Reduction'] = dim_reduction_fig(sample_batch.latent_distmat(),
                                                      cluster_labels,
                                                      clusters_to_analyze,
                                                      cluster_color,
                                                      basin_inds)

    if do_explicit_probs:
        fig_dict['boltzmann'] = boltzmann_fig(sample_energy, kT, learned_log_Z, logp_est)

    for key, fig in fig_dict.items():
        if key == 'boltzmann':
            width = 700
            height = 700
            fig.update_layout(
                width=width,
                height=height,
                font_size=20
            )

        elif key == 'staircase_fig':
            width = 2000
            height = 1300
            fig.update_layout(
                width=width,
                height=height,
                font_size=24
            )

        elif key == 'std_marginals_fig':
            width = 1920
            height = 1080
            fig.update_layout(
                width=width,
                height=height,
                font_size=20
            )

        elif key == 'density_funnel_fig':
            width = 700
            height = 700
            fig.update_layout(
                width=width,
                height=height,
                font_size=24
            )

        elif key == 'clusters':
            width = 1800
            height = 1200
            fig.update_layout(
                width=width,
                height=height,
                font_size=24,
            )
            fig.update_annotations(font_size=24)

        elif key == 'Thermo Table':
            width = 800
            height = 300
            fig.update_layout(
                width=width,
                height=height
            )

        elif key == 'Dim Reduction':
            width = 800
            height = 800
            fig.update_layout(
                width=width,
                height=height
            )

        if show_figs:
            fig.show()
        if write_figs:
            fig.write_image(
                rf"C:\Users\mikem\OneDrive\NYU\CSD\papers\generator\{run_name}_{key.replace(' ', '_')}.png",
                height=height,
                width=width,
                scale=2)

        aa = 1

    'best samples analysis'  # todo replace with probability maximum
    best_samples = [samples[ind] for ind in top_cluster_inds[:clusters_to_analyze]]
    best_batch = collate_data_list(best_samples)

    bin_edges = torch.linspace(0, 6, sample_batch.rdf.shape[-1], )
    dmat = compute_rdf_distmat(sample_batch.rdf[top_cluster_inds[:clusters_to_analyze]], bin_edges,
                               chunk_size=10000)
    go.Figure(go.Heatmap(z=dmat)).show()

    matchess, rmsdss = [], []
    for ind in range(best_batch.num_graphs):
        matches, rmsds = batch_compack([ind for ind in range(best_batch.num_graphs)],
                                       best_samples,
                                       collate_data_list([best_samples[ind]]).mol2cluster(cutoff=6))
        matchess.append(matches)
        rmsdss.append(rmsds)

    clusters = best_batch.mol2cluster(cutoff=6)
    clusters.visualize(mode='unit cell')

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

    # 'RDF space MC'
    # rdf_device = 'cuda'
    # bin_edges = torch.linspace(0, 6, sample_batch.rdf.shape[-1], device=rdf_device)
    # dmat = compute_rdf_distmat(sample_batch.rdf.to(rdf_device), bin_edges, chunk_size=10000).cpu()
    # best, best_energy = graph_MC(sample_energy,
    #                              traj_len=500,
    #                              kT=2.5,
    #                              cval=1.0,
    #                              dmat=dmat)
    # 
    # if show_plots:
    #     go.Figure(go.Histogram(x=best, nbinsx=len(best.unique()))).show()
    #     masks = np.array([best == ind for ind in np.unique(best)])
    #     mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    #     sample_batch.plot_batch_cell_params(space='real',
    #                                         aux_dists=[sample_batch.full_cell_parameters()[m] for m in
    #                                                    masks[mask_sorts[:10]] if
    #                                                    sum(m) > 1])
    
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


'''
