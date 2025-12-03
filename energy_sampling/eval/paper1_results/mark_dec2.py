import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from umap import UMAP

from energy_sampling.eval.paper1_results.utils import cluster_dendro_fig, marginal_cluster_1d, coupling_ratio, \
    correlate_mask, top_joint_correlates, latent_dendro_fig, plot_marginals, get_highp_correlations, \
    compute_dim_weights, sample_from_gfn, analyze_samples, cluster_hdbscan_to_df, estimate_logp_with_convergence
from energy_sampling.models import GFN
from mxtaltools.common.utils import log_rescale_positive, get_point_density
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)

if __name__ == '__main__':
    device = 'cuda'
    num_samples = 10000
    energy_function = 'uma'  # 'elj', 'lj'
    molecule_path = r'D:\crystal_datasets\protonated_nicoam\nicoam0.pkl'
    dataset_path = r'D:/crystal_datasets/opt_outputs/nic_2_zp1.pt'

    if energy_function == 'uma':
        pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
        predictor = init_uma_crystal_predictor(pred_path, device=device)
    else:
        predictor = None

    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    dataset = [elem for elem in dataset if elem.sg_ind == 2]
    data_batch = collate_data_list(dataset, max_z_prime=1)

    gen_samples = torch.load('mark_dec2_cold_uma.pt', weights_only=False)
    sample_batch = collate_data_list(gen_samples, max_z_prime=1)

    """Analyses"""
    if energy_function == 'uma':
        sample_energy = sample_batch.uma_pot / sample_batch.sym_mult - sample_batch.uma_gas_pot
    elif energy_function == 'elj':
        sample_energy = sample_batch.elj
    else:
        sample_energy = sample_batch.lj

    "Dimension Reduction"
    real_params = sample_batch.full_cell_parameters()
    whitened_cell_params = (real_params - real_params.mean(0)) / real_params.std(0)
    umap_model = UMAP(n_components=6, n_neighbors=10, min_dist=0.001)
    sample_embedding = umap_model.fit_transform(whitened_cell_params)  # [low_en_bools])

    "Clustering in umap dims"
    clust_df, clust_labels = cluster_hdbscan_to_df(torch.Tensor(sample_embedding), sample_energy)
    clust_df = clust_df.sort_values('p', ascending=False)
    masks = [clust_labels == ind for ind in np.unique(clust_labels)]


    def density_funnel():

        energy = sample_energy.cpu().detach().numpy()
        cp = sample_batch.packing_coeff.cpu().detach()[energy < np.quantile(energy, 0.95)]
        energy = energy[energy < np.quantile(energy, 0.95)]

        xy = np.vstack([cp, energy])
        try:
            z = get_point_density(xy, bins=25)
        except:
            z = np.ones(xy.shape[1])

        scatter_dict = {'energy': energy,
                        'packing_coefficient': cp,
                        }

        cscale = px.colors.cyclical.IceFire
        color_tag = 'Point Density'
        scatter_dict.update({'Point Density': z})

        opacity = max(0.25, 1 - sample_batch.num_graphs / 5e4)
        df = pd.DataFrame.from_dict(scatter_dict)

        fig = px.scatter(df,
                         x='packing_coefficient', y='energy',
                         color=color_tag,
                         color_continuous_scale=cscale,
                         marginal_x='violin', marginal_y='violin',
                         color_discrete_sequence=px.colors.qualitative.Set3 if color_tag == 'Space Group' else None,
                         opacity=opacity
                         )

        fig.update_layout(yaxis_title='Energy', xaxis_title='Packing Coeff')
        fig.update_layout(yaxis_range=[np.amin(df['energy']) - np.ptp(df['energy']) * 0.05,
                                       min(10, np.amax(df['energy']) + np.ptp(df['energy']) * 0.05)],
                          xaxis_range=[max(0, np.amin(df['packing_coefficient']) * 0.95),
                                       min(1, np.amax(df['packing_coefficient']) * 1.05)],
                          )

        fig.update_layout(
            xaxis_title=r"Packing coefficient",
            yaxis_title=r"UMA Lattice Energy, (kJ/mol)",
        )
        fig.update_traces(
            marker=dict(
                size=5,
                line=dict(width=0.3, color='rgba(0,0,0,0.3)'),
                opacity=opacity,
            )
        )
        if color_tag == 'Point Density':
            fig.update_layout(coloraxis_colorbar=dict(
                title="Point Density",
                tickfont=dict(size=18),
                title_font=dict(size=18),
            ))
        fig.update_traces(selector=dict(type='violin'), spanmode='hard')
        fig.update_traces(selector=dict(type='violin'), line=dict(width=0.6, color='black'))
        fig.update_layout(
            font=dict(family="Helvetica", size=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=40, b=50),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.15)',
            gridwidth=0.8,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True,
            row=1, col=1  # target the main scatter subplot
        )


    def staircase():

        space = 'real'
        mode = 'contour'
        cmap = 'icefire'
        nbins = 25
        colorbar = False

        labels = sample_batch._build_feature_labels()
        samples = sample_batch._get_samples(space)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()
        N, D = samples.shape

        # Create D×D subplots (upper triangle empty)
        fig = make_subplots(
            rows=D, cols=D,
            horizontal_spacing=0.005, vertical_spacing=0.005,
            shared_xaxes=True, shared_yaxes=True,
        )

        # Loop over lower triangle
        for i in range(D):
            for j in range(D):
                if j >= i:
                    continue  # keep lower triangle only

                x = samples[:, j]
                y = samples[:, i]

                if mode == 'contour':
                    trace = go.Histogram2dContour(
                        x=x, y=y,
                        ncontours=32,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                        contours=dict(coloring='fill', showlines=False, start=0, end=None, size=None),
                        line=dict(smoothing=0.85, width=0),
                        nbinsx=nbins,
                        nbinsy=nbins,
                    )
                elif mode == 'heatmap':
                    trace = go.Histogram2d(
                        x=x, y=y,
                        nbinsx=nbins, nbinsy=nbins,
                        colorscale=cmap,
                        showscale=colorbar and (i == D - 1 and j == 0),
                    )
                else:
                    raise ValueError("mode must be 'contour' or 'heatmap'")
                fig.add_trace(trace, row=i + 1, col=j + 1)

        for i in range(D):
            fig.update_xaxes(title_text=labels[i], row=D, col=i + 1)
            fig.update_yaxes(title_text=labels[i], row=i + 1, col=1)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            # height=1000,
            # width=1000,
            showlegend=False,
        )
        fig.update_layout(
            font=dict(family="Helvetica", size=20),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=30, r=30, t=20, b=30),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1, nticks=4, tickfont_size=16)
        fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1, nticks=4, tickfont_size=16)


    def cell_params():


        space = 'real'
        quantiles = None
        ref_dist = data_batch.full_cell_parameters()
        aux_dists = None
        split_by_sg = False
        split_by_zp = False
        override_energy=sample_energy
        n_kde=200
        bw_factor=0.05
        lattice_features = sample_batch._build_feature_labels()
        samples = sample_batch._get_samples(space)
        num_dists, dist_names, dists = sample_batch._collect_sample_dists(samples, ref_dist, quantiles, split_by_sg,
                                                                          split_by_zp, aux_dists, override_energy)
        # delete or NaN unused higher Z' elements
        # 1d Histograms
        lattice_features[0] = "a Length (Å)"
        lattice_features[1] = "b Length (Å)"
        lattice_features[2] = "c Length (Å)"
        dist_names[0] = "Prior"
        dist_names[1] = "Generator Samples"
        fig = make_subplots(rows=2 + 2 * sample_batch.max_z_prime,
                            cols=3,
                            subplot_titles=lattice_features)
        colors = sample_batch._get_color_set(num_dists)
        data_ranges = []
        for di in range(num_dists):
            data_ranges.append(sample_batch._set_cell_ranges(space, dists[di]))

        ranges = data_ranges[0]
        if num_dists > 1:
            for di in range(num_dists): # extract the widest range of the dists under study
                for key in data_ranges[di].keys():
                    if data_ranges[di][key][0] < ranges[key][0]:
                        ranges[key][0] = data_ranges[di][key][0]
                    if data_ranges[di][key][1] > ranges[key][1]:
                        ranges[key][1] = data_ranges[di][key][1]

        for i in range(len(lattice_features)):
            for j in range(num_dists):
                sample_batch._add_violin(
                    fig, dists[j][:, i], dist_names[j], colors[j], i, ranges[i],
                    n_kde, bw_factor
                )

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          violinmode='overlay',
                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.05,
                              xanchor="center",
                              x=0.5,
                              bgcolor='rgba(0,0,0,0)',

                          ),
                          margin=dict(l=40, r=20, t=50, b=50),
                          font=dict(family="Helvetica", size=12, color='black'),
                          )

        for i in range(6 + sample_batch.max_z_prime * 6):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.update_xaxes(range=ranges[i], row=row, col=col)
        fig.update_xaxes(
            showgrid=False, zeroline=False, ticks='outside',
            tickwidth=1, mirror=True
        )
        fig.update_layout(font_size=20)
        fig.update_annotations(font_size=20)
        fig.update_traces(opacity=0.5)

    def crystals_vis():

        # pick exemplars
        rep_inds = []
        mean_energies = []
        best_energies = []
        for clust in np.unique(clust_labels):
            clust_inds = np.argwhere(clust_labels == clust).flatten()
            ens = sample_energy[clust_inds]
            best_ind = np.argmin(ens)
            rep_inds.append(int(clust_inds[best_ind]))
            mean_energies.append(sample_energy[clust_inds].mean())
            best_energies.append(sample_energy[best_ind])

        # visualize crystals
        rep_batch = collate_data_list([gen_samples[ind] for ind in rep_inds])
        rep_clust = rep_batch.mol2cluster()
        rep_clust.visualize(mode='unit cell',cutoff=4)
        clust, p = np.unique(clust_labels, return_counts=True)


    if False:  # energy_function == 'uma':
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

    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in masks[:10] if
                                                   sum(m) > 1])

    end = 1


'''
# other analyses
    
    
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
