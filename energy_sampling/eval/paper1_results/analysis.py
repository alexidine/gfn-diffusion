import os

import numpy as np
import torch

from energy_sampling.eval.paper1_results.figures import sparkbar_table
from energy_sampling.eval.paper1_results.utils import generator_reward, new_local_analysis, sample_and_analyze, \
    load_experimental_structure
from energy_sampling.utils import load_yaml, dict2namespace
from mxtaltools.analysis.crystal_rdf import rdf_radial_graph
from mxtaltools.dataset_utils.utils import collate_data_list

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def basins_table(results_dir, run_name, sample_metrics, sample_inds):
    display_names = {
        'energy': 'E',
        'density': 'ρ<sub>Emin</sub>',
        'local_max_density': 'ρ<sub>max</sub>',
        'local_dist_mean': 'd̄',
        'local_dist_var': 'σ²(d)',
        'local_en_mean': 'Ē',
        'local_en_var': 'σ²(E)',
    }
    metric_keys = ['energy',
                   'density', 'local_max_density',
                   'local_dist_mean', 'local_dist_var',
                   'local_en_mean', 'local_en_var']
    sparkbar_table(
        {display_names[k]: sample_metrics[k][sample_inds] for k in metric_keys},
        save_path=os.path.join(results_dir, run_name + '_basins_table.png'))


def vars_table(results_dir, run_name, sample_metrics, sample_inds):
    group_labels = {
        'Cell Lengths': ['a', 'b', 'c'],
        'Cell Angles': ['α', 'β', 'γ'],
        'Mol. Centroid': ['u', 'v', 'w'],
        'Mol. Orientation': ['θ', 'φ', 'r'],
    }
    dim_names = []
    for dims in group_labels.values():
        dim_names.extend(dims)
    # Map var0..var11 to display names
    var_display = {f'var{i}': f'σ²({dim_names[i]})' for i in range(12)}
    group_colors = {
        'Cell Lengths': 'rgba(70, 130, 180, 0.12)',
        'Cell Angles': 'rgba(180, 120, 70, 0.12)',
        'Mol. Centroid': 'rgba(70, 180, 100, 0.12)',
        'Mol. Orientation': 'rgba(160, 70, 180, 0.12)',
    }
    # Build color per column
    col_colors = {}
    for group, dims in group_labels.items():
        for d in dims:
            col_colors[f'σ²({d})'] = group_colors[group]
    metric_keys = [f'var{i}' for i in range(12)]
    columns = {var_display[k]: sample_metrics[k][sample_inds] for k in metric_keys}
    sparkbar_table(columns,
                   title='Latent Variance by Dimension',
                   col_colors=col_colors,
                   show_values=True,
                   save_path=os.path.join(results_dir, run_name + '_var_table.png'))


if __name__ == '__main__':

    config = dict2namespace(load_yaml('analysis.yaml'))
    results_path = os.path.join(config.results_dir, rf"{config.run_name}.pt")
    basins_path = os.path.join(config.results_dir, config.run_name + '_basins.pt')

    "Load Relevant Dataset"
    molecule = torch.load(config.molecule_path, weights_only=False)
    dataset_file = torch.load(config.dataset_path, weights_only=False)
    dataset = dataset_file['prior_batch'].batch_to_list()
    en_scaling_factor = dataset_file['thermal_scaling_factor']
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])

    "load presampled results"
    if config.reload_results and os.path.exists(results_path):
        results_dict = torch.load(results_path, weights_only=False)
    else:
        results_dict = sample_and_analyze(config.config_path,
                                          config.model_path,
                                          config.device,
                                          config.num_samples,
                                          max_z_prime,
                                          config.n_steps,
                                          config.batch_size,
                                          config.sg_ind,
                                          config.zp,
                                          config.energy_function,
                                          molecule,
                                          config.save_results,
                                          results_path,
                                          config.overwrite_results)
    samples = results_dict['samples']

    "analyze experimental samples"
    if config.exp_sample_path is not None:
        esamples = load_experimental_structure(config.exp_sample_path,
                                               config.sg_ind,
                                               config.zp,
                                               config.device, max_z_prime, molecule,
                                               config.energy_function)
        samples = samples + esamples

    """get sample info"""
    sample_batch = collate_data_list(samples)
    if hasattr(sample_batch, 'elj'):  # appropriately rescale energy
        sample_batch.elj *= en_scaling_factor
    sample_energy = sample_batch[config.energy_function]
    sample_cp = sample_batch.packing_coeff
    rewards = generator_reward(
        sample_batch,
        None,
        max_z_prime,
        energy_function=config.energy_function,
        temperature=config.kT,
        energy_clip=None
    )

    """start neighborhood analysis"""
    d_metrics = []
    d_cuts = [0.6]
    for d_cut in d_cuts:
        with torch.no_grad():
            bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1], device='cuda')
            neighbor_lists, neighbor_dists = rdf_radial_graph(
                sample_batch.rdf,
                'cuda',
                d_cut=d_cut,
                bins=bins,
                chunk_size=24
            )
            sample_metrics = new_local_analysis(sample_batch,
                                                sample_energy,
                                                neighbor_lists,
                                                neighbor_dists,
                                                samples_to_analyze=np.arange(sample_batch.num_graphs),
                                                d_kernel=d_cut / 3,
                                                e_cut=config.alpha * config.kT,
                                                )
            ndims = sample_batch.latent_params().shape[-1]
            sample_metrics.update({'energy': sample_energy})
            var = sample_metrics['var']
            for ind in range(ndims):
                sample_metrics.update({f'var{ind}': np.nan_to_num(var[:, ind]) / np.ptp(np.nan_to_num(var[:, ind]))})
            dnorm = sample_metrics['density'].sum()
            sample_metrics['local_mean_density'] /= dnorm
            sample_metrics['local_max_density'] /= dnorm
            sample_metrics['density'] /= dnorm

            d_metrics.append(sample_metrics)

    results_dict.update(d_metrics[-1])
    torch.save(results_dict, results_path)
    aa = 1
    #
    # for sample_metrics, d_cut in zip(d_metrics, d_cuts):
    #     # basin_mask_energy, basin_assignments_energy = assign_basins(sample_metrics['local_energy_minimum_id'],
    #     #                                                             sample_batch.num_graphs)
    #     # basin_mask_density, basin_assignments_density = assign_basins(sample_metrics['local_density_maximum_id'],
    #     #                                                               sample_batch.num_graphs)
    #
    #     minima_inds = np.argwhere(sample_metrics['is_local_en_minimum']).flatten()
    #     sorted_minima_inds = minima_inds[np.argsort(sample_energy[minima_inds])]
    #     maxima_inds = np.argwhere(
    #         sample_metrics['is_local_density_maximum'] & (sample_energy < np.median(sample_energy)).numpy()).flatten()
    #     sorted_maxima_inds = maxima_inds[np.argsort(sample_metrics['density'][maxima_inds])][::-1]
    #     target_ind = sample_batch.num_graphs - 1
    #     sample_inds = np.concatenate([np.ones(1).astype(np.long) * target_ind, sorted_minima_inds[:8]])
    #     # sample_inds = np.concatenate([sorted_minima_inds[:12], sorted_maxima_inds[:12]])  # add experimental state
    #
    #     basins_table(config.results_dir, config.run_name, sample_metrics, sample_inds)
    #     vars_table(config.results_dir, config.run_name, sample_metrics, sample_inds)
    #
    # fig_dict = {}
    # fig_dict = general_figs(fig_dict,
    #                         sample_batch,
    #                         sample_energy,
    #                         'kJ/mol')
    #
    # x = results_dict['log_pbs'] + rewards[:-1]
    # y = results_dict['log_pfs'] + results_dict['learned_log_z']
    # fig_dict['TB_fig'] = parity_fig(x, y, "Pb + R", "Pf + Z")
    #
    # """rdf embedding"""
    # from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
    #
    # dmat = compute_rdf_distmat(sample_batch.rdf.cuda(), bins.cuda())
    # umap_model = UMAP(n_components=2, n_neighbors=500, min_dist=0.5,
    #                   init='pca', metric='precomputed', low_memory=True, n_jobs=-1)
    # sample_embedding = umap_model.fit_transform(dmat.cpu().numpy().astype(np.float32))
    #
    # basin_mask_energy, basin_assignments_energy = assign_basins(d_metrics[-1]['local_energy_minimum_id'], sample_batch.num_graphs)
    # #basin_mask_density, basin_assignments_density = assign_basins(d_metrics[-1]['local_density_maximum_id'],sample_batch.num_graphs)
    # en_basin_inds = np.unique(basin_assignments_energy)
    # en_basin_inds = en_basin_inds[en_basin_inds >= 0]
    #
    # den_basin_inds = np.zeros_like(en_basin_inds)
    # for ind in range(len(en_basin_inds)):
    #     local_densities = d_metrics[-1]['density'][basin_mask_energy[ind]]
    #     max_den = local_densities.argmax()
    #     den_basin_inds[ind] = basin_mask_energy[ind].argwhere().flatten()[max_den]
    #
    # basins = basin_mask_energy
    # basin_sort_inds = torch.argsort(basins.sum(-1), descending=True).flatten()
    # n_basins = 10
    # top_basins = basins[basin_sort_inds][:n_basins]
    # en_basin_inds = en_basin_inds[basin_sort_inds][:n_basins]
    # den_basin_inds = den_basin_inds[basin_sort_inds][:n_basins]
    #
    # fig = go.Figure()
    # fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], marker_color='grey', mode='markers',
    #                 opacity=0.5, marker_size=6)
    # for c_ind, basin in enumerate(top_basins):
    #     fig.add_scatter(x=sample_embedding[basin, 0], y=sample_embedding[basin, 1], mode='markers', opacity=0.65,
    #                     name=f"{c_ind} : {basin.sum().item()}")
    #
    # # Energy minima — big stars
    # fig.add_scatter(x=sample_embedding[en_basin_inds, 0],
    #                 y=sample_embedding[en_basin_inds, 1],
    #                 mode='markers+text', name='E minima',
    #                 text=[str(i) for i in range(10)], textposition='top center',
    #                 marker=dict(size=16, symbol='star', color='black', line=dict(width=1, color='white')))
    #
    # # Density maxima — big diamonds
    # fig.add_scatter(x=sample_embedding[den_basin_inds, 0],
    #                 y=sample_embedding[den_basin_inds, 1],
    #                 mode='markers+text', name='ρ maxima',
    #                 text=[str(i) for i in range(10)], textposition='top center',
    #                 marker=dict(size=14, symbol='diamond', color='red', line=dict(width=1, color='white')))
    #
    # fig.add_scatter(x=sample_embedding[target_ind:target_ind + 1, 0],
    #                 y=sample_embedding[target_ind:target_ind + 1, 1],
    #                 mode='markers', name='Experimental Polymorph',
    #                 marker=dict(size=18, symbol='x', color='green',
    #                             line=dict(width=2, color='darkgreen')))
    #
    # fig_dict['embeddings'] = fig
    # for key, fig in fig_dict.items():
    #     fig.show()
    #     fig.write_image(os.path.join(config.results_dir, config.run_name + f'_{key}.png'))
    #
    # aa = 1

    #
    # X = sample_batch.latent_params()
    # from umap import UMAP
    #
    # umap_model = UMAP(n_components=2, n_neighbors=50, min_dist=0.01,
    #                   init='pca', metric='euclidean', low_memory=True, n_jobs=-1)
    # sample_embedding = umap_model.fit_transform(X.numpy().astype(np.float32))
    #
    #
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # basins = basin_mask_energy
    # top_basins = basins[torch.argsort(basins.sum(-1), descending=True).flatten()][:10]
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # fig = go.Figure()
    # fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], marker_color='grey', mode='markers', opacity=0.5)
    # for c_ind, basin in enumerate(top_basins):
    #     fig.add_scatter(x=sample_embedding[basin, 0], y=sample_embedding[basin, 1], mode='markers',
    #                     name=f"{c_ind} : {basin.sum().item()}")
    # fig.show()
    #
    # top_basins = basins[torch.argsort(basins.sum(-1), descending=True).flatten()][:10]
    # sample_batch.plot_batch_cell_params(space='latent',
    #                                     aux_dists=[sample_batch.latent_params()[bas] for bas in top_basins]
    #                                     )
    #
    # """rdf embedding"""
    # from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
    #
    # dmat = compute_rdf_distmat(sample_batch.rdf.cuda(), bins.cuda())
    # umap_model = UMAP(n_components=2, n_neighbors=50, min_dist=0.01,
    #                   init='pca', metric='precomputed', low_memory=True, n_jobs=-1)
    # sample_embedding = umap_model.fit_transform(dmat.cpu().numpy().astype(np.float32))
    #
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # basins = basin_mask_energy
    # top_basins = basins[torch.argsort(basins.sum(-1), descending=True).flatten()][:10]
    # import plotly.graph_objects as go
    # from plotly.subplots import make_subplots
    #
    # fig = go.Figure()
    # fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], marker_color='grey', mode='markers', opacity=0.5)
    # for c_ind, basin in enumerate(top_basins):
    #     fig.add_scatter(x=sample_embedding[basin, 0], y=sample_embedding[basin, 1], mode='markers',
    #                     name=f"{c_ind} : {basin.sum().item()}")
    # fig.show()
    #
    #
    # n_cols = 3
    # keys = list(sample_metrics.keys())
    # keys = ['energy', 'count', 'density', 'local_dist_mean', 'local_dist_var', 'local_laplacian']
    # n_rows = len(keys) // n_cols + int((len(keys) % n_cols != 0))
    # fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=keys)
    # good_inds = sample_energy < sample_energy.median()
    # for ind, key in enumerate(keys):
    #     if key != 'var':
    #         row = ind // n_cols + 1
    #         col = ind % n_cols + 1
    #         fig.add_scatter(x=sample_embedding[:, 0][good_inds], y=sample_embedding[:, 1][good_inds],
    #                         marker_color=sample_metrics[key][good_inds],
    #                         mode='markers', opacity=0.5, row=row, col=col, marker_colorscale='plasma',
    #                         showlegend=False)
    # fig.show()
    #
    # #
    #
    # d_cut = 3.5
    # sample_metrics = new_local_analysis(d_cut,  # tuned d_cut
    #                                     sample_batch,
    #                                     sample_energy,
    #                                     rmsdmat)
    #

    # minima_inds = np.argwhere(sample_metrics['is_local_en_minimum']).flatten()
    # sorted_minima_inds = minima_inds[np.argsort(sample_energy[minima_inds])]
    #
    # maxima_inds = np.argwhere(
    #     sample_metrics['is_local_density_maximum'] & (sample_energy < np.median(sample_energy)).numpy()).flatten()
    # sorted_maxima_inds = maxima_inds[np.argsort(sample_metrics['density'][maxima_inds])][::-1]
    #
    # sample_inds = np.concatenate([sorted_minima_inds[:12], sorted_maxima_inds[:12]])  # add experimental state

    # fig = sample_summary_table(sample_metrics, sample_energy, sample_inds)
    # fig_dict[f'thermo_table'] = fig
    # fig = var_table(sample_metrics, sample_energy, sample_inds)
    # fig_dict[f'var_table'] = fig

    """distance calculations & comparisons
    
    n_dist_samples = 250
    asupos = init_asupos(sample_batch)
    dmats = torch.zeros((3, n_dist_samples, n_dist_samples))
    bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1])
    latents = sample_batch.latent_params()[:n_dist_samples]
    rdfs = sample_batch.rdf[:n_dist_samples]
    asuposes = asupos[:n_dist_samples]
    for ind in tqdm(range(n_dist_samples)):
        dmats[0,ind] = compute_asu_rmsd(asuposes, ind)
        dmats[1,ind] = compute_rdf_distance(rdfs, rdfs[ind], bins)
        dmats[2,ind] = simple_latent_distance(latents, latents[None, ind])

    from plotly.subplots import make_subplots

    from mxtaltools.common.utils import get_point_density

    fig = make_subplots(rows=1, cols=4, subplot_titles=['asu vs rdf', 'asu vs latent', 'rdf vs latent'])
    cc = 1
    for i1 in range(3):
        for i2 in range(3):
            if i2 > i1:
                x = dmats[i1].flatten()
                y = dmats[i2].flatten()
                x = x[x > 0].log10()
                y = y[y > 0].log10()
                c = get_point_density(np.stack([x.cpu().numpy(), y.cpu().numpy()]))
                fig.add_scatter(x=x, y=y, marker_color=c, mode='markers', col=cc, row=1)
                cc += 1
    fig.show()
    """
