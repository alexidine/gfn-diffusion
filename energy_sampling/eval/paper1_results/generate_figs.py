"""analysis and figures for the results section of the paper"""
from copy import copy
import plotly.express as px
import numpy as np
import torch
from ase.io import write
from umap import UMAP
import plotly.graph_objects as go

from energy_sampling.eval.paper1_results.figures import parity_fig, dual_energy_marginal_fig, \
    rdf_embedding_fig, polymorph_summary_table, pes_cartoon, plot_dual_density_contour
from energy_sampling.eval.paper1_results.utils import load_experimental_structure, generator_reward
from examples.crystal_search_reporting import batch_compack
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.dataset_utils.utils import collate_data_list

if __name__ == '__main__':
    """
    load up elj distribution, uma distribution, and experimental polymorphs
    """
    uma_results = torch.load(r"D:\crystal_datasets\gfn_results\mipcas_uma.pt", weights_only=False)
    elj_results = torch.load(r"D:\crystal_datasets\gfn_results\mipcas_elj.pt", weights_only=False)
    molecule = torch.load(r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt", weights_only=False)
    esamples = load_experimental_structure(
        r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt",
        2,
        1,
        'cuda', 1, molecule,
        'uma')

    ebatch = collate_data_list(esamples)
    eparams = ebatch.latent_params()
    eparams = eparams.repeat(1000, 1)
    eparams += torch.randn_like(eparams) * 0.01

    """
    generate figures
    """
    fig_dict = {}
    '''TB plot'''
    prior_data = torch.load(r"D:\crystal_datasets\mipcas\mipcas_elj_prior_dataset.pt", weights_only=False)
    en_scaling_factor = prior_data['thermal_scaling_factor']
    elj_batch = elj_results['sample_batch'].clone()
    elj_batch.elj = elj_batch.elj * en_scaling_factor
    rewards = generator_reward(
        elj_batch,
        None,
        1,
        energy_function='elj',
        temperature=2.5,
        energy_clip=None
    )
    x = elj_results['log_pbs'] + rewards
    y = elj_results['log_pfs'] + elj_results['learned_log_z']
    fig_dict['elj_TB_fig'] = parity_fig(x, y,
                                        "log(P<sub>b</sub>) + log(R)", "log(P<sub>f</sub>) + log(Z<sub>θ</sub>)",
                                        quantile_cut=0.99)

    uma_batch = uma_results['sample_batch']
    rewards = generator_reward(
        uma_batch,
        None,
        1,
        energy_function='uma',
        temperature=2.5,
        energy_clip=None
    )
    x = uma_results['log_pbs'] + rewards
    y = uma_results['log_pfs'] + uma_results['learned_log_z']
    fig_dict['uma_TB_fig'] = parity_fig(x, y,
                                    "log(P<sub>b</sub>) + log(R)", "log(P<sub>f</sub>) + log(Z<sub>θ</sub>)",
                                    quantile_cut=0.99)

    uma_batch = uma_results['sample_batch'].clone()
    elj_batch = elj_results['sample_batch'].clone()
    prior_data = torch.load(r"D:\crystal_datasets\mipcas\mipcas_elj_prior_dataset.pt", weights_only=False)
    en_scaling_factor = prior_data['thermal_scaling_factor']
    uma_en = uma_batch.uma
    elj_en = elj_batch.elj * en_scaling_factor

    uma_cp = uma_batch.packing_coeff
    elj_cp = elj_batch.packing_coeff
    '''Density funnel overlay'''
    fig_dict['density_contours'] = plot_dual_density_contour(elj_cp, elj_en.clip(max=0), uma_cp, uma_en.clip(max=0),
                                                             label_a="LJ", label_b="UMA", ncontours=8,
                                                             max_y_quantile=0.95,
                                                             x_min=0.625, x_max=0.9,
                                                             y_min=-145, y_max=-100,
                                                             bw=0.1,
                                                             show=False,
                                                             return_fig=True,
                                                             yaxis_title="Lattice Energy (kJ/mol)",
                                                             xaxis_title="Packing Coefficient"
                                                             )

    '''Energy marginals'''
    fig_dict['energy_marginal'] = dual_energy_marginal_fig(elj_en, uma_en,
                                                           "LJ", "UMA"
                                                           )

    '''12 marginals'''
    uma_batch = uma_results['sample_batch'].clone()
    elj_batch = elj_results['sample_batch'].clone()
    marginals_fig = uma_batch.plot_batch_cell_params(
        space='latent',
        ref_dist=elj_batch.latent_params(),
        aux_dists=[eparams],
        return_fig=True,
        show=False
    )
    marginals_fig.data[1].name = "UMA Model"
    marginals_fig.data[0].name = "LJ Model"
    marginals_fig.data[2].name = "Experimental Structures"
    subplot_labels = [
        '(a) Normed a',
        '(b) Normed b',
        '(c) Normed c',
        '(d) Scaled α',
        '(e) Scaled β',
        '(f) Scaled γ',
        '(g) Aunit frac. x',
        '(h) Aunit frac. y',
        '(i) Aunit frac. z',
        '(j) Scaled θ',
        '(k) Scaled φ',
        '(l) Scaled r',
    ]
    for ii, annotation in enumerate(marginals_fig.layout.annotations):
        annotation.text = subplot_labels[ii]
    fig_dict['marginals_fig'] = marginals_fig

    '''staircase plots'''
    elj_staircase = elj_batch.plot_batch_staircase(space='latent',
                                                   ref_dist=ebatch.latent_params(),
                                                   return_fig=True,
                                                   show=False)
    uma_staircase = uma_batch.plot_batch_staircase(space='latent',
                                                   ref_dist=ebatch.latent_params(),
                                                   return_fig=True,
                                                   show=False)
    fig_dict['elj_staircase'] = elj_staircase
    fig_dict['uma_staircase'] = uma_staircase

    '''Big 'ol combo plot'''

    rdf_bins = torch.linspace(0, 100, uma_batch.rdf.shape[-1])
    with torch.no_grad():
        dmat = compute_rdf_distmat(torch.cat([uma_batch.rdf, ebatch.rdf], dim=0).cuda(), rdf_bins.cuda()).cpu()

    sample_energy = torch.cat([uma_batch.uma, ebatch.uma], dim=0)
    minima_inds = np.argwhere(uma_results['is_local_en_minimum']).flatten()
    sorted_minima_inds = minima_inds[np.argsort(sample_energy[minima_inds])]
    related_maxima = uma_results['local_density_maximum_id'][sorted_minima_inds]
    callout_samples = []
    polymorph_inds = [len(uma_results['density']) - 1 - ind for ind in range(ebatch.num_graphs)]
    callout_samples.extend(polymorph_inds)
    callout_samples.extend(sorted_minima_inds)
    callout_samples = list(set(callout_samples))
    callout_samples = [int(elem) for elem in callout_samples]

    umap_model = UMAP(n_components=2, n_neighbors=100, min_dist=1,
                      init='pca', metric='precomputed', low_memory=True, n_jobs=-1)
    sample_embedding = umap_model.fit_transform(dmat.cpu().numpy().astype(np.float32))
    uma_free_energy = -2.5 * np.log(uma_results['density'])[:len(uma_en)]

    rank_map = {sample_id: rank + 1 for rank, sample_id in enumerate(sorted_minima_inds)}
    basin_inds = np.array([rank_map.get(id, 0) for id in uma_results['local_energy_minimum_id']])
    n_basins = len(np.unique(basin_inds)) - 1
    colorscale = px.colors.qualitative.Set1[:n_basins + 1]
    colorscale[0] = 'rgb(100, 100, 100)'
    basin_colors = [colorscale[i] for i in basin_inds]
    fig_dict['embedding_fig'] = rdf_embedding_fig(sample_embedding, uma_en, uma_free_energy, sorted_minima_inds,
                                                  related_maxima,
                                                  polymorph_inds, basin_colors)

    'top samples analysis'
    clean_callouts = [callout_samples[ind] for ind in range(len(callout_samples)) if
                      callout_samples[ind] not in polymorph_inds]
    # matches, rmsds = batch_compack(clean_callouts, uma_batch.batch_to_list(), ebatch.mol2cluster(cutoff=10))

    uma_results['free_energy'] = uma_free_energy
    analysis_keys = ['sample_energy', 'sample_cp', 'density', 'local_en_mean', 'local_en_var', 'local_max_density',
                     'local_mean_density', 'free_energy']  # ,'local_energy_minimum_id']
    stats = {key:
                 uma_results[key][sorted_minima_inds] for key in analysis_keys
             }
    matches = [20, 10, 8, 7, 3]
    rmsds = [.261, .495, .279, .38, .122]
    fig_dict['summary_table'] = polymorph_summary_table(stats, sorted_minima_inds, polymorph_inds, colorscale, matches,
                                                        rmsds)

    '''
    
    samples = uma_results['sample_batch'].batch_to_list()
    cbatch = collate_data_list([samples[ind] for ind in sorted_minima_inds])
    clustbatch = cbatch.mol2cluster(cutoff=2)
    mols = []
    
    for ind in range(cbatch.num_graphs):
        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='convolve with')
        write(f"{esamples[0].identifier}_poly_{ind}.png", mol)
    
        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='unit cell')
        mol.write(f"{esamples[0].identifier}_ucell_{ind}.cif")
    
    clustbatch.visualize(mode='convolve with')
    '''

    '''pes cartoon'''
    fig_dict['pes_cartoon'] = pes_cartoon()

    """Exports"""

    pub_style = dict(
        font=dict(family="Arial", size=18),
        width=1920,
        height=1080,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations_font_size=18,
        scale=2,
    )


    def custom_style(key):
        dd = {}
        if key == 'pes_cartoon':
            dd = {
                'width': 1200,
                'height': 500
            }
        if key == 'summary_table':
            dd = {
                'font_size': 12,
                'annotations_font_size': 12,
                'scale': 4,
                'width': 900,
                'height': 400,
            }
        if 'staircase' in key:
            dd = {
                'font_size': 18,
                'annotations_font_size': 18,
                'width': 2400,
                'height': 1200,
                'scale': 3
            }
        if key == 'marginals_fig':
            dd = {
                'font_size': 32,
                'annotations_font_size': 32,
            }
        if key == 'embedding_fig':
            dd = {
                'width': 1500,
                'height': 650,
            }
        if 'TB' in key:
            dd = {
                'height': 620,
                'width': 620
            }
        if key == 'density_contours':
            dd = {
                'height': 500,
                'width': 1000
            }
        if key == 'energy_marginal':
            dd = {
                'height': 400,
                'width': 1000
            }
        return dd


    for key, fig in fig_dict.items():
        # fig.show()
        style = copy(pub_style)
        style.update(custom_style(key))
        scale = style['scale']
        style.pop('scale')
        for skey in style.keys():
            if 'annotation' in skey:
                fs = style[skey]
                fig.update_annotations(font_size=fs)
        style = {skey: value for skey, value in style.items() if 'annotation' not in skey}
        fig.update_layout(**style)
        fig.write_image(f'_{key}.png', width=fig.layout.width, height=fig.layout.height, scale=scale)

    aa = 1
