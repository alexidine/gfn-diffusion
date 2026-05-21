"""analysis and figures for the results section of the paper"""
from copy import copy

import torch
from ase.io import write

from energy_sampling.eval.paper1_results.figures import parity_fig, dual_energy_marginal_fig, \
    pes_cartoon, plot_dual_density_contour, combo_fig
from energy_sampling.eval.paper1_results.utils import combo_fig_analysis
from energy_sampling.eval.paper1_results.utils import generator_reward
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor


def do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path):
    uma_results = torch.load(uma_results_path, weights_only=False)
    elj_results = torch.load(elj_results_path, weights_only=False)

    cut_ind = 0  # only doing one now #np.argwhere(uma_results['n_clusts'] == 4).flatten()[0]
    uma_thermos = uma_results['metrics'][cut_ind]

    pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
    predictor = init_uma_crystal_predictor(pred_path, device='cuda')
    exp_crystals = torch.load(exp_path, weights_only=False)
    ebatch = collate_data_list(exp_crystals, max_z_prime=1)
    with torch.no_grad():
        ebatch.cuda()
        ebatch.analyze(['elj', 'uma', 'rdf'], elementwise=False, atomwise=True, assign_outputs=True,
                       predictor=predictor)
        ebatch.cpu()
    num_polymorphs = ebatch.num_graphs
    eparams = ebatch.latent_params()
    eparams = eparams.repeat(1000, 1)
    eparams += torch.randn_like(eparams) * 0.01

    """
    generate figures
    """
    fig_dict = {}
    '''TB plot'''
    prior_data = torch.load(prior_path, weights_only=False)
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
    prior_data = torch.load(prior_path, weights_only=False)
    en_scaling_factor = prior_data['thermal_scaling_factor']
    uma_en = uma_batch.uma
    elj_en = elj_batch.elj * en_scaling_factor

    uma_cp = uma_batch.packing_coeff
    elj_cp = elj_batch.packing_coeff
    '''Density funnel overlay'''
    combo_en = torch.cat([elj_en, uma_en])
    fig_dict['density_contours'] = plot_dual_density_contour(elj_cp, elj_en.clip(max=0), uma_cp, uma_en.clip(max=0),
                                                             label_a="LJ",
                                                             label_b="UMA",
                                                             ncontours=8,
                                                             max_y_quantile=0.99,
                                                             x_min=0.55, x_max=0.9,
                                                             y_min=combo_en.amin() - 5, y_max=combo_en.quantile(0.95),
                                                             bw=0.1,
                                                             show=False,
                                                             return_fig=True,
                                                             yaxis_title="Lattice Energy (kJ/mol)",
                                                             xaxis_title="Packing Coefficient"
                                                             )

    '''Energy marginals'''
    fig_dict['energy_marginal'] = dual_energy_marginal_fig(elj_en, uma_en,
                                                           "LJ"
                                                           "UMA"
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
    marginals_fig.data[2].name = "Experimental"
    subplot_labels = [
        '(a) Normed a',
        '(b) Normed b',
        '(c) Normed c',
        '(d) α',
        '(e) β',
        '(f) γ',
        '(g) u',
        '(h) v',
        '(i) w',
        '(j) θ',
        '(k) φ',
        '(l) r',
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
    (basin_colorscale, basin_min_batch, indexed_cluster_labels,
     n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
     polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
     sample_energy, sample_inds, stats) = combo_fig_analysis(
        ebatch, elj_batch, elj_results, num_polymorphs, uma_batch, uma_results, uma_thermos, 'uma')

    fig_dict['cluster_analysis'] = combo_fig(
        num_polymorphs,
        n_basins,
        packing_coeffs,
        stats,
        uma_thermos,
        sample_inds,
        sample_embedding,
        p_maxima,
        polymorph_inds,
        new_min_inds,
        sample_energy,
        polymorph_colorscale,
        sample_colors,
        indexed_cluster_labels,
        basin_colorscale,
        basin_min_batch,
        polymorph_basin_index,
        energy_function='uma'
    )
    # fig_dict['cluster_analysis'].show()

    "save the top crystals"
    esamples = ebatch.batch_to_list()
    samples = uma_results['sample_batch'].batch_to_list() + esamples
    cbatch = collate_data_list([samples[ind] for ind in p_maxima])
    clustbatch = cbatch.mol2cluster(cutoff=2)
    mols = []

    for ind in range(cbatch.num_graphs):
        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='convolve with')
        write(f"{esamples[0].identifier}_poly_{ind}.png", mol)

        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='unit cell')

        cif_path = f"{samples[0].identifier}_ucell_{ind}.cif"
        mol.write(cif_path)

        with open(cif_path, 'r') as f:
            content = f.read()
        content = content.replace('data_image0', f'data_{samples[0].identifier}_ucell_{ind}', 1)
        with open(cif_path, 'w') as f:
            f.write(content)
    #
    # clustbatch.visualize(mode='convolve with')
    # clustbatch.visualize(mode='unit cell')

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
        if key == 'summary_table' or key == 'polymorph_table':
            dd = {
                'font_size': 16,
                'annotations_font_size': 16,
                'scale': 4,
                'width': 800,
                'height': 1100,
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
                'height': 800,
                'font_size': 32,
                'annotations_font_size': 32,
                'scale': 3,
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
                'width': 1000,
                'font_size': 30,
                'annotations_font_size': 30,
                'scale': 3,
            }
        if key == 'energy_marginal':
            dd = {
                'height': 500,
                'width': 1000,
                'font_size': 30,
                'scale': 3,
                'annotations_font_size': 30,
            }
        if key == 'cluster_analysis':
            dd = {
                'width': 1500,
                'height': 800,
                'annotations_font_size': 24,
                'font_size': 24,
                'legend_font_size': 16,
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
        fig.write_image(rf'C:\Users\mikem\OneDrive\NYU\CSD\papers\generator\{mol_name}_{key}.png',
                        width=fig.layout.width, height=fig.layout.height, scale=scale)

    aa = 1


if __name__ == '__main__':
    """
    load up elj distribution, uma distribution, and experimental polymorphs
    """
    mol_name = 'mipcas'
    exp_path = r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt"
    uma_results_path = r"D:\crystal_datasets\gfn_results\mipcas_uma.pt"
    elj_results_path = r"D:\crystal_datasets\gfn_results\mipcas_elj.pt"
    prior_path = r"D:\crystal_datasets\mipcas\mipcas_elj_prior_dataset.pt"
    do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path)

    mol_name = 'nehzor'
    exp_path = r"D:\crystal_datasets\nehzor\NEHZOR_structures_std_conf.pt"
    uma_results_path = r"D:\crystal_datasets\gfn_results\nehzor_uma.pt"
    elj_results_path = r"D:\crystal_datasets\gfn_results\nehzor_elj.pt"
    prior_path = r"D:\crystal_datasets\nehzor\nehzor_elj_prior_dataset.pt"
    do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path)

    aa = 1
