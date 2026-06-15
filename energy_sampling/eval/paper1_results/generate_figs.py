"""analysis and figures for the results section of the paper"""
from copy import copy

import torch
from ase.io import write
from tqdm import tqdm

from energy_sampling.eval.paper1_results.figures import parity_fig, dual_energy_marginal_fig, \
    pes_cartoon, plot_dual_density_contour, combo_fig
from energy_sampling.eval.paper1_results.utils import combo_fig_analysis, augment_dmat2, clustering, clustering3
from energy_sampling.eval.paper1_results.utils import generator_reward
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


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
        ebatch.analyze(['elj', 'uma', 'rdf'], rdf_mode='atomwise', assign_outputs=True,
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

    fig_dict['duo_embedding'] = duo_embedding_fig(ebatch, elj_batch, elj_results, en_scaling_factor, uma_batch,
                                                  uma_results)

    # todo compute a summary statistic that makes this metric real/meaningful
    # todo consider then maybe whether to combine these as well

    '''Big 'ol combo plot'''
    cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering(uma_results, uma_thermos, max_n_clusters=6,min_basin_size=100)
    #cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering3(uma_results, uma_thermos, max_n_clusters=6,min_basin_size=100)
    # cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering2(uma_results, uma_thermos, n_clusters=10, n_keep=6)

    (basin_colorscale, basin_min_batch, indexed_cluster_labels,
     n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
     polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
     sample_energy, sample_inds, stats) = combo_fig_analysis(
        ebatch, elj_batch, elj_results, num_polymorphs, uma_batch, uma_results, 'uma', cluster_labels, p_maxima,
        n_basins, indexed_cluster_labels)

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


def duo_embedding_fig(ebatch, elj_batch, elj_results, en_scaling_factor, uma_batch, uma_results):
    subsamp = 5000
    dmat_a = uma_results['dmat'].clone()
    dmat_b = elj_results['dmat'].clone()
    rdf_a = uma_batch.rdf.clone()
    rdf_b = elj_batch.rdf.clone()
    sample_energy = uma_batch['uma']
    good_inds = (sample_energy < sample_energy.amin() + 15 * 2.5).argwhere().flatten()
    rdf_a = rdf_a[good_inds]
    sample_energy = elj_batch['elj'] * en_scaling_factor
    good_inds = (sample_energy < sample_energy.amin() + 15 * 2.5).argwhere().flatten()
    rdf_b = rdf_b[good_inds]
    rdf_a = rdf_a[:subsamp]
    rdf_b = rdf_b[:subsamp]
    dmat_a = dmat_a[:subsamp][:, :subsamp]
    dmat_b = dmat_b[:subsamp][:, :subsamp]
    n_a, n_b = dmat_a.shape[0], dmat_b.shape[0]
    assert dmat_a.shape == (n_a, n_a) and dmat_b.shape == (n_b, n_b)
    assert rdf_a.shape[0] == n_a and rdf_b.shape[0] == n_b
    device = 'cuda'
    dtype = torch.float32
    bins = torch.linspace(0, 10, rdf_a.shape[-1], device=device)
    rdf_a = rdf_a.to(device=device, dtype=dtype)
    rdf_b = rdf_b.to(device=device, dtype=dtype)
    a_vs_b = torch.empty(n_a, n_b, device=device, dtype=dtype)
    chunk = 32
    min_chunk = 1
    i = 0
    pbar = tqdm(total=n_a, desc="A vs B")
    while i < n_a:
        j = min(i + chunk, n_a)
        try:
            a_vs_b[i:j] = compute_rdf_distance(
                rdf_a[i:j, None],
                rdf_b[None],
                bins,
            )
            pbar.update(j - i)
            i = j
        except torch.cuda.OutOfMemoryError:
            if chunk <= min_chunk:
                raise
            torch.cuda.empty_cache()
            chunk = max(min_chunk, chunk // 2)
            pbar.write(f"OOM at chunk={j - i}, reducing to {chunk}")
    pbar.close()
    a_vs_b = a_vs_b.to('cpu')
    merged = torch.empty(
        n_a + n_b, n_a + n_b,
        device=dmat_a.device, dtype=dmat_a.dtype,
    )
    merged[:n_a, :n_a] = dmat_a
    merged[n_a:, n_a:] = dmat_b.to(device=dmat_a.device, dtype=dmat_a.dtype)
    merged[:n_a, n_a:] = a_vs_b
    merged[n_a:, :n_a] = a_vs_b.T
    merged = augment_dmat2(ebatch.rdf, torch.cat([rdf_a.cpu(), rdf_b.cpu()], dim=0), merged)
    polymorph_inds = torch.arange(ebatch.num_graphs) + len(merged) - ebatch.num_graphs
    import numpy as np
    from umap import UMAP
    umap_model = UMAP(n_components=2, n_neighbors=300, min_dist=0.75,
                      init='spectral', metric='precomputed', low_memory=True,
                      # densmap=True,
                      repulsion_strength=2.0,
                      n_jobs=-1)
    sample_embedding = umap_model.fit_transform(merged.cpu().numpy().astype(np.float32))
    import plotly.graph_objects as go
    uma_embed = sample_embedding[:subsamp]
    lj_embed = sample_embedding[subsamp:-ebatch.num_graphs]
    from scipy.stats import gaussian_kde
    def density_alpha(pts, lo=0.15, hi=0.8, log=False):
        d = gaussian_kde(pts.T)(pts.T)
        if log:
            d = np.log(d + 1e-12)
        d = (d - d.min()) / (d.max() - d.min() + 1e-12)
        return lo + (hi - lo) * d

    uma_a = density_alpha(uma_embed)
    lj_a = density_alpha(lj_embed)
    uma_colors = [f'rgba(65,105,225,{a:.3f})' for a in uma_a]
    lj_colors = [f'rgba(220,20,60,{a:.3f})' for a in lj_a]
    fig = go.Figure()
    fig.add_scatter(x=uma_embed[:, 0], y=uma_embed[:, 1], mode='markers', name='UMA', marker_color=uma_colors)
    fig.add_scatter(x=lj_embed[:, 0], y=lj_embed[:, 1], mode='markers', name='LJ', marker_color=lj_colors)
    poly_x = np.atleast_1d(sample_embedding[polymorph_inds, 0])
    poly_y = np.atleast_1d(sample_embedding[polymorph_inds, 1])

    fig.add_scatter(x=poly_x, y=poly_y,
                    mode='markers',
                    marker_symbol='x-thin',
                    marker_color='black',
                    marker_line_color='black',
                    marker_line_width=8,
                    marker_size=28,
                    opacity=1.0,
                    showlegend=False)

    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    fig.update_layout(
        xaxis1_title='CV1', yaxis1_title='CV2',
        font_size=20,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig


if __name__ == '__main__':
    """
    load up elj distribution, uma distribution, and experimental polymorphs
    """
    # mol_name = 'mipcas'
    # exp_path = r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt"
    # uma_results_path = r"D:\crystal_datasets\gfn_results\mipcas_uma.pt"
    # elj_results_path = r"D:\crystal_datasets\gfn_results\mipcas_elj.pt"
    # prior_path = r"D:\crystal_datasets\mipcas\mipcas_elj_prior_dataset.pt"
    # do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path)

    mol_name = 'nehzor'
    exp_path = r"D:\crystal_datasets\nehzor\NEHZOR_structures_std_conf.pt"
    uma_results_path = r"D:\crystal_datasets\gfn_results\nehzor_uma.pt"
    elj_results_path = r"D:\crystal_datasets\gfn_results\nehzor_elj.pt"
    prior_path = r"D:\crystal_datasets\nehzor\nehzor_elj_prior_dataset.pt"
    do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path)

    aa = 1
