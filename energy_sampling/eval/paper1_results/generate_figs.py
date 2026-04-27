"""analysis and figures for the results section of the paper"""
from copy import copy
from energy_sampling.eval.paper1_results.utils import mean_shift_density

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from ase.io import write
from umap import UMAP

from energy_sampling.eval.paper1_results.figures import parity_fig, dual_energy_marginal_fig, \
    pes_cartoon, plot_dual_density_contour, combo_fig
from energy_sampling.eval.paper1_results.utils import generator_reward
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
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
    sample_energy = uma_batch.uma
    # good_inds = (sample_energy < (sample_energy.amin() + 15*2.5)).argwhere().flatten()
    good_inds = (sample_energy < sample_energy.amin() + 15 * 2.5).argwhere().flatten()
    num_filtered_samples = len(good_inds)

    # good_inds = (sample_energy < (sample_energy.quantile(0.95))).argwhere().flatten()
    sample_energy = uma_batch.uma[good_inds]
    sample_cps = uma_batch.packing_coeff[good_inds]
    #
    cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering(uma_results, uma_thermos)
    e_minima = np.array([(cluster_labels == ind).argwhere().flatten()[np.argmin(sample_energy[cluster_labels == ind])]
                         for ind in p_maxima])  # in order of p_maxima
    basin_opt_ens = sample_energy[e_minima]
    ss = uma_batch.batch_to_list()
    ss = [ss[ind] for ind in good_inds]
    basin_min_batch = collate_data_list([ss[ind] for ind in e_minima])
    # opt_samples = uma_results['basin_opts']
    # opt_batch = collate_data_list(opt_samples)
    # basin_opt_ens = opt_batch.uma.reshape(n_basins, -1)

    assert len(basin_opt_ens) == n_basins, "optimizer output mismatch"

    # basin_mins = torch.argmin(basin_opt_ens, dim=1)
    # basin_min_inds = []
    # samples_per_basin = basin_opt_ens.shape[-1]
    # cursor = 0
    # for ind in range(n_basins):
    #     basin_min_inds.append(basin_mins[ind] + cursor)
    #     cursor += samples_per_basin
    # basin_min_batch = collate_data_list([opt_samples[ind] for ind in basin_min_inds])

    # full_dmat rows:
    # [0 : len(good_inds)]                              → filtered uma samples
    # [len(good_inds) : len(good_inds)+num_polymorphs]  → polymorph references (ebatch)
    # [len(good_inds)+num_polymorphs : ]                → basin minima
    full_dmat = augment_dmat(basin_min_batch, ebatch, uma_batch, uma_results, good_inds)

    assert full_dmat.shape[0] == len(good_inds) + num_polymorphs + n_basins, "full_dmat shape mismatch"

    polymorph_inds = [len(good_inds) + ind for ind in range(num_polymorphs)]
    new_min_inds = [len(good_inds) + num_polymorphs + ind for ind in range(basin_min_batch.num_graphs)]

    umap_model = UMAP(n_components=2, n_neighbors=300, min_dist=0.75,
                      init='pca', metric='precomputed', low_memory=True, n_jobs=-1)
    sample_embedding = umap_model.fit_transform(full_dmat.cpu().numpy().astype(np.float32))

    basin_colorscale = px.colors.qualitative.Vivid[:n_basins + 2]
    basin_colorscale[0] = 'rgb(100, 100, 100)'
    sample_colors = [basin_colorscale[i + 1] for i in indexed_cluster_labels]

    """get e minima and polymorph state probs under uma"""
    d_kernel = uma_results['d_cuts'][0] / 3
    old_dens = (torch.exp(-(full_dmat[:num_filtered_samples, :num_filtered_samples] ** 2) / (2 * d_kernel ** 2)).sum(
        dim=1) - 1)
    dnorm = old_dens.sum()
    new_dens = (torch.exp(-(full_dmat[num_filtered_samples:, :num_filtered_samples] ** 2) / (2 * d_kernel ** 2)).sum(
        dim=1)) / dnorm  # no minus one because we have deleted the self term
    old_dens /= dnorm

    """get polymorph & basin probs under elj"""
    edmat = elj_results['dmat']

    bins = torch.linspace(0, 10, elj_batch.rdf.shape[-1], device='cuda')
    all_new_rdf = torch.cat([uma_batch.rdf[p_maxima], ebatch.rdf, basin_min_batch.rdf], dim=0)
    # New samples vs. all original samples
    dists_to_elj = torch.stack([
        compute_rdf_distance(all_new_rdf[ii], elj_batch.rdf[good_inds], bins.cpu())
        for ii in range(all_new_rdf.shape[0])
    ])  # [n, 10k]

    n_elj_samples = edmat.shape[0]
    old_elj_dens = (torch.exp(-(edmat ** 2) / (2 * d_kernel ** 2)).sum(
        dim=1) - 1)
    dnorm = old_elj_dens.sum()
    new_elj_dens = (torch.exp(-(dists_to_elj ** 2) / (2 * d_kernel ** 2)).sum(
        dim=1)) / dnorm  # no minus one because we have deleted the self term
    old_elj_dens /= dnorm

    new_elj_dens /= old_elj_dens.amax()

    stats = {}
    stats['sample_energy'] = torch.cat([basin_min_batch.uma, ebatch.uma]).numpy()
    stats['sample_cp'] = torch.cat([sample_cps[p_maxima], ebatch.packing_coeff]).numpy()
    stats['density'] = torch.cat(
        [old_dens[p_maxima], new_dens[:num_polymorphs]]).numpy() / old_dens.amax()  # replace with per-basin maxima
    stats['elj_density'] = new_elj_dens[:num_polymorphs + n_basins].numpy()
    # stats['local_en_var'] = uma_thermos['local_en_var'][p_maxima]
    poly_to_basin_dists = torch.zeros((num_polymorphs, n_basins))
    for ind in range(num_polymorphs):
        for b_ind, bb in enumerate(p_maxima):
            binds = (cluster_labels == bb).argwhere().flatten()
            dists = full_dmat[binds, num_filtered_samples + ind]
            poly_to_basin_dists[ind][b_ind] = dists.amin()

    polymorph_basin_index = poly_to_basin_dists.argmin(dim=1)
    polymorph_colorscale = [basin_colorscale[1 + ind] for ind in polymorph_basin_index]

    num_orig_samples = len(good_inds)
    packing_coeffs = sample_cps
    sample_inds = np.arange(num_orig_samples)

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


def clustering(uma_results, uma_thermos):
    """"""

    "get basin anchors"
    dmat = uma_results['dmat']
    d_cuts = uma_results['d_cuts']
    b_rec = []
    b_sz_rec = []
    for cc in torch.linspace(0.33, 2, 50):
        cluster_labels = mean_shift_density(len(dmat), 100, dmat, d_cuts[0] * cc, uma_thermos['density'])

        i, c = np.unique(cluster_labels, return_counts=True)
        b_rec.append(i)
        b_sz_rec.append(c)
        if len(i) <= 4:
            break
    anchors = i

    "assign samples to basins"
    sig = d_cuts[0] / 3
    weights = np.exp(-dmat[anchors] ** 2 / (2 * sig ** 2))
    weights = weights / weights.sum(0)
    assignments = weights.argmax(dim=0)
    assignments[weights.amax(dim=0) < 0.8] = -1
    cluster_labels = torch.tensor([i[ass] if ass > -1 else -1 for ass in assignments])

    p_maxima, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    p_maxima = p_maxima[1:]

    density = np.asarray(uma_thermos['density'])
    order = np.argsort(-density[p_maxima])
    ordered_p_maxima = p_maxima[order]

    label_map = {old.item(): new for new, old in enumerate(ordered_p_maxima, start=1)}
    label_map.update({-1: 0})
    indexed_cluster_labels = torch.tensor([label_map[l.item()] for l in cluster_labels])
    indexed_cluster_labels -= 1
    num_filtered_samples = len(cluster_labels)

    n_basins = len(np.unique(cluster_labels)) - 1

    assert torch.all(cluster_labels[p_maxima] == p_maxima)
    assert len(cluster_labels) == num_filtered_samples, "cluster_labels mismatch"

    return cluster_labels, indexed_cluster_labels, ordered_p_maxima, n_basins


def augment_dmat(basin_min_batch, ebatch, uma_batch, uma_results, good_inds):
    dmat = uma_results['dmat']
    num_polymorphs = ebatch.num_graphs
    bins = torch.linspace(0, 10, uma_batch.rdf.shape[-1], device='cuda')
    all_new_rdf = torch.cat([ebatch.rdf, basin_min_batch.rdf], dim=0)
    # New samples vs. all original samples
    new_vs_original = torch.stack([
        compute_rdf_distance(all_new_rdf[ii], uma_batch.rdf[good_inds], bins.cpu())
        for ii in range(all_new_rdf.shape[0])
    ])  # [n, 10k]
    # New samples vs. each other
    new_vs_new = torch.stack([
        compute_rdf_distance(all_new_rdf[ii], all_new_rdf, bins.cpu())
        for ii in range(all_new_rdf.shape[0])
    ])  # [n, n]
    n_orig = dmat.shape[0]
    n_new = all_new_rdf.shape[0]
    dmat_augmented = torch.zeros(
        n_orig + n_new, n_orig + n_new,
        device=dmat.device, dtype=dmat.dtype
    )
    dmat_augmented[:n_orig, :n_orig] = dmat
    dmat_augmented[n_orig:, :n_orig] = new_vs_original
    dmat_augmented[:n_orig, n_orig:] = new_vs_original.T
    dmat_augmented[n_orig:, n_orig:] = new_vs_new
    return dmat_augmented


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
