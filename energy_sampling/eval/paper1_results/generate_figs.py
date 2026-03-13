"""analysis and figures for the results section of the paper"""
import numpy as np
import plotly.graph_objects as go
from umap import UMAP

from energy_sampling.eval.paper1_results.figures import parity_fig, energy_marginal_fig, dual_energy_marginal_fig, \
    rdf_embedding_fig
from energy_sampling.eval.paper1_results.utils import load_experimental_structure, generator_reward, \
    bottom_up_cluster_w_dmat
import torch

from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.dataset_utils.data_class_methods.crystal_ops import plot_dual_density_contour
from mxtaltools.dataset_utils.utils import collate_data_list

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
                                    "log(Pb) + log(R)", "log(Pf) + log(Z)",
                                    quantile_cut=0.01)

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
                                    "log(Pb) + log(R)", "log(Pf) + log(Z)",
                                    quantile_cut=0.01)

'''Density funnel overlay'''
uma_batch = uma_results['sample_batch'].clone()
elj_batch = elj_results['sample_batch'].clone()
prior_data = torch.load(r"D:\crystal_datasets\mipcas\mipcas_elj_prior_dataset.pt", weights_only=False)
en_scaling_factor = prior_data['thermal_scaling_factor']
uma_en = uma_batch.uma
elj_en = elj_batch.elj * en_scaling_factor

uma_cp = uma_batch.packing_coeff
elj_cp = elj_batch.packing_coeff
fig_dict['density_contours'] = plot_dual_density_contour(uma_cp, uma_en.clip(max=0), elj_cp, elj_en.clip(max=0),
                                                         label_a="UMA", label_b="LJ", ncontours=8,
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

uma_batch = uma_results['sample_batch']
elj_batch = elj_results['sample_batch']
marginals_fig = uma_batch.plot_batch_cell_params(
    space='latent',
    ref_dist=elj_batch.latent_params(),
    aux_dists=[eparams],
    return_fig=True,
    show=False
)  # todo update formatting and legend
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
# add here the callout samples
# then, add callout samples
# always take experimental polymorphs

uma_rdf = uma_batch.rdf
rdf_bins = torch.linspace(0,100, uma_rdf.shape[-1])
with torch.no_grad():
    dmat = compute_rdf_distmat(uma_rdf.cuda(), rdf_bins.cuda()).cpu()

umap_model = UMAP(n_components=2, n_neighbors=100, min_dist=1,
                  init='pca', metric='precomputed', low_memory=True, n_jobs=-1)
sample_embedding = umap_model.fit_transform(dmat.cpu().numpy().astype(np.float32))
uma_free_energy = -2.5 * np.log(uma_results['density'] + np.quantile(uma_results['density'], 0.01))[:-1]

fig = rdf_embedding_fig(sample_embedding, uma_en, uma_free_energy)
callout_samples = []
polymorph_inds = [len(uma_results['density']) - 1 - ind for ind in range(ebatch.num_graphs)]
callout_samples.extend(polymorph_inds)
c_inds = bottom_up_cluster_w_dmat(uma_batch.uma, 3, uma_batch.uma.quantile(0.25), 100, 'cpu', dmat)
callout_samples.extend(c_inds[:10])
callout_samples = list(set(callout_samples))
fig.add_scatter(x=sample_embedding[c_inds, 0], y=sample_embedding[c_inds, 1],
                mode='markers', marker_color='white', marker_line_color='black', marker_line_width=4, marker_size=12)
fig.add_scatter(x=sample_embedding[polymorph_inds, 0], y=sample_embedding[polymorph_inds, 1],
                mode='markers', marker_color='white', marker_line_color='black',marker_line_width=4, marker_size=16)




fig_dict['embedding_fig'] = fig


for fig in fig_dict.values():
    fig.show()

aa = 1
