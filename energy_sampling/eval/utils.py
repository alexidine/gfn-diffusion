import json
import sys

import numpy as np
import plotly
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from tqdm import tqdm

from energy_sampling.utils import uniform_discretizer, get_gfn_init_state, embed_dataset, logmeanexp
from mxtaltools.dataset_utils.utils import collate_data_list


def sample_crystals(
        generator: str,  # 'generator' or 'random'
        gfn_model,
        batch_size,
        mol_list,
        space_group,
        n_steps,
        samples_per_mol,
        device,
        energy_function,
        encoder=None,
        optim_kwargs=None,
        do_opt: bool = False
):
    """
    :param gfn_model:
    :param batch_size:
    :param mol_list:
    :param space_group:
    :param n_steps:
    :param samples_per_mol:
    :param device:
    :param energy_function:
    :return:
    """

    """
    initialize useful things
    """
    if optim_kwargs is None:
        optim_kwargs = dict(
            optim_target='silu',
            show_tqdm=True,
            lr=1e-4,
            convergence_eps=1e-3,
            compression_factor=0.1,
            max_num_steps=300,
            do_box_restriction=True,
            enforce_niggli=True,
            cutoff=6,
            optimizer_func=torch.optim.Rprop,
        )

    with torch.no_grad():
        discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)

        num_batches = len(mol_list) // batch_size
        if len(mol_list) % batch_size != 0:
            num_batches += 1

        energy_function.space_groups = [space_group]
        init_state = get_gfn_init_state(batch_size, energy_function.data_ndim, device)

        params_record = np.zeros((samples_per_mol, len(mol_list), 12))
        energy_record = np.zeros((samples_per_mol, len(mol_list)))
        density_record = np.zeros_like(energy_record)
        sample_record = []
        if do_opt:
            opt_params_record = np.zeros((samples_per_mol, len(mol_list), 12))
            opt_energy_record = np.zeros((samples_per_mol, len(mol_list)))
            opt_density_record = np.zeros_like(energy_record)
            opt_sample_record = []

        if generator == 'generator':
            """embed the dataset"""
            if hasattr(mol_list[0], 'embedding'):
                if mol_list[0].embedding is not None:
                    pass
                else:
                    mol_list = embed_dataset(mol_list, encoder=encoder)
            else:
                mol_list = embed_dataset(mol_list, encoder=encoder)

        """sample"""

        for s_ind in tqdm(range(samples_per_mol)):
            ssample_record = []
            if do_opt:
                opt_ssample_record = []

            for b_ind in range(num_batches):
                batch_inds = np.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                mol_batch = collate_data_list([mol_list[ind] for ind in batch_inds]).to(device)

                if generator == 'generator':
                    (_, samples, log_r, _, _,
                     _, sample_batch, _, _, _, _,
                     _, _, _, _,
                     log_T_tensor) = sample_eval_fwd_trajs(
                        init_state, gfn_model, discretizer, energy_function, mol_batch)
                # # DEPRECATED
                # elif generator == 'random':
                #     crystal_batch = collate_data_list(
                #         mol_to_blank_crystal_list(mol_batch, [space_group for _ in range(mol_batch.num_graphs)], ))
                #     samples = sample_crystal_prior(crystal_batch, 1)
                #     log_T_tensor = torch.ones(crystal_batch.num_graphs, device=device) * energy_function.temperature
                #     log_r, sample_batch = energy_function.log_reward(
                #         samples, mol_batch=mol_batch,
                #         log_temperature=log_T_tensor,
                #         return_exp=True)

                params_record[s_ind, batch_inds] = samples.cpu().detach().numpy()
                energy_record[s_ind, batch_inds] = sample_batch.lj.cpu().detach().numpy()
                density_record[s_ind, batch_inds] = sample_batch.packing_coeff.cpu().detach().numpy()
                ssample_record.append(sample_batch.cpu().detach().to_data_list())

                if do_opt:
                    opt_batch = sample_batch.clone()
                    opt_batch = opt_batch.to(device)
                    opt_traj = opt_batch.optimize_crystal_parameters(**optim_kwargs)
                    opt_batch = opt_batch.cpu()

                    finished_batch = collate_data_list(opt_traj[-1])

                    opt_params_record[s_ind, batch_inds] = finished_batch.latent_params().cpu().detach().numpy()
                    opt_energy_record[s_ind, batch_inds] = finished_batch.lj.cpu().detach().numpy()
                    opt_density_record[s_ind, batch_inds] = finished_batch.packing_coeff.cpu().detach().numpy()
                    opt_ssample_record.append(opt_traj[-1])

            sample_record.append(ssample_record)
            if do_opt:
                opt_sample_record.append(opt_ssample_record)

    if do_opt:
        return params_record, energy_record, density_record, sample_record, \
            opt_params_record, opt_energy_record, opt_density_record, opt_sample_record


    else:
        return params_record, energy_record, density_record, sample_record


@torch.no_grad()
def sample_eval_fwd_trajs(initial_state, gfn, discretizer, energy_function, mol_batch,
                          sg_inds=None):
    mol_batch, log_T_tensor, sg_inds, zps, condition = (
        energy_function.condition_samples(mol_batch, sg_inds=sg_inds, z_primes=mol_batch.z_prime))

    condition = condition.to(gfn.device)

    mol_batch.sg_ind = sg_inds

    (states, log_pfs, log_pbs, log_flow,
     means_f, logvars_f, means_b, logvars_b) = gfn.get_traj_fwd(initial_state,
                                                                discretizer,
                                                                None,
                                                                condition,
                                                                return_gauss_params=True)
    gauss_params = {'means_f': means_f.cpu().detach(),
                    'logvars_f': logvars_f.cpu().detach(),
                    'means_b': means_b.cpu().detach(),
                    'logvars_b': logvars_b.cpu().detach()}

    log_r, sample_batch = energy_function.log_reward(
        states[:, -1], mol_batch=mol_batch,
        log_temperature=log_T_tensor,
        return_exp=True)

    log_weight = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)

    log_Z = logmeanexp(log_weight)
    log_Z_lb = log_weight.mean()
    log_Z_learned = log_flow.mean()

    outputs = (states, states[:, -1],
               log_r, log_Z, log_Z_lb, log_Z_learned,
               sample_batch, condition,
               log_pfs, log_pbs, log_flow,
               gauss_params, log_T_tensor)
    outputs = (o if isinstance(o, dict) else o.cpu().detach()
               for o in outputs)
    return outputs


# @torch.no_grad()
# def mean_log_likelihood(terminal_state, gfn, log_reward_fn, num_evals=10):
#     bsz = terminal_state.shape[0]
#     terminal_state = terminal_state.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
#     states, log_pfs, log_pbs, log_fs = gfn.get_traj_bwd(terminal_state, None, log_reward_fn)
#     log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
#     return logmeanexp(log_weight, dim=1).mean()

#
# def crystal_list_rdf(samples, batch_size, device):
#     num_batches = len(samples) // batch_size
#     if len(samples) % batch_size != 0:
#         num_batches += 1
#
#     rdfs = []
#     for b_ind in range(num_batches):
#         batch_inds = np.arange(b_ind * batch_size, min(len(samples), (b_ind + 1) * batch_size))
#         mol_batch = collate_data_list([samples[ind] for ind in batch_inds]).to(device)
#         rdf, rr = get_rdfs(mol_batch)
#         rdfs.append(rdf)
#
#     return torch.cat(rdfs), rr
#
#
# def get_rdfs(crystal_batch):
#     with torch.no_grad():
#         cluster_batch = crystal_batch.mol2cluster(cutoff=6)
#         cluster_batch.construct_radial_graph(cutoff=6)
#         rdf, rr, _ = crystal_rdf(cluster_batch,
#                                  cluster_batch.edges_dict,
#                                  rrange=[0, 6], bins=2000,
#                                  mode='intermolecular',
#                                  elementwise=True,
#                                  raw_density=True,
#                                  cpu_detach=False)
#
#     return rdf.cpu().detach(), rr

#
# @torch.no_grad()
# def sample_csd_rdf_dists(csd_mols, csd_sampling_dict, eval_batch_size, device):
#     sample_rdfs = []
#     for ind in tqdm(range(len(csd_mols))):
#         identifier = csd_mols[ind].identifier
#         for ind2 in range(len(csd_sampling_dict[identifier]['samples'])):
#             samples = csd_sampling_dict[identifier]['samples'][ind2]
#             samples = [item for sublist in samples for item in sublist]
#
#             rdf, rr = crystal_list_rdf(samples, eval_batch_size, device)
#             sample_rdfs.append(rdf)
#
#     per_csd_rdfs = []
#     ii = 0
#     for ind in range(len(csd_mols)):
#         ss_rdf = []
#         for ind2 in range(len(csd_sampling_dict[identifier]['samples'])):
#             ss_rdf.append(sample_rdfs[ii])
#             ii += 1
#         per_csd_rdfs.append(torch.cat(ss_rdf))
#
#     sample_rdfs = torch.stack(per_csd_rdfs)
#     csd_rdfs, rr = crystal_list_rdf(csd_mols,
#                                     eval_batch_size,
#                                     device)
#
#     rdf_dists = torch.zeros_like(sample_rdfs[:, :, 0, 0])
#     for ind in range(len(csd_mols)):
#         rdf_dists[ind] = compute_rdf_distance(csd_rdfs[ind].to(device), sample_rdfs[ind].to(device), rr)
#     return rdf_dists, rr

#
# def sample_csd_lattice_divs(csd_mols, csd_sampling_dict):
#     identifiers = [elem.identifier for elem in csd_mols]
#     js_divs = []
#     for ind, ident in enumerate(identifiers):
#         box_matrix = csd_mols[ind].T_fc[0].T.cpu().detach().numpy()
#         csd_dists = lattice_distance_spectrum(box_matrix,
#                                               max_radius=50,
#                                               resolution=0.01)
#         samples = []
#         for elem in csd_sampling_dict[identifiers[ind]]['samples']:
#             samples.extend(elem)
#         samples = [item for sublist in samples for item in sublist]
#         hist1, hr = np.histogram(csd_dists, bins=100, range=[0, 50])
#         divs = []
#
#         for j in range(len(samples)):
#             box_matrix = samples[j].T_fc[0].T.cpu().detach().numpy()
#             sample_dists = lattice_distance_spectrum(box_matrix,
#                                                      max_radius=50,
#                                                      resolution=0.01)
#             hist2, hr = np.histogram(sample_dists, bins=100, range=[0, 50])
#             divs.append(jensenshannon(hist1, hist2))
#
#         js_divs.append(divs)
#
#     return js_divs


# def lattice_distance_spectrum(cell_matrix, max_radius=0.0, resolution=0.01):
#     """Compute sorted inter-point distances for lattice defined by 3x3 cell_matrix"""
#     max_index = int(np.ceil(max_radius / np.min(np.linalg.norm(cell_matrix, axis=1))))
#     shifts = np.mgrid[-max_index:max_index + 1, -max_index:max_index + 1, -max_index:max_index + 1].reshape(3, -1).T
#     distances = np.linalg.norm(shifts @ cell_matrix, axis=1)
#     distances = distances[(distances > 1e-8) & (distances < max_radius)]
#     distances = np.sort(np.round(distances / resolution) * resolution)  # bin by resolution
#     return distances


def get_plotly_fig_size_mb(fig) -> float:
    # Convert Plotly figure to JSON string
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return sys.getsizeof(fig_json) / (1024 * 1024)


def big_staircse_comparison(dbatch, ebatch):
    gen_samples = dbatch.latent_params()
    elats = ebatch.latent_params()
    if torch.is_tensor(gen_samples):
        gen_samples = gen_samples.detach().cpu().numpy()
    N, D = gen_samples.shape

    # Create D×D subplots (upper triangle empty)
    fig = make_subplots(
        rows=D, cols=D,
        horizontal_spacing=0.01, vertical_spacing=0.01,
        shared_xaxes=True, shared_yaxes=True,
    )

    # Loop over lower triangle
    for i in range(D):
        for j in range(D):
            if j >= i:
                continue  # keep lower triangle only

            x = gen_samples[:, j]
            y = gen_samples[:, i]

            trace = go.Histogram2dContour(
                x=x, y=y,
                ncontours=100,
                colorscale='icefire',
                showscale=False,
                contours=dict(coloring='fill', showlines=False, start=0, end=None, size=None),
                line=dict(smoothing=0.85, width=0),
                nbinsx=100,
                nbinsy=100,
            )
            fig.add_trace(trace, row=i + 1, col=j + 1)

            trace = go.Scatter(x=elats[:, j], y=elats[:, i], mode='markers',
                               marker_color='yellow', marker_line_width=4, opacity=0.5,
                               marker_line_color='black', marker_size=14, showlegend=False)
            fig.add_trace(trace, row=i + 1, col=j + 1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        # height=1000,
        # width=1000,
        showlegend=False,
    )
    fig.update_layout(
        font=dict(family="Helvetica", size=12),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=30, r=30, t=20, b=30),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)  # , range=[-1,1])
    fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', tickwidth=1)  # , range=[-1,1])
    fig.update_layout(height=2400, width=3000)
    return fig
