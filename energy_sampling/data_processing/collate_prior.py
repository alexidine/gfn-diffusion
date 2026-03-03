import torch
import plotly.graph_objects as go

from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.geometry_utils import compute_latent_distance
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

opt_outs = torch.load(r"D:\crystal_datasets\opt_outputs\mipcas_test.pt", weights_only=False)
batch = collate_data_list(opt_outs)
target = torch.load(r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt", weights_only=False)

target_latent = target.latent_params()
tbatch = collate_data_list([target])
tlats = target_latent.repeat(1000, 1)
tlats += torch.randn_like(tlats) * 0.01

#batch.plot_batch_cell_params(space='latent', ref_dist=tlats)
#batch.plot_batch_staircase(space='latent', ref_dist=target_latent)

lat_dist = compute_latent_distance(target_latent, batch.latent_params())

pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
predictor = init_uma_crystal_predictor(pred_path, device='cuda')

best_inds = lat_dist.topk(100, largest=False).indices
best_batch = collate_data_list([opt_outs[ind] for ind in best_inds])
best_batch = best_batch.cuda()
tbatch = tbatch.cuda()

best_batch.analyze(['rdf','uma', 'elj'],assign_outputs=True, elementwise=False, atomwise=True, predictor=predictor)
tbatch.analyze(['rdf','uma', 'elj'],assign_outputs=True, elementwise=False, atomwise=True, predictor=predictor)
bins = torch.linspace(0, 10, 500)

rdf_dist = compute_rdf_distance(best_batch.rdf.cpu(), tbatch.rdf, bins)
go.Figure(go.Scatter(x=lat_dist[best_inds], y=rdf_dist, mode='markers')).show()


aa = 1


# from plotly.subplots import make_subplots
#
# import os
# import glob
# import torch
# import plotly.graph_objects as go
# from tqdm import tqdm
#
# from mxtaltools.dataset_utils.utils import collate_data_list
#
#
# def norm_lj(lj_en):
#     lj_mean, lj_std, uma_mean, uma_std = [-20.6, 5.7, -3.4, 1.5]
#     atomwise_energy = lj_en / (target_mol.num_atoms / zp)
#     atomwise_fixed = (atomwise_energy - lj_mean) / lj_std * uma_std + uma_mean
#     return atomwise_fixed * (target_mol.num_atoms / zp)
#
#
# def param_dist(ref, sample, scale):
#     """
#     :param ref: [n, k]
#     :param sample: [k]
#     :param scale: [k]
#     :return: [k]
#     """
#     return (ref - sample[None, :]).abs() / scale
#
#
# def thin_points_indices(traj, cutoff, eps=1e-8):
#     """
#     traj: [n, l, k]
#
#     Returns:
#         indices: list of 1D LongTensors, one per trajectory
#     """
#
#     L, N, D = traj.shape
#
#     # per-sample, per-dim scale
#     ptp = traj.quantile(0.95, dim=0) - traj.quantile(0.05, dim=0)  # [N, D]
#     scale = ptp.clamp_min(eps)  # avoid div-by-zero
#
#     all_indices = []
#
#     for i in range(N):
#         t = traj[:, i]  # [L, D]
#         s = scale[i]  # [D]
#
#         keep = [L - 1]
#         last = t[L - 1]
#
#         for j in range(L - 2, -1, -1):
#             diff = (t[j] - last).abs() / s  # per-dim scaled diff
#
#             # fire if any dimension exceeds cutoff
#             if (diff >= cutoff).any():
#                 keep.append(j)
#                 last = t[j]
#
#         keep.reverse()  # restore chronological order
#         all_indices.append(torch.tensor(keep, device=traj.device))
#
#     return all_indices
#
# if __name__ == '__main__':
#
#     search_output_dir = r'D:\crystal_datasets\opt_outputs'
#     #run_name = 'acridine_14_local'
#     #target_path = r"D:\crystal_datasets\acridine\ACRDIN04_standardized_match.pt"
#     run_name = 'xul_61_local'
#     target_path = r"D:\crystal_datasets\xuldud\xul_csd.pkl"
#     identifier = 'XULDUD'
#     energy_function = 'elj'
#
#     os.chdir(search_output_dir)
#     traj_records = glob.glob(os.path.join(search_output_dir, run_name + '_traj*1.pt'))
#
#     target_mol = torch.load(target_path, weights_only=False)
#     target_mol = [elem for elem in target_mol if elem.identifier == identifier][0]
#     target_mol.aunit_handedness = target_mol.aunit_handedness.abs()
#     zp = target_mol.z_prime
#     sg = target_mol.sg_ind
#
#     all_params = []
#     all_ens = []
#     for tpath in tqdm(traj_records):
#         record = torch.load(tpath, weights_only=False)
#
#         traj = record['params']
#         indices = thin_points_indices(traj, 0.1)
#
#         ens = record[energy_function]
#         thinned_traj = torch.cat([traj[idx, i] for i, idx in enumerate(indices)])
#         thinned_energies = torch.cat([ens[idx, i] for i, idx in enumerate(indices)])
#
#         all_ens.extend(thinned_energies)
#         all_params.append(thinned_traj)
#
#     all_params = torch.cat(all_params, dim=0)
#     all_ens = torch.tensor(all_ens)
#
#
#     anchors = []
#
#     eps = 1e-3
#     scale = torch.quantile(all_params, 0.95, dim=0) - torch.quantile(all_params, 0.05, dim=0) + eps
#     diff = param_dist(all_params, target_mol.zp1_cell_parameters().cpu()[0], scale)
#
#     d_cut = 0.05
#
#     e_cut = torch.quantile(all_ens[all_ens < 0], 0.25)
#     en_sort_inds = torch.argsort(all_ens, descending=False).flatten()
#     anchors.append(en_sort_inds[0])
#     for ind in tqdm(en_sort_inds[1:]):
#         if all_ens[ind] > e_cut:
#             break
#         sample = all_params[ind]
#         ref = all_params[torch.tensor(anchors)]
#         diff = param_dist(ref, sample, scale)
#         keep = (diff > d_cut).any(dim=1).all()
#
#         if keep:
#             anchors.append(ind)
#
#     print(f"anchors found: {len(anchors)}")
#
#     '''
#     make nice dataset
#     '''
#
#     batch = collate_data_list([target_mol.clone() for _ in range(len(anchors))], max_z_prime=zp)
#     batch.set_cell_parameters(all_params[anchors])
#     batch.analyze(['elj', 'reduction_en', 'lj'], cutoff=10, supercell_size=10, assign_outputs=True)
#     samples = batch.batch_to_list()
#
#     prior_dataset = {
#         'anchor_samples': samples,
#         'noisy_samples': all_params,
#         'noisy_energies': all_ens,
#     }
#     torch.save(prior_dataset, os.path.join(search_output_dir,target_mol.identifier + '_prior_dataset.pt'))
#
#     #
#     # fig = make_subplots(rows=4, cols=3)
#     # for ind in range(12):
#     #     row = ind // 3 + 1
#     #     col = ind % 3 + 1
#     #     fig.add_histogram(x=all_params[:, ind], row=row, col=col, nbinsx=100, histnorm='probability density',
#     #                       marker_color='red')
#     #     fig.add_histogram(x=all_params[anchors, ind], row=row, col=col, nbinsx=100, histnorm='probability density',
#     #                       marker_color='blue')
#     # fig.show()
#     #
#     # batch = collate_data_list([target_mol.clone() for _ in range(len(anchors))], max_z_prime=zp)
#     # batch.reset_sg_info(14)
#     # batch.set_cell_parameters(all_params[anchors])
#     # batch.clean_cell_parameters(
#     #     mode='hard',
#     #     canonicalize_orientations=True,
#     # )
#     #
#     # en2 = batch.analyze(['elj', 'reduction_en'], cutoff=10, supercell_size=10)['elj']
#     # en2 = norm_lj(en2)
#     # go.Figure(go.Scatter(x=all_ens[anchors], y=en2, mode='markers')).show()
#
#     aa = 1
