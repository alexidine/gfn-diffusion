from copy import copy

import plotly.graph_objects as go
import os

import numpy as np
import torch
from tqdm import tqdm
import hdbscan

from energy_sampling.eval.paper1_results.figures import sparkbar_table
from energy_sampling.eval.paper1_results.utils import generator_reward, new_local_analysis, sample_and_analyze, \
    load_experimental_structure, dmat_local_analysis, mean_shift_density, basin_opt
from energy_sampling.utils import load_yaml, dict2namespace
from mxtaltools.analysis.crystal_rdf import rdf_radial_graph, compute_rdf_distance, compute_rdf_distmat
from mxtaltools.crystal_search.run_search import crystal_search
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)

if __name__ == '__main__':

    config = dict2namespace(load_yaml('analysis.yaml'))
    results_path = os.path.join(config.results_dir, rf"{config.run_name}.pt")
    basins_path = os.path.join(config.results_dir, config.run_name + '_basins.pt')

    "Load Relevant Dataset"
    molecule = torch.load(config.molecule_path, weights_only=False)
    if isinstance(molecule, list):
        molecule = molecule[0]
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
        pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
        predictor = init_uma_crystal_predictor(pred_path, device='cuda')
        exp_crystals = torch.load(config.exp_sample_path, weights_only=False)
        ebatch = collate_data_list(exp_crystals, max_z_prime=1)
        computes = ['lj', 'qlj', 'elj', 'silu', 'rdf', 'reduction_en']
        if config.energy_function == 'uma':
            computes.append('uma')
        with torch.no_grad():
            ebatch.cuda()
            ebatch.analyze(computes, elementwise=False, atomwise=True, assign_outputs=True,
                           predictor=predictor)
            ebatch.cpu()
        esamples = ebatch.batch_to_list()
        # samples = samples + esamples
    else:
        predictor = None

    """get sample info"""
    sample_batch = collate_data_list(samples)

    sample_energy = sample_batch[config.energy_function]
    if hasattr(sample_batch, 'elj'):  # appropriately rescale energy
        sample_batch.elj *= en_scaling_factor
    good_ens = (sample_energy < sample_energy.amin() + 15 * config.kT).argwhere().flatten()
    sample_batch = collate_data_list([samples[i] for i in good_ens])
    sample_energy = sample_batch[config.energy_function]
    if hasattr(sample_batch, 'elj'):  # appropriately rescale energy
        sample_batch.elj *= en_scaling_factor
    sample_cp = sample_batch.packing_coeff

    if (not 'dmat' in list(results_dict.keys())) or (not config.reload_results):
        rdf_bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1])
        with torch.no_grad():
            dmat = compute_rdf_distmat(sample_batch.rdf.cuda(), rdf_bins.cuda()).cpu()
        results_dict['dmat'] = dmat
    else:
        dmat = results_dict['dmat']

    """start neighborhood analysis"""
    d_metrics = []
    bins = torch.linspace(0, 10, sample_batch.rdf.shape[-1], device='cuda')
    d = torch.cat([compute_rdf_distance(sample_batch.rdf[ii], sample_batch.rdf, bins.cpu()) for ii in range(5)])
    d_cuts = [d.quantile(0.15)]

    for d_cut in d_cuts:
        with torch.no_grad():
            sample_metrics = dmat_local_analysis(sample_energy,
                                                 dmat,
                                                 d_cut=d_cut,
                                                 d_kernel=d_cut / 3,
                                                 e_cut=None,
                                                 )
            ndims = sample_batch.latent_params().shape[-1]
            d_metrics.append(sample_metrics)

    results_dict.update({'metrics': d_metrics,
                         'd_cuts': d_cuts,
                         }, )

    #
    # samples_per = 20
    # n_basins = sum(np.unique(cleaned_cluster_labels) > -1)
    # basin_minima = []
    # for bind in np.unique(cleaned_cluster_labels):
    #     if bind == -1:
    #         continue
    #     b_inds = torch.argwhere(cleaned_cluster_labels == bind).flatten()
    #     b_ens = sample_energy[b_inds]
    #     topk = b_ens.topk(samples_per, largest=False).indices
    #     basin_minima.extend(b_inds[topk])
    #
    # num_polymorphs = ebatch.num_graphs
    # poly_dists = torch.stack([compute_rdf_distance(ebatch.rdf[ind].cpu(), sample_batch.rdf.cpu(), bins.cpu()) for ind in range(num_polymorphs)]) #dmat[:-num_polymorphs, -num_polymorphs:]
    # results_dict.update({'poly_dists': poly_dists})
    # closest = poly_dists.argsort(dim=1, descending=False)[:, :samples_per]
    # samples = sample_batch.batch_to_list()
    # opt_samples = [samples[ind] for ind in closest.T.flatten()] + esamples + [samples[ind] for ind in basin_minima]
    #
    # opt_config = {
    #     'device': 'cuda',
    #     'mol_path': None,
    #     'dataset_path': None,
    #     'target_path': None,
    #     'umbrella_path': None,
    #     'target_identifier': 'temp_opt',
    #     'out_dir': 'D:/crystal_datasets/opt_outputs',
    #     'run_name': 'temp_opt',
    #     'save_trajs': False,
    #     'uma_predictor_path': 'D:/crystal_datasets/esen_s.pt',
    #     'init_sample_method': 'in_config',
    #     'samples_to_optim': opt_samples,
    #     'init_reduced': False,
    #     'mol_seed': 0,
    #     'opt_seed': 0,
    #     'sampling_mode': 'all',
    #     'mols_to_sample': None,
    #     'num_samples': 10000,
    #     'sgs_to_search': [opt_samples[0].sg_ind],
    #     'zp_to_search': [1],
    #     'batch_size': 200,
    #     'grow_batch_size': False,
    #     'init_target_cp': None,
    #     'opt': [
    #         {
    #             'optim_target': config.energy_function,
    #             'enforce_reduced': True,
    #             'compression_factor': 0.0,
    #             'cutoff': 10,
    #             'init_lr': 0.001,
    #             'convergence_eps': 0.0001,
    #             'optimizer_func': 'rprop',
    #             'anneal_lr': False,
    #             'grad_norm_clip': 0.1,
    #             'show_tqdm': True,
    #             'max_num_steps': 500,
    #             'rdf_warmup': None,
    #             'target_packing_coeff': None,
    #             'umbrella': False,
    #             'umbrella_sigma': 0.25,
    #             'umbrella_epsilon': 40.0,
    #         },
    #     ],
    # }
    # os.remove(r"D:\crystal_datasets\opt_outputs\temp_opt.pt")
    # opt_outs = crystal_search(dict2namespace(opt_config))
    # results_dict.update({'poly_opts': opt_outs[:(num_polymorphs * samples_per) + num_polymorphs],
    #                      'basin_opts': opt_outs[(num_polymorphs * samples_per) + num_polymorphs:],})

    torch.save(results_dict, results_path)
    aa = 1
