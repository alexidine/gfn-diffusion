import os

import os

import torch

from energy_sampling.eval.paper1_results.utils import sample_and_analyze, \
    dmat_local_analysis
from energy_sampling.utils import load_yaml, dict2namespace
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance, compute_rdf_distmat
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def run_analysis(config):
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
        elif config.energy_function == 'mace':
            computes.append('mace')
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

    torch.save(results_dict, results_path)
    aa = 1


if __name__ == '__main__':
    raw = load_yaml('analysis.yaml')
    base, runs = raw['base'], raw['runs']

    for run in runs:
        config = dict2namespace({**base, **run})
        results_path = os.path.join(config.results_dir, f"{config.run_name}.pt")
        basins_path = os.path.join(config.results_dir, f"{config.run_name}_basins.pt")

        print(f"\n=== {config.run_name} ===")
        run_analysis(config)  # whatever your entry point is


        aa = 1