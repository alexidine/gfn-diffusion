import glob
import os
import re

import torch
from tqdm import tqdm

from energy_sampling.data_processing.utils import calibrate_energy_function_vs_uma
from energy_sampling.utils import new_calibrate_prior_noise
from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.common.clustering import greedy_bottom_up_anchors
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def collate_generate_prior():
    global sample, cell_params, en_scaling_factor, log_noise_range, thinned_batch
    "load samples"
    opt_samples = []
    for tpath in tqdm(traj_records):
        opt_samples.extend(torch.load(tpath, weights_only=False))
    # huge waste of space here for some reason
    for sample in opt_samples:
        if hasattr(sample, 'rdf'):
            del sample.rdf
        if hasattr(sample, 'fingerprint'):
            del sample.fingerprint
    """ 
    make sure we are latent-safe, and filter degenerate boxes (ultra-flat cells)
    """
    sample_batch = collate_data_list(opt_samples, exclude_keys=['elj'])
    sample_batch.latent_to_cell_params(sample_batch.latent_params())
    analyses = ['lj', 'vdw', 'vdw_max']
    analyses.append(energy_function)
    sample_batch = adaptive_batched_analysis(
        sample_batch, analyses=analyses, state={},
        initial_batch_size=10000, predictor=predictor,
        device=device, show_tqdm=True
    )
    sample_batch = sample_batch.to('cpu')
    opt_samples = sample_batch.batch_to_list()  # corrected latents
    # manually filter severe vdW overlaps
    params = sample_batch.latent_params()
    ens = sample_batch[energy_function]
    cell_params = sample_batch.full_cell_parameters()
    lengths = cell_params[:, :3]
    angles = cell_params[:, 3:6]
    volumes = sample_batch.cell_volume
    angular_factor = volumes / lengths.prod(dim=-1)
    good_inds = ((angular_factor > 0.1) & (sample_batch.vdw_max < 1.5)).argwhere().flatten()  # nearly degenerate cells
    good_samples = [opt_samples[ind] for ind in good_inds]
    sample_batch = collate_data_list(good_samples)
    sample_batch.latent_to_cell_params(sample_batch.latent_params())  # get safely into the latent space
    """
    Calibrate distance cutoff on this energy function
    """
    "very coarse diverse basin selection"
    sample_batch = sample_batch.cuda()
    params = sample_batch.latent_params()
    ens = sample_batch[energy_function]
    cps = sample_batch.packing_coeff
    anchors = greedy_bottom_up_anchors(params, cps, ens, d_cut=0.1, e_cut=torch.quantile(ens, 0.1))
    anchors = anchors[:1000]  # 1000 is plenty
    "run calibration"
    coarse_batch = collate_data_list([good_samples[ind] for ind in anchors])
    coarse_batch.latent_to_cell_params(coarse_batch.latent_params())
    if energy_function == 'uma':
        en_scaling_factor = 1
    else:
        en_scaling_factor = calibrate_energy_function_vs_uma(search_output_dir,
                                                             energy_function,
                                                             run_name,
                                                             coarse_batch[energy_function]
                                                             )
    log_noise_range = new_calibrate_prior_noise(coarse_batch,
                                                energy_function,
                                                en_scaling_factor,
                                                kT=kT,
                                                low_cut=0.05,
                                                high_cut=6.0,
                                                predictor=predictor)
    """
    do actual thinning with physics-informed cutoff
    """
    thermal_ens = ens * en_scaling_factor
    d_cut = 10 ** log_noise_range[1]
    anchors = greedy_bottom_up_anchors(params, cps, thermal_ens, d_cut=d_cut, e_cut=thermal_ens.amin() + 6 * kT)
    thinned_batch = collate_data_list([good_samples[ind] for ind in anchors])


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # # mipcas
    # search_output_dir = r"D:\crystal_datasets\mipcas"
    # run_name = 'mipcas_elj'
    # identifier = 'MIPCAS'
    # energy_function = 'elj'
    # target_path = r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt"
    # uma_model_path = r"D:\crystal_datasets\esen_s.pt"
    # device = 'cuda'
    # # nehzor
    # search_output_dir = r"D:\crystal_datasets\nehzor\p6"
    # run_name = 'nehzor_4_uma'
    # identifier = 'NEHZOR'
    # energy_function = 'uma'
    # target_path = r"D:\crystal_datasets\nehzor\NEHZOR01_standardized.pt"
    # uma_model_path = r"D:\crystal_datasets\esen_s.pt"
    # device = 'cuda'
    # acridine
    search_output_dir = r"D:\crystal_datasets\acridine"
    run_name = 'may_acridine_sg14_zp1'
    identifier = 'ACRDIN01'
    energy_function = 'mace'
    target_path = r"D:\crystal_datasets\acridine\prot_acrdin_crystals.pt"
    uma_model_path = r"D:\crystal_datasets\esen_s.pt"
    mace_model_path = r"C:\Users\mikem\Downloads\acr_112025_mh1_stagetwo.model"
    device = 'cuda'

    kT = 2.5
    tot_noised_samples = 200000
    if energy_function == 'uma':
        predictor = init_uma_crystal_predictor(uma_model_path, device)
    elif energy_function == 'mace':
        predictor = load_mace_model(mace_model_path, device, torch.float32)

    "get files"
    os.chdir(search_output_dir)
    pattern = os.path.join(search_output_dir, run_name + '_*.pt')
    files = glob.glob(pattern)
    traj_records = glob.glob(f"{run_name}*")
    traj_records = [elem for elem in traj_records if 'prior' not in elem]
    # traj_records = [
    #     f for f in files
    #     if re.search(rf'{re.escape(run_name)}_\d{{1,2}}\.pt$', os.path.basename(f))
    # ]

    target_mol = torch.load(target_path, weights_only=False)
    if isinstance(target_mol, list):
        target_mol = [elem for elem in target_mol if elem.identifier == identifier][0]
    target_mol.aunit_handedness = target_mol.aunit_handedness.abs()
    zp = target_mol.z_prime
    sg = target_mol.sg_ind

    # todo we need to filter samples which are obviously insane but UMA for some reason doesn't grok

    dataset_filename = run_name + '_prior_dataset.pt'

    if os.path.exists(dataset_filename):
        dd = torch.load(dataset_filename, weights_only=False)
        noised_samples = dd['noised_batch'].batch_to_list()
        thinned_batch = dd['prior_batch']
        log_noise_range = dd['log_noise_range']
        en_scaling_factor = dd['thermal_scaling_factor']
    else:
        noised_samples = []
        collate_generate_prior()

    """
    preliminary noising
    """
    state = {}
    samples_per_anchor = ((tot_noised_samples - len(noised_samples)) // thinned_batch.num_graphs) + 1
    for _ in tqdm(range(samples_per_anchor)):
        batch = thinned_batch.clone()
        batch = batch.to(device)
        batch.log_noise_latent_parameters(log_noise_range[0], log_noise_range[1])
        batch, state = adaptive_batched_analysis(
            batch, analyses=[energy_function, 'reduction_en'],
            state=state,
            initial_batch_size=10000,
            predictor=predictor,
            return_state=True,
            device=device,
            show_tqdm=False,
        )
        batch = batch.to('cpu')
        noised_samples_i = batch.batch_to_list()
        valid = torch.argwhere(
            (batch.reduction_en < 1e-3) & (batch.packing_coeff > 0.55) & (batch.packing_coeff < 0.95)).flatten()
        noised_samples.extend([noised_samples_i[ind] for ind in valid])
        del batch

        dataset_dict = {
            'thermal_scaling_factor': en_scaling_factor,
            'log_noise_range': log_noise_range,
            'prior_batch': thinned_batch.cpu(),
            'noised_batch': collate_data_list(noised_samples),
        }
        torch.save(dataset_dict, dataset_filename)

    aa = 1  # thin out reduction_en and save reward stuff here as well - maybe save one big batch?
