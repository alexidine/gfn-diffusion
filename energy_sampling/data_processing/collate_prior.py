import glob
import os

import torch
from tqdm import tqdm

from energy_sampling.data_processing.utils import calibrate_energy_function_vs_uma
from energy_sampling.utils import new_calibrate_prior_noise
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
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
        try:
            opt_samples.extend(torch.load(tpath, weights_only=False))
        except:
            print(tpath, "bad file")
            pass
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
    analyses = ['lj', 'vdw', 'vdw_max', 'rdf', energy_function]
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
    if energy_function in ['uma', 'mace']:
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


def confirm_polymorphs_in_prior():
    global identifier, thinned_batch, target_path, target_mol, tbatch, tbatch
    results = []
    for identifier in identifiers:
        for rdtype in ['envwise']:  # [None, 'elementwise', 'atomwise', 'envwise']:
            thinned_batch = dd['prior_batch'].clone()
            del thinned_batch.rdf

            # target_path = r"D:\crystal_datasets\opt_outputs\acrdin_mace_test.pt"
            target_path = r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"
            target_mol = torch.load(target_path, weights_only=False).batch_to_list()
            if isinstance(target_mol, list):
                target_mol = [elem for elem in target_mol if elem.identifier in identifier]
            tbatch = collate_data_list(target_mol)

            # target_path = r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"
            # tbatch = torch.load(target_path, weights_only=False)

            tbatch.aunit_handedness = tbatch.aunit_handedness.abs()
            zp = tbatch.z_prime
            sg = tbatch.sg_ind
            tbatch.analyze(['rdf'], rdf_mode=rdtype, cutoff=10, rdf_cutoff=10, assign_outputs=True)
            thinned_batch.to('cuda')
            analyses = ['rdf']
            thinned_batch = adaptive_batched_analysis(
                thinned_batch, analyses=analyses, state={},
                initial_batch_size=1000, predictor=predictor,
                device=device, show_tqdm=True, rdf_mode=rdtype, cutoff=10, rdf_cutoff=10
            )
            thinned_batch.to('cpu')
            bins = torch.linspace(0, 10, 100)
            ds = []
            ds.append(compute_rdf_distance(tbatch.rdf, thinned_batch.rdf, bins))

            dstack = torch.stack(ds)
            dinds = dstack.argsort(dim=1, descending=False)[:, :20].flatten()

            samps = thinned_batch.batch_to_list()
            cb = collate_data_list([samps[ind] for ind in dinds])
            cbatch = cb.mol2cluster()

            tbatch.mol2ucell()
            tbatch.write_cif(torch.arange(tbatch.num_graphs), 'acr_ref', 'unit cell')
            matches, rmsds = cb.batch_compack('acr_ref_0.cif', torch.arange(cb.num_graphs))
            results.append([identifiers, rdtype, matches, rmsds])


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
    search_output_dir = r"D:\crystal_datasets\acridine\prior_chunks"
    run_name = 'may_acridine_sg14_zp2'
    identifiers = None #['ACRDIN04', 'ACRDIN12']
    energy_function = 'mace'
    target_path = r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"
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

    target_mol = torch.load(target_path, weights_only=False).batch_to_list()
    if isinstance(target_mol, list):
        if identifiers is not None:
            target_mol = [elem for elem in target_mol if elem.identifier in identifiers]
    tbatch = collate_data_list(target_mol)
    tbatch.aunit_handedness = tbatch.aunit_handedness.abs()
    zp = tbatch.z_prime
    sg = tbatch.sg_ind

    dataset_filename = run_name + '_prior_dataset.pt'

    if os.path.exists(dataset_filename):
        dd = torch.load(dataset_filename, weights_only=False)

        thinned_batch = dd['prior_batch']
        log_noise_range = dd['log_noise_range']
        en_scaling_factor = dd['thermal_scaling_factor']
        if hasattr(dd, 'noised_batch'):
            noised_samples = dd['noised_batch'].batch_to_list()
        else:
            noised_samples = []
    else:
        noised_samples = []
        collate_generate_prior()
        dataset_dict = {
            'thermal_scaling_factor': en_scaling_factor,
            'log_noise_range': log_noise_range,
            'prior_batch': thinned_batch.cpu(),
        }
        torch.save(dataset_dict, dataset_filename)

    """
    confirm that target structures are in the distribution
    """

    #confirm_polymorphs_in_prior()

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
