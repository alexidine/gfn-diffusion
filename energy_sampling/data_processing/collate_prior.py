import argparse
import glob
import os

import torch
from tqdm import tqdm

from energy_sampling.data_processing.utils import (UMA_ENERGY_STATE,
                                                   calibrate_energy_function_vs_uma,
                                                   load_search_chunks)
from energy_sampling.utils import new_calibrate_prior_noise
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.adaptive_batching import adaptive_batched_analysis
from mxtaltools.common.clustering import greedy_bottom_up_anchors, greedy_bottom_up_anchors2
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def collate_generate_prior():
    global sample, cell_params, en_scaling_factor, log_noise_range, thinned_batch
    "load samples"
    # prefers f047_rescored chunks where they exist, and refuses stale UMA energies
    # outright on the uma route -- an unstamped chunk is pre-F-047 by definition
    opt_samples = load_search_chunks(search_output_dir, run_name,
                                     require_uma_state=(energy_function == 'uma'))
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
    # NO_GRAD: fairchem leaves grad ENABLED (its _run_inference picks nullcontext
    # whenever direct_forces is False, which our crystal predictor sets), so the
    # scored energies come back carrying grad_fn and pin their activations --
    # measured at 100-250 MB retained PER CRYSTAL. Nothing here is differentiated.
    # This is an offline scan, so the caller owns the decision; the shared energy
    # path must NOT do this, or reward_grads silently reads zero.
    with torch.no_grad():
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
    anchors = greedy_bottom_up_anchors2(params, cps, thermal_ens, d_cut=d_cut, e_cut=thermal_ens.amin() + 6 * kT)
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
            with torch.no_grad():          # offline scan; see collate_generate_prior
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

    #: The arms, as DATA rather than commented-out assignments. Selecting one by
    #: uncommenting is how two of these silently drifted out of step with the code:
    #: the nehzor and mipcas blocks set `identifier` (singular) where the flow reads
    #: `identifiers`, and named 'NEHZOR' where the target file carries 'NEHZOR01',
    #: so either one would have filtered the target list down to nothing.
    TARGETS = {
        'mipcas_elj': dict(
            search_output_dir=r"D:\crystal_datasets\mipcas", run_name='mipcas_elj',
            identifiers=['MIPCAS'], energy_function='elj',
            target_path=r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt"),
        'mipcas_uma': dict(
            search_output_dir=r"D:\crystal_datasets\mipcas", run_name='mipcas_uma',
            identifiers=['MIPCAS'], energy_function='uma',
            target_path=r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt"),
        'nehzor_elj': dict(
            search_output_dir=r"D:\crystal_datasets\nehzor\p6", run_name='nehzor_4_elj',
            identifiers=['NEHZOR01'], energy_function='elj',
            target_path=r"D:\crystal_datasets\nehzor\NEHZOR01_standardized.pt"),
        'nehzor_uma': dict(
            search_output_dir=r"D:\crystal_datasets\nehzor\p6", run_name='nehzor_4_uma',
            identifiers=['NEHZOR01'], energy_function='uma',
            target_path=r"D:\crystal_datasets\nehzor\NEHZOR01_standardized.pt"),
        # acridine forms, kept with the identifier->polymorph mapping that is the
        # only record of which forms each arm targets
        'acridine_sg14_zp1': dict(
            search_output_dir=r"D:\crystal_datasets\acridine\prior_chunks",
            run_name='may_acridine_sg14_zp1',
            identifiers=['ACRDIN04', 'ACRDIN12'],      # forms II and IX
            energy_function='mace',
            target_path=r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"),
        'acridine_sg14_zp2': dict(
            search_output_dir=r"D:\crystal_datasets\acridine\prior_chunks",
            run_name='may_acridine_sg14_zp2',
            identifiers=['ACRDIN07', 'ACRDIN06'],      # forms III and VII
            energy_function='mace',
            target_path=r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"),
        'acridine_sg9_zp2': dict(
            search_output_dir=r"D:\crystal_datasets\acridine\prior_chunks",
            run_name='may_acridine_sg9_zp2',
            identifiers=['ACRDIN05', 'ACRDIN_VIII'],   # forms VI and VIII
            energy_function='mace',
            target_path=r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"),
        'acridine_sg19_zp3': dict(
            search_output_dir=r"D:\crystal_datasets\acridine\prior_chunks",
            run_name='may_acridine_sg19_zp3',
            identifiers=['ACRDIN08'],                  # form IV
            energy_function='mace',
            target_path=r"D:\crystal_datasets\acridine\std_acridine_polymorphs.pt"),
    }

    ap = argparse.ArgumentParser(description='Build one prior dataset from search output.')
    ap.add_argument('--target', required=True, choices=sorted(TARGETS))
    ap.add_argument('--uma-model', default=r"D:\crystal_datasets\esen_s.pt")
    ap.add_argument('--mace-model', default=r"D:\crystal_datasets\acr_112025_mh1_stagetwo.model")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--noised-samples', type=int, default=50000)
    ap.add_argument('--dataset-suffix', default='',
                    help="appended to the output name, e.g. '_f047' -> "
                         "<run>_f047_prior_dataset.pt. A REBUILD MUST USE ONE: the "
                         "resume branch below reuses any existing file of the target "
                         "name outright, so re-running after an energy-function change "
                         "silently returns the old prior and reports success.")
    cli = ap.parse_args()

    cfg = TARGETS[cli.target]
    search_output_dir = cfg['search_output_dir']
    run_name = cfg['run_name']
    identifiers = cfg['identifiers']
    energy_function = cfg['energy_function']
    target_path = cfg['target_path']
    uma_model_path, mace_model_path, device = cli.uma_model, cli.mace_model, cli.device
    print(f'building prior for {cli.target}: {run_name} ({energy_function}) '
          f'from {search_output_dir}')

    kT = 2.5
    tot_noised_samples = cli.noised_samples
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

    # the acridine targets are a BATCH of polymorphs; the nehzor/mipcas ones are a
    # single crystal, where batch_to_list asserts. Handle both rather than assuming.
    target_obj = torch.load(target_path, weights_only=False)
    target_mol = (target_obj.batch_to_list() if getattr(target_obj, 'is_batch', False)
                  else [target_obj])
    if identifiers is not None:
        target_mol = [elem for elem in target_mol if elem.identifier in identifiers]
    if not target_mol:
        raise ValueError(
            f'no target structure matched identifiers {identifiers} in {target_path}; '
            f'available: {sorted({getattr(e, "identifier", "?") for e in (target_obj.batch_to_list() if getattr(target_obj, "is_batch", False) else [target_obj])})}')
    tbatch = collate_data_list(target_mol)
    tbatch.aunit_handedness = tbatch.aunit_handedness.abs()
    zp = tbatch.z_prime
    sg = tbatch.sg_ind

    dataset_filename = run_name + cli.dataset_suffix + '_prior_dataset.pt'

    if os.path.exists(dataset_filename):
        dd = torch.load(dataset_filename, weights_only=False)

        thinned_batch = dd['prior_batch']
        log_noise_range = dd['log_noise_range']
        en_scaling_factor = dd['thermal_scaling_factor']
        if 'noised_batch' in dd:
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
            # which energy state selected these structures. The THINNING is what the
            # stamp is about: it cuts at e_min + 6kT, so energies that were wrong
            # change which structures are here, not merely what they are labelled.
            'uma_energy_state': UMA_ENERGY_STATE if energy_function == 'uma' else None,
        }
        torch.save(dataset_dict, dataset_filename)

    """
    confirm that target structures are in the distribution
    """

    # confirm_polymorphs_in_prior()

    """
    preliminary noising
    """
    # DELETE RDFS
    if hasattr(thinned_batch, 'rdf'):
        del thinned_batch.rdf
    state = {}
    samples_per_anchor = ((tot_noised_samples - len(noised_samples)) // thinned_batch.num_graphs) + 1
    for _ in tqdm(range(samples_per_anchor)):
        batch = thinned_batch.clone()
        batch = batch.to(device)
        batch.log_noise_latent_parameters(log_noise_range[0], log_noise_range[1])
        # NO_GRAD, and it matters most here: `state` is sticky across iterations, so
        # one retention-driven OOM ratchets the batch size down for the whole noising
        # loop and never recovers. See collate_generate_prior for the mechanism.
        with torch.no_grad():
            batch, state = adaptive_batched_analysis(
                batch,
                analyses=[energy_function, 'reduction_en'],
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
            'uma_energy_state': UMA_ENERGY_STATE if energy_function == 'uma' else None,
            'noised_batch': collate_data_list(noised_samples),
        }
        torch.save(dataset_dict, dataset_filename)

    aa = 1  # thin out reduction_en and save reward stuff here as well - maybe save one big batch?
