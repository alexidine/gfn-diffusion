"""
Detailed / expensive evaluations for the GFN generator

1. Broad sampling over evaluation molecules and space groups
2. Report distributions of energies and diversity vs. SG, and molecule correlates (size, composition, functional group, spherical defect, shape factors)

"""
import os

import numpy as np
import torch

from energy_sampling.energies.molecular_crystal import MolecularCrystal
from energy_sampling.evaluations import log_partition_function
from energy_sampling.train import embed_dataset
from energy_sampling.utils import uniform_discretizer, get_gfn_init_state, load_yaml, dict2namespace
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import load_encoder


def sample_from_generator(
        gfn_model,
        batch_size,
        mol_list,
        space_group,
        n_steps,
        samples_per_mol,
        device,
        energy_function,
        encoder=None,
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
    with torch.no_grad():
        discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)

        num_batches = len(mol_list) // batch_size
        if len(mol_list) % batch_size != 0:
            num_batches += 1

        energy_function.space_groups = [space_group]
        init_state = get_gfn_init_state(batch_size, energy_function.data_ndim, device)

        sample_record = np.zeros((samples_per_mol, len(mol_list), 12))
        energy_record = np.zeros((samples_per_mol, len(mol_list)))
        density_record = np.zeros_like(energy_record)

        """embed the dataset"""
        if hasattr(mol_list[0], 'embedding'):
            if mol_list[0].embedding is not None:
                pass
            else:
                mol_list = embed_dataset(mol_list, encoder=encoder)

        """sample"""
        for s_ind in range(samples_per_mol):
            for b_ind in range(num_batches):
                batch_inds = np.arange(b_ind * batch_size, (b_ind + 1) * batch_size)
                mol_batch = collate_data_list([mol_list[ind] for ind in batch_inds]).to(device)
                (flow_states, samples, log_r, log_Z, log_Z_lb,
                 log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_fs,
                 f_means_f, f_vars_f, f_means_b, f_vars_b,
                 log_T_tensor) = log_partition_function(
                    init_state, gfn_model, discretizer, energy_function, mol_batch)

                sample_record[s_ind, batch_inds] = flow_states[:, -1].cpu().detach().numpy()
                energy_record[s_ind, batch_inds] = sample_batch.silu_energy.cpu().detach().numpy()
                density_record[s_ind, batch_inds] = sample_batch.packing_coeff.cpu().detach().numpy()

    return sample_record, energy_record, density_record


"""load relevant data"""
args = dict2namespace(load_yaml(eval_config))
override_resample = args.override_resample
generator_path = 'p1'  # generator model
eval_molecules_path = 'p2'  # molecules to evaluate on
encoder_model_path = 'p3'  # autoencoder model
opt_crystals_path = 'p4'  # pre-optimized evaluation samples
csd_crystals_path = 'p5'  # qm9-like experimental samples
fully_sampled_crystals_path = 'p6'  # a subset of crystals which have been exhaustively sampled
eval_config = 'p7'

os.mkdir('eval_results')
# load stuff here
gfn_model = torch.load(generator_path)
encoder = load_encoder(encoder_model_path)
gfn_model.eval()
encoder.eval()
energy_function = MolecularCrystal(device=args.device,
                                   energy_function=args.energy_function,
                                   min_temperature=args.energy_min_temperature,
                                   max_temperature=args.energy_max_temperature,
                                   temperature_scaling_factor=args.temperature_scaling_factor,
                                   temperature_conditioning=args.temperature_conditioning,
                                   temperature=args.energy_static_temperature,
                                   density_coeff=args.energy_density_coeff,
                                   energy_clip=args.energy_clip,
                                   ellipsoid_scale=args.ellipsoid_scale,
                                   core_coeff=args.energy_core_coeff,
                                   lj_coeff=args.energy_lj_coeff,
                                   lj_turnover_pot=args.lj_turnover_pot,
                                   lj_repulsion=args.lj_repulsion,
                                   molecule_conditioning=args.molecule_conditioning,
                                   sg_conditioning=args.sg_conditioning,
                                   space_groups=args.space_groups,
                                   )


"""
for each space group, for the eval set, sample and record energies, cell parameters
"""
if not os.path.exists('sg_results.npy') or override_resample:
    eval_mols = torch.load(eval_molecules_path)
    sg_sampling_dict = {sg: {} for sg in args.sgs_to_sample}
    for sg in args.sgs_to_sample:
        (sg_sampling_dict[sg]['samples'],
         sg_sampling_dict[sg]['energies'],
         sg_sampling_dict[sg]['densities']) = sample_from_generator(
            gfn_model, args.eval_batch_size, eval_mols,
            sg, args.eval_T, args.eval_samples_per_mol,
            args.device, energy_function, encoder
        )
    np.save('sg_results', sg_sampling_dict)

"""
for each csd molecule, sample and record energies, cell parameters in the target SG
"""
if not os.path.exists('csd_results.npy') or override_resample:
    csd_mols = torch.load(csd_crystals_path)
    identifiers = [mol.identifier for mol in csd_mols]
    csd_sampling_dict = {ident: {} for ident in identifiers}
    for ind in range(10):
        id = identifiers[ind]
        mol = csd_mols[ind]
        sg = mol.sg_ind
        (csd_sampling_dict[id]['samples'],
         csd_sampling_dict[id]['energies'],
         csd_sampling_dict[id]['densities']) = sample_from_generator(
            gfn_model, args.eval_batch_size, [mol for _ in range(args.eval_batch_size)],
            sg, args.eval_T, args.csd_samples_per_mol,
            args.device, energy_function, encoder
        )
    np.save('csd_results', csd_sampling_dict)

"""
For each fully sampled molecule, sample with the generator
"""
if not os.path.exists('sampled_results.npy') or override_resample:
    sampled_mols = torch.load(fully_sampled_crystals_path)
    identifiers = [mol.identifier for mol in sampled_mols]
    full_sampling_dict = {ident: {} for ident in identifiers}
    for ind in range(10):
        id = identifiers[ind]
        mol = sampled_mols[ind]
        sg = mol.sg_ind
        (full_sampling_dict[id]['samples'],
         full_sampling_dict[id]['energies'],
         full_sampling_dict[id]['densities']) = sample_from_generator(
            gfn_model, args.eval_batch_size, [mol for _ in range(args.eval_batch_size)],
            sg, args.eval_T, args.csd_samples_per_mol,
            args.device, energy_function, encoder
        )
    np.save('sampled_results', full_sampling_dict)

"""
Reporting
- on eval set:
    - energy & diversity per-sg
    - correlate mol features with energy & diversity
- on csd samples
    - niggli cell distance to real crystal
    - energy difference to real crystal
    - sample density near real crystal
- on opt crystals
    - check coverage
    - check distribution vs well depths
- on fully sampled crystals
    - check coverage
    - check density
    - estimate sample efficiency
"""
