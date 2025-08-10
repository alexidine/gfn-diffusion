"""
Detailed / expensive evaluations for the GFN generator

1. Broad sampling over evaluation molecules and space groups
2. Report distributions of energies and diversity vs. SG, and molecule correlates (size, composition, functional group, spherical defect, shape factors)

"""
import os

import numpy as np
import torch
from tqdm import tqdm

from energy_sampling.energies.molecular_crystal import MolecularCrystal
from energy_sampling.eval.offline_figs import create_energy_distribution_plot, create_density_distribution_plot, \
    create_cell_params_variance_plot, crystal_sample_funnel_plot
from energy_sampling.eval.utils import sample_from_generator, sample_csd_rdf_dists, sample_csd_lattice_divs
from energy_sampling.models import GFN
from energy_sampling.utils import load_yaml, dict2namespace
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.models.utils import load_encoder
from scipy.spatial.distance import cdist

"""load relevant data"""
eval_config = 'eval.yaml'
args = dict2namespace(load_yaml(eval_config))
override_resample = args.override_resample
generator_path = args.generator_path  # generator model
generator_config_path = generator_path.replace('model.pt', 'model_config.npy')
eval_molecules_path = args.eval_molecules_path  # molecules to evaluate on
encoder_model_path = args.autoencoder_path  # autoencoder model
opt_crystals_path = args.opt_crystals_path  # pre-optimized evaluation samples
csd_crystals_path = args.csd_crystals_path  # qm9-like experimental samples
fully_sampled_crystals_path = 'p6'  # a subset of crystals which have been exhaustively sampled

if not os.path.exists('eval_results'):
    os.mkdir('eval_results')
# load stuff here
gfn_model_config = np.load(generator_config_path, allow_pickle=True).item()
gfn_model_state_dict = torch.load(generator_path, weights_only=True)
gfn_model = GFN(**gfn_model_config)
gfn_model.load_state_dict(gfn_model_state_dict)
gfn_model.device = args.device
gfn_model.to(args.device)

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
                                   bounding_coeff=args.bounding_coeff,
                                   niggli_coeff=args.niggli_coeff,
                                   )

"""
for each space group, for the eval set, sample and record energies, cell parameters
"""
if not os.path.exists('sg_results.npy') or override_resample:

    eval_mols = torch.load(eval_molecules_path, weights_only=False)[:args.eval_mols_to_sample]
    sg_sampling_dict = {sg: {} for sg in args.sgs_to_sample}

    for sg in args.sgs_to_sample:
        (sg_sampling_dict[sg]['cell_params'],
         sg_sampling_dict[sg]['energies'],
         sg_sampling_dict[sg]['densities'],
         sg_sampling_dict[sg]['samples']) = sample_from_generator(
            gfn_model,
            args.eval_batch_size,
            eval_mols,
            sg,
            args.eval_T,
            args.eval_samples_per_mol,
            args.device,
            energy_function,
            encoder
        )
    np.save('sg_results', sg_sampling_dict)

"""
for each csd molecule, sample and record energies, cell parameters in the target SG
"""
if not os.path.exists('csd_results.npy') or override_resample:
    csd_mols = torch.load(csd_crystals_path, weights_only=False)
    csd_mols = [mol for mol in csd_mols if int(mol.sg_ind) == 2]  # todo relax in future
    csd_mols = csd_mols[:args.csd_mols_to_sample]

    identifiers = [mol.identifier for mol in csd_mols]
    csd_sampling_dict = {ident: {} for ident in identifiers}

    for ind in tqdm(range(len(csd_mols))):
        id = identifiers[ind]
        mol = csd_mols[ind]
        sg = mol.sg_ind
        (csd_sampling_dict[id]['cell_params'],
         csd_sampling_dict[id]['energies'],
         csd_sampling_dict[id]['densities'],
         csd_sampling_dict[id]['samples']) = sample_from_generator(
            gfn_model,
            args.eval_batch_size,
            [mol for _ in range(args.eval_batch_size)],
            sg,
            args.eval_T,
            int(max(1, args.csd_samples_per_mol / args.eval_batch_size)),
            args.device,
            energy_function,
            encoder
        )
    np.save('csd_results', csd_sampling_dict)
    print('getting RDFs')
    csd_rdf_dists, rr = sample_csd_rdf_dists(csd_mols,
                                             csd_sampling_dict,
                                             args.eval_batch_size,
                                             args.device)
    np.save('csd_rdfs', csd_rdf_dists.cpu().detach().numpy())

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

"""SG and molecule properties distributions"""
if args.show_figs:
    sg_sampling_dict = np.load('sg_results.npy', allow_pickle=True).item()

    # Create the three figures
    fig1 = create_energy_distribution_plot(sg_sampling_dict)
    fig2 = create_density_distribution_plot(sg_sampling_dict)
    fig3 = create_cell_params_variance_plot(sg_sampling_dict)

    # Display the figures
    fig1.show(renderer='browser')
    fig2.show(renderer='browser')
    fig3.show(renderer='browser')

"""Hit rate vs synthetic"""

"""Hit rate vs CSD"""
# analyze & preprocess CSD samples
csd_mols = torch.load(csd_crystals_path, weights_only=False)
csd_mols = [mol for mol in csd_mols if int(mol.sg_ind) == 2]  # todo relax in future
csd_mols = csd_mols[:args.csd_mols_to_sample]
identifiers = [elem.identifier for elem in csd_mols]
csd_batch = collate_data_list(csd_mols)
csd_batch.box_analysis()
csd_clusters = csd_batch.mol2cluster(cutoff=6)
csd_clusters.construct_radial_graph(cutoff=6)
ref_energies = csd_clusters.compute_silu_energy()

# load samples
csd_sampling_dict = np.load('csd_results.npy', allow_pickle=True).item()
csd_rdf_dists = np.load('csd_rdfs.npy', allow_pickle=True)

# funnel figs
if args.show_figs:
    funnel_figs = []
    for ind in range(len(csd_mols)):
        samples = csd_sampling_dict[identifiers[ind]]
        funnel_figs.append(crystal_sample_funnel_plot(
            packing_coeff=samples['densities'].flatten(),
            energies=samples['energies'].flatten(),
            dists=torch.tensor(csd_rdf_dists)[ind],
            ref_energies=torch.tensor([ref_energies[ind]]),
            ref_packing_coeff=csd_clusters[ind].packing_coeff
        ))
    [f.show(renderer='browser') for f in funnel_figs]

# Divergence between lattice distance sets
js_divs = np.array(sample_csd_lattice_divs(csd_mols, csd_sampling_dict))
#
# dmats = []
# for ind in range(len(csd_mols)):
#     samples = csd_sampling_dict[identifiers[ind]]
#     dens = samples['densities'].flatten()
#     ens = samples['energies'].flatten()
#     std_den = (dens - np.mean(dens)) / np.std(dens)
#     std_en = (ens - np.mean(ens)) / np.std(ens)
#
#     ref_en = np.array(float(ref_energies[ind]))[None]
#     ref_den = np.array(float(csd_clusters[ind].packing_coeff))[None]
#     std_ref_en = (ref_en - np.mean(ens)) / np.std(ens)
#     std_ref_den = (ref_den - np.mean(dens)) / np.std(dens)
#
#     scat_dists = cdist(np.stack([std_den, std_en]).T, np.stack([std_ref_den, std_ref_en]).T)
#
#     js_dists = js_divs[ind]
#     import plotly.graph_objects as go
#
#     go.Figure(
#         go.Scatter(x=scat_dists.flatten(), y=js_dists, mode='markers', marker_color=np.log10(csd_rdf_dists[ind]))).show(
#         renderer='browser', marker_colorscale='viridis')

aa = 1



""" # Niggli checks
samples_list = sg_sampling_dict[2]['samples']
samplist = []
for i in range(len(samples_list)):
    for j in range(len(samples_list[i])):
        samplist.extend(samples_list[i][j])
crystal_batch = collate_data_list(samplist)
crystal_batch.box_analysis()
reduced_opt_lengths = torch.zeros((crystal_batch.num_graphs, 6), dtype=torch.float32)
for ind in tqdm(range(len(reduced_opt_lengths))):
    reduced_opt_lengths[ind] = torch.tensor(get_niggli_cell(crystal_batch, ind))

csd_mols = torch.load(csd_crystals_path)
samples_list = csd_mols
crystal_batch = collate_data_list(samples_list)
crystal_batch.box_analysis()
reduced_opt_lengths = torch.zeros((crystal_batch.num_graphs, 6), dtype=torch.float32)
for ind in tqdm(range(len(reduced_opt_lengths))):
    reduced_opt_lengths[ind] = torch.tensor(get_niggli_cell(crystal_batch, ind))

"""
