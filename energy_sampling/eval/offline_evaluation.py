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
from energy_sampling.eval.utils import sample_csd_rdf_dists, sample_csd_lattice_divs, \
    sample_crystals
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
gfn_model_config['conditional_flow_model'] = True
gfn_model_config['conditions_dim'] = 64 * 3
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
                                   lj_coeff=args.energy_lj_coeff,
                                   lj_turnover_pot=args.lj_turnover_pot,
                                   molecule_conditioning=args.molecule_conditioning,
                                   sg_conditioning=args.sg_conditioning,
                                   space_groups=args.space_groups,
                                   bounding_coeff=args.bounding_coeff,
                                   niggli_coeff=args.niggli_coeff,
                                   # todo add z prime info
                                   )

"""
for each space group, for the eval set, sample and record energies, cell parameters
"""
if not os.path.exists('sg_gen_results.npy') or override_resample:

    eval_mols = torch.load(eval_molecules_path, weights_only=False)[:args.eval_mols_to_sample]
    sg_gen_sampling_dict = {sg: {} for sg in args.sgs_to_sample}

    for sg in args.sgs_to_sample:
        (sg_gen_sampling_dict[sg]['cell_params'],
         sg_gen_sampling_dict[sg]['energies'],
         sg_gen_sampling_dict[sg]['densities'],
         sg_gen_sampling_dict[sg]['samples'],
         sg_gen_sampling_dict[sg]['opt_cell_params'],
         sg_gen_sampling_dict[sg]['opt_energies'],
         sg_gen_sampling_dict[sg]['opt_densities'],
         sg_gen_sampling_dict[sg]['opt_samples']
         ) = sample_crystals(
            'generator',
            gfn_model,
            args.eval_batch_size,
            eval_mols,
            sg,
            args.eval_T,
            args.eval_samples_per_mol,
            args.device,
            energy_function,
            encoder,
            do_opt=True,
        )
    np.save('sg_gen_results', sg_gen_sampling_dict)

if not os.path.exists('sg_rand_results.npy') or override_resample:
    eval_mols = torch.load(eval_molecules_path, weights_only=False)[:args.eval_mols_to_sample]
    sg_rand_sampling_dict = {sg: {} for sg in args.sgs_to_sample}

    for sg in args.sgs_to_sample:
        (sg_rand_sampling_dict[sg]['cell_params'],
         sg_rand_sampling_dict[sg]['energies'],
         sg_rand_sampling_dict[sg]['densities'],
         sg_rand_sampling_dict[sg]['samples'],
         sg_rand_sampling_dict[sg]['opt_cell_params'],
         sg_rand_sampling_dict[sg]['opt_energies'],
         sg_rand_sampling_dict[sg]['opt_densities'],
         sg_rand_sampling_dict[sg]['opt_samples']
         ) = sample_crystals(
            'random',
            gfn_model,
            args.eval_batch_size,
            eval_mols,
            sg,
            args.eval_T,
            args.eval_samples_per_mol,
            args.device,
            energy_function,
            encoder,
            do_opt=True,
        )
    np.save('sg_rand_results', sg_rand_sampling_dict)

"""
for each csd molecule, sample and record energies, cell parameters in the target SG
"""
if not os.path.exists('csd_opt_results.npy') or override_resample:
    csd_mols = torch.load(csd_crystals_path, weights_only=False)
    csd_mols = [mol for mol in csd_mols if int(mol.sg_ind) == 2]  # todo relax in future
    csd_mols = csd_mols[:args.csd_mols_to_sample]

    identifiers = [mol.identifier for mol in csd_mols]
    csd_opt_sampling_dict = {ident: {} for ident in identifiers}

    for ind in tqdm(range(len(csd_mols))):
        id = identifiers[ind]
        mol = csd_mols[ind]
        sg = mol.sg_ind
        (csd_opt_sampling_dict[id]['cell_params'],
         csd_opt_sampling_dict[id]['energies'],
         csd_opt_sampling_dict[id]['densities'],
         csd_opt_sampling_dict[id]['samples'],
         csd_opt_sampling_dict[id]['opt_cell_params'],
         csd_opt_sampling_dict[id]['opt_energies'],
         csd_opt_sampling_dict[id]['opt_densities'],
         csd_opt_sampling_dict[id]['opt_samples']
         ) = sample_crystals(
            'generator',
            gfn_model,
            args.eval_batch_size,
            [mol for _ in range(args.eval_batch_size)],
            sg,
            args.eval_T,
            int(max(1, args.csd_samples_per_mol / args.eval_batch_size)),
            args.device,
            energy_function,
            encoder,
            do_opt=True,
        )
    np.save('csd_opt_results', csd_opt_sampling_dict)

    csd_rand_sampling_dict = {ident: {} for ident in identifiers}

    for ind in tqdm(range(len(csd_mols))):
        id = identifiers[ind]
        mol = csd_mols[ind]
        sg = mol.sg_ind
        (csd_rand_sampling_dict[id]['cell_params'],
         csd_rand_sampling_dict[id]['energies'],
         csd_rand_sampling_dict[id]['densities'],
         csd_rand_sampling_dict[id]['samples'],
         csd_rand_sampling_dict[id]['opt_cell_params'],
         csd_rand_sampling_dict[id]['opt_energies'],
         csd_rand_sampling_dict[id]['opt_densities'],
         csd_rand_sampling_dict[id]['opt_samples']
         ) = sample_crystals(
            'random',
            gfn_model,
            args.eval_batch_size,
            [mol for _ in range(args.eval_batch_size)],
            sg,
            args.eval_T,
            int(max(1, args.csd_samples_per_mol / args.eval_batch_size)),
            args.device,
            energy_function,
            encoder,
            do_opt=True,
        )
    np.save('csd_rand_results', csd_rand_sampling_dict)
    print('getting RDFs')
    csd_rdf_dists, rr = sample_csd_rdf_dists(csd_mols,
                                             csd_opt_sampling_dict,
                                             args.rdfs_batch_size,
                                             args.device)
    np.save('csd_opt_rdfs', csd_rdf_dists.cpu().detach().numpy())
    csd_rdf_dists, rr = sample_csd_rdf_dists(csd_mols,
                                             csd_rand_sampling_dict,
                                             args.rdfs_batch_size,
                                             args.device)
    np.save('csd_rand_rdfs', csd_rdf_dists.cpu().detach().numpy())

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
    sg_gen_sampling_dict = np.load('sg_gen_results.npy', allow_pickle=True).item()
    sg_rand_sampling_dict = np.load('sg_rand_results.npy', allow_pickle=True).item()

    # Create the three figures
    fig1 = create_energy_distribution_plot(sg_gen_sampling_dict, sg_rand_sampling_dict)
    fig2 = create_density_distribution_plot(sg_gen_sampling_dict, sg_rand_sampling_dict)
    fig3 = create_cell_params_variance_plot(sg_gen_sampling_dict)

    # Display the figures
    fig1.show(renderer='browser')
    fig2.show(renderer='browser')
    fig3.show(renderer='browser')

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    space_groups = sorted(sg_gen_sampling_dict.keys())
    for sg in space_groups:
        gen_dict = sg_gen_sampling_dict[sg]
        rand_dict = sg_rand_sampling_dict[sg]
        fig = go.Figure()

        fig.add_scatter(x=gen_dict['opt_densities'].flatten(), y=gen_dict['opt_energies'].flatten(),
                        marker_color='blue', marker_size=1,
                        showlegend=False, mode='markers')

        fig.add_scatter(x=rand_dict['opt_densities'].flatten(), y=rand_dict['opt_energies'].flatten(),
                        marker_color='red', marker_size=1,
                        showlegend=False, mode='markers')
    fig.update_xaxes(range=[0.55, 0.95])
    fig.update_yaxes(range=[-500, 0])
    fig.show(renderer='browser')

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px

    space_groups = sorted(sg_gen_sampling_dict.keys())
    for sg in space_groups:
        gen_dict = sg_gen_sampling_dict[sg]
        rand_dict = sg_rand_sampling_dict[sg]
        fig = make_subplots(rows=2, cols=2, subplot_titles=['Gen', 'Gen + Opt', 'Rand', 'Rand + Opt'])

        fig.add_scatter(x=gen_dict['densities'].flatten(), y=gen_dict['energies'].flatten(), row=1, col=1,
                        showlegend=False, mode='markers')
        fig.add_scatter(x=gen_dict['opt_densities'].flatten(), y=gen_dict['opt_energies'].flatten(), row=1, col=2,
                        showlegend=False, mode='markers')
        fig.add_scatter(x=rand_dict['densities'].flatten(), y=rand_dict['energies'].flatten(), row=2, col=1,
                        showlegend=False, mode='markers')
        fig.add_scatter(x=rand_dict['opt_densities'].flatten(), y=rand_dict['opt_energies'].flatten(), row=2, col=2,
                        showlegend=False, mode='markers')
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[-500, 500])
    fig.show(renderer='browser')

    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_2']
    # 1d Histograms
    colors = ['red', 'blue']
    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for sind, samples in enumerate([gen_dict['opt_cell_params'], rand_dict['opt_cell_params']]):
        samples = np.concatenate(samples)

        for i in range(12):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(go.Violin(
                x=samples[:, i], y=[0 for _ in range(len(samples))], side='positive', orientation='h', width=4,
                name='rand' if sind == 1 else 'gen',
                legendgroup='rand' if sind == 1 else 'gen',
                showlegend=True if i == 0 else False,
                meanline_visible=True, bandwidth=float(np.ptp(samples[:, i]) / 100), points=False,
                line_color=colors[sind],
            ),
                row=row, col=col
            )

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', violinmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.show(renderer='browser')

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
ref_energies = csd_clusters.compute_LJ_energy()
ref_densities = csd_clusters.packing_coeff

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
csd_opt_traj = csd_batch.optimize_crystal_parameters(**optim_kwargs)
csd_opt_batch = collate_data_list(csd_opt_traj[-1])
opt_ref_energies = csd_opt_batch.lj
opt_ref_densities = csd_opt_batch.packing_coeff

# load samples
csd_opt_sampling_dict = np.load('csd_opt_results.npy', allow_pickle=True).item()
csd_opt_rdf_dists = np.load('csd_opt_rdfs.npy', allow_pickle=True)
csd_rand_sampling_dict = np.load('csd_rand_results.npy', allow_pickle=True).item()
csd_rand_rdf_dists = np.load('csd_rand_rdfs.npy', allow_pickle=True)


# funnel figs
if args.show_figs:
    funnel_figs = []
    for ind in range(len(csd_mols)):
        samples = csd_opt_sampling_dict[identifiers[ind]]
        funnel_figs.append(crystal_sample_funnel_plot(
            packing_coeff=samples['opt_densities'].flatten(),
            energies=samples['opt_energies'].flatten(),
            dists=torch.tensor(csd_opt_rdf_dists)[ind],
            ref_energies=opt_ref_energies[ind, None],
            ref_packing_coeff=opt_ref_densities[ind, None]
        ))
    [f.show(renderer='browser') for f in funnel_figs]

# Divergence between lattice distance sets
js_divs = np.array(sample_csd_lattice_divs(csd_mols, csd_opt_sampling_dict))
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
