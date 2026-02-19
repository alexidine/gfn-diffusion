import os

import numpy as np
import torch
from ase.spacegroup import Spacegroup
from plotly.subplots import make_subplots
from umap import UMAP

from energy_sampling.eval.paper1_results.figures import general_figs, parity_fig, sample_summary_table
from energy_sampling.eval.paper1_results.utils import get_gfn_samples, \
    generator_reward, get_rmsdmat, local_analysis, analyze_samples, load_experimental_structure
from energy_sampling.models import GFN
from energy_sampling.utils import uniform_discretizer, thin_large_dmat_block, featurize_dataset
from examples.crystal_search_reporting import batch_compack
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.common.geometry_utils import crystal_parameter_distmat
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def dataset_tb_analysis():
    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])

    # canonicalize rotvecs (upper half-plane)
    batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    batch.canonicalize_orientation()
    orientations = batch.aunit_orientation
    for ind, elem in enumerate(dataset):
        elem.aunit_orientation = orientations[ind][None, ...]

    # canonicalize aunit parameterizations
    batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    batch.canonicalize_zp_aunits()
    aunits = batch.aunit_centroid
    for ind, elem in enumerate(dataset):
        elem.aunit_centroid = aunits[ind][None, ...]

    # filter invalid latents
    batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    latents = batch.latent_params()
    good_inds = torch.argwhere(torch.all(latents.abs() <= 1, dim=1))  # valid latent space
    dataset = [dataset[ind] for ind in good_inds]

    # filter reasonable densities
    dataset = [elem for elem in dataset if elem.packing_coeff >= 0.55]
    dataset = [elem for elem in dataset if elem.packing_coeff <= 0.95]

    # filter near-identical samples
    d_cut = 0.05  # should be relatively sparse or the local density bias becomes large
    latents = collate_data_list(dataset).latent_params()
    keep = thin_large_dmat_block(latents.to(device),
                                 torch.tensor([elem.lj for elem in dataset], device=device),
                                 d_cut).cpu()
    keep_inds = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    dataset = [dataset[i] for i in keep_inds]

    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()

    pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
    dataset = featurize_dataset(dataset,
                                device,
                                energy_function,
                                uma_path=pred_path,
                                batch_size=10000)

    dataset = [elem for elem in dataset if elem.packing_coeff >= 0.55]
    dataset = [elem for elem in dataset if elem.packing_coeff <= 0.95]
    dataset = [elem for elem in dataset if elem.reduction_en <= 1e-3]

    data_batch = collate_data_list(dataset)

    terminal_states = data_batch.latent_params()
    discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)
    condition = torch.zeros((len(terminal_states), 1), device=gfn_model.device)

    with torch.no_grad():
        states, log_pfs, log_pbs, log_flow = gfn_model.get_traj_bwd(
            terminal_states.clone().to(gfn_model.device),
            discretizer, condition, return_gauss_params=False
        )
        pbs = log_pbs.sum(-1).cpu().detach()
        pfs = log_pfs.sum(-1).cpu().detach()
        del log_pbs, log_pfs

    if energy_function == 'uma':
        denergy = data_batch.uma_pot / (data_batch.sym_mult * data_batch.z_prime) - data_batch.uma_gas_pot
        data_batch.uma = denergy
    else:
        lj_mean, lj_std, uma_mean, uma_std = [-20.6, 5.7, -3.4, 1.5]
        atomwise_energy = data_batch.elj / (data_batch.num_atoms / data_batch.z_prime)
        atomwise_fixed = (atomwise_energy - lj_mean) / lj_std * uma_std + uma_mean
        denergy = atomwise_fixed * (data_batch.num_atoms / data_batch.z_prime)
    rewards = generator_reward(
        data_batch,
        None,
        max_z_prime,
        energy_function=energy_function,
        temperature=kT,
        energy_clip=None
    )
    parity_fig(
        y_raw=pfs + gfn_model.flow_model().item(),
        x_raw=pbs + rewards,
        y_label=r'$log(P_f(\tau)) + log(R(x_T))$',
        x_label=r'$log(P_b(\tau|x_T)) + log(Z_\theta)$',
    ).show()
    data_batch.plot_batch_density_funnel(override_energy=denergy)

    import plotly.graph_objects as go
    alpha = 1.0
    eps = 1.0e-2
    z = gfn_model.flow_model().item()
    log_importance_weight = (rewards - z) - (pfs - pbs)
    log_importance_weight = log_importance_weight.clip(max=log_importance_weight.quantile(0.99))
    importance_weight = (alpha * log_importance_weight).exp() + 1e-6
    importance_weight = (1 - eps) * importance_weight
    importance_weight += eps / len(importance_weight)
    importance_weight = importance_weight.numpy().astype(np.float64)
    importance_weight /= importance_weight.sum()
    go.Figure(go.Scatter(y=pfs + z, x=pbs + rewards, mode='markers', marker_color=np.log10(importance_weight),
                         marker_colorscale='viridis', opacity=0.5)).show()
    go.Figure(
        go.Scatter(x=data_batch.packing_coeff, y=denergy, marker_color=np.log10(importance_weight), mode='markers',
                   opacity=0.5)).show()

    exp_crystals = torch.load(exp_sample_path, weights_only=False)
    exp_crystals = [cry for cry in exp_crystals if (cry.sg_ind == sg_ind) and (cry.z_prime == zp)]
    exp_crystals = [exp_crystals[0]]
    ebatch = collate_data_list(exp_crystals, max_z_prime=zp)
    pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
    predictor = init_uma_crystal_predictor(pred_path, device=device)
    esamples = analyze_samples(
        None,  # ebatch.latent_params(),
        [molecule] * ebatch.num_graphs,
        max_z_prime,
        device,
        1000,
        sg_ind,
        zp,
        do_uma=energy_function == 'uma',
        predictor=predictor,
        overwrite_latents=False,
    )

    n_bins = esamples[0].rdf.shape[-1]
    dd = compute_rdf_distance(esamples[0].rdf, data_batch.rdf, torch.linspace(0, 6, n_bins))
    good_inds = torch.argsort(dd)[:10]
    best_samples = [dataset[ind] for ind in good_inds]
    best_batch = collate_data_list(best_samples)
    clusters = best_batch.mol2cluster(cutoff=6)

    for ii in range(len(best_samples)):
        mol = ase_mol_from_crystaldata(clusters, index=ii, mode='unit cell')
        mol.info['spacegroup'] = Spacegroup(sg_ind, setting=1)
        mol.write(os.path.join(results_dir,
                               rf'C:\Users\mikem\Projects\mxt_gfn\gfn_diffusion\energy_sampling\eval\paper1_results\{run_name}_{ii}.cif'))

    matches, rmsds = batch_compack(best_samples,
                                   dataset,
                                   ebatch.mol2cluster(cutoff=6))

    aa = 1


def sample_and_analyze():
    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()
    "Sample from GFN & process samples"
    sample_batch, sample_latents, sample_energy, sample_cp, samples, pfs, pbs = get_gfn_samples(
        num_samples, max_z_prime,
        device, n_steps, batch_size, gfn_model,
        energy_function, molecule, sg_ind, zp
    )

    results_dict = {
        'sample_batch': sample_batch,
        'sample_latents': sample_latents,
        'sample_energy': sample_energy,
        'sample_cp': sample_cp,
        'samples': samples,
        'log_pfs': pfs,
        'log_pbs': pbs,
        'learned_log_z': gfn_model.flow_model().item()
    }

    if save_results:
        if os.path.exists(results_path):
            if overwrite_results:
                torch.save(results_dict, results_path)
            else:
                pass
        else:
            torch.save(results_dict, results_path)

    return results_dict


def view_and_save_figs(fig_dict):
    for key, fig in fig_dict.items():
        if key == 'tb':
            width = 700
            height = 700
            fig.update_layout(
                width=width,
                height=height,
                font_size=24
            )

        elif key == 'staircase_fig':
            width = 2000
            height = 1300
            fig.update_layout(
                width=width,
                height=height,
                font_size=24
            )

        elif key == 'std_marginals_fig':
            width = 1920
            height = 1080
            fig.update_layout(
                width=width,
                height=height,
                font_size=20
            )

        elif key == 'density_funnel_fig':
            width = 700
            height = 700
            fig.update_layout(
                width=width,
                height=height,
                font_size=24
            )

        elif key == 'clusters':
            width = 1800
            height = 1000
            fig.update_layout(
                width=width,
                height=height,
                font_size=24,
            )
            fig.update_annotations(font_size=24)

        elif key == 'Thermo Table':
            width = 800
            height = 300
            fig.update_layout(
                width=width,
                height=height
            )

        elif key == 'Dim Reduction':
            width = 800
            height = 800
            fig.update_layout(
                width=width,
                height=height
            )

        if show_figs:
            fig.show()

        if write_figs:
            fig.write_image(
                rf"C:\Users\mikem\OneDrive\NYU\CSD\papers\generator\{run_name}_{key.replace(' ', '_')}.png",
                height=height,
                width=width,
                scale=2)


def recompute_ens(sample_batch, energy_function):
    if energy_function == 'uma':
        sample_batch.uma = sample_batch.uma_pot / (
                sample_batch.sym_mult * sample_batch.z_prime) - sample_batch.uma_gas_pot
        sample_energy = sample_batch.uma
    elif energy_function == 'elj':
        lj_mean, lj_std, uma_mean, uma_std = [-20.6, 5.7, -3.4, 1.5]
        atomwise_energy = sample_batch.elj / (sample_batch.num_atoms / sample_batch.z_prime)
        atomwise_fixed = (atomwise_energy - lj_mean) / lj_std * uma_std + uma_mean
        sample_energy = atomwise_fixed * (sample_batch.num_atoms / sample_batch.z_prime)
    else:
        sample_energy = sample_batch.lj

    return sample_energy


if __name__ == '__main__':
    # acridine lj config
    # run = 'xuldud_61_uma'
    # run = 'xuldud_61_elj'
    # run = 'acridine_14_uma'
    run = 'acridine_14_elj'

    if run == 'acridine_14_uma':
        run_name = 'acr_uma'
        device = 'cuda'
        num_samples = 1000
        batch_size = 1000
        energy_function = 'uma'  # 'elj', 'lj' 'uma
        n_steps = 100  # critical to get this right!
        sg_ind = 14
        zp = 1
        kT = 2.5
        units = 'kJ/mol'

        model_path = rf"D:\crystal_datasets\acridine\acr11_sg{sg_ind}_zp{zp}_model_eval.pt"
        config_path = rf"D:\crystal_datasets\acridine\acr11_sg{sg_ind}_zp{zp}_model_config.npy"
        molecule_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
        dataset_path = rf"D:\crystal_datasets\acridine\acridine_sg{sg_ind}_zp{zp}.pt"
        results_dir = rf"D:\crystal_datasets\gfn_results"
        results_path = os.path.join(results_dir, rf"{run_name}_sg{sg_ind}_zp{zp}.pt")

        exp_sample_path = r"D:\crystal_datasets\acridine\prot_acrdin_crystals.pt"

    elif run == 'acridine_14_elj':
        run_name = 'acr_lj'
        device = 'cuda'
        num_samples = 10000
        batch_size = 10000
        energy_function = 'elj'  # 'elj', 'lj' 'uma
        n_steps = 100  # critical to get this right!
        sg_ind = 14
        zp = 1
        kT = 2.5
        units = 'kJ/mol'

        model_path = rf"D:\crystal_datasets\acridine\acr17_sg{sg_ind}_zp{zp}_6_model_eval.pt"
        config_path = rf"D:\crystal_datasets\acridine\acr17_sg{sg_ind}_zp{zp}_6_model_config.npy"
        molecule_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
        dataset_path = rf"D:\crystal_datasets\acridine\acridine_sg{sg_ind}_zp{zp}.pt"
        results_dir = rf"D:\crystal_datasets\gfn_results"
        results_path = os.path.join(results_dir, rf"{run_name}_sg{sg_ind}_zp{zp}.pt")

        exp_sample_path = r"D:\crystal_datasets\acridine\prot_acrdin_crystals.pt"

    elif run == 'xuldud_61_lj':
        device = 'cuda'
        num_samples = 20000
        batch_size = 2000
        energy_function = 'elj'  # 'elj', 'lj' 'uma
        n_steps = 100  # critical to get this right!
        sg_ind = 61
        run_name = f'xul_{sg_ind}'
        zp = 1
        kT = 2.5
        clusters_to_analyze = 20
        units = 'kJ/mol'
        model_path = rf"D:\crystal_datasets\xuldud\xul4_sg{sg_ind}_zp{zp}_1_model_eval.pt"
        config_path = rf"D:\crystal_datasets\xuldud\xul4_sg{sg_ind}_zp{zp}_1_model_config.npy"
        molecule_path = r"D:\crystal_datasets\xuldud\xuldud.pt"
        dataset_path = rf"D:\crystal_datasets\xuldud\xuldud_sg{sg_ind}_zp{zp}.pt"
        results_dir = rf"D:\crystal_datasets\gfn_results"
        results_path = os.path.join(results_dir, rf"{run_name}_sg{sg_ind}_zp{zp}.pt")
        exp_sample_path = r"D:\crystal_datasets\xuldud\xul_csd.pkl"

    elif run == 'xuldud_61_uma':
        device = 'cuda'
        num_samples = 10000
        batch_size = 100
        energy_function = 'uma'  # 'elj', 'lj' 'uma
        n_steps = 100  # critical to get this right!
        sg_ind = 61
        run_name = run
        zp = 1
        kT = 2.5
        clusters_to_analyze = 20
        units = 'kJ/mol'
        model_path = rf"D:\crystal_datasets\xuldud\xul6_sg{sg_ind}_zp{zp}_3_model_eval.pt"
        config_path = rf"D:\crystal_datasets\xuldud\xul6_sg{sg_ind}_zp{zp}_3_model_config.npy"
        molecule_path = r"D:\crystal_datasets\xuldud\xuldud.pt"
        dataset_path = rf"D:\crystal_datasets\xuldud\xuldud_sg{sg_ind}_zp{zp}.pt"
        results_dir = rf"D:\crystal_datasets\gfn_results"
        results_path = os.path.join(results_dir, rf"{run_name}_sg{sg_ind}_zp{zp}.pt")
        exp_sample_path = r"D:\crystal_datasets\xuldud\xul_csd.pkl"

    k_vals = [50, 100, 200]  # , 50, 100]
    reload_results = False
    show_figs = True
    write_figs = False
    save_results = True
    overwrite_results = False

    "Load Relevant Dataset"
    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])
    data_batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    data_latents = data_batch.latent_params()

    #dataset_tb_analysis()


    "load presampled results"
    if reload_results and os.path.exists(results_path):
        results_dict = torch.load(results_path, weights_only=False)
    else:
        results_dict = sample_and_analyze()
    samples = results_dict['samples']

    "analyze experimental samples"
    if exp_sample_path is not None:
        esamples = load_experimental_structure(exp_sample_path, sg_ind, zp, device, max_z_prime, molecule,
                                               energy_function)
        samples = samples + esamples

    sample_batch = collate_data_list(samples)
    sample_energy = recompute_ens(sample_batch, energy_function)

    sample_cp = sample_batch.packing_coeff
    sample_latents = sample_batch.latent_params()

    rewards = generator_reward(
        sample_batch,
        None,
        max_z_prime,
        energy_function=energy_function,
        temperature=kT,
        energy_clip=None
    )

    """distance calculations"""
    rmsdmat = get_rmsdmat(sample_batch.clone())
    lat_dmat = crystal_parameter_distmat(sample_latents).fill_diagonal_(0).detach()
    bin_edges = torch.linspace(0, 6, sample_batch.rdf.shape[-1], )
    # rdf_dmat = compute_rdf_distmat(sample_batch.rdf, bin_edges, chunk_size=10000)
    sample_metrics = local_analysis(k_vals, sample_batch, sample_energy, rmsdmat)

    """visual and local thermo analysis"""
    umap_model = UMAP(n_components=2, n_neighbors=50, min_dist=0.01, metric='precomputed')
    sample_embedding = umap_model.fit_transform(rmsdmat.numpy().astype(np.float64))

    candidates_to_eval = np.argwhere(sample_metrics[200]['is_local_en_minimum']).flatten()
    candi2 = np.argwhere(sample_metrics[200]['is_local_rho_maximum']).flatten()
    candidates_to_eval = np.unique(
        np.concatenate([candidates_to_eval, [sample_batch.num_graphs - (len(esamples) + 1)],
                        candi2]))  # add experimental state

    candidates_to_eval = candidates_to_eval[np.argsort(sample_energy[candidates_to_eval])]

    fig_dict = {}

    for k in k_vals:
        # todo nice column names
        # todo nice column groups
        fig = sample_summary_table(sample_metrics, [k], candidates_to_eval,
                                   good_keys=[
                                       'basin_min_en',
                                       'basin_std_en',
                                       'log_rho',
                                       'basin_max_rho',
                                       'grad_mag',
                                       'd_eff',
                                       'softness',
                                       'is_local_en_minimum',
                                       'is_local_rho_maximum',
                                   ])
        fig_dict[f'table_{k}'] = fig

    for k in k_vals:
        thermos = np.stack(list(sample_metrics[k].values()))
        x = sample_embedding[:, 0]
        y = sample_embedding[:, 1]
        n_thermos = thermos.shape[0]
        fig = make_subplots(rows=n_thermos // 4 + int(n_thermos % 4 > 0), cols=4,
                            subplot_titles=list(sample_metrics[200].keys()), vertical_spacing=0.05,
                            horizontal_spacing=0.05)
        for ind in range(len(thermos)):
            c = thermos[ind]
            fig.add_scatter(x=x, y=y, marker_color=c, mode='markers',
                            opacity=0.7, marker_size=4, showlegend=False,
                            marker_colorscale='icefire',
                            row=ind // 4 + 1, col=ind % 4 + 1)
        fig_dict[f'umap_{k}'] = fig

    for fig in fig_dict.values():
        fig.show()

    # todo qualitative UMAP fig over PCs of these metrics. Typically density dominates PC1 and anisotropy (log_cond) PC2. gauss_entropy active in both

    # todo per-sample reporting
    # energy, local density, d_eff, log_cond, grad_mag, escape barrier (kinetic), over a few representative samples

    import plotly.graph_objects as go

    labels = list(sample_metrics[k].keys())
    corr = np.corrcoef(thermos / (1e-3 + np.std(thermos, axis=1)[:, None]))
    np.fill_diagonal(corr, 0)
    go.Figure(go.Heatmap(z=corr, x=labels, y=labels)).show()

    fig_dict = general_figs(fig_dict, sample_batch, sample_energy, data_batch, units=units)

    fig_dict['tb'] = parity_fig(
        y_raw=results_dict['log_pfs'] + results_dict['learned_log_z'],
        x_raw=results_dict['log_pbs'] + rewards[len(results_dict['log_pbs'])],
        y_label=r'$log(P_f(\tau)) + log(R(x_T))$',
        x_label=r'$log(P_b(\tau|x_T)) + log(Z_\theta)$',
    )

    view_and_save_figs(fig_dict)

    'best samples analysis'
    # best_samples = [samples[ind] for ind in top_cluster_inds[:clusters_to_analyze]]
    # best_batch = collate_data_list(best_samples)
    # clusters = best_batch.mol2cluster(cutoff=6)
    #
    # for ii, ind in enumerate(top_cluster_inds[:clusters_to_analyze]):
    #     mol = ase_mol_from_crystaldata(clusters, index=ii, mode='unit cell')
    #     mol.info['spacegroup'] = Spacegroup(sg_ind, setting=1)
    #     mol.write(os.path.join(results_dir, f'{run_name}_{ii}.cif'))
    #
    # bin_edges = torch.linspace(0, 6, sample_batch.rdf.shape[-1], )
    # dmat = compute_rdf_distmat(sample_batch.rdf[top_cluster_inds[:clusters_to_analyze]], bin_edges,
    #                            chunk_size=10000)
    # go.Figure(go.Heatmap(z=dmat)).show()
    #
    # clusters.visualize(mode='unit cell')
    #
    # matchess, rmsdss = [], []
    # for ind in range(best_batch.num_graphs):
    #     matches, rmsds = batch_compack([ind for ind in range(best_batch.num_graphs)],
    #                                    best_samples,
    #                                    collate_data_list([best_samples[ind]]).mol2cluster(cutoff=6))
    #     matchess.append(matches)
    #     rmsdss.append(rmsds)
    #
    # matched = np.stack(matchess)
    # rmsds = np.stack(rmsdss)
    # go.Figure(go.Scatter(x=(rmsds / matched).flatten(), y=dmat.flatten(), mode='markers')).show()

"analyze experimental samples"
if False:  # exp_sample_path is not None:
    "analyze dataset"
    dbatch = collate_data_list(dataset)
    dsamples = analyze_samples(
        dbatch.latent_params(),
        [molecule] * dbatch.num_graphs,
        max_z_prime,
        device,
        1000,
        sg_ind,
        zp,
        do_uma=False,
        # predictor=predictor
    )
    exp_crystals = torch.load(exp_sample_path, weights_only=False)
    exp_crystals = [cry for cry in exp_crystals if cry.sg_ind == sg_ind]
    ebatch = collate_data_list(exp_crystals)
    esamples = analyze_samples(
        ebatch.latent_params(),
        [molecule] * ebatch.num_graphs,
        max_z_prime,
        device,
        1000,
        sg_ind,
        zp,
        do_uma=False,
        # predictor=predictor
    )

    pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
    predictor = init_uma_crystal_predictor(pred_path, device=device)
    calc = FAIRChemCalculator(predictor, task_name="omc")

    exp_crystals = torch.load(exp_sample_path, weights_only=False)
    exp_crystals = [cry for cry in exp_crystals if cry.sg_ind == sg_ind]
    ebatch = collate_data_list(exp_crystals)

    "get tb loss on experimental states"
    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()

    bwd_repeats = 250
    terminal_states = ebatch.latent_params().repeat(bwd_repeats, 1)
    discretizer = lambda bsz: uniform_discretizer(bsz, n_steps)
    condition = torch.zeros((len(terminal_states), 1), device=gfn_model.device)

    states, log_pfs, log_pbs, log_flow = gfn_model.get_traj_bwd(
        terminal_states.clone().to(gfn_model.device),
        discretizer, condition, return_gauss_params=False
    )
    gas_en = ebatch.compute_lattice_gas_phase_uma(predictor,
                                                  std_orientation=True).cpu().detach() * 96.485
    cry_en = ebatch.compute_crystal_uma(predictor=predictor,
                                        std_orientation=True).cpu().detach() * 96.485
    exp_uma = cry_en / (ebatch.sym_mult * ebatch.z_prime) - gas_en
    ebatch.uma_gas_pot = gas_en
    ebatch.uma_pot = cry_en
    ebatch.uma = exp_uma

    exp_rewards = generator_reward(
        ebatch,
        None,
        max_z_prime,
        energy_function=energy_function,
        temperature=kT,
        energy_clip=None
    ).repeat(bwd_repeats)

    parity_fig(
        y_raw=log_pfs.sum(-1).cpu().detach() + gfn_model.flow_model().item(),
        x_raw=log_pbs.sum(-1).cpu().detach() + exp_rewards.cpu().detach(),
        y_label=r'$log(P_f(\tau)) + log(R(x_T))$',
        x_label=r'$log(P_b(\tau|x_T)) + log(Z_\theta)$',
    ).show()

    alist = batch_to_ase_ucell_list(ebatch, std_orientation=True, pbc=True)

    for atoms in alist:
        atoms.calc = calc

        # Stage 1: fast, robust descent
        opt = LBFGS(UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0]), memory=50, maxstep=0.2)
        opt.run(fmax=0.1, steps=100)

        # Stage 2: clean convergence
        opt = FIRE(UnitCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0]))
        opt.run(fmax=0.03, steps=200)

    # get terminal energy

    uma_batch = atomicdata_list_to_batch([AtomicData.from_ase(atoms, task_name='omc') for atoms in alist])

    out, crashed = safe_predict_uma(predictor, uma_batch)
    cry_en = out['energy'].cpu().detach() * 96.485
    # convert back to our batch data type

    gas_en = ebatch.compute_lattice_gas_phase_uma(predictor,
                                                  std_orientation=True).cpu().detach() * 96.485
    cry_en = ebatch.compute_crystal_uma(predictor=predictor,
                                        std_orientation=True).cpu().detach() * 96.485
    exp_uma = cry_en / (ebatch.sym_mult * ebatch.z_prime) - gas_en

    fig = go.Figure(go.Scatter(x=sample_batch.packing_coeff, y=sample_energy, mode='markers'))
    fig.add_scatter(x=ebatch.packing_coeff, y=exp_uma, mode='markers', marker_size=20)
    fig.update_layout(yaxis_range=[-np.inf, 0])
    fig.show()

    nn = ebatch.full_cell_parameters().repeat(200, 1)
    nn += torch.randn_like(nn) * 0.05
    sample_batch.plot_batch_cell_params(space='real', ref_dist=nn,
                                        show=True)

    '''
    # compare RMSD to latent dist
    sample_batch.pose_aunit(std_orientation=True)
    sample_batch.build_unit_cell()
    upos = sample_batch.unit_cell_pos.reshape(sample_batch.num_graphs,
                                              sample_batch.sym_mult[0] * sample_batch.num_atoms[0], 3)
    d2 = torch.zeros_like(dmat)
    for ind in range(sample_batch.num_graphs):
        d2[ind] = (upos[ind, None, ...] - upos).norm(dim=-1).mean(-1)
    fig = go.Figure(go.Histogram2dContour(
                x=dmat[:100].flatten().numpy(),
                y=d2[:100].flatten().numpy(),
                ncontours=12,
                showscale=False,  # colorbar and (i == D - 1 and j == 0),
                #contours=dict(coloring='none', showlines=True, start=0.0001, end=0.1, size=0.04),
                #line=dict(smoothing=1.0, color='grey', width=2),
                nbinsx=50,
                nbinsy=50,
                histnorm='probability',
                showlegend=False))
    fig.show()
    '''

    #     basin_weights, cluster_labels, cluster_prob, basin_inds = umap_hdbscan_clustering(
    #         dmat, sample_energy,
    #         n_components=6,
    #         n_neighbors=10,
    #         min_dist=0.01,
    #         min_cluster_size=50,
    #         min_samples=10,
    #         kT=kT,
    #     )
    #     top_cluster_inds = torch.argsort(basin_weights.sum(0), descending=True).flatten()
    #
    #     """
    #     Make Figures
    #     """
    #     masks = np.array([cluster_labels == ind for ind in np.unique(cluster_labels)])
    #     mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    #     sorted_masks = masks[mask_sorts]
    #     sample_batch.plot_batch_cell_params(space='real',
    #                                         aux_dists=[sample_batch.full_cell_parameters()[m] for m in
    #                                                    masks[mask_sorts[:10]] if
    #                                                    sum(m) > 1])
    #
    #     significant_weight_clusters = sum(basin_weights.sum(0) > (basin_weights.sum(0)[0] * 0.1))
    #     clusters_to_analyze = min(len(top_cluster_inds), min(clusters_to_analyze, significant_weight_clusters))
    #     cluster_color = get_color_set(clusters_to_analyze, alpha=0.7)
    #
    # min_ens, Zb, Fb, basin_probs, mean_rho, Sb, mean_E = cluster_thermo_analysis(basin_weights, sample_energy, kT,
    #                                                                              sample_cp, cluster_labels,
    #                                                                              top_cluster_inds)
    #
    # fig_dict['clusters'] = cluster_comparison_fig(top_cluster_inds,
    #                                               sample_cp, sample_energy, cluster_labels,
    #                                               sample_batch, 8, sample_latents,
    #                                               cluster_color,
    #                                               )
    # fig_dict['Thermo Table'] = make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho,
    #                                              cluster_labels, clusters_to_analyze,
    #                                              units=units)
    #
    # fig_dict['Dim Reduction'] = dim_reduction_fig(dmat,
    #                                               cluster_labels,
    #                                               clusters_to_analyze,
    #                                               cluster_color,
    #                                               basin_inds)
