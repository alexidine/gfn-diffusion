import os

import numpy as np
import plotly.graph_objects as go
import torch
from ase.spacegroup import Spacegroup

from energy_sampling.eval.paper1_results.figures import general_figs, cluster_comparison_fig, dim_reduction_fig, \
    make_thermo_table, parity_fig
from energy_sampling.eval.paper1_results.utils import get_gfn_samples, \
    cluster_thermo_analysis, get_color_set, umap_hdbscan_clustering, generator_reward
from energy_sampling.models import GFN
from energy_sampling.utils import uniform_discretizer
from examples.crystal_search_reporting import batch_compack
from mxtaltools.analysis.crystal_rdf import compute_rdf_distmat
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.common.geometry_utils import crystal_parameter_distmat
from mxtaltools.dataset_utils.utils import collate_data_list

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


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


if __name__ == '__main__':
    # acridine lj config
    run = 'xuldud'

    if run == 'acridine':
        run_name = 'acr_lj'
        device = 'cuda'
        num_samples = 10000
        batch_size = 1000
        energy_function = 'elj'  # 'elj', 'lj' 'uma
        n_steps = 100  # critical to get this right!
        sg_ind = 2
        zp = 1
        kT = 2.5
        clusters_to_analyze = 12
        units = 'LJ en /mol'

        model_path = rf"D:\crystal_datasets\acridine\best_acr_lj_sg{sg_ind}_zp{zp}_2_model_eval.pt"
        config_path = rf"D:\crystal_datasets\acridine\acr_lj_sg{sg_ind}_zp{zp}_2_model_config.npy"
        molecule_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
        dataset_path = rf"D:\crystal_datasets\acridine\acridine_sg{sg_ind}_zp{zp}.pt"
        results_dir = rf"D:\crystal_datasets\gfn_results"
        results_path = os.path.join(results_dir, rf"{run_name}_sg{sg_ind}_zp{zp}.pt")

        exp_sample_path = None


    elif run == 'xuldud':
        device = 'cuda'
        num_samples = 50000
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

    reload_results = False
    show_figs = True
    write_figs = True
    save_results = True
    overwrite_results = True

    "Load Relevant Dataset"
    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])
    data_batch = collate_data_list(dataset, max_z_prime=max_z_prime)
    data_latents = data_batch.latent_params()

    "load presampled results"
    if reload_results and os.path.exists(results_path):
        results_dict = torch.load(results_path, weights_only=False)
    else:
        results_dict = sample_and_analyze()

    sample_batch = results_dict['sample_batch']
    samples = results_dict['samples']
    sample_energy = results_dict['sample_energy']
    sample_cp = results_dict['sample_cp']
    sample_latents = sample_batch.latent_params()  # results_dict['sample_latents']  # these are sometimes wrong for xul
    if energy_function == 'uma':
        sample_batch.uma = sample_batch.uma_pot / (
                sample_batch.sym_mult * sample_batch.z_prime) - sample_batch.uma_gas_pot

    rewards = generator_reward(
        sample_batch,
        None,
        max_z_prime,
        energy_function=energy_function,
        temperature=kT,
        energy_clip=None
    )

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

        sample_batch.plot_batch_cell_params(space='real', ref_dist=ebatch.full_cell_parameters().repeat(2, 1),
                                            show=True)

    "Clustering"
    dmat = crystal_parameter_distmat(sample_latents).fill_diagonal_(0).detach()
    basin_weights, cluster_labels, cluster_prob, basin_inds = umap_hdbscan_clustering(
        dmat, sample_energy,
        n_components=6,
        n_neighbors=10,
        min_dist=0.01,
        min_cluster_size=50,
        min_samples=10,
        kT=kT,
    )
    top_cluster_inds = torch.argsort(basin_weights.sum(0), descending=True).flatten()

    """
    Make Figures
    """
    masks = np.array([cluster_labels == ind for ind in np.unique(cluster_labels)])
    mask_sorts = np.argsort([sum(m) for m in masks])[::-1]
    sorted_masks = masks[mask_sorts]
    sample_batch.plot_batch_cell_params(space='real',
                                        aux_dists=[sample_batch.full_cell_parameters()[m] for m in
                                                   masks[mask_sorts[:10]] if
                                                   sum(m) > 1])

    significant_weight_clusters = sum(basin_weights.sum(0) > (basin_weights.sum(0)[0] * 0.1))
    clusters_to_analyze = min(len(top_cluster_inds), min(clusters_to_analyze, significant_weight_clusters))
    cluster_color = get_color_set(clusters_to_analyze, alpha=0.7)

    fig_dict = {}
    fig_dict = general_figs(fig_dict, sample_batch, sample_energy, data_batch, units=units)

    min_ens, Zb, Fb, basin_probs, mean_rho, Sb, mean_E = cluster_thermo_analysis(basin_weights, sample_energy, kT,
                                                                                 sample_cp, cluster_labels,
                                                                                 top_cluster_inds)

    fig_dict['clusters'] = cluster_comparison_fig(top_cluster_inds,
                                                  sample_cp, sample_energy, cluster_labels,
                                                  sample_batch, 8, sample_latents,
                                                  cluster_color,
                                                  )
    fig_dict['Thermo Table'] = make_thermo_table(Zb, basin_probs, Fb, mean_E, min_ens, Sb, mean_rho,
                                                 cluster_labels, clusters_to_analyze,
                                                 units=units)

    fig_dict['Dim Reduction'] = dim_reduction_fig(dmat,
                                                  cluster_labels,
                                                  clusters_to_analyze,
                                                  cluster_color,
                                                  basin_inds)
    fig_dict['tb'] = parity_fig(
        y_raw=results_dict['log_pfs'] + results_dict['learned_log_z'],
        x_raw=results_dict['log_pbs'] + rewards,
        y_label=r'$log(P_f(\tau)) + log(R(x_T))$',
        x_label=r'$log(P_b(\tau|x_T)) + log(Z_\theta)$',
    )

    view_and_save_figs(fig_dict)

    'best samples analysis'
    best_samples = [samples[ind] for ind in top_cluster_inds[:clusters_to_analyze]]
    best_batch = collate_data_list(best_samples)
    clusters = best_batch.mol2cluster(cutoff=6)

    for ii, ind in enumerate(top_cluster_inds[:clusters_to_analyze]):
        mol = ase_mol_from_crystaldata(clusters, index=ii, mode='unit cell')
        mol.info['spacegroup'] = Spacegroup(sg_ind, setting=1)
        mol.write(os.path.join(results_dir, f'{run_name}_{ii}.cif'))

    bin_edges = torch.linspace(0, 6, sample_batch.rdf.shape[-1], )
    dmat = compute_rdf_distmat(sample_batch.rdf[top_cluster_inds[:clusters_to_analyze]], bin_edges,
                               chunk_size=10000)
    go.Figure(go.Heatmap(z=dmat)).show()

    clusters.visualize(mode='unit cell')

    matchess, rmsdss = [], []
    for ind in range(best_batch.num_graphs):
        matches, rmsds = batch_compack([ind for ind in range(best_batch.num_graphs)],
                                       best_samples,
                                       collate_data_list([best_samples[ind]]).mol2cluster(cutoff=6))
        matchess.append(matches)
        rmsdss.append(rmsds)

    matched = np.stack(matchess)
    rmsds = np.stack(rmsdss)
    go.Figure(go.Scatter(x=(rmsds / matched).flatten(), y=dmat.flatten(), mode='markers')).show()
