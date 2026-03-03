import os

import numpy as np
import torch
from umap import UMAP

from energy_sampling.eval.paper1_results.figures import sample_summary_table, var_table
from energy_sampling.eval.paper1_results.utils import generator_reward, new_local_analysis, get_kinetic_basins_sparse, \
    sample_and_analyze, get_kinetic_basins_light, recompute_ens, init_ucell_pos
from mxtaltools.dataset_utils.utils import collate_data_list

torch.cuda.set_per_process_memory_fraction(0.9, device=0)

if __name__ == '__main__':
    run = 'acridine_14_elj'

    if run == 'acridine_14_uma':
        run_name = 'acr_uma'
        device = 'cuda'
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

    reload_results = True
    show_figs = True
    write_figs = False
    save_results = True
    overwrite_results = True
    num_samples = 50000
    batch_size = 1000
    alpha = 5.0

    "Load Relevant Dataset"
    molecule = torch.load(molecule_path, weights_only=False)
    dataset = torch.load(dataset_path, weights_only=False)
    max_z_prime = max([int(elem.max_z_prime) for elem in dataset])

    "load presampled results"
    if reload_results and os.path.exists(results_path):
        results_dict = torch.load(results_path, weights_only=False)
    else:
        results_dict = sample_and_analyze(config_path, model_path, device, num_samples, max_z_prime, n_steps,
                                          batch_size, sg_ind, zp,
                                          energy_function, molecule, save_results, results_path, overwrite_results)
    samples = results_dict['samples']

    "analyze experimental samples"
    # if exp_sample_path is not None:
    #     esamples = load_experimental_structure(exp_sample_path, sg_ind, zp, device, max_z_prime, molecule,
    #                                            energy_function)
    #     samples = samples + esamples

    if energy_function != 'uma':
        kk = ['uma', 'uma_pot', 'uma_gas_pot']
    else:
        kk = None
    sample_batch = collate_data_list(samples, exclude_keys=kk)
    sample_energy = recompute_ens(sample_batch, energy_function)

    sample_cp = sample_batch.packing_coeff

    rewards = generator_reward(
        sample_batch,
        None,
        max_z_prime,
        energy_function=energy_function,
        temperature=kT,
        energy_clip=None
    )

    upos = init_ucell_pos(sample_batch)
    if os.path.exists('basins.pt'):
        basins = torch.load('basins.pt', weights_only=False)
    else:
        basins = get_kinetic_basins_light(sample_energy, kT, samples,
                                          sample_batch.latent_params(),
                                          upos,
                                          alpha=alpha,
                                          k_rad=10000,
                                          n_points=50,
                                          max_step_length=3)
    # torch.save(basins, 'basins.pt')
    basins = torch.load('basins.pt', weights_only=False)


    sample_metrics = new_local_analysis(sample_batch,
                                        sample_energy,
                                        d_cut=3.5,
                                        basins=basins
                                        )

    masked_energy = sample_energy.unsqueeze(0).clone()  # [1, N]
    masked_energy = masked_energy.expand(basins.shape[0], -1)  # [n_basins, N]

    masked_energy = masked_energy.masked_fill(~basins, float('inf'))

    sample_inds = masked_energy.argmin(dim=1)

    good_basins = basins.sum(-1) > 50
    fig_dict = {}
    d_cut = 3.5
    fig = sample_summary_table(sample_metrics, sample_energy[sample_inds], sample_inds = torch.arange(len(sample_inds))[good_basins])
    fig_dict[f'thermo_table'] = fig
    fig = var_table(sample_metrics, sample_energy[sample_inds], sample_inds = torch.arange(len(sample_inds))[good_basins])
    fig_dict[f'var_table'] = fig
    for fig in fig_dict.values():
        fig.show()


    X = upos.reshape(upos.shape[0], upos.shape[-1] * upos.shape[-2])  # [N, D]
    from umap import UMAP

    umap_model = UMAP(n_components=2, n_neighbors=50, min_dist=0.01,
                      init='pca', metric='euclidean', low_memory=True, n_jobs=-1)
    sample_embedding = umap_model.fit_transform(X.numpy().astype(np.float32))

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(x=sample_embedding[:, 0], y=sample_embedding[:, 1], marker_color='grey', mode='markers')
    for c_ind in range(len(basins)):
        basin = basins[c_ind]
        if sum(basin) > 25:
            fig.add_scatter(x=sample_embedding[basin, 0], y=sample_embedding[basin, 1], mode='markers',
                            name=f"{c_ind} : {basin.sum().item()}")
    fig.show()

    top_basins = basins[torch.argsort(basins.sum(-1), descending=True).flatten()][:10]
    sample_batch.plot_batch_cell_params(space='latent',
                                        aux_dists=[sample_batch.latent_params()[bas] for bas in top_basins]
                                        )
    aa = 1

    #
    # d_cut = 3.5
    # sample_metrics = new_local_analysis(d_cut,  # tuned d_cut
    #                                     sample_batch,
    #                                     sample_energy,
    #                                     rmsdmat)
    #

    # minima_inds = np.argwhere(sample_metrics['is_local_en_minimum']).flatten()
    # sorted_minima_inds = minima_inds[np.argsort(sample_energy[minima_inds])]
    #
    # maxima_inds = np.argwhere(
    #     sample_metrics['is_local_density_maximum'] & (sample_energy < np.median(sample_energy)).numpy()).flatten()
    # sorted_maxima_inds = maxima_inds[np.argsort(sample_metrics['density'][maxima_inds])][::-1]
    #
    # sample_inds = np.concatenate([sorted_minima_inds[:12], sorted_maxima_inds[:12]])  # add experimental state

    # fig = sample_summary_table(sample_metrics, sample_energy, sample_inds)
    # fig_dict[f'thermo_table'] = fig
    # fig = var_table(sample_metrics, sample_energy, sample_inds)
    # fig_dict[f'var_table'] = fig
