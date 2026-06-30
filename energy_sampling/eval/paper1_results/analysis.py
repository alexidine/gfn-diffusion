import os

import os

import torch

from energy_sampling.eval.paper1_results.figures import combo_fig
from energy_sampling.eval.paper1_results.utils import sample_and_analyze, \
    dmat_local_analysis, combo_fig_analysis, clustering
from energy_sampling.utils import load_yaml, dict2namespace
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance, compute_rdf_distmat
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.AL_mace_utils import load_mace_model
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
                                          config.overwrite_results,
                                          config.rdf_type)
    samples = results_dict['samples']

    "analyze experimental samples"
    if config.exp_sample_path is not None:
        if config.energy_function == 'uma':
            pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
            predictor = init_uma_crystal_predictor(pred_path, device='cuda')
        elif config.energy_function == 'mace':
            pred_path = r"C:\Users\mikem\Downloads\acr_112025_mh1_stagetwo.model"
            predictor = load_mace_model(pred_path, device='cuda', dtype=torch.torch.float32)
        else:
            predictor = None

        exp_crystals = torch.load(config.exp_sample_path, weights_only=False)
        if hasattr(exp_crystals, 'is_batch'):
            if exp_crystals.is_batch:
                exp_crystals = exp_crystals.batch_to_list()
        if hasattr(config, 'identifiers'):
            if config.identifiers is not None:
                exp_crystals = [elem for elem in exp_crystals if elem.identifier in config.identifiers]

        ebatch = collate_data_list(exp_crystals)
        computes = ['lj', 'qlj', 'elj', 'silu', 'rdf', 'reduction_en']
        computes.append(config.energy_function)
        computes = list(set(computes))

        with torch.no_grad():
            ebatch.cuda()
            ebatch.analyze(computes, rdf_mode=config.rdf_type,
                           assign_outputs=True,
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
        rdf_bins = sample_batch.rdf_bins[0]
        with torch.no_grad():
            dmat = compute_rdf_distmat(sample_batch.rdf.cuda(), rdf_bins.cuda()).cpu()
        results_dict['dmat'] = dmat
    else:
        dmat = results_dict['dmat']

    """start neighborhood analysis"""
    d_metrics = []
    bins = sample_batch.rdf_bins[0].to('cuda')
    d = torch.cat([compute_rdf_distance(sample_batch.rdf[ii], sample_batch.rdf, bins.cpu()) for ii in range(50)])
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

    if config.save_results:
        if os.path.exists(results_path):
            if config.overwrite_results:
                torch.save(results_dict, results_path)
        else:
            torch.save(results_dict, results_path)

    aa = 1
    #
    # thermos = results_dict['metrics'][0]
    # cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering(results_dict, thermos, max_n_clusters = 6, min_basin_size=100)
    #
    # (basin_colorscale, basin_min_batch, indexed_cluster_labels,
    #  n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
    #  polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
    #  sample_energy, sample_inds, stats) = combo_fig_analysis(
    #     ebatch, sample_batch, results_dict, ebatch.num_graphs, sample_batch, results_dict, config.energy_function, cluster_labels, p_maxima, n_basins, indexed_cluster_labels)
    #
    # fig = combo_fig(
    #     ebatch.num_graphs,
    #     n_basins,
    #     packing_coeffs,
    #     stats,
    #     thermos,
    #     sample_inds,
    #     sample_embedding,
    #     p_maxima,
    #     polymorph_inds,
    #     new_min_inds,
    #     sample_energy,
    #     polymorph_colorscale,
    #     sample_colors,
    #     indexed_cluster_labels,
    #     basin_colorscale,
    #     basin_min_batch,
    #     polymorph_basin_index,
    #     config.energy_function
    # )
    # fig.show()
    # aa = 1

    """
    # RMSD business
    
    data = sample_batch.batch_to_list()
    if isinstance(exp_crystals, list):
        edata = exp_crystals
    else:
        edata = [exp_crystals]
    for pind in range(len(edata)):
        dists = compute_rdf_distance(ebatch.rdf[pind], sample_batch.rdf, sample_batch.rdf_bins[0])
        closest_rdfs = dists[:len(dmat)].topk(20, dim=-1, largest=False, sorted=True).indices
        close_batch = collate_data_list([data[ind] for ind in closest_rdfs])
    
        target_batch = collate_data_list([edata[pind]])
        target_batch.mol2ucell()
        target_batch.write_cif(torch.arange(target_batch.num_graphs), 'pap_ref', 'unit_cell')
        matches, rmsds = close_batch.batch_compack('pap_ref_0.cif', torch.arange(close_batch.num_graphs))
    
    """

    #
    # from mxtaltools.common.clustering import greedy_bottom_up_anchors
    #
    # anchors = greedy_bottom_up_anchors(sample_batch.latent_params(),
    #                                    sample_batch.packing_coeff,
    #                                    sample_energy,
    #                                    0.5,
    #                                    sample_energy.amin() + 3)
    # ss = sample_batch.batch_to_list()
    # expbatch = collate_data_list([ss[ind] for ind in anchors])
    # expbatch.mol2ucell()
    # expbatch.write_cif(torch.arange(expbatch.num_graphs), 'acrdin_sg14_zp1_low_en_structures', mode='unit cell')


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


'''

    # for passing top-k samples

    """
    Cluster a precomputed RDF distance matrix, take the top-N clusters by
    population, and pull the lowest-energy sample from each.
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    import plotly.graph_objects as go
    import umap

    def umap_embed(dist_matrix: np.ndarray, n_neighbors: int = 15,
                   min_dist: float = 0.1, seed: int = 0) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=2,
            metric="precomputed",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
        return reducer.fit_transform(dist_matrix)

    def plot_clusters(
            embedding: np.ndarray,
            labels: np.ndarray,
            energies: np.ndarray,
            reps: list,
            title: str = "UMAP of RDF distance matrix",
    ):
        """
        embedding : (N, 2) UMAP coords
        labels    : (N,) cluster assignments
        energies  : (N,) for hover info
        reps      : list of dicts from top_cluster_representatives (to mark reps)
        """
        rep_ids = {r["cluster_id"] for r in reps}
        rep_indices = {r["rep_index"] for r in reps}

        fig = go.Figure()

        # Background: non-top clusters in grey, single trace
        bg_mask = ~np.isin(labels, list(rep_ids))
        if bg_mask.any():
            fig.add_trace(go.Scatter(
                x=embedding[bg_mask, 0],
                y=embedding[bg_mask, 1],
                mode="markers",
                marker=dict(size=5, color="lightgrey", opacity=0.5),
                name="other",
                hovertext=[f"idx={i}<br>cluster={labels[i]}<br>E={energies[i]:.3f}"
                           for i in np.where(bg_mask)[0]],
                hoverinfo="text",
            ))

        # Top clusters: one trace each so legend toggles work
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
        for k, r in enumerate(reps):
            cid = r["cluster_id"]
            mask = labels == cid
            idxs = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode="markers",
                marker=dict(size=7, color=palette[k % len(palette)], opacity=0.85),
                name=f"cluster {cid} (n={r['size']})",
                hovertext=[f"idx={i}<br>cluster={cid}<br>E={energies[i]:.3f}"
                           for i in idxs],
                hoverinfo="text",
            ))

        # Reps as a star overlay
        rep_idx_arr = np.array(sorted(rep_indices))
        fig.add_trace(go.Scatter(
            x=embedding[rep_idx_arr, 0],
            y=embedding[rep_idx_arr, 1],
            mode="markers",
            marker=dict(size=16, color="black", symbol="star",
                        line=dict(width=1, color="white")),
            name="rep (min-E)",
            hovertext=[f"REP idx={i}<br>cluster={labels[i]}<br>E={energies[i]:.3f}"
                       for i in rep_idx_arr],
            hoverinfo="text",
        ))

        fig.update_layout(
            title=title,
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            template="plotly_white",
            width=900, height=700,
            legend=dict(itemsizing="constant"),
        )
        return fig

    def top_cluster_representatives(
            dist_matrix: np.ndarray,
            energies: np.ndarray,
            n_clusters: int = 10,
            top_n: int = 5,
            linkage: str = "average",
    ):
        """
        Parameters
        ----------
        dist_matrix : (N, N) symmetric, zero diagonal, precomputed distances (e.g. RDF-EMD)
        energies    : (N,) per-sample energies
        n_clusters  : total clusters to form. Set larger than top_n so the top-N
                      selection is meaningful; ~2-3x top_n is a reasonable default.
        top_n       : how many of the largest clusters to return reps for.
        linkage     : 'average' (default), 'complete', or 'single'.

        Returns
        -------
        reps : list of dicts, one per selected cluster, ordered by population (desc).
               Each dict has: cluster_id, size, rep_index, rep_energy, member_indices.
        labels : (N,) cluster assignments for all samples.
        """
        dist_matrix = np.asarray(dist_matrix)
        energies = np.asarray(energies)
        assert dist_matrix.shape[0] == dist_matrix.shape[1] == energies.shape[0]

        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
        labels = clusterer.fit_predict(dist_matrix)

        # Rank clusters by population
        cluster_ids, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        top_clusters = cluster_ids[order][:top_n]

        reps = []
        for cid in top_clusters:
            members = np.where(labels == cid)[0]
            rep = members[np.argmin(energies[members])]
            reps.append({
                "cluster_id": int(cid),
                "size": int(members.size),
                "rep_index": int(rep),
                "rep_energy": float(energies[rep]),
                "member_indices": members,
            })

        return reps, labels

    # --- load your data ---
    # dist_matrix = np.load("rdf_5emd_distances.npy")
    # energies    = np.load("energies.npy")

    # demo
    dist_matrix = dmat
    energies = sample_energy

    reps, labels = top_cluster_representatives(
        dist_matrix, energies, n_clusters=15, top_n=5
    )

    print(f"{'rank':>4} {'cid':>4} {'size':>5} {'rep_idx':>8} {'rep_E':>10}")
    for rank, r in enumerate(reps, 1):
        print(f"{rank:>4} {r['cluster_id']:>4} {r['size']:>5} "
              f"{r['rep_index']:>8} {r['rep_energy']:>10.4f}")

    rep_indices = [r["rep_index"] for r in reps]

    emb = umap_embed(dist_matrix, n_neighbors=500, min_dist=0.75)
    fig = plot_clusters(emb, labels, energies, reps)
    # fig.write_html("umap_clusters.html")
    fig.show()
    anchors = torch.tensor(rep_indices)
    ss = sample_batch.batch_to_list()
    expbatch = collate_data_list([ss[ind] for ind in anchors]).to('cuda')
    opt_config = {
        "optim_target": "mace",  # lj qlj elj silu ellipsoid classification_score rdf_score rdf_dist latent_dist
        "enforce_reduced": False,
        "compression_factor": 0.0,
        "cutoff": 10,  # can be as low as 6 for SiLU, 10 otherwise
        "init_lr": 1e-4,
        "convergence_eps": 1e-8,
        "optimizer_func": "rprop",  # NOTE rprop is by far the fastest and most reliable
        "anneal_lr": False,
        "grad_norm_clip": 0.1,
        "show_tqdm": True,
        "max_num_steps": 500,
        "rdf_warmup": None,
        "target_packing_coeff": None,
        "umbrella": False,
        "umbrella_sigma": 0.25,
        "umbrella_epsilon": 40.0,
        'predictor': predictor,
    }
    opt_out, opt_record = expbatch.optimize_crystal_parameters(return_record=True, **opt_config)
    opbatch = collate_data_list(opt_out)
    opbatch.mol2ucell()
    opbatch.write_cif(torch.arange(opbatch.num_graphs), 'acrdin_sg14_zp1_low_en_structures2', mode='unit cell')

'''