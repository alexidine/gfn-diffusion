"""analysis and figures for the results section of the paper"""
import os
from copy import copy

import numpy as np
import torch
from ase.io import write
from tqdm import tqdm

from energy_sampling.eval.paper1_results.figures import parity_fig, dual_energy_marginal_fig, \
    pes_cartoon, plot_dual_density_contour, combo_fig, fused_duo_embedding_fig, render_molecule_image, \
    scale_embedding_fig
from energy_sampling.eval.paper1_results.utils import combo_fig_analysis, augment_dmat2, clustering, clustering3, \
    dmat_row_indices
from mxtaltools.analysis.crystal_rdf import compute_rdf_distance
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor

# GPU_MEM_FRACTION overrides, e.g. 0.25 when a local training run shares the card
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_MEM_FRACTION', 0.9)), device=0)


PAPER_FIG_DIR = r'C:\Users\mikem\OneDrive\NYU\CSD\papers\generator'


def elj_prescale_factor(elj_results, en_scaling_factor):
    """Factor still to apply to a results file's stored `.elj`.

    new_analysis.py stamps lj_coeff before analyze(), so `.elj` is stored ALREADY
    scaled and its results carry an 'lj_coeff' key. The June analysis.py stored the
    raw sum and had no such key. Multiplying a pre-scaled value again is a silent
    x0.36 (mipcas) / x0.16 (nehzor) error, and it also shifts the 15 kT filter off
    the rows the stored dmat was built from."""
    return 1.0 if 'lj_coeff' in elj_results else float(en_scaling_factor)


def require_rdf_mode(uma_results, elj_results):
    """Both runs, and the experimental references scored here, must share one RDF mode.

    For a 13-atom molecule atomwise and envwise both give 91 channels, so a mismatch
    raises nothing and simply measures the wrong distances."""
    modes = {uma_results.get('rdf_mode'), elj_results.get('rdf_mode')}
    assert None not in modes, (
        "a results file carries no 'rdf_mode' (it predates new_analysis.py's rdf_mode key); "
        "rerun the analysis so the reference RDFs can be matched to it")
    assert len(modes) == 1, f"UMA and eLJ results use different RDF modes: {modes}"
    return modes.pop()


def require_log_r(results, tag):
    assert 'log_r' in results, (
        f"{tag} results have no 'log_r' (the exact training reward). Rerun new_analysis.py with "
        f"`train_config` set for that run; the old generator_reward was not the training reward.")
    return results['log_r']


# d_cut at the N/20 scale: the RDF distance below which 95% of sampled structure pairs match at least N of the 20
# molecules of a COMPACK shell at CCDC defaults (logistic fits to 291 MIPCAS and 333 NEHZOR p23 sample pairs,
# 2026-10-02; 20 of 20 gives 0.116 / 0.118, too tight for 100k draws to resolve). The kernel is sigma = d_cut / 3,
# as everywhere else in the analysis. KERNEL_SCALES=1 draws one diagnostic landscape per scale.
SCALE_RADII = {'mipcas': {'match15of20': 0.169, 'match10of20': 0.242},
               'nehzor': {'match15of20': 0.177, 'match10of20': 0.319}}
KERNEL_REACH = 3.0   # sigma = d_cut / 3, i.e. the kernel's d_cut IS the COMPACK distance above


def do_figs(mol_name, exp_path, uma_results_path, elj_results_path, prior_path, out_dir=PAPER_FIG_DIR,
            uma_bundle=None, elj_bundle=None):
    # ONLY_FIG=a,b regenerates just those figures. duo_embedding (UMAP over 5000x5000) and
    # cluster_analysis (clustering + its own embedding) are the expensive steps; pes_cartoon needs no data,
    # so asking for it alone skips loading the multi-GB results files entirely.
    only_fig = {k.strip() for k in os.environ.get('ONLY_FIG', '').split(',') if k.strip()}
    want = lambda *keys: (not only_fig) or bool(only_fig.intersection(keys))
    DATA_FIGS = ('elj_TB_fig', 'uma_TB_fig', 'density_contours', 'energy_marginal', 'marginals_fig',
                 'elj_staircase', 'uma_staircase', 'duo_embedding', 'cluster_analysis', 'crystals')
    if not want(*DATA_FIGS):
        print(f"  ONLY_FIG={sorted(only_fig)}: no data figure requested, skipping the results files")
        export_figs({'pes_cartoon': pes_cartoon()}, mol_name, out_dir)
        return
    uma_results = torch.load(uma_results_path, weights_only=False)
    elj_results = torch.load(elj_results_path, weights_only=False)
    rdf_mode = require_rdf_mode(uma_results, elj_results)

    cut_ind = 0  # only doing one now #np.argwhere(uma_results['n_clusts'] == 4).flatten()[0]
    uma_thermos = uma_results['metrics'][cut_ind]

    pred_path = r"D:\crystal_datasets\esen_s.pt"  # smaller mol crystal model
    predictor = init_uma_crystal_predictor(pred_path, device='cuda')
    exp_crystals = torch.load(exp_path, weights_only=False)
    ebatch = collate_data_list(exp_crystals, max_z_prime=1)
    with torch.no_grad():
        ebatch.cuda()
        ebatch.analyze(['elj', 'uma', 'rdf'], rdf_mode=rdf_mode, assign_outputs=True,
                       predictor=predictor)
        ebatch.cpu()
    num_polymorphs = ebatch.num_graphs
    eparams = ebatch.latent_params()
    eparams = eparams.repeat(1000, 1)
    eparams += torch.randn_like(eparams) * 0.01

    """
    generate figures
    """
    fig_dict = {}
    '''TB plot'''
    # log R is the EXACT training reward written by new_analysis.py
    # (MolecularCrystal.prebuilt_sample_to_reward), not a hand-rolled copy
    # the prior is only read for results that stored the raw eLJ sum (some prior files are several GB)
    en_scaling_factor = 1.0 if 'lj_coeff' in elj_results else elj_prescale_factor(
        elj_results, torch.load(prior_path, weights_only=False)['thermal_scaling_factor'])
    for tag, res, res_path in (('elj', elj_results, elj_results_path), ('uma', uma_results, uma_results_path)):
        if not (res.get('provenance') or {}).get('pb_frozen_installed'):
            # the landscape draws predate the frozen-P_B fix; tb_draws.py writes fresh ones for this figure only
            tb_path = os.path.join(os.path.dirname(res_path), 'tb',
                                   os.path.basename(res_path).replace('_red0.pt', '.pt').replace('.pt', '_tb.pt'))
            if os.path.exists(tb_path):
                tb = torch.load(tb_path, weights_only=False)
                assert tb['provenance']['step_ind'] == res['provenance']['step_ind'], (
                    f"{tb_path} is from checkpoint step {tb['provenance']['step_ind']}, the landscape results from "
                    f"{res['provenance']['step_ind']}")
                print(f"  {tag} TB parity from {tb['n']} fresh draws ({os.path.basename(tb_path)})")
                res = tb
        # results saved before load_gfn installed the frozen P_B snapshot carry log P_B from the drifted live head
        # (1-2.4 nats off, ~2 nats of extra scatter on the p23 arms); refuse them rather than plot them
        if not (res.get('provenance') or {}).get('pb_frozen_installed'):
            print(f"  !! {tag} TB parity SKIPPED: results predate the frozen-P_B fix (no pb_frozen_installed in "
                  f"provenance); resample to regenerate it")
            continue
        if 'log_r' not in res:
            print(f"  !! {tag} TB parity SKIPPED: results carry no 'log_r' (analysis run without train_config)")
            continue
        x = res['log_pbs'] + require_log_r(res, tag).to(res['log_pbs'].dtype)
        y = res['log_pfs'] + res['learned_log_z']
        fig_dict[f'{tag}_TB_fig'] = parity_fig(x, y,
                                               "log(P<sub>b</sub>) + log(R)",
                                               "log(P<sub>f</sub>) + log(Z<sub>θ</sub>)",
                                               quantile_cut=0.99)

    # A *_red0 results file keeps EVERY sample in sample_batch and applies max_reduction_en only to
    # good_ens / dmat, so the density figures below must apply it themselves (the TB parity figure
    # above stays on all samples: it checks the model's fit to the reward over everything it draws).
    def reduced_only(res):
        batch = res['sample_batch']
        thr = res.get('max_reduction_en')
        if thr is None:
            return batch.clone()
        keep = (batch.reduction_en.flatten() <= float(thr)).nonzero().flatten().tolist()
        rows = batch.batch_to_list()
        print(f"  density figures: reduced cells only, {len(keep)} of {len(rows)} samples (reduction_en <= {thr:g})")
        return collate_data_list([rows[i] for i in keep])

    def bundle_draws(bundle, res):
        """Reduced-cell draws of a dens_engine bundle: energy, packing coefficient, and a batch carrying their cell
        parameters (built from the stored latents on one of the results' own crystals; no neighbour lists)."""
        d = torch.load(os.path.join(bundle, 'draws.pt'), weights_only=False)
        keep = d['reduction_en'].flatten() <= float(res.get('max_reduction_en') or float('inf'))
        print(f"  density figures: {int(keep.sum())} of {len(keep)} bundle draws (reduced cells), "
              f"{os.path.basename(bundle)}")
        template = res['sample_batch'].batch_to_list()[0]
        batch = collate_data_list([template.clone() for _ in range(int(keep.sum()))])
        batch.latent_to_cell_params(d['latents'][keep])
        return d['energy'][keep].flatten(), d['packing_coeff'][keep].flatten(), batch

    bundled = uma_bundle is not None and elj_bundle is not None
    # the per-draw figures below are skipped when ONLY_FIG asks for none of them (they rebuild 100k-draw batches)
    if want('density_contours', 'energy_marginal', 'marginals_fig', 'elj_staircase', 'uma_staircase'):
        if bundled:
            uma_en, uma_cp, uma_batch = bundle_draws(uma_bundle, uma_results)
            elj_en, elj_cp, elj_batch = bundle_draws(elj_bundle, elj_results)   # the bundle stores the scaled eLJ
        else:
            uma_batch = reduced_only(uma_results)
            elj_batch = reduced_only(elj_results)
            uma_en = uma_batch.uma
            elj_en = elj_batch.elj * en_scaling_factor
            uma_cp = uma_batch.packing_coeff
            elj_cp = elj_batch.packing_coeff
        '''Density funnel overlay'''
        combo_en = torch.cat([elj_en, uma_en])
        fig_dict['density_contours'] = plot_dual_density_contour(elj_cp, elj_en.clip(max=0), uma_cp, uma_en.clip(max=0),
                                                                 label_a="LJ",
                                                                 label_b="eSEN",
                                                                 ncontours=8,
                                                                 max_y_quantile=0.99,
                                                                 x_min=0.55, x_max=0.9,
                                                                 y_min=combo_en.amin() - 5, y_max=combo_en.quantile(0.95),
                                                                 bw=0.1,
                                                                 show=False,
                                                                 return_fig=True,
                                                                 yaxis_title="Lattice Energy (kJ/mol)",
                                                                 xaxis_title="Packing Coefficient"
                                                                 )

        '''Energy marginals'''
        fig_dict['energy_marginal'] = dual_energy_marginal_fig(elj_en, uma_en,
                                                               "LJ",
                                                               "eSEN"
                                                               )

        '''12 marginals'''
        marginals_fig = uma_batch.plot_batch_cell_params(
            space='latent',
            ref_dist=elj_batch.latent_params(),
            aux_dists=[eparams],
            return_fig=True,
            show=False
        )
        marginals_fig.data[1].name = "eSEN Model"
        marginals_fig.data[0].name = "LJ Model"
        marginals_fig.data[2].name = "Experimental"
        subplot_labels = [
            '(a) Normed a',
            '(b) Normed b',
            '(c) Normed c',
            '(d) α',
            '(e) β',
            '(f) γ',
            '(g) u',
            '(h) v',
            '(i) w',
            '(j) θ',
            '(k) φ',
            '(l) r',
        ]
        for ii, annotation in enumerate(marginals_fig.layout.annotations):
            annotation.text = subplot_labels[ii]
        fig_dict['marginals_fig'] = marginals_fig

        '''staircase plots'''
        # axis_range=(-1, 1): the latent domain. Without it each panel bins over its OWN data range, so a
        # near-delta dimension is binned across float noise and draws as stripes instead of a narrow peak.
        elj_staircase = elj_batch.plot_batch_staircase(space='latent',
                                                       ref_dist=ebatch.latent_params(),
                                                       axis_range=(-1, 1),
                                                       return_fig=True,
                                                       show=False)
        uma_staircase = uma_batch.plot_batch_staircase(space='latent',
                                                       ref_dist=ebatch.latent_params(),
                                                       axis_range=(-1, 1),
                                                       return_fig=True,
                                                       show=False)
        fig_dict['elj_staircase'] = elj_staircase
        fig_dict['uma_staircase'] = uma_staircase

    # the embedding and clustering index by dmat_row_indices, i.e. rows of the FULL sample batch
    # (the dmat itself already carries the energy window and any reduction filter)
    uma_batch = uma_results['sample_batch'].clone()
    elj_batch = elj_results['sample_batch'].clone()
    if want('duo_embedding'):
        if bundled:   # every window & reduced draw of both bundles, embedded jointly
            from energy_sampling.eval.paper1_results.landscape_100k import bundle_duo
            fig_dict['duo_embedding'], duo = duo_embedding_fig(None, None, None, None, None, None,
                                                               precomputed=bundle_duo(uma_bundle, elj_bundle))
        else:
            fig_dict['duo_embedding'], duo = duo_embedding_fig(ebatch, elj_batch, elj_results, en_scaling_factor,
                                                               uma_batch, uma_results)
        # kept for the fused multi-molecule figure (fused_duo_embedding); the molecule is the first reference's
        first = ebatch.batch == 0
        np.savez(os.path.join(out_dir, f'{mol_name}_duo_embedding.npz'), z=ebatch.z[first].cpu().numpy(),
                 pos=ebatch.pos[first].cpu().numpy(), **duo)

    # todo compute a summary statistic that makes this metric real/meaningful
    # todo consider then maybe whether to combine these as well

    '''Big 'ol combo plot'''
    if not want('cluster_analysis', 'crystals'):
        fig_dict['pes_cartoon'] = pes_cartoon()
        export_figs(fig_dict, mol_name, out_dir)
        return
    if bundled:
        # the whole landscape over the bundle's draws: the 10k basins extended to every draw (landscape_100k)
        from energy_sampling.eval.paper1_results.landscape_100k import bundle_landscape
        # The paper figure: basins, shading and the first percentile rows at the 10/20 scale (95% of pairs at that
        # RDF distance match at least 10 of 20 molecules); the same points' percentiles at the 15/20 scale as extra
        # table rows. KERNEL_SCALES=1 instead draws one diagnostic landscape per scale, each with its own mean shift.
        kernel = {name: 3 * r / KERNEL_REACH for name, r in SCALE_RADII[mol_name].items()}
        scales = {'': kernel['match10of20']}
        diagnostic = bool(os.environ.get('KERNEL_SCALES'))
        if diagnostic:
            scales = {f'_{name}': dc for name, dc in kernel.items()}
        for suffix, d_cut in scales.items():
            print(f"  landscape{suffix}: kernel d_cut {d_cut:.4f} (sigma {d_cut / 3:.4f})")
            landscape, fig_thermos = bundle_landscape(
                uma_bundle, elj_bundle, uma_results, uma_thermos, ebatch, 'uma', max_n_clusters=4,
                min_basin_size=100, d_cut=d_cut, max_basins=6 if diagnostic else 4,
                scale_label=None if diagnostic else '10/20',
                extra_scales=None if diagnostic else {'15/20': kernel['match15of20']})
            (basin_colorscale, basin_min_batch, indexed_cluster_labels,
             n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
             polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
             sample_energy, sample_inds, stats) = landscape
            fig_dict['cluster_analysis' + suffix] = combo_fig(
                num_polymorphs, n_basins, packing_coeffs, stats, fig_thermos, sample_inds, sample_embedding, p_maxima,
                polymorph_inds, new_min_inds, sample_energy, polymorph_colorscale, sample_colors,
                indexed_cluster_labels, basin_colorscale, basin_min_batch, polymorph_basin_index,
                energy_function='uma', marker_scale=(10000 / len(sample_inds)) ** 0.5)
            if diagnostic:
                fig_dict = {'cluster_analysis' + suffix: fig_dict['cluster_analysis' + suffix]}
                export_figs(fig_dict, mol_name, out_dir)
        if diagnostic:
            return
        print("  basin-maximum crystal export skipped: the maxima are bundle draws, which are not stored as crystals")
        fig_dict['pes_cartoon'] = pes_cartoon()
        export_figs(fig_dict, mol_name, out_dir)
        return
    if os.environ.get('KERNEL_SCALES'):
        # the 10k analysis set at each COMPACK-anchored scale: density, mean shift and table at that kernel width
        for name, r in SCALE_RADII[mol_name].items():
            d_cut = 3 * r / KERNEL_REACH
            res_s = dict(uma_results, d_cuts=[d_cut])
            dm = uma_results['dmat'].double()
            th = {'density': (torch.exp(-dm ** 2 / (2 * (d_cut / 3) ** 2)).sum(-1) - 1).numpy()}
            cl, icl, pm, nb = clustering(res_s, th, max_n_clusters=4, min_basin_size=100)
            print(f"  scale {name} (10k): d_cut {d_cut:.4f}; {nb} basins of 100+; sizes "
                  f"{[int((icl == k).sum()) for k in range(nb)]}; unassigned {100 * float((icl < 0).float().mean()):.1f}%")
            if nb == 0:
                print("    no basin of 100+ samples at this width: no figure")
                continue
            if nb > 6:   # the figure grid holds six: keep the six largest, in density order
                sizes = torch.tensor([int((icl == k).sum()) for k in range(nb)])
                keep = torch.sort(torch.argsort(-sizes)[:6]).values
                remap = torch.full((nb,), -1, dtype=torch.long)
                remap[keep] = torch.arange(6)
                cl = torch.where(torch.isin(icl, keep), cl, torch.full_like(cl, -1))
                icl = torch.where(icl >= 0, remap[icl.clamp(min=0)], icl)
                pm, nb = pm[keep.numpy()], 6
            (basin_colorscale, basin_min_batch, icl, nb, new_min_inds, pm, packing_coeffs, polymorph_basin_index,
             polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding, sample_energy, sample_inds,
             stats) = combo_fig_analysis(ebatch, elj_batch, elj_results, num_polymorphs, uma_batch, res_s, 'uma', cl,
                                         pm, nb, icl)
            export_figs({f'cluster_analysis_{name}_10k': combo_fig(
                num_polymorphs, nb, packing_coeffs, stats, th, sample_inds, sample_embedding, pm, polymorph_inds,
                new_min_inds, sample_energy, polymorph_colorscale, sample_colors, icl, basin_colorscale,
                basin_min_batch, polymorph_basin_index, energy_function='uma')}, mol_name, out_dir)
        return
    scale_kw = {}
    if mol_name in SCALE_RADII:
        # the paper's definition, on the 10k analysis set: density, mean shift, shading and the first percentile
        # rows at the 10/20 kernel; the same points' percentiles at 15/20 as extra rows
        kernel = {name: 3 * r / KERNEL_REACH for name, r in SCALE_RADII[mol_name].items()}
        d10 = kernel['match10of20']
        uma_results = dict(uma_results, d_cuts=[d10])
        uma_thermos = {'density': (torch.exp(-uma_results['dmat'].double() ** 2 / (2 * (d10 / 3) ** 2)).sum(-1) - 1).numpy()}
        scale_kw = dict(scale_label='10/20', extra_scales={'15/20': kernel['match15of20']})
        print(f"  landscape (10k): kernel d_cut {d10:.4f}; {100 * float(np.mean(uma_thermos['density'] < 1)):.1f}% of "
              f"samples below one sample's weight")
    cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering(
        uma_results, uma_thermos, max_n_clusters=4, min_basin_size=100, cc_max=3.0 if scale_kw else 2.0)
    if n_basins > 4:   # the scan ran out above four modes: keep the four largest, in density order
        sizes = torch.tensor([int((indexed_cluster_labels == k).sum()) for k in range(n_basins)])
        keep = torch.sort(torch.argsort(-sizes)[:4]).values
        remap = torch.full((n_basins,), -1, dtype=torch.long)
        remap[keep] = torch.arange(4)
        print(f"  {n_basins} basins of 100+ at the end of the scan (sizes {sizes.tolist()}); the four largest are drawn")
        cluster_labels = torch.where(torch.isin(indexed_cluster_labels, keep), cluster_labels,
                                     torch.full_like(cluster_labels, -1))
        indexed_cluster_labels = torch.where(indexed_cluster_labels >= 0,
                                             remap[indexed_cluster_labels.clamp(min=0)], indexed_cluster_labels)
        p_maxima, n_basins = p_maxima[keep.numpy()], 4
    print(f"  basins: {[int((indexed_cluster_labels == k).sum()) for k in range(n_basins)]}; unassigned "
          f"{100 * float((indexed_cluster_labels < 0).float().mean()):.1f}%")
    #cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering3(uma_results, uma_thermos, max_n_clusters=6,min_basin_size=100)
    # cluster_labels, indexed_cluster_labels, p_maxima, n_basins = clustering2(uma_results, uma_thermos, n_clusters=10, n_keep=6)

    (basin_colorscale, basin_min_batch, indexed_cluster_labels,
     n_basins, new_min_inds, p_maxima, packing_coeffs, polymorph_basin_index,
     polymorph_colorscale, polymorph_inds, sample_colors, sample_embedding,
     sample_energy, sample_inds, stats) = combo_fig_analysis(
        ebatch, elj_batch, elj_results, num_polymorphs, uma_batch, uma_results, 'uma', cluster_labels, p_maxima,
        n_basins, indexed_cluster_labels, **scale_kw)

    fig_dict['cluster_analysis'] = combo_fig(
        num_polymorphs,
        n_basins,
        packing_coeffs,
        stats,
        uma_thermos,
        sample_inds,
        sample_embedding,
        p_maxima,
        polymorph_inds,
        new_min_inds,
        sample_energy,
        polymorph_colorscale,
        sample_colors,
        indexed_cluster_labels,
        basin_colorscale,
        basin_min_batch,
        polymorph_basin_index,
        energy_function='uma'
    )
    # fig_dict['cluster_analysis'].show()

    "save the top crystals"
    esamples = ebatch.batch_to_list()
    # p_maxima are rows of the FILTERED set (the dmat rows); map them back to the full
    # sample list. Indexing the full list directly exported unrelated structures
    # whenever the energy filter had removed any sample before a maximum.
    rows = dmat_row_indices(uma_results['sample_batch'], uma_results, 'uma')
    samples = uma_results['sample_batch'].batch_to_list()
    cbatch = collate_data_list([samples[int(rows[ind])] for ind in p_maxima])
    clustbatch = cbatch.mol2cluster(cutoff=2)
    mols = []

    for ind in range(cbatch.num_graphs):
        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='convolve with')
        write(f"{esamples[0].identifier}_poly_{ind}.png", mol)

        mol = ase_mol_from_crystaldata(clustbatch, ind, mode='unit cell')

        cif_path = f"{samples[0].identifier}_ucell_{ind}.cif"
        mol.write(cif_path)

        with open(cif_path, 'r') as f:
            content = f.read()
        content = content.replace('data_image0', f'data_{samples[0].identifier}_ucell_{ind}', 1)
        with open(cif_path, 'w') as f:
            f.write(content)
    #
    # clustbatch.visualize(mode='convolve with')
    # clustbatch.visualize(mode='unit cell')

    fig_dict['pes_cartoon'] = pes_cartoon()
    export_figs(fig_dict, mol_name, out_dir)


def export_figs(fig_dict, mol_name, out_dir):
    """one styling pass and one PNG per figure in fig_dict"""
    pub_style = dict(
        font=dict(family="Arial", size=18),
        width=1920,
        height=1080,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations_font_size=18,
        scale=2,
    )

    def custom_style(key):
        dd = {}
        if key == 'pes_cartoon':
            dd = {
                'width': 900,
                'height': 540,  # 60% of the square version
                'font_size': 22,
            }
        if key == 'summary_table' or key == 'polymorph_table':
            dd = {
                'font_size': 16,
                'annotations_font_size': 16,
                'scale': 4,
                'width': 800,
                'height': 1100,
            }
        if 'staircase' in key:
            dd = {
                'font_size': 18,
                'annotations_font_size': 18,
                'width': 2400,
                'height': 1200,
                'scale': 3
            }
        if key == 'marginals_fig':
            dd = {
                'height': 800,
                'font_size': 32,
                'annotations_font_size': 32,
                'scale': 3,
            }
        if key == 'embedding_fig':
            dd = {
                'width': 1500,
                'height': 650,
            }
        if 'TB' in key:
            dd = {
                'height': 620,
                'width': 620
            }
        if key == 'density_contours':
            dd = {
                'height': 500,
                'width': 1000,
                'font_size': 30,
                'annotations_font_size': 30,
                'scale': 3,
            }
        if key == 'energy_marginal':
            dd = {
                'height': 500,
                'width': 1000,
                'font_size': 30,
                'scale': 3,
                'annotations_font_size': 30,
                'pad': 24,  # white border added to the PNG: plotly's automargin puts the axis titles flush to the edge
            }
        if key == 'duo_embedding':
            dd = {'width': 1000, 'height': 1000, 'margin': dict(l=20, r=20, t=20, b=20)}
        if key == 'embedding_scales':
            dd = {'width': 2100, 'height': 700, 'margin': dict(l=20, r=30, t=60, b=20), 'font_size': 22,
                  'annotations_font_size': 28}
        if key == 'duo_embedding_fused':
            dd = {'width': 2000, 'height': 1000, 'margin': dict(l=20, r=20, t=20, b=20), 'annotations_font_size': 38}
        if key.startswith('cluster_analysis'):
            dd = {
                'width': 1500,
                'height': 900,  # the summary table carries up to six data rows
                'annotations_font_size': 24,
                'font_size': 24,
                'legend_font_size': 16,
            }
        return dd

    for key, fig in fig_dict.items():
        # fig.show()
        style = copy(pub_style)
        style.update(custom_style(key))
        scale = style['scale']
        style.pop('scale')
        pad = style.pop('pad', 0)
        for skey in style.keys():
            if 'annotation' in skey:
                fs = style[skey]
                fig.update_annotations(font_size=fs, selector=lambda a: a.name != 'callout')
        style = {skey: value for skey, value in style.items() if 'annotation' not in skey}
        fig.update_layout(**style)
        fig.write_image(os.path.join(out_dir, f'{mol_name}_{key}.png'),
                        width=fig.layout.width, height=fig.layout.height, scale=scale)
        if pad:
            from PIL import Image, ImageOps
            path = os.path.join(out_dir, f'{mol_name}_{key}.png')
            ImageOps.expand(Image.open(path).convert('RGB'), border=int(pad * scale), fill='white').save(path)

    aa = 1


def duo_embedding_fig(ebatch, elj_batch, elj_results, en_scaling_factor, uma_batch, uma_results, precomputed=None):
    """precomputed: {'uma': [n, 2], 'lj': [m, 2], 'poly': [k, 2]} embedding coordinates (landscape_100k.bundle_duo);
    the 5000 + 5000 sample embedding below is then skipped and only the figure is drawn."""
    if precomputed is not None:
        return _duo_plot(np.asarray(precomputed['uma']), np.asarray(precomputed['lj']),
                         np.atleast_2d(precomputed['poly']))
    subsamp = 5000
    dmat_a = uma_results['dmat'].clone()
    dmat_b = elj_results['dmat'].clone()
    rdf_a = uma_batch.rdf.clone()
    rdf_b = elj_batch.rdf.clone()
    # each run's rdf read through the rows ITS dmat was built from; the saved index
    # set is used when present rather than re-deriving the 15 kT filter
    rdf_a = rdf_a[dmat_row_indices(uma_batch, uma_results, 'uma')]
    if 'good_ens' not in elj_results:  # legacy file: the filter needs the scaled energy
        elj_batch = elj_batch.clone()
        elj_batch.elj = elj_batch.elj * en_scaling_factor
    rdf_b = rdf_b[dmat_row_indices(elj_batch, elj_results, 'elj')]
    rdf_a = rdf_a[:subsamp]
    rdf_b = rdf_b[:subsamp]
    dmat_a = dmat_a[:subsamp][:, :subsamp]
    dmat_b = dmat_b[:subsamp][:, :subsamp]
    n_a, n_b = dmat_a.shape[0], dmat_b.shape[0]
    assert dmat_a.shape == (n_a, n_a) and dmat_b.shape == (n_b, n_b)
    assert rdf_a.shape[0] == n_a and rdf_b.shape[0] == n_b
    device = 'cuda'
    dtype = torch.float32
    bins = torch.linspace(0, 10, rdf_a.shape[-1], device=device)
    rdf_a = rdf_a.to(device=device, dtype=dtype)
    rdf_b = rdf_b.to(device=device, dtype=dtype)
    a_vs_b = torch.empty(n_a, n_b, device=device, dtype=dtype)
    chunk = 32
    min_chunk = 1
    i = 0
    pbar = tqdm(total=n_a, desc="A vs B")
    while i < n_a:
        j = min(i + chunk, n_a)
        try:
            a_vs_b[i:j] = compute_rdf_distance(
                rdf_a[i:j, None],
                rdf_b[None],
                bins,
            )
            pbar.update(j - i)
            i = j
        except torch.cuda.OutOfMemoryError:
            if chunk <= min_chunk:
                raise
            torch.cuda.empty_cache()
            chunk = max(min_chunk, chunk // 2)
            pbar.write(f"OOM at chunk={j - i}, reducing to {chunk}")
    pbar.close()
    a_vs_b = a_vs_b.to('cpu')
    merged = torch.empty(
        n_a + n_b, n_a + n_b,
        device=dmat_a.device, dtype=dmat_a.dtype,
    )
    merged[:n_a, :n_a] = dmat_a
    merged[n_a:, n_a:] = dmat_b.to(device=dmat_a.device, dtype=dmat_a.dtype)
    merged[:n_a, n_a:] = a_vs_b
    merged[n_a:, :n_a] = a_vs_b.T
    merged = augment_dmat2(ebatch.rdf, torch.cat([rdf_a.cpu(), rdf_b.cpu()], dim=0), merged)
    polymorph_inds = torch.arange(ebatch.num_graphs) + len(merged) - ebatch.num_graphs
    from umap import UMAP
    umap_model = UMAP(n_components=2, n_neighbors=300, min_dist=0.75,
                      init='spectral', metric='precomputed', low_memory=True,
                      # densmap=True,
                      repulsion_strength=2.0,
                      n_jobs=-1)
    sample_embedding = umap_model.fit_transform(merged.cpu().numpy().astype(np.float32))
    return _duo_plot(sample_embedding[:n_a], sample_embedding[n_a:n_a + n_b],
                     np.atleast_2d(sample_embedding[polymorph_inds]))


def _duo_plot(uma_embed, lj_embed, poly_xy):
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    def density_alpha(pts, lo=0.15, hi=0.8, log=False):
        if len(pts) > 20000:   # a smoothed 2D histogram: gaussian_kde is O(n^2)
            from scipy.ndimage import gaussian_filter
            h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=300)
            h = gaussian_filter(h, 3)
            d = h[np.clip(np.searchsorted(xe, pts[:, 0]) - 1, 0, 299), np.clip(np.searchsorted(ye, pts[:, 1]) - 1, 0, 299)]
        else:
            d = gaussian_kde(pts.T)(pts.T)
        if log:
            d = np.log(d + 1e-12)
        d = (d - d.min()) / (d.max() - d.min() + 1e-12)
        return lo + (hi - lo) * d

    uma_a = density_alpha(uma_embed)
    lj_a = density_alpha(lj_embed)
    uma_colors = [f'rgba(65,105,225,{a:.3f})' for a in uma_a]
    lj_colors = [f'rgba(220,20,60,{a:.3f})' for a in lj_a]
    fig = go.Figure()
    ms = 6 if len(uma_embed) + len(lj_embed) < 20000 else 2.5
    fig.add_scatter(x=uma_embed[:, 0], y=uma_embed[:, 1], mode='markers', name='eSEN', marker_color=uma_colors,
                    marker_size=ms)
    fig.add_scatter(x=lj_embed[:, 0], y=lj_embed[:, 1], mode='markers', name='LJ', marker_color=lj_colors,
                    marker_size=ms)
    poly_x, poly_y = poly_xy[:, 0], poly_xy[:, 1]

    fig.add_scatter(x=poly_x, y=poly_y,
                    mode='markers',
                    marker_symbol='x-thin',
                    marker_color='black',
                    marker_line_color='black',
                    marker_line_width=8,
                    marker_size=28,
                    opacity=1.0,
                    showlegend=False)

    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    fig.update_layout(
        xaxis1_title='CV1', yaxis1_title='CV2',
        font_size=20,
        # itemsizing keeps the swatches at one size instead of the (small) scatter marker size
        legend=dict(font=dict(size=30), itemsizing='constant', x=0.99, y=0.99, xanchor='right', yanchor='top',
                    bgcolor='rgba(255,255,255,0.65)'),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    # exported square (export_figs), so one data unit is the same length on both axes
    fig.update_xaxes(title_font_size=40)
    fig.update_yaxes(title_font_size=40, scaleanchor='x', scaleratio=1)
    duo = dict(uma=uma_embed, lj=lj_embed, uma_alpha=uma_a, lj_alpha=lj_a,
               poly=np.stack([poly_x, poly_y], axis=1))
    return fig, duo


def fused_duo_embedding(mol_names, out_dir):
    """One figure from the per-molecule embeddings do_figs saved (`<mol>_duo_embedding.npz`)."""
    paths = [os.path.join(out_dir, f'{m}_duo_embedding.npz') for m in mol_names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  fused LJ-vs-UMA figure SKIPPED: no saved embedding at {missing}")
        return
    fig = fused_duo_embedding_fig([dict(np.load(p)) for p in paths])
    export_figs({'duo_embedding_fused': fig}, '_'.join(mol_names), out_dir)


def scale_embeddings(mol_name, uma_bundle, out_dir):
    """The UMA model's 100k embedding (window & reduced draws) coloured three ways: lattice energy, and the P(x)
    percentile at the 10/20 and 15/20 kernel scales (sigma = d_cut / 3, sums from the bundle's histograms)."""
    from scipy.stats import rankdata
    from energy_sampling.eval.paper1_results.dens_engine import kernel_sums_from_hist
    from energy_sampling.eval.paper1_results.landscape_100k import embed_bundles
    draws = torch.load(os.path.join(uma_bundle, 'draws.pt'), weights_only=False)
    keep = draws['window_reduced'].numpy()
    n = len(keep)
    emb = embed_bundles([uma_bundle])
    energy = draws['energy'].numpy()[keep]
    panels = [dict(values=energy, title='Lattice energy', colorbar='E (kJ/mol)', colorscale='Turbo_r',
                   # colour scale capped 10 kT above the lowest draw; draws above it (up to the 15 kT window) take the end colour
                   range=(float(energy.min()), float(energy.min()) + 10 * float(draws['kT'])), top='low')]
    for name, label in (('match10of20', '10/20'), ('match15of20', '15/20')):
        s = kernel_sums_from_hist(uma_bundle, 3 * SCALE_RADII[mol_name][name] / KERNEL_REACH)[:n][keep]
        panels.append(dict(values=100 * (rankdata(s) - 0.5) / len(s), title=f'P<sup>{label}</sup>(x) percentile',
                           colorbar='percentile', colorscale='Turbo', range=(0, 100), top='high'))
        print(f"  {mol_name} {label}: {100 * float(np.mean(s < 1)):.1f}% of draws below one draw's weight")
    fig = scale_embedding_fig(emb['draws'][0][keep], panels, emb['refs'])
    export_figs({'embedding_scales': fig}, mol_name, out_dir)


def scale_embeddings_10k(name, results_path, exp_path, identifiers, scales, out_dir, kT=2.5):
    """scale_embeddings for a 10k analysis results file (new_analysis.py) instead of a 100k bundle: its analysed set
    (dmat rows) embedded by UMAP with the landscape figure's settings, together with the experimental forms, and
    coloured by lattice energy (capped 10 kT above the lowest sample) and by the P(x) percentile at each kernel scale
    of `scales` ({label: d_cut}; sigma = d_cut / 3, sums in samples' worth of weight, own sample excluded)."""
    from scipy.stats import rankdata
    from umap import UMAP
    R = torch.load(results_path, map_location='cpu', weights_only=False)
    good = torch.as_tensor(R['good_ens']).long()
    D = R['dmat'].double()
    D = 0.5 * (D + D.T)
    energy = R['sample_energy'][good].numpy()
    rdf_s = R['sample_batch'].rdf[good].float()
    exp = torch.load(exp_path, map_location='cpu', weights_only=False)
    exp = exp if isinstance(exp, list) else exp.batch_to_list()
    exp = [e for e in exp if str(e.identifier) in identifiers]
    assert [str(e.identifier) for e in exp] == list(identifiers), f"{exp_path}: {[e.identifier for e in exp]}"
    eb = collate_data_list([e.clone() for e in exp])
    eb.analyze(['rdf'], cutoff=10, std_orientation=bool(R['std_orientation']), assign_outputs=True,
               rdf_mode=R['rdf_mode'])
    rdf_r = eb.rdf.float()
    bins = torch.linspace(0, 10, rdf_s.shape[-1])
    d_rs = torch.stack([compute_rdf_distance(rdf_r[i], rdf_s, bins) for i in range(len(exp))]).double()
    d_rr = torch.stack([compute_rdf_distance(rdf_r[i], rdf_r, bins) for i in range(len(exp))]).double()
    print(f"  {name}: nearest sample to " + ', '.join(f"{lab} {float(d_rs[i].min()):.3f}" for i, lab in enumerate(identifiers)))
    full = torch.cat([torch.cat([D, d_rs.T], 1), torch.cat([d_rs, 0.5 * (d_rr + d_rr.T)], 1)], 0)
    full.fill_diagonal_(0)
    n = len(good)
    xy = UMAP(n_components=2, init='pca', metric='precomputed', low_memory=True, n_neighbors=300, min_dist=0.75,
              random_state=0).fit_transform(full.numpy())
    panels = [dict(values=energy, title='Lattice energy', colorbar='E (kJ/mol)', colorscale='Turbo_r',
                   range=(float(energy.min()), float(energy.min()) + 10 * kT), top='low')]
    for label, d_cut in scales.items():
        s = (torch.exp(-D ** 2 / (2 * (d_cut / 3) ** 2)).sum(1) - 1).numpy()
        panels.append(dict(values=100 * (rankdata(s) - 0.5) / len(s), title=f'P<sup>{label}</sup>(x) percentile',
                           colorbar='percentile', colorscale='Turbo', range=(0, 100), top='high'))
        print(f"  {name} {label} (d_cut {d_cut:.3f}): {100 * float(np.mean(s < 1)):.1f}% of samples below one "
              f"sample's weight")
    fig = scale_embedding_fig(xy[:n], panels, xy[n:], poly_labels=list(identifiers))
    export_figs({'embedding_scales': fig}, name, out_dir)


def fused_energies(mol_names, out_dir, row_height=1500):
    """One figure from the exported `<mol>_energy_marginal.png` and `<mol>_density_contours.png`: a row per
    molecule (marginals, then contours), lettered panels, a row label, a dashed rule between rows, and the
    molecule drawn in the empty tail of its UMA marginal when `<mol>_duo_embedding.npz` holds its geometry."""
    from PIL import Image, ImageDraw, ImageFont
    need = [os.path.join(out_dir, f'{m}_{k}.png') for m in mol_names for k in ('energy_marginal', 'density_contours')]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        print(f"  fused energies figure SKIPPED: missing {missing}")
        return
    H = row_height
    head, label_w, gap, sep = 130, 170, 60, 90
    font, label_font = ImageFont.truetype('arial.ttf', 84), ImageFont.truetype('arial.ttf', 92)
    # left edge of each plot area as a fraction of its source image width (LJ panel, UMA panel)
    marg_x, cont_x = (0.115, 0.585), (0.125, 0.572)
    rows, letter = [], 0
    for mol in mol_names:
        marg = Image.open(os.path.join(out_dir, f'{mol}_energy_marginal.png')).convert('RGB')
        cont = Image.open(os.path.join(out_dir, f'{mol}_density_contours.png')).convert('RGB')
        marg = marg.resize((round(marg.width * H / marg.height), H), Image.LANCZOS)
        cont = cont.resize((round(cont.width * H / cont.height), H), Image.LANCZOS)
        geom = os.path.join(out_dir, f'{mol}_duo_embedding.npz')
        if os.path.exists(geom):
            g = np.load(geom)
            m = render_molecule_image(g['z'], g['pos'], px=900)
            s = min(0.21 * marg.width / m.width, 0.42 * H / m.height)
            m = m.resize((round(m.width * s), round(m.height * s)), Image.LANCZOS)
            marg.paste(m, (round(0.755 * marg.width), round(0.20 * H)), m)
        else:
            print(f"  fused energies figure: no {geom}, {mol} drawn without its molecule")
        row = Image.new('RGB', (label_w + marg.width + gap + cont.width, head + H), 'white')
        row.paste(marg, (label_w, head))
        row.paste(cont, (label_w + marg.width + gap, head))
        d = ImageDraw.Draw(row)
        for img, x0, fracs in ((marg, label_w, marg_x), (cont, label_w + marg.width + gap, cont_x)):
            for f in fracs:
                d.text((x0 + f * img.width, 10), f'({chr(97 + letter)})', font=font, fill='black')
                letter += 1
        tag = Image.new('RGBA', (H, label_w), (255, 255, 255, 0))
        td = ImageDraw.Draw(tag)
        td.text(((H - td.textlength(mol.upper(), font=label_font)) / 2, 30), mol.upper(), font=label_font,
                fill='black')
        tag = tag.rotate(90, expand=True)
        row.paste(tag, (0, head), tag)
        rows.append(row)
    W = max(r.width for r in rows)
    out = Image.new('RGB', (W, sum(r.height for r in rows) + sep * (len(rows) - 1)), 'white')
    d = ImageDraw.Draw(out)
    y = 0
    for k, row in enumerate(rows):
        if k:
            for x in range(0, W, 50):
                d.line([(x, y + sep // 2), (x + 28, y + sep // 2)], fill='black', width=5)
            y += sep
        out.paste(row, (0, y))
        y += row.height
    out.save(os.path.join(out_dir, '_'.join(mol_names) + '_energies_fused.png'))


if __name__ == '__main__':
    """
    load up elj distribution, uma distribution, and experimental polymorphs
    """
    # prod_sep12 pairs: results from new_analysis.py (make_figures: false).
    # prior_path is the eLJ prior; its thermal_scaling_factor only matters for
    # legacy results that stored raw .elj.
    RES = r"D:\crystal_datasets\gfn_results"
    PRIORS = r"D:\crystal_datasets\conditional\priors"
    PAIRS = {
        'mipcas': dict(exp_path=r"D:\crystal_datasets\mipcas\MIPCAS_standardized.pt",
                       uma_results_path=os.path.join(RES, 'p12_mipu_lr0p0625_best_10k.pt'),
                       elj_results_path=os.path.join(RES, 'p12_mip_lr1_best_10k.pt'),
                       prior_path=os.path.join(PRIORS, 'mipcas_sg2_zp1_elj_200k_prior_dataset.pt')),
        'nehzor': dict(exp_path=r"D:\crystal_datasets\nehzor\NEHZOR_structures_std_conf.pt",
                       uma_results_path=os.path.join(RES, 'p12_nehu_lr0p125_best_10k.pt'),
                       elj_results_path=os.path.join(RES, 'p12_neh_lr0p5_best_10k.pt'),
                       prior_path=os.path.join(PRIORS, 'nehzor_sg14_zp1_elj_prior_dataset.pt')),
    }
    # PAIR_SET=p20: the converged prod_sep20 arms on their ROLLING checkpoints, scored under the
    # post-change walls (mipcas *_niggli_v2 triclinic Niggli, nehzor *_w3 monoclinic per-class).
    # p12 stays the default so the paper's existing figures regenerate unchanged.
    PAIRS_P20 = {
        'mipcas': dict(exp_path=r'D:\crystal_datasets\mipcas\MIPCAS_standardized_niggli.pt',
                       uma_results_path=os.path.join(RES, 'p20_mipu_n5_running_10k.pt'),
                       elj_results_path=os.path.join(RES, 'p20_mip_n5_running_10k.pt'),
                       prior_path=os.path.join(PRIORS, 'mipcas_sg2_zp1_elj_200k_prior_dataset_niggli_v2.pt')),
        'nehzor': dict(exp_path=r'D:\crystal_datasets\nehzor\NEHZOR_structures_std_conf.pt',
                       uma_results_path=os.path.join(RES, 'p20_nehu_n5_running_10k.pt'),
                       elj_results_path=os.path.join(RES, 'p20_neh_n5_running_10k.pt'),
                       prior_path=os.path.join(PRIORS, 'nehzor_sg14_zp1_elj_prior_dataset_w3.pt')),
    }
    # PAIR_SET=p23 / p23_red0: the prod_sep23_ft fine-tunes of those arms (box and reduction penalties
    # ramped 10 -> 1000), all samples / reduced cells only (reduction_en <= 1e-6, *_red0 results).
    PAIRS_P23 = {mol: dict(p, uma_results_path=p['uma_results_path'].replace('p20_', 'p23_').replace('_n5_', '_ft_'),
                           elj_results_path=p['elj_results_path'].replace('p20_', 'p23_').replace('_n5_', '_ft_'))
                 for mol, p in PAIRS_P20.items()}
    PAIRS_P23_RED0 = {mol: dict(p, uma_results_path=p['uma_results_path'].replace('.pt', '_red0.pt'),
                                elj_results_path=p['elj_results_path'].replace('.pt', '_red0.pt'))
                      for mol, p in PAIRS_P23.items()}
    # PAIR_SET=p23_red0_100k: the same figures, with the landscape table's percentile rows read against the 100k-draw
    # dens_engine bundles of the same checkpoints (distance pass on the red0 figures' kernel, *_k_red0)
    def bundle_of(results_path):
        run = os.path.basename(results_path).replace('_red0.pt', '')
        return os.path.join(RES, 'dens', f'{run}_100000_k_red0')
    PAIRS_P23_RED0_100K = {mol: dict(p, uma_bundle=bundle_of(p['uma_results_path']),
                                     elj_bundle=bundle_of(p['elj_results_path']))
                           for mol, p in PAIRS_P23_RED0.items()}
    # PAIR_SET=ppl_red0: the prod_pool_oct07 arms (pooled oct05 priors), reduced cells only
    PAIRS_PPL_RED0 = {
        mol: dict(exp_path=PAIRS_P20[mol]['exp_path'],
                  uma_results_path=os.path.join(RES, f'ppl_{fam}u_n5_running_10k_red0.pt'),
                  elj_results_path=os.path.join(RES, f'ppl_{fam}_n5_running_10k_red0.pt'),
                  prior_path=os.path.join(PRIORS, f'{mol}_elj_pooled_oct05_prior.pt'))
        for mol, fam in (('mipcas', 'mip'), ('nehzor', 'neh'))}
    # PAIR_SET=pft_red0: the ft_pool_oct12 fine-tunes of those arms (box + reduction penalties ramped), reduced cells only
    PAIRS_PFT_RED0 = {
        mol: dict(p, uma_results_path=os.path.join(RES, f'pft_{fam}u_ft_running_10k_red0.pt'),
                  elj_results_path=os.path.join(RES, f'pft_{fam}_ft_running_10k_red0.pt'))
        for (mol, p), fam in zip(PAIRS_PPL_RED0.items(), ('mip', 'neh'))}
    PAIRS = {'p12': PAIRS, 'p20': PAIRS_P20, 'p23': PAIRS_P23, 'p23_red0': PAIRS_P23_RED0,
             'p23_red0_100k': PAIRS_P23_RED0_100K, 'ppl_red0': PAIRS_PPL_RED0,
             'pft_red0': PAIRS_PFT_RED0}[os.environ.get('PAIR_SET', 'p12')]
    only = {m.strip() for m in os.environ.get('ONLY_MOL', '').split(',') if m.strip()}
    out_dir = os.environ.get('FIG_OUT_DIR', PAPER_FIG_DIR)
    # ONLY_FIG=embedding_scales_10k: the three-panel embedding figure from a 10k results file, for a system without
    # paired UMA/LJ runs; the kernel scales are that system's own COMPACK calibration
    # (compack_calibration/landscape_compack.py), read from its <tag>_scales.json
    SCALE_EMB_10K = {
        'acridine_inn_final': dict(
            results_path=os.path.join(RES, 'inn_acr23_base_u65361_10k_red0.pt'),
            exp_path=r'D:\crystal_datasets\acridine\opt_acrdin.pt', identifiers=('ACRDIN04', 'ACRDIN12'),
            scales_json=os.path.join(RES, 'compack_calibration', 'inn_acr23_final_scales.json'),
            # acridine (envwise RDF): P(>= N of 20) = 0.95 lies below nearly every sample pair (15/20 extrapolates under
            # all of them; at 10/20 every sample is under one sample's weight), so this figure takes the 50% distances
            scale_key='scales_p50')}
    if 'embedding_scales_10k' in {k.strip() for k in os.environ.get('ONLY_FIG', '').split(',')}:
        import json
        import sys
        for name, spec in SCALE_EMB_10K.items():
            if only and name not in only:
                continue
            key = spec.pop('scale_key', 'scales')
            cal = json.load(open(spec.pop('scales_json')))[key]
            tag = '' if key == 'scales' else ', 50%'
            scale_embeddings_10k(name, **spec, out_dir=out_dir,
                                 scales={f'10/20{tag}': cal['match10of20']['r'], f'15/20{tag}': cal['match15of20']['r']})
        sys.exit(0)
    for mol_name, paths in PAIRS.items():
        if only and mol_name not in only:
            continue
        print(f"\n=== {mol_name} -> {out_dir} ===")
        do_figs(mol_name, out_dir=out_dir, **paths)
    only_fig = {k.strip() for k in os.environ.get('ONLY_FIG', '').split(',') if k.strip()}
    if not only_fig or only_fig.intersection(('duo_embedding', 'duo_embedding_fused')):
        fused_duo_embedding([m for m in PAIRS if not only or m in only], out_dir)
    if not only_fig or 'embedding_scales' in only_fig:
        for m, paths in PAIRS.items():
            if paths.get('uma_bundle') and (not only or m in only):
                scale_embeddings(m, paths['uma_bundle'], out_dir)
    if not only_fig or only_fig.intersection(('energy_marginal', 'density_contours', 'energies_fused')):
        fused_energies([m for m in PAIRS if not only or m in only], out_dir)
