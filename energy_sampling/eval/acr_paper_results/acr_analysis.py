import glob
import os

import numpy as np
import torch
from ase.spacegroup import Spacegroup

from energy_sampling.eval.paper1_results.utils import get_gfn_samples, simple_dedupe
from energy_sampling.models import GFN
from mxtaltools.common.ase_interface import ase_mol_from_crystaldata
from mxtaltools.dataset_utils.utils import collate_data_list

torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def sample_and_analyze(model_path, config_path, molecule, max_z_prime, batch_size, sg_ind, zp, energy_function, device,
                       num_samples):
    gfn_model = GFN(**np.load(config_path, allow_pickle=True).item())
    gfn_model.load_state_dict(torch.load(model_path, weights_only=True))
    gfn_model.to(device)
    gfn_model.eval()

    "Sample from GFN & process samples"
    sample_batch, sample_latents, sample_energy, sample_cp, samples = get_gfn_samples(
        num_samples, max_z_prime,
        device, n_steps, batch_size, gfn_model,
        energy_function, molecule, sg_ind, zp
    )

    return samples


if __name__ == '__main__':
    "Configs & args"
    # acridine lj config
    run_name = 'acr_search'
    device = 'cuda'
    num_samples = 1000
    batch_size = 10
    energy_function = 'uma'  # 'elj', 'lj' 'uma
    n_steps = 100  # critical to get this right!
    zp = 1  # todo fix zp>1 pre-processing
    max_z_prime = zp
    d_cut = 0.1
    rdf_cut = 0.02
    tol = 1e-2
    model_path = rf"D:\crystal_datasets\acridine"
    model_prefix = "best_acr7"
    molecule_path = r"D:\crystal_datasets\acridine\acridine_conformer.pt"
    results_path = rf"D:\crystal_datasets\gfn_results\{run_name}.pt"
    samples_path = rf"D:\crystal_datasets\gfn_results\{run_name}"
    reload_results = False

    os.chdir(model_path)
    models = glob.glob(f'{model_prefix}*eval.pt')

    "Load Relevant Dataset"
    molecule = torch.load(molecule_path, weights_only=False)

    "Do CSP"
    if reload_results and os.path.exists(results_path):
        all_samples = torch.load(results_path, weights_only=False)
    else:# os.path.exists(results_path):
        all_samples = []
        for m_path in models:
            sg_ind = int(m_path.replace('rerun__','').split('_')[2].split('sg')[-1])
            config_path = m_path.replace('best_', '').replace('model_eval.pt', 'model_config.npy')
            samples = sample_and_analyze(m_path, config_path, molecule, max_z_prime, batch_size, sg_ind, zp,
                                         energy_function, device, num_samples)

            all_samples.append(samples)
            torch.save(all_samples, results_path)


    "Extract top candidates from each space group"
    sg_reps = {}
    for samples in all_samples:
        sg_reps[samples[0].sg_ind.item()] = simple_dedupe(samples, d_cut, rdf_cut)
    sorted_sgs = np.sort(list(sg_reps.keys()))

    "write to cif archive"
    if not os.path.exists(samples_path):
        os.mkdir(samples_path)
    for sg_ind in range(len(all_samples)):
        batch = collate_data_list(sg_reps[sorted_sgs[sg_ind]])
        cluster_batch = batch.mol2cluster(cutoff=6)
        energy = batch.uma_pot / (batch.sym_mult * batch.z_prime) - batch.uma_gas_pot
        sort_inds = torch.argsort(energy)
        for ind in sort_inds:  # write in order of lowest to highest energy
            mol = ase_mol_from_crystaldata(cluster_batch, index=ind, mode='unit cell')
            mol.info['spacegroup'] = Spacegroup(int(cluster_batch.sg_ind[sg_ind]), setting=1)
            mol.write(os.path.join(samples_path, f'sg_{sorted_sgs[sg_ind]}_sample_{ind}.cif'))

    "visualize"
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from mxtaltools.constants.space_group_info import SPACE_GROUPS
    from mxtaltools.common.utils import get_point_density

    fig = make_subplots(rows=3, cols=6, subplot_titles=[str(ii) + '_' + SPACE_GROUPS[int(ii)] for ii in sorted_sgs])

    all_ens = []
    all_cps = []
    for ind in range(len(sorted_sgs)):
        row = ind // 6 + 1
        col = ind % 6 + 1
        batch = collate_data_list(sg_reps[sorted_sgs[ind]])
        energy = batch.uma_pot / (batch.sym_mult * batch.z_prime) - batch.uma_gas_pot
        cp = batch.packing_coeff
        all_ens.extend(energy)
        all_cps.extend(cp)

    yrange = [np.amin(all_ens), np.quantile(all_ens, 0.9)]
    xrange = [np.quantile(all_cps, 0.1), np.amax(all_cps)]
    for ind in range(len(sorted_sgs)):
        row = ind // 6 + 1
        col = ind % 6 + 1
        batch = collate_data_list(sg_reps[sorted_sgs[ind]])
        energy = batch.uma_pot / (batch.sym_mult * batch.z_prime) - batch.uma_gas_pot
        cp = batch.packing_coeff

        x = cp
        y = energy
        xy = np.vstack([x.cpu().detach().numpy(), y.cpu().detach().numpy()])
        try:
            c = get_point_density(xy, bins=25)
        except:
            c = np.ones(len(x))

        fig.add_scatter(
            x=cp,
            y=energy,
            marker_size=8,
            opacity=1.0,
            row=row,
            col=col,
            marker_color=c,
            mode='markers',
            showlegend=False
        )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    fig.show()


    aa = 0
