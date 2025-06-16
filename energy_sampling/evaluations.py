import os
import torch
import wandb
from matplotlib import cm
from matplotlib.colors import to_hex
from mxtaltools.reporting.online import simple_embedding_fig, simple_cell_hist, simple_cell_scatter_fig, \
    log_crystal_samples, simple_latent_hist
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

from plot_utils import get_plotly_fig_size_mb
from sample_metrics import compute_distribution_distances
from utils import logmeanexp


@torch.no_grad()
def log_partition_function(initial_state, gfn, energy_function, mol_batch):
    condition = energy_function.get_conditioning_tensor(mol_batch)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state,
                                                              None,
                                                              energy_function,
                                                              condition)
    log_r, sample_batch = energy_function.log_reward(states[:, -1], mol_batch=mol_batch, log_temperature=condition[:, 0], return_exp=True)
    log_weight = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)

    log_Z = logmeanexp(log_weight)
    log_Z_lb = log_weight.mean()
    log_Z_learned = log_fs[:, 0].mean()

    return states, states[:, -1], log_r, log_Z, log_Z_lb, log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_fs


@torch.no_grad()
def mean_log_likelihood(data, gfn, log_reward_fn, num_evals=10):
    bsz = data.shape[0]
    data = data.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(data, None, log_reward_fn)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
    return logmeanexp(log_weight, dim=1).mean()


@torch.no_grad()
def get_sample_metrics(samples, gt_samples=None, final_eval=False):
    if gt_samples is None:
        return

    return compute_distribution_distances(samples.unsqueeze(1), gt_samples.unsqueeze(1), final_eval)


def eval_step(energy_function,
              gfn_model,
              init_state,
              do_figures: bool = True,
              mol_batch=None,
              conditional_flow_model: bool = False):
    gfn_model.eval()

    metrics = {}
    fig_dict = {}
    (flow_states, samples, log_r, log_Z, log_Z_lb,
     log_Z_learned, sample_batch, condition, log_pfs, log_pbs, log_fs) = log_partition_function(
        init_state, gfn_model, energy_function, mol_batch)

    "Scalar / distribution metrics"
    metrics['eval/log_Z'] = log_Z.cpu().detach().numpy()
    metrics['eval/log_Z_lb'] = log_Z_lb.cpu().detach().numpy()
    metrics['eval/log_Z_learned'] = log_Z_learned.cpu().detach().numpy()
    metrics['eval/packing_coeff'] = sample_batch.packing_coeff.mean().cpu().detach().numpy()
    metrics['eval/silu_potential'] = sample_batch.silu_pot.mean().cpu().detach().numpy()
    metrics['mean sample energy'] = sample_batch.gfn_energy.mean().cpu().detach().numpy()
    metrics['sample energy distribution'] = sample_batch.gfn_energy.cpu().detach().numpy()
    metrics['mean sample reward'] = log_r.mean().cpu().detach().numpy()
    metrics['sample reward distribution'] = log_r.cpu().detach().numpy()
    metrics['Crystal Log Temperature'] = condition[:, 0]
    metrics['Crystal Mean Log Temperature'] = condition[:, 0].mean()
    metrics['Crystal Min Temperature'] = energy_function.min_temperature
    metrics['Crystal Max Temperature'] = energy_function.max_temperature
    metrics['Ellipsoid Scale'] = energy_function.ellipsoid_scale
    metrics['Temperature Scaling Factor'] = energy_function.temperature_scaling_factor
    metrics['Density Loss Coefficient'] = energy_function.density_coeff
    if hasattr(sample_batch, 'ellipsoid_overlap'):
        metrics['mean ellipsoid overlap'] = sample_batch.ellipsoid_overlap.mean().cpu().detach().numpy()
        metrics['ellipsoid overlap'] = sample_batch.ellipsoid_overlap.clip(min=1e-3).log10().cpu().detach().numpy()

    "Custom Figures"
    if do_figures:
        # todo figs to add
        # pairwise dists to eval sample and dataset
        # RDF dists of same
        # clustering / mode counting / basin counting/m            fig = go.Figure()
        #             fig.add_histogram2d(x=condition[:, 0].cpu().detach().numpy(),
        #                                 y=sample_batch.gfn_energy.cpu().detach().numpy(),
        #                                 showscale=False,
        #                                 nbinsx=25, nbinsy=50)
        #             fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Sample Energy')apping
        # known mode coverage
        # diversity vs T / E

        if conditional_flow_model:  # todo update this with molecule conditioning when the time comes
            fig_dict['Learned Z vs T'] = Z_vs_T_fig(gfn_model, init_state)
            fig_dict['T vs Energy'] = T_vs_E_fig(condition, sample_batch)

        fig_dict['TB Parity Plot'] = flow_parity_plot(log_r, log_fs[:, 0], log_pbs, log_pfs)
        fig_dict['Lattice Latents Trajectories'] = visualize_latent_trajs(flow_states.cpu().detach().numpy(),
                                                                          20,
                                                                          log_r.cpu().detach().numpy())
        fig_dict['Lattice Features Distribution'] = simple_cell_hist(sample_batch)
        fig_dict['Lattice Latents Distribution'] = simple_latent_hist(sample_batch)
        fig_dict['Sample Scatter'] = simple_cell_scatter_fig(sample_batch,
                                                             (condition[:,
                                                              0].cpu().detach().numpy()) if condition is not None else None,
                                                             aux_scalar_name='log_temperature' if condition is not None else None)
        fig_dict['Sample Embedding'] = simple_embedding_fig(sample_batch,
                                                            sample_batch.silu_pot.cpu().detach().numpy())  #condition[:, 0].cpu().detach().numpy() if condition is not None else None)
        for key in fig_dict.keys():
            fig = fig_dict[key]
            if get_plotly_fig_size_mb(fig) > 1:  # bigger than 1 MB
                fig.write_image(key + 'fig.png', width=720,
                                height=512)  # save the image rather than the fig, for size reasons
                fig_dict[key] = wandb.Image(key + 'fig.png')
        metrics.update(fig_dict)

    "Crystal samples"
    samples_to_log, filenames = log_crystal_samples(sample_batch=sample_batch, return_filenames=True)
    [wandb.log({f'crystal_sample_{ind}': samples_to_log[ind]}, commit=False) for ind in range(len(samples_to_log))]
    try:
        [os.remove(file) for file in filenames]  # delete this cif as a temporary file
    except:
        pass

    gfn_model.train()
    return metrics


def Z_vs_T_fig(gfn_model, init_state):
    log_temps = torch.linspace(-2, 2, 100).to(init_state.device)[:, None].flatten()
    Z_at_T = gfn_model.flow_model(
        gfn_model.conditions_embedding_model(log_temps[:, None])).cpu().detach().flatten()
    fig = go.Figure(go.Scatter(x=log_temps.cpu().detach(), y=Z_at_T.cpu().detach(), mode='lines+markers'))
    fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Log Partition Function')
    return fig


def T_vs_E_fig(condition, sample_batch):
    fig = go.Figure()
    fig.add_histogram2d(x=condition[:, 0].cpu().detach().numpy(),
                        y=sample_batch.gfn_energy.cpu().detach().numpy(),
                        showscale=False,
                        nbinsx=25, nbinsy=50)
    fig.update_layout(xaxis_title='Log Temperature', yaxis_title='Sample Energy')
    return fig


import numpy as np
import plotly.colors as pc


def diverse_n_colors(n, colorscale='Viridis'):
    """
    Generate `n` visually distinct colors from a Plotly continuous colorscale.

    Args:
        n (int): Number of colors to generate.
        colorscale (str or list): Name of Plotly continuous colorscale, or a custom list.

    Returns:
        list[str]: List of color strings (e.g., 'rgb(68,1,84)')
    """
    if isinstance(colorscale, str):
        try:
            base_scale = pc.get_colorscale(colorscale)
        except ValueError:
            raise ValueError(f"Unknown Plotly colorscale: {colorscale}")
    else:
        base_scale = colorscale

    # Normalize to 0-1 spacing, but skip too-close colors by spreading out non-linearly
    positions = np.linspace(0, 1, n)

    # Use Plotly's built-in interpolation
    interpolated = pc.sample_colorscale(base_scale, positions, colortype='rgb')
    return interpolated


def visualize_latent_trajs(states, n_trajs, log_r):
    lattice_features = ['cell_a', 'cell_b', 'cell_c',
                        'cell_alpha', 'cell_beta', 'cell_gamma',
                        'aunit_x', 'aunit_y', 'aunit_z',
                        'orientation_1', 'orientation_2', 'orientation_3']
    # 1d Histograms
    n_crystal_features = 12
    n_trajs = min(n_trajs, states.shape[0])
    steps = np.arange(states.shape[1])
    trajs = states

    cmap = cm.get_cmap('bwr')  # same as Bluered
    norm_log_r = (log_r - log_r[:n_trajs].min()) / (log_r[:n_trajs].max() - log_r[:n_trajs].min())

    color_hex = [to_hex(cmap(val)) for val in norm_log_r]

    fig = make_subplots(rows=4, cols=3, subplot_titles=lattice_features)
    for i in range(n_crystal_features):
        for j in range(n_trajs):
            row = i // 3 + 1
            col = i % 3 + 1
            fig.add_trace(go.Scatter(
                x=trajs[j, :, i], y=steps,
                name=f"Traj {j}",
                legendgroup=f"Traj {j}",
                opacity=0.5,
                mode='lines',
                marker_line_width=0.5,
                showlegend=True if i == 0 else False,
                marker_color=color_hex[j],
                marker_colorscale='bluered',
            ),
                row=row, col=col
            )
    return fig


def flow_parity_plot(log_r, log_Z_learned, log_pbs, log_pfs):
    # Compute x and y
    x = (log_r + log_pbs.sum(-1)).cpu().detach().numpy()
    y = (log_Z_learned + log_pfs.sum(-1)).cpu().detach().numpy()

    # Get symmetric limits
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    margin = 0.05 * (max_val - min_val)
    lim_low = min_val - margin
    lim_high = max_val + margin
    mae = np.mean(np.abs(x - y))
    r_value, _ = pearsonr(x, y)
    # Build figure
    fig = go.Figure()

    # Identity line
    fig.add_trace(go.Scatter(
        x=[lim_low, lim_high], y=[lim_low, lim_high],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=6, opacity=0.7, color=log_r.cpu().detach()),
        name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
    ))

    # Layout adjustments
    fig.update_layout(
        title='Parity Plot',
        xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
        yaxis=dict(title='log_f + log_pf', range=[lim_low, lim_high], scaleanchor='x', scaleratio=1),
        # width=600,
        # height=600,
        template='plotly_white'
    )

    return fig

''' # version for anywhere-deployment
from scipy.stats import pearsonr

log_Z_learned = log_fs[:, 0]
x = (log_r + log_pbs.sum(-1)).cpu().detach().numpy()
y = (log_Z_learned + log_pfs.sum(-1)).cpu().detach().numpy()

# Get symmetric limits
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
margin = 0.05 * (max_val - min_val)
lim_low = min_val - margin
lim_high = max_val + margin
mae = np.mean(np.abs(x - y))
r_value, _ = pearsonr(x, y)
# Build figure
fig = go.Figure()

# Identity line
fig.add_trace(go.Scatter(
    x=[lim_low, lim_high], y=[lim_low, lim_high],
    mode='lines',
    line=dict(color='gray', dash='dash'),
    showlegend=False,
    hoverinfo='skip'
))

# Scatter points
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(size=6, opacity=0.7, color=log_r.cpu().detach()),
    name=f'Samples<br>MAE = {mae:.3f}<br>R = {r_value:.3f}'
))

# Layout adjustments
fig.update_layout(
    title='Parity Plot',
    xaxis=dict(title='log_r + log_pb', range=[lim_low, lim_high]),
    yaxis=dict(title='log_f + log_pf', range=[lim_low, lim_high], scaleanchor='x', scaleratio=1),
    # width=600,
    # height=600,
    template='plotly_white'
)
fig.show()

'''