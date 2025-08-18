import argparse
import gc
import math
import os
import random
from argparse import Namespace
from pathlib import Path

import psutil

import PIL
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.common.geometry_utils import batch_molecule_principal_axes_torch, batch_cell_vol_torch
from mxtaltools.crystal_building.crystal_latent_transforms import enforce_niggli_plane
from mxtaltools.models.utils import load_encoder


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def dcp(tensor):
    return tensor.detach().cpu()


def gaussian_params(tensor):
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow,
                      conditional_flow_model=False,
                      use_weight_decay=False, weight_decay=1e-7):
    param_groups = [{'params': gfn_model.t_model.parameters()},
                    {'params': gfn_model.s_model.parameters()},
                    {'params': gfn_model.forward_policy.parameters()},
                    {'params': gfn_model.backward_policy.parameters()},
                    {'params': gfn_model.flow_model.parameters(), 'lr': lr_flow}
                    ]

    if conditional_flow_model:
        param_groups += [{'params': gfn_model.conditions_embedding_model.parameters(),
                          'lr': lr_policy}]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer


def get_exploration_std(iter, exploratory, max_steps: int = 5000, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        exploration_factor = 0
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / max_steps)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl


def get_name(args):
    name = ''
    if args.langevin:
        name = f'langevin_'
        if args.langevin_scaling_per_dimension:
            name = f'langevin_scaling_per_dimension_'
    if args.exploratory and (args.exploration_factor is not None):
        if args.exploration_wd:
            name = f'exploration_wd_{args.exploration_factor}_{name}_'
        else:
            name = f'exploration_{args.exploration_factor}_{name}_'

    if args.learn_pb:
        name = f'{name}learn_pb_scale_range_{args.pb_scale_range}_'

    if args.clipping:
        name = f'{name}clipping_lgv_{args.lgv_clip}_gfn_{args.gfn_clip}_'

    if args.mode_fwd == 'subtb':
        mode_fwd = f'subtb_subtb_lambda_{args.subtb_lambda}'
        if args.partial_energy:
            mode_fwd = f'{mode_fwd}_{args.partial_energy}'
    else:
        mode_fwd = args.mode_fwd

    if args.both_ways:
        ways = f'fwd_bwd/fwd_{mode_fwd}_bwd_{args.mode_bwd}'
    elif args.bwd:
        ways = f'bwd/bwd_{args.mode_bwd}'
    else:
        ways = f'fwd/fwd_{mode_fwd}'

    if args.local_search:
        local_search = f'local_search_iter_{args.max_iter_ls}_burn_{args.burn_in}_cycle_{args.ls_cycle}_step_{args.ld_step}_beta_{args.beta}_rankw_{args.rank_weight}_prioritized_{args.prioritized}'
        ways = f'{ways}/{local_search}'

    if args.pis_architectures:
        results = 'results_pis_architectures'
    else:
        results = 'results'

    name = f'{results}/{args.energy}/{name}gfn/{ways}/T_{args.T}/tscale_{args.t_scale}/lvr_{args.log_var_range}/'

    name = f'{name}/seed_{args.seed}/'

    return name


def get_train_args():
    parser = argparse.ArgumentParser(description='GFN Linear Regression')
    args, remaining = parser.parse_known_args()

    return dict2namespace(load_yaml(remaining[1]))


def load_yaml(path):
    """
    Safely load yaml file as dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


def dict2namespace(data_dict: dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def get_gfn_init_state(batch_size, ndim, device):
    init_state = torch.zeros(batch_size, ndim).to(device)

    return init_state


def uniform_discretizer(bsz, trajectory_length):
    return torch.linspace(0, 1, trajectory_length + 1).repeat(bsz, 1)


def random_discretizer(bsz, trajectory_length, max_ratio):
    x = (torch.rand(bsz, trajectory_length) * (max_ratio - 1) + 1).cumsum(1)
    x = torch.cat([torch.zeros(bsz, 1), x], 1) / x[:, -1].unsqueeze(1)
    return x


def low_discrepancy_discretizer(bsz, traj_length=2):
    u = torch.rand(1, traj_length - 1)
    u_sorted, _ = torch.sort(u, dim=-1, descending=False)
    # print(u_sorted)
    # print(u_sorted.shape)
    shift_vector = (torch.arange(bsz) / bsz).unsqueeze(1).repeat(1, traj_length - 1)
    timestep = u + shift_vector
    timesteps_in_range = timestep % 1.0
    timesteps_sorted, indices = torch.sort(timesteps_in_range, dim=-1, descending=False)
    x = torch.cat([torch.zeros(bsz, 1), timesteps_sorted, torch.ones(bsz, 1)], dim=1)
    return x

    # old code below:
    # u = torch.rand(1)
    # shift_vector = torch.arange(bsz)/bsz
    # timestep = u + shift_vector
    # timestep_in_range = timestep % 1.0
    # timestep_in_range = timestep_in_range.unsqueeze(-1)
    # x = torch.cat([torch.zeros(bsz, 1), timestep_in_range, torch.ones(bsz, 1)], 1)
    # return x


def low_discrepancy_discretizer2(bsz, traj_length=2):
    s = traj_length - 1
    u = torch.rand(1, s)
    shift_vector = torch.arange(bsz) / bsz
    timestep = u + shift_vector.unsqueeze(-1)
    timestep_in_range = timestep % 1.0
    x = (timestep_in_range + torch.arange(s).unsqueeze(0)) / s
    x = torch.stack([col[torch.randperm(col.size(0))] for col in x.t()]).t()
    return x


def shifted_equidistant(bsz, traj_length, eps=1e-4):
    bound = 1 / traj_length - eps
    noise = torch.empty(bsz, 1).uniform_(- bound, bound)
    steps = (torch.arange(1, traj_length) / traj_length).unsqueeze(0) + noise
    return torch.cat([torch.zeros(bsz, 1), steps, torch.ones(bsz, 1)], dim=1)


def report_mem(tag=""):
    gc.collect()
    print(f"[{tag}] RSS = {psutil.Process(os.getpid()).memory_info().rss / 1e6:.2f} MB")


def compute_sample_overlap(ref_x, sample_x=None, ga: float = 1.0, agg='sum'):
    if sample_x is None:
        d = torch.cdist(ref_x, ref_x) + torch.eye(len(ref_x), device=ref_x.device) * 100
    else:
        d = torch.cdist(ref_x, sample_x)

    if agg == 'sum':
        return torch.exp(-ga * d ** 2).sum(dim=0)
    elif agg == 'mean':
        return torch.exp(-ga * d ** 2).mean(dim=0)


def smoothstep(x, t, delta):
    if not torch.is_tensor(x):
        x = torch.tensor([x])
    x_clipped = torch.clamp((x - t) / delta, 0.0, 1.0)
    return float(x_clipped ** 2 * (3 - 2 * x_clipped))  # smoothstep polynomial


def triangle_schedule(it, init, maxval, minval, on, off):
    if it <= on:
        # ramp up
        frac = (it / on) if on > 0 else 1.0
        return init * (1 - frac) + maxval * frac
    elif on < it <= off:
        # ramp down
        frac = (it - on) / (off - on) if (off - on) > 0 else 1.0
        return maxval * (1 - frac) + minval * frac
    else:
        return minval


@torch.inference_mode()
def featurize_dataset(dataset, device, ellipsoid_scale, lj_repulsion, batch_size: int = 500):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False
    )
    overlaps = []
    silus = []
    ljs = []
    niggli_overlaps = []

    for crystal_batch in tqdm(loader):
        crystal_batch = crystal_batch.to(device)
        crystal_batch.box_analysis()
        cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                  supercell_size=10,
                                                  align_to_standardized_orientation=True)

        cluster_batch.construct_radial_graph(cutoff=6)

        lj_energy, normed_lj_energy = cluster_batch.compute_LJ_energy()
        silu_energy = cluster_batch.compute_silu_energy(
            repulsion=lj_repulsion,
        )

        # simplified ellipsoid energy testing
        _, _, _, _, _, _, normed_ellipsoid_overlap \
            = cluster_batch.compute_ellipsoidal_overlap(
            surface_padding=ellipsoid_scale,
            return_details=True)

        niggli_overlap = cluster_batch.compute_niggli_overlap()

        overlaps.extend(normed_ellipsoid_overlap.cpu().detach())
        silus.extend(silu_energy.cpu().detach())
        ljs.extend(lj_energy.cpu().detach())
        niggli_overlaps.extend(niggli_overlap.cpu().detach())

    overlaps = torch.tensor(overlaps)
    silus = torch.tensor(silus)
    ljs = torch.tensor(ljs)
    niggli_overlaps = torch.tensor(niggli_overlaps)
    for ind, elem in enumerate(dataset):
        elem.ellipsoid_overlap = torch.ones(1) * overlaps[ind]
        elem.silu_pot = torch.ones(1) * silus[ind]
        elem.lj_pot = torch.ones(1) * ljs[ind]
        elem.niggli_overlap = torch.ones(1) * niggli_overlaps[ind]

    # exclude negative niggli overlaps
    dataset = [elem for elem in dataset if elem.niggli_overlap >= 0]
    [elem.box_analysis() for elem in dataset]

    return dataset

@torch.inference_mode()
def embed_dataset(dataset, autoencoder_path=None, device=None, encoder=None, embedding_type='autoencoder'):
    batch_size = 500
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False
    )
    with torch.no_grad():
        if encoder is None and embedding_type == 'autoencoder':
            encoder = load_encoder(autoencoder_path).to(device).eval()

        embeddings = []

        for crystal_batch in tqdm(loader):
            crystal_batch = crystal_batch.to(device)
            # for now, make all the embeddings exactly standardized,
            # and we'll generate also in the standardized basis
            crystal_batch.orient_molecule(mode='standardized',
                                          target_handedness=torch.ones_like(crystal_batch.radius)
                                          )
            if embedding_type == 'autoencoder':
                embeddings.append(encoder.encode(crystal_batch).clone().cpu())
            elif embedding_type == 'principal_axes':
                v_embedding_i, s_embedding_i, _ = batch_molecule_principal_axes_torch(
                    crystal_batch.pos,
                    crystal_batch.batch,
                    crystal_batch.num_graphs,
                    crystal_batch.num_atoms,
                )
                embeddings.append((v_embedding_i * s_embedding_i[:, :, None]).cpu())

            del crystal_batch

        embeddings = torch.cat(embeddings, dim=0)
        for ind, elem in enumerate(dataset):
            elem.embedding = embeddings[None, ind]

    return dataset


def get_conditioning_dim(args):
    conditioning_dim = 0
    if args.temperature_conditioning:
        conditioning_dim += 1
    if args.molecule_conditioning:
        if args.mol_embedding_type == 'autoencoder':
            conditioning_dim += 64 * 3
        elif args.mol_embedding_type == 'principal_axes':
            conditioning_dim += 9
        else:
            assert False
    if args.sg_conditioning:
        conditioning_dim += 237
    return conditioning_dim


def anneal_reward(it, temp_annealing_lambda, energy_function, args):
    """anneal reward function"""
    if args.anneal_temperature:
        if args.temperature_conditioning:
            if energy_function.temperature_scaling_factor < 1:
                energy_function.temperature_scaling_factor *= temp_annealing_lambda
        else:
            if energy_function.temperature > args.energy_min_temperature:
                energy_function.temperature *= temp_annealing_lambda

    if args.core_start_time > 0:
        energy_function.core_coeff = round(
            args.energy_core_coeff * F.sigmoid(torch.tensor((it - args.core_start_time) / 50)).item(), 2)
    if args.lj_start_time > 0:
        energy_function.lj_coeff = round(
            args.energy_lj_coeff * F.sigmoid(torch.tensor((it - args.lj_start_time) / 50)).item(), 2)


def set_loss_coeffs(it, args):
    """anneal reward function"""
    if it == 0:
        args.fwd_loss_schedule = parse_loss_schedules(args.fwd_loss_coeffs)
        args.bwd_loss_schedule = parse_loss_schedules(args.bwd_loss_coeffs)

        args.fwd_loss_coeffs = dict2namespace({k: 0.0 for k in args.fwd_loss_schedule})
        args.bwd_loss_coeffs = dict2namespace({k: 0.0 for k in args.bwd_loss_schedule})

    update_loss_schedule(it, args.fwd_loss_schedule, args.fwd_loss_coeffs.__dict__)
    update_loss_schedule(it, args.bwd_loss_schedule, args.bwd_loss_coeffs.__dict__)


def parse_loss_schedules(loss_coeffs_config):
    """
    Parse loss coefficient configuration into standardized format.

    Input formats:
    - Single value: coeff_name: 1.5 -> constant schedule
    - List of [step, value] pairs: coeff_name: [[0, 0.0], [1000, 2.0]]

    Returns dict of {coeff_name: [(step, value), ...]}
    """
    schedules = {}

    # Handle both dict and namespace objects
    if hasattr(loss_coeffs_config, '__dict__'):
        config_dict = loss_coeffs_config.__dict__
    else:
        config_dict = loss_coeffs_config

    for key, value in config_dict.items():
        if isinstance(value, (int, float)):
            # Single constant value
            schedules[key] = [(0, float(value))]
        elif isinstance(value, list) and len(value) > 0:
            # List of [step, value] pairs
            if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
                # Validate and sort by step
                schedule = [(int(step), float(val)) for step, val in value]
                schedule.sort(key=lambda x: x[0])  # Sort by step

                # Validate steps are non-negative and ascending
                for i, (step, val) in enumerate(schedule):
                    if step < 0:
                        raise ValueError(f"Step {step} for {key} must be non-negative")
                    if i > 0 and step < schedule[i - 1][0]:
                        raise ValueError(f"Steps for {key} must be in ascending order")

                schedules[key] = schedule
            else:
                raise ValueError(f"Invalid schedule format for {key}: {value}")
        else:
            raise ValueError(f"Invalid schedule format for {key}: {value}")

    return schedules


def evaluate_schedule(step, schedule):
    """
    Evaluate a piecewise linear schedule at given step.

    Args:
        step: Current training step
        schedule: List of (step, value) tuples, sorted by step

    Returns:
        Interpolated value at the given step
    """
    if len(schedule) == 1:
        # Constant schedule
        return schedule[0][1]

    # Find the appropriate segment
    for i in range(len(schedule) - 1):
        step1, val1 = schedule[i]
        step2, val2 = schedule[i + 1]

        if step <= step1:
            return val1
        elif step1 < step <= step2:
            # Linear interpolation between points
            if step1 == step2:  # Avoid division by zero
                return val2

            alpha = (step - step1) / (step2 - step1)
            return val1 + alpha * (val2 - val1)

    # Past the last point, return final value
    return schedule[-1][1]


def update_loss_schedule(it, loss_schedules, active_coeffs):
    """Update active coefficients based on current iteration and schedules"""
    for key, schedule in loss_schedules.items():
        active_coeffs[key] = evaluate_schedule(it, schedule)


def sample_crystal_prior(crystal_batch, std):
    rands = torch.randn((crystal_batch.num_graphs, 12), device=crystal_batch.device) * std

    # enforce the random prior is in the positive niggli plane
    if not hasattr(crystal_batch, 'latent_transform'):
        crystal_batch.init_latent_transform()
    temp_params = crystal_batch.latent_transform.inverse(rands,
                                                         crystal_batch.sg_ind,
                                                         crystal_batch.radius)
    cell_lengths = temp_params[:, :3]
    cell_angles = temp_params[:, 3:6]

    # rescale cell lengths for a good packing coeff
    target_packing_coeff = (torch.randn(crystal_batch.num_graphs, device=crystal_batch.device) * 0.075 + 0.65).clip(
        min=0.55, max=0.95)
    vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
    cp1 = crystal_batch.mol_volume * crystal_batch.sym_mult / vol1
    correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
    cell_lengths *= correction_ratio[:, None]

    # enforce positive side of niggli plane
    cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')
    temp_params[:, 3:6] = cell_angles

    prior_samples = crystal_batch.latent_transform.forward(temp_params,
                                                           crystal_batch.sg_ind,
                                                           crystal_batch.radius
                                                           ).clip(min=-6, max=6)

    return prior_samples
