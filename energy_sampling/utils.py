import argparse
import gc
import math
import os
import random
from argparse import Namespace
from asyncio import sleep
from pathlib import Path
from typing import Optional

import PIL
import numpy as np
import psutil
import torch
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mxtaltools.common.config_processing import dict2namespace
from mxtaltools.common.geometry_utils import batch_molecule_principal_axes_torch
from mxtaltools.common.utils import log_rescale_positive
# from mxtaltools.crystal_building.crystal_latent_transforms import enforce_niggli_plane
from mxtaltools.dataset_utils.data_classes import MolCrystalData
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.mlip_interfaces.uma_utils import init_uma_crystal_predictor
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
    # dt = x.diff(dim=1)
    # too_small = dt < 5e-3

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


@torch.no_grad()
def featurize_dataset(dataset, device, energy_function: str, batch_size: int = 500,
                      uma_path: Optional[str] = None, ):
    outputs = []

    cutoff = 10
    computes = ['lj', 'reduction_en']
    if energy_function != 'lj' and energy_function != 'uma':
        computes.append(energy_function)

    if energy_function == 'uma':
        uma_predictor = init_uma_crystal_predictor(uma_path, device=device)

    cursor = 0
    pbar = tqdm(total=len(dataset), unit="reparameterized samples")

    while cursor < len(dataset):
        try:
            crystal_batch = collate_data_list(
                [dataset[ind] for ind in range(cursor, min(len(dataset), cursor + batch_size))])
            crystal_batch = crystal_batch  # .to(device)

            crystal_batch.box_analysis()
            out = crystal_batch.analyze(computes,
                                        cutoff=cutoff,
                                        supercell_size=5,
                                        std_orientation=True,
                                        )
            if energy_function == 'uma':
                cry_en = crystal_batch.compute_crystal_uma(
                    predictor=uma_predictor,
                    std_orientation=True).cpu().detach() * 96.485  # output in kJ/mol (of unit cells)
                gas_en = crystal_batch.compute_lattice_gas_phase_uma(
                    predictor=uma_predictor, std_orientation=True).cpu().detach() * 96.485
                out.update({'uma_gas_pot': gas_en,
                            'uma_pot': cry_en,
                            'uma': cry_en / (crystal_batch.sym_mult * crystal_batch.z_prime) - gas_en})  # lattice energy

            outputs.append(out)
            cursor += batch_size
            pbar.update(min(batch_size, len(dataset) - cursor))  # safe final update
            batch_size += 1

        except (RuntimeError, ValueError) as e:
            if is_cuda_oom(e):
                batch_size = max(int(batch_size * 0.6), 1)
                print(f"OOM error: dropping batch size to {batch_size}")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sleep(0.1)
            else:
                raise e

    keys = outputs[0].keys()

    full = {
        key: torch.cat([batch[key] for batch in outputs], dim=0)
        for key in keys
    }

    for key, tensor in full.items():
        for i, elem in enumerate(dataset):
            setattr(elem, key, tensor[None, i])

    if 'lj' in full:
        scaled = log_rescale_positive(full['lj'])
        for i, elem in enumerate(dataset):
            setattr(elem, 'scaled_lj', scaled[None, i])

    [elem.box_analysis() for elem in dataset]

    return dataset


@torch.no_grad()
def embed_dataset(dataset, autoencoder_path=None, device=None, encoder=None, embedding_type='autoencoder',
                  ):
    batch_size = 500
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False
    )
    if encoder is None and embedding_type == 'autoencoder':
        encoder = load_encoder(autoencoder_path).to(device).eval()

    embeddings = []

    for crystal_batch in tqdm(loader):
        crystal_batch = crystal_batch.to(device)
        # for now, make all the embeddings exactly standardized,
        # and we'll generate also in the standardized basis
        crystal_batch.orient_molecule(mode='standardized',
                                      target_handedness=torch.ones_like(crystal_batch.radius)[:, None],
                                      )
        if embedding_type == 'autoencoder':
            # NOTE - autoencoder trained on global centroids, but recenter molecules in orient_molecule works on heavy atoms now
            crystal_batch.recenter_molecules(center_on_heavy_atoms=False)
            embeddings.append(encoder.encode(crystal_batch).clone().cpu())
        elif embedding_type == 'principal_axes':
            v_embedding_i, s_embedding_i, _ = batch_molecule_principal_axes_torch(
                crystal_batch.pos,
                crystal_batch.batch,
                crystal_batch.num_graphs,
                crystal_batch.num_atoms,
                heavy_atoms_only=True,
                atom_types=crystal_batch.z
            )
            embeddings.append((v_embedding_i * s_embedding_i[:, :, None]).cpu())

        del crystal_batch

    embeddings = torch.cat(embeddings, dim=0)
    for ind, elem in enumerate(dataset):
        elem.embedding = embeddings[None, ind]

    return dataset


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
    assert False, "This method needs to be rewritten as the latent prior is no longer std normal"
    # rands = torch.randn((crystal_batch.num_graphs, 12), device=crystal_batch.device) * std
    #
    # # enforce the random prior is in the positive niggli plane
    # if not hasattr(crystal_batch, 'latent_transform'):
    #     crystal_batch.init_latent_transform()
    # temp_params = crystal_batch.latent_transform.inverse(rands,
    #                                                      crystal_batch.sg_ind,
    #                                                      crystal_batch.radius)
    # cell_lengths = temp_params[:, :3]
    # cell_angles = temp_params[:, 3:6]
    #
    # # rescale cell lengths for a good packing coeff
    # target_packing_coeff = (torch.randn(crystal_batch.num_graphs, device=crystal_batch.device) * 0.075 + 0.65).clip(
    #     min=0.55, max=0.95)
    # vol1 = batch_cell_vol_torch(cell_lengths, cell_angles)
    # cp1 = crystal_batch.mol_volume * crystal_batch.sym_mult / vol1
    # correction_ratio = (cp1 / target_packing_coeff) ** (1 / 3)
    # cell_lengths *= correction_ratio[:, None]
    #
    # # enforce positive side of niggli plane
    # cell_angles = enforce_niggli_plane(cell_lengths, cell_angles, mode='mirror')
    # temp_params[:, 3:6] = cell_angles
    #
    # prior_samples = crystal_batch.latent_transform.forward(temp_params,
    #                                                        crystal_batch.sg_ind,
    #                                                        crystal_batch.radius
    #                                                        ).clip(min=-6, max=6)
    #
    # return prior_samples


@torch.no_grad()
def update_ema(model, ema_model, decay=0.9999):
    """
    Update ema_model parameters towards model parameters.

    Args:
        model:      nn.Module, the training model
        ema_model:  nn.Module, the EMA copy
        decay:      float, EMA decay (close to 1.0 gives long memory)
    """
    if decay is not None:
        if decay > 0:
            msd = model.state_dict()
            emsd = ema_model.state_dict()
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    emsd[k].mul_(decay).add_(msd[k], alpha=1 - decay)
                else:
                    emsd[k] = msd[k]  # copy over non-float buffers (e.g. ints, bools)
    else:
        msd = model.state_dict()
        emsd = ema_model.state_dict()
        for k in msd.keys():  # simply overwrite state dict to EMA model
            emsd[k] = msd[k]


def manual_batch_to_data_list(batch):
    ptr = batch.ptr
    num_graphs = batch.num_graphs

    # Pre-split all tensor attributes into lists of [num_graphs] length
    attr_splits = {}
    for key in batch.keys():
        if key not in ['batch', 'ptr', 'edges_dict',
                       'niggli_energy',
                       'reduction_en',
                       'core_energy',
                       'density_energy',
                       'lj_energy',
                       'bounding_energy',
                       'asym_unit_dict',
                       'latent_transform', ]:
            value = batch[key]
            if torch.is_tensor(value) and value.size(0) == ptr[-1]:
                # node-level attributes
                attr_splits[key] = torch.split(value, torch.diff(ptr).tolist())
            elif torch.is_tensor(value) and value.size(0) == num_graphs:
                # graph-level attributes
                attr_splits[key] = value.unsqueeze(1)
            elif len(value) == num_graphs:
                # graph-level, list attrubutes
                attr_splits[key] = value

    data_list = []
    for ind in range(num_graphs):
        data = MolCrystalData()

        # here assign attributes to object
        for key, splits in attr_splits.items():
            setattr(data, key, splits[ind])

        data_list.append(data)

    return data_list


def iter_forever(loader):
    while True:
        for batch in loader:
            yield batch


def is_cuda_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    s = str(e).lower()
    return (
            ("cuda" in s and "memory" in s)
            or ("cublas" in s and "alloc" in s)
            or ("cusolver" in s and "alloc" in s)
            or ("out of memory" in s)
            or ("nonzero is not supported for tensors with more than int_max elements" in s)
    )


def get_annealing_factor(start_value, stop_value, total_time, step_iters):
    assert stop_value > 0, "Setting final value as zero breaks this module"
    return (stop_value / start_value) ** (1 / (total_time / step_iters))


def substitute_prior(loss_coeffs, condition, crystal_batch, energy_function, rewards, samples, buffer):
    # noise buffer samples with gaussian magnitude steps
    rand_dir = torch.randn_like(samples)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True)
    rand_magnitude = torch.randn(len(samples), device=samples.device).abs() * loss_coeffs.noise_level
    noised_samples = (samples + rand_dir * rand_magnitude[:, None]).clip(min=-1, max=1)
    new_samples = samples.clone()
    if loss_coeffs.noised_fraction < 1:
        num_to_replace = max(1, int(len(samples) * loss_coeffs.noised_fraction))
        inds_to_replace = np.random.choice(len(samples), num_to_replace, replace=False)
        new_samples[inds_to_replace] = noised_samples[inds_to_replace]
    else:
        new_samples = noised_samples

    # have to update the rewards if we are using any loss functions that require them
    # otherwise, if we're not using the reward, just pass the raw sample
    # recondition and rescore
    if any([
        loss_coeffs.tb > 0,
        loss_coeffs.vg_lb > 0,
        loss_coeffs.vg_lme > 0,
    ]):
        log_T_tensor, sg_inds, condition = energy_function.get_conditioning_tensor(crystal_batch,
                                                                                   sg_inds=crystal_batch.sg_ind,
                                                                                   z_primes=crystal_batch.z_prime)
        if log_T_tensor is not None:
            log_temperature = log_T_tensor
        else:
            log_temperature = None
        with torch.no_grad():
            # todo update this when the time comes for conditional rotation business
            crystal_batch.orient_molecule(mode='std')
            new_rewards = energy_function.log_reward(new_samples.to(energy_function.device),
                                                     crystal_batch.to(energy_function.device),
                                                     log_temperature.to(energy_function.device),
                                                     False)

    return condition, new_rewards, new_samples, crystal_batch
