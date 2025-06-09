import gc
import os
from time import time

import numpy as np
import torch
import wandb
from mxtaltools.dataset_utils.utils import collate_data_list
from mxtaltools.reporting.online import simple_cell_hist, simple_cell_scatter_fig, log_crystal_samples
from tqdm import trange

from buffer import ReplayBuffer, CrystalReplayBuffer
from energies.molecular_crystal import MolecularCrystal
from energy_sampling.plot_utils import get_plotly_fig_size_mb
from energy_sampling.utils import get_train_args
from evaluations import log_partition_function, mean_log_likelihood, get_sample_metrics
from langevin import langevin_dynamics
from models import GFN
from utils import set_seed, cal_subtb_coef_matrix, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std

args = get_train_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

if args.pis_architectures:
    args.zero_init = True

device = args.device  #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

times = {}


def eval_step(energy, gfn_model, batch_size):
    gfn_model.eval()
    metrics = dict()
    fig_dict = {}
    init_state = torch.zeros(batch_size, energy.data_ndim).to(device)
    samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
        'eval/log_Z_learned'], sample_batch = log_partition_function(
        init_state, gfn_model, energy)

    "Scalar metrics"
    metrics['eval/packing_coeff'] = sample_batch.packing_coeff.mean().cpu().detach().numpy()
    metrics['eval/silu_potential'] = sample_batch.silu_pot.mean().cpu().detach().numpy()
    metrics['eval/energy'] = sample_batch.gfn_energy.mean().cpu().detach().numpy()
    metrics['eval/ellipsoid_overlap'] = sample_batch.ellipsoid_overlap.mean().cpu().detach().numpy()

    "Custom Figures"
    fig_dict['Lattice Features Distribution'] = simple_cell_hist(sample_batch)
    fig_dict['Sample Scatter'] = simple_cell_scatter_fig(sample_batch)
    for key in fig_dict.keys():
        fig = fig_dict[key]
        if get_plotly_fig_size_mb(fig) > 0.1:  # bigger than .1 MB
            fig.write_image(key + 'fig.png', width=1024,
                            height=512)  # save the image rather than the fig, for size reasons
            fig_dict[key] = wandb.Image(key + 'fig.png')
    metrics.update(fig_dict)

    "Crystal samples"
    samples_to_log = log_crystal_samples(sample_batch=sample_batch)
    [wandb.log({f'crystal_sample_{ind}': samples_to_log[ind]}, commit=False) for ind in range(len(samples_to_log))]

    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, buffer, exploration_factor,
               exploration_wd):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if False:  #TODO REIMPLEMENT # args.sampling == 'buffer':
                loss, states, _, _, log_r = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                buffer.add(states[:, -1], log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(gfn_model.parameters(),
                                   args.gradient_norm_clip)  # gradient clipping
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss


def bwd_train_step(energy, gfn_model, buffer, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':
        # if args.local_search:  # We are just not doing on the fly local search any more (slow), but using a prebuilt dataset.
        #     if it % args.ls_cycle < 2:
        #         if args.energy == 'molecular_crystal':
        #             # sample reasonable diffuse crystals
        #             samples = energy.sample(args.batch_size,
        #                                     reasonable_only=True,
        #                                     target_packing_coeff=0.5)
        #             # optimize them
        #             local_search_samples, log_r = energy.local_opt(samples,
        #                                                            args.max_iter_ls,
        #                                                            args.samples_per_opt,
        #                                                            )
        #         else:
        #             samples, rewards = buffer.sample()
        #             local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
        #         buffer_ls.add(local_search_samples, log_r)
        #
        #     samples, rewards = buffer_ls.sample()
        # else:
        samples, rewards = buffer.sample()
    else:
        assert False, f"sampling method {args.sampling} not implemented"

    loss = get_gfn_backward_loss(args.mode_bwd, samples.to(device), gfn_model, energy.log_reward,
                                 exploration_std=exploration_std)
    return loss


def train():
    times['initialization_start'] = time()
    name = args.run_name
    if not os.path.exists(name):
        os.makedirs(name)

    energy_function = MolecularCrystal(device=device, temperature=args.energy_temperature)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)

    gfn_model = GFN(energy_function.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
                    langevin=args.langevin, learned_variance=args.learned_variance,
                    partial_energy=args.partial_energy, log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, dropout=args.dropout, norm=args.norm,
                    zero_init=args.zero_init, device=device).to(device)

    wandb.watch(gfn_model, log_graph=True, log_freq=100)  # for gradient logging

    gfn_optimizer = get_gfn_optimizer(gfn_model, args.lr_policy, args.lr_flow, args.lr_back, args.learn_pb,
                                      args.conditional_flow_model, args.use_weight_decay, args.weight_decay)

    buffer = CrystalReplayBuffer(
        args.buffer_size,
        device,
        energy_function,
        args.batch_size,
        beta=args.beta,
        rank_weight=args.rank_weight,
        prioritized=args.prioritized)

    if args.learn_pb and args.dataset_path is not None:  # preload samples into the buffer
        buffer = add_dataset_to_buffer(args.dataset_path, buffer)

    gfn_model.train()

    times['initialization_end'] = time()
    loss_record = []
    oomed_out = False
    old_params = [p.clone().detach() for p in gfn_model.parameters()]
    for i in trange(args.epochs + 1):
        metrics = dict()
        times['train_step_start'] = time()
        try:
            metrics['train/loss'] = train_step(energy_function, gfn_model, gfn_optimizer, i, args.exploratory,
                                               buffer, args.exploration_factor, args.exploration_wd)
            loss_record.append(metrics['train/loss'])
            if not oomed_out:
                if args.batch_size < args.max_batch_size and args.grow_batch_size:
                    args.batch_size = max(args.batch_size + 1,
                                          int(args.batch_size * 1.01))  # gradually increment batch size

        except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
            if "CUDA out of memory" in str(
                    e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
                args.batch_size = handle_oom(args.batch_size)
                oomed_out = True
                print(f"Reducing batch size to {args.batch_size}")
            else:
                raise e  # will simply raise error if other or if training on CPU
        times['train_step_end'] = time()

        if (i % args.eval_period == 0 and i > 0) or i == 50:

            times['eval_step_start'] = time()
            metrics.update({'Batch Size': args.batch_size})
            eval_batch_size = min(max(args.batch_size, 500), 1000)
            metrics.update(eval_step(energy_function, gfn_model, eval_batch_size))

            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']

            if args.energy == 'molecular_crystal' and args.anneal_energy:
                with torch.no_grad():
                    total_change = 0.0
                    total_norm = 0.0
                    for p, old_p in zip(gfn_model.parameters(), old_params):
                        delta = (p - old_p).norm()
                        total_change += delta.item()
                        total_norm += p.norm().item()
                    relative_change = total_change / (total_norm + 1e-8)
                    metrics['relative_gradient_change'] = relative_change
                    old_params = [p.clone().detach() for p in gfn_model.parameters()]

                anneal_energy(energy_function, relative_change < 2e-3)

            metrics.update({'Crystal Temperature': energy_function.temperature,
                            'Crystal Turnover Potential': energy_function.turnover_pot})
            times['eval_step_end'] = time()
            metrics.update(log_elapsed_times())
            wandb.log(metrics, step=i)

        elif i % 10 == 0:
            metrics.update(log_elapsed_times())
            wandb.log(metrics, step=i)

        if i % 100 == 0:
            torch.save(gfn_model.state_dict(), f'{name}model.pt')

    torch.save(gfn_model.state_dict(), f'{name}model_final.pt')


def log_elapsed_times():
    elapsed_times = {}
    for key in times.keys():
        if 'start' in key:
            start_key = key
            end_key = start_key.split('_start')[0] + '_end'
            if end_key in times.keys():
                elapsed_times[start_key.split('_start')[0] + '_time'] = times[end_key] - times[start_key]

    return elapsed_times


def anneal_energy(energy_function, trigger: bool, min_temperature: float = 0.01):
    # values = np.array(values)[-100:]
    # # Avoid log(0) or log of very small numbers
    # values = np.clip(values, 1e-12, None)
    # log_values = np.log(values)
    #
    # slope, _ = np.polyfit(np.arange(len(log_values)), log_values, 1)
    # curvature, _, _ = np.polyfit(np.arange(len(log_values)), values, 2)

    if trigger: #slope > -1e-3 and curvature > -1e-1:  # if the loss isn't decaying fast enough / the loss is saturated
        if energy_function.temperature > min_temperature:
            energy_function.temperature *= 0.9
            print("Annealing energy function")


def handle_oom(batch_size):
    gc.collect()
    torch.cuda.empty_cache()
    batch_size = int(batch_size * 0.9)
    return batch_size


def add_dataset_to_buffer(dataset_path, buffer):
    print("Loading prebuilt buffer")
    dataset = torch.load(dataset_path)
    buffer.add(dataset)
    print(f"Buffer loaded with {len(dataset)} samples")

    return buffer


def update_buffer_reward(buffer,
                         energy):
    rescaled_reward = -energy.soften_LJ_energy(buffer.raw_reward_dataset.rewards)
    buffer.reward_dataset.rewards = rescaled_reward
    buffer.reward_dataset.raw_tsrs = rescaled_reward
    return buffer


if __name__ == '__main__':
    train()
