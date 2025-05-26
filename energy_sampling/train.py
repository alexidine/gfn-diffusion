import gc

import yaml
from mxtaltools.reporting.online import simple_cell_hist, simple_cell_scatter_fig, log_crystal_samples

from energies.molecular_crystal import MolecularCrystal
from plot_utils import *
import argparse
import torch
import os

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name
from buffer import ReplayBuffer
from langevin import langevin_dynamics
from models import GFN
from gflownet_losses import *
from energies import *
from evaluations import *

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--run_name', type=str, default='test')
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--norm', type=str, default=None)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--max_batch_size', type=int, default=300)
parser.add_argument('--grow_batch_size', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--eval_period', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str,
                    default='molecular_crystal')  # this thing is mostly hardcoded for molecular crystals now
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)
parser.add_argument('--samples_per_opt', type=int, default=10)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)

# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################

parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)

# args for molecular crystal energy
parser.add_argument('--energy_temperature', type=float, default=1)
parser.add_argument('--anneal_energy', type=bool, default=False)
parser.add_argument('--energy_annealing_threshold', type=float, default=0)

args, remaining = parser.parse_known_args()

if 'config' in remaining[0]:  # load external yaml config file
    with open(remaining[1], 'r') as f:
        config_args = yaml.safe_load(f)
    for key, value in config_args.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            parser.error(f"Unknown config key: {key}")

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

eval_batch_size = min(args.batch_size, 1000)

def get_energy():
    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    elif args.energy == 'molecular_crystal':
        energy = MolecularCrystal(device=device, temperature=args.energy_temperature)
    else:
        assert False, f"{args.energy} is not a valid energy function"
    return energy


def eval_step(energy, gfn_model, batch_size):
    gfn_model.eval()
    metrics = dict()
    init_state = torch.zeros(batch_size, energy.data_ndim).to(device)
    samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
        'eval/log_Z_learned'], sample_batch = log_partition_function(
        init_state, gfn_model, energy)
    metrics['Lattice Features Distribution'] = simple_cell_hist(sample_batch)
    metrics['Sample Scatter'] = simple_cell_scatter_fig(sample_batch)
    metrics['eval/packing_coeff'] = sample_batch.packing_coeff.mean().cpu().detach().numpy()
    metrics['eval/energy'] = sample_batch.silu_pot.mean().cpu().detach().numpy()
    samples_to_log = log_crystal_samples(sample_batch=sample_batch)
    [wandb.log({f'crystal_sample_{ind}': samples_to_log[ind]}, commit=False) for ind in range(len(samples_to_log))]

    gfn_model.train()
    return metrics


def train_step(energy, gfn_model, gfn_optimizer, it, exploratory, buffer, buffer_ls, exploration_factor,
               exploration_wd):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(it, exploratory, exploration_factor, exploration_wd)

    if args.both_ways:
        if it % 2 == 0:
            if args.sampling == 'buffer':
                loss, states, _, _, log_r = fwd_train_step(energy, gfn_model, exploration_std, return_exp=True)
                buffer.add(states[:, -1], log_r)
            else:
                loss = fwd_train_step(energy, gfn_model, exploration_std)
        else:
            loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)

    elif args.bwd:
        loss = bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std, it=it)
    else:
        loss = fwd_train_step(energy, gfn_model, exploration_std)

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(energy, gfn_model, exploration_std, return_exp=False):
    init_state = torch.zeros(args.batch_size, energy.data_ndim).to(device)
    loss = get_gfn_forward_loss(args.mode_fwd, init_state, gfn_model, energy.log_reward, coeff_matrix,
                                exploration_std=exploration_std, return_exp=return_exp)
    return loss


def bwd_train_step(energy, gfn_model, buffer, buffer_ls, exploration_std=None, it=0):
    if args.sampling == 'sleep_phase':
        samples = gfn_model.sleep_phase_sample(args.batch_size, exploration_std).to(device)
    elif args.sampling == 'energy':
        samples = energy.sample(args.batch_size).to(device)
    elif args.sampling == 'buffer':  # todo cleanup the logic here - for molecular crystal we should really just have one method
        if args.local_search:
            if it % args.ls_cycle < 2:
                if args.energy == 'molecular_crystal':
                    # sample reasonable diffuse crystals
                    samples = energy.sample(args.batch_size,
                                            reasonable_only=True,
                                            target_packing_coeff=0.5)
                    # optimize them
                    local_search_samples, log_r = energy.local_opt(samples,
                                                                   args.max_iter_ls,
                                                                   args.samples_per_opt,
                                                                   )
                else:
                    samples, rewards = buffer.sample()
                    local_search_samples, log_r = langevin_dynamics(samples, energy.log_reward, device, args)
                buffer_ls.add(local_search_samples, log_r)

            samples, rewards = buffer_ls.sample()
        else:
            samples, rewards = buffer.sample()

    loss = get_gfn_backward_loss(args.mode_bwd, samples.to(device), gfn_model, energy.log_reward,
                                 exploration_std=exploration_std)
    return loss


def train():
    #name = get_name(args)  # the mkdirs is bugging with long names
    name = args.run_name
    if not os.path.exists(name):
        os.makedirs(name)

    energy = get_energy()
    eval_data = None  # we are not using this right now - later maybe we will load from a prebuilt buffer
    # energy.sample(eval_data_size).to(device)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)

    gfn_model = GFN(energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
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

    print(gfn_model)

    buffer = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size, data_ndim=energy.data_ndim,
                          beta=args.beta,
                          rank_weight=args.rank_weight, prioritized=args.prioritized)
    buffer_ls = ReplayBuffer(args.buffer_size, device, energy.log_reward, args.batch_size, data_ndim=energy.data_ndim,
                             beta=args.beta,
                             rank_weight=args.rank_weight, prioritized=args.prioritized)

    gfn_model.train()

    oomed_out = False
    for i in trange(args.epochs + 1):
        metrics = dict()
        try:
            metrics['train/loss'] = train_step(energy, gfn_model, gfn_optimizer, i, args.exploratory,
                                               buffer, buffer_ls, args.exploration_factor, args.exploration_wd)
            if not oomed_out:
                if args.batch_size < args.max_batch_size and args.grow_batch_size:
                    args.batch_size = max(args.batch_size + 1,
                                          int(args.batch_size * 1.01))  # gradually increment batch size
                    metrics.update({'Batch Size': args.batch_size})

        except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
            if "CUDA out of memory" in str(
                    e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
                args.batch_size = handle_oom(args.batch_size)
                oomed_out = True
                print(f"Reducing batch size to {args.batch_size}")
            else:
                raise e  # will simply raise error if other or if training on CPU

        if (i % args.eval_period == 0 and i > 0) or i == 50:
            metrics.update(eval_step(energy, gfn_model, eval_batch_size))
            if 'tb-avg' in args.mode_fwd or 'tb-avg' in args.mode_bwd:
                del metrics['eval/log_Z_learned']
            if args.energy == 'molecular_crystal' and args.anneal_energy:
                anneal_energy(energy,
                              metrics['eval/energy'],
                              args.energy_annealing_threshold,
                              10)
            metrics.update({'Crystal Temperature': energy.temperature,
                            'Crystal Turnover Potential': energy.turnover_pot})
            wandb.log(metrics, step=i)

        elif i % 10 == 0:
            wandb.log(metrics, step=i)

        if i % 100 == 0:
            torch.save(gfn_model.state_dict(), f'{name}model.pt')

    torch.save(gfn_model.state_dict(), f'{name}model_final.pt')


def anneal_energy(energy_function, sample_energies, threshold, max_turnover_pot):
    if sample_energies.mean() < threshold:
        if energy_function.turnover_pot < max_turnover_pot:
            energy_function.turnover_pot *= 1.1


def handle_oom(batch_size):
    gc.collect()  # TODO not clear to me that these are effective
    torch.cuda.empty_cache()
    batch_size = int(batch_size * 0.9)
    return batch_size




if __name__ == '__main__':
    if args.eval:
        eval()
    else:
        train()
