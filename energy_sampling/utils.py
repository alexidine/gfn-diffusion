import argparse
import random
import numpy as np
import math
import PIL
import yaml

from gflownet_losses import *


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


def get_gfn_optimizer(gfn_model, lr_policy, lr_flow, lr_back, back_model=False, conditional_flow_model=False,
                      use_weight_decay=False, weight_decay=1e-7):
    param_groups = [{'params': gfn_model.t_model.parameters()},
                    {'params': gfn_model.s_model.parameters()},
                    {'params': gfn_model.joint_model.parameters()},
                    {'params': gfn_model.langevin_scaling_model.parameters()}]
    if conditional_flow_model:
        param_groups += [{'params': gfn_model.flow_model.parameters(), 'lr': lr_flow}]
    else:
        param_groups += [{'params': [gfn_model.flow_model], 'lr': lr_flow}]

    if back_model:
        param_groups += [{'params': gfn_model.back_model.parameters(), 'lr': lr_back}]

    if use_weight_decay:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy, weight_decay=weight_decay)
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, lr_policy)
    return gfn_optimizer


def get_gfn_forward_loss(mode, init_state, gfn_model, log_reward, coeff_matrix, exploration_std=None, return_exp=False):
    if mode == 'tb':
        loss = fwd_tb(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'tb-avg':
        loss = fwd_tb_avg(init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp)
    elif mode == 'db':
        loss = db(init_state, gfn_model, log_reward, exploration_std)
    elif mode == 'subtb':
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    return loss


def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, exploration_std=None):
    if mode == 'tb':
        loss = bwd_tb(samples, gfn_model, log_reward, exploration_std)
    elif mode == 'tb-avg':
        loss = bwd_tb_avg(samples, gfn_model, log_reward, exploration_std)
    elif mode == 'mle':
        loss = bwd_mle(samples, gfn_model, log_reward, exploration_std)
    return loss


def get_exploration_std(iter, exploratory, exploration_factor=0.1, exploration_wd=False):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1. - iter / 5000.)
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
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr_policy', type=float, default=1e-3)
    parser.add_argument('--lr_flow', type=float, default=1e-2)
    parser.add_argument('--lr_back', type=float, default=1e-3)
    parser.add_argument('--gradient_norm_clip', type=float, default=10)
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
    parser.add_argument('--dataset_path', type=str, default=None)
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
    parser.add_argument('--energy_annealing_threshold', type=float, default=1e-3)
    args, remaining = parser.parse_known_args()

    if 'config' in remaining[0]:  # load external yaml config file
        with open(remaining[1], 'r') as f:
            config_args = yaml.safe_load(f)
        for key, value in config_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                parser.error(f"Unknown config key: {key}")

    return args
