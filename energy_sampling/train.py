import gc
import os
from time import time

import numpy as np
import plotly.graph_objects as go
import torch
import wandb
from mxtaltools.dataset_utils.data_classes import MolData
from mxtaltools.dataset_utils.utils import collate_data_list
from torch.optim import lr_scheduler
from torch_geometric.loader import DataLoader
from tqdm import trange

from buffer import CrystalReplayBuffer
from energies.molecular_crystal import MolecularCrystal
from evaluations import eval_step
from models import GFN
from utils import get_train_args, get_gfn_init_state, set_seed, cal_subtb_coef_matrix, \
    get_gfn_optimizer, get_exploration_std, random_discretizer, low_discrepancy_discretizer, \
    low_discrepancy_discretizer2, shifted_equidistant, uniform_discretizer
from gflownet_losses import get_gfn_forward_loss, get_gfn_backward_loss

args = get_train_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

device = args.device
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

times = {}


def train_step(energy_function, gfn_model,
               fwd_gfn_optimizer,
               bwd_gfn_optimizer,
               it, exploration_std, buffer, mol_loader, repeats: int = 10):
    do_forward = False
    do_backward = False
    add_to_buffer = False
    if args.both_ways:
        if it % args.fwd_train_each == 0:
            if args.sampling == 'buffer':
                add_to_buffer = True
            do_forward = True
        else:
            do_backward = True
    elif args.bwd:  # backward ONLY
        do_backward = True
    else:  # forward ONLY
        do_forward = True

    discretizer = get_discretizer(args.discretizer)

    if do_forward:
        fwd_gfn_optimizer.zero_grad()
        mol_batch = next(iter(mol_loader)).to(device)
        loss, states, log_pfs, log_pbs, log_r, log_fs, crystal_batch = fwd_train_step(energy_function,
                                                                                      gfn_model,
                                                                                      discretizer,
                                                                                      exploration_std,
                                                                                      mol_batch,
                                                                                      return_exp=True,
                                                                                      repeats=repeats,
                                                                                      )
        if add_to_buffer:
            buffer.add(crystal_batch.cpu().detach().to_data_list())

    elif do_backward:
        bwd_gfn_optimizer.zero_grad()
        loss, states, log_pfs, log_pbs, log_r, log_fs = bwd_train_step(gfn_model,
                                                                       discretizer,
                                                                       buffer,
                                                                       exploration_std,
                                                                       repeats=repeats,
                                                                       return_exp=True)
    else:
        assert False

    loss.backward()
    torch.nn.utils.clip_grad_norm_(gfn_model.parameters(),
                                   args.gradient_norm_clip)  # gradient clipping
    if do_forward:
        fwd_gfn_optimizer.step()
    elif do_backward:
        bwd_gfn_optimizer.step()
    return loss.item(), "Forward" if do_forward else "Backward"


def get_discretizer(discretization_type):
    # discretizer = lambda bsz: uniform_discretizer(bsz, args.T)
    # discretizer = lambda bsz: uniform_discretizer(bsz, np.random.randint(10,args.T+1))
    # discretizer = lambda bsz: random_discretizer(bsz, args.T, 10)
    if args.traj_length_strategy == 'static':
        traj_length = args.T
    elif args.traj_length_strategy == 'sampled':
        traj_length = np.random.randint(low=args.min_traj_length, high=args.max_traj_length + 1)
    if discretization_type == 'random':
        discretizer = lambda bsz: random_discretizer(bsz, traj_length, max_ratio=args.discretizer_max_ratio)
    elif discretization_type == 'low_discrepancy':
        discretizer = lambda bsz: low_discrepancy_discretizer(bsz, traj_length)
    elif discretization_type == 'low_discrepancy2':
        discretizer = lambda bsz: low_discrepancy_discretizer2(bsz, traj_length)
    elif discretization_type == 'equidistant':
        discretizer = lambda bsz: shifted_equidistant(bsz, traj_length)
    elif discretization_type == 'uniform':
        discretizer = lambda bsz: uniform_discretizer(bsz, traj_length)
    else:
        assert False
    return discretizer


def fwd_train_step(energy_function, gfn_model, discretizer, exploration_std, mol_batch, return_exp=False,
                   repeats: int = 10):
    init_state = get_gfn_init_state(args.batch_size, energy_function.data_ndim, device)
    condition = energy_function.get_conditioning_tensor(mol_batch)
    return get_gfn_forward_loss(args.mode_fwd,
                                init_state,
                                gfn_model,
                                energy_function.log_reward,
                                discretizer,
                                mol_batch,
                                exploration_std=exploration_std,
                                return_exp=return_exp,
                                condition=condition,
                                repeats=repeats)


def bwd_train_step(gfn_model, discretizer, buffer, exploration_std=None, repeats: int = 10, return_exp=False):
    if args.sampling == 'buffer':
        samples, rewards, crystal_batch, condition = buffer.sample(
            return_conditioning=True,
            override_batch=int(buffer.batch_size * args.bwd_batch_multiplier))
    else:
        assert False, f"sampling method {args.sampling} not implemented"

    return get_gfn_backward_loss(args.mode_bwd,
                                 samples.to(device),
                                 gfn_model,
                                 rewards.to(device),
                                 discretizer,
                                 exploration_std=exploration_std,
                                 condition=condition.to(device),
                                 repeats=repeats,
                                 return_exp=return_exp)


def train():
    times['initialization_start'] = time()
    name = args.run_name
    if not os.path.exists(name):
        os.makedirs(name)

    energy_function = MolecularCrystal(device=device,
                                       energy_function=args.energy_function,
                                       min_temperature=args.energy_min_temperature,
                                       max_temperature=args.energy_max_temperature,
                                       temperature_scaling_factor=args.temperature_scaling_factor,
                                       temperature_conditioning=args.temperature_conditioning,
                                       temperature=args.energy_static_temperature,
                                       density_coeff=args.energy_density_coeff)

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy", config=config, name=name)
    conditioning_dim = 1 if args.temperature_conditioning else 0  # probably will not run without this right now
    gfn_model = GFN(energy_function.data_ndim, args.s_emb_dim, args.hidden_dim,
                    conditioning_dim, args.harmonics_dim,
                    args.t_emb_dim, args.bwd_policy, condition_embedding_dim=args.condition_emb_dim,
                    trajectory_length=args.T, clipping=args.clipping,
                    gfn_clip=args.gfn_clip,
                    learned_variance=args.learned_variance,
                    log_var_range=args.log_var_range,
                    pb_scale_range=args.pb_scale_range,
                    t_scale=args.t_scale,
                    conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
                    lgv_layers=args.lgv_layers,
                    joint_layers=args.joint_layers, dropout=args.dropout, norm=args.norm,
                    zero_init=args.zero_init, device=device).to(device)

    wandb.watch(gfn_model, log_graph=True, log_freq=500)  # for gradient logging

    forward_optimizer, backward_optimizer, fwd_scheduler1, fwd_scheduler2, bwd_scheduler1, bwd_scheduler2 = init_schedulers_optimizers(
        gfn_model)
    buffer, mol_loader = init_buffers_datasets(energy_function)

    times['initialization_end'] = time()
    #loss_record, energy_record, learned_Z_record = [], [], []
    oomed_out = False
    prev_rewards_dist = None
    lr_warmup_finished = False
    # maxes out at 1, triggering every 10 steps
    # go from initial to final scaling value in annealing_max_steps
    if args.conditional_flow_model:
        annealing_lambda = (1 / args.temperature_scaling_factor) ** (1 / (args.annealing_max_steps / 10))
    else:
        annealing_lambda = (args.energy_min_temperature / args.energy_static_temperature) ** (
                    1 / (args.annealing_max_steps / 10))

    fwd_loss, bwd_loss = 0, 0
    gfn_model.train()
    for i in trange(args.epochs + 1):
        metrics = dict()
        exploration_std = get_exploration_std(i,
                                              args.exploratory,
                                              args.wd_max_steps,
                                              args.exploration_factor,
                                              args.exploration_wd)
        metrics['train/expl'] = exploration_std(0) if exploration_std is not None else 0

        times['train_step_start'] = time()
        try:
            #torch.autograd.set_detect_anomaly(True)  # for debugging
            train_loss, step_type = train_step(energy_function,
                                               gfn_model,
                                               forward_optimizer,
                                               backward_optimizer,
                                               i,
                                               exploration_std,
                                               buffer,
                                               mol_loader,
                                               repeats=args.repeats
                                               )
            if step_type == 'Forward':
                fwd_loss = train_loss
            elif step_type == 'Backward':
                bwd_loss = train_loss
            if not oomed_out:
                buffer, mol_loader = grow_batch_size(buffer, mol_loader)

        except (RuntimeError, ValueError) as e:  # if we do hit OOM, slash the batch size
            oomed_out, buffer, mol_loader = handle_train_epoch_error(e, oomed_out, buffer, mol_loader)
        times['train_step_end'] = time()

        if (i % args.eval_period == 0 and i > 0) or i == 50:
            torch.save(gfn_model.state_dict(), f'{name}model.pt')
            metrics = do_evaluation(energy_function, buffer, gfn_model, i, metrics, mol_loader)
            wandb.log(metrics, step=i)

        elif i % 10 == 0 and i > 9:
            lr_warmup_finished, lr = step_lr_schedule(bwd_scheduler1, bwd_scheduler2,
                                                      forward_optimizer,
                                                      fwd_scheduler1, fwd_scheduler2,
                                                      lr_warmup_finished)
            anneal_reward(annealing_lambda, energy_function, args)
            metrics.update({'lr': forward_optimizer.param_groups[0]['lr']})
            metrics.update(log_elapsed_times())
            metrics['Forward Loss'] = fwd_loss
            metrics['Backward Loss'] = bwd_loss
            wandb.log(metrics, step=i)

    torch.save(gfn_model.state_dict(), f'{name}_model_final.pt')


def anneal_reward(annealing_lambda, energy_function, args):
    """anneal reward function"""
    if args.anneal_energy:
        if args.conditional_flow_model:
            if energy_function.temperature_scaling_factor < 1:
                energy_function.temperature_scaling_factor *= annealing_lambda
        else:
            if energy_function.temperature > args.energy_min_temperature:
                energy_function.temperature *= annealing_lambda


def step_lr_schedule(bwd_scheduler1, bwd_scheduler2, forward_optimizer, fwd_scheduler1, fwd_scheduler2,
                     lr_warmup_finished):
    if args.scheduler:
        lr = forward_optimizer.param_groups[0]['lr']
        if not lr_warmup_finished:
            fwd_scheduler1.step()
            bwd_scheduler1.step()
            if lr >= args.lr_policy:
                lr_warmup_finished = True

        elif lr > args.min_lr:
            fwd_scheduler2.step()
            bwd_scheduler2.step()
        return lr_warmup_finished, lr
    else:
        return False, None


def init_schedulers_optimizers(gfn_model):
    if args.scheduler:
        init_policy_lr = args.lr_policy / 100
        init_flow_lr = args.lr_flow / 100
        init_back_lr = args.lr_back / 100
    else:
        init_policy_lr = args.lr_policy
        init_flow_lr = args.lr_flow
        init_back_lr = args.lr_back
    forward_optimizer = get_gfn_optimizer(gfn_model,
                                          init_policy_lr,
                                          init_flow_lr,
                                          args.conditional_flow_model,
                                          args.use_weight_decay,
                                          args.weight_decay)
    backward_optimizer = get_gfn_optimizer(gfn_model,
                                           init_back_lr,
                                           init_flow_lr,
                                           args.conditional_flow_model,
                                           args.use_weight_decay,
                                           args.weight_decay)
    if args.scheduler:
        lr_warmup_lambda = (100) ** (1 / (args.lr_warmup_time / 10))  # grow over 2 orders
        lr_annealing_lambda = (args.min_lr / args.lr_policy) ** (1 / (args.lr_anneal_time / 10))
        fwd_scheduler1 = lr_scheduler.MultiplicativeLR(forward_optimizer, lr_lambda=lambda epoch: lr_warmup_lambda)
        fwd_scheduler2 = lr_scheduler.MultiplicativeLR(forward_optimizer, lr_lambda=lambda epoch: lr_annealing_lambda)
        bwd_scheduler1 = lr_scheduler.MultiplicativeLR(backward_optimizer, lr_lambda=lambda epoch: lr_warmup_lambda)
        bwd_scheduler2 = lr_scheduler.MultiplicativeLR(backward_optimizer, lr_lambda=lambda epoch: lr_annealing_lambda)
    else:
        fwd_scheduler1, fwd_scheduler2, bwd_scheduler1, bwd_scheduler2 = None, None, None, None
    return forward_optimizer, backward_optimizer, fwd_scheduler1, fwd_scheduler2, bwd_scheduler1, bwd_scheduler2


def init_buffers_datasets(energy_function):
    # load dataset of prebuilt and scored molecular crystals into the buffer
    buffer = CrystalReplayBuffer(
        args.buffer_size,
        device,
        energy_function,
        args.batch_size,
        beta=args.beta,
        rank_weight=args.rank_weight,
        prioritized=args.prioritized)
    if (args.both_ways or args.bwd) and args.buffer_path is not None:  # preload samples into the buffer
        buffer = add_dataset_to_buffer(args.buffer_path, buffer)
    # load dataset of just molecules
    # mols_list = torch.load(args.molecules_path)
    # good_mol = mols_list[17]  # a nice molecule
    atom_coords = torch.tensor([  # stick with urea for just now
        [-1.3042, - 0.0008, 0.0001],
        [0.6903, - 1.1479, 0.0001],
        [0.6888, 1.1489, 0.0001],
        [- 0.0749, - 0.0001, - 0.0003],
    ], dtype=torch.float32, device='cpu')
    atom_coords -= atom_coords.mean(0)
    atom_types = torch.tensor([8, 7, 7, 6], dtype=torch.long, device='cpu')
    good_mol = MolData(
        z=atom_types,
        pos=atom_coords,
        x=atom_types,
        skip_mol_analysis=False,
    )
    mols_list = [good_mol for _ in range(int(args.max_batch_size * 1.5))]
    mol_loader = DataLoader(
        mols_list,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return buffer, mol_loader


def grow_batch_size(buffer, mol_loader):
    if args.batch_size < args.max_batch_size and args.grow_batch_size:
        new_batch_size = max(args.batch_size + 1,
                             int(args.batch_size * 1.01))
        args.batch_size = new_batch_size  # gradually increment batch size

        if len(buffer) > 0:
            buffer.batch_size = new_batch_size

        mol_loader = DataLoader(
            mol_loader.dataset,
            batch_size=new_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    return buffer, mol_loader


def handle_train_epoch_error(e, oomed_out, buffer, mol_loader):
    if "CUDA out of memory" in str(
            e) or "nonzero is not supported for tensors with more than INT_MAX elements" in str(e):
        args.batch_size = handle_oom(args.batch_size)

        if len(buffer) > 0:  # cut also the buffer size in case it's getting too big
            args.buffer_size = max([10000, int(args.buffer_size * 0.9)])
            buffer.loader = DataLoader(
                buffer.dataset,
                batch_size=args.batch_size,
                sampler=buffer.sampler,
                num_workers=0,
                pin_memory=True,
                drop_last=True)

        mol_loader = DataLoader(
            mol_loader.dataset,
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        oomed_out = True
        print(f"Reducing batch size to {args.batch_size}")
    else:
        raise e  # will simply raise error if other or if training on CPU
    return oomed_out, buffer, mol_loader


def do_evaluation(energy_function, buffer, gfn_model, i, metrics, mol_loader):
    times['eval_step_start'] = time()

    eval_discretizer = lambda bsz: uniform_discretizer(bsz, args.eval_T)

    do_figures = i % args.figs_period == 0
    eval_batch_size = args.eval_batch_size

    eval_rands = np.random.randint(len(mol_loader.dataset), size=eval_batch_size)
    mol_batch = collate_data_list([mol_loader.dataset[ind] for ind in eval_rands]).to(device)

    init_state = get_gfn_init_state(eval_batch_size, energy_function.data_ndim, args.device)
    metrics.update(
        eval_step(energy_function,
                  gfn_model,
                  eval_discretizer,
                  init_state,
                  buffer,
                  do_figures,
                  mol_batch,
                  bwd_training=len(buffer) > 0))

    metrics.update({'Batch Size': args.batch_size})
    metrics.update(log_elapsed_times())

    times['eval_step_end'] = time()

    return metrics


def log_elapsed_times():
    elapsed_times = {}
    for key in times.keys():
        if 'start' in key:
            start_key = key
            end_key = start_key.split('_start')[0] + '_end'
            if end_key in times.keys():
                elapsed_times[start_key.split('_start')[0] + '_time'] = times[end_key] - times[start_key]

    return elapsed_times


def handle_oom(batch_size):
    gc.collect()
    torch.cuda.empty_cache()
    batch_size = int(batch_size * 0.9)
    return batch_size


def add_dataset_to_buffer(dataset_path, buffer):
    print("Loading prebuilt buffer")
    dataset = torch.load(dataset_path)
    if args.energy_function == 'ellipsoid_overlap':  # add ellipsoid overlaps to each sample here, as they weren't in the original optimization
        print("Adding ellipsoid information to buffer")

        from tqdm import tqdm
        batch_size = 500
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False
        )
        overlaps = []

        for crystal_batch in tqdm(loader):
            crystal_batch = crystal_batch.to('cuda')
            crystal_batch.box_analysis()
            cluster_batch = crystal_batch.mol2cluster(cutoff=6,
                                                      supercell_size=10,
                                                      align_to_standardized_orientation=True)

            cluster_batch.construct_radial_graph(cutoff=6)
            # simplified ellipsoid energy testing
            _, _, _, _, _, _, normed_ellipsoid_overlap \
                = cluster_batch.compute_ellipsoidal_overlap(
                semi_axis_scale=1,
                return_details=True)

            overlaps.extend(normed_ellipsoid_overlap.cpu().detach().numpy())

        overlaps = torch.tensor(overlaps)
        for ind, elem in enumerate(dataset):
            elem.ellipsoid_overlap = torch.ones(1) * overlaps[ind]

    buffer.add(dataset)
    print(f"Buffer loaded with {len(dataset)} samples")

    return buffer


if __name__ == '__main__':
    train()
